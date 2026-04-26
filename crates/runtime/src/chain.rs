//! N-peer pipeline client.
//!
//! A `Chain` connects to an ordered list of peers, opens a session with
//! each (one owns `[s0..s1)`, next owns `[s1..s2)`, …, tail owns
//! `[s_{n-1}..N)`), and funnels hidden states through them in order for
//! each forward step. The driver keeps the front slice `[0..s0)` (if
//! any), plus the embedding and the head.
//!
//! Used by both `pipe_driver` (as a canonical example) and the CLI's
//! `RunMode::Network` path, so any hardening — timeouts, reconnects,
//! wire-level trace — shows up in both places at once.

use std::io;
use std::net::SocketAddr;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use bytes::BytesMut;
use intelnav_core::types::{LayerRange, Quant};
use intelnav_core::{PeerId, SessionId};
use intelnav_ggml::{decode_hidden, encode_hidden_with, Hidden, HiddenPayload};
use intelnav_wire::{self as wire, Dtype, Msg, Phase};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::timeout;

use crate::generate::SamplingCfg;
use crate::model::ModelHandle;
use crate::sample::{Sampler, SamplerCfg};
use crate::telemetry::{StepEvent, StepPhase, Telemetry};
use crate::tokenizer::Tok;

/// Per-peer link state. One TCP connection, one session, one inbound
/// read buffer. The driver owns a `Vec<Link>` and walks them in order.
pub struct Link {
    pub addr:       SocketAddr,
    pub layers:     LayerRange,
    pub session:    SessionId,
    sock:           TcpStream,
    rx:             BytesMut,
    seq:            u64,
}

impl Link {
    pub fn short(&self) -> String {
        format!("{}", self.addr)
    }
}

/// What went wrong while walking the chain. Keeping this structured
/// lets the TUI surface `peer 2 (192.168.1.4:7718) dropped` instead of
/// a generic "io error".
#[derive(Debug, thiserror::Error)]
pub enum ChainError {
    #[error("peer {index} ({addr}) refused connection: {source}")]
    Connect {
        index:  usize,
        addr:   SocketAddr,
        #[source]
        source: io::Error,
    },
    #[error("peer {index} ({addr}): {reason}")]
    Handshake { index: usize, addr: SocketAddr, reason: String },
    #[error("peer {index} ({addr}) timed out after {:?}", .timeout)]
    Timeout { index: usize, addr: SocketAddr, timeout: Duration },
    #[error("peer {index} ({addr}) dropped mid-step: {reason}")]
    Dropped { index: usize, addr: SocketAddr, reason: String },
    #[error("peer {index} ({addr}) aborted: {reason}")]
    Aborted { index: usize, addr: SocketAddr, reason: String },
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Chain config. `splits[0]` is the layer boundary between the driver's
/// front slice and peer 0; `splits[i]` is the boundary between peer
/// `i-1` and peer `i`. The tail implicitly extends to `n_blocks`.
///
/// Invariants (checked in `connect`):
/// * `peers.len() == splits.len()`
/// * `splits` is strictly increasing
/// * `0 < splits[0]` (so the driver owns at least the embedding and one block — keeps front-half runtime warm)
/// * `*splits.last() < n_blocks`
#[derive(Clone, Debug)]
pub struct ChainCfg {
    pub peers:      Vec<SocketAddr>,
    pub splits:     Vec<u16>,
    pub proto_ver:  u32,
    pub model_cid:  String,
    pub max_seq:    u32,
    /// Per-request wall-clock budget. A peer that exceeds this on any
    /// read or write is counted as dropped and the session is torn down.
    pub step_timeout: Duration,
    /// Connect-attempt budget. Separate from `step_timeout` because a
    /// cold start on a big model legitimately takes seconds to load.
    pub connect_timeout: Duration,
    /// Activation dtype on the wire. `Fp16` is lossless enough for the
    /// Q4_K_M noise floor; `Int8` halves bytes per step at negligible
    /// quality cost. See [`crate::hidden`] for the layout.
    pub wire_dtype: Dtype,
}

impl ChainCfg {
    pub fn single(peer: SocketAddr, split: u16) -> Self {
        Self {
            peers:           vec![peer],
            splits:          vec![split],
            proto_ver:       1,
            model_cid:       "local".into(),
            max_seq:         2048,
            step_timeout:    Duration::from_secs(30),
            connect_timeout: Duration::from_secs(10),
            wire_dtype:      Dtype::Fp16,
        }
    }

    pub fn many(peers: Vec<SocketAddr>, splits: Vec<u16>) -> Self {
        Self { peers, splits, ..Self::single("127.0.0.1:7717".parse().unwrap(), 0) }
    }
}

/// A live chain of peers, mid-session.
pub struct Chain {
    cfg:        ChainCfg,
    links:      Vec<Link>,
    front_end:  u16,
    /// Optional broadcast handle for live demos. When set, every peer
    /// hop in [`Chain::step_with_truncate`] emits a `StepEvent` with
    /// timing + byte counts. Subscribers (the gateway's
    /// `/v1/swarm/events` SSE) see hops in real time.
    telemetry:  Option<Telemetry>,
}

impl Chain {
    pub fn front_range(&self) -> (usize, usize) { (0, self.front_end as usize) }

    /// How many peers we're fronting.
    pub fn peer_count(&self) -> usize { self.links.len() }

    /// Attach a telemetry handle so subsequent steps publish hop
    /// events. Idempotent — replaces any previously attached handle.
    pub fn attach_telemetry(&mut self, telemetry: Telemetry) {
        self.telemetry = Some(telemetry);
    }

    /// Connect + handshake + open a session on every peer. Fails fast on
    /// the first peer that can't open — no partial state retained.
    pub async fn connect(cfg: ChainCfg, n_blocks: u16) -> std::result::Result<Self, ChainError> {
        if cfg.peers.is_empty() {
            return Err(ChainError::Other(anyhow!("chain needs at least one peer")));
        }
        if cfg.peers.len() != cfg.splits.len() {
            return Err(ChainError::Other(anyhow!(
                "peers ({}) and splits ({}) length mismatch",
                cfg.peers.len(), cfg.splits.len()
            )));
        }
        for pair in cfg.splits.windows(2) {
            if pair[0] >= pair[1] {
                return Err(ChainError::Other(anyhow!(
                    "splits must be strictly increasing, got {:?}", cfg.splits
                )));
            }
        }
        if cfg.splits[0] == 0 {
            return Err(ChainError::Other(anyhow!(
                "first split must be > 0 (driver owns [0..s0))"
            )));
        }
        if let Some(&last) = cfg.splits.last() {
            if last >= n_blocks {
                return Err(ChainError::Other(anyhow!(
                    "last split {last} must be < n_blocks {n_blocks}"
                )));
            }
        }

        let mut links = Vec::with_capacity(cfg.peers.len());
        for (i, &addr) in cfg.peers.iter().enumerate() {
            let start = cfg.splits[i];
            let end = cfg.splits.get(i + 1).copied().unwrap_or(n_blocks);
            let layers = LayerRange::new(start, end);

            let sock = timeout(cfg.connect_timeout, TcpStream::connect(addr))
                .await
                .map_err(|_| ChainError::Timeout { index: i, addr, timeout: cfg.connect_timeout })?
                .map_err(|source| ChainError::Connect { index: i, addr, source })?;
            sock.set_nodelay(true).ok();
            let mut link = Link {
                addr,
                layers,
                session: SessionId::random(),
                sock,
                rx: BytesMut::with_capacity(64 * 1024),
                seq: 0,
            };

            // Hello → Hello
            let hello = Msg::Hello {
                peer_id:          PeerId::new([0u8; 32]),
                proto_ver:        cfg.proto_ver,
                supported_quants: vec![Quant::Q4KM, Quant::FP16],
            };
            Self::write_with_timeout(&mut link, &hello, cfg.step_timeout, i).await?;
            match Self::read_with_timeout(&mut link, cfg.step_timeout, i).await? {
                Some(Msg::Hello { proto_ver, .. }) if proto_ver == cfg.proto_ver => {}
                Some(Msg::Hello { proto_ver, .. }) => {
                    return Err(ChainError::Handshake {
                        index: i, addr,
                        reason: format!(
                            "proto_ver mismatch: driver v{}, peer v{proto_ver}", cfg.proto_ver
                        ),
                    });
                }
                other => {
                    return Err(ChainError::Handshake {
                        index: i, addr,
                        reason: format!("expected Hello, got {other:?}"),
                    });
                }
            }

            // SessionInit → SessionAck
            let init = Msg::SessionInit {
                session_id:        link.session,
                client_x25519_pub: [0u8; 32],
                model_cid:         cfg.model_cid.clone(),
                layer_range:       layers,
                max_seq:           cfg.max_seq,
            };
            Self::write_with_timeout(&mut link, &init, cfg.step_timeout, i).await?;
            match Self::read_with_timeout(&mut link, cfg.step_timeout, i).await? {
                Some(Msg::SessionAck { session_id, .. }) if session_id == link.session => {}
                Some(Msg::AbortSession { reason, .. }) => {
                    return Err(ChainError::Aborted { index: i, addr, reason });
                }
                other => {
                    return Err(ChainError::Handshake {
                        index: i, addr,
                        reason: format!("expected SessionAck, got {other:?}"),
                    });
                }
            }

            links.push(link);
        }

        let front_end = cfg.splits[0];
        let _ = n_blocks;
        Ok(Self { cfg, links, front_end, telemetry: None })
    }

    /// Run one forward step: push the front hidden through each peer in
    /// order, then return the tail hidden for the driver to feed into
    /// `head`. Callers compute the front slice locally.
    pub async fn step(
        &mut self,
        front_hidden: Hidden,
        phase:        Phase,
    ) -> std::result::Result<Hidden, ChainError> {
        self.step_with_truncate(front_hidden, phase, None).await
    }

    /// Same as [`step`], but each peer first truncates its KV cache to
    /// `keep` entries along the sequence dimension. Used by speculative
    /// decoding to roll back rejected draft tokens.
    pub async fn step_with_truncate(
        &mut self,
        front_hidden: Hidden,
        phase:        Phase,
        keep:         Option<u32>,
    ) -> std::result::Result<Hidden, ChainError> {
        let mut cur = front_hidden;
        let step_phase = match phase {
            Phase::Prefill => StepPhase::Prefill,
            Phase::Decode  => StepPhase::Decode,
        };
        for i in 0..self.links.len() {
            let hop_started = std::time::Instant::now();
            let link = &mut self.links[i];
            link.seq += 1;
            let seq = link.seq;
            let session = link.session;
            let addr = link.addr;

            // Hidden is rank-3 [batch, seq, hidden]; the wire carries
            // the same shape as [u32; 3].
            let (exp_b, exp_s, exp_h) = match cur.shape.as_slice() {
                [b, s, h] => (*b, *s, *h),
                _ => return Err(ChainError::Other(anyhow!(
                    "front_hidden must be rank-3 [batch, seq, hidden], got shape {:?}",
                    cur.shape
                ))),
            };
            let wire_shape = [exp_b as u32, exp_s as u32, exp_h as u32];
            let p = encode_hidden_with(&cur.data, wire_shape, self.cfg.wire_dtype)
                .map_err(ChainError::from)?;
            // Captured here because `p.bytes` is moved into the Msg below.
            let bytes_up = p.bytes.len() as u64;

            let req = Msg::ForwardHidden {
                session_id: session,
                seq,
                phase,
                dtype:      p.dtype,
                shape:      p.shape,
                payload:    p.bytes,
                kv_delta:   None,
                kv_truncate_to: keep,
            };
            Self::write_with_timeout(link, &req, self.cfg.step_timeout, i).await?;

            let reply = Self::read_with_timeout(link, self.cfg.step_timeout, i).await?;
            let (r_dtype, r_shape, r_payload) = match reply {
                Some(Msg::ForwardHidden {
                    session_id: sid, dtype, shape, payload, ..
                }) if sid == session => (dtype, shape, payload),
                Some(Msg::AbortSession { reason, .. }) => {
                    return Err(ChainError::Aborted { index: i, addr, reason });
                }
                Some(other) => {
                    return Err(ChainError::Dropped {
                        index: i, addr,
                        reason: format!("unexpected message {other:?}"),
                    });
                }
                None => {
                    return Err(ChainError::Dropped {
                        index: i, addr,
                        reason: "peer closed before reply".into(),
                    });
                }
            };
            if !matches!(r_dtype, Dtype::Fp16 | Dtype::Int8) {
                return Err(ChainError::Dropped {
                    index: i, addr,
                    reason: format!(
                        "peer returned dtype {r_dtype:?}, only fp16/int8 supported"
                    ),
                });
            }
            let (out_shape, out_data) = decode_hidden(&HiddenPayload {
                dtype: r_dtype, shape: r_shape, bytes: r_payload,
            }).map_err(ChainError::from)?;
            if out_shape[0] as usize != exp_b || out_shape[1] as usize != exp_s {
                return Err(ChainError::Dropped {
                    index: i, addr,
                    reason: format!(
                        "hidden shape mismatch: sent [{exp_b}, {exp_s}, _], got {:?}", out_shape
                    ),
                });
            }
            let bytes_down = out_data.len() as u64
                * std::mem::size_of::<f32>() as u64;
            cur = Hidden::new(
                out_data,
                vec![out_shape[0] as usize, out_shape[1] as usize, out_shape[2] as usize],
            ).map_err(ChainError::from)?;

            // Publish the per-hop event. Cheap when no subscribers
            // (broadcast::send is a no-op without receivers).
            if let Some(t) = &self.telemetry {
                if t.has_subscribers() {
                    let rtt_ms = hop_started.elapsed().as_secs_f32() * 1000.0;
                    t.emit(StepEvent {
                        seq:        0,                 // assigned by Telemetry::emit
                        at_ms:      0,                 // assigned by Telemetry::emit
                        peer_index: i,
                        peer_id:    short_peer_label(&addr),
                        phase:      step_phase,
                        rtt_ms,
                        bytes_up,
                        bytes_down,
                        synthetic:  false,
                    });
                }
            }
        }
        Ok(cur)
    }

    /// Send a courtesy `AbortSession` to every peer so their server logs
    /// line ends with "ended cleanly". Best-effort — errors are eaten.
    pub async fn close(&mut self, reason: &str) {
        for link in &mut self.links {
            let msg = Msg::AbortSession {
                session_id: link.session,
                reason:     reason.into(),
            };
            let mut out = BytesMut::with_capacity(256);
            if wire::encode_frame(&mut out, &msg).is_ok() {
                let _ = link.sock.write_all(&out).await;
                let _ = link.sock.flush().await;
            }
        }
    }

    async fn write_with_timeout(
        link:  &mut Link,
        msg:   &Msg,
        limit: Duration,
        index: usize,
    ) -> std::result::Result<(), ChainError> {
        let mut out = BytesMut::with_capacity(256);
        wire::encode_frame(&mut out, msg)
            .map_err(|e| ChainError::Other(anyhow!("encode: {e}")))?;
        let fut = async {
            link.sock.write_all(&out).await?;
            link.sock.flush().await?;
            io::Result::Ok(())
        };
        match timeout(limit, fut).await {
            Err(_) => Err(ChainError::Timeout { index, addr: link.addr, timeout: limit }),
            Ok(Err(source)) => Err(ChainError::Dropped {
                index, addr: link.addr, reason: format!("write: {source}"),
            }),
            Ok(Ok(())) => Ok(()),
        }
    }

    async fn read_with_timeout(
        link:  &mut Link,
        limit: Duration,
        index: usize,
    ) -> std::result::Result<Option<Msg>, ChainError> {
        let fut = async {
            loop {
                if let Some(msg) = wire::decode_frame(&mut link.rx)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("decode: {e}")))?
                {
                    return io::Result::Ok(Some(msg));
                }
                let had_any = !link.rx.is_empty();
                let n = link.sock.read_buf(&mut link.rx).await?;
                if n == 0 {
                    return if had_any {
                        Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            format!("truncated frame at EOF ({} bytes buffered)", link.rx.len()),
                        ))
                    } else {
                        Ok(None)
                    };
                }
            }
        };
        match timeout(limit, fut).await {
            Err(_) => Err(ChainError::Timeout { index, addr: link.addr, timeout: limit }),
            Ok(Err(source)) => Err(ChainError::Dropped {
                index, addr: link.addr, reason: format!("read: {source}"),
            }),
            Ok(Ok(opt)) => Ok(opt),
        }
    }
}

/// Helper: apply the driver's `[0..front_end)` on `tokens` and return
/// the hidden state ready for the chain's first peer.
pub fn front_forward(
    model:     &mut ModelHandle,
    tokens:    &[u32],
    index_pos: usize,
    front_end: usize,
) -> Result<Hidden> {
    let pl = model.pipelined().ok_or_else(|| anyhow!("model not pipelined"))?;
    let embedded = pl.embed(tokens)?;
    pl.forward_range(&embedded, index_pos, 0, front_end)
        .context("front forward_range")
}

/// Helper: apply the driver's `head` on the tail hidden returned by
/// the chain. Returns logits shape `[batch, vocab]`.
pub fn head_forward(model: &mut ModelHandle, tail_hidden: &Hidden) -> Result<Hidden> {
    let pl = model.pipelined().ok_or_else(|| anyhow!("model not pipelined"))?;
    pl.head(tail_hidden).map_err(Into::into)
}

/// Like [`head_forward`] but returns logits for every position:
/// shape `[batch, seq, vocab]`. Used by speculative decoding to score
/// k draft positions in one verify forward.
pub fn head_all_forward(model: &mut ModelHandle, tail_hidden: &Hidden) -> Result<Hidden> {
    let pl = model.pipelined().ok_or_else(|| anyhow!("model not pipelined"))?;
    pl.head_all(tail_hidden).map_err(Into::into)
}

/// End-to-end turn over a chain: connect, prefill, decode, close.
///
/// The driver owns `model` + `tok`, runs the front slice `[0..s0)` and
/// the head locally, and funnels the middle through `chain`. Calls
/// `on_token(text)` for each decoded text chunk.
///
/// Blocks on synchronous ggml forward passes in-between `await` points
/// — callers should spawn it on a current-thread runtime dedicated to
/// this turn.
pub async fn run_turn<F: FnMut(&str) -> Result<()>>(
    model:   &mut ModelHandle,
    tok:     &Tok,
    chain:   &mut Chain,
    prompt:  &str,
    cfg:     &SamplingCfg,
    mut on_token: F,
) -> Result<usize> {
    let front_end = chain.front_range().1;
    model.reset_cache();

    let mut tokens = tok.encode(prompt)?;
    if tokens.is_empty() {
        return Err(anyhow!("prompt tokenized to zero tokens"));
    }

    let mut sampler = Sampler::new(cfg.seed, cfg.temperature, cfg.top_p);
    let scfg = SamplerCfg { repeat_penalty: cfg.repeat_penalty, repeat_ctx: cfg.repeat_ctx };
    let mut decoder = tok.incremental();
    let mut index_pos: usize = 0;

    // Prefill.
    let front = front_forward(model, &tokens, index_pos, front_end)?;
    let tail = chain.step(front, Phase::Prefill).await.map_err(|e| anyhow!("{e}"))?;
    let logits = head_forward(model, &tail)?;
    index_pos += tokens.len();
    let mut next = sampler.sample(&logits.data, &tokens, &scfg)?;

    let mut n_gen = 0usize;
    loop {
        if tok.is_eos(next) || n_gen >= cfg.max_new_tokens { break; }
        tokens.push(next);
        n_gen += 1;
        if let Some(txt) = decoder.push(next)? {
            on_token(&txt)?;
        }

        let step_input = [next];
        let front = front_forward(model, &step_input, index_pos, front_end)?;
        let tail = chain.step(front, Phase::Decode).await.map_err(|e| anyhow!("{e}"))?;
        let logits = head_forward(model, &tail)?;
        index_pos += 1;
        next = sampler.sample(&logits.data, &tokens, &scfg)?;
    }

    Ok(n_gen)
}

/// Render a `SocketAddr` as the same bs58-short id the gateway's
/// static directory would assign for `host:port`. Lets telemetry
/// events match peer-card ids in the SPA without a lookup.
fn short_peer_label(addr: &SocketAddr) -> String {
    let s = addr.to_string();
    let h = blake3::hash(s.as_bytes());
    let b58 = bs58::encode(h.as_bytes()).into_string();
    let mut out = String::with_capacity(14);
    out.push_str(&b58[..6]);
    out.push('…');
    out.push_str(&b58[b58.len() - 6..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cfg_single_is_valid() {
        let c = ChainCfg::single("127.0.0.1:7717".parse().unwrap(), 12);
        assert_eq!(c.peers.len(), 1);
        assert_eq!(c.splits, vec![12]);
    }

    #[test]
    fn cfg_many_preserves_inputs() {
        let c = ChainCfg::many(
            vec!["127.0.0.1:1".parse().unwrap(), "127.0.0.1:2".parse().unwrap()],
            vec![4, 8],
        );
        assert_eq!(c.splits, vec![4, 8]);
    }
}
