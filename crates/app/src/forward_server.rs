//! In-process forward listener for `intelnav-node`.
//!
//! Hosts the inference TCP endpoint advertised as `forward_url` in
//! every provider record. Replaces the standalone `pipe_peer` example
//! binary so a host runs *one* process (`intelnav-node`) instead of
//! N + 1 (`intelnav-node` + one `pipe_peer` per slice).
//!
//! Behaviour:
//!
//! * Bind `config.forward_addr` once at daemon boot.
//! * On each connection: read `Hello`, then `SessionInit` with a
//!   `(model_cid, layer_range)`. Look up [`ControlState`] for that
//!   key — if not Announcing, send `AbortSession` and close. The
//!   consumer's chain driver fails over to its alternate.
//! * If admitted, hold a [`ChainGuard`]. Lazy-load the model for the
//!   slice (stitched-subset GGUF cached on disk after first build).
//!   Dispatch `ForwardHidden` messages until the consumer hangs up
//!   or aborts.
//! * Multiple slices: each `(cid, start, end)` gets its own
//!   `ModelHandle` cached behind an `Arc<Mutex<…>>` so concurrent
//!   chains on different slices don't serialise.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use bytes::BytesMut;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use intelnav_core::{PeerId, SessionId};
use intelnav_core::types::Quant;
use intelnav_ggml::{decode_hidden, encode_hidden_with, Hidden, HiddenPayload};
use intelnav_runtime::{DevicePref, ModelHandle};
use intelnav_wire::{self as wire, Msg};

use crate::contribute::{shard_dir, DisabledRanges, KeptRanges};
use crate::control::{ChainGuard, ControlState, SliceKey};

pub fn spawn(
    addr_raw: &str,
    state: Arc<ControlState>,
    models_dir: PathBuf,
) -> Option<JoinHandle<()>> {
    let addr: SocketAddr = match addr_raw.parse() {
        Ok(a) => a,
        Err(e) => { warn!(?e, addr_raw, "forward_addr is not host:port"); return None; }
    };
    Some(tokio::spawn(async move {
        if let Err(e) = serve(addr, state, models_dir).await {
            warn!(?e, "forward listener exited");
        }
    }))
}

async fn serve(addr: SocketAddr, state: Arc<ControlState>, models_dir: PathBuf) -> Result<()> {
    let listener = TcpListener::bind(addr).await
        .with_context(|| format!("bind forward listener {addr}"))?;
    info!(%addr, "forward listener up");
    let cache: ModelCache = Arc::new(Mutex::new(HashMap::new()));
    loop {
        let (sock, peer) = match listener.accept().await {
            Ok(v) => v,
            Err(e) => { warn!(?e, "forward accept"); continue; }
        };
        let st = state.clone();
        let cache = cache.clone();
        let dir = models_dir.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_connection(sock, peer, st, cache, dir).await {
                debug!(?e, %peer, "forward session ended");
            }
        });
    }
}

/// Per-slice model cache: load once, reuse for every chain that hits us
/// for that (cid, start, end). The Mutex serializes forward calls on a
/// single slice (KV cache + libllama state are not Send/Sync-clean).
type ModelCache = Arc<Mutex<HashMap<SliceKey, Arc<Mutex<LoadedSlice>>>>>;

struct LoadedSlice {
    handle: ModelHandle,
    /// Subtract from REAL layer indices before calling libllama's
    /// `forward_range`. 0 for full-GGUF mode; `start` for stitched
    /// (the stitched file renumbers tensors from 0).
    local_offset: usize,
}

async fn handle_connection(
    mut sock: TcpStream,
    peer: SocketAddr,
    state: Arc<ControlState>,
    cache: ModelCache,
    models_dir: PathBuf,
) -> Result<()> {
    sock.set_nodelay(true).ok();
    let mut buf = BytesMut::with_capacity(64 * 1024);
    let mut session_id: Option<SessionId> = None;
    let mut index_pos: usize = 0;
    let mut last_seq: u64 = 0;
    let mut active_slice: Option<SliceKey> = None;
    let mut _guard: Option<ChainGuard> = None;
    let mut loaded: Option<Arc<Mutex<LoadedSlice>>> = None;

    loop {
        let msg = match read_frame(&mut sock, &mut buf).await? {
            Some(m) => m,
            None    => return Ok(()),
        };
        match msg {
            Msg::Hello { peer_id, proto_ver, .. } => {
                let reply = Msg::Hello {
                    peer_id: PeerId::new([0u8; 32]),
                    proto_ver,
                    supported_quants: vec![Quant::Q4KM, Quant::FP16],
                };
                write_frame(&mut sock, &reply).await?;
                debug!(%peer, %proto_ver, "hello from {}", peer_id.short());
            }
            Msg::SessionInit { session_id: sid, layer_range, model_cid, .. } => {
                let key: SliceKey = (
                    model_cid.clone(),
                    layer_range.start,
                    layer_range.end,
                );
                // Refuse if not announcing — the consumer fails over.
                let Some(g) = state.accept_chain(&key).await else {
                    let reply = Msg::AbortSession {
                        session_id: sid,
                        reason: format!(
                            "slice {model_cid} [{}..{}) not announcing on this peer",
                            layer_range.start, layer_range.end,
                        ),
                    };
                    write_frame(&mut sock, &reply).await?;
                    return Ok(());
                };
                _guard = Some(g);
                active_slice = Some(key.clone());

                // Lazy-load the model for this slice (cached after first hit).
                let slice = match prepare_slice(&cache, &models_dir, &key).await {
                    Ok(s) => s,
                    Err(e) => {
                        let reply = Msg::AbortSession {
                            session_id: sid,
                            reason: format!("load slice failed: {e}"),
                        };
                        write_frame(&mut sock, &reply).await?;
                        return Err(e);
                    }
                };
                {
                    let mut g = slice.lock().await;
                    if let Some(pl) = g.handle.pipelined() {
                        pl.reset_cache();
                    } else {
                        g.handle.reset_cache();
                    }
                }
                loaded = Some(slice);
                session_id = Some(sid);
                index_pos = 0;
                last_seq = 0;

                let ack = Msg::SessionAck {
                    session_id: sid,
                    shard_x25519_pub: [0u8; 32], // crypto deferred
                };
                write_frame(&mut sock, &ack).await?;
                debug!(%peer, "session opened for {model_cid} [{}..{})",
                    layer_range.start, layer_range.end);
            }
            Msg::ForwardHidden { session_id: sid, seq, phase: _, dtype, shape, payload, kv_delta: _, kv_truncate_to } => {
                let cur = session_id.ok_or_else(|| anyhow!("ForwardHidden before SessionInit"))?;
                if cur != sid { return Err(anyhow!("ForwardHidden session mismatch")); }
                if seq <= last_seq && last_seq > 0 {
                    return Err(anyhow!("ForwardHidden out-of-order: last={last_seq} got={seq}"));
                }
                last_seq = seq;

                let slice = loaded.as_ref().ok_or_else(|| anyhow!("session not loaded"))?;
                let (real_start, real_end) = active_slice.as_ref()
                    .map(|(_, s, e)| (*s as usize, *e as usize))
                    .ok_or_else(|| anyhow!("session has no slice"))?;

                let mut g = slice.lock().await;
                let local_offset = g.local_offset;
                let pl = g.handle.pipelined()
                    .ok_or_else(|| anyhow!("model not pipelined"))?;
                if let Some(keep) = kv_truncate_to {
                    pl.truncate_kv_to(keep as usize)?;
                    index_pos = keep as usize;
                }
                let wire_payload = HiddenPayload { dtype, shape, bytes: payload };
                let (in_shape, in_data) = decode_hidden(&wire_payload)?;
                let hidden = Hidden::new(
                    in_data,
                    vec![in_shape[0] as usize, in_shape[1] as usize, in_shape[2] as usize],
                )?;
                let seq_len = hidden.shape[1];

                let local_start = real_start - local_offset;
                let local_end   = real_end - local_offset;
                let _t = Instant::now();
                let out = pl.forward_range(&hidden, index_pos, local_start, local_end)?;
                index_pos += seq_len;

                let out_shape = [out.shape[0] as u32, out.shape[1] as u32, out.shape[2] as u32];
                let p = encode_hidden_with(&out.data, out_shape, dtype)?;
                let reply = Msg::ForwardHidden {
                    session_id: sid,
                    seq,
                    phase: intelnav_wire::Phase::Decode,
                    dtype: p.dtype,
                    shape: p.shape,
                    payload: p.bytes,
                    kv_delta: None,
                    kv_truncate_to: None,
                };
                drop(g);
                write_frame(&mut sock, &reply).await?;
            }
            Msg::AbortSession { session_id: sid, reason } => {
                debug!(%peer, ?sid, "abort: {reason}");
                return Ok(());
            }
            Msg::Heartbeat { .. } | Msg::Advertise { .. } | Msg::Gossip { .. } => {}
            Msg::Prompt { .. } | Msg::Token { .. } => {
                return Err(anyhow!("peer received driver-only message"));
            }
            Msg::SessionAck { .. } => {
                return Err(anyhow!("peer received SessionAck — that's a driver message"));
            }
        }
    }
}

/// Resolve the slice's GGUF (full or stitched-subset) and load it
/// once. Cached per (cid, start, end).
async fn prepare_slice(
    cache: &ModelCache,
    models_dir: &std::path::Path,
    key: &SliceKey,
) -> Result<Arc<Mutex<LoadedSlice>>> {
    {
        let g = cache.lock().await;
        if let Some(s) = g.get(key) { return Ok(s.clone()); }
    }
    let (cid, start, end) = key.clone();
    let root = shard_dir(models_dir, &cid);
    let kept_path = root.join("kept_ranges.json");
    let bytes = tokio::fs::read(&kept_path).await
        .with_context(|| format!("read {}", kept_path.display()))?;
    let kept: KeptRanges = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse {}", kept_path.display()))?;

    // Find the (start, end) entry. We already validated via accept_chain
    // that this slice is announcing, but the sidecar might have been
    // edited concurrently — re-check.
    let disabled = DisabledRanges::load(&root);
    if disabled.contains(start, end) {
        anyhow::bail!("slice {start}..{end} is disabled on disk");
    }
    if !kept.kept.iter().any(|&(s, e)| s == start && e == end) {
        anyhow::bail!("slice {start}..{end} not present in kept_ranges.json");
    }

    let (gguf_path, local_offset) = if !kept.gguf_path.as_os_str().is_empty()
        && kept.gguf_path.exists()
    {
        // Hub→split path: the full GGUF is on disk, real layer indices.
        (kept.gguf_path.clone(), 0usize)
    } else {
        // Swarm-pulled path: stitch a subset GGUF from manifest+chunks.
        let stitched = stitch_subset_for(&root, start, end, kept.block_count).await?;
        (stitched, start as usize)
    };

    let handle = tokio::task::spawn_blocking({
        let p = gguf_path.clone();
        move || ModelHandle::load(&p, DevicePref::Auto)
            .with_context(|| format!("load {}", p.display()))
    })
        .await
        .map_err(|e| anyhow!("join: {e}"))??;

    let slice = Arc::new(Mutex::new(LoadedSlice { handle, local_offset }));
    let mut g = cache.lock().await;
    g.insert(key.clone(), slice.clone());
    info!(%cid, start, end, gguf = %gguf_path.display(), "slice loaded");
    Ok(slice)
}

async fn stitch_subset_for(
    root: &std::path::Path,
    start: u16,
    end: u16,
    block_count: u16,
) -> Result<PathBuf> {
    let manifest_path = root.join("manifest.json");
    let out_path = root.join(format!("stitched-{start}-{end}.gguf"));
    if out_path.exists() {
        return Ok(out_path);
    }
    let root_t   = root.to_path_buf();
    let manifest = manifest_path.clone();
    let out      = out_path.clone();
    tokio::task::spawn_blocking(move || -> Result<PathBuf> {
        use intelnav_model_store::{stitch_subset, Manifest, StitchRange};
        let bytes = std::fs::read(&manifest)?;
        let m = Manifest::from_json_bytes(&bytes)?;
        let range = StitchRange {
            start: start as u32,
            end:   end as u32,
            include_embed: start == 0,
            include_head:  end == block_count,
        };
        let outcome = stitch_subset(&m, &root_t, &range, &out)?;
        Ok(outcome.path)
    })
    .await
    .map_err(|e| anyhow!("join: {e}"))?
}

async fn read_frame(sock: &mut TcpStream, buf: &mut BytesMut) -> Result<Option<Msg>> {
    loop {
        if let Some(msg) = wire::decode_frame(buf)? { return Ok(Some(msg)); }
        let had_any = !buf.is_empty();
        let n = sock.read_buf(buf).await?;
        if n == 0 {
            return if had_any {
                Err(anyhow!("truncated frame at EOF ({} bytes buffered)", buf.len()))
            } else {
                Ok(None)
            };
        }
    }
}

async fn write_frame(sock: &mut TcpStream, msg: &Msg) -> Result<()> {
    let mut out = BytesMut::with_capacity(256);
    wire::encode_frame(&mut out, msg)?;
    sock.write_all(&out).await?;
    sock.flush().await?;
    Ok(())
}
