//! `pipe_peer` — tail-half pipeline peer, localhost smoke.
//!
//! Loads a GGUF, owns layer range `[start..end)`, binds a TCP port, and
//! serves one session at a time:
//!
//! ```text
//!   driver                                       peer
//!   ──────                                       ────
//!   embed + forward_range(0..start)
//!       ─── ForwardHidden (prefill) ─────────►  forward_range(start..end)
//!                                                     │
//!       ◄────────────── ForwardHidden ──────────      │ hidden out
//!   head + sample
//!   loop decode steps …
//! ```
//!
//! **Scope** (M1 smoke):
//! - Plaintext TCP, no Noise, no AES-GCM, no auth. Localhost only.
//! - One active session at a time. A second `SessionInit` aborts the first.
//! - Peer tracks `index_pos` locally from hidden-state seq length —
//!   requires in-order delivery, which TCP on localhost gives us.
//!
//! Every one of those shortcuts graduates to a real M2 concern
//! (handshake, multiplexing, resume-on-drop). For now they just have
//! to not be in the way of proving `ForwardHidden` works.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use bytes::BytesMut;
use clap::Parser;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use intelnav_core::{PeerId, SessionId};
use intelnav_core::types::Quant;
use intelnav_ggml::{decode_hidden, encode_hidden_with, Hidden, HiddenPayload};
use intelnav_runtime::probe::Probe;
use intelnav_runtime::{DevicePref, ModelHandle};
use intelnav_wire::{self as wire, Msg};

/// Live counters the probe HTTP endpoint serves. Updated from the
/// chain handler each time a forward pass completes.
#[derive(Default)]
struct RuntimeStats {
    forward_calls:    u64,
    /// Total tokens processed (sum of seq_len per forward call).
    tokens:           u64,
    /// Total wall-clock time spent inside `forward_range`, in
    /// microseconds. Divide by `tokens` × 1e6 for tok/s on this
    /// peer's slice.
    forward_us:       u64,
    /// Number of layers this peer owns — handed to the SPA so it
    /// can normalize tok/s across heterogeneous shards.
    blocks_owned:     u16,
    /// Wall-clock time of the most recent forward, ms — gives the
    /// SPA an immediate "is this peer busy right now" signal.
    last_forward_ms:  f64,
    /// Wall-clock instant of the most recent forward (ms since
    /// peer start) so the SPA can show "idle for Ns".
    last_forward_at:  u64,
    started_at:       Option<Instant>,
}

impl RuntimeStats {
    fn record(&mut self, tokens: u64, dur: std::time::Duration) {
        self.forward_calls   += 1;
        self.tokens          += tokens;
        self.forward_us      += dur.as_micros() as u64;
        self.last_forward_ms  = dur.as_secs_f64() * 1000.0;
        self.last_forward_at  = self.started_at
            .map(|t| t.elapsed().as_millis() as u64)
            .unwrap_or(0);
    }
    fn snapshot_json(&self, probe: &Probe) -> String {
        let tok_per_s = if self.forward_us == 0 {
            0.0
        } else {
            (self.tokens as f64) / (self.forward_us as f64 / 1_000_000.0)
        };
        // No GPU vendor-specific probe yet — surface backend +
        // RAM headroom so the SPA at least stops reading 0/0.
        let ram_total = probe.memory.total_bytes;
        let ram_used  = ram_total.saturating_sub(probe.memory.available_bytes);
        format!(
            "{{\"backend\":\"{backend}\",\"cores\":{cores},\
             \"ram_total\":{ram_total},\"ram_used\":{ram_used},\
             \"blocks_owned\":{blocks},\"forward_calls\":{calls},\
             \"tokens\":{tokens},\"tok_per_s\":{tok_per_s:.3},\
             \"last_forward_ms\":{last_ms:.3},\"last_forward_at_ms\":{last_at}}}",
            backend  = probe.backends.recommended,
            cores    = probe.cpu.logical_cores,
            ram_total = ram_total,
            ram_used  = ram_used,
            blocks   = self.blocks_owned,
            calls    = self.forward_calls,
            tokens   = self.tokens,
            tok_per_s = tok_per_s,
            last_ms  = self.last_forward_ms,
            last_at  = self.last_forward_at,
        )
    }
}

#[derive(Parser, Debug)]
#[command(name = "pipe_peer", about = "IntelNav pipeline peer — runs one layer slice")]
struct Args {
    /// Full GGUF file to load. Mutually exclusive with `--manifest`:
    /// use this when the peer holds the entire model on disk (the
    /// pre-Path-B mode). A peer only needs the tensors for its
    /// layer range, so `--manifest` is preferred at scale.
    #[arg(long, conflicts_with = "manifest")]
    gguf: Option<PathBuf>,

    /// Path B: points at a manifest. Two forms accepted:
    ///
    ///   * A local path to `manifest.json` — requires `--chunk-cache`.
    ///   * An HTTP(S) URL ending in `manifest.json` — the peer fetches
    ///     the manifest and the bundles it needs into a local cache
    ///     (see `--chunk-cache` to override the default location).
    ///
    /// Either way, a subset GGUF is stitched on the fly.
    #[arg(long)]
    manifest: Option<String>,

    /// Directory where fetched chunks live (or, for local manifests,
    /// where `manifest.json` + `chunks/<cid>.bin` already exist).
    /// When fetching, defaults to `~/.cache/intelnav/models/<cid>/`.
    #[arg(long)]
    chunk_cache: Option<PathBuf>,

    /// Where to place the stitched GGUF. If unset, a path under the
    /// chunk cache directory is used. Overwrites any existing file.
    #[arg(long)]
    stitched_path: Option<PathBuf>,

    /// First layer this peer owns, inclusive.
    #[arg(long, default_value_t = 0)]
    start: u16,

    /// One-past-last layer this peer owns. Pass `--end N` where N is
    /// the model's block count to own the tail.
    #[arg(long)]
    end: u16,

    /// Bind address. `127.0.0.1:7717` for localhost, `0.0.0.0:7717` for
    /// LAN peers. Takes a full `host:port` so IPv6 and non-default
    /// ports work without a separate flag.
    #[arg(long, default_value = "127.0.0.1:7717")]
    bind: SocketAddr,

    /// Backend: `auto`, `cpu`, `cuda[:N]`, `metal[:N]`.
    #[arg(long, default_value = "cpu")]
    device: DevicePref,

    /// Sideband HTTP probe port — gateway scrapes `GET /probe` to
    /// fill in real RAM/CPU/tok-per-s for the SPA. Default = bind
    /// port + 1000. Pass `0` to disable the probe entirely.
    #[arg(long)]
    probe_port: Option<u16>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(
                "warn,intelnav_runtime=info,pipe_peer=info")))
        .init();

    let args = Args::parse();
    if args.start >= args.end {
        return Err(anyhow!("invalid layer range [{}..{})", args.start, args.end));
    }
    let t0 = Instant::now();

    // Resolve the GGUF path — either the caller pointed us at a full
    // GGUF on disk, or we stitch one from chunks. `local_offset` is
    // how much to subtract from real layer indices before passing
    // them to libllama: for a stitched model it's `args.start`
    // (tensors were renumbered to 0-based); for a full GGUF it's 0.
    let (gguf_path, local_offset, mode): (PathBuf, usize, &'static str) = match (&args.gguf, &args.manifest) {
        (Some(p), None) => (p.clone(), 0, "full-gguf"),
        (None, Some(manifest_ref)) => {
            let (manifest_path, chunk_cache) = resolve_manifest(
                manifest_ref,
                args.chunk_cache.as_deref(),
                args.start as u32,
                args.end as u32,
            ).await?;
            let stitched = stitch_from_manifest(
                &manifest_path,
                &chunk_cache,
                args.start as u32,
                args.end as u32,
                args.stitched_path.as_deref(),
            )?;
            (stitched, args.start as usize, "stitched-subset")
        }
        (Some(_), Some(_)) => return Err(anyhow!("pass --gguf or --manifest, not both")),
        (None, None) => return Err(anyhow!("must pass either --gguf or --manifest")),
    };

    let mut model = ModelHandle::load(&gguf_path, args.device)?;
    let total_blocks = model.block_count();
    // In stitched mode, `total_blocks` already equals `end-start`
    // (the stitched GGUF lied about `block_count` so libllama
    // allocates the right layer vector). Validation uses the real
    // range either way.
    let expected_blocks = match mode {
        "full-gguf" => args.end as usize,
        _ /* stitched */ => (args.end - args.start) as usize,
    };
    if expected_blocks > total_blocks {
        return Err(anyhow!(
            "layer range [{}..{}) needs {} blocks but model has {}",
            args.start, args.end, expected_blocks, total_blocks
        ));
    }
    if model.pipelined().is_none() {
        return Err(anyhow!(
            "model arch {:?} is not pipelined in this build (only qwen2 today)",
            model.kind()
        ));
    }
    eprintln!(
        "pipe_peer: loaded {:?} from {} ({} blocks, mode={}) in {:.2?}, serving layers [{}..{}) on {}",
        model.kind(),
        gguf_path.display(),
        total_blocks,
        mode,
        t0.elapsed(),
        args.start,
        args.end,
        args.bind,
    );

    let listener = TcpListener::bind(args.bind).await
        .with_context(|| format!("binding {}", args.bind))?;

    // Stats handle shared between the chain handler (writes) and the
    // probe HTTP listener (reads).
    let stats = Arc::new(Mutex::new(RuntimeStats {
        blocks_owned: (args.end - args.start),
        started_at:   Some(Instant::now()),
        ..Default::default()
    }));
    let probe = Arc::new(Probe::collect());

    // Sideband HTTP probe — gateway hits GET /probe to render real
    // hardware numbers in the SPA. Default port is bind+1000 so the
    // demo script doesn't have to track another flag.
    let probe_port = args.probe_port.unwrap_or_else(|| args.bind.port() + 1000);
    if probe_port != 0 {
        let probe_addr = SocketAddr::new(args.bind.ip(), probe_port);
        let s = stats.clone(); let p = probe.clone();
        tokio::spawn(async move {
            if let Err(e) = serve_probe(probe_addr, s, p).await {
                eprintln!("pipe_peer: probe listener ended: {e:#}");
            }
        });
        eprintln!("pipe_peer: probe at http://{probe_addr}/probe");
    }

    // Serve one connection at a time. The smoke is two processes, not a
    // multi-tenant shard — a concurrent second connection just waits.
    loop {
        let (sock, peer) = listener.accept().await?;
        eprintln!("pipe_peer: accepted {peer}");
        if let Err(e) = handle(
            sock, &mut model,
            args.start as usize, args.end as usize, local_offset,
            stats.clone(),
        ).await {
            eprintln!("pipe_peer: session ended with error: {e:#}");
        } else {
            eprintln!("pipe_peer: session ended cleanly");
        }
    }
}

/// Hand-rolled HTTP/1.1 probe server. We only answer one path,
/// don't need keep-alive, don't need a router — bringing in axum
/// for two routes would more than double pipe_peer's compile time.
async fn serve_probe(
    addr:  SocketAddr,
    stats: Arc<Mutex<RuntimeStats>>,
    probe: Arc<Probe>,
) -> Result<()> {
    let lst = TcpListener::bind(addr).await
        .with_context(|| format!("binding probe on {addr}"))?;
    loop {
        let (mut sock, _peer) = match lst.accept().await {
            Ok(v) => v,
            Err(_) => continue,
        };
        let stats_c = stats.clone();
        let probe_c = probe.clone();
        tokio::spawn(async move {
            // Slurp until we see CRLFCRLF or hit a small ceiling. We
            // don't care about headers — just need to know the path.
            let mut buf = [0u8; 1024];
            let mut acc = Vec::with_capacity(256);
            loop {
                let n = match sock.read(&mut buf).await { Ok(0) | Err(_) => 0, Ok(n) => n };
                if n == 0 { break; }
                acc.extend_from_slice(&buf[..n]);
                if acc.windows(4).any(|w| w == b"\r\n\r\n") || acc.len() > 4096 { break; }
            }
            let req = String::from_utf8_lossy(&acc);
            let first_line = req.lines().next().unwrap_or("");
            let body;
            let status_line;
            if first_line.contains(" /probe") || first_line.contains(" /") {
                let json = stats_c.lock().unwrap().snapshot_json(&probe_c);
                body = json;
                status_line = "HTTP/1.1 200 OK";
            } else {
                body = "{}".into();
                status_line = "HTTP/1.1 404 Not Found";
            }
            let resp = format!(
                "{status_line}\r\n\
                 Content-Type: application/json\r\n\
                 Access-Control-Allow-Origin: *\r\n\
                 Content-Length: {}\r\n\
                 Connection: close\r\n\r\n{body}",
                body.len(),
            );
            let _ = sock.write_all(resp.as_bytes()).await;
            let _ = sock.shutdown().await;
        });
    }
}

/// Resolve the `--manifest` argument into (local manifest path,
/// chunk cache dir). For an HTTP(S) URL, fetch the manifest and the
/// bundles needed for `[real_start..real_end)` into the cache. For a
/// filesystem path, require `--chunk-cache` alongside it.
async fn resolve_manifest(
    manifest_ref: &str,
    chunk_cache: Option<&std::path::Path>,
    real_start: u32,
    real_end: u32,
) -> Result<(PathBuf, PathBuf)> {
    if manifest_ref.starts_with("http://") || manifest_ref.starts_with("https://") {
        use intelnav_model_store::{
            fetch_chunks, fetch_manifest_only, FetchOptions, FetchPlan,
        };
        let mut opts = FetchOptions::default();
        if let Some(cache) = chunk_cache {
            opts.cache_root = cache.to_path_buf();
        }
        // Pull JUST the manifest first — small JSON over the wire —
        // so we can plan the minimum set of bundles this peer needs.
        // A mid-slice peer serving 2 of 80 layers must not download
        // the other 78.
        let fetched = fetch_manifest_only(manifest_ref, &opts).await
            .context("fetching manifest")?;
        let plan = FetchPlan::for_range(&fetched.manifest, real_start, real_end);
        let out = fetch_chunks(&fetched, &plan, &opts).await
            .context("fetching per-range chunks")?;
        eprintln!(
            "pipe_peer: fetched {} ({}B downloaded, {}B reused) into {}",
            out.manifest_cid, out.bytes_downloaded, out.bytes_reused, out.dir.display(),
        );
        Ok((out.dir.join("manifest.json"), out.dir))
    } else {
        let manifest_path = PathBuf::from(manifest_ref);
        let cache = chunk_cache
            .map(|p| p.to_path_buf())
            .ok_or_else(|| anyhow!("local --manifest requires --chunk-cache"))?;
        Ok((manifest_path, cache))
    }
}

/// Stitch a subset GGUF from a manifest + chunk cache. Returns the
/// path to the stitched file on disk.
fn stitch_from_manifest(
    manifest_path: &std::path::Path,
    chunk_cache: &std::path::Path,
    real_start: u32,
    real_end: u32,
    explicit_out: Option<&std::path::Path>,
) -> Result<PathBuf> {
    use intelnav_model_store::{stitch_subset, Manifest, StitchRange};

    let bytes = std::fs::read(manifest_path)
        .with_context(|| format!("reading manifest {}", manifest_path.display()))?;
    let manifest = Manifest::from_json_bytes(&bytes)
        .with_context(|| format!("parsing manifest {}", manifest_path.display()))?;

    let range = StitchRange {
        start: real_start,
        end:   real_end,
        include_embed: real_start == 0,
        include_head:  real_end == manifest.n_layers,
    };
    let out_path = explicit_out
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| {
            chunk_cache.join(format!(
                "stitched-{}-{}.gguf",
                real_start, real_end,
            ))
        });
    let outcome = stitch_subset(&manifest, chunk_cache, &range, &out_path)
        .context("stitching subset GGUF from chunks")?;
    eprintln!(
        "pipe_peer: stitched subset {} (n_tensors={}, n_kv={}, size={})",
        outcome.path.display(), outcome.n_tensors, outcome.n_kv, outcome.size,
    );
    Ok(outcome.path)
}

async fn handle(
    mut sock: TcpStream,
    model: &mut ModelHandle,
    real_start: usize,
    real_end: usize,
    // Subtract from real layer indices before calling libllama.
    // 0 for full-GGUF; real_start for stitched (tensors renumbered).
    local_offset: usize,
    stats: Arc<Mutex<RuntimeStats>>,
) -> Result<()> {
    let mut buf = BytesMut::with_capacity(64 * 1024);
    let mut session_id: Option<SessionId> = None;
    let mut index_pos: usize = 0;
    let mut last_seq: u64 = 0;

    loop {
        // Read until we have at least one complete frame, or EOF.
        let msg = match read_frame(&mut sock, &mut buf).await? {
            Some(m) => m,
            None    => return Ok(()), // clean EOF
        };

        match msg {
            Msg::Hello { peer_id, proto_ver, .. } => {
                let reply = Msg::Hello {
                    peer_id: PeerId::new([0u8; 32]),
                    proto_ver,
                    supported_quants: vec![Quant::Q4KM, Quant::FP16],
                };
                write_frame(&mut sock, &reply).await?;
                eprintln!("pipe_peer: hello from {}, proto v{proto_ver}", peer_id.short());
            }

            Msg::SessionInit { session_id: sid, layer_range, .. } => {
                // Enforce that the driver's view of our layer range
                // matches what we actually own. Defensive — if they drift
                // the forward computes garbage silently. The driver
                // always speaks in REAL layer indices regardless of
                // whether this peer is full-GGUF or stitched.
                if (layer_range.start as usize) != real_start
                    || (layer_range.end as usize) != real_end
                {
                    let reply = Msg::AbortSession {
                        session_id: sid,
                        reason: format!(
                            "peer owns [{}..{}), driver requested [{}..{})",
                            real_start, real_end, layer_range.start, layer_range.end,
                        ),
                    };
                    write_frame(&mut sock, &reply).await?;
                    continue;
                }
                // Reset KV cache and per-session counters.
                if let Some(pl) = model.pipelined() {
                    pl.reset_cache();
                } else {
                    model.reset_cache();
                }
                session_id = Some(sid);
                index_pos = 0;
                last_seq = 0;

                let ack = Msg::SessionAck {
                    session_id: sid,
                    shard_x25519_pub: [0u8; 32], // crypto deferred
                };
                write_frame(&mut sock, &ack).await?;
                eprintln!("pipe_peer: session {} opened", hex_short(&sid));
            }

            Msg::ForwardHidden { session_id: sid, seq, phase: _, dtype, shape, payload, kv_delta: _, kv_truncate_to } => {
                let cur = session_id.ok_or_else(||
                    anyhow!("ForwardHidden arrived before SessionInit"))?;
                if cur != sid {
                    return Err(anyhow!("ForwardHidden session mismatch"));
                }
                if seq <= last_seq && last_seq > 0 {
                    return Err(anyhow!("ForwardHidden out-of-order: last_seq={last_seq}, seq={seq}"));
                }
                last_seq = seq;

                // Speculative decoding may ask us to roll the KV cache
                // back before running this step. Apply first so the
                // forward sees the rolled-back state.
                if let Some(keep) = kv_truncate_to {
                    let keep = keep as usize;
                    let pl = model.pipelined().ok_or_else(|| anyhow!("model not pipelined"))?;
                    pl.truncate_kv_to(keep)?;
                    index_pos = keep;
                }

                let wire_payload = HiddenPayload { dtype, shape, bytes: payload };
                let (in_shape, in_data) = decode_hidden(&wire_payload)?;
                let hidden = Hidden::new(
                    in_data,
                    vec![in_shape[0] as usize, in_shape[1] as usize, in_shape[2] as usize],
                )?;
                let seq_len = hidden.shape[1];

                let pl = model.pipelined().ok_or_else(|| anyhow!("model not pipelined"))?;
                // Translate REAL layer indices into the local namespace
                // libllama actually knows about. For full-GGUF mode
                // local_offset is 0 so this is a no-op; for stitched
                // mode it shifts [start..end) down to [0..end-start).
                let local_start = real_start - local_offset;
                let local_end   = real_end - local_offset;
                let t_fwd = Instant::now();
                let out = pl.forward_range(&hidden, index_pos, local_start, local_end)?;
                stats.lock().unwrap().record(seq_len as u64, t_fwd.elapsed());
                index_pos += seq_len;

                // Echo the driver-chosen dtype on the reply so the whole
                // chain stays at one wire precision per turn.
                let out_shape = [out.shape[0] as u32, out.shape[1] as u32, out.shape[2] as u32];
                let p = encode_hidden_with(&out.data, out_shape, dtype)?;
                let reply = Msg::ForwardHidden {
                    session_id: sid,
                    seq:        seq, // echo the seq so the driver can correlate
                    phase:      intelnav_wire::Phase::Decode, // meaning irrelevant for tail output
                    dtype:      p.dtype,
                    shape:      p.shape,
                    payload:    p.bytes,
                    kv_delta:   None,
                    kv_truncate_to: None,
                };
                write_frame(&mut sock, &reply).await?;
            }

            Msg::AbortSession { session_id: sid, reason } => {
                eprintln!("pipe_peer: abort session {} — {reason}", hex_short(&sid));
                return Ok(());
            }

            // Heartbeats / gossip / advertises: out-of-scope for the smoke.
            // Acknowledge by silence rather than crashing — the driver shouldn't
            // send them, but a sloppier peer might.
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

/// Block until one full CBOR-framed message arrives, or the peer closes
/// cleanly (returns `Ok(None)` on clean EOF before any bytes).
async fn read_frame(sock: &mut TcpStream, buf: &mut BytesMut) -> Result<Option<Msg>> {
    loop {
        if let Some(msg) = wire::decode_frame(buf)? {
            return Ok(Some(msg));
        }
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

fn hex_short(sid: &SessionId) -> String {
    let b = sid.as_bytes();
    format!("{:02x}{:02x}{:02x}{:02x}", b[0], b[1], b[2], b[3])
}
