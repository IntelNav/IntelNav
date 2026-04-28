//! Long-lived libp2p host for the CLI.
//!
//! On `chat` startup we spawn a [`Libp2pNode`], dial the configured
//! bootstrap peers, scan `<models_dir>/.shards/*/kept_ranges.json`
//! for slices we host, and start a periodic announce task that
//! republishes our (cid, range) provider records to the DHT.
//!
//! The TUI gets back a [`SwarmIndex`] handle for `/models` lookups
//! plus a join handle on the announce task (held by the
//! [`SwarmHandle`]) so it can be aborted on quit.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result};
use tokio::task::JoinHandle;
use tracing::{info, warn};

use intelnav_core::Config;
use intelnav_crypto::Identity;
use intelnav_net::{
    identity_to_keypair, spawn_libp2p_node, Libp2pNode, Multiaddr, ProviderRecord, SwarmIndex,
};

use crate::contribute::{DisabledRanges, KeptRanges};
use crate::control::{self, ControlServer, ControlState, SliceStatus};

const ANNOUNCE_INTERVAL: Duration = Duration::from_secs(5 * 60);
const DRAIN_GRACE:       Duration = Duration::from_secs(5 * 60);
const WATCHDOG_INTERVAL: Duration = Duration::from_secs(30);

pub struct SwarmHandle {
    pub node:        Arc<Libp2pNode>,
    pub index:       SwarmIndex,
    /// Daemon-only: mutable view of which slices are Announcing /
    /// Draining / Stopped. The control RPC mutates it; the announce
    /// loop honours it. `None` for chat clients.
    pub control:     Option<Arc<ControlState>>,
    /// Background re-announce task. `_` because we hold it for the
    /// drop side-effect (`abort` on Drop kills the task cleanly).
    pub _announce:   Option<JoinHandle<()>>,
    /// Drain-timeout watchdog: force-stops Draining slices whose grace
    /// period has elapsed.
    pub _watchdog:   Option<JoinHandle<()>>,
    /// Multi-shard chunk HTTP server bound to `config.chunks_addr`.
    pub _chunks:     Option<JoinHandle<()>>,
    /// Forward TCP listener bound to `config.forward_addr`.
    pub _forward:    Option<JoinHandle<()>>,
    /// Daemon-only: control RPC server. Drop removes the socket file.
    pub _control_server: Option<ControlServer>,
}

impl Drop for SwarmHandle {
    fn drop(&mut self) {
        if let Some(h) = self._announce.take() { h.abort(); }
        if let Some(h) = self._watchdog.take() { h.abort(); }
        if let Some(h) = self._chunks.take()   { h.abort(); }
        if let Some(h) = self._forward.take()  { h.abort(); }
        // _control_server::Drop runs automatically.
    }
}

/// Spawn a *client-only* libp2p host: routing table, DHT queries,
/// no announce loop, no scan of kept_ranges sidecars. Used by the
/// chat CLI so opening `/models` populates the swarm rows but
/// closing the chat doesn't take any slices off the network.
pub async fn spawn_client_only(config: &Config) -> Result<SwarmHandle> {
    spawn_inner(config, None).await
}

/// Maximum time we'll wait for in-flight chains to finish during a
/// graceful shutdown. After this, we abort and exit anyway — better
/// to drop a wedged chain than to block a logout/reboot indefinitely.
const SHUTDOWN_DRAIN_BUDGET: Duration = Duration::from_secs(30);

/// How often we re-check for active chains during the drain wait.
const SHUTDOWN_POLL: Duration = Duration::from_millis(500);

/// Run the host daemon forever. Spawns the swarm, hosts the kept
/// slices, and blocks until SIGINT (Ctrl+C) or SIGTERM (systemd stop /
/// logout / reboot). On either signal, transitions every Announcing
/// slice to Draining, waits up to [`SHUTDOWN_DRAIN_BUDGET`] for active
/// chains to finish, then exits. Used by the `intelnav-node` binary.
pub async fn serve_forever(config: &Config, models_dir: PathBuf) -> Result<()> {
    let handle = spawn(config, models_dir).await?;
    info!(
        peer_id = %handle.node.peer_id,
        "intelnav-node ready — SIGINT or SIGTERM for graceful shutdown",
    );

    wait_for_shutdown_signal().await?;
    info!("shutdown signal received — beginning graceful drain");

    // Tell the announce loop + forward listener that everything is
    // draining: stop re-announcing provider records, refuse new chain
    // sessions. In-flight chains keep streaming until they finish.
    let attached_chains = if let Some(state) = handle.control.as_ref() {
        let slices = state.snapshot_hosted().await;
        for s in &slices {
            if matches!(s.state, control::SliceStatus::Announcing) {
                state.begin_drain(&(s.cid.clone(), s.start, s.end)).await;
            }
        }
        slices.iter().map(|s| s.active_chains).sum::<u32>()
    } else { 0 };
    info!(in_flight = attached_chains, budget = ?SHUTDOWN_DRAIN_BUDGET, "draining");

    // Poll until every slice is idle or budget expires.
    if let Some(state) = handle.control.as_ref() {
        let deadline = std::time::Instant::now() + SHUTDOWN_DRAIN_BUDGET;
        while std::time::Instant::now() < deadline {
            let still_active: u32 = state.snapshot_hosted().await
                .iter().map(|s| s.active_chains).sum();
            if still_active == 0 { break; }
            tokio::time::sleep(SHUTDOWN_POLL).await;
        }
        let final_active: u32 = state.snapshot_hosted().await
            .iter().map(|s| s.active_chains).sum();
        if final_active > 0 {
            warn!(stuck = final_active, "drain budget exhausted — force-exiting with chains still attached");
        } else {
            info!("drain complete — all chains finished cleanly");
        }
    }

    drop(handle);
    Ok(())
}

/// Block until the daemon receives SIGINT or SIGTERM, whichever fires
/// first. SIGINT is what Ctrl+C sends in a foreground shell; SIGTERM
/// is what systemd, init scripts, and `kill <pid>` send.
async fn wait_for_shutdown_signal() -> Result<()> {
    use tokio::signal::unix::{signal, SignalKind};
    let mut sigterm = signal(SignalKind::terminate())
        .map_err(|e| anyhow::anyhow!("install SIGTERM handler: {e}"))?;
    tokio::select! {
        r = tokio::signal::ctrl_c() => {
            r.map_err(|e| anyhow::anyhow!("install SIGINT handler: {e}"))?;
            info!("SIGINT (Ctrl+C) received");
        }
        _ = sigterm.recv() => {
            info!("SIGTERM received");
        }
    }
    Ok(())
}

/// Spawn the host daemon: libp2p + bootstrap dials + sidecar scan +
/// 5-minute announce loop. Best-effort failures are warned, never
/// fatal — a peer that can't reach a bootstrap is still useful.
pub async fn spawn(config: &Config, models_dir: PathBuf) -> Result<SwarmHandle> {
    spawn_inner(config, Some(models_dir)).await
}

/// Internal driver: when `models_dir` is `Some`, run the announce
/// loop that publishes kept_ranges sidecars. When `None`, skip the
/// scan + announce — the caller is a read-only consumer of the DHT.
async fn spawn_inner(config: &Config, models_dir: Option<PathBuf>) -> Result<SwarmHandle> {
    let identity = load_or_generate_identity()?;
    let keypair = identity_to_keypair(&identity)
        .context("identity_to_keypair")?;
    let listen: Multiaddr = config.libp2p_listen.parse()
        .with_context(|| format!("libp2p_listen `{}` is not a valid multiaddr", config.libp2p_listen))?;
    let node = Arc::new(spawn_libp2p_node(keypair, listen).await
        .context("spawn libp2p")?);
    info!(peer_id = %node.peer_id, listen_addrs = ?node.listen_addrs, "libp2p node up");

    // Resolve the bootstrap seed list. If the user pinned one in config we
    // honour it (escape hatch for self-hosted swarms / dev). Otherwise pull
    // the curated manifest from GitHub releases — falls back to cache if
    // the network is down.
    let bootstrap: Vec<String> = if config.bootstrap.is_empty() {
        crate::bootstrap::load_seeds().await
    } else {
        config.bootstrap.clone()
    };
    for boot in &bootstrap {
        let addr: Multiaddr = match boot.parse() {
            Ok(a)  => a,
            Err(e) => { warn!(?e, %boot, "skipping malformed bootstrap"); continue; }
        };
        if let Err(e) = node.dial(addr.clone()).await {
            warn!(?e, %addr, "bootstrap dial failed");
        }
    }
    if !bootstrap.is_empty() {
        let _ = node.bootstrap().await;
    }

    let (announce, watchdog, chunks_task, forward_task, control_state, control_server) = match models_dir {
        Some(dir) => {
            let state = ControlState::new(node.peer_id.to_base58());
            // Seed control state from the on-disk sidecars so a fresh
            // boot reflects what we'll be publishing.
            let kept = scan_kept_ranges(&dir);
            for k in &kept {
                for (start, end) in &k.kept {
                    state.upsert_announcing(
                        (k.model_cid.clone(), *start, *end),
                        k.display_name.clone(),
                    ).await;
                }
            }
            let announce = spawn_announce_loop(&node, config, &dir, state.clone()).await;
            let watchdog = spawn_drain_watchdog(state.clone(), dir.clone());
            let chunks = spawn_chunk_server(config, &dir);
            let forward = spawn_forward_listener(config, state.clone(), &dir);
            let server = control::spawn_server(state.clone(), control::default_socket_path())?;
            (announce, Some(watchdog), chunks, forward, Some(state), Some(server))
        }
        None => (None, None, None, None, None, None),
    };

    let index = SwarmIndex::new(node.clone());
    Ok(SwarmHandle {
        node,
        index,
        control: control_state,
        _announce: announce,
        _watchdog: watchdog,
        _chunks: chunks_task,
        _forward: forward_task,
        _control_server: control_server,
    })
}

fn spawn_forward_listener(
    config: &Config,
    state: Arc<ControlState>,
    models_dir: &std::path::Path,
) -> Option<JoinHandle<()>> {
    let raw = config.forward_addr.as_deref()?;
    crate::forward_server::spawn(raw, state, models_dir.to_path_buf())
}

/// Spawn the multi-shard chunk HTTP server bound to `chunks_addr`. No-op
/// if the user hasn't set one (still useful — they can later set it via
/// config and restart). The server serves every shard under
/// `<models_dir>/.shards/` keyed by manifest_cid.
fn spawn_chunk_server(config: &Config, models_dir: &std::path::Path) -> Option<JoinHandle<()>> {
    let raw = config.chunks_addr.clone()?;
    let addr: std::net::SocketAddr = match raw.parse() {
        Ok(a) => a,
        Err(e) => { warn!(?e, %raw, "chunks_addr is not host:port"); return None; }
    };
    let shards_root = models_dir.join(".shards");
    Some(tokio::spawn(async move {
        if let Err(e) = intelnav_model_store::serve::serve_multi(&shards_root, addr).await {
            warn!(?e, "chunk server exited");
        }
    }))
}

/// Drain-timeout watchdog. Every [`WATCHDOG_INTERVAL`] it scans for
/// slices that have been Draining longer than [`DRAIN_GRACE`] and
/// force-stops them — releasing whatever in-flight chains were
/// holding the slice up. Without this a wedged consumer could pin a
/// host slice forever.
fn spawn_drain_watchdog(state: Arc<ControlState>, models_dir: PathBuf) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(WATCHDOG_INTERVAL);
        loop {
            tick.tick().await;
            let expired = state.expired_draining(DRAIN_GRACE).await;
            if expired.is_empty() { continue; }
            for (cid, start, end) in &expired {
                warn!(%cid, start, end, "drain grace elapsed — force-stopping slice");
            }
            // Force-stop everything draining at once (the snapshot above
            // is ground truth for "expired"; force_stop_all_draining
            // covers any extra that hit the grace between calls).
            state.force_stop_all_draining().await;
            persist_drained_slices(&state, &models_dir).await;
        }
    })
}

/// Boot the periodic announce loop. Returns `None` when the peer
/// hosts no slices on disk (nothing to publish), in which case
/// dropping the SwarmHandle has nothing to abort.
///
/// The loop re-reads `kept_ranges.json` sidecars on every tick so a
/// `Join` via the control RPC (which writes a fresh sidecar) is
/// picked up automatically. It also consults `ControlState` to skip
/// slices in Draining / Stopped so a `Leave` stops announces
/// immediately, even before the 30-min DHT TTL expires.
async fn spawn_announce_loop(
    node: &Arc<Libp2pNode>,
    config: &Config,
    models_dir: &std::path::Path,
    state: Arc<ControlState>,
) -> Option<JoinHandle<()>> {
    let initial = scan_kept_ranges(models_dir);
    if initial.is_empty() { return None; }

    let chunks = config.chunks_addr.clone();
    let forward = config.forward_addr.clone();
    let peer_id_b58 = node.peer_id.to_base58();
    let listen_strings: Vec<String> = node.listen_addrs.iter()
        .map(|m| m.to_string()).collect();

    // Initial announce so a freshly-booted peer is visible immediately.
    announce_all(
        node, &initial, &state,
        &peer_id_b58, &listen_strings,
        chunks.as_deref(), forward.as_deref(),
    ).await;

    let node_t   = node.clone();
    let chunks_t = chunks;
    let forward_t = forward;
    let models_t = models_dir.to_path_buf();
    let state_t  = state.clone();
    Some(tokio::spawn(async move {
        let mut tick = tokio::time::interval(ANNOUNCE_INTERVAL);
        // Skip the first immediate fire — initial announce already done.
        tick.tick().await;
        loop {
            tick.tick().await;
            // Persist any drains that completed since the last tick so a
            // crash mid-drain doesn't resurrect a slice the user left.
            persist_drained_slices(&state_t, &models_t).await;
            let kept = scan_kept_ranges(&models_t);
            announce_all(
                &node_t, &kept, &state_t,
                &peer_id_b58, &listen_strings,
                chunks_t.as_deref(), forward_t.as_deref(),
            ).await;
        }
    }))
}

/// Walk `<models_dir>/.shards/<cid>/kept_ranges.json` and collect
/// every (cid, range) we still host. Ranges listed in
/// `disabled_ranges.json` next to the sidecar are subtracted — that's
/// the persistent record of a user-initiated `/leave`. Missing or
/// malformed sidecars are skipped silently.
fn scan_kept_ranges(models_dir: &std::path::Path) -> Vec<KeptRanges> {
    let shards = models_dir.join(".shards");
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(&shards) else { return out; };
    for entry in rd.flatten() {
        let shard_root = entry.path();
        let path = shard_root.join("kept_ranges.json");
        let Ok(bytes) = std::fs::read(&path) else { continue; };
        let mut k: KeptRanges = match serde_json::from_slice(&bytes) {
            Ok(k)  => k,
            Err(e) => { warn!(?e, file = %path.display(), "malformed kept_ranges.json"); continue; }
        };
        let disabled = DisabledRanges::load(&shard_root);
        k.kept.retain(|&(s, e)| !disabled.contains(s, e));
        if !k.kept.is_empty() {
            out.push(k);
        }
    }
    out
}

/// After ControlState flips a slice to Stopped, persist that decision
/// so a daemon restart honours it. Called on every announce tick.
async fn persist_drained_slices(
    state: &Arc<ControlState>,
    models_dir: &std::path::Path,
) {
    let stopped = state.drain_idle().await;
    if stopped.is_empty() { return; }
    for (cid, start, end) in stopped {
        let root = models_dir.join(".shards").join(&cid);
        let mut disabled = DisabledRanges::load(&root);
        disabled.add(start, end);
        if let Err(e) = disabled.save(&root) {
            warn!(?e, %cid, start, end, "persist disabled range failed");
        } else {
            info!(%cid, start, end, "slice drained — disabled persisted");
        }
    }
}

/// Look up the manifest_cid we wrote next to the chunks. The
/// chunker writes it to `<shards>/<cid>/manifest.json`'s file path,
/// not the contents — we hash the bytes here to derive the canonical
/// CID. For chunks pulled from a peer, the source manifest cid is
/// stashed in `pull_source.json`.
fn manifest_cid_for(shard_root: &std::path::Path) -> Option<String> {
    if let Ok(bytes) = std::fs::read(shard_root.join("pull_source.json")) {
        if let Ok(src) = serde_json::from_slice::<crate::swarm_contribute::PullSource>(&bytes) {
            return Some(src.manifest_cid);
        }
    }
    let manifest_path = shard_root.join("manifest.json");
    let bytes = std::fs::read(&manifest_path).ok()?;
    Some(intelnav_model_store::Manifest::from_json_bytes(&bytes).ok()
        .map(|_| intelnav_model_store::cid::cid_string_for(&bytes))?)
}

async fn announce_all(
    node: &Libp2pNode,
    kept: &[KeptRanges],
    state: &Arc<ControlState>,
    peer_id_b58: &str,
    listen_addrs: &[String],
    chunks_addr: Option<&str>,
    forward_addr: Option<&str>,
) {
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut published = 0usize;
    for k in kept {
        let shard_root = k.kept.first()
            .and_then(|_| crate::contribute::shard_dir(&PathBuf::new(), &k.model_cid).into());
        let manifest_cid = match shard_root {
            Some(root) => manifest_cid_for(&root),
            None => None,
        };
        let record = ProviderRecord {
            peer_id:      peer_id_b58.to_string(),
            addrs:        listen_addrs.to_vec(),
            chunks_url:   chunks_addr.map(str::to_string),
            manifest_cid,
            forward_url:  forward_addr.map(str::to_string),
            minted_at:    now,
        };
        for (start, end) in &k.kept {
            // Honour the control-state veto: if a slice has been told to
            // drain, stop republishing it immediately. The DHT's 30-min
            // TTL eventually flushes any stale records.
            let key = (k.model_cid.clone(), *start, *end);
            if !state.is_announcing(&key).await {
                // Make sure new sidecars (RPC Join) get registered.
                if let Some(SliceStatus::Stopped) | None = state_status(state, &key).await {
                    state.upsert_announcing(key.clone(), k.display_name.clone()).await;
                } else {
                    continue;
                }
            }
            if let Err(e) = node.announce_shard(&k.model_cid, *start, *end, record.clone()).await {
                warn!(?e, cid = %k.model_cid, start, end, "announce_shard failed");
            } else {
                published += 1;
            }
        }
        // Also publish the model envelope (idempotent).
        let env = intelnav_net::ModelEnvelope {
            cid:           k.model_cid.clone(),
            display_name:  k.display_name.clone(),
            arch:          String::new(),
            block_count:   k.block_count as u32,
            total_bytes:   0,
            quant:         String::new(),
        };
        if let Err(e) = node.announce_model(&env).await {
            warn!(?e, cid = %k.model_cid, "announce_model failed");
        }
    }
    info!(n = published, "DHT announces published");
}

async fn state_status(state: &Arc<ControlState>, key: &control::SliceKey) -> Option<SliceStatus> {
    state.slices.lock().await.get(key).map(|e| e.state)
}

fn load_or_generate_identity() -> Result<Identity> {
    let path = directories::ProjectDirs::from("io", "intelnav", "intelnav")
        .map(|p| p.data_dir().join("peer.key"))
        .unwrap_or_else(|| PathBuf::from("./peer.key"));
    if path.exists() {
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("read {}", path.display()))?;
        let bytes = hex::decode(raw.trim())
            .with_context(|| format!("decode {}", path.display()))?;
        let seed: [u8; 32] = bytes.as_slice().try_into()
            .map_err(|_| anyhow::anyhow!("peer.key must be 32-byte hex seed"))?;
        Ok(Identity::from_seed(&seed))
    } else {
        if let Some(p) = path.parent() { let _ = std::fs::create_dir_all(p); }
        let id = Identity::generate();
        let _ = std::fs::write(&path, hex::encode(id.seed()));
        Ok(id)
    }
}
