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

use crate::contribute::KeptRanges;

const ANNOUNCE_INTERVAL: Duration = Duration::from_secs(5 * 60);

pub struct SwarmHandle {
    pub node:        Arc<Libp2pNode>,
    pub index:       SwarmIndex,
    /// Background re-announce task. `_` because we hold it for the
    /// drop side-effect (`abort` on Drop kills the task cleanly).
    pub _announce:   Option<JoinHandle<()>>,
}

impl Drop for SwarmHandle {
    fn drop(&mut self) {
        if let Some(h) = self._announce.take() {
            h.abort();
        }
    }
}

/// Spawn a *client-only* libp2p host: routing table, DHT queries,
/// no announce loop, no scan of kept_ranges sidecars. Used by the
/// chat CLI so opening `/models` populates the swarm rows but
/// closing the chat doesn't take any slices off the network.
pub async fn spawn_client_only(config: &Config) -> Result<SwarmHandle> {
    spawn_inner(config, None).await
}

/// Run the host daemon forever. Spawns the swarm, hosts the kept
/// slices, and blocks on Ctrl+C. Used by the `intelnav-node` binary.
pub async fn serve_forever(config: &Config, models_dir: PathBuf) -> Result<()> {
    let handle = spawn(config, models_dir).await?;
    info!(
        peer_id = %handle.node.peer_id,
        "intelnav-node ready — Ctrl+C to stop",
    );
    tokio::signal::ctrl_c().await
        .map_err(|e| anyhow::anyhow!("install ctrl_c handler: {e}"))?;
    info!("ctrl_c received, shutting down");
    drop(handle);
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

    // Dial bootstrap peers. Each is an inline multiaddr ending in /p2p/<peer_id>.
    for boot in &config.bootstrap {
        let addr: Multiaddr = match boot.parse() {
            Ok(a)  => a,
            Err(e) => { warn!(?e, %boot, "skipping malformed bootstrap"); continue; }
        };
        if let Err(e) = node.dial(addr.clone()).await {
            warn!(?e, %addr, "bootstrap dial failed");
        }
    }
    if !config.bootstrap.is_empty() {
        let _ = node.bootstrap().await;
    }

    let announce = match models_dir {
        Some(dir) => spawn_announce_loop(&node, config, &dir).await,
        None => None,
    };

    let index = SwarmIndex::new(node.clone());
    Ok(SwarmHandle { node, index, _announce: announce })
}

/// Boot the periodic announce loop. Returns `None` when the peer
/// hosts no slices (nothing to publish), in which case dropping
/// the SwarmHandle has nothing to abort.
async fn spawn_announce_loop(
    node: &Arc<Libp2pNode>,
    config: &Config,
    models_dir: &std::path::Path,
) -> Option<JoinHandle<()>> {
    let kept = scan_kept_ranges(models_dir);
    if kept.is_empty() { return None; }

    let chunks = config.chunks_addr.clone();
    let forward = config.forward_addr.clone();
    let peer_id_b58 = node.peer_id.to_base58();
    let listen_strings: Vec<String> = node.listen_addrs.iter()
        .map(|m| m.to_string()).collect();

    // Initial announce so a freshly-booted peer is visible immediately.
    announce_all(
        node, &kept,
        &peer_id_b58, &listen_strings,
        chunks.as_deref(), forward.as_deref(),
    ).await;

    let node_t   = node.clone();
    let chunks_t = chunks;
    let forward_t = forward;
    Some(tokio::spawn(async move {
        let mut tick = tokio::time::interval(ANNOUNCE_INTERVAL);
        // Skip the first immediate fire — initial announce already done.
        tick.tick().await;
        loop {
            tick.tick().await;
            announce_all(
                &node_t, &kept,
                &peer_id_b58, &listen_strings,
                chunks_t.as_deref(), forward_t.as_deref(),
            ).await;
        }
    }))
}

/// Walk `<models_dir>/.shards/<cid>/kept_ranges.json` and collect
/// every (cid, range) we host. Missing or malformed sidecars are
/// skipped silently — a corrupted sidecar shouldn't take the swarm
/// node down.
fn scan_kept_ranges(models_dir: &std::path::Path) -> Vec<KeptRanges> {
    let shards = models_dir.join(".shards");
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(&shards) else { return out; };
    for entry in rd.flatten() {
        let path = entry.path().join("kept_ranges.json");
        let Ok(bytes) = std::fs::read(&path) else { continue; };
        match serde_json::from_slice::<KeptRanges>(&bytes) {
            Ok(k)  => out.push(k),
            Err(e) => warn!(?e, file = %path.display(), "malformed kept_ranges.json"),
        }
    }
    out
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
    peer_id_b58: &str,
    listen_addrs: &[String],
    chunks_addr: Option<&str>,
    forward_addr: Option<&str>,
) {
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
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
            if let Err(e) = node.announce_shard(&k.model_cid, *start, *end, record.clone()).await {
                warn!(?e, cid = %k.model_cid, start, end, "announce_shard failed");
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
    info!(n = kept.len(), "DHT announces published");
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
