//! Bootstrap seed list, fetched from a versioned GitHub release.
//!
//! The seed list is the small set of long-lived libp2p peers that
//! a fresh client dials to populate its routing table. Hard-coding
//! it would mean a release every time we rotate seeds; instead we
//! pull a tiny JSON manifest from a stable release tag and cache it
//! locally. Offline clients keep working off the cache until it
//! goes stale.
//!
//! On-disk cache: `<config_dir>/bootstrap.cache.json`.

use std::path::PathBuf;
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

/// Stable URL we expect the manifest to live at. The `bootstrap` tag
/// is moved (re-tagged) whenever the seed list changes; `latest` works
/// too if you'd rather rotate via release naming.
pub const BOOTSTRAP_URL: &str =
    "https://github.com/IntelNav/intelnav/releases/download/bootstrap/bootstrap.json";

/// How long a cached manifest is considered fresh. The remote also
/// declares a `ttl_seconds`; we honour the smaller of the two.
const CACHE_TTL: Duration = Duration::from_secs(7 * 24 * 60 * 60); // 7 days

/// Network timeout for the manifest fetch. Short — if GitHub is slow
/// we'd rather use the cache.
const FETCH_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BootstrapManifest {
    /// Schema version. Bump when fields change incompatibly.
    pub version: u32,
    /// When the manifest was minted (unix seconds).
    pub minted_at: u64,
    /// Server-side TTL hint (seconds).
    pub ttl_seconds: u64,
    /// Bootstrap multiaddrs ending in `/p2p/<peer_id>`.
    pub seeds: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CachedManifest {
    fetched_at: u64,
    manifest: BootstrapManifest,
}

fn cache_path() -> Option<PathBuf> {
    directories::ProjectDirs::from("io", "intelnav", "intelnav")
        .map(|p| p.config_dir().join("bootstrap.cache.json"))
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn read_cache() -> Option<CachedManifest> {
    let path = cache_path()?;
    let bytes = std::fs::read(&path).ok()?;
    serde_json::from_slice::<CachedManifest>(&bytes).ok()
}

fn write_cache(manifest: &BootstrapManifest) -> Result<()> {
    let path = cache_path().context("no config dir for bootstrap cache")?;
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p)
            .with_context(|| format!("create {}", p.display()))?;
    }
    let cached = CachedManifest { fetched_at: now_secs(), manifest: manifest.clone() };
    let bytes = serde_json::to_vec_pretty(&cached)?;
    std::fs::write(&path, bytes)
        .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

/// Cached manifest is fresh if neither our TTL nor the server's TTL
/// has elapsed.
fn is_fresh(c: &CachedManifest) -> bool {
    let age = now_secs().saturating_sub(c.fetched_at);
    let server_ttl = Duration::from_secs(c.manifest.ttl_seconds);
    age < CACHE_TTL.as_secs().min(server_ttl.as_secs())
}

/// Fetch the manifest over HTTP. Short timeout; failures bubble up
/// so the caller can fall back to the cache.
async fn fetch_remote() -> Result<BootstrapManifest> {
    let client = reqwest::Client::builder()
        .timeout(FETCH_TIMEOUT)
        .user_agent(concat!("intelnav/", env!("CARGO_PKG_VERSION")))
        .build()?;
    let resp = client.get(BOOTSTRAP_URL).send().await
        .with_context(|| format!("GET {BOOTSTRAP_URL}"))?
        .error_for_status()?;
    let manifest = resp.json::<BootstrapManifest>().await
        .context("decode bootstrap manifest")?;
    if manifest.version != 1 {
        anyhow::bail!("unsupported bootstrap manifest version {}", manifest.version);
    }
    Ok(manifest)
}

/// Get the seed list — fresh cache > live fetch > stale cache > empty.
///
/// Strategy:
/// 1. If we have a *fresh* cache, return it (no network).
/// 2. Otherwise try the network (5s timeout) and refresh the cache.
/// 3. If the network fails, fall back to whatever we had cached
///    (even if stale) so an offline user still boots with seeds.
/// 4. If there's no cache at all, return empty — the caller logs and
///    moves on. The DHT will populate via mDNS or manual peers.
pub async fn load_seeds() -> Vec<String> {
    if let Some(cached) = read_cache() {
        if is_fresh(&cached) {
            debug!(n = cached.manifest.seeds.len(), "using fresh bootstrap cache");
            return cached.manifest.seeds;
        }
    }
    match fetch_remote().await {
        Ok(manifest) => {
            if let Err(e) = write_cache(&manifest) {
                warn!(?e, "failed to write bootstrap cache");
            }
            info!(n = manifest.seeds.len(), "fetched bootstrap manifest");
            manifest.seeds
        }
        Err(e) => {
            if let Some(cached) = read_cache() {
                warn!(?e, "remote bootstrap fetch failed, using stale cache");
                cached.manifest.seeds
            } else {
                warn!(?e, "no bootstrap cache and remote fetch failed");
                Vec::new()
            }
        }
    }
}

/// Force a refresh — used by the TUI on first run and on user-initiated
/// "refresh seeds". Returns the new seed count for the status line.
pub async fn refresh() -> Result<usize> {
    let manifest = fetch_remote().await?;
    let n = manifest.seeds.len();
    write_cache(&manifest)?;
    Ok(n)
}
