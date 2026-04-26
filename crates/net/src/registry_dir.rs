//! `RegistryDirectory` — a [`PeerDirectory`] backed by the HTTP shard registry.
//!
//! Polls `GET /v1/shards/:model` on a configurable interval, converts the
//! peer/part table into [`PeerRecord`]s, and exposes them through the
//! standard `PeerDirectory` trait. The registry is the authoritative source
//! of *role* (volunteer vs. cloud) and *live/standby* state; peer wire
//! addresses still come from mDNS/DHT.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use tracing::{debug, warn};

use intelnav_core::types::{Backend, CapabilityV1, Quant, Role, ShardRoute};
use intelnav_core::{ModelId, PeerId};

use crate::directory::{PeerDirectory, PeerRecord};

// ----------------------------------------------------------------------
//  Wire shape (subset of ModelSnapshot / PartSnapshot / PeerSummary)
// ----------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct WireModel {
    cid:          String,
    #[allow(dead_code)]
    name:         String,
    quant:        Quant,
    #[allow(dead_code)]
    total_layers: u16,
    parts:        Vec<WirePart>,
}

#[derive(Debug, Deserialize)]
struct WirePart {
    #[allow(dead_code)]
    id:          String,
    layer_range: [u16; 2],
    #[serde(default)]
    peers:       Vec<WirePeer>,
}

#[derive(Debug, Deserialize)]
struct WirePeer {
    peer_id: String,                   // 64-char hex
    role:    Role,
    status:  String,                   // "live" | "standby" | "draining" | "evicted"
}

// ----------------------------------------------------------------------
//  Directory
// ----------------------------------------------------------------------

pub struct RegistryDirectory {
    base_url:  String,
    model_id:  ModelId,
    inner:     Arc<RwLock<HashMap<PeerId, PeerRecord>>>,
}

impl RegistryDirectory {
    /// Spawn a background poller. `base_url` is the registry root
    /// (e.g. `http://127.0.0.1:8787`). `model_id` is the human-facing
    /// model label the gateway advertises.
    pub fn spawn(
        base_url: impl Into<String>,
        model_id: ModelId,
        poll_interval: Duration,
    ) -> Arc<Self> {
        let dir = Arc::new(Self {
            base_url: base_url.into(),
            model_id,
            inner:    Arc::new(RwLock::new(HashMap::new())),
        });

        let dir_clone = Arc::clone(&dir);
        tokio::spawn(async move {
            let client = reqwest::Client::builder()
                .user_agent(concat!("intelnav-registry-dir/", env!("CARGO_PKG_VERSION")))
                .timeout(Duration::from_secs(5))
                .build()
                .unwrap_or_default();
            loop {
                if let Err(e) = dir_clone.poll_once(&client).await {
                    warn!(error = %e, "registry directory poll failed");
                }
                tokio::time::sleep(poll_interval).await;
            }
        });

        dir
    }

    async fn poll_once(&self, client: &reqwest::Client) -> Result<(), String> {
        // The registry keys models by CID. The caller configures RegistryDirectory
        // with a CID (or human id that happens to be a CID). We just pass it through.
        let url = format!(
            "{}/v1/shards/{}",
            self.base_url.trim_end_matches('/'),
            self.model_id.as_str()
        );
        let resp = client.get(&url).send().await.map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("{} -> {}", url, resp.status()));
        }
        let body: WireModel = resp.json().await.map_err(|e| e.to_string())?;

        // Fold per-part peers into one PeerRecord per peer.
        let mut fresh: HashMap<PeerId, PeerRecord> = HashMap::new();
        let last_seen = now_s();
        for part in &body.parts {
            let [start, end] = part.layer_range;
            for peer in &part.peers {
                // only live peers get routed to
                if peer.status != "live" { continue; }
                let pid = match parse_peer_id(&peer.peer_id) {
                    Some(p) => p,
                    None => continue,
                };
                let route = ShardRoute { cid: body.cid.clone(), start, end };
                fresh.entry(pid)
                    .and_modify(|r| {
                        if !r.capability.layers.iter().any(|l|
                            l.cid == route.cid && l.start == route.start && l.end == route.end
                        ) {
                            r.capability.layers.push(route.clone());
                        }
                        r.last_seen = last_seen;
                    })
                    .or_insert_with(|| PeerRecord {
                        peer_id:    pid,
                        addrs:      vec![],
                        capability: CapabilityV1 {
                            peer_id:     pid,
                            backend:     Backend::LlamaCpp,
                            quants:      vec![body.quant],
                            vram_bytes:  0,
                            ram_bytes:   0,
                            tok_per_sec: 0.0,
                            max_seq:     0,
                            models:      vec![self.model_id.clone()],
                            layers:      vec![route.clone()],
                            role:        peer.role,
                        },
                        last_seen,
                    });
            }
        }

        debug!(peers = fresh.len(), part_count = body.parts.len(), "registry snapshot");
        *self.inner.write().unwrap() = fresh;
        Ok(())
    }
}

#[async_trait]
impl PeerDirectory for RegistryDirectory {
    async fn all(&self) -> Vec<PeerRecord> {
        self.inner.read().unwrap().values().cloned().collect()
    }
    fn name(&self) -> &'static str { "registry" }
}

// ----------------------------------------------------------------------

fn parse_peer_id(hex_s: &str) -> Option<PeerId> {
    if hex_s.len() != 64 { return None; }
    let mut out = [0u8; 32];
    hex::decode_to_slice(hex_s, &mut out).ok()?;
    Some(PeerId::new(out))
}

fn now_s() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}
