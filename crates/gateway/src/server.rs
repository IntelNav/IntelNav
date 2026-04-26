//! axum server wiring.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use std::time::Duration;

use intelnav_core::types::{Backend, CapabilityV1, Quant, Role, ShardRoute};
use intelnav_core::{Config, ModelId, PeerId, Result};
use intelnav_net::{DhtDirectory, MdnsDirectory, PeerRecord, RegistryDirectory, StaticDirectory};

use crate::api;
use crate::state::GatewayState;

/// Build the axum router.
pub fn router(state: GatewayState) -> Router {
    Router::new()
        // Demo SPA at `/`; the plaintext banner moves to `/banner` so
        // `curl gateway:8787` still works without HTML soup.
        .route("/",                     get(api::demo_index))
        .route("/banner",               get(api::banner))
        .route("/v1/models",            get(api::list_models))
        .route("/v1/network/peers",     get(api::peers))
        .route("/v1/network/health",    get(api::health))
        .route("/v1/swarm/topology",    get(api::swarm_topology))
        .route("/v1/chat/completions",  post(api::chat_completions))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Seed the static directory from `config.peers` + `config.splits`.
///
/// Each entry in `config.peers` (e.g. `"127.0.0.1:7717"`) becomes one
/// `PeerRecord` with a deterministic [`PeerId`] derived from the
/// address (so the same peer keeps the same id across restarts) and
/// a `ShardRoute` covering the layer range that peer owns. The split
/// list `[s1, s2, …]` against N peers maps to ranges
/// `[0..s1) [s1..s2) … [s_{N-1}..u16::MAX)` — `u16::MAX` is the open
/// "tail" sentinel until we know the model's actual block count;
/// the chain driver clamps it to the real layer count when it
/// connects.
///
/// Demo-friendly: even without a registry / mDNS / DHT, an operator
/// can spin up three local `pipe_peer`s, point the gateway at them
/// in `config.peers`, and the SPA's `/v1/swarm/topology` lights up
/// immediately.
fn seed_static_directory(dir: &StaticDirectory, config: &Config) {
    if config.peers.is_empty() {
        return;
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    // Build peer ranges from the splits list. With 3 peers + splits
    // [a,b], the ranges are [0..a) [a..b) [b..MAX).
    let n = config.peers.len();
    let mut ranges: Vec<(u16, u16)> = Vec::with_capacity(n);
    let mut prev: u16 = 0;
    for i in 0..n {
        let end = config.splits.get(i).copied().unwrap_or(u16::MAX);
        ranges.push((prev, end));
        prev = end;
    }

    let model_cid = config.registry_model.clone()
        .or_else(|| Some(config.default_model.clone()))
        .unwrap_or_else(|| "default".to_string());

    for (i, addr) in config.peers.iter().enumerate() {
        let (start, end) = ranges[i];
        let peer_id = peer_id_from_addr(addr);
        let shard = ShardRoute { cid: model_cid.clone(), start, end };
        let cap = CapabilityV1 {
            peer_id,
            backend:     Backend::LlamaCpp,
            quants:      vec![Quant::Q4KM],
            vram_bytes:  0,
            ram_bytes:   0,
            tok_per_sec: 0.0,
            max_seq:     2048,
            models:      vec![ModelId::new(model_cid.clone())],
            layers:      vec![shard],
            role:        Role::Volunteer,
        };
        dir.insert(PeerRecord {
            peer_id,
            addrs:      vec![addr.clone()],
            capability: cap,
            last_seen:  now,
        });
    }
}

/// Deterministic [`PeerId`] for a `host:port` string. Same address
/// produces the same id every run — the SPA can rely on a stable
/// short id when rendering. blake3 of the bytes; truncated to 32B.
fn peer_id_from_addr(addr: &str) -> PeerId {
    let h = blake3::hash(addr.as_bytes());
    PeerId::new(*h.as_bytes())
}

/// Start the gateway and block until cancelled.
pub async fn run(config: Config, enable_mdns: bool) -> Result<()> {
    let http = reqwest::Client::builder()
        .user_agent(concat!("intelnav/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| intelnav_core::Error::Http(e.to_string()))?;

    let registry_dir = match (&config.registry_url, &config.registry_model) {
        (Some(url), Some(model)) => {
            tracing::info!(%url, %model, "subscribing to shard registry");
            Some(RegistryDirectory::spawn(
                url.clone(),
                ModelId::new(model.clone()),
                Duration::from_secs(5),
            ))
        }
        (Some(_), None) => {
            tracing::warn!("registry_url set but registry_model is empty — skipping");
            None
        }
        _ => None,
    };

    let static_dir = Arc::new(StaticDirectory::new());
    seed_static_directory(&static_dir, &config);

    let state = GatewayState {
        config:     Arc::new(config.clone()),
        http,
        static_dir,
        dht_dir:    Arc::new(DhtDirectory::new()),
        mdns_dir:   if enable_mdns {
            match MdnsDirectory::spawn(None) {
                Ok(m)  => Some(Arc::new(m)),
                Err(e) => {
                    tracing::warn!(?e, "mdns disabled");
                    None
                }
            }
        } else {
            None
        },
        registry_dir,
        started_at: std::time::Instant::now(),
    };

    let addr: SocketAddr = config
        .gateway_bind
        .parse()
        .map_err(|e: std::net::AddrParseError| intelnav_core::Error::Config(e.to_string()))?;

    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "gateway listening");
    axum::serve(listener, router(state))
        .await
        .map_err(|e| intelnav_core::Error::Http(e.to_string()))?;
    Ok(())
}
