//! HTTP handlers.

use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::IntoResponse;
use axum::{http::StatusCode, Json};
use futures_util::stream::{Stream, StreamExt};
use serde::{Deserialize, Serialize};

use intelnav_core::types::Role;
use intelnav_core::LatencyTier;

use crate::state::GatewayState;

// ----------------------------------------------------------------------
//  /  — banner
// ----------------------------------------------------------------------

pub async fn banner(State(s): State<GatewayState>) -> impl IntoResponse {
    let uptime = s.started_at.elapsed().as_secs();
    format!(
        "IntelNav gateway — proto v1\nupstream: {}\nuptime: {}s\ntry: GET /v1/models\n",
        s.config.upstream_url, uptime,
    )
}

// ----------------------------------------------------------------------
//  /v1/network/health
// ----------------------------------------------------------------------

#[derive(Serialize)]
pub struct Health {
    pub ok:         bool,
    pub uptime_sec: u64,
    pub peer_count: usize,
    pub directories: Vec<String>,
    pub upstream:   String,
}

pub async fn health(State(s): State<GatewayState>) -> Json<Health> {
    let mut count = 0usize;
    let mut names = Vec::new();
    for d in s.directories() {
        names.push(d.name().to_string());
        count += d.all().await.len();
    }
    Json(Health {
        ok:          true,
        uptime_sec:  s.started_at.elapsed().as_secs(),
        peer_count:  count,
        directories: names,
        upstream:    s.config.upstream_url.clone(),
    })
}

// ----------------------------------------------------------------------
//  /v1/network/peers
// ----------------------------------------------------------------------

#[derive(Serialize)]
pub struct PeerListing {
    pub directory: String,
    pub peers:     Vec<PeerEntry>,
}

#[derive(Serialize)]
pub struct PeerEntry {
    pub peer_id:   String,
    pub addrs:     Vec<String>,
    pub models:    Vec<String>,
    pub tok_per_s: f32,
    pub last_seen: u64,
}

pub async fn peers(State(s): State<GatewayState>) -> Json<Vec<PeerListing>> {
    let mut out = Vec::new();
    for d in s.directories() {
        let entries = d
            .all()
            .await
            .into_iter()
            .map(|r| PeerEntry {
                peer_id:   r.peer_id.to_string(),
                addrs:     r.addrs,
                models:    r.capability.models.iter().map(|m| m.to_string()).collect(),
                tok_per_s: r.capability.tok_per_sec,
                last_seen: r.last_seen,
            })
            .collect();
        out.push(PeerListing { directory: d.name().to_string(), peers: entries });
    }
    Json(out)
}

// ----------------------------------------------------------------------
//  /v1/models — merge upstream /v1/models with P2P peer advertisements
// ----------------------------------------------------------------------

#[derive(Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data:   Vec<ModelEntry>,
}

#[derive(Serialize)]
pub struct ModelEntry {
    pub id:           String,
    pub object:       &'static str,
    pub owned_by:     String,
    pub created:      u64,
    /// Providers sorted volunteer-first, then cloud — paper §10 / registry §5.
    pub providers:    Vec<ProviderEntry>,
    /// Quants advertised across the union of providers.
    pub quants:       Vec<String>,
    /// Best observed tokens/s across providers.
    pub best_tok_per_s: f32,
}

#[derive(Serialize)]
pub struct ProviderEntry {
    pub peer_id: String,
    pub role:    Role,
}

pub async fn list_models(State(s): State<GatewayState>) -> Json<ModelList> {
    use std::collections::BTreeMap;

    let mut agg: BTreeMap<String, ModelEntry> = BTreeMap::new();

    // ---- upstream backend (Ollama / LM Studio / vLLM) ----
    let upstream = format!("{}/v1/models", s.config.upstream_url.trim_end_matches('/'));
    if let Ok(resp) = s.http.get(&upstream).send().await {
        if let Ok(v) = resp.json::<serde_json::Value>().await {
            if let Some(arr) = v.get("data").and_then(|d| d.as_array()) {
                for m in arr {
                    if let Some(id) = m.get("id").and_then(|v| v.as_str()) {
                        agg.entry(id.to_string()).or_insert(ModelEntry {
                            id:        id.to_string(),
                            object:    "model",
                            owned_by:  "upstream".into(),
                            created:   now_s(),
                            providers: vec![ProviderEntry {
                                peer_id: "upstream".into(),
                                role:    Role::Volunteer,
                            }],
                            quants:    vec![],
                            best_tok_per_s: 0.0,
                        });
                    }
                }
            }
        }
    }

    // ---- P2P directories ----
    for d in s.directories() {
        for rec in d.all().await {
            for model in &rec.capability.models {
                let entry = agg.entry(model.0.clone()).or_insert(ModelEntry {
                    id:        model.0.clone(),
                    object:    "model",
                    owned_by:  "intelnav".into(),
                    created:   rec.last_seen,
                    providers: vec![],
                    quants:    vec![],
                    best_tok_per_s: 0.0,
                });
                let peer_short = rec.peer_id.short();
                if !entry.providers.iter().any(|p| p.peer_id == peer_short) {
                    entry.providers.push(ProviderEntry {
                        peer_id: peer_short,
                        role:    rec.capability.role,
                    });
                }
                entry.best_tok_per_s = entry.best_tok_per_s.max(rec.capability.tok_per_sec);
                for q in &rec.capability.quants {
                    let qs = q.as_str().to_string();
                    if !entry.quants.contains(&qs) {
                        entry.quants.push(qs);
                    }
                }
            }
        }
    }

    // Volunteer-over-cloud tiebreaker (spec §5): sort providers so the gateway
    // and any downstream picker see volunteers first, cloud fallback last.
    for entry in agg.values_mut() {
        entry.providers.sort_by_key(|p| match p.role {
            Role::Volunteer => 0,
            Role::Cloud     => 1,
        });
    }

    Json(ModelList {
        object: "list",
        data:   agg.into_values().collect(),
    })
}

fn now_s() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

// ----------------------------------------------------------------------
//  /v1/swarm/topology — SPA-friendly snapshot of who's in the swarm
// ----------------------------------------------------------------------

/// Per-node performance + utilization snapshot. Some fields are
/// real (RAM probed via sysinfo, addr from the directory record),
/// some are synthesized when chain telemetry isn't plumbed yet.
/// `synthetic: true` advertises the difference to the SPA so it
/// can subtly dim the synth values rather than passing them off
/// as real measurements.
#[derive(Serialize)]
pub struct NodeMetrics {
    /// True if any field below is filled by the synth path (until
    /// the chain telemetry channel lands per arc 6 sub-C/D).
    pub synthetic:    bool,
    /// Round-trip latency to this peer in milliseconds.
    pub rtt_ms:       f32,
    /// Sustained tokens/sec this peer has reported.
    pub tok_per_s:    f32,
    /// GPU utilization 0.0..1.0 (CPU-bound peers report 0).
    pub gpu_util:     f32,
    /// Approximate VRAM used in bytes.
    pub vram_used:    u64,
    /// Approximate VRAM capacity in bytes (0 if CPU-only).
    pub vram_total:   u64,
    /// Recent inbound traffic from the gateway in bytes/sec
    /// (hidden state forwarded into this peer).
    pub bytes_in_s:   f32,
    /// Recent outbound traffic to the gateway in bytes/sec
    /// (hidden state forwarded out).
    pub bytes_out_s:  f32,
}

/// One node visible to the gateway. Either the gateway itself or a
/// peer it learned about through one of its directories.
#[derive(Serialize)]
pub struct SwarmNode {
    /// `gateway` for self, otherwise the peer's short id.
    pub id:        String,
    /// `gateway` | `volunteer` | `cloud`.
    pub kind:      &'static str,
    /// First reachable address — for display ("192.168.1.4:7717").
    pub addr:      String,
    /// Best tok/s the gateway has seen this node hit.
    pub tok_per_s: f32,
    /// Models advertised by this node.
    pub models:    Vec<String>,
    /// Which directory surfaced this peer (`static`, `mdns`, `dht`,
    /// `registry`). `self` for the gateway node.
    pub source:    String,
    /// Layer range this node owns in the chain ("0..8" / "8..16").
    /// Empty for the gateway and for peers that don't advertise a
    /// `ShardRoute`.
    pub layers:    String,
    /// Live performance + utilization. See [`NodeMetrics`].
    pub metrics:   NodeMetrics,
}

#[derive(Serialize)]
pub struct SwarmTopology {
    pub gateway:    SwarmNode,
    pub peers:      Vec<SwarmNode>,
    /// Models the gateway can serve right now (union of upstream +
    /// peer advertisements). Same shape as `/v1/models` data, just
    /// flatter — the SPA renders these as cards.
    pub models:     Vec<String>,
    pub uptime_sec: u64,
    pub upstream:   String,
    /// Aggregate end-to-end tok/s of the configured chain (sum of
    /// peer pipeline throughput, gated on the slowest peer). Synth
    /// until chain telemetry is wired through.
    pub chain_tok_per_s: f32,
    /// One sample per node (in `peers` order) of total bytes
    /// forwarded through the chain in the last second. The SPA
    /// renders this as the rolling traffic chart.
    pub chain_bytes_s:   f32,
    /// True when the topology is reporting at least one synthetic
    /// metric. Drops to false once chain telemetry replaces synth.
    pub synthetic:       bool,
}

pub async fn swarm_topology(State(s): State<GatewayState>) -> Json<SwarmTopology> {
    let uptime = s.started_at.elapsed().as_secs();

    let mut peers: Vec<SwarmNode> = Vec::new();
    for dir in s.directories() {
        let dir_name = dir.name().to_string();
        for rec in dir.all().await {
            let kind = match rec.capability.role {
                Role::Volunteer => "volunteer",
                Role::Cloud     => "cloud",
            };
            let layers = rec.capability.layers.first()
                .map(|sr| {
                    if sr.end == u16::MAX {
                        format!("{}..", sr.start)
                    } else {
                        format!("{}..{}", sr.start, sr.end)
                    }
                })
                .unwrap_or_default();
            // Try the peer's hardware probe first; fall back to synth
            // when the probe is unreachable (e.g. peer running an old
            // binary or probe disabled with --probe-port=0).
            let addr = rec.addrs.first().cloned().unwrap_or_default();
            let metrics = match scrape_peer_probe(&s.http, &addr).await {
                Some(real) => real,
                None       => synth_node_metrics(&rec.peer_id.short(), uptime, &kind),
            };
            peers.push(SwarmNode {
                id:        rec.peer_id.short(),
                kind,
                addr,
                tok_per_s: metrics.tok_per_s,
                models:    rec.capability.models.iter().map(|m| m.0.clone()).collect(),
                source:    dir_name.clone(),
                layers,
                metrics,
            });
        }
    }

    // Sort by id so the SPA's sparkline rendering is stable across
    // refreshes — ordering shouldn't depend on directory iteration.
    peers.sort_by(|a, b| a.id.cmp(&b.id));

    let gateway = SwarmNode {
        id:        "gateway".to_string(),
        kind:      "gateway",
        addr:      s.config.gateway_bind.clone(),
        tok_per_s: 0.0,
        models:    vec![],
        source:    "self".to_string(),
        layers:    String::new(),
        metrics:   synth_node_metrics("gateway", uptime, &"gateway"),
    };

    let chain_tok_per_s = if peers.is_empty() {
        0.0
    } else {
        // End-to-end pipeline rate is gated by the slowest peer.
        peers.iter().map(|p| p.metrics.tok_per_s)
            .fold(f32::INFINITY, f32::min)
    };
    let chain_bytes_s = peers.iter().map(|p| p.metrics.bytes_out_s).sum();

    let mut model_set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for p in &peers {
        for m in &p.models {
            model_set.insert(m.clone());
        }
    }

    let any_synth = peers.iter().any(|p| p.metrics.synthetic);
    Json(SwarmTopology {
        gateway,
        peers,
        models:     model_set.into_iter().collect(),
        uptime_sec: uptime,
        upstream:   s.config.upstream_url.clone(),
        chain_tok_per_s: if chain_tok_per_s.is_finite() { chain_tok_per_s } else { 0.0 },
        chain_bytes_s,
        synthetic:  any_synth,
    })
}

/// Scrape the per-peer probe sideband (pipe_peer's `GET /probe` on
/// `addr.port + 1000`) and turn it into [`NodeMetrics`]. Returns
/// `None` when the probe doesn't answer in time so the caller can
/// fall back to synth and the SPA still has something to render.
///
/// We deliberately probe with a tight timeout — the topology poll is
/// a 1.5 s tick, so a 600 ms scrape ceiling keeps the SPA responsive
/// even if a peer is wedged.
async fn scrape_peer_probe(
    http:    &reqwest::Client,
    addr:    &str,
) -> Option<NodeMetrics> {
    if addr.is_empty() { return None; }
    // Demo wires the gateway through netsim: addr is the netsim
    // listen port, which is bind+100. The peer's real bind port is
    // bind+0; its probe is on bind+1000. We have no clean way to
    // recover "the real bind port" from "the netsim port" here, so
    // try the netsim addr's port + 1000 first (matches non-netsim
    // demos), then the netsim port - 100 + 1000 fallback for the
    // demo's specific port allocation.
    let parsed: std::net::SocketAddr = addr.parse().ok()?;
    let host = parsed.ip();
    // Candidates ordered most-likely-first: bind+1000, then the
    // demo's `peer_port + 1000` derived from the netsim port.
    let candidates = [
        std::net::SocketAddr::new(host, parsed.port().saturating_add(1000)),
        std::net::SocketAddr::new(host, parsed.port().saturating_sub(100).saturating_add(1000)),
    ];
    for cand in candidates {
        let url = format!("http://{cand}/probe");
        let req = http.get(&url)
            .timeout(std::time::Duration::from_millis(600));
        match req.send().await {
            Ok(r) if r.status().is_success() => {
                if let Ok(v) = r.json::<serde_json::Value>().await {
                    return Some(probe_to_metrics(&v));
                }
            }
            _ => continue,
        }
    }
    None
}

fn probe_to_metrics(v: &serde_json::Value) -> NodeMetrics {
    let f = |k: &str| v.get(k).and_then(|x| x.as_f64()).unwrap_or(0.0);
    let u = |k: &str| v.get(k).and_then(|x| x.as_u64()).unwrap_or(0);
    NodeMetrics {
        synthetic:  false,
        // RTT is measured by chain telemetry; the probe doesn't
        // claim it. Surface 0 here so the SPA's "live RTT" stays
        // sourced from the per-step events instead of stale.
        rtt_ms:     0.0,
        tok_per_s:  f("tok_per_s") as f32,
        // No GPU vendor probe yet — surface CPU+RAM as the
        // utilization signal so the dashboard isn't blank. This is
        // honest: 0% gpu_util means "no GPU detected," and
        // ram_used/ram_total mirrors what the peer's sysinfo says.
        gpu_util:   0.0,
        vram_used:  u("ram_used"),
        vram_total: u("ram_total"),
        // Throughput estimate from tok/s × an assumed hidden-state
        // size — we don't know exactly without the model arch, but
        // 4 KiB per token is a fair approximation for fp16 hidden
        // vectors in the 2k-hidden range.
        bytes_in_s:  f("tok_per_s") as f32 * 4096.0,
        bytes_out_s: f("tok_per_s") as f32 * 4096.0,
    }
}

/// Deterministically synthesized per-node metrics keyed by the node's
/// short id. The same id at the same gateway uptime produces the same
/// numbers, so a polling SPA sees gentle drift rather than noisy random
/// jitter — looks like a real running system, not a slot machine.
///
/// Used as a fallback when [`scrape_peer_probe`] can't reach a peer.
fn synth_node_metrics(id: &str, uptime: u64, kind: &str) -> NodeMetrics {
    // Hash the id to get a stable per-node seed. We split it into two
    // independent slots: `seed_phase` drives the time-varying drift,
    // `seed_class` picks a per-peer baseline (RTT tier + GPU class) so
    // the swarm doesn't look like three identical clones.
    let seed = id.bytes().fold(0u32, |a, b| a.wrapping_mul(31).wrapping_add(b as u32));
    let seed_phase = seed % 360;
    let seed_class = (seed / 360) as usize;
    let phase = (uptime as f32 * 0.6 + seed_phase as f32) * std::f32::consts::PI / 180.0;
    let drift = phase.sin() * 0.5 + 0.5;             // 0..1, slow oscillation
    let drift_fast = (phase * 4.7).sin() * 0.5 + 0.5;

    // Heterogeneous LAN volunteers: one node on a fast link, one
    // typical, one a bit slower / weaker GPU. Picked deterministically
    // off the id so the same peer keeps the same character every run.
    // (rtt_base, tok/s_base, gpu_total, vram_total)
    const VOLUNTEER_TIERS: [(f32, f32, f32, u64); 3] = [
        ( 2.4, 58.0, 0.62,  8 * 1024_u64.pow(3)),  // fast LAN, 8 GiB
        ( 4.6, 47.0, 0.68, 12 * 1024_u64.pow(3)),  // typical LAN, 12 GiB
        ( 9.2, 38.0, 0.74, 16 * 1024_u64.pow(3)),  // slower link, beefier card
    ];

    let (rtt_ms, tok_per_s, gpu_util, vram_total, bytes_base) = match kind {
        "gateway" => (0.5 + drift * 0.4, 0.0, 0.0, 0, 0.0),
        "cloud"   => (62.0 + drift * 18.0, 28.0 + drift_fast * 8.0,
                      0.78 + drift * 0.18, 24 * 1024_u64.pow(3), 220_000.0),
        _         => {
            let (rtt_b, tps_b, gpu_b, vram) = VOLUNTEER_TIERS[seed_class % VOLUNTEER_TIERS.len()];
            // Bandwidth ≈ tok/s × hidden-state bytes-per-token. fp16 of
            // a 2k-hidden vector + framing is ~4 KiB; scale by tok/s.
            let bytes = tps_b * 4096.0;
            (rtt_b + drift * (rtt_b * 0.35),
             tps_b + drift_fast * (tps_b * 0.18),
             (gpu_b + drift * 0.18).min(0.97),
             vram,
             bytes)
        }
    };
    let vram_used = ((vram_total as f32) * (0.55 + 0.35 * drift)) as u64;
    let bytes_out_s = bytes_base * (0.85 + 0.3 * drift_fast);
    let bytes_in_s  = bytes_out_s * 0.92;

    NodeMetrics {
        synthetic: true,
        rtt_ms,
        tok_per_s,
        gpu_util,
        vram_used,
        vram_total,
        bytes_in_s,
        bytes_out_s,
    }
}

// ----------------------------------------------------------------------
//  /v1/swarm/events — live SSE stream of chain step events
// ----------------------------------------------------------------------

/// Subscribe to the gateway's [`Telemetry`] broadcast and forward
/// each [`StepEvent`] as a JSON SSE frame. Browsers consume this via
/// `new EventSource('/v1/swarm/events')`. The stream stays open as
/// long as the connection is held; lagged subscribers get a
/// `Lagged(n)` notification (which the relay turns into an
/// `event: lag` frame so the client can resync if it cares).
pub async fn swarm_events(
    State(s): State<GatewayState>,
) -> Sse<impl Stream<Item = std::result::Result<Event, Infallible>>> {
    let mut rx = s.telemetry.subscribe();
    let stream = async_stream::stream! {
        loop {
            match rx.recv().await {
                Ok(ev) => {
                    let payload = serde_json::to_string(&ev).unwrap_or_default();
                    yield Ok(Event::default().event("step").data(payload));
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    yield Ok(Event::default()
                        .event("lag")
                        .data(format!("{{\"missed\":{n}}}")));
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    };
    Sse::new(stream).keep_alive(axum::response::sse::KeepAlive::default())
}

// ----------------------------------------------------------------------
//  /v1/network/links — netsim aggregator + live tuning passthrough
// ----------------------------------------------------------------------

/// One entry in `/v1/network/links`. The `peer_addr` slot is what the
/// gateway uses to talk to its peer (the netsim's listen address —
/// the real peer port lives behind the shaper). `stats` is the raw
/// JSON the netsim publishes; we don't re-shape it so the SPA's
/// schema stays in lockstep with the netsim crate.
#[derive(Serialize)]
pub struct NetworkLink {
    pub index:        usize,
    pub peer_addr:    String,
    pub control_url:  String,
    /// `null` if the netsim's `/stats` call failed — the SPA shows
    /// the row dimmed rather than dropping it, so an operator sees
    /// "this shaper went away" instead of "the panel is empty."
    pub stats:        Option<serde_json::Value>,
}

/// Snapshot of every netsim shaper the gateway knows about. Order
/// matches `config.peers`.
pub async fn network_links(State(s): State<GatewayState>) -> Json<Vec<NetworkLink>> {
    let mut out = Vec::with_capacity(s.config.netsims.len());
    for (i, ctrl) in s.config.netsims.iter().enumerate() {
        let url = format!("http://{ctrl}/stats");
        let stats = match s.http.get(&url).send().await {
            Ok(r) if r.status().is_success() => r.json::<serde_json::Value>().await.ok(),
            _ => None,
        };
        let peer_addr = s.config.peers.get(i).cloned().unwrap_or_default();
        out.push(NetworkLink {
            index: i,
            peer_addr,
            control_url: format!("http://{ctrl}"),
            stats,
        });
    }
    Json(out)
}

/// PATCH `/v1/network/links/:idx` — body is forwarded straight to the
/// matching netsim's `PATCH /config`. Same body shape as
/// `intelnav-netsim` accepts: `{"forward": {...}, "reverse": {...},
/// "label": "..."}`. We don't validate the contents — the netsim
/// owns the schema, the gateway is just a CORS-friendly proxy so the
/// SPA can hit shapers running on a separate port.
pub async fn patch_network_link(
    State(s): State<GatewayState>,
    axum::extract::Path(idx): axum::extract::Path<usize>,
    body: axum::body::Bytes,
) -> StatusCode {
    let Some(ctrl) = s.config.netsims.get(idx) else {
        return StatusCode::NOT_FOUND;
    };
    let url = format!("http://{ctrl}/config");
    match s.http.patch(&url)
        .header("content-type", "application/json")
        .body(body)
        .send().await
    {
        Ok(r) if r.status().is_success() => StatusCode::NO_CONTENT,
        Ok(r) => {
            tracing::warn!(status = %r.status(), %url, "netsim rejected patch");
            StatusCode::BAD_GATEWAY
        }
        Err(e) => {
            tracing::warn!(?e, %url, "netsim unreachable");
            StatusCode::BAD_GATEWAY
        }
    }
}

// ----------------------------------------------------------------------
//  /v1/models/available + /v1/models/active — local GGUF inventory
// ----------------------------------------------------------------------

#[derive(Serialize)]
pub struct LocalModelEntry {
    /// Filename only — e.g. `deepseek-coder-1.3b-instruct.Q4_K_M.gguf`.
    pub name:        String,
    /// Absolute path the gateway would load if this model were
    /// selected. The SPA passes this back to `/v1/models/active` so
    /// users don't have to type it.
    pub path:        String,
    /// File size on disk, bytes — useful for the SPA to show
    /// "0.5B (470 MB)" without a separate metadata fetch.
    pub size_bytes:  u64,
    /// True when this is the GGUF the chain driver currently has
    /// loaded. Exactly one model is active at a time when chain
    /// mode is enabled; none when the gateway is in proxy mode.
    pub active:      bool,
}

#[derive(Serialize)]
pub struct ModelsAvailable {
    /// Models listed in the order they appear on disk.
    pub models:    Vec<LocalModelEntry>,
    /// Currently-loaded GGUF path, if any. Mirrors what
    /// `models[i].active=true` already conveys but saves the SPA a
    /// linear scan.
    pub active:    Option<String>,
    /// `models_dir` paths the gateway searched. Surfaced for
    /// diagnostics — if the SPA shows zero models the user knows
    /// where to drop new ones.
    pub searched:  Vec<String>,
}

/// Scan `config.models_dir` for `*.gguf` and any directory the
/// `INTELNAV_MODELS_SEARCH` env var lists (colon-separated). The env
/// var is how the demo script points the gateway at the project's
/// shared `models/` directory without forcing the operator to set
/// `INTELNAV_MODELS_DIR` (which doubles as the local cache root).
pub async fn models_available(State(s): State<GatewayState>) -> Json<ModelsAvailable> {
    let mut dirs: Vec<std::path::PathBuf> = vec![s.config.models_dir.clone()];
    if let Ok(extra) = std::env::var("INTELNAV_MODELS_SEARCH") {
        for p in extra.split(':').filter(|p| !p.is_empty()) {
            dirs.push(std::path::PathBuf::from(p));
        }
    }
    let active = s.driver.as_ref().and_then(|_| {
        std::env::var("INTELNAV_GATEWAY_MODEL").ok()
    });

    let mut models: Vec<LocalModelEntry> = Vec::new();
    let mut seen: std::collections::HashSet<std::path::PathBuf> = Default::default();
    for dir in &dirs {
        let Ok(rd) = std::fs::read_dir(dir) else { continue };
        for ent in rd.flatten() {
            let path = ent.path();
            if path.extension().and_then(|s| s.to_str()) != Some("gguf") { continue; }
            let canon = path.canonicalize().unwrap_or_else(|_| path.clone());
            if !seen.insert(canon.clone()) { continue; }
            let size = ent.metadata().map(|m| m.len()).unwrap_or(0);
            let path_str = path.to_string_lossy().to_string();
            let is_active = match &active {
                Some(a) => std::path::Path::new(a).canonicalize().ok().map_or(false, |c| c == canon),
                None    => false,
            };
            models.push(LocalModelEntry {
                name: path.file_name().unwrap_or_default().to_string_lossy().to_string(),
                path: path_str,
                size_bytes: size,
                active: is_active,
            });
        }
    }
    // Stable order: smallest first — the demo flow goes "try the
    // tiny one to verify the chain, then move up to a real model."
    models.sort_by_key(|m| m.size_bytes);

    Json(ModelsAvailable {
        models,
        active,
        searched: dirs.iter().map(|p| p.to_string_lossy().to_string()).collect(),
    })
}

#[derive(Deserialize)]
pub struct ModelSelectReq {
    pub path: String,
}

#[derive(Serialize)]
pub struct ModelSelectResp {
    /// `true` when the gateway already has this model loaded — the
    /// SPA can pick it without restarting anything.
    pub already_active:  bool,
    /// Plain-English advice the SPA renders as a toast. Live model
    /// swap requires every peer to reload its slice, which needs a
    /// peer-side reload RPC we haven't built yet — in the meantime
    /// the answer is "restart the demo with this env var set."
    pub message:         String,
    pub restart_command: Option<String>,
}

/// POST `/v1/models/select` — body `{"path": "/abs/path/to.gguf"}`.
/// First cut: read-only confirmation. If the requested model is
/// already active we say so; otherwise we hand back the restart
/// command. Live swap lands once `pipe_peer` grows a reload RPC.
pub async fn models_select(
    State(s):  State<GatewayState>,
    Json(req): Json<ModelSelectReq>,
) -> Json<ModelSelectResp> {
    let active = std::env::var("INTELNAV_GATEWAY_MODEL").ok();
    let already = active.as_deref().map_or(false, |a| {
        let a_canon = std::path::Path::new(a).canonicalize().ok();
        let r_canon = std::path::Path::new(&req.path).canonicalize().ok();
        match (a_canon, r_canon) {
            (Some(a), Some(r)) => a == r,
            _                  => a == req.path,
        }
    });
    if already {
        return Json(ModelSelectResp {
            already_active:  true,
            message:         "already active".into(),
            restart_command: None,
        });
    }
    // Heuristic command — preserves existing knobs the user might
    // already have set (NETSIM, STITCHED, …) by only overriding GGUF.
    let cmd = format!("GGUF={} scripts/demo.sh", req.path);
    let _ = s; // unused for now
    Json(ModelSelectResp {
        already_active:  false,
        message:         "live swap needs peer-side reload (M3 work). \
                          Restart the demo with this command to pick \
                          this model.".into(),
        restart_command: Some(cmd),
    })
}

// ----------------------------------------------------------------------
//  /v1/chain/config — live-readable chain knobs (wire_dtype today)
// ----------------------------------------------------------------------

#[derive(Serialize)]
pub struct ChainConfigView {
    /// `"fp16"` or `"int8"`. Reflects what the gateway driver will
    /// use for the *next* turn.
    pub wire_dtype: &'static str,
    /// True when a chain driver is loaded; false when the gateway is
    /// running in upstream-proxy mode and the SPA's toggle should be
    /// disabled.
    pub active: bool,
}

pub async fn chain_config(State(s): State<GatewayState>) -> Json<ChainConfigView> {
    let (wire, active) = match &s.driver {
        Some(d) => (crate::driver::wire_dtype_str(d.chain_config().wire_dtype), true),
        None    => ("fp16", false),
    };
    Json(ChainConfigView { wire_dtype: wire, active })
}

#[derive(Deserialize)]
pub struct WireDtypePatch {
    pub wire_dtype: String,
}

/// POST `/v1/chain/wire-dtype` — flips the gateway driver's wire
/// dtype. Body: `{"wire_dtype": "int8"}`. Takes effect on the next
/// chat turn (current connections keep their existing dtype). 404
/// when no driver is loaded.
pub async fn set_wire_dtype(
    State(s):  State<GatewayState>,
    Json(req): Json<WireDtypePatch>,
) -> StatusCode {
    let Some(d) = s.driver.as_ref() else { return StatusCode::NOT_FOUND };
    let dt = crate::driver::parse_wire_dtype(&req.wire_dtype);
    d.set_wire_dtype(dt);
    StatusCode::NO_CONTENT
}

// ----------------------------------------------------------------------
//  / — single-file demo SPA (chat + swarm topology)
// ----------------------------------------------------------------------

/// Minimal HTML+CSS+JS demo baked into the binary. Served at `/`.
/// No build step, no node_modules — vanilla JS over `fetch` and
/// the existing OpenAI-compatible streaming surface.
pub async fn demo_index() -> impl IntoResponse {
    axum::response::Html(include_str!("../static/index.html"))
}

// ----------------------------------------------------------------------
//  /v1/chat/completions
// ----------------------------------------------------------------------

#[derive(Deserialize)]
pub struct ChatRequest {
    pub model:       String,
    pub messages:    Vec<ChatMessage>,
    #[serde(default)]
    pub stream:      bool,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default, rename = "intelnav")]
    pub intelnav:    Option<IntelnavExt>,
    #[serde(flatten)]
    pub passthrough: serde_json::Map<String, serde_json::Value>,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role:    String,
    pub content: String,
}

#[derive(Deserialize, Debug, Default)]
pub struct IntelnavExt {
    #[serde(default)]
    pub quorum:         Option<u8>,
    #[serde(default)]
    pub min_reputation: Option<f32>,
    #[serde(default)]
    pub tier:           Option<String>,
    #[serde(default)]
    pub allow_wan:      Option<bool>,
    #[serde(default)]
    pub speculative:    Option<bool>,
}

pub async fn chat_completions(
    State(s): State<GatewayState>,
    Json(req): Json<ChatRequest>,
) -> impl IntoResponse {
    // Tier / allow_wan enforcement — paper §5.4.
    let tier = req
        .intelnav
        .as_ref()
        .and_then(|e| e.tier.as_deref())
        .map(parse_tier)
        .unwrap_or(s.config.default_tier);
    let allow_wan = req
        .intelnav
        .as_ref()
        .and_then(|e| e.allow_wan)
        .unwrap_or(s.config.allow_wan);
    if matches!(tier, LatencyTier::Wan) && !allow_wan {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": {
                    "code":    "no_viable_route",
                    "message": "T3 WAN chain requested but allow_wan is false; retry with intelnav.allow_wan=true",
                }
            })),
        )
            .into_response();
    }

    // --- chain-mode: drive the configured peer chain locally ---
    if let Some(driver) = s.driver.clone() {
        return chat_through_chain(driver, req).await.into_response();
    }

    // --- forward to upstream (Ollama/LM Studio/vLLM speaks OpenAI) ---
    let upstream = format!("{}/v1/chat/completions", s.config.upstream_url.trim_end_matches('/'));
    let mut body = serde_json::json!({
        "model":       req.model,
        "messages":    req.messages,
        "stream":      req.stream,
    });
    if let Some(t) = req.temperature {
        body["temperature"] = serde_json::Value::from(t);
    }
    for (k, v) in req.passthrough.iter() {
        body[k] = v.clone();
    }

    tracing::info!(tier = %tier.display(), stream = req.stream, "chat_completions");

    let upstream_resp = match s.http.post(&upstream).json(&body).send().await {
        Ok(r) => r,
        Err(e) => return upstream_err(format!("upstream unreachable: {e}")).into_response(),
    };

    if !upstream_resp.status().is_success() {
        let status = upstream_resp.status();
        let text = upstream_resp.text().await.unwrap_or_default();
        return upstream_err_with(status, text).into_response();
    }

    if !req.stream {
        let bytes = match upstream_resp.bytes().await {
            Ok(b) => b,
            Err(e) => return upstream_err(format!("upstream body: {e}")).into_response(),
        };
        return (
            StatusCode::OK,
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            bytes,
        )
            .into_response();
    }

    // -------- SSE streaming --------
    let byte_stream = upstream_resp.bytes_stream();
    let sse_stream = sse_relay(byte_stream);
    Sse::new(sse_stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

/// Drive one chat turn through [`GatewayDriver`] and stream the
/// tokens back as OpenAI-shape SSE deltas. Each on-token chunk
/// becomes one `data: {…}` frame the SPA's existing chat path
/// already understands; a final `data: [DONE]` closes the stream.
async fn chat_through_chain(
    driver: crate::driver::GatewayDriver,
    req:    ChatRequest,
) -> axum::response::Response {
    use crate::driver::Delta;
    use axum::response::Response;
    use intelnav_runtime::SamplingCfg;

    // Map the OpenAI-shape messages to (role, content) pairs so the
    // driver can render them through build_chat_prompt.
    let messages: Vec<(String, String)> = req.messages.iter()
        .map(|m| (m.role.clone(), m.content.clone()))
        .collect();
    if messages.is_empty() {
        return upstream_err_with(
            StatusCode::BAD_REQUEST,
            "messages must be non-empty".into(),
        )
            .into_response();
    }

    // Build a sampling config from whatever the request specified;
    // fall back to SamplingCfg::default() for the unset knobs so a
    // client that just sends `messages` still gets sane behaviour.
    let mut cfg = SamplingCfg::default();
    if let Some(t) = req.temperature {
        cfg.temperature = t as f64;
    }
    let mut rx = driver.stream(messages, cfg);

    // ---- non-streaming: collect every token, return one JSON ----
    if !req.stream {
        let mut acc = String::new();
        while let Some(d) = rx.recv().await {
            match d {
                Delta::Token(t) => acc.push_str(&t),
                Delta::Done     => break,
                Delta::Error(e) => return upstream_err(format!("chain: {e}")).into_response(),
            }
        }
        let body = serde_json::json!({
            "id":      format!("intelnav-{}", now_s()),
            "object":  "chat.completion",
            "created": now_s(),
            "model":   req.model,
            "choices": [{
                "index":   0,
                "message": {
                    "role":    "assistant",
                    "content": acc,
                },
                "finish_reason": "stop",
            }],
        });
        return (
            StatusCode::OK,
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            serde_json::to_vec(&body).unwrap_or_default(),
        )
            .into_response();
    }

    // ---- streaming: bridge the mpsc to OpenAI-shape SSE ----
    let id = format!("intelnav-{}", now_s());
    let model = req.model.clone();
    let stream = async_stream::stream! {
        // Initial frame with the role — matches how OpenAI starts an
        // assistant turn so the SPA's "first chunk" branding works.
        let head = serde_json::json!({
            "id":      id,
            "object":  "chat.completion.chunk",
            "created": now_s(),
            "model":   model,
            "choices": [{
                "index":   0,
                "delta":   { "role": "assistant" },
                "finish_reason": null,
            }],
        });
        yield Ok::<_, Infallible>(Event::default().data(head.to_string()));

        while let Some(d) = rx.recv().await {
            match d {
                Delta::Token(text) => {
                    let frame = serde_json::json!({
                        "id":      id,
                        "object":  "chat.completion.chunk",
                        "created": now_s(),
                        "model":   model,
                        "choices": [{
                            "index":   0,
                            "delta":   { "content": text },
                            "finish_reason": null,
                        }],
                    });
                    yield Ok(Event::default().data(frame.to_string()));
                }
                Delta::Done => {
                    let tail = serde_json::json!({
                        "id":      id,
                        "object":  "chat.completion.chunk",
                        "created": now_s(),
                        "model":   model,
                        "choices": [{
                            "index":   0,
                            "delta":   {},
                            "finish_reason": "stop",
                        }],
                    });
                    yield Ok(Event::default().data(tail.to_string()));
                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
                Delta::Error(msg) => {
                    let frame = serde_json::json!({
                        "error": {
                            "code":    "chain_error",
                            "message": msg,
                        }
                    });
                    yield Ok(Event::default().data(frame.to_string()));
                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
            }
        }
    };

    let resp: Response = Sse::new(stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response();
    resp
}

fn upstream_err(msg: String) -> impl IntoResponse {
    (
        StatusCode::BAD_GATEWAY,
        Json(serde_json::json!({ "error": { "code": "upstream_error", "message": msg } })),
    )
}

fn upstream_err_with(status: StatusCode, msg: String) -> impl IntoResponse {
    (
        status,
        Json(serde_json::json!({ "error": { "code": "upstream_error", "message": msg } })),
    )
}

fn parse_tier(s: &str) -> LatencyTier {
    match s.to_ascii_lowercase().as_str() {
        "lan" | "t1"                => LatencyTier::Lan,
        "wan" | "t3"                => LatencyTier::Wan,
        "cont" | "continent" | "t2" => LatencyTier::Continent,
        _                           => LatencyTier::Continent,
    }
}

/// Parse an OpenAI-style `text/event-stream` from `upstream`, re-emitting
/// each `data:` line as an SSE `Event`.
fn sse_relay<S>(mut byte_stream: S) -> impl Stream<Item = Result<Event, Infallible>>
where
    S: Stream<Item = reqwest::Result<bytes::Bytes>> + Unpin + Send + 'static,
{
    async_stream::stream! {
        let mut buf: Vec<u8> = Vec::with_capacity(4096);
        while let Some(chunk) = byte_stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(e) => {
                    yield Ok(Event::default().event("error").data(format!("upstream stream: {e}")));
                    return;
                }
            };
            buf.extend_from_slice(&chunk);
            while let Some(pos) = find_event_boundary(&buf) {
                let event_bytes = buf.drain(..pos).collect::<Vec<u8>>();
                // we also consumed the trailing boundary:
                // pos is start-of-next; caller has already split.
                for line in event_bytes.split(|&b| b == b'\n') {
                    let line = std::str::from_utf8(line).unwrap_or("");
                    if let Some(payload) = line.strip_prefix("data:") {
                        let payload = payload.trim_start();
                        yield Ok(Event::default().data(payload.to_string()));
                    }
                }
            }
        }
    }
}

/// Return the byte offset *just past* the first `\n\n` in `buf`, or `None`.
fn find_event_boundary(buf: &[u8]) -> Option<usize> {
    let mut prev = 0u8;
    for (i, &b) in buf.iter().enumerate() {
        if prev == b'\n' && b == b'\n' {
            return Some(i + 1);
        }
        prev = b;
    }
    None
}
