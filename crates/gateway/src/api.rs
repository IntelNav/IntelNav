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
            let metrics = synth_node_metrics(&rec.peer_id.short(), uptime, &kind);
            peers.push(SwarmNode {
                id:        rec.peer_id.short(),
                kind,
                addr:      rec.addrs.first().cloned().unwrap_or_default(),
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

    Json(SwarmTopology {
        gateway,
        peers,
        models:     model_set.into_iter().collect(),
        uptime_sec: uptime,
        upstream:   s.config.upstream_url.clone(),
        chain_tok_per_s: if chain_tok_per_s.is_finite() { chain_tok_per_s } else { 0.0 },
        chain_bytes_s,
        synthetic:  true,
    })
}

/// Deterministically synthesized per-node metrics keyed by the node's
/// short id. The same id at the same gateway uptime produces the same
/// numbers, so a polling SPA sees gentle drift rather than noisy random
/// jitter — looks like a real running system, not a slot machine.
///
/// Drops out completely once the chain telemetry channel from arc 6
/// sub-C lands; the SPA's `synthetic` flag is what keeps users honest
/// about which numbers are real today.
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
