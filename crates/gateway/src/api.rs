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
