//! Axum handlers for the registry HTTP API (spec §3).

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};
use tracing::warn;

use intelnav_core::Role;

use crate::envelope::{now_unix, SignedEnvelope};
use crate::state::{AssignError, ClaimError, HeartbeatError, ReleaseError, RegistryState};

pub type Shared = Arc<RegistryState>;

// ----------------------------------------------------------------------
//  small helpers
// ----------------------------------------------------------------------

#[derive(Serialize)]
pub struct ErrBody<'a> { pub error: &'a str }

fn err(code: StatusCode, msg: &str) -> (StatusCode, Json<ErrBody<'_>>) {
    (code, Json(ErrBody { error: msg }))
}

fn verify_or_unauthorized(
    body: &SignedEnvelope,
    model_cid: &str,
    part_id: &str,
    replay_window_s: u32,
) -> Result<String, (StatusCode, Json<ErrBody<'static>>)> {
    match body.verify(model_cid, part_id, replay_window_s, now_unix()) {
        Ok(_) => Ok(body.envelope.peer_id.clone()),
        Err(e) => {
            warn!("envelope rejected: {e}");
            Err((StatusCode::UNAUTHORIZED, Json(ErrBody { error: "bad envelope" })))
        }
    }
}

// ----------------------------------------------------------------------
//  GET /v1/shards/:model
// ----------------------------------------------------------------------

#[derive(Serialize)]
pub struct ModelSnapshot {
    pub cid:          String,
    pub name:         String,
    pub quant:        intelnav_core::Quant,
    pub total_layers: u16,
    pub parts:        Vec<PartSnapshot>,
}

#[derive(Serialize)]
pub struct PartSnapshot {
    pub id:                   String,
    pub layer_range:          [u16; 2],
    pub weight_url:           String,
    pub sha256:               String,
    pub size_bytes:           u64,
    pub desired_k:            u8,
    pub counted_volunteers:   usize,    // "toward k" count
    pub live_total:           usize,
    pub peers:                Vec<PeerSummary>,
}

#[derive(Serialize)]
pub struct PeerSummary {
    pub peer_id: String,
    pub role:    Role,
    pub status:  crate::state::SeederStatus,
}

pub async fn get_model(
    State(state): State<Shared>,
    Path(cid): Path<String>,
) -> Result<Json<ModelSnapshot>, (StatusCode, Json<ErrBody<'static>>)> {
    let manifest = state.manifest_snapshot();
    if manifest.model.cid != cid {
        return Err(err(StatusCode::NOT_FOUND, "unknown model cid"));
    }
    let counts = state.replication_counts();

    let parts = manifest.parts.iter().map(|p| {
        let (counted, live) = counts.get(&p.id).copied().unwrap_or((0, 0));
        let peers = state.seeders_of(&p.id).into_iter()
            .map(|s| PeerSummary { peer_id: s.peer_id, role: s.role, status: s.status })
            .collect();
        PartSnapshot {
            id:                 p.id.clone(),
            layer_range:        p.layer_range,
            weight_url:         p.weight_url.clone(),
            sha256:             p.sha256.clone(),
            size_bytes:         p.size_bytes,
            desired_k:          manifest.desired_k_of(p),
            counted_volunteers: counted,
            live_total:         live,
            peers,
        }
    }).collect();

    Ok(Json(ModelSnapshot {
        cid:          manifest.model.cid.clone(),
        name:         manifest.model.name.clone(),
        quant:        manifest.model.quant,
        total_layers: manifest.model.total_layers,
        parts,
    }))
}

// ----------------------------------------------------------------------
//  POST /v1/shards/:model/assign
// ----------------------------------------------------------------------

#[derive(Deserialize)]
pub struct AssignReq {
    pub peer_id:    String,
    pub role:       Role,
    pub vram_bytes: u64,
}

pub async fn post_assign(
    State(state): State<Shared>,
    Path(cid): Path<String>,
    Json(req): Json<AssignReq>,
) -> impl IntoResponse {
    let manifest = state.manifest_snapshot();
    if manifest.model.cid != cid {
        return err(StatusCode::NOT_FOUND, "unknown model cid").into_response();
    }
    match state.try_assign(&req.peer_id, req.role, req.vram_bytes) {
        Ok(out) => Json(out).into_response(),
        Err(AssignError::RateLimited) =>
            err(StatusCode::TOO_MANY_REQUESTS, "rate limited").into_response(),
        Err(AssignError::NoPartFits) =>
            err(StatusCode::UNPROCESSABLE_ENTITY, "no part fits this peer").into_response(),
    }
}

// ----------------------------------------------------------------------
//  POST /v1/shards/:model/:part/claim
// ----------------------------------------------------------------------

#[derive(Deserialize)]
pub struct ClaimReq {
    pub envelope: crate::envelope::Envelope,
    pub sig:      String,
    pub role:     Role,
}

pub async fn post_claim(
    State(state): State<Shared>,
    Path((cid, part)): Path<(String, String)>,
    Json(req): Json<ClaimReq>,
) -> impl IntoResponse {
    let manifest = state.manifest_snapshot();
    if manifest.model.cid != cid {
        return err(StatusCode::NOT_FOUND, "unknown model cid").into_response();
    }
    let se = SignedEnvelope { envelope: req.envelope, sig: req.sig };
    let peer = match verify_or_unauthorized(&se, &cid, &part, manifest.defaults.replay_window_s) {
        Ok(p) => p,
        Err(resp) => return resp.into_response(),
    };
    match state.try_claim(&part, &peer, req.role) {
        Ok(()) => (StatusCode::OK, Json(serde_json::json!({"ok": true}))).into_response(),
        Err(ClaimError::UnknownPart) =>
            err(StatusCode::NOT_FOUND, "unknown part").into_response(),
        Err(ClaimError::AlreadyLive) =>
            err(StatusCode::CONFLICT, "already live").into_response(),
    }
}

// ----------------------------------------------------------------------
//  POST /v1/shards/:model/:part/heartbeat
// ----------------------------------------------------------------------

#[derive(Deserialize)]
pub struct HeartbeatReq {
    pub envelope: crate::envelope::Envelope,
    pub sig:      String,
}

pub async fn post_heartbeat(
    State(state): State<Shared>,
    Path((cid, part)): Path<(String, String)>,
    Json(req): Json<HeartbeatReq>,
) -> impl IntoResponse {
    let manifest = state.manifest_snapshot();
    if manifest.model.cid != cid {
        return err(StatusCode::NOT_FOUND, "unknown model cid").into_response();
    }
    let se = SignedEnvelope { envelope: req.envelope, sig: req.sig };
    let peer = match verify_or_unauthorized(&se, &cid, &part, manifest.defaults.replay_window_s) {
        Ok(p) => p,
        Err(resp) => return resp.into_response(),
    };
    match state.try_heartbeat(&part, &peer) {
        Ok(out) => Json(out).into_response(),
        Err(HeartbeatError::UnknownPart) =>
            err(StatusCode::NOT_FOUND, "unknown part").into_response(),
        Err(HeartbeatError::NotLive) =>
            err(StatusCode::GONE, "peer not in live set").into_response(),
    }
}

// ----------------------------------------------------------------------
//  POST /v1/shards/:model/:part/release
// ----------------------------------------------------------------------

#[derive(Deserialize)]
pub struct ReleaseReq {
    pub envelope: crate::envelope::Envelope,
    pub sig:      String,
}

pub async fn post_release(
    State(state): State<Shared>,
    Path((cid, part)): Path<(String, String)>,
    Json(req): Json<ReleaseReq>,
) -> impl IntoResponse {
    let manifest = state.manifest_snapshot();
    if manifest.model.cid != cid {
        return err(StatusCode::NOT_FOUND, "unknown model cid").into_response();
    }
    let se = SignedEnvelope { envelope: req.envelope, sig: req.sig };
    let peer = match verify_or_unauthorized(&se, &cid, &part, manifest.defaults.replay_window_s) {
        Ok(p) => p,
        Err(resp) => return resp.into_response(),
    };
    match state.try_release(&part, &peer) {
        Ok(()) => (StatusCode::OK, Json(serde_json::json!({"ok": true}))).into_response(),
        Err(ReleaseError::UnknownPart) =>
            err(StatusCode::NOT_FOUND, "unknown part").into_response(),
        Err(ReleaseError::NotLive) =>
            err(StatusCode::GONE, "peer not in live set").into_response(),
    }
}

// ----------------------------------------------------------------------
//  GET /v1/shards/:model/:part/peers
// ----------------------------------------------------------------------

pub async fn get_peers(
    State(state): State<Shared>,
    Path((cid, part)): Path<(String, String)>,
) -> impl IntoResponse {
    let manifest = state.manifest_snapshot();
    if manifest.model.cid != cid {
        return err(StatusCode::NOT_FOUND, "unknown model cid").into_response();
    }
    if manifest.part(&part).is_none() {
        return err(StatusCode::NOT_FOUND, "unknown part").into_response();
    }
    let peers: Vec<PeerSummary> = state.seeders_of(&part).into_iter()
        .map(|s| PeerSummary { peer_id: s.peer_id, role: s.role, status: s.status })
        .collect();
    Json(peers).into_response()
}

// ----------------------------------------------------------------------
//  GET /v1/shards/health
// ----------------------------------------------------------------------

pub async fn get_health(State(state): State<Shared>) -> Json<serde_json::Value> {
    let manifest = state.manifest_snapshot();
    let counts = state.replication_counts();
    Json(serde_json::json!({
        "status":      "ok",
        "model_cid":   manifest.model.cid,
        "model_name":  manifest.model.name,
        "parts":       manifest.parts.iter().map(|p| {
            let (counted, live) = counts.get(&p.id).copied().unwrap_or((0, 0));
            serde_json::json!({
                "id": p.id,
                "counted_volunteers": counted,
                "live_total": live,
                "desired_k": manifest.desired_k_of(p),
            })
        }).collect::<Vec<_>>(),
    }))
}
