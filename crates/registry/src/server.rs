//! Axum server + reaper task (spec §3, §5).

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use axum::routing::{get, post};
use axum::Router;
use tower_http::trace::TraceLayer;
use tracing::info;

use crate::api;
use crate::manifest::Manifest;
use crate::state::RegistryState;

#[derive(Clone, Debug)]
pub struct RegistryConfig {
    pub bind:              SocketAddr,
    pub manifest_path:     PathBuf,
    pub reaper_interval_s: u64,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            bind:              "0.0.0.0:8787".parse().unwrap(),
            manifest_path:     PathBuf::from("manifest.toml"),
            reaper_interval_s: 10,
        }
    }
}

pub fn router(state: Arc<RegistryState>) -> Router {
    Router::new()
        .route("/v1/shards/health",                  get(api::get_health))
        .route("/v1/shards/:model",                  get(api::get_model))
        .route("/v1/shards/:model/assign",           post(api::post_assign))
        .route("/v1/shards/:model/:part/claim",      post(api::post_claim))
        .route("/v1/shards/:model/:part/heartbeat",  post(api::post_heartbeat))
        .route("/v1/shards/:model/:part/release",    post(api::post_release))
        .route("/v1/shards/:model/:part/peers",      get(api::get_peers))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

pub async fn serve(cfg: RegistryConfig) -> Result<()> {
    let manifest = Manifest::load(&cfg.manifest_path)
        .with_context(|| format!("loading manifest {}", cfg.manifest_path.display()))?;
    info!(
        model_cid = %manifest.model.cid,
        parts     = manifest.parts.len(),
        "manifest loaded"
    );

    let state = Arc::new(RegistryState::new(manifest));

    // reaper: periodic eviction + directive recompute
    let reaper_state = Arc::clone(&state);
    let reaper_period = Duration::from_secs(cfg.reaper_interval_s);
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(reaper_period);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            ticker.tick().await;
            reaper_state.tick();
        }
    });

    let app = router(state);
    let listener = tokio::net::TcpListener::bind(cfg.bind)
        .await
        .with_context(|| format!("binding {}", cfg.bind))?;
    info!(bind = %cfg.bind, "registry listening");
    axum::serve(listener, app).await.context("axum serve")?;
    Ok(())
}
