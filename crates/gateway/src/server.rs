//! axum server wiring.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use std::time::Duration;

use intelnav_core::{Config, ModelId, Result};
use intelnav_net::{DhtDirectory, MdnsDirectory, RegistryDirectory, StaticDirectory};

use crate::api;
use crate::state::GatewayState;

/// Build the axum router.
pub fn router(state: GatewayState) -> Router {
    Router::new()
        .route("/",                     get(api::banner))
        .route("/v1/models",            get(api::list_models))
        .route("/v1/network/peers",     get(api::peers))
        .route("/v1/network/health",    get(api::health))
        .route("/v1/chat/completions",  post(api::chat_completions))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
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

    let state = GatewayState {
        config:     Arc::new(config.clone()),
        http,
        static_dir: Arc::new(StaticDirectory::new()),
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
