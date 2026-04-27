//! `intelnav-netsim` — TCP shaper with a live HTTP control plane.
//!
//! Spawned per peer in the demo. Listens on `--bind`, forwards to
//! `--upstream`, and exposes `--control` for `GET /stats` and
//! `PATCH /config` so the SPA can reach in and twist the knobs while
//! a chat is mid-flight.
//!
//! Knob format on the CLI mirrors what `tc qdisc add netem` accepts
//! conceptually: `delay=40,jitter=4,bw=100,loss=0.01`. See
//! [`intelnav_netsim::parse_params`] for the parser.

use std::net::SocketAddr;

use anyhow::{Context, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, patch};
use axum::{Json, Router};
use clap::Parser;
use serde::Deserialize;
use tower_http::cors::CorsLayer;

use intelnav_netsim::{parse_params, LinkParams, Shaper, ShaperConfig};

#[derive(Parser, Debug)]
#[command(name = "intelnav-netsim", version)]
struct Cli {
    /// Address the shaper listens on for incoming connections.
    #[arg(long)]
    bind:     SocketAddr,
    /// Address the shaper forwards to (the real peer).
    #[arg(long)]
    upstream: SocketAddr,
    /// HTTP control-plane bind. SPA hits this for stats + live tuning.
    #[arg(long)]
    control:  SocketAddr,
    /// Tier label, e.g. `LAN`, `Metro`, `WAN`, `Lossy WAN`.
    #[arg(long, default_value = "")]
    label:    String,
    /// Forward leg params (`delay=40,jitter=4,bw=100,loss=0.01`).
    /// Empty string = perfect link.
    #[arg(long, default_value = "")]
    forward:  String,
    /// Reverse leg params. Defaults to mirroring `--forward`.
    #[arg(long)]
    reverse:  Option<String>,
}

#[derive(Clone)]
struct AppState {
    shaper: Shaper,
}

#[derive(Deserialize)]
struct PatchBody {
    /// `Some(...)` replaces the leg's params; `None` leaves it alone.
    #[serde(default)]
    forward: Option<LinkParams>,
    #[serde(default)]
    reverse: Option<LinkParams>,
    #[serde(default)]
    label:   Option<String>,
}

async fn stats(State(s): State<AppState>) -> impl IntoResponse {
    Json(s.shaper.snapshot().await)
}

async fn patch_config(
    State(s): State<AppState>,
    Json(body): Json<PatchBody>,
) -> impl IntoResponse {
    if let Some(p) = body.forward { s.shaper.set_link(true,  p).await; }
    if let Some(p) = body.reverse { s.shaper.set_link(false, p).await; }
    if let Some(l) = body.label   { s.shaper.set_label(l).await; }
    StatusCode::NO_CONTENT
}

async fn run_control(addr: SocketAddr, shaper: Shaper) -> Result<()> {
    let app = Router::new()
        .route("/stats",  get(stats))
        .route("/config", patch(patch_config))
        .layer(CorsLayer::permissive())
        .with_state(AppState { shaper });
    let lst = tokio::net::TcpListener::bind(addr).await
        .with_context(|| format!("bind control plane on {addr}"))?;
    tracing::info!(%addr, "netsim control listening");
    axum::serve(lst, app).await?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();
    let forward = parse_params(&cli.forward).context("parsing --forward")?;
    let reverse = match cli.reverse.as_deref() {
        Some(s) => parse_params(s).context("parsing --reverse")?,
        None    => forward.clone(),
    };

    let cfg = ShaperConfig {
        upstream: cli.upstream,
        forward,
        reverse,
        label: cli.label,
    };

    let shaper = Shaper::new(cfg);
    let bind = cli.bind;
    let ctrl = cli.control;
    let s_ctrl = shaper.clone();

    tracing::info!(%bind, upstream = %cli.upstream, control = %ctrl,
                   "starting intelnav-netsim");

    let serve  = tokio::spawn(async move { shaper.serve(bind).await });
    let ctrl_h = tokio::spawn(async move { run_control(ctrl, s_ctrl).await });

    tokio::select! {
        r = serve  => r??,
        r = ctrl_h => r??,
    }
    Ok(())
}
