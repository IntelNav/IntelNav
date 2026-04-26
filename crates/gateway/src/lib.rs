//! `intelnav-gateway` — OpenAI-compatible HTTP surface.
//!
//! Paper §10: "A deployment is only adopted if it is easy to adopt."
//!
//! Endpoints implemented here:
//!
//! | Method | Path                      | Purpose                                |
//! | ------ | ------------------------- | -------------------------------------- |
//! | POST   | `/v1/chat/completions`    | Streamed chat; proxies to upstream.    |
//! | GET    | `/v1/models`              | Union of upstream + P2P-discovered.    |
//! | GET    | `/v1/network/peers`       | Every known peer across directories.   |
//! | GET    | `/v1/network/health`      | Gateway liveness + counts.             |
//! | GET    | `/`                       | Friendly banner.                       |
//!
//! The `intelnav` request extension (paper §10) is parsed and surfaced to
//! the route planner; today it is honored best-effort and logged.

#![forbid(unsafe_code)]

pub mod api;
pub mod server;
pub mod state;

pub use server::run;
pub use state::GatewayState;
