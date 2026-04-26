//! `intelnav-core` — shared types, errors, configuration.
//!
//! This crate is dependency-free w.r.t. libp2p / crypto primitives; downstream
//! crates layer their own libraries on top of these shape-only types.

#![forbid(unsafe_code)]

pub mod config;
pub mod error;
pub mod ids;
pub mod tier;
pub mod types;

pub use config::{Config, RunMode};
pub use error::{Error, Result};
pub use ids::{ModelId, PeerId, SessionId};
pub use tier::LatencyTier;
pub use types::{Backend, CapabilityV1, LayerRange, Quant, Role, ShardRoute};
