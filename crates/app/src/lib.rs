//! Substantive layer for IntelNav.
//!
//! This crate holds the modules the binaries (`intelnav`, `intelnav-node`)
//! call into. The binaries themselves are thin: they parse args, set up
//! tracing, and hand off here.

#![deny(unsafe_code)]

pub mod banner;
pub mod browser;
pub mod catalog;
pub mod chain_driver;
pub mod cmd;
pub mod contribute;
pub mod delta;
pub mod download;
pub mod local;
pub mod shimmer;
pub mod slash;
pub mod swarm_contribute;
pub mod swarm_node;
pub mod theme;
pub mod tui;
