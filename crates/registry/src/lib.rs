//! `intelnav-registry` — HTTP shard registry service.
//!
//! See [`specs/shard-registry-v1.md`](../../../specs/shard-registry-v1.md)
//! for the normative spec. This crate is the reference implementation.

#![forbid(unsafe_code)]

pub mod api;
pub mod envelope;
pub mod gguf;
pub mod manifest;
pub mod server;
pub mod state;

pub use manifest::{Manifest, ManifestDefaults, ManifestModel, ManifestPart};
pub use server::{serve, RegistryConfig};
pub use state::{Directive, RegistryState, SeederEntry, SeederStatus};
