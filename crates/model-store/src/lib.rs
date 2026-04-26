//! Content-addressed tensor chunker for IntelNav models.
//!
//! This is phase 1 of "Path B" (see `docs/dev/RUNTIME_DECISION.md` and
//! the task chain #24-#27): take a GGUF file and split it into
//! individually addressable chunks, each identified by a CIDv1 with a
//! SHA-256 multihash. A manifest ties the chunks back together.
//!
//! One chunk per tensor. The GGUF header (magic + version + counts +
//! KV metadata + tensor index) goes into a single additional "header
//! chunk" — the loader side needs its bytes to reconstruct a valid
//! GGUF on-disk image for libllama.
//!
//! CIDs are raw-codec (0x55) because we hash raw bytes — no DAG-CBOR
//! encoding on the chunks themselves. That keeps dedup trivial (same
//! bytes ⇒ same CID, across models).

pub mod bundle;
pub mod cid;
pub mod chunker;
pub mod gguf;
pub mod http;
pub mod manifest;
#[cfg(feature = "p2p")]
pub mod p2p;
#[cfg(feature = "serve")]
pub mod serve;
pub mod stitch;

pub use bundle::{classify_tensor, BundleEntry, BundleKind, BundleMember};
pub use chunker::{chunk_gguf, verify_chunks, ChunkOutcome, ChunkerOptions};
pub use http::{
    default_cache_root,
    fetch_chunks, fetch_manifest_and_chunks, fetch_manifest_only,
    FetchOptions, FetchOutcome, FetchPlan, FetchedManifest,
};
pub use manifest::Manifest;
pub use stitch::{stitch_subset, StitchOutcome, StitchRange};
