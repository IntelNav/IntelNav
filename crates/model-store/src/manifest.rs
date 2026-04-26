//! Manifest: the single document a peer needs to locate and verify
//! every chunk of a model.
//!
//! Serialized as pretty-printed JSON on disk. The manifest file's
//! own CID (SHA-256 of its bytes, raw codec) is the "model CID" — the
//! single handle you'd share with a friend for them to be able to
//! fetch this model from any content-addressed store.
//!
//! Layout (v2):
//!
//! * `header_chunk` — the original GGUF header bytes (magic, version,
//!   counts, KV, tensor index), addressable by its own CID. A Phase-2
//!   loader uses these bytes to learn the layout, then rewrites them
//!   for a subset of tensors before handing the assembled image to
//!   libllama.
//! * `bundles[]` — one fetchable unit per peer-servable segment
//!   (`embed`, `blk.<N>`, `head`). Each bundle's bytes are the
//!   concatenation of its member tensors, in the order they appear in
//!   the GGUF tensor index. The bundle's CID is the hash of those
//!   concatenated bytes.
//! * `bundles[].members[]` — the per-tensor provenance record: name,
//!   shape, dtype, original offset, per-tensor CID. Kept so a future
//!   phase can migrate to finer fetch granularity without manifest
//!   churn.

use serde::{Deserialize, Serialize};

use crate::bundle::BundleEntry;

/// The on-disk manifest schema version. Bump on any breaking change.
pub const MANIFEST_VERSION: u32 = 2;

/// Top-level manifest — one per model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    /// Identifier: `"intelnav-model"` so a sniffer can distinguish
    /// our manifests from other random JSON blobs.
    pub format: String,
    /// Schema version; see [`MANIFEST_VERSION`].
    pub version: u32,
    /// Human-friendly display name (copied from the source GGUF's
    /// `general.name` KV when available, else derived from the file
    /// stem). Not load-bearing — diagnostic only.
    pub name: Option<String>,
    /// Architecture tag from GGUF's `general.architecture` KV
    /// (e.g. `"qwen2"`, `"llama"`). Diagnostic only.
    pub architecture: Option<String>,
    /// Number of transformer layers (derived from `*.block_count` KV
    /// or from the highest `blk.<N>` index observed).
    pub n_layers: u32,

    /// Everything you need to rebuild an on-disk GGUF from chunks.
    pub gguf: GgufInfo,

    /// The header chunk covers bytes `[0, gguf.tensor_data_offset)`
    /// of the original file — magic + version + counts + KV + tensor
    /// index, plus any pre-alignment padding.
    pub header_chunk: Chunk,

    /// Fetchable bundles (`embed`, `blk.<N>`, `head`), in a stable
    /// order: `embed` first, then blocks in index order, then `head`.
    pub bundles: Vec<BundleEntry>,
}

/// Parameters taken from the source GGUF's header that future loader
/// phases will need to reconstruct a valid file image.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GgufInfo {
    pub gguf_version: u32,
    /// Per-tensor alignment from `general.alignment` KV (default 32
    /// if not present).
    pub alignment: u64,
    /// File offset where the tensor data blob begins — equals the
    /// size of `header.bin`, since the header chunk is literally the
    /// first `tensor_data_offset` bytes of the source file.
    pub tensor_data_offset: u64,
    /// Count of KV entries in the header (informational).
    pub n_kv: u64,
    /// Count of tensors (informational). Equal to the sum of bundle
    /// member counts.
    pub n_tensors: u64,
}

/// A chunk of bytes addressed by its SHA-256-backed CID.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Chunk {
    /// CIDv1 (raw codec, sha2-256 multihash) as the canonical base32
    /// string — e.g. `"bafkrei..."`.
    pub cid: String,
    /// Size of the chunk in bytes.
    pub size: u64,
}

impl Manifest {
    /// Serialize this manifest as pretty JSON with a trailing newline
    /// (the newline makes it git-friendly and matches most editors).
    pub fn to_json_bytes(&self) -> serde_json::Result<Vec<u8>> {
        let mut v = serde_json::to_vec_pretty(self)?;
        v.push(b'\n');
        Ok(v)
    }

    /// Deserialize a manifest from JSON bytes.
    pub fn from_json_bytes(bytes: &[u8]) -> serde_json::Result<Self> {
        serde_json::from_slice(bytes)
    }
}
