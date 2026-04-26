//! Bundle layer: group per-tensor chunks into peer-servable segments.
//!
//! A bundle is the fetch unit on the wire. One bundle per transformer
//! block, plus one `embed` and one `head`. A peer owning layer range
//! `[a, b)` fetches `embed` if `a == 0`, blocks `a..b`, and `head` if
//! `b == n_layers`.
//!
//! Per-tensor CIDs are retained inside each bundle's `members` list so
//! verification stays granular and a future migration to finer fetch
//! granularity is a one-manifest bump away.

use serde::{Deserialize, Serialize};

/// Logical bundle identity. `Embed` and `Head` are singletons; `Block`
/// carries its 0-based layer index.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BundleKind {
    Embed,
    Block(u32),
    Head,
}

impl BundleKind {
    pub fn name(&self) -> String {
        match self {
            Self::Embed => "embed".into(),
            Self::Block(n) => format!("blk.{n}"),
            Self::Head => "head".into(),
        }
    }

    /// Half-open layer range this bundle occupies, for the manifest.
    /// `None` for embed/head (which aren't transformer blocks).
    pub fn layer_range(&self) -> Option<[u32; 2]> {
        match self {
            Self::Block(n) => Some([*n, n + 1]),
            _ => None,
        }
    }
}

/// Route a tensor to its bundle based on its name. The GGUF
/// convention (consistent across llama/qwen/mistral/deepseek) is:
///
/// * `token_embd.*` → embed
/// * `blk.<N>.*`    → block N
/// * `output*`, `output_norm*` → head
///
/// Anything unrecognized falls through to `Head` so loaders still see
/// it (tokenizer-bound tensors, auxiliary norms, etc.); the manifest
/// records which bucket we chose so that downstream tools can audit
/// the routing.
pub fn classify_tensor(name: &str) -> BundleKind {
    if name.starts_with("token_embd") {
        BundleKind::Embed
    } else if let Some(rest) = name.strip_prefix("blk.") {
        // blk.<N>.<field>  — parse up to the next '.'.
        let n_str = rest.split('.').next().unwrap_or("");
        match n_str.parse::<u32>() {
            Ok(n) => BundleKind::Block(n),
            Err(_) => BundleKind::Head,
        }
    } else {
        // output.weight, output_norm.weight, rope_freqs, etc.
        BundleKind::Head
    }
}

/// A tensor's placement inside a bundle — kept verbatim in the
/// manifest for provenance and per-tensor verification.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BundleMember {
    pub name: String,
    pub shape: Vec<i64>,
    pub dtype: String,
    pub dtype_code: u32,
    /// Original GGUF tensor offset, relative to that file's tensor
    /// data blob. Preserved so a Phase-2 loader that stitches a new
    /// GGUF can either reuse these offsets (if it keeps all tensors)
    /// or recompute them.
    pub data_offset_rel: u64,
    /// Per-tensor CID (raw-codec sha2-256). Identical bytes across
    /// models ⇒ identical CID; the loader MAY verify each member
    /// after splitting a bundle back into tensors.
    pub cid: String,
    /// Byte size of this tensor (equals the size of its chunk).
    pub size: u64,
    /// Byte offset of this tensor WITHIN the bundle's concatenated
    /// bytes. Loaders need this to carve the bundle back apart.
    pub offset_in_bundle: u64,
}

/// Top-level bundle entry in the manifest.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BundleEntry {
    /// `"embed"`, `"blk.<N>"`, or `"head"`.
    pub name: String,
    /// Layer range occupied, if applicable.
    pub layer_range: Option<[u32; 2]>,
    /// CID of the concatenated bundle bytes.
    pub cid: String,
    /// Total size of the bundle on the wire.
    pub size: u64,
    /// Tensors inside this bundle, in GGUF tensor-index order.
    pub members: Vec<BundleMember>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_known_names() {
        assert_eq!(classify_tensor("token_embd.weight"), BundleKind::Embed);
        assert_eq!(classify_tensor("blk.0.attn_q.weight"), BundleKind::Block(0));
        assert_eq!(classify_tensor("blk.17.ffn_down.weight"), BundleKind::Block(17));
        assert_eq!(classify_tensor("output.weight"), BundleKind::Head);
        assert_eq!(classify_tensor("output_norm.weight"), BundleKind::Head);
        assert_eq!(classify_tensor("rope_freqs.weight"), BundleKind::Head);
    }

    #[test]
    fn block_layer_range_is_single_layer() {
        assert_eq!(BundleKind::Block(5).layer_range(), Some([5, 6]));
        assert_eq!(BundleKind::Embed.layer_range(), None);
        assert_eq!(BundleKind::Head.layer_range(), None);
    }

    #[test]
    fn names_round_trip() {
        assert_eq!(BundleKind::Embed.name(), "embed");
        assert_eq!(BundleKind::Block(3).name(), "blk.3");
        assert_eq!(BundleKind::Head.name(), "head");
    }
}
