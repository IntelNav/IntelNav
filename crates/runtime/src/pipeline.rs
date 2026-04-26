//! Minimum contract every model backend has to meet.
//!
//! `Forwarding` — full forward only. Every backend implements it.
//! `Pipelined` — layer-range support for multi-peer pipelines.
//!
//! Since task #18 the trait's currency is `intelnav_ggml::Hidden`
//! (flat `Vec<f32>` + shape) rather than `candle::Tensor`. Drops the
//! per-call conversion overhead the adapter used to do, removes
//! candle from the runtime's hot path, and makes multi-batch /
//! per-sequence-length layouts tractable for M3's continuous batching.

use anyhow::Result;

pub use intelnav_ggml::Hidden;

/// Full forward only.
pub trait Forwarding {
    fn block_count(&self) -> usize;

    /// Full forward: tokens → logits. Returns `Hidden` with shape
    /// `[batch, vocab]` — one logit row per requested output position
    /// (typically the last).
    fn forward(&mut self, input_ids: &[u32], index_pos: usize) -> Result<Hidden>;

    fn reset_cache(&mut self);
}

/// Layer-range support for multi-peer pipelines. A peer hosting
/// `layers[start..end)` implements [`Pipelined::forward_range`]; the
/// first peer in the chain calls [`Pipelined::embed`], the last
/// calls [`Pipelined::head`].
///
/// **Correctness contract** — verified by the 5-scenario bit-
/// identical suite in `crates/ggml/tests/bit_identical.rs` for
/// Qwen2 and TinyLlama (max_abs_diff = 0):
///
/// ```text
/// head(forward_range(embed(ids), 0, N))
///   ==
/// head(forward_range(forward_range(embed(ids), 0, split), split, N))
/// ```
pub trait Pipelined: Forwarding {
    /// Tokens → embedding lookup. Output shape: `[batch, seq, hidden]`.
    fn embed(&mut self, input_ids: &[u32]) -> Result<Hidden>;

    /// Run layers `[start, end)` over `hidden`. `index_pos` is the
    /// absolute position of the first token in the sequence, same
    /// semantics as `llama_batch.pos[]`. Output shape matches input:
    /// `[batch, seq, hidden]`.
    fn forward_range(
        &mut self,
        hidden:    &Hidden,
        index_pos: usize,
        start:     usize,
        end:       usize,
    ) -> Result<Hidden>;

    /// Final `output_norm + lm_head`, sliced to the last sequence
    /// position. Output shape: `[batch, vocab]`.
    fn head(&mut self, hidden: &Hidden) -> Result<Hidden>;

    /// Final `output_norm + lm_head` for every sequence position.
    /// Output shape: `[batch, seq, vocab]`. Needed by speculative
    /// decoding to verify k draft tokens in a single forward.
    fn head_all(&mut self, hidden: &Hidden) -> Result<Hidden>;

    /// Truncate each layer's KV cache to `keep` entries along the
    /// sequence dimension. Rolls back rejected draft tokens.
    fn truncate_kv_to(&mut self, keep: usize) -> Result<()>;
}
