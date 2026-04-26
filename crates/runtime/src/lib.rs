//! `intelnav-runtime` — layer-range inference backend.
//!
//! Exposes a **layer-range forward pass** so peers can run slices
//! `[i..j]` of a model and pass hidden states to the next peer in the
//! pipeline.
//!
//! Backend: the IntelNav-patched libllama via [`intelnav_ggml`]. Every
//! supported architecture (qwen2 / llama / mistral / deepseek /
//! deepseek2 / mixtral) goes through `GgmlBackend` and ggml's
//! universal GPU dispatch, so the runtime crate carries no per-arch
//! Rust code.

#![forbid(unsafe_code)]

pub mod chain;
pub mod device;
pub mod generate;
pub mod ggml_backend;
pub mod model;
pub mod pipeline;
pub mod probe;
pub mod sample;
pub mod spec;
pub mod tokenizer;

pub use chain::{
    front_forward, head_all_forward, head_forward, run_turn, Chain, ChainCfg, ChainError,
};
pub use device::DevicePref;
pub use ggml_backend::GgmlBackend;
pub use generate::{build_chat_prompt, generate, qwen_chat_prompt, ChatTurn, SamplingCfg};
// Re-export so CLI callers depending only on `intelnav-runtime` can
// spell `Dtype` without pulling in `intelnav-wire` directly.
pub use intelnav_wire::Dtype;
pub use model::{sniff_arch, ModelHandle, ModelKind};
pub use pipeline::{Forwarding, Pipelined};
pub use probe::Probe;
pub use sample::{Sampler, SamplerCfg};
pub use spec::{run_turn_spec, SpecCfg};
pub use tokenizer::Tok;
