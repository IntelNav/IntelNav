//! `intelnav-runtime` — layer-range inference backend.
//!
//! The goal of this crate is to expose a **layer-range forward pass**
//! so peers can run slices `[i..j]` of a model and pass hidden states
//! to the next peer in the pipeline (paper §5, M1/M2).
//!
//! Backend: [`candle`] with its `quantized_qwen2::ModelWeights` loader.
//! We wrap `ModelWeights` rather than forking it, so upstream bug
//! fixes come along for the ride. Layer slicing happens in our
//! [`Session::forward_range`] (to be added once load is verified).

#![forbid(unsafe_code)]

pub mod chain;
pub mod device;
pub mod generate;
pub mod ggml_backend;
pub mod model;
pub mod pipeline;
pub mod probe;
pub mod spec;
pub mod tokenizer;

pub use chain::{
    front_forward, head_all_forward, head_forward, run_turn, Chain, ChainCfg, ChainError,
};
pub use device::{pick_device, pick_device_with, DevicePref};
pub use ggml_backend::GgmlBackend;
pub use generate::{build_chat_prompt, generate, qwen_chat_prompt, ChatTurn, SamplingCfg};
// Re-export so CLI callers depending only on `intelnav-runtime` can
// spell `Dtype` without pulling in `intelnav-wire` directly.
pub use intelnav_wire::Dtype;
pub use model::{sniff_arch, ModelHandle, ModelKind};
pub use pipeline::{Forwarding, Pipelined};
pub use probe::Probe;
pub use spec::{run_turn_spec, SpecCfg};
pub use tokenizer::Tok;
