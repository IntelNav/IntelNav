//! GGUF loading + architecture dispatch.
//!
//! Since task #12 (2026-04-23) this is a thin façade over
//! [`GgmlBackend`] — every supported arch goes through libllama. The
//! old candle-fork path (`Qwen2Weights` / `LlamaWeights`) was removed
//! after the ggml path was proven bit-identical + end-to-end over TCP
//! and running at GPU speed on AMD hardware.

use std::path::Path;

use anyhow::{Context, Result};
use candle_core::Device;

use crate::ggml_backend::GgmlBackend;
use crate::pipeline::{Forwarding, Hidden, Pipelined};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    /// Every GGUF architecture libllama supports — the universal
    /// backend via [`GgmlBackend`]. Chat-template choice is deferred
    /// to the caller (or to the model's `tokenizer.chat_template`
    /// metadata once task #14 wires that through).
    Ggml,
}

impl ModelKind {
    /// Every supported model today goes through the pipelined path.
    pub fn is_pipelined(self) -> bool {
        true
    }
}

/// Read just the GGUF header and return its architecture tag.
/// Kept for source-compat with callers that used to route per-arch;
/// the ggml backend handles every arch uniformly so this is now a
/// stub that returns [`ModelKind::Ggml`] whenever the file opens.
pub fn sniff_arch(_path: &Path) -> Result<ModelKind> {
    Ok(ModelKind::Ggml)
}

pub enum ModelHandle {
    Ggml(GgmlBackend),
}

impl ModelHandle {
    pub fn kind(&self) -> ModelKind {
        match self {
            Self::Ggml(_) => ModelKind::Ggml,
        }
    }

    /// Load a GGUF model through [`GgmlBackend`]. Supports every
    /// architecture libllama does and every GPU ggml is built for
    /// (CUDA, ROCm, Metal, Vulkan, SYCL, CPU). Env overrides:
    ///
    ///   * `INTELNAV_NGL=<int>` — ggml's `n_gpu_layers`. `0` forces
    ///     CPU, `-1` (default) offloads every layer onto whatever
    ///     backend the linked libllama was compiled with, positive
    ///     N offloads that many.
    ///   * `INTELNAV_NCTX` — context window (default 2048).
    ///   * `INTELNAV_NBATCH` — physical batch size (default 512).
    ///
    /// `device` is kept in the signature for source-compatibility with
    /// pre-task-#12 callers; the ggml path does not consult it.
    /// Hardware selection is purely via which libllama was linked and
    /// the `INTELNAV_NGL` value.
    pub fn load(path: &Path, _device: &Device) -> Result<Self> {
        Self::load_ggml_from_env(path)
    }

    /// Explicit entry point — bypasses the env-var dispatcher. Useful
    /// for tests and benches that want to pin specific values.
    pub fn load_ggml(
        path:         &Path,
        n_ctx:        u32,
        n_batch:      u32,
        n_gpu_layers: i32,
    ) -> Result<Self> {
        let b = GgmlBackend::load(path, n_ctx, n_batch, n_gpu_layers)
            .with_context(|| format!("loading {} via GgmlBackend", path.display()))?;
        Ok(Self::Ggml(b))
    }

    fn load_ggml_from_env(path: &Path) -> Result<Self> {
        let n_ctx:    u32 = env_or("INTELNAV_NCTX",   2048);
        let n_batch:  u32 = env_or("INTELNAV_NBATCH",  512);
        let n_gpu_layers: i32 = env_or("INTELNAV_NGL",  -1);
        Self::load_ggml(path, n_ctx, n_batch, n_gpu_layers)
    }

    // ---- forwarding delegation ----

    pub fn forwarding(&mut self) -> &mut dyn Forwarding {
        match self {
            Self::Ggml(m) => m,
        }
    }

    /// Returns the pipelined-capable handle. Always `Some` on the
    /// ggml path — every supported arch is pipelined.
    pub fn pipelined(&mut self) -> Option<&mut dyn Pipelined> {
        match self {
            Self::Ggml(m) => Some(m),
        }
    }

    pub fn block_count(&self) -> usize {
        match self {
            Self::Ggml(m) => <GgmlBackend as Forwarding>::block_count(m),
        }
    }

    pub fn forward(&mut self, input_ids: &[u32], index_pos: usize) -> Result<Hidden> {
        self.forwarding().forward(input_ids, index_pos)
    }

    pub fn reset_cache(&mut self) {
        self.forwarding().reset_cache()
    }
}

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}
