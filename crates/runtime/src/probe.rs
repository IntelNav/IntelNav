//! Hardware capability probe.
//!
//! A small synchronous report the CLI, gateway, and registry can use
//! to answer "what backend do I have, and how big a model slice can
//! this machine carry." Intentionally side-effect-free — loading a
//! model is the caller's job.

use std::fmt;

use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct Probe {
    pub backends:       Backends,
    pub cpu:            CpuInfo,
    pub memory:         MemoryInfo,
    /// Best-effort summary string, ready to log or show in a UI.
    pub summary:        String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Backends {
    /// Compile-time features enabled in this build.
    pub compiled:       Vec<&'static str>,
    /// Backends reported as usable by candle at runtime.
    pub runtime:        Vec<&'static str>,
    /// The backend `pick_device(Auto)` would select right now.
    pub auto_choice:    &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct CpuInfo {
    pub logical_cores:  usize,
    pub brand:          String,
}

#[derive(Clone, Debug, Serialize)]
pub struct MemoryInfo {
    pub total_bytes:    u64,
    pub available_bytes: u64,
}

impl Probe {
    pub fn collect() -> Self {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        sys.refresh_cpu_all();

        let cpu = CpuInfo {
            logical_cores: sys.cpus().len(),
            brand:         sys.cpus().first().map(|c| c.brand().to_string()).unwrap_or_default(),
        };

        let memory = MemoryInfo {
            total_bytes:     sys.total_memory(),
            available_bytes: sys.available_memory(),
        };

        let mut compiled = Vec::new();
        #[cfg(feature = "cuda")]  compiled.push("cuda");
        #[cfg(feature = "metal")] compiled.push("metal");
        if compiled.is_empty() { compiled.push("cpu"); }

        let mut runtime: Vec<&'static str> = Vec::new();
        #[cfg(feature = "cuda")]
        if candle_core::utils::cuda_is_available() { runtime.push("cuda"); }
        #[cfg(feature = "metal")]
        if candle_core::utils::metal_is_available() { runtime.push("metal"); }
        runtime.push("cpu");

        let auto_choice = runtime.first().copied().unwrap_or("cpu");

        let summary = format!(
            "{auto_choice} · {} cores · {} RAM ({} free)",
            cpu.logical_cores,
            human(memory.total_bytes),
            human(memory.available_bytes),
        );

        Self { backends: Backends { compiled, runtime, auto_choice }, cpu, memory, summary }
    }

    /// Rough estimate of the largest model this machine can hold,
    /// given a target dtype. For a GPU host we'd ideally check VRAM,
    /// but candle doesn't expose VRAM uniformly across backends —
    /// fall back to "assume 80% of system RAM / VRAM available for
    /// weights" as a conservative ceiling.
    pub fn max_model_bytes(&self) -> u64 {
        // 80% of *available* RAM is a safe ceiling for a quantized
        // GGUF load. Actual runtime KV cache + activations need
        // headroom on top.
        self.memory.available_bytes.saturating_mul(4) / 5
    }
}

impl fmt::Display for Probe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "backend:   {}", self.backends.auto_choice)?;
        writeln!(f, "  compiled: {}", self.backends.compiled.join(", "))?;
        writeln!(f, "  runtime:  {}", self.backends.runtime.join(", "))?;
        writeln!(f, "cpu:       {} × {}", self.cpu.logical_cores, self.cpu.brand)?;
        writeln!(f, "memory:    {} total / {} available",
                 human(self.memory.total_bytes),
                 human(self.memory.available_bytes))?;
        writeln!(f, "max model: ~{} (80% of available RAM)",
                 human(self.max_model_bytes()))?;
        Ok(())
    }
}

fn human(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;
    if bytes >= GB { format!("{:.1} GiB", bytes as f64 / GB as f64) }
    else if bytes >= MB { format!("{:.1} MiB", bytes as f64 / MB as f64) }
    else if bytes >= KB { format!("{:.1} KiB", bytes as f64 / KB as f64) }
    else                { format!("{bytes} B") }
}
