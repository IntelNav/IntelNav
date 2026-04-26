//! Hardware capability probe.
//!
//! A small synchronous report the CLI, gateway, and registry can use
//! to answer "what backend do I have, and how big a model slice can
//! this machine carry." Intentionally side-effect-free — loading a
//! model is the caller's job.
//!
//! Backend availability is delegated to [`intelnav_ggml::probe`] which
//! checks driver-library presence (`libcuda.so.1`, `libhsa-runtime64.so`,
//! `libvulkan.so.1`, …) without dlopen'ing them, and runs `nvidia-smi`
//! / `rocminfo` when present. CPU-side data (cores, RAM) comes from
//! `sysinfo`.

use std::fmt;

use intelnav_ggml::probe::GgmlProbe;
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
    /// Backends ggml's probe says have their driver libraries
    /// installed on this host. Order is highest-preference first
    /// (CUDA → ROCm → Metal → Vulkan → CPU); always ends in `cpu`.
    pub available: Vec<&'static str>,
    /// The backend a libllama loader would try first given what's
    /// installed. Same as `available[0]` unless empty (then `cpu`).
    pub recommended: &'static str,
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

        let gg = GgmlProbe::collect();
        let backends = Backends {
            available:   gg.preferred.clone(),
            recommended: gg.recommended(),
        };

        let summary = format!(
            "{} · {} cores · {} RAM ({} free)",
            backends.recommended,
            cpu.logical_cores,
            human(memory.total_bytes),
            human(memory.available_bytes),
        );

        Self { backends, cpu, memory, summary }
    }

    /// Rough estimate of the largest model this machine can hold.
    /// We fall back to "80% of available RAM" because VRAM isn't
    /// exposed uniformly across backends — a conservative ceiling
    /// for a quantized GGUF load. Actual runtime KV cache +
    /// activations need headroom on top.
    pub fn max_model_bytes(&self) -> u64 {
        self.memory.available_bytes.saturating_mul(4) / 5
    }
}

impl fmt::Display for Probe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "backend:   {}", self.backends.recommended)?;
        writeln!(f, "  available: {}", self.backends.available.join(", "))?;
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
