//! Device selection with runtime fallback.
//!
//! Build matrix:
//!
//! * default features → CPU only, builds everywhere.
//! * `--features cuda` → enables candle's CUDA kernels. The binary
//!   will still fall back to CPU if no CUDA runtime is present.
//! * `--features metal` → same but for Apple GPUs.
//!
//! The picker honours [`DevicePref`] so the operator can force a
//! backend, with "auto" as the sensible default.

use anyhow::Result;
use candle_core::Device;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DevicePref {
    #[default]
    Auto,
    Cpu,
    Cuda(usize),
    Metal(usize),
}

impl std::str::FromStr for DevicePref {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        let s = s.trim().to_ascii_lowercase();
        if s == "auto" || s.is_empty() { return Ok(Self::Auto); }
        if s == "cpu"                   { return Ok(Self::Cpu); }
        if let Some(rest) = s.strip_prefix("cuda") {
            let ord = rest.trim_start_matches([':', '=']).parse::<usize>().unwrap_or(0);
            return Ok(Self::Cuda(ord));
        }
        if let Some(rest) = s.strip_prefix("metal") {
            let ord = rest.trim_start_matches([':', '=']).parse::<usize>().unwrap_or(0);
            return Ok(Self::Metal(ord));
        }
        Err(anyhow::anyhow!("unknown device `{s}` (expected auto|cpu|cuda[:N]|metal[:N])"))
    }
}

pub fn pick_device() -> Result<Device> {
    pick_device_with(DevicePref::Auto)
}

pub fn pick_device_with(pref: DevicePref) -> Result<Device> {
    match pref {
        DevicePref::Cpu => {
            tracing::info!("runtime: using CPU (forced)");
            Ok(Device::Cpu)
        }
        DevicePref::Cuda(i) => {
            let d = Device::new_cuda(i)
                .map_err(|e| anyhow::anyhow!("CUDA requested but unavailable: {e}"))?;
            tracing::info!("runtime: using CUDA device {i} (forced)");
            Ok(d)
        }
        DevicePref::Metal(i) => {
            let d = Device::new_metal(i)
                .map_err(|e| anyhow::anyhow!("Metal requested but unavailable: {e}"))?;
            tracing::info!("runtime: using Metal device {i} (forced)");
            Ok(d)
        }
        DevicePref::Auto => {
            #[cfg(feature = "cuda")]
            {
                if candle_core::utils::cuda_is_available() {
                    if let Ok(d) = Device::new_cuda(0) {
                        tracing::info!("runtime: using CUDA device 0");
                        return Ok(d);
                    }
                }
            }
            #[cfg(feature = "metal")]
            {
                if candle_core::utils::metal_is_available() {
                    if let Ok(d) = Device::new_metal(0) {
                        tracing::info!("runtime: using Metal device 0");
                        return Ok(d);
                    }
                }
            }
            tracing::info!("runtime: using CPU (no GPU backend available)");
            Ok(Device::Cpu)
        }
    }
}
