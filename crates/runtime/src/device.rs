//! Operator-facing device hint.
//!
//! Hardware selection in IntelNav flows through the linked
//! `libllama-<backend>.so` plus `INTELNAV_NGL` (`n_gpu_layers`); ggml
//! itself decides which kernels to dispatch on. So all this module
//! exposes is a small enum the operator can pass on the command line
//! (`--device cpu | auto | cuda | metal`) and a translation to the
//! `n_gpu_layers` value the runtime hands ggml.
//!
//! `DevicePref::Cpu` forces `n_gpu_layers = 0`; everything else means
//! "offload as many layers as ggml can fit" (`n_gpu_layers = -1`),
//! relying on the libllama build that was actually loaded to do the
//! right thing. The ordinal on `Cuda(N)` / `Metal(N)` is currently
//! advisory — multi-GPU support is a later milestone.
//!
//! `INTELNAV_NGL` env var, if set, wins over `DevicePref` so an
//! operator can always pin the value out-of-band.

use anyhow::Result;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DevicePref {
    #[default]
    Auto,
    Cpu,
    Cuda(usize),
    Metal(usize),
}

impl DevicePref {
    /// Translate the hint into ggml's `n_gpu_layers`. `0` keeps every
    /// layer on CPU; `-1` lets ggml offload everything it can to
    /// whatever backend libllama was compiled with.
    pub fn n_gpu_layers(self) -> i32 {
        match self {
            DevicePref::Cpu => 0,
            DevicePref::Auto | DevicePref::Cuda(_) | DevicePref::Metal(_) => -1,
        }
    }

    /// Short label for logs / probe output.
    pub fn label(self) -> &'static str {
        match self {
            DevicePref::Auto    => "auto",
            DevicePref::Cpu     => "cpu",
            DevicePref::Cuda(_) => "cuda",
            DevicePref::Metal(_) => "metal",
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_common_forms() {
        assert_eq!("auto".parse::<DevicePref>().unwrap(), DevicePref::Auto);
        assert_eq!("CPU".parse::<DevicePref>().unwrap(), DevicePref::Cpu);
        assert_eq!("cuda".parse::<DevicePref>().unwrap(), DevicePref::Cuda(0));
        assert_eq!("cuda:1".parse::<DevicePref>().unwrap(), DevicePref::Cuda(1));
        assert_eq!("metal=2".parse::<DevicePref>().unwrap(), DevicePref::Metal(2));
        assert!("nope".parse::<DevicePref>().is_err());
    }

    #[test]
    fn cpu_forces_no_offload() {
        assert_eq!(DevicePref::Cpu.n_gpu_layers(), 0);
        assert_eq!(DevicePref::Auto.n_gpu_layers(), -1);
        assert_eq!(DevicePref::Cuda(0).n_gpu_layers(), -1);
        assert_eq!(DevicePref::Metal(1).n_gpu_layers(), -1);
    }
}
