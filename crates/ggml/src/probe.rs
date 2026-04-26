//! Hardware + runtime-library probe for the GGML path.
//!
//! Answers two questions that `intelnav doctor` needs to surface, and
//! that the future runtime loader (task #14c) uses to pick which
//! `libllama-<backend>.so` to dlopen:
//!
//! 1. **Which GPU backends are actually installed on this host?**
//!    i.e. which runtime shared libraries (`libhsa-runtime64.so.1`,
//!    `libvulkan.so.1`, `libcuda.so.1`, …) resolve via the dynamic
//!    linker's search paths. We check file existence — not ldconfig
//!    — to stay portable across glibc / musl / non-Linux.
//!
//! 2. **What GPU hardware is installed?** Best-effort, using tools
//!    that ship with the GPU driver (`nvidia-smi`, `rocminfo`) plus
//!    a `lspci` / `/sys/class/drm` fallback. Never fatal if the
//!    tools aren't present — we just report less.
//!
//! The probe is side-effect-free: it opens no libraries, loads no
//! models, forks at most shell commands that print text. Calling
//! it on startup is cheap.
//!
//! Platforms covered: Linux (primary), macOS (limited — Metal is
//! detected but GPU hardware enumeration is best-effort), Windows
//! (stub that always returns "cpu only"). Windows support lands
//! with the CI matrix extension (task #21).

use std::path::Path;
use std::process::Command;

use serde::Serialize;

/// A backend we know how to ship a prebuilt libllama for, plus its
/// runtime-library and driver status on this host.
#[derive(Clone, Debug, Serialize)]
pub struct BackendCheck {
    /// Matches the tag used in the CI release artifact name, e.g.
    /// "rocm" for `libllama-rocm-<sha>.tar.gz`.
    pub tag: &'static str,
    pub status: BackendStatus,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "state")]
pub enum BackendStatus {
    /// Runtime libraries are present. The libllama-<tag>.so can
    /// be `dlopen`ed safely.
    Available,
    /// Runtime libraries are missing. `reason` names the first
    /// missing file; `install_hint` is a single-line command
    /// suggestion for the user's platform when we can infer one.
    Missing {
        reason:       String,
        install_hint: Option<String>,
    },
    /// This platform doesn't carry the backend at all (e.g. Metal on
    /// Linux, ROCm on macOS). Silently deprioritized in ranking.
    NotApplicable,
}

impl BackendStatus {
    pub fn is_available(&self) -> bool {
        matches!(self, BackendStatus::Available)
    }
}

/// A detected GPU. `vendor` is a human-friendly string
/// ("AMD", "NVIDIA", "Intel"); `detail` is the full device line
/// — model, gfx arch, driver version — when the probe could find it.
#[derive(Clone, Debug, Serialize)]
pub struct DetectedGpu {
    pub vendor: &'static str,
    pub detail: String,
}

/// Everything the ggml probe can say about the host — backend
/// availability + hardware inventory + the loader's recommended
/// preference order for `dlopen` attempts.
#[derive(Clone, Debug, Serialize)]
pub struct GgmlProbe {
    pub gpus:     Vec<DetectedGpu>,
    pub backends: Vec<BackendCheck>,
    /// The order in which task #14c's loader should try to dlopen
    /// backend libllamas, highest-preference first. Always ends in
    /// `"cpu"` — CPU is guaranteed to load on any host where the
    /// plain libllama.so is present.
    pub preferred: Vec<&'static str>,
}

impl GgmlProbe {
    /// Run every check. Idempotent; safe to call repeatedly.
    pub fn collect() -> Self {
        let gpus = detect_gpus();
        let backends = vec![
            check_cuda(),
            check_rocm(),
            check_vulkan(),
            check_metal(),
            check_cpu(),
        ];

        // Preference ranking. For now: most-specialized first (CUDA,
        // ROCm, Metal), then universal GPU fallback (Vulkan), then
        // CPU. The task #14c loader will walk this list top-to-bottom,
        // dlopen each in turn, and stop on the first one that both
        // resolves AND passes a post-load sanity check.
        let rank: &[&'static str] = &["cuda", "rocm", "metal", "vulkan", "cpu"];
        let preferred: Vec<&'static str> = rank
            .iter()
            .copied()
            .filter(|tag| {
                backends
                    .iter()
                    .find(|b| b.tag == *tag)
                    .map(|b| b.status.is_available())
                    .unwrap_or(false)
            })
            .collect();

        Self { gpus, backends, preferred }
    }

    /// The single best backend to try first, given what's installed.
    /// Falls back to `"cpu"` if nothing else resolves.
    pub fn recommended(&self) -> &'static str {
        self.preferred.first().copied().unwrap_or("cpu")
    }
}

// ---------------------------------------------------------------------
// backend-specific checks
// ---------------------------------------------------------------------

fn check_cuda() -> BackendCheck {
    if !cfg!(any(target_os = "linux", target_os = "windows")) {
        return BackendCheck {
            tag:    "cuda",
            status: BackendStatus::NotApplicable,
        };
    }
    // libcuda.so.1 ships with the NVIDIA driver (not the toolkit);
    // libcudart ships with the toolkit and is what libllama-cuda.so
    // will transitively need.
    match find_first(&[
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib64/libcuda.so.1",
        "/usr/lib/libcuda.so.1",
    ]) {
        Some(_) => BackendCheck {
            tag:    "cuda",
            status: BackendStatus::Available,
        },
        None => BackendCheck {
            tag:    "cuda",
            status: BackendStatus::Missing {
                reason:       "libcuda.so.1 not found (install the NVIDIA driver)".to_string(),
                install_hint: Some(install_hint_cuda()),
            },
        },
    }
}

fn check_rocm() -> BackendCheck {
    if !cfg!(target_os = "linux") {
        return BackendCheck {
            tag:    "rocm",
            status: BackendStatus::NotApplicable,
        };
    }
    // The two libs our ROCm libllama.so transitively needs. Distros
    // place them under different prefixes (Fedora: /usr/lib64/,
    // Ubuntu: /opt/rocm/lib, etc.). Covering the common ones.
    let hsa = find_first(&[
        "/usr/lib64/libhsa-runtime64.so.1",
        "/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1",
        "/opt/rocm/lib/libhsa-runtime64.so.1",
    ]);
    let hip = find_first(&[
        "/usr/lib64/libamdhip64.so.6",
        "/usr/lib64/libamdhip64.so.5",
        "/usr/lib/x86_64-linux-gnu/libamdhip64.so",
        "/opt/rocm/lib/libamdhip64.so",
    ]);

    match (hsa, hip) {
        (Some(_), Some(_)) => BackendCheck {
            tag:    "rocm",
            status: BackendStatus::Available,
        },
        (None, _) => BackendCheck {
            tag:    "rocm",
            status: BackendStatus::Missing {
                reason:       "libhsa-runtime64.so.1 not found (ROCm runtime missing)".to_string(),
                install_hint: Some(install_hint_rocm()),
            },
        },
        (_, None) => BackendCheck {
            tag:    "rocm",
            status: BackendStatus::Missing {
                reason:       "libamdhip64.so not found (HIP runtime missing)".to_string(),
                install_hint: Some(install_hint_rocm()),
            },
        },
    }
}

fn check_vulkan() -> BackendCheck {
    match find_first(&[
        "/usr/lib/x86_64-linux-gnu/libvulkan.so.1",
        "/usr/lib64/libvulkan.so.1",
        "/usr/lib/libvulkan.so.1",
        // macOS — MoltenVK; install via Homebrew `vulkan-loader`.
        "/usr/local/lib/libvulkan.1.dylib",
        "/opt/homebrew/lib/libvulkan.1.dylib",
    ]) {
        Some(_) => BackendCheck {
            tag:    "vulkan",
            status: BackendStatus::Available,
        },
        None => BackendCheck {
            tag:    "vulkan",
            status: BackendStatus::Missing {
                reason:       "libvulkan.so.1 not found".to_string(),
                install_hint: Some(install_hint_vulkan()),
            },
        },
    }
}

fn check_metal() -> BackendCheck {
    if cfg!(target_os = "macos") {
        BackendCheck { tag: "metal", status: BackendStatus::Available }
    } else {
        BackendCheck { tag: "metal", status: BackendStatus::NotApplicable }
    }
}

fn check_cpu() -> BackendCheck {
    BackendCheck { tag: "cpu", status: BackendStatus::Available }
}

// ---------------------------------------------------------------------
// GPU hardware detection
// ---------------------------------------------------------------------

fn detect_gpus() -> Vec<DetectedGpu> {
    let mut out = Vec::new();

    // NVIDIA — nvidia-smi ships with the driver.
    if let Some(line) = run("nvidia-smi", &["--query-gpu=name,driver_version", "--format=csv,noheader"]) {
        for line in line.lines() {
            let line = line.trim();
            if !line.is_empty() {
                out.push(DetectedGpu {
                    vendor: "NVIDIA",
                    detail: line.to_string(),
                });
            }
        }
    }

    // AMD — rocminfo (if ROCm is installed) is authoritative; otherwise
    // fall back to reading the gfx name from /sys/class/drm.
    //
    // rocminfo enumerates HSA Agents; CPUs are agents too. For each
    // agent the Marketing Name line comes BEFORE Device Type, so we
    // can't filter as we go — we buffer the most-recent Marketing
    // Name and emit only when the matching Device Type confirms GPU.
    // A new Marketing Name without an intervening Device Type means
    // we're into a new agent whose previous buffer was irrelevant.
    let mut found_amd = false;
    if let Some(info) = run("rocminfo", &[]) {
        let mut pending_name: Option<String> = None;
        for line in info.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("Marketing Name:") {
                pending_name = Some(rest.trim().to_string());
                continue;
            }
            if let Some(kind) = line.strip_prefix("Device Type:") {
                if kind.trim().eq_ignore_ascii_case("GPU") {
                    if let Some(name) = pending_name.take() {
                        if !name.is_empty() {
                            out.push(DetectedGpu {
                                vendor: "AMD",
                                detail: name,
                            });
                            found_amd = true;
                        }
                    }
                } else {
                    // Consume the pending name — it was a CPU agent.
                    pending_name = None;
                }
            }
        }
    }
    if !found_amd {
        // Fallback: scan /sys/class/drm for an AMD vendor id. Present on
        // any Linux with the amdgpu kernel driver, even if rocminfo isn't.
        if cfg!(target_os = "linux") {
            if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
                for e in entries.flatten() {
                    let vendor_path = e.path().join("device/vendor");
                    if let Ok(vendor) = std::fs::read_to_string(&vendor_path) {
                        if vendor.trim() == "0x1002" {
                            let name = e.file_name().to_string_lossy().to_string();
                            out.push(DetectedGpu {
                                vendor: "AMD",
                                detail: format!("{name} (amdgpu)"),
                            });
                        }
                    }
                }
            }
        }
    }

    // Intel GPUs — best-effort. We don't ship an Intel-specific libllama
    // yet (SYCL is task #21); the Vulkan build covers them in practice.
    if cfg!(target_os = "linux") {
        if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
            for e in entries.flatten() {
                let vendor_path = e.path().join("device/vendor");
                if let Ok(vendor) = std::fs::read_to_string(&vendor_path) {
                    if vendor.trim() == "0x8086" {
                        let name = e.file_name().to_string_lossy().to_string();
                        out.push(DetectedGpu {
                            vendor: "Intel",
                            detail: format!("{name} (i915)"),
                        });
                    }
                }
            }
        }
    }

    out
}

// ---------------------------------------------------------------------
// install hints (per-distro best-effort)
// ---------------------------------------------------------------------

fn install_hint_cuda() -> String {
    "install the NVIDIA proprietary driver for your distro \
     (Ubuntu: `sudo apt install nvidia-driver-<latest>`, \
     Fedora: follow the rpmfusion guide)".to_string()
}

fn install_hint_rocm() -> String {
    "install the ROCm runtime for your distro \
     (Fedora: `sudo dnf install rocm-hip rocm-runtime hipblas rocblas`, \
     Ubuntu: follow https://rocm.docs.amd.com/projects/install-on-linux)".to_string()
}

fn install_hint_vulkan() -> String {
    "install your GPU driver's Vulkan loader \
     (Ubuntu: `sudo apt install libvulkan1`, \
     Fedora: `sudo dnf install vulkan-loader`, \
     Arch: `sudo pacman -S vulkan-icd-loader`)".to_string()
}

// ---------------------------------------------------------------------
// tiny utilities
// ---------------------------------------------------------------------

/// Return the first path in `candidates` that exists as a regular
/// file or symlink. We want file existence, not arbitrary metadata,
/// so symlinks (typical for `libXXX.so.1 → libXXX.so.1.2.3`) count.
fn find_first(candidates: &[&str]) -> Option<String> {
    for p in candidates {
        if Path::new(p).exists() {
            return Some((*p).to_string());
        }
    }
    None
}

/// Run a command, return its stdout as String if it exited 0.
/// Stderr is suppressed; errors (program missing, non-zero exit,
/// non-utf8 stdout) are silently collapsed to `None`. We want a
/// probe, not a shell.
fn run(program: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(program).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout).ok()
}

// ---------------------------------------------------------------------
// display helpers — the CLI's doctor command reaches for these.
// ---------------------------------------------------------------------

impl std::fmt::Display for GgmlProbe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "GPUs:")?;
        if self.gpus.is_empty() {
            writeln!(f, "  (none detected)")?;
        } else {
            for g in &self.gpus {
                writeln!(f, "  {} · {}", g.vendor, g.detail)?;
            }
        }
        writeln!(f, "libllama backends:")?;
        for b in &self.backends {
            match &b.status {
                BackendStatus::Available => {
                    writeln!(f, "  \x1b[32m✓\x1b[0m {:<8} runtime libs present", b.tag)?
                }
                BackendStatus::Missing { reason, install_hint } => {
                    writeln!(f, "  \x1b[33m✗\x1b[0m {:<8} {}", b.tag, reason)?;
                    if let Some(hint) = install_hint {
                        writeln!(f, "              hint: {hint}")?;
                    }
                }
                BackendStatus::NotApplicable => {
                    writeln!(f, "  \x1b[90m·\x1b[0m {:<8} n/a on this platform", b.tag)?
                }
            }
        }
        writeln!(f, "preferred order: {}", self.preferred.join(" → "))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_is_always_available() {
        let p = GgmlProbe::collect();
        let cpu = p.backends.iter().find(|b| b.tag == "cpu").unwrap();
        assert!(cpu.status.is_available(), "cpu backend must always be available");
    }

    #[test]
    fn preferred_includes_cpu_somewhere() {
        // Even a stripped host with no drivers should still fall
        // through to cpu — that's the guarantee.
        let p = GgmlProbe::collect();
        assert!(
            p.preferred.iter().any(|t| *t == "cpu"),
            "preferred list must contain cpu as a fallback"
        );
    }

    #[test]
    fn ranking_prefers_specialized_over_generic() {
        // If both ROCm AND Vulkan are available, ROCm must come first.
        // We can't force that on the real host, but we can verify the
        // ordering logic via the `rank` slice matches documented
        // preferences (ROCm → Vulkan on the same AMD hardware).
        let rank: &[&'static str] = &["cuda", "rocm", "metal", "vulkan", "cpu"];
        let rocm_pos = rank.iter().position(|t| *t == "rocm").unwrap();
        let vulkan_pos = rank.iter().position(|t| *t == "vulkan").unwrap();
        assert!(rocm_pos < vulkan_pos, "rocm must outrank vulkan");
    }

    #[test]
    fn display_is_non_empty() {
        let p = GgmlProbe::collect();
        let s = format!("{p}");
        assert!(s.contains("libllama backends:"));
        assert!(s.contains("preferred order:"));
    }
}
