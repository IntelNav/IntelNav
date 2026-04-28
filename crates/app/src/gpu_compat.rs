//! Auto-detect AMD GPU and set runtime overrides ROCm needs to run
//! kernels we built for an adjacent arch.
//!
//! Background: ROCm precompiles SPIR-V/HSACO bytecode per-arch. Our
//! libllama tarballs target a curated set (gfx1030, gfx1100, gfx908,
//! gfx90a). Cards in the gfx10.3.x family (RX 6600 = gfx1032, RX 6700
//! XT = gfx1031, etc.) are *binary compatible* with gfx1030 once you
//! tell ROCm to pretend so via `HSA_OVERRIDE_GFX_VERSION=10.3.0`.
//! Same idea on RDNA3 with `11.0.0`.
//!
//! The user shouldn't need to know this. We probe the GPU at startup
//! (cheap `rocminfo` call) and set the env var if needed, before
//! libllama is loaded via dlopen. If `HSA_OVERRIDE_GFX_VERSION` is
//! already in the environment we leave it alone — power-user override
//! always wins.

use std::process::Command;

use tracing::{debug, info};

/// Probe the GPU and set `HSA_OVERRIDE_GFX_VERSION` if our libllama
/// tarballs would otherwise fail on this card. Idempotent.
///
/// Call this BEFORE any `intelnav-ggml` / libllama operation. Right
/// place is `main()` of either binary, immediately after the
/// `firstrun::ensure_initialized()` call.
pub fn ensure_runtime_overrides() {
    if std::env::var_os("HSA_OVERRIDE_GFX_VERSION").is_some() {
        debug!("HSA_OVERRIDE_GFX_VERSION already set; leaving alone");
        return;
    }
    let Some(arch) = detect_gfx_arch() else {
        debug!("no AMD GPU detected (or rocminfo unavailable)");
        return;
    };
    let target = override_for(&arch);
    let Some(target) = target else {
        debug!(detected = %arch, "no override needed for this arch");
        return;
    };
    // SAFETY: set_var is unsafe in 2024 edition because env access in
    // multi-threaded programs is racy. We're in single-threaded main
    // before any worker threads spawn, which is the documented safe
    // window. The libllama dlopen / tokio runtime starts after this.
    #[allow(unsafe_code)]
    unsafe {
        std::env::set_var("HSA_OVERRIDE_GFX_VERSION", target);
    }
    info!(arch = %arch, override = %target,
        "auto-set HSA_OVERRIDE_GFX_VERSION so libllama gfx10x0 kernels run on this card");
}

/// Map a real GPU arch to the override value that lets adjacent-arch
/// bytecode run on it. `None` means "no adjustment needed."
fn override_for(arch: &str) -> Option<&'static str> {
    match arch {
        // RDNA2 (Navi 2x) — all binary-compat with gfx1030 once told.
        // Includes RX 6600 (gfx1032), RX 6700 XT (gfx1031), 6800/6900
        // (gfx1030 native), and the embedded variants.
        "gfx1031" | "gfx1032" | "gfx1033" | "gfx1034" | "gfx1035" | "gfx1036" => Some("10.3.0"),
        // RDNA3 (Navi 3x) — gfx1100 covers most discrete cards; the
        // override unlocks gfx1101/1102 (laptop / OEM variants) when
        // only gfx1100 bytecode is shipped.
        "gfx1101" | "gfx1102" | "gfx1103" => Some("11.0.0"),
        // RDNA3.5 (Strix-class APUs).
        "gfx1150" | "gfx1151" | "gfx1152" | "gfx1200" | "gfx1201" => Some("11.0.0"),
        _ => None,
    }
}

/// Run `rocminfo` and pull out the first non-generic gfx target name.
/// Returns `None` if rocminfo is missing, errors, or doesn't list a
/// real GPU agent (CPU-only systems).
fn detect_gfx_arch() -> Option<String> {
    let out = Command::new("rocminfo").output().ok()?;
    if !out.status.success() { return None; }
    let s = String::from_utf8_lossy(&out.stdout);
    // rocminfo output has lines like:
    //   Name:                    amdgcn-amd-amdhsa--gfx1032
    // and generic placeholders we want to skip:
    //   Name:                    amdgcn-amd-amdhsa--gfx10-3-generic
    for line in s.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("Name:") {
            let rest = rest.trim();
            if let Some(arch) = rest.strip_prefix("amdgcn-amd-amdhsa--") {
                // Take the leading token; drop generic placeholders.
                let arch = arch.split_whitespace().next().unwrap_or("");
                if arch.starts_with("gfx")
                    && arch.chars().nth(3).map(|c| c.is_ascii_digit()).unwrap_or(false)
                    && !arch.contains("generic")
                {
                    return Some(arch.to_string());
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn override_map_covers_rdna2_oems() {
        assert_eq!(override_for("gfx1032"), Some("10.3.0"));
        assert_eq!(override_for("gfx1031"), Some("10.3.0"));
        assert_eq!(override_for("gfx1030"), None);
    }

    #[test]
    fn override_map_covers_rdna3() {
        assert_eq!(override_for("gfx1101"), Some("11.0.0"));
        assert_eq!(override_for("gfx1100"), None);
    }

    #[test]
    fn override_map_skips_unrelated() {
        assert_eq!(override_for("gfx900"), None);
        assert_eq!(override_for("gfx942"), None);
    }
}
