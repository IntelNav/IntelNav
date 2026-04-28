//! Mandatory contribution gate.
//!
//! Chat is unlocked when *one* of these is true:
//!
//! 1. The user hosts at least one slice (`<models_dir>/.shards/*` has
//!    a non-empty kept_ranges.json with at least one range that isn't
//!    in `disabled_ranges.json`).
//! 2. `config.relay_only = true` — the user opted out of hosting
//!    inference but still runs the daemon as a DHT relay.
//!
//! Any other state returns [`GateState::NeedsContribution`] with a
//! suggested slice that fits the user's hardware, so the TUI can
//! prompt without forcing the user to learn the catalog.

use std::path::Path;

use intelnav_core::Config;
use intelnav_runtime::probe::Probe;

use crate::catalog::{catalog, CatalogEntry, Fit};
use crate::contribute::{DisabledRanges, KeptRanges};

#[derive(Clone, Debug)]
pub enum GateState {
    /// Chat is allowed: user is contributing one way or the other.
    Pass(GatePass),
    /// User must pick a slice (or enable relay mode) before chatting.
    NeedsContribution { suggestion: Option<Suggestion> },
}

#[derive(Clone, Copy, Debug)]
pub enum GatePass {
    /// Hosting at least one slice.
    Hosting,
    /// DHT-only relay mode.
    Relay,
}

/// What the gate suggests the user host. Picked as the smallest slice
/// of the smallest model that comfortably fits — the bar is "the user
/// will see a model running, not a memory error."
#[derive(Clone, Debug)]
pub struct Suggestion {
    pub entry: &'static CatalogEntry,
    pub range: (u16, u16),
    pub fit:   Fit,
}

pub fn check(config: &Config) -> GateState {
    if hosts_any_slice(&config.models_dir) {
        return GateState::Pass(GatePass::Hosting);
    }
    if config.relay_only {
        return GateState::Pass(GatePass::Relay);
    }
    let probe = Probe::collect();
    GateState::NeedsContribution { suggestion: best_suggestion(&probe) }
}

fn hosts_any_slice(models_dir: &Path) -> bool {
    let shards = models_dir.join(".shards");
    let Ok(rd) = std::fs::read_dir(&shards) else { return false; };
    for entry in rd.flatten() {
        let root = entry.path();
        let Ok(bytes) = std::fs::read(root.join("kept_ranges.json")) else { continue; };
        let Ok(k): Result<KeptRanges, _> = serde_json::from_slice(&bytes) else { continue; };
        let disabled = DisabledRanges::load(&root);
        if k.kept.iter().any(|&(s, e)| !disabled.contains(s, e)) {
            return true;
        }
    }
    false
}

/// Pick a single (model, slice) the user can host comfortably. We
/// rank by smallest model first, then smallest slice within that
/// model — the goal is "lowest bar to onboard," not "best fit."
fn best_suggestion(probe: &Probe) -> Option<Suggestion> {
    let mut best: Option<Suggestion> = None;
    for entry in catalog() {
        let splits = entry.swarm_ranges();
        if splits.is_empty() { continue; }
        // Per-slice RAM bound: assume layers dominate cost. Allocate
        // proportional to layer count; round up so we err on the side
        // of "won't OOM."
        let total_layers = entry.block_count.max(1) as u64;
        for &(start, end) in splits.iter() {
            let span = (end - start) as u64;
            let slice_ram = entry.ram_bytes_min.saturating_mul(span) / total_layers;
            // Treat per-slice the same way `CatalogEntry::fit` treats
            // the full model.
            let free = probe.memory.available_bytes;
            let fit = if free >= slice_ram.saturating_mul(13) / 10 { Fit::Fits }
                      else if free >= slice_ram { Fit::Tight }
                      else { Fit::TooBig };
            if matches!(fit, Fit::TooBig) { continue; }
            let candidate = Suggestion { entry, range: (start, end), fit };
            best = Some(match best {
                None => candidate,
                Some(prev) => {
                    // Prefer Fits over Tight, then smaller models, then
                    // smaller slices.
                    let better = match (prev.fit, candidate.fit) {
                        (Fit::Tight, Fit::Fits) => true,
                        (Fit::Fits,  Fit::Tight) => false,
                        _ => {
                            if candidate.entry.size_bytes != prev.entry.size_bytes {
                                candidate.entry.size_bytes < prev.entry.size_bytes
                            } else {
                                let p_span = (prev.range.1 - prev.range.0) as u64;
                                let c_span = (candidate.range.1 - candidate.range.0) as u64;
                                c_span < p_span
                            }
                        }
                    };
                    if better { candidate } else { prev }
                }
            });
        }
    }
    best
}

/// Persist `relay_only = true` to the user's config.toml so subsequent
/// launches skip the gate. Idempotent.
pub fn enable_relay_mode() -> anyhow::Result<()> {
    let Some(path) = Config::config_path() else {
        anyhow::bail!("could not resolve config dir");
    };
    let mut cfg = Config::load()?;
    cfg.relay_only = true;
    let toml_str = toml::to_string_pretty(&cfg)?;
    std::fs::write(&path, toml_str)?;
    Ok(())
}
