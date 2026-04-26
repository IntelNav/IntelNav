//! Manifest: the authoritative partitioning of a model into shard parts.
//!
//! Loaded once at startup from TOML. Spec: `shard-registry-v1.md` §2.

use std::fs;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

use intelnav_core::Quant;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Manifest {
    pub model: ManifestModel,
    #[serde(rename = "part", default)]
    pub parts: Vec<ManifestPart>,
    #[serde(default)]
    pub defaults: ManifestDefaults,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ManifestModel {
    pub cid:          String,
    pub name:         String,
    pub quant:        Quant,
    pub total_layers: u16,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ManifestPart {
    pub id:             String,
    pub layer_range:    [u16; 2],      // inclusive [start, end]
    pub weight_url:     String,
    pub sha256:         String,
    pub size_bytes:     u64,
    #[serde(default)]
    pub min_vram_bytes: u64,
    pub desired_k:      Option<u8>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ManifestDefaults {
    pub desired_k:                u8,
    pub heartbeat_interval_s:     u32,
    pub heartbeat_miss_tolerance: u32,
    pub min_live_seconds:         u32,
    pub reservation_ttl_s:        u32,
    pub replay_window_s:          u32,
    pub assign_rate_per_hour:     u32,
}

impl Default for ManifestDefaults {
    fn default() -> Self {
        Self {
            desired_k:                3,
            heartbeat_interval_s:     20,
            heartbeat_miss_tolerance: 3,
            min_live_seconds:         600,
            reservation_ttl_s:        1800,
            replay_window_s:          120,
            assign_rate_per_hour:     6,
        }
    }
}

impl Manifest {
    pub fn load(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("reading manifest {}", path.display()))?;
        let mut m: Manifest = toml::from_str(&text)
            .with_context(|| format!("parsing manifest {}", path.display()))?;
        m.validate()?;
        Ok(m)
    }

    fn validate(&mut self) -> Result<()> {
        if self.parts.is_empty() {
            return Err(anyhow!("manifest has zero [[part]] entries"));
        }
        let mut covered: u32 = 0;
        for p in &self.parts {
            let [s, e] = p.layer_range;
            if e < s {
                return Err(anyhow!("part {}: layer_range {:?} is inverted", p.id, p.layer_range));
            }
            if p.sha256.len() != 64 {
                return Err(anyhow!("part {}: sha256 must be 64 hex chars", p.id));
            }
            if p.weight_url.is_empty() {
                return Err(anyhow!("part {}: weight_url is empty", p.id));
            }
            covered += (e - s + 1) as u32;
        }
        if covered != self.model.total_layers as u32 {
            return Err(anyhow!(
                "manifest coverage {} != model.total_layers {}",
                covered, self.model.total_layers
            ));
        }
        Ok(())
    }

    pub fn part(&self, id: &str) -> Option<&ManifestPart> {
        self.parts.iter().find(|p| p.id == id)
    }

    pub fn desired_k_of(&self, part: &ManifestPart) -> u8 {
        part.desired_k.unwrap_or(self.defaults.desired_k)
    }
}
