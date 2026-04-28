//! Domain value types mirroring paper §4.2, §4.5 and §10.

use serde::{Deserialize, Serialize};

use crate::ids::{ModelId, PeerId};

/// Inference backend advertised by a contributor. Paper §2.3.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Backend {
    #[serde(alias = "llama.cpp")]
    LlamaCpp,
    #[serde(alias = "vllm")]
    Vllm,
    #[serde(alias = "mlx-lm")]
    MlxLm,
    #[serde(alias = "ollama")]
    Ollama,
    #[serde(other)]
    Unknown,
}

/// Quantization tier. Paper §4.1 / §B.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum Quant {
    #[serde(rename = "Q4_K_M")]
    Q4KM,
    #[serde(rename = "Q5_K_M")]
    Q5KM,
    #[serde(rename = "Q8_0")]
    Q8_0,
    #[serde(rename = "fp16")]
    FP16,
    #[serde(rename = "bf16")]
    BF16,
}

impl Quant {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Q4KM => "Q4_K_M",
            Self::Q5KM => "Q5_K_M",
            Self::Q8_0 => "Q8_0",
            Self::FP16 => "fp16",
            Self::BF16 => "bf16",
        }
    }
}

/// Inclusive layer range advertised by a contributor shard.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LayerRange {
    pub start: u16,
    pub end: u16,
}
impl LayerRange {
    pub fn new(start: u16, end: u16) -> Self {
        Self { start, end }
    }
    pub fn len(&self) -> u16 {
        self.end.saturating_sub(self.start) + 1
    }
    pub fn is_empty(&self) -> bool {
        self.end < self.start
    }
}

/// Peer role — decides routing preference (volunteers outrank cloud).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    #[default]
    Volunteer,
    Cloud,
}

/// Capability vector advertised in the DHT. Paper §4.2.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CapabilityV1 {
    pub peer_id:     PeerId,
    pub backend:     Backend,
    pub quants:      Vec<Quant>,
    pub vram_bytes:  u64,
    pub ram_bytes:   u64,
    pub tok_per_sec: f32,
    pub max_seq:     u32,
    pub models:      Vec<ModelId>,
    pub layers:      Vec<ShardRoute>,
    #[serde(default)]
    pub role:        Role,
}

/// Single entry in a contributor's `layers` vector — "peer holds layers
/// `start..=end` of model CID".
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardRoute {
    pub cid:   String,
    pub start: u16,
    pub end:   u16,
}
