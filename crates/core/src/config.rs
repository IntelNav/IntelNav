//! Configuration loader — TOML file + environment overlay.
//!
//! Resolution order (later wins):
//!   1. Compiled-in defaults ([`Config::default`]).
//!   2. `$XDG_CONFIG_HOME/intelnav/config.toml` if present.
//!   3. `INTELNAV_*` environment variables (see individual `Env:` comments).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::tier::LatencyTier;

/// Where does the CLI send its turns?
///
/// * `Local` — run in-process against `models_dir`.
/// * `Network` — route through the configured peer chain
///   (`peers` + `splits`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RunMode {
    #[default]
    Local,
    Network,
}

impl RunMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Local   => "local",
            Self::Network => "network",
        }
    }
}

impl std::str::FromStr for RunMode {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "local"   | "offline"   => Ok(Self::Local),
            "network" | "remote"    => Ok(Self::Network),
            other => Err(Error::Config(format!(
                "unknown run mode `{other}` (expected local|network)"
            ))),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// How the CLI picks its backend. Env: `INTELNAV_MODE`.
    pub mode: RunMode,

    /// Default latency tier for outbound chains. Env: `INTELNAV_TIER`.
    pub default_tier: LatencyTier,

    /// Allow T3 (WAN) routes by default. Env: `INTELNAV_ALLOW_WAN`.
    pub allow_wan: bool,

    /// Quorum over disjoint shard chains. Paper §8.3. Env: `INTELNAV_QUORUM`.
    pub quorum: u8,

    /// Bootstrap multiaddrs (libp2p). Env: `INTELNAV_BOOTSTRAP` (comma-sep).
    pub bootstrap: Vec<String>,

    /// Default model shown in the CLI header.
    pub default_model: String,

    /// Directory scanned for local GGUFs. Env: `INTELNAV_MODELS_DIR`.
    pub models_dir: PathBuf,

    /// Preferred device for the in-process runtime:
    /// `auto`, `cpu`, `cuda[:N]`, `metal[:N]`. Env: `INTELNAV_DEVICE`.
    pub device: String,

    /// Optional shard-registry URL. When set, the CLI subscribes to
    /// `GET /v1/shards/<registry_model>` and surfaces its peers alongside
    /// mDNS/DHT entries. Env: `INTELNAV_REGISTRY_URL`.
    #[serde(default)]
    pub registry_url: Option<String>,

    /// Model CID the registry serves (used as the path segment in
    /// `/v1/shards/:model`). Env: `INTELNAV_REGISTRY_MODEL`.
    #[serde(default)]
    pub registry_model: Option<String>,

    /// Hand-configured pipeline peer chain for `RunMode::Network`.
    /// `peers = ["a:7717", "b:7717"]` + `splits = [6, 12]` means the
    /// driver owns `[0..6)`, peer A owns `[6..12)`, peer B owns
    /// `[12..N)`. Env: `INTELNAV_PEERS` (comma-sep), `INTELNAV_SPLITS`
    /// (comma-sep).
    #[serde(default)]
    pub peers:  Vec<String>,
    #[serde(default)]
    pub splits: Vec<u16>,

    /// Path to a small GGUF used as the **draft** model for speculative
    /// decoding. Must share the target's tokenizer (Qwen2 family today:
    /// e.g. 0.5B drafting for a 7B target). Empty = spec-dec disabled.
    /// Env: `INTELNAV_DRAFT_MODEL`.
    #[serde(default)]
    pub draft_model: Option<PathBuf>,

    /// Draft proposals per spec-dec round. 0 or 1 = spec-dec disabled;
    /// sane values are 3..8. Env: `INTELNAV_SPEC_K`.
    #[serde(default)]
    pub spec_k: u16,

    /// Activation dtype on the chain wire: `"fp16"` (baseline, lossless
    /// vs the Q4_K_M weight noise floor) or `"int8"` (per-token
    /// symmetric quant, ~2× smaller bytes/step on LAN). Env:
    /// `INTELNAV_WIRE_DTYPE`.
    #[serde(default = "default_wire_dtype")]
    pub wire_dtype: String,

    /// libp2p listen multiaddr for the swarm node. Empty = pick an
    /// ephemeral TCP port on every interface. Env: `INTELNAV_LIBP2P_LISTEN`.
    #[serde(default = "default_libp2p_listen")]
    pub libp2p_listen: String,

    /// Public `host:port` of this peer's chunk-server, advertised in
    /// DHT provider records so others can pull our slice's bundles.
    /// `None` means "I host slices but won't seed them" — others may
    /// still route inference through us. Env: `INTELNAV_CHUNKS_ADDR`.
    #[serde(default)]
    pub chunks_addr: Option<String>,

    /// Public `host:port` of this peer's `pipe_peer` inference TCP
    /// listener, advertised so others can include us in chains.
    /// `None` means "I'm a leech — I run inference but don't host."
    /// Env: `INTELNAV_FORWARD_ADDR`.
    #[serde(default)]
    pub forward_addr: Option<String>,

    /// Hard contribution gate: when `true`, chat is unlocked even if
    /// the user hosts no slices, on the understanding that they're
    /// still contributing DHT routing (the daemon participates in
    /// Kademlia regardless). Off by default — first run forces the
    /// user through the picker. Env: `INTELNAV_RELAY_ONLY`.
    #[serde(default)]
    pub relay_only: bool,
}

fn default_wire_dtype() -> String { "fp16".into() }
fn default_libp2p_listen() -> String { "/ip4/0.0.0.0/tcp/0".into() }

impl Default for Config {
    fn default() -> Self {
        Self {
            mode:          RunMode::Local,
            default_tier:  LatencyTier::Continent,
            allow_wan:     false,
            quorum:        1,
            bootstrap:     vec![],
            default_model: "deepseek-coder:33b".into(),
            models_dir:    default_models_dir(),
            device:        "auto".into(),
            registry_url:  None,
            registry_model: None,
            peers:         Vec::new(),
            splits:        Vec::new(),
            draft_model:   None,
            spec_k:        0,
            wire_dtype:    default_wire_dtype(),
            libp2p_listen: default_libp2p_listen(),
            chunks_addr:   None,
            forward_addr:  None,
            relay_only:    false,
        }
    }
}

/// `$XDG_DATA_HOME/intelnav/models` (or `~/.local/share/intelnav/models`).
pub fn default_models_dir() -> PathBuf {
    directories::ProjectDirs::from("io", "intelnav", "intelnav")
        .map(|p| p.data_dir().join("models"))
        .unwrap_or_else(|| PathBuf::from("./models"))
}

impl Config {
    pub fn config_path() -> Option<PathBuf> {
        directories::ProjectDirs::from("io", "intelnav", "intelnav")
            .map(|p| p.config_dir().join("config.toml"))
    }

    /// `$XDG_STATE_HOME/intelnav/intelnav.log` (falls back to a temp
    /// file if the dirs lookup fails). The TUI redirects tracing +
    /// raw stderr here so stray log lines can't paint over the UI.
    pub fn log_path(&self) -> PathBuf {
        directories::ProjectDirs::from("io", "intelnav", "intelnav")
            .map(|p| p.state_dir()
                    .map(|s| s.to_path_buf())
                    .unwrap_or_else(|| p.cache_dir().to_path_buf())
                    .join("intelnav.log"))
            .unwrap_or_else(|| std::env::temp_dir().join("intelnav.log"))
    }

    pub fn load() -> Result<Self> {
        let mut cfg = Self::default();
        if let Some(p) = Self::config_path() {
            if p.exists() {
                let raw = std::fs::read_to_string(&p)?;
                cfg = toml::from_str(&raw)
                    .map_err(|e| Error::Config(format!("{}: {e}", p.display())))?;
            }
        }
        cfg.apply_env();
        Ok(cfg)
    }

    pub fn apply_env(&mut self) {
        use std::env::var;
        if let Ok(v) = var("INTELNAV_MODE") {
            if let Ok(m) = v.parse::<RunMode>() { self.mode = m; }
        }
        if let Ok(v) = var("INTELNAV_MODELS_DIR") { self.models_dir = PathBuf::from(v); }
        if let Ok(v) = var("INTELNAV_DEVICE")     { self.device     = v; }
        if let Ok(v) = var("INTELNAV_ALLOW_WAN")    {
            self.allow_wan = matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes");
        }
        if let Ok(v) = var("INTELNAV_QUORUM") {
            if let Ok(n) = v.parse::<u8>() { self.quorum = n.max(1); }
        }
        if let Ok(v) = var("INTELNAV_BOOTSTRAP") {
            self.bootstrap = v.split(',').filter(|s| !s.is_empty()).map(String::from).collect();
        }
        if let Ok(v) = var("INTELNAV_REGISTRY_URL") {
            self.registry_url = if v.is_empty() { None } else { Some(v) };
        }
        if let Ok(v) = var("INTELNAV_REGISTRY_MODEL") {
            self.registry_model = if v.is_empty() { None } else { Some(v) };
        }
        if let Ok(v) = var("INTELNAV_PEERS") {
            self.peers = v.split(',').filter(|s| !s.is_empty()).map(String::from).collect();
        }
        if let Ok(v) = var("INTELNAV_SPLITS") {
            self.splits = v.split(',').filter_map(|s| s.trim().parse::<u16>().ok()).collect();
        }
        if let Ok(v) = var("INTELNAV_DRAFT_MODEL") {
            self.draft_model = if v.is_empty() { None } else { Some(PathBuf::from(v)) };
        }
        if let Ok(v) = var("INTELNAV_SPEC_K") {
            if let Ok(n) = v.parse::<u16>() { self.spec_k = n; }
        }
        if let Ok(v) = var("INTELNAV_WIRE_DTYPE") {
            if !v.is_empty() { self.wire_dtype = v; }
        }
        if let Ok(v) = var("INTELNAV_LIBP2P_LISTEN") {
            if !v.is_empty() { self.libp2p_listen = v; }
        }
        if let Ok(v) = var("INTELNAV_CHUNKS_ADDR") {
            self.chunks_addr = if v.is_empty() { None } else { Some(v) };
        }
        if let Ok(v) = var("INTELNAV_FORWARD_ADDR") {
            self.forward_addr = if v.is_empty() { None } else { Some(v) };
        }
        if let Ok(v) = var("INTELNAV_RELAY_ONLY") {
            self.relay_only = matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes");
        }
        if let Ok(v) = var("INTELNAV_TIER") {
            self.default_tier = match v.to_ascii_lowercase().as_str() {
                "lan" | "t1"                 => LatencyTier::Lan,
                "wan" | "t3"                 => LatencyTier::Wan,
                "cont" | "continent" | "t2"  => LatencyTier::Continent,
                _                            => self.default_tier,
            };
        }
    }
}
