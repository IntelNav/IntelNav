//! First-run initialization.
//!
//! When the user runs `intelnav` for the first time we want them to
//! drop straight into the TUI — no manual `intelnav init`, no editing
//! of `config.toml`. This module bootstraps everything that's missing:
//!
//! - The XDG config directory.
//! - `config.toml` with sane defaults + auto-picked ports.
//! - `peer.key` (Ed25519 identity, 0600 perms).
//! - `models_dir`.
//!
//! If anything is already there we leave it alone — this is idempotent.

use std::net::TcpListener;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tracing::info;

use intelnav_core::Config;
use intelnav_crypto::Identity;

/// Outcome of the first-run check, returned so the TUI can show
/// "welcome, generated your config" once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InitReport {
    pub wrote_config:   bool,
    pub wrote_identity: bool,
    pub created_models_dir: bool,
}

impl InitReport {
    pub fn is_first_run(&self) -> bool {
        self.wrote_config || self.wrote_identity
    }
}

/// Make sure config + identity + models_dir exist on disk. Idempotent.
///
/// Picks free ports for `chunks_addr` and `forward_addr` on the first
/// write — those are then permanent (we don't want a host's advertised
/// port to drift across restarts and confuse the DHT).
pub fn ensure_initialized() -> Result<InitReport> {
    let cfg_path = Config::config_path()
        .context("could not resolve XDG config directory")?;
    let id_path = identity_path();
    let default_models = intelnav_core::config::default_models_dir();

    if let Some(parent) = cfg_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create {}", parent.display()))?;
    }

    let wrote_config = if !cfg_path.exists() {
        let mut cfg = Config::default();
        cfg.mode = intelnav_core::RunMode::Network;
        cfg.chunks_addr = Some(format!("0.0.0.0:{}", pick_free_port(8765)));
        cfg.forward_addr = Some(format!("0.0.0.0:{}", pick_free_port(7717)));
        let toml_str = toml::to_string_pretty(&cfg)?;
        std::fs::write(&cfg_path, toml_str)
            .with_context(|| format!("write {}", cfg_path.display()))?;
        info!(path = %cfg_path.display(), "wrote default config");
        true
    } else {
        false
    };

    let wrote_identity = if !id_path.exists() {
        if let Some(p) = id_path.parent() {
            std::fs::create_dir_all(p)
                .with_context(|| format!("create {}", p.display()))?;
        }
        let id = Identity::generate();
        std::fs::write(&id_path, hex::encode(id.seed()))
            .with_context(|| format!("write {}", id_path.display()))?;
        set_private_perms(&id_path)?;
        info!(path = %id_path.display(), peer_id = %id.peer_id(), "generated identity");
        true
    } else {
        false
    };

    let created_models_dir = if !default_models.exists() {
        std::fs::create_dir_all(&default_models)
            .with_context(|| format!("create {}", default_models.display()))?;
        true
    } else {
        false
    };

    Ok(InitReport { wrote_config, wrote_identity, created_models_dir })
}

/// Try to bind `preferred`; if it's taken let the OS pick. Either way
/// return *a* port number. We don't actually keep the listener — this
/// is a "is the port likely free right now" probe.
fn pick_free_port(preferred: u16) -> u16 {
    if let Ok(l) = TcpListener::bind(("0.0.0.0", preferred)) {
        let port = l.local_addr().map(|a| a.port()).unwrap_or(preferred);
        drop(l);
        return port;
    }
    if let Ok(l) = TcpListener::bind(("0.0.0.0", 0)) {
        if let Ok(addr) = l.local_addr() {
            let port = addr.port();
            drop(l);
            return port;
        }
    }
    preferred
}

fn identity_path() -> PathBuf {
    directories::ProjectDirs::from("io", "intelnav", "intelnav")
        .map(|p| p.data_dir().join("peer.key"))
        .unwrap_or_else(|| PathBuf::from("./peer.key"))
}

#[cfg(unix)]
fn set_private_perms(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(path)?.permissions();
    perms.set_mode(0o600);
    std::fs::set_permissions(path, perms)?;
    Ok(())
}

#[cfg(not(unix))]
fn set_private_perms(_path: &Path) -> Result<()> { Ok(()) }
