//! Service-manager integration.
//!
//! The user should never need to type `systemctl`. This module
//! installs `intelnav-node` as a systemd user unit on first call,
//! then everything is `systemctl --user` — no further sudo.
//!
//! Linux only for v1. macOS and Windows are deferred until the
//! Linux flow has been tested in the wild.

use anyhow::Result;

#[cfg(target_os = "linux")]
pub mod systemd;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServiceStatus {
    /// Service unit is installed and enabled — survives reboot.
    Active,
    /// Daemon is reachable on its control socket but the service unit
    /// isn't installed (user is running `intelnav-node` manually).
    RunningAdHoc,
    /// Daemon not running at all.
    Inactive,
    /// Platform doesn't support automated service install.
    Unsupported,
}

/// Detect the current state. Best-effort; returns [`Inactive`] on any
/// platform-specific failure rather than bubbling errors — the TUI
/// just wants something to display.
pub fn status() -> ServiceStatus {
    #[cfg(target_os = "linux")]
    return systemd::status();
    #[cfg(not(target_os = "linux"))]
    ServiceStatus::Unsupported
}

/// Install the service unit and start the daemon. Pops at most one
/// elevation prompt. Idempotent: if the service is already installed,
/// this is a no-op.
pub async fn install() -> Result<()> {
    #[cfg(target_os = "linux")]
    return systemd::install().await;
    #[cfg(not(target_os = "linux"))]
    anyhow::bail!("automatic service install is Linux-only for now")
}

/// Stop and remove the service unit. Pops one elevation prompt.
pub async fn uninstall() -> Result<()> {
    #[cfg(target_os = "linux")]
    return systemd::uninstall().await;
    #[cfg(not(target_os = "linux"))]
    anyhow::bail!("automatic service uninstall is Linux-only for now")
}
