//! systemd integration for `intelnav-node`.
//!
//! Strategy: install a *user* unit at `~/.config/systemd/user/intelnav-node.service`,
//! then `loginctl enable-linger` so the unit survives logout/reboot.
//! The first time we install, `pkexec loginctl enable-linger <user>`
//! pops an elevation prompt (~once per machine). After that everything
//! is `systemctl --user` — no further sudo.
//!
//! Why user units instead of system units? They live in the user's
//! home, run as the user (no daemon UID), and uninstall cleanly. The
//! only system-level state we touch is the linger flag.

use std::path::PathBuf;

use anyhow::{Context, Result};
use tokio::process::Command;
use tracing::{info, warn};

use super::ServiceStatus;

const UNIT_NAME: &str = "intelnav-node.service";

fn user_unit_path() -> PathBuf {
    directories::BaseDirs::new()
        .map(|b| b.config_dir().join("systemd/user").join(UNIT_NAME))
        .unwrap_or_else(|| PathBuf::from(format!("/tmp/{UNIT_NAME}")))
}

fn current_user() -> String {
    std::env::var("USER")
        .or_else(|_| std::env::var("LOGNAME"))
        .unwrap_or_else(|_| "root".into())
}

/// Locate the `intelnav-node` binary the unit should `ExecStart`. We
/// prefer the same binary the user installed (look up the running
/// `intelnav` and assume the daemon lives next to it). Falls back to
/// `intelnav-node` on $PATH.
fn locate_node_binary() -> PathBuf {
    if let Ok(self_exe) = std::env::current_exe() {
        if let Some(parent) = self_exe.parent() {
            let candidate = parent.join("intelnav-node");
            if candidate.exists() { return candidate; }
        }
    }
    // Fall back to PATH lookup at runtime — systemd will resolve
    // it via $PATH at unit start.
    PathBuf::from("intelnav-node")
}

fn render_unit() -> String {
    let exec = locate_node_binary();
    format!(
        "[Unit]\n\
         Description=IntelNav swarm node\n\
         After=network-online.target\n\
         Wants=network-online.target\n\
         \n\
         [Service]\n\
         Type=simple\n\
         ExecStart={exec}\n\
         Restart=on-failure\n\
         RestartSec=5\n\
         \n\
         [Install]\n\
         WantedBy=default.target\n",
        exec = exec.display(),
    )
}

pub fn status() -> ServiceStatus {
    if !user_unit_path().exists() {
        // Maybe the user is running the daemon ad-hoc — check the
        // control socket as a fallback signal.
        let sock = crate::control::default_socket_path();
        if sock.exists() {
            return ServiceStatus::RunningAdHoc;
        }
        return ServiceStatus::Inactive;
    }
    // Unit exists; ask systemd whether it's active. Cheap, no elevation.
    let output = std::process::Command::new("systemctl")
        .args(["--user", "is-active", UNIT_NAME])
        .output();
    match output {
        Ok(o) if o.status.success() => ServiceStatus::Active,
        _ => ServiceStatus::Inactive,
    }
}

pub async fn install() -> Result<()> {
    let path = user_unit_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create {}", parent.display()))?;
    }
    std::fs::write(&path, render_unit())
        .with_context(|| format!("write {}", path.display()))?;
    info!(unit = %path.display(), "wrote systemd user unit");

    // Enable linger so the unit survives logout / reboot. This is the
    // ONE step that needs root. Use pkexec so the user gets a single
    // graphical prompt rather than a tty password.
    let user = current_user();
    let linger = Command::new("pkexec")
        .args(["loginctl", "enable-linger", &user])
        .status()
        .await
        .context("invoke pkexec loginctl")?;
    if !linger.success() {
        warn!("loginctl enable-linger failed — daemon will only run while logged in");
    }

    // Reload + enable + start. None of these need root for user units.
    run("systemctl", ["--user", "daemon-reload"]).await?;
    run("systemctl", ["--user", "enable", "--now", UNIT_NAME]).await?;
    info!("intelnav-node service installed and started");
    Ok(())
}

pub async fn uninstall() -> Result<()> {
    let path = user_unit_path();
    let _ = run("systemctl", ["--user", "disable", "--now", UNIT_NAME]).await;
    if path.exists() {
        std::fs::remove_file(&path)
            .with_context(|| format!("remove {}", path.display()))?;
    }
    let _ = run("systemctl", ["--user", "daemon-reload"]).await;
    info!("intelnav-node service uninstalled");
    Ok(())
}

async fn run<I, S>(program: &str, args: I) -> Result<()>
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    let status = Command::new(program)
        .args(args)
        .status()
        .await
        .with_context(|| format!("invoke {program}"))?;
    if !status.success() {
        anyhow::bail!("{program} exited with {status}");
    }
    Ok(())
}
