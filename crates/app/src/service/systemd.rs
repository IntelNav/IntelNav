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
use std::time::Duration;

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

/// Locate the `intelnav-node` binary as an *absolute* path. systemd's
/// ExecStart path resolution is restrictive (limited PATH search), so
/// passing a bare name fails with status 203/EXEC.
///
/// Returns `Err` with an actionable message when the binary genuinely
/// can't be found — the install bails before generating a unit that
/// would just crash-loop.
fn locate_node_binary() -> Result<PathBuf> {
    // 1. Sibling of the running `intelnav` binary (the common case
    //    when the user just `cargo build --release`'d both).
    if let Ok(self_exe) = std::env::current_exe() {
        if let Some(parent) = self_exe.parent() {
            let candidate = parent.join("intelnav-node");
            if candidate.exists() { return Ok(candidate); }
        }
    }
    // 2. Anywhere on $PATH, but resolved to absolute now (not deferred
    //    to systemd's restricted PATH at start time).
    if let Ok(path) = std::env::var("PATH") {
        for dir in path.split(':') {
            let candidate = PathBuf::from(dir).join("intelnav-node");
            if candidate.exists() { return Ok(candidate); }
        }
    }
    anyhow::bail!(
        "couldn't find `intelnav-node` next to the running `intelnav` or on $PATH. \
         Build it first: `cargo build --release -p intelnav-node`"
    )
}

/// Forward env vars the daemon needs at runtime. systemd starts user
/// units with a minimal environment — your shell's `INTELNAV_*` vars
/// (notably `INTELNAV_LIBLLAMA_DIR`) won't be there unless we copy
/// them into the unit explicitly.
fn forwarded_env() -> Vec<(String, String)> {
    let keys = [
        "INTELNAV_LIBLLAMA_DIR",
        "INTELNAV_LIBP2P_LISTEN",
        "INTELNAV_CHUNKS_ADDR",
        "INTELNAV_FORWARD_ADDR",
        "INTELNAV_MODELS_DIR",
        "INTELNAV_DEVICE",
        "INTELNAV_BOOTSTRAP",
        // gpu_compat sets this if needed; carry it into systemd so the
        // daemon's own libllama load also runs on adjacent-arch cards.
        "HSA_OVERRIDE_GFX_VERSION",
    ];
    keys.iter()
        .filter_map(|k| std::env::var(k).ok().map(|v| (k.to_string(), v)))
        .collect()
}

fn render_unit() -> Result<String> {
    let exec = locate_node_binary()?;
    let env_lines: String = forwarded_env()
        .iter()
        .map(|(k, v)| format!("Environment={k}={v}\n"))
        .collect();
    Ok(format!(
        "[Unit]\n\
         Description=IntelNav swarm node\n\
         Documentation=https://intelnav.net\n\
         After=network-online.target\n\
         Wants=network-online.target\n\
         \n\
         [Service]\n\
         Type=simple\n\
         ExecStart={exec}\n\
         Restart=on-failure\n\
         RestartSec=5\n\
         {env_lines}\n\
         [Install]\n\
         WantedBy=default.target\n",
        exec = exec.display(),
    ))
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
    let unit = render_unit()?;
    let path = user_unit_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create {}", parent.display()))?;
    }
    std::fs::write(&path, unit)
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

    // Wait briefly for the unit to actually reach Active. A returning
    // `enable --now` only means systemd accepted the start command —
    // the unit may immediately exit (binary missing, env wrong, ...).
    if !wait_for_active(Duration::from_secs(4)).await {
        let logs = recent_journal().await.unwrap_or_default();
        anyhow::bail!(
            "service installed but failed to start. Recent journal:\n{logs}"
        );
    }
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

async fn wait_for_active(budget: Duration) -> bool {
    let deadline = std::time::Instant::now() + budget;
    while std::time::Instant::now() < deadline {
        let ok = std::process::Command::new("systemctl")
            .args(["--user", "is-active", UNIT_NAME])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if ok { return true; }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    false
}

async fn recent_journal() -> Option<String> {
    let out = Command::new("journalctl")
        .args(["--user", "-u", UNIT_NAME, "--no-pager", "-n", "10"])
        .output()
        .await
        .ok()?;
    if !out.status.success() { return None; }
    let mut s = String::from_utf8_lossy(&out.stdout).to_string();
    // Trim each line to a reasonable width so the TUI message stays readable.
    let trimmed: String = s.lines().map(|l| {
        if l.len() > 140 { format!("{}…", &l[..140]) } else { l.to_string() }
    }).collect::<Vec<_>>().join("\n");
    s = trimmed;
    Some(s)
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
