//! `intelnav` — chat client for the IntelNav swarm.
//!
//! Thin binary: parses args, sets up tracing, and hands off to
//! `intelnav-app`. The substantive code lives there so the
//! `intelnav-node` daemon can share it.

#![deny(unsafe_code)]

use anyhow::Result;
use clap::{Parser, Subcommand};

use intelnav_app::{cmd, firstrun, gate, gpu_compat, tui};
use intelnav_core::{Config, RunMode};

#[derive(Parser)]
#[command(
    name = "intelnav",
    version,
    about = "IntelNav — chat through a decentralized inference swarm",
    long_about = None,
)]
struct Cli {
    /// Path to an alternate config file (defaults to XDG).
    #[arg(long, global = true)]
    config: Option<std::path::PathBuf>,

    /// Backend mode: local | network. Env: INTELNAV_MODE.
    #[arg(long, global = true)]
    mode: Option<RunMode>,

    /// Increase logging verbosity (-v, -vv).
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    verbose: u8,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Interactive chat REPL (default).
    Chat {
        #[arg(short, long)]
        model: Option<String>,
        #[arg(short, long)]
        quorum: Option<u8>,
        #[arg(long)]
        allow_wan: bool,
    },

    /// Non-interactive one-shot query.
    Ask {
        #[arg(short, long)]
        model: Option<String>,
        prompt: Option<String>,
    },

    /// Bridge to the Python shard server.
    Node {
        #[arg(long, default_value = "/tmp/intelnav_shard.sock")]
        shard: String,
    },

    /// List local models in `models_dir`.
    Models {
        #[arg(long)]
        json: bool,
    },

    /// Preflight checks.
    Doctor,

    /// Write a default config file and generate a peer identity.
    Init {
        #[arg(long)]
        force: bool,
    },
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Auto-init on first run: writes config.toml + peer.key + models_dir
    // if any are missing. Idempotent on subsequent runs.
    let init_report = firstrun::ensure_initialized()?;
    // Probe the GPU and set HSA_OVERRIDE_GFX_VERSION if our libllama
    // tarballs would otherwise fail on this card. Must run BEFORE any
    // libllama operation — we're still single-threaded here.
    gpu_compat::ensure_runtime_overrides();

    let mut config = Config::load()?;
    if let Some(m) = cli.mode {
        config.mode = m;
    }
    let _ = init_report;

    let is_tui = matches!(cli.command, None | Some(Command::Chat { .. }));

    // Run the contribution gate BEFORE we redirect stderr to the log
    // file (which the TUI bootstrap below does to keep tracing from
    // painting over Ratatui). If we ran it after, the gate's "you must
    // contribute" explainer would silently disappear into the log.
    if is_tui {
        match gate::check(&config) {
            gate::GateState::Pass(_) => {}
            gate::GateState::NeedsContribution { suggestion, hardware_tier } => {
                print_gate_block(suggestion, hardware_tier);
                return Ok(());
            }
        }
    }

    let level = match cli.verbose {
        0 => "intelnav=info,warn",
        1 => "intelnav=debug,info",
        _ => "intelnav=trace,debug",
    };
    let filter = tracing_subscriber::EnvFilter::try_new(level).unwrap();

    if is_tui {
        let log_path = config.log_path();
        if let Some(parent) = log_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let file = std::fs::OpenOptions::new()
            .create(true).append(true).open(&log_path)?;
        // Rebind raw FD 2 so native deps that write directly to
        // stderr can't paint over the Ratatui canvas.
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            let target = file.as_raw_fd();
            #[allow(unsafe_code)]
            unsafe { libc::dup2(target, 2); }
        }
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_ansi(false)
            .with_writer(std::sync::Mutex::new(file))
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(std::io::stderr)
            .init();
    }

    match cli.command.unwrap_or(Command::Chat { model: None, quorum: None, allow_wan: false }) {
        Command::Chat { model, quorum, allow_wan } => {
            // Gate already checked above (must run before stderr is
            // redirected to the TUI log file).
            tui::run(&config, config.mode, model, quorum, allow_wan).await
        }
        Command::Ask { model, prompt } => {
            let text = match prompt {
                Some(p) => p,
                None => {
                    use std::io::Read;
                    let mut s = String::new();
                    std::io::stdin().read_to_string(&mut s)?;
                    s
                }
            };
            cmd::ask(&config, config.mode, model, &text).await
        }
        Command::Node { shard } => cmd::node(&config, &shard).await,
        Command::Models { json } => cmd::models(&config, json).await,
        Command::Doctor          => cmd::doctor(&config).await,
        Command::Init { force }  => cmd::init(force).await,
    }
}

/// Pre-TUI explainer for users who haven't picked a slice yet.
///
/// Writes to stdout (the user's terminal) before the TUI's stderr
/// redirect kicks in, so the message actually reaches them.
///
/// Shape of the message depends on hardware tier — capable hardware
/// doesn't see relay-only as a suggested option (they'd be hurting
/// the network by leeching). The env var still works as an override
/// for power users who know what they're doing.
fn print_gate_block(
    suggestion: Option<intelnav_app::gate::Suggestion>,
    tier: intelnav_app::gate::HardwareTier,
) {
    use intelnav_app::gate::HardwareTier;

    println!();
    println!("\x1b[1mIntelNav requires every peer to contribute.\x1b[0m");
    println!();

    match tier {
        HardwareTier::Capable => {
            println!("Your hardware is plenty for hosting. \x1b[1mPlease host a slice.\x1b[0m");
            println!("The network only works because capable peers commit their hardware.");
            println!();
            if let Some(s) = suggestion {
                let (start, end) = s.range;
                println!("  Suggested:  \x1b[32m{}\x1b[0m  layers [{start}..{end})",
                    s.entry.display_name);
            }
            println!();
            println!("  How to host:");
            println!("    \x1b[1mINTELNAV_RELAY_ONLY=1 intelnav\x1b[0m   (one-time, to reach the TUI)");
            println!("    Inside the TUI: \x1b[1m/models\x1b[0m → highlight a row → press \x1b[1mc\x1b[0m");
            println!("    Then \x1b[1m/service install\x1b[0m to make it permanent across reboots.");
            println!();
        }
        HardwareTier::Modest => {
            println!("You're not hosting a slice yet. Two ways forward:");
            println!();
            if let Some(s) = suggestion {
                let (start, end) = s.range;
                let fit_label = match s.fit {
                    intelnav_app::catalog::Fit::Fits  => "comfortable",
                    intelnav_app::catalog::Fit::Tight => "tight (close to RAM limit)",
                    intelnav_app::catalog::Fit::TooBig => "too big",
                };
                println!("  1. \x1b[32mHost a slice\x1b[0m \x1b[1m(strongly preferred)\x1b[0m:");
                println!("       \x1b[1m{}\x1b[0m  layers [{start}..{end})  ({fit_label})",
                    s.entry.display_name);
                println!();
                println!("       \x1b[1mINTELNAV_RELAY_ONLY=1 intelnav\x1b[0m to reach the TUI,");
                println!("       then \x1b[1m/models\x1b[0m → highlight → press \x1b[1mc\x1b[0m to contribute.");
            }
            println!();
            println!("  2. \x1b[36mRelay only\x1b[0m — DHT routing, no inference:");
            println!("       \x1b[1mINTELNAV_RELAY_ONLY=1 intelnav\x1b[0m");
            println!("       (Make permanent: set \x1b[1mrelay_only = true\x1b[0m in ~/.config/intelnav/config.toml)");
            println!();
        }
        HardwareTier::Constrained => {
            println!("Your hardware is below the catalog's hosting floor.");
            println!("\x1b[36mRelay-only mode\x1b[0m is the right path:");
            println!();
            println!("    \x1b[1mINTELNAV_RELAY_ONLY=1 intelnav\x1b[0m");
            println!();
            println!("Your daemon will participate in the DHT (which still helps the network)");
            println!("but won't run inference layers.");
            println!("Make permanent: set \x1b[1mrelay_only = true\x1b[0m in ~/.config/intelnav/config.toml.");
            println!();
        }
    }
}
