//! `intelnav` — chat client for the IntelNav swarm.
//!
//! Thin binary: parses args, sets up tracing, and hands off to
//! `intelnav-app`. The substantive code lives there so the
//! `intelnav-node` daemon can share it.

#![deny(unsafe_code)]

use anyhow::Result;
use clap::{Parser, Subcommand};

use intelnav_app::{cmd, tui};
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

    let mut config = Config::load()?;
    if let Some(m) = cli.mode {
        config.mode = m;
    }

    let is_tui = matches!(cli.command, None | Some(Command::Chat { .. }));
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
