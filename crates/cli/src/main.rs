//! `intelnav` — the user-facing CLI.
//!
//! Mirrors the ergonomics of Claude Code: a primary interactive `chat` REPL,
//! terse one-shot `ask`, plus operator commands (`gateway`, `models`, `peers`,
//! `health`, `doctor`, `init`).

#![deny(unsafe_code)]

mod banner;
mod browser;
mod catalog;
mod chain_driver;
mod chat;
mod cmd;
mod download;
mod local;
mod shimmer;
mod slash;
mod theme;
mod tui;

use anyhow::Result;
use clap::{Parser, Subcommand};

use intelnav_core::{Config, RunMode};

#[derive(Parser)]
#[command(
    name = "intelnav",
    version,
    about = "IntelNav — decentralized pipeline-parallel LLM inference",
    long_about = None,
)]
struct Cli {
    /// Path to an alternate config file (defaults to XDG).
    #[arg(long, global = true)]
    config: Option<std::path::PathBuf>,

    /// Gateway URL override. Also settable via INTELNAV_GATEWAY_URL.
    #[arg(long, global = true, env = "INTELNAV_GATEWAY_URL")]
    gateway: Option<String>,

    /// Backend mode: auto | local | network. Env: INTELNAV_MODE.
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
        /// Model to use; overrides config default.
        #[arg(short, long)]
        model: Option<String>,

        /// Quorum over disjoint shard chains.
        #[arg(short, long)]
        quorum: Option<u8>,

        /// Opt in to cross-continent (T3) routes.
        #[arg(long)]
        allow_wan: bool,
    },

    /// Non-interactive one-shot query.
    Ask {
        /// Model to use.
        #[arg(short, long)]
        model: Option<String>,

        /// Prompt text. If omitted, reads from stdin.
        prompt: Option<String>,
    },

    /// Run the local OpenAI-compatible gateway (paper §10).
    Gateway {
        /// Bind address.  Default: 127.0.0.1:8787
        #[arg(long)]
        bind: Option<String>,

        /// Disable mDNS peer discovery.
        #[arg(long)]
        no_mdns: bool,
    },

    /// Run a contributor (shard) node.  Bridges to the Python shard server.
    Node {
        /// Address of the local shard server's Unix socket or TCP endpoint.
        #[arg(long, default_value = "/tmp/intelnav_shard.sock")]
        shard: String,
    },

    /// List models available on the network.
    Models {
        /// Print as JSON instead of a formatted table.
        #[arg(long)]
        json: bool,
    },

    /// List peers known to the gateway.
    Peers {
        #[arg(long)]
        json: bool,
    },

    /// Gateway + upstream + network health snapshot.
    Health,

    /// Preflight checks (gateway reachable, identity valid, mDNS, etc.).
    Doctor,

    /// Write a default config file and generate a peer identity.
    Init {
        /// Overwrite an existing config.
        #[arg(long)]
        force: bool,
    },
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // ---- config ----
    let mut config = Config::load()?;
    if let Some(g) = cli.gateway {
        config.gateway_url = g;
    }
    if let Some(m) = cli.mode {
        config.mode = m;
    }

    // The interactive TUI owns the screen — stderr writes would paint
    // over the Ratatui canvas. For that one command, send tracing to
    // a log file (and redirect raw stderr there too, to catch any
    // stray `eprintln!` from deps). All other commands keep the usual
    // stderr writer so operators see logs live.
    let is_tui = matches!(
        cli.command,
        None | Some(Command::Chat { .. })
    );
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
        // Rebind raw FD 2 so llama.cpp / reqwest / any native dep
        // that writes directly to stderr goes to the log file
        // instead of painting over the Ratatui canvas.
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
            let mode = resolve_mode(&config).await;
            tui::run(&config, mode, model, quorum, allow_wan).await
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
            let mode = resolve_mode(&config).await;
            cmd::ask(&config, mode, model, &text).await
        }
        Command::Gateway { bind, no_mdns } => {
            let mut cfg = config.clone();
            if let Some(b) = bind { cfg.gateway_bind = b; }
            intelnav_gateway::run(cfg, !no_mdns).await.map_err(Into::into)
        }
        Command::Node { shard } => cmd::node(&config, &shard).await,
        Command::Models { json } => cmd::models(&config, json).await,
        Command::Peers { json }  => cmd::peers(&config, json).await,
        Command::Health          => cmd::health(&config).await,
        Command::Doctor          => cmd::doctor(&config).await,
        Command::Init { force }  => cmd::init(force).await,
    }
}

/// Resolve `RunMode::Auto` into a concrete `Local` or `Network` choice
/// by pinging the configured gateway with a short timeout. Anything
/// other than `Auto` passes through unchanged.
async fn resolve_mode(cfg: &Config) -> RunMode {
    match cfg.mode {
        RunMode::Local   => RunMode::Local,
        RunMode::Network => RunMode::Network,
        RunMode::Auto    => {
            let url = format!("{}/v1/network/health", cfg.gateway_url.trim_end_matches('/'));
            let client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_millis(350))
                .build();
            let ok = match client {
                Ok(c) => c.get(&url).send().await.map(|r| r.status().is_success()).unwrap_or(false),
                Err(_) => false,
            };
            if ok { RunMode::Network } else { RunMode::Local }
        }
    }
}
