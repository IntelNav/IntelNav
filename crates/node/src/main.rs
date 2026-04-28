//! `intelnav-node` — long-running host daemon.
//!
//! Hosts the libp2p swarm, the periodic shard announce loop, the
//! chunk-server, and the inference forward listener. Stays up across
//! `intelnav` chat sessions so closing the chat window doesn't drop
//! the host off the swarm.
//!
//! Stub: this binary will spawn the full host stack once the daemon
//! integration in #28 lands. For now it parses args, sets up tracing,
//! and exits with a TODO so the build system has something to wire
//! into release packaging.

#![deny(unsafe_code)]

use anyhow::Result;
use clap::Parser;

use intelnav_core::Config;

#[derive(Parser)]
#[command(
    name = "intelnav-node",
    version,
    about = "IntelNav — host slices on the decentralized inference swarm",
    long_about = None,
)]
struct Cli {
    /// Path to an alternate config file (defaults to XDG).
    #[arg(long)]
    config: Option<std::path::PathBuf>,

    /// Increase logging verbosity (-v, -vv).
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let level = match cli.verbose {
        0 => "intelnav=info,warn",
        1 => "intelnav=debug,info",
        _ => "intelnav=trace,debug",
    };
    let filter = tracing_subscriber::EnvFilter::try_new(level).unwrap();
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();

    intelnav_app::firstrun::ensure_initialized()?;
    intelnav_app::firstrun::auto_discover_libllama_dir();
    intelnav_app::gpu_compat::ensure_runtime_overrides();
    let config = Config::load()?;
    intelnav_app::swarm_node::serve_forever(&config, config.models_dir.clone()).await
}
