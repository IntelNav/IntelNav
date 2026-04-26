//! `intelnav-chunk` — split a GGUF into content-addressed chunks.
//!
//! Usage:
//!
//!     intelnav-chunk chunk <input.gguf> <output_dir> [--overwrite] [--dry-run]
//!     intelnav-chunk verify <output_dir>
//!
//! The `chunk` subcommand prints the manifest CID on the last line of
//! stdout, so it's easy to pipe: `MODEL_CID=$(intelnav-chunk chunk ...)`.

#[cfg(feature = "serve")]
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use intelnav_model_store::chunker::{chunk_gguf, verify_chunks, ChunkerOptions};
use intelnav_model_store::http::{
    fetch_chunks, fetch_manifest_and_chunks, fetch_manifest_only, FetchOptions, FetchPlan,
};

#[derive(Parser, Debug)]
#[command(name = "intelnav-chunk", about = "Chunk a GGUF file for content-addressed distribution")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Chunk a GGUF file into per-tensor CIDs + manifest.
    Chunk {
        /// Source GGUF.
        input: PathBuf,
        /// Directory to write chunks/ and manifest.json into.
        output_dir: PathBuf,
        /// Allow writing into an existing non-empty directory.
        #[arg(long)]
        overwrite: bool,
        /// Compute CIDs but don't touch disk.
        #[arg(long)]
        dry_run: bool,
    },
    /// Re-hash every chunk on disk and confirm it matches the manifest.
    Verify {
        /// Directory containing manifest.json and chunks/.
        output_dir: PathBuf,
    },
    /// Fetch a manifest + selected chunks from an HTTP URL.
    Fetch {
        /// URL to the remote `manifest.json`.
        url: String,
        /// Local cache root. Default: ~/.cache/intelnav/models.
        #[arg(long)]
        cache_root: Option<PathBuf>,
        /// Layer range to fetch (inclusive start, exclusive end).
        /// With both flags omitted, the full model is fetched.
        #[arg(long)]
        start: Option<u32>,
        #[arg(long)]
        end: Option<u32>,
    },
    /// Serve a chunk directory over HTTP. Requires the `serve` feature.
    #[cfg(feature = "serve")]
    Serve {
        /// Directory containing manifest.json + chunks/.
        root: PathBuf,
        /// Bind address.
        #[arg(long, default_value = "127.0.0.1:8645")]
        bind: SocketAddr,
    },
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");
    let res: anyhow::Result<()> = rt.block_on(async {
        match cli.cmd {
            Cmd::Chunk { input, output_dir, overwrite, dry_run } => {
                run_chunk(input, output_dir, overwrite, dry_run)
            }
            Cmd::Verify { output_dir } => run_verify(output_dir),
            Cmd::Fetch { url, cache_root, start, end } => {
                run_fetch(&url, cache_root, start, end).await
            }
            #[cfg(feature = "serve")]
            Cmd::Serve { root, bind } => {
                intelnav_model_store::serve::serve(root, bind).await
            }
        }
    });

    match res {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run_chunk(input: PathBuf, output_dir: PathBuf, overwrite: bool, dry_run: bool) -> anyhow::Result<()> {
    let opts = ChunkerOptions {
        output_dir,
        overwrite,
        dry_run,
    };
    let out = chunk_gguf(&input, &opts)?;
    let mode = if dry_run { "(dry-run) " } else { "" };
    eprintln!(
        "{mode}chunked {t} tensors into {b} bundles from {path}: header {hcid}, manifest {mcid}, wrote {bytes} bytes",
        t = out.n_tensors,
        b = out.n_bundles,
        path = input.display(),
        hcid = out.manifest.header_chunk.cid,
        mcid = out.manifest_cid,
        bytes = out.bytes_written
    );
    // Final line on stdout = the model CID, for shell scripting.
    println!("{}", out.manifest_cid);
    Ok(())
}

fn run_verify(output_dir: PathBuf) -> anyhow::Result<()> {
    verify_chunks(&output_dir)?;
    eprintln!("ok: all chunks in {} verified", output_dir.display());
    Ok(())
}

async fn run_fetch(
    url: &str,
    cache_root: Option<PathBuf>,
    start: Option<u32>,
    end: Option<u32>,
) -> anyhow::Result<()> {
    let mut opts = FetchOptions::default();
    if let Some(root) = cache_root {
        opts.cache_root = root;
    }
    let out = match (start, end) {
        (Some(s), Some(e)) => {
            // Manifest-first so we only pull the bundles this range
            // actually needs. `FetchPlan::Full` would waste bandwidth.
            let fetched = fetch_manifest_only(url, &opts).await?;
            let plan = FetchPlan::for_range(&fetched.manifest, s, e);
            fetch_chunks(&fetched, &plan, &opts).await?
        }
        (None, None) => fetch_manifest_and_chunks(url, &FetchPlan::Full, &opts).await?,
        _ => anyhow::bail!("pass both --start and --end, or neither"),
    };
    eprintln!(
        "fetched manifest {} to {} ({}B downloaded, {}B reused from cache)",
        out.manifest_cid, out.dir.display(), out.bytes_downloaded, out.bytes_reused,
    );
    println!("{}", out.manifest_cid);
    Ok(())
}
