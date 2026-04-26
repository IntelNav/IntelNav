//! End-to-end Phase 4 test: seed a chunk dir, fetch it over a real
//! libp2p swarm on loopback, confirm byte-identity with the seed.
//!
//! Uses the synthetic GGUF fixture so there's no model download.
//! Two in-process nodes on 127.0.0.1:
//!
//!   * Seeder: calls `announce` for every chunk CID + the manifest.
//!   * Fetcher: calls `fetch_manifest_and_chunks_p2p(root_cid, ...)`
//!     with the seeder's listen addr as bootstrap. Pulls manifest,
//!     plans, pulls chunks, verifies.
//!
//! Only built with the `p2p` feature; default CI skips it.

#![cfg(feature = "p2p")]

mod common;

use std::time::Duration;

use libp2p::{identity, Multiaddr};
use intelnav_model_store::{
    chunk_gguf, stitch_subset, ChunkerOptions, StitchRange,
    p2p::{fetch_manifest_and_chunks_p2p_with, spawn_node, P2pFetchOptions},
    FetchPlan,
};

#[tokio::test]
async fn p2p_roundtrip_seeder_to_fetcher() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn,intelnav_model_store=debug")))
        .try_init();

    let work = common::unique_tmpdir("p2p-roundtrip");
    std::fs::create_dir_all(&work).unwrap();
    let gguf_path = work.join("synth.gguf");
    common::synth_gguf(&gguf_path).expect("build synthetic GGUF");

    // Chunk on the seeder's disk.
    let seed_dir = work.join("seed");
    let chunked = chunk_gguf(
        &gguf_path,
        &ChunkerOptions { output_dir: seed_dir.clone(), overwrite: true, dry_run: false },
    ).unwrap();

    // -- Start seeder node and announce everything it holds --
    let seeder_kp = identity::Keypair::generate_ed25519();
    let seeder = spawn_node(
        seeder_kp,
        "/ip4/127.0.0.1/tcp/0".parse().unwrap(),
        Some(seed_dir.clone()),
    ).await.unwrap();
    let seeder_addr: Multiaddr = format!(
        "{}/p2p/{}",
        seeder.listen_addrs.first().expect("seeder bound"),
        seeder.peer_id,
    ).parse().unwrap();
    eprintln!("seeder at {seeder_addr}");

    // Park the manifest bytes in the seed dir so the seeder can
    // serve them by CID — `load_chunk_from_dir` checks both
    // `chunks/<cid>.bin` and the flat `<cid>.bin` layout.
    std::fs::write(
        seed_dir.join(format!("{}.bin", chunked.manifest_cid)),
        &chunked.manifest_bytes,
    ).unwrap();

    // -- Start fetcher BEFORE announcing, and connect them --
    //
    // Kad's `start_providing` only publishes the record to peers
    // already in the routing table. If the seeder announces in
    // isolation, its provider records sit dead locally. We connect
    // first, give identify time to exchange, THEN announce.
    let fetcher_kp = identity::Keypair::generate_ed25519();
    let fetcher = spawn_node(
        fetcher_kp.clone(),
        "/ip4/127.0.0.1/tcp/0".parse().unwrap(),
        None,
    ).await.unwrap();
    fetcher.dial(seeder_addr.clone()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(500)).await;

    seeder.announce(&chunked.manifest_cid).await.unwrap();
    seeder.announce(&chunked.manifest.header_chunk.cid).await.unwrap();
    for b in &chunked.manifest.bundles {
        seeder.announce(&b.cid).await.unwrap();
    }
    // Let provider records replicate across the two routing tables.
    tokio::time::sleep(Duration::from_millis(500)).await;

    // -- Fetch manifest + a mid-slice plan via the already-connected fetcher node --
    let cache_root = work.join("fetcher-cache");
    std::fs::create_dir_all(&cache_root).unwrap();
    let plan = FetchPlan::Bundles(vec!["blk.1".to_string()]);
    let out = fetch_manifest_and_chunks_p2p_with(
        &fetcher,
        &chunked.manifest_cid,
        &plan,
        &cache_root,
        1 << 20,
    ).await.expect("p2p fetch");
    assert!(out.bytes_downloaded > 0);

    // Silence unused-import warning on P2pFetchOptions import path.
    let _ = P2pFetchOptions::default;

    // Stitch from the P2P-fetched cache and from the seed cache —
    // must be byte-identical (same correctness gate as http_roundtrip).
    let range = StitchRange { start: 1, end: 2, include_embed: false, include_head: false };
    let seed_out = work.join("stitch-seed.gguf");
    let p2p_out  = work.join("stitch-p2p.gguf");
    stitch_subset(&chunked.manifest, &seed_dir, &range, &seed_out).unwrap();
    stitch_subset(&out.manifest,      &out.dir,   &range, &p2p_out).unwrap();
    assert_eq!(std::fs::read(&seed_out).unwrap(), std::fs::read(&p2p_out).unwrap(),
        "p2p-fetched stitch differs from seed-stitched");
}
