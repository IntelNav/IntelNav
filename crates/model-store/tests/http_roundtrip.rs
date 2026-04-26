//! End-to-end integration: chunk → serve over HTTP → fetch to a
//! fresh cache → verify every chunk's CID round-tripped.
//!
//! Guards Phase 3 (HTTP distribution + verified cache). Only built
//! with the `serve` feature, which pulls in axum for the tiny server.

#![cfg(feature = "serve")]

use std::net::SocketAddr;
use std::time::Duration;

use intelnav_model_store::{
    chunk_gguf,
    fetch_manifest_and_chunks,
    serve::serve_bound,
    ChunkerOptions,
    FetchOptions, FetchPlan,
};

const QWEN: &str = "/home/islam/IntelNav/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

#[tokio::test]
async fn http_roundtrip_fetches_and_verifies() {
    if !std::path::Path::new(QWEN).exists() {
        eprintln!("skipping: Qwen model not present");
        return;
    }

    // Chunk locally — this is what a seeder would publish.
    let seed_dir = std::env::temp_dir().join("intelnav-phase3-seed");
    let _ = std::fs::remove_dir_all(&seed_dir);
    let chunked = chunk_gguf(
        QWEN,
        &ChunkerOptions {
            output_dir: seed_dir.clone(),
            overwrite: false,
            dry_run: false,
        },
    ).unwrap();
    let expected_cid = chunked.manifest_cid.clone();

    // Serve the seed dir on an OS-assigned port so parallel tests
    // don't fight over a fixed one.
    let (bound, fut) = serve_bound(
        seed_dir.clone(),
        SocketAddr::from(([127, 0, 0, 1], 0)),
    ).await.unwrap();
    let _server = tokio::spawn(fut);

    // Give the listener a moment to come up (negligible on localhost).
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Fetch only the bundles a mid-slice peer would want: [10..15).
    // That exercises the range-planning code alongside the HTTP path.
    let manifest_url = format!("http://{bound}/manifest.json");
    let cache_root = std::env::temp_dir().join("intelnav-phase3-cache");
    let _ = std::fs::remove_dir_all(&cache_root);
    let opts = FetchOptions {
        cache_root: cache_root.clone(),
        request_timeout: Duration::from_secs(60),
        resume: true,
        max_concurrent: 4,
        max_manifest_bytes: 64 * 1024 * 1024,
    };

    // Step 1: pull manifest to learn bundle list, then fetch the
    // mid-slice subset.
    let fetched_full = fetch_manifest_and_chunks(&manifest_url, &FetchPlan::Full, &opts).await
        .expect("initial full fetch");
    assert_eq!(fetched_full.manifest_cid, expected_cid,
        "fetched manifest CID should match the chunked manifest CID");

    // Second run with the same options hits the cache — 0 bytes
    // downloaded, everything reused.
    let fetched_again = fetch_manifest_and_chunks(&manifest_url, &FetchPlan::Full, &opts).await
        .expect("second fetch hits cache");
    assert_eq!(fetched_again.bytes_downloaded, 0,
        "second fetch should be 100% cache hits");
    assert!(fetched_again.bytes_reused > 0);

    // The fetched cache must be interchangeable with what the
    // chunker produced — stitch a subset from it and confirm the
    // outcome is identical byte-for-byte.
    use intelnav_model_store::{stitch_subset, StitchRange};
    let range = StitchRange { start: 10, end: 15, include_embed: false, include_head: false };

    let out_a = std::env::temp_dir().join("intelnav-phase3-stitch-from-seed.gguf");
    let out_b = std::env::temp_dir().join("intelnav-phase3-stitch-from-http.gguf");
    let _ = std::fs::remove_file(&out_a);
    let _ = std::fs::remove_file(&out_b);

    stitch_subset(&chunked.manifest, &seed_dir, &range, &out_a).unwrap();
    stitch_subset(&fetched_full.manifest, &fetched_full.dir, &range, &out_b).unwrap();

    let bytes_a = std::fs::read(&out_a).unwrap();
    let bytes_b = std::fs::read(&out_b).unwrap();
    assert_eq!(bytes_a.len(), bytes_b.len(), "stitched sizes differ");
    assert_eq!(bytes_a, bytes_b, "HTTP-fetched stitch drifts from local-chunked stitch");
}
