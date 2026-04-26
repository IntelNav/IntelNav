//! CI-safe end-to-end Path B test. Uses a synthetic GGUF (see
//! `tests/common/mod.rs`) so it runs on any OS / architecture
//! without external downloads.
//!
//! What it covers, in one test:
//!
//! 1. The chunker produces a stable manifest CID for identical input
//!    (streaming-write path stays deterministic).
//! 2. `verify_chunks` roundtrips — every bundle and every member CID
//!    matches what was written.
//! 3. The stitcher assembles a subset for a layer range without
//!    allocating per-tensor heap copies.
//! 4. The stitched output parses back as a valid GGUF via our own
//!    parser (we can't assert libllama acceptance here — the
//!    synthetic arch isn't a real one).
//!
//! The corresponding libllama-acceptance tests live in
//! `intelnav-ggml/tests/stitched_load.rs` and need an installed
//! libllama + a real GGUF.

mod common;

use intelnav_model_store::{
    chunk_gguf, stitch_subset, verify_chunks, ChunkerOptions, StitchRange,
};
use intelnav_model_store::gguf::Gguf;

#[test]
fn synth_gguf_full_path_b_roundtrip() {
    let work = common::unique_tmpdir("ci-roundtrip");
    std::fs::create_dir_all(&work).unwrap();
    let gguf_path = work.join("synth.gguf");
    common::synth_gguf(&gguf_path).expect("build synthetic GGUF");

    // -- Chunker: two runs must produce the same CIDs. --
    let seed_dir = work.join("seed");
    let opts = ChunkerOptions { output_dir: seed_dir.clone(), overwrite: true, dry_run: false };
    let a = chunk_gguf(&gguf_path, &opts).unwrap();
    let b = chunk_gguf(&gguf_path, &opts).unwrap();
    assert_eq!(a.manifest_cid, b.manifest_cid, "chunker output drifted between runs");
    assert_eq!(a.n_bundles, 4, "synth model has embed + 2 blocks + head");
    assert_eq!(a.manifest.n_layers, 2);

    // -- Verifier: every chunk's CID matches what's on disk. --
    verify_chunks(&seed_dir).expect("verify_chunks");

    // -- Stitcher: full-range subset. --
    let full_out = work.join("stitched-full.gguf");
    let range = StitchRange { start: 0, end: 2, include_embed: true, include_head: true };
    let outcome = stitch_subset(&a.manifest, &seed_dir, &range, &full_out).unwrap();
    assert_eq!(outcome.n_tensors, a.manifest.gguf.n_tensors);
    // Our parser must re-read the stitched file cleanly.
    let g = Gguf::open(&full_out).expect("stitched full gguf reparses");
    assert_eq!(g.n_tensors, a.manifest.gguf.n_tensors);

    // -- Stitcher: mid-slice subset. Renumbers + emits the smaller index. --
    let mid_out = work.join("stitched-mid.gguf");
    let mid_range = StitchRange { start: 1, end: 2, include_embed: false, include_head: false };
    let mid_outcome = stitch_subset(&a.manifest, &seed_dir, &mid_range, &mid_out).unwrap();
    // Only block 1's tensors should be kept (2 tensors: attn_norm + ffn_norm).
    assert_eq!(mid_outcome.n_tensors, 2);
    let g = Gguf::open(&mid_out).unwrap();
    let names: Vec<String> = g.tensors().unwrap().iter().map(|t| t.name.to_string()).collect();
    assert!(names.iter().all(|n| n.starts_with("blk.0.")),
        "mid-slice should renumber to blk.0; got {names:?}");
}

#[cfg(feature = "serve")]
#[tokio::test]
async fn synth_gguf_http_roundtrip() {
    use std::net::SocketAddr;
    use std::time::Duration;
    use intelnav_model_store::{
        fetch_chunks, fetch_manifest_only, serve::serve_bound,
        FetchOptions, FetchPlan,
    };

    let work = common::unique_tmpdir("ci-http");
    std::fs::create_dir_all(&work).unwrap();
    let gguf_path = work.join("synth.gguf");
    common::synth_gguf(&gguf_path).expect("build synthetic GGUF");

    let seed_dir = work.join("seed");
    let chunked = chunk_gguf(
        &gguf_path,
        &ChunkerOptions { output_dir: seed_dir.clone(), overwrite: true, dry_run: false },
    ).unwrap();

    // Stand up the dev server on a random port.
    let (bound, fut) = serve_bound(&seed_dir, SocketAddr::from(([127, 0, 0, 1], 0)))
        .await.unwrap();
    let _server = tokio::spawn(fut);
    tokio::time::sleep(Duration::from_millis(50)).await;
    let manifest_url = format!("http://{bound}/manifest.json");

    // Fetch manifest only, then the mid-slice plan.
    let cache_root = work.join("cache");
    let opts = FetchOptions {
        cache_root,
        request_timeout: Duration::from_secs(30),
        resume: true,
        max_concurrent: 2,
        max_manifest_bytes: 1 << 20,
    };
    let fetched = fetch_manifest_only(&manifest_url, &opts).await.unwrap();
    assert_eq!(fetched.manifest_cid, chunked.manifest_cid);

    let plan = FetchPlan::for_range(&fetched.manifest, 1, 2);
    let out = fetch_chunks(&fetched, &plan, &opts).await.unwrap();
    // Mid-slice: header chunk + one bundle = 2 chunks.
    assert!(out.bytes_downloaded > 0);

    // Second run: 100% cache hits.
    let out2 = fetch_chunks(&fetched, &plan, &opts).await.unwrap();
    assert_eq!(out2.bytes_downloaded, 0);
    assert!(out2.bytes_reused > 0);

    // Stitch from the fetched cache — must match stitch from seed byte-for-byte.
    let range = StitchRange { start: 1, end: 2, include_embed: false, include_head: false };
    let a = work.join("stitch-seed.gguf");
    let b = work.join("stitch-http.gguf");
    stitch_subset(&chunked.manifest, &seed_dir, &range, &a).unwrap();
    stitch_subset(&fetched.manifest, &fetched.dir, &range, &b).unwrap();
    assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap(),
        "HTTP-fetched stitch drifts from seed-stitched");
}
