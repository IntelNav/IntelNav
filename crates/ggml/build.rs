// Build script for intelnav-ggml.
//
// Since task #14 phase 2 (dlopen refactor), libllama is NOT linked at
// build time. It's dlopened at runtime via the `Loader` (see
// src/loader.rs) which looks the library up from:
//
//   1. $INTELNAV_LIBLLAMA_PATH                  — absolute file path
//   2. $INTELNAV_LIBLLAMA_DIR/lib{llama,intelnav_llama}.so
//   3. $XDG_CACHE_HOME/intelnav/libllama/…       — the artifact cache
//      (future: task #14 phase 3 downloader populates it)
//   4. system loader defaults (ldconfig)
//
// This file remains only to surface the rerun-if-env-changed hooks
// so Cargo re-invokes the loader search when the env flips. No
// compilation happens here.

use std::env;

fn main() {
    for key in [
        "INTELNAV_LIBLLAMA_PATH",
        "INTELNAV_LIBLLAMA_DIR",
        "XDG_CACHE_HOME",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
        let _ = env::var(key);
    }
}
