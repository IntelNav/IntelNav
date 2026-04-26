//! Runtime loader for the IntelNav-patched libllama.
//!
//! Opens the library via `libloading` at a path chosen by
//! [`find_libllama`] (env overrides → cache → system), resolves
//! every symbol the FFI needs up-front, and stores each as a bare
//! function pointer. Call-time cost is one pointer deref — same as
//! a direct `extern "C"` call would be.
//!
//! Lifetime discipline: the `Loader` owns the `libloading::Library`,
//! which owns the `dlopen` handle. Dropping the Loader closes the
//! library. All `Model` / `Context` / `Session` handles hold an
//! `Arc<Loader>`, so the library is guaranteed to outlive the last
//! llama_model / llama_context that points into it. This is the same
//! safety pattern the candle-linked build had — just enforced via
//! Arc rather than rustc's static link graph.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context as _, Result};
use libloading::{Library, Symbol};

use crate::sys;

/// Handle to a loaded libllama. One process typically holds a single
/// Loader shared via `Arc`; the [`default`] helper lazy-inits one.
pub struct Loader {
    // Held to keep the dlopen handle alive; accessed via resolved
    // function pointers below (which point INTO this library's .text).
    #[allow(dead_code)]
    lib: Library, // last field so it drops last (after any fn pointers go out of scope)

    /// Companion ggml shared libraries (libggml-base, libggml-cpu-*,
    /// and any backend-specific ones that ship next to libllama).
    /// Preloaded BEFORE libllama so dlopen-by-SONAME succeeds without
    /// needing `LD_LIBRARY_PATH`. Held here so their dlopen handles
    /// stay alive for the life of the Loader — dropping them too
    /// early would unmap symbols libllama still uses.
    #[allow(dead_code)]
    companions: Vec<Library>,

    pub ggml_backend_load_all:      sys::ggml_backend_load_all_fn,
    pub ggml_backend_load_all_from_path: sys::ggml_backend_load_all_from_path_fn,

    pub llama_model_free:           sys::llama_model_free_fn,
    pub llama_model_get_vocab:      sys::llama_model_get_vocab_fn,
    pub llama_model_n_embd:         sys::llama_model_n_embd_fn,
    pub llama_model_n_layer:        sys::llama_model_n_layer_fn,

    pub llama_vocab_n_tokens:       sys::llama_vocab_n_tokens_fn,
    pub llama_tokenize:             sys::llama_tokenize_fn,

    pub llama_free:                 sys::llama_free_fn,

    pub llama_batch_init:           sys::llama_batch_init_fn,
    pub llama_batch_free:           sys::llama_batch_free_fn,

    pub llama_decode:               sys::llama_decode_fn,
    pub llama_embed_only:           sys::llama_embed_only_fn,
    pub llama_decode_layers:        sys::llama_decode_layers_fn,
    pub llama_head_only:            sys::llama_head_only_fn,

    pub llama_get_logits_ith:       sys::llama_get_logits_ith_fn,
    pub llama_get_embeddings_ith:   sys::llama_get_embeddings_ith_fn,

    pub llama_get_memory:           sys::llama_get_memory_fn,
    pub llama_memory_seq_rm:        sys::llama_memory_seq_rm_fn,

    pub llama_set_embeddings:       sys::llama_set_embeddings_fn,

    pub intelnav_load_model:        sys::intelnav_load_model_fn,
    pub intelnav_new_context:       sys::intelnav_new_context_fn,
    pub intelnav_trip_abort:        sys::intelnav_trip_abort_fn,

    /// Path the library was loaded from — for diagnostics + `doctor`.
    pub loaded_from: PathBuf,
}

// Safety: all fields are either `Send + Sync` primitives (the fn
// pointers) or a `libloading::Library`, which is Send + Sync per its
// crate docs. The C runtime state libllama keeps is protected by
// our own Context (not Sync) wrappers.
unsafe impl Send for Loader {}
unsafe impl Sync for Loader {}

impl Loader {
    /// Open libllama from `path` and resolve every symbol the FFI uses.
    /// Returns an error if the library can't be opened or if any
    /// required symbol is missing (which usually means the library
    /// wasn't built from the IntelNav fork).
    ///
    /// Before loading libllama itself, we preload any companion ggml
    /// shared libraries (libggml-base.so, libggml-cpu-*.so, …) from
    /// the same directory. This is what makes the loader work without
    /// LD_LIBRARY_PATH or DYLD_LIBRARY_PATH: libllama's implicit SONAME
    /// lookup finds the already-loaded handles instead of searching
    /// the system loader path. On Windows the same mechanic applies
    /// via the DLL load order.
    pub fn open(path: impl AsRef<Path>) -> Result<Arc<Self>> {
        let path = path.as_ref().to_path_buf();
        let companions = preload_companions(&path);

        // Safety: we only load libraries from paths chosen by
        // `find_libllama` — env override, cache, or system default.
        // A malicious path would need write access to one of those
        // locations, at which point the attacker already owns the box.
        let lib = unsafe { Library::new(&path) }
            .with_context(|| format!("dlopen {}", path.display()))?;

        let loader = Loader {
            ggml_backend_load_all:      *load_sym(&lib, b"ggml_backend_load_all")?,
            ggml_backend_load_all_from_path: *load_sym(&lib, b"ggml_backend_load_all_from_path")?,
            llama_model_free:           *load_sym(&lib, b"llama_model_free")?,
            llama_model_get_vocab:      *load_sym(&lib, b"llama_model_get_vocab")?,
            llama_model_n_embd:         *load_sym(&lib, b"llama_model_n_embd")?,
            llama_model_n_layer:        *load_sym(&lib, b"llama_model_n_layer")?,

            llama_vocab_n_tokens:       *load_sym(&lib, b"llama_vocab_n_tokens")?,
            llama_tokenize:             *load_sym(&lib, b"llama_tokenize")?,

            llama_free:                 *load_sym(&lib, b"llama_free")?,

            llama_batch_init:           *load_sym(&lib, b"llama_batch_init")?,
            llama_batch_free:           *load_sym(&lib, b"llama_batch_free")?,

            llama_decode:               *load_sym(&lib, b"llama_decode")?,
            llama_embed_only:           *load_sym(&lib, b"llama_embed_only")?,
            llama_decode_layers:        *load_sym(&lib, b"llama_decode_layers")?,
            llama_head_only:            *load_sym(&lib, b"llama_head_only")?,

            llama_get_logits_ith:       *load_sym(&lib, b"llama_get_logits_ith")?,
            llama_get_embeddings_ith:   *load_sym(&lib, b"llama_get_embeddings_ith")?,

            llama_get_memory:           *load_sym(&lib, b"llama_get_memory")?,
            llama_memory_seq_rm:        *load_sym(&lib, b"llama_memory_seq_rm")?,

            llama_set_embeddings:       *load_sym(&lib, b"llama_set_embeddings")?,

            intelnav_load_model:        *load_sym(&lib, b"intelnav_load_model")?,
            intelnav_new_context:       *load_sym(&lib, b"intelnav_new_context")?,
            intelnav_trip_abort:        *load_sym(&lib, b"intelnav_trip_abort")?,

            loaded_from: path,

            companions,
            lib, // moved in last — lives as long as the Loader
        };
        Ok(Arc::new(loader))
    }
}

/// Preload every companion ggml shared library sitting alongside
/// libllama. Silent on failure — a missing `libggml-cuda.so` just
/// means the CUDA backend won't load, which is diagnostic-worthy but
/// not fatal. Returns the live dlopen handles; callers must keep
/// them alive for libllama's lifetime.
fn preload_companions(libllama_path: &Path) -> Vec<Library> {
    let Some(dir) = libllama_path.parent() else { return Vec::new(); };
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new(); };

    let mut handles = Vec::new();
    // Load order matters: ggml-base is a hard dependency of every
    // backend plugin and libllama itself, so it goes first. The
    // loader itself tolerates SONAMEs loading out-of-order (the
    // kernel dynamic linker resolves forward refs), but we prefer
    // deterministic order for diagnosability.
    let mut candidates: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            let Some(name) = p.file_name().and_then(OsStr::to_str) else { return false; };
            is_companion_filename(name) && p != libllama_path
        })
        .collect();
    candidates.sort_by_key(|p| {
        let n = p.file_name().and_then(OsStr::to_str).unwrap_or("");
        // ggml-base first, then cpu variants, then GPU backends.
        if n.contains("ggml-base") { 0 }
        else if n.contains("ggml-cpu") { 1 }
        else { 2 }
    });

    for p in candidates {
        // Safety: same argument as the main library load — we trust
        // the directory the caller pointed us at, and these are
        // content we shipped alongside libllama.
        if let Ok(lib) = unsafe { Library::new(&p) } {
            handles.push(lib);
        }
    }
    handles
}

/// Match ggml library filenames we care about preloading. Deliberately
/// narrow: only ggml's own libraries that ship next to libllama in the
/// pack tarball. Third-party `.so` files in the same directory are
/// ignored.
///
/// Accepted patterns (Linux/macOS/Windows all covered):
///
///   * `libggml.so`, `libggml.so.0`, `libggml.so.0.10.0` — umbrella
///   * `libggml-base.so*`, `libggml-cpu-*.so*`, `libggml-cuda.so*`, etc.
///   * `ggml.dll`, `ggml-base.dll`, `ggml-cpu-*.dll`, …
///   * `libggml.dylib`, `libggml-*.dylib`, …
fn is_companion_filename(name: &str) -> bool {
    // Linux/macOS prefix is `libggml`, Windows has no `lib` prefix.
    let body = name.strip_prefix("lib").unwrap_or(name);
    let starts_ok = body.starts_with("ggml.") || body.starts_with("ggml-");
    if !starts_ok { return false; }

    // Must be a recognisable shared library — either a plain suffix
    // or a versioned `.so.N[.M[.K]]` chain (ELF SONAME style).
    name.ends_with(".so")
        || name.ends_with(".dylib")
        || name.ends_with(".dll")
        || name.contains(".so.")
}

fn load_sym<'lib, T: Copy>(lib: &'lib Library, name: &[u8]) -> Result<Symbol<'lib, T>> {
    unsafe { lib.get::<T>(name) }
        .with_context(|| {
            let name = String::from_utf8_lossy(name.trim_ascii_end_null()).to_string();
            format!(
                "dlsym: libllama missing symbol `{name}` — make sure it was built from the \
                 IntelNav fork of llama.cpp (github.com/IntelNav/llama.cpp)"
            )
        })
}

trait TrimAsciiEndNull {
    fn trim_ascii_end_null(&self) -> &[u8];
}
impl TrimAsciiEndNull for [u8] {
    fn trim_ascii_end_null(&self) -> &[u8] {
        let mut end = self.len();
        while end > 0 && self[end - 1] == 0 {
            end -= 1;
        }
        &self[..end]
    }
}

// ------------------------- library discovery -------------------------

/// Locate `libllama.so` (or the platform equivalent) for the current
/// process. Walks these sources, first hit wins:
///
/// 1. `$INTELNAV_LIBLLAMA_PATH` — absolute file path, unconditional.
/// 2. `$INTELNAV_LIBLLAMA_DIR` — directory; appends the platform's
///    library file name.
/// 3. The cache under `$XDG_CACHE_HOME/intelnav/libllama/**` —
///    populated by the future downloader (task #14 phase 3). The
///    newest `libllama.*` found under the cache wins.
/// 4. System default: hand a bare `libllama` / `intelnav_llama` to
///    `libloading`, which consults `ldconfig` / `DYLD_*` / `PATH`.
///
/// Returns the resolved absolute path. Callers pass it to [`Loader::open`].
pub fn find_libllama() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("INTELNAV_LIBLLAMA_PATH") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Ok(p);
        }
        return Err(anyhow!(
            "INTELNAV_LIBLLAMA_PATH={} does not exist or is not a file",
            p.display()
        ));
    }

    if let Ok(dir) = std::env::var("INTELNAV_LIBLLAMA_DIR") {
        let dir = PathBuf::from(dir);
        for name in platform_lib_names() {
            let p = dir.join(name);
            if p.is_file() {
                return Ok(p);
            }
        }
        return Err(anyhow!(
            "INTELNAV_LIBLLAMA_DIR={} has no {} — check your build output",
            dir.display(),
            platform_lib_names().join("/")
        ));
    }

    if let Some(cache_hit) = find_in_cache()? {
        return Ok(cache_hit);
    }

    // System default. We just hand `libloading` the bare name and let
    // `dlopen` / `LoadLibrary` resolve it against the OS search path.
    // `Library::new` accepts either a relative or absolute path; when
    // relative, the OS loader walks its default search dirs.
    let fallback = platform_lib_names()[0];
    Ok(PathBuf::from(fallback))
}

fn platform_lib_names() -> &'static [&'static str] {
    if cfg!(target_os = "windows") {
        &["llama.dll", "intelnav_llama.dll"]
    } else if cfg!(target_os = "macos") {
        &["libllama.dylib", "libintelnav_llama.dylib"]
    } else {
        &["libllama.so", "libintelnav_llama.so"]
    }
}

fn find_in_cache() -> Result<Option<PathBuf>> {
    let cache = cache_root()?;
    if !cache.is_dir() {
        return Ok(None);
    }

    // Walk `$CACHE/libllama/**` and return the newest matching file by
    // mtime. The future downloader will key by
    // `<INTELNAV_SHA>/<backend>/`, so the newest tarball mtime
    // corresponds to the most-recently-installed version.
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    walk(cache.as_path(), &mut |p| {
        let name = match p.file_name().and_then(OsStr::to_str) {
            Some(n) => n,
            None => return,
        };
        if !platform_lib_names().iter().any(|s| *s == name) {
            return;
        }
        let ts = match std::fs::metadata(p).and_then(|m| m.modified()) {
            Ok(t) => t,
            Err(_) => return,
        };
        match &best {
            Some((bt, _)) if *bt >= ts => {}
            _ => best = Some((ts, p.to_path_buf())),
        }
    });
    Ok(best.map(|(_, p)| p))
}

fn cache_root() -> Result<PathBuf> {
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        return Ok(PathBuf::from(xdg).join("intelnav").join("libllama"));
    }
    if let Some(home) = std::env::var_os("HOME") {
        return Ok(PathBuf::from(home).join(".cache").join("intelnav").join("libllama"));
    }
    Err(anyhow!("no $XDG_CACHE_HOME or $HOME — can't locate the libllama cache"))
}

fn walk<F: FnMut(&Path)>(dir: &Path, f: &mut F) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, f);
        } else {
            f(&p);
        }
    }
}

// ------------------------- singleton helper --------------------------

use std::sync::OnceLock;

static DEFAULT_LOADER: OnceLock<Arc<Loader>> = OnceLock::new();

/// Lazy-initialize a process-wide `Loader` using [`find_libllama`] and
/// return an `Arc` to it. Subsequent calls return the cached handle;
/// a library path change requires process restart.
///
/// Errors: bubble up from `find_libllama` or `Loader::open` the first
/// time; on subsequent calls the cached result — including its error
/// — is returned directly.
pub fn default_loader() -> Result<Arc<Loader>> {
    // Double-init guard: if the first call errors we don't cache, so
    // a later call can retry after e.g. the user fixes
    // INTELNAV_LIBLLAMA_PATH. This matches the "best-effort" spirit
    // of doctor-driven recovery.
    if let Some(l) = DEFAULT_LOADER.get() {
        return Ok(l.clone());
    }
    let path = find_libllama()?;
    let loader = Loader::open(&path)?;
    let _ = DEFAULT_LOADER.set(loader.clone());
    Ok(loader)
}
