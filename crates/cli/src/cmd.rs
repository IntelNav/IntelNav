//! Non-interactive subcommands: `ask`, `models`, `peers`, `health`, `doctor`,
//! `init`, `node`.

use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde_json::Value;

use intelnav_core::{Config, RunMode};
use intelnav_crypto::Identity;
use intelnav_runtime::{DevicePref, Probe, SamplingCfg};

use crate::banner;
use crate::chain_driver::{ChainDriver, ChainTarget, DraftTarget};
use crate::chat::{self, ChatMessage, ChatRequest, Delta};
use crate::local::{self, LocalDriver};

// ======================================================================
//  ask — stream a single prompt to stdout
// ======================================================================

pub async fn ask(cfg: &Config, mode: RunMode, model: Option<String>, prompt: &str) -> Result<()> {
    let messages = vec![ChatMessage { role: "user".into(), content: prompt.into() }];
    let mut rx = match mode {
        RunMode::Local => {
            let scan = local::list_models(&cfg.models_dir);
            let requested = model.clone().unwrap_or_else(|| {
                scan.iter().filter(|m| m.is_usable())
                    .min_by_key(|m| m.size_bytes)
                    .map(|m| m.name.clone())
                    .unwrap_or_else(|| cfg.default_model.clone())
            });
            let Some(m) = local::resolve(&scan, &requested) else {
                anyhow::bail!(
                    "no local model matches `{requested}`. Drop a .gguf into {}",
                    cfg.models_dir.display()
                );
            };
            if !m.is_usable() {
                anyhow::bail!("{}", m.status_line());
            }
            let device: DevicePref = cfg.device.parse().unwrap_or(DevicePref::Auto);
            let driver = LocalDriver::new(device);
            driver.stream(m, messages, SamplingCfg::default())
        }
        RunMode::Network if !cfg.peers.is_empty() => {
            let target = ChainTarget::from_config(&cfg.peers, &cfg.splits)?;
            let scan = local::list_models(&cfg.models_dir);
            let requested = model.clone().unwrap_or_else(|| cfg.default_model.clone());
            let Some(m) = local::resolve(&scan, &requested) else {
                anyhow::bail!(
                    "peer chain needs the GGUF locally for the front slice + head; \
                     no match for `{requested}` in {}", cfg.models_dir.display()
                );
            };
            if !m.is_usable() {
                anyhow::bail!("{}", m.status_line());
            }
            let device: DevicePref = cfg.device.parse().unwrap_or(DevicePref::Auto);
            let driver = ChainDriver::new(device);
            driver.set_target(Some(target));
            if let (Some(path), k) = (cfg.draft_model.clone(), cfg.spec_k) {
                if k >= 2 {
                    driver.set_draft(Some(DraftTarget { path, k: k as usize }));
                }
            }
            {
                let (dtype, _) = crate::chain_driver::parse_wire_dtype(&cfg.wire_dtype);
                driver.set_wire_dtype(dtype);
            }
            driver.stream(m, messages, SamplingCfg::default())
        }
        RunMode::Network | RunMode::Auto => chat::stream(ChatRequest {
            gateway:   cfg.gateway_url.clone(),
            model:     model.unwrap_or_else(|| cfg.default_model.clone()),
            messages,
            quorum:    Some(cfg.quorum),
            allow_wan: cfg.allow_wan,
        }),
    };
    let mut stdout = std::io::stdout();
    while let Some(delta) = rx.recv().await {
        match delta {
            Delta::Token(t) => {
                stdout.write_all(t.as_bytes())?;
                stdout.flush()?;
            }
            Delta::Done => {
                writeln!(stdout)?;
                break;
            }
            Delta::Error(e) => {
                writeln!(std::io::stderr(), "\nerror: {e}")?;
                std::process::exit(1);
            }
        }
    }
    Ok(())
}

// ======================================================================
//  models
// ======================================================================

pub async fn models(cfg: &Config, json: bool) -> Result<()> {
    let local_scan = local::list_models(&cfg.models_dir);

    // Try the gateway; tolerate failures so local listing still works.
    let url = format!("{}/v1/models", cfg.gateway_url.trim_end_matches('/'));
    let remote: Option<Value> = match reqwest::get(&url).await {
        Ok(r) if r.status().is_success() => r.json().await.ok(),
        _ => None,
    };

    if json {
        let out = serde_json::json!({
            "local":  local_scan.iter().map(|m| serde_json::json!({
                "name": m.name, "path": m.path, "size_bytes": m.size_bytes,
                "arch": format!("{:?}", m.arch), "usable": m.is_usable(),
            })).collect::<Vec<_>>(),
            "remote": remote,
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    // Local first.
    println!("── local ({}) ──", cfg.models_dir.display());
    if local_scan.is_empty() {
        println!("  (no .gguf files found)");
    } else {
        for m in &local_scan {
            let tag = if m.is_usable() { "·" } else { "!" };
            println!("  {tag} {}", m.status_line());
        }
    }
    println!();

    // Remote table.
    println!("── network ({}) ──", cfg.gateway_url);
    let Some(body) = remote else {
        println!("  gateway unreachable — start one with: intelnav gateway");
        return Ok(());
    };
    let empty: Vec<Value> = vec![];
    let arr = body.get("data").and_then(|v| v.as_array()).unwrap_or(&empty);
    if arr.is_empty() {
        println!("  no network models.");
        return Ok(());
    }
    println!("{:<40} {:>8}  {:<18}  providers", "MODEL", "TOK/S", "QUANTS");
    println!("{}", "─".repeat(90));
    for m in arr {
        let id = m.get("id").and_then(|v| v.as_str()).unwrap_or("?");
        let tps = m.get("best_tok_per_s").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let quants: String = m
            .get("quants")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(","))
            .unwrap_or_default();
        let provs: Vec<String> = m
            .get("providers")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
            .unwrap_or_default();
        let providers = if provs.is_empty() { "—".into() } else { provs.join(", ") };
        println!("{id:<40} {tps:>8.1}  {quants:<18}  {providers}");
    }
    Ok(())
}

// ======================================================================
//  peers
// ======================================================================

pub async fn peers(cfg: &Config, json: bool) -> Result<()> {
    let url = format!("{}/v1/network/peers", cfg.gateway_url.trim_end_matches('/'));
    let body: Value = reqwest::get(&url)
        .await
        .with_context(|| format!("GET {url}"))?
        .error_for_status()?
        .json()
        .await?;

    if json {
        println!("{}", serde_json::to_string_pretty(&body)?);
        return Ok(());
    }

    let empty: Vec<Value> = vec![];
    let listings = body.as_array().unwrap_or(&empty);
    if listings.is_empty() {
        println!("no peer directories registered.");
        return Ok(());
    }
    for listing in listings {
        let dir = listing.get("directory").and_then(|v| v.as_str()).unwrap_or("?");
        let peers = listing.get("peers").and_then(|v| v.as_array()).cloned().unwrap_or_default();
        println!("── {dir} ── ({} peers)", peers.len());
        for p in peers {
            let id  = p.get("peer_id").and_then(|v| v.as_str()).unwrap_or("?");
            let tps = p.get("tok_per_s").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let models: Vec<String> = p
                .get("models")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
                .unwrap_or_default();
            println!("  {id}  {tps:>6.1} tok/s  [{}]", models.join(", "));
        }
    }
    Ok(())
}

// ======================================================================
//  health
// ======================================================================

pub async fn health(cfg: &Config) -> Result<()> {
    let url = format!("{}/v1/network/health", cfg.gateway_url.trim_end_matches('/'));
    match reqwest::get(&url).await {
        Ok(resp) if resp.status().is_success() => {
            let v: Value = resp.json().await?;
            println!("gateway    : OK  ({})", cfg.gateway_url);
            println!("upstream   : {}", v.get("upstream").and_then(|v| v.as_str()).unwrap_or("?"));
            println!("uptime_sec : {}",  v.get("uptime_sec").and_then(|v| v.as_u64()).unwrap_or(0));
            println!("peers      : {}",  v.get("peer_count").and_then(|v| v.as_u64()).unwrap_or(0));
            if let Some(dirs) = v.get("directories").and_then(|v| v.as_array()) {
                let names: Vec<String> = dirs.iter().filter_map(|x| x.as_str().map(String::from)).collect();
                println!("directories: {}", names.join(", "));
            }
        }
        Ok(resp) => anyhow::bail!("gateway returned {}", resp.status()),
        Err(e)   => anyhow::bail!("gateway unreachable at {url}: {e}"),
    }
    Ok(())
}

// ======================================================================
//  doctor — preflight
// ======================================================================

pub async fn doctor(cfg: &Config) -> Result<()> {
    println!("{}", banner::BANNER);
    println!("IntelNav doctor — preflight checks");
    println!();

    // Collect actionable hints as we go so the end-of-report summary
    // can tell the user "here are the N things to fix, in order."
    // Purely user-facing; the checks themselves stay independent.
    let mut actions: Vec<String> = Vec::new();

    ok_or_warn("config path",
        match Config::config_path() {
            Some(p) => format!("{}", p.display()),
            None    => "<no XDG dir>".into(),
        },
        true);

    let id_path = identity_path();
    let (ident_status, ident_ok) = if id_path.exists() {
        (format!("loaded from {}", id_path.display()), true)
    } else {
        ("not yet generated — run `intelnav init`".into(), false)
    };
    ok_or_warn("peer identity", ident_status, ident_ok);
    if !ident_ok {
        actions.push("run `intelnav init` to generate a peer identity".into());
    }

    ok_or_warn("mode", cfg.mode.as_str().to_string(), true);

    // Local runtime probe.
    let probe = Probe::collect();
    ok_or_warn("runtime", probe.summary.clone(), true);
    println!("      available: {}", probe.backends.available.join(", "));
    println!("      preferred: {}", probe.backends.recommended);

    // GGML-path probe: which libllama-<backend>.so would load, which
    // runtime libs are missing, what GPU hardware is installed.
    let gg = intelnav_ggml::GgmlProbe::collect();
    ok_or_warn(
        "ggml backend",
        format!("preferred: {}", gg.preferred.join(" → ")),
        true,
    );
    for g in &gg.gpus {
        println!("      GPU: {} · {}", g.vendor, g.detail);
    }
    for b in &gg.backends {
        match &b.status {
            intelnav_ggml::BackendStatus::Available => {
                println!("      \x1b[32m✓\x1b[0m {:<7} libs available", b.tag);
            }
            intelnav_ggml::BackendStatus::Missing { reason, install_hint } => {
                println!("      \x1b[33m✗\x1b[0m {:<7} {}", b.tag, reason);
                if let Some(hint) = install_hint {
                    println!("          hint: {hint}");
                }
            }
            intelnav_ggml::BackendStatus::NotApplicable => {}
        }
    }

    // ---- Live libllama load test ------------------------------------
    //
    // The GgmlProbe above only inspects the filesystem ("what COULD
    // load"). This actually invokes `find_libllama()` + `Loader::open()`
    // + `backend_load_all()` — the same code path pipe_peer / the
    // gateway / every runtime entry point will take. If this fails,
    // nothing else in IntelNav runs. Surface the dlopen error with
    // the fix attached rather than letting the user discover it
    // on their first forward pass.
    println!();
    println!("  libllama load test");
    match intelnav_ggml::find_libllama() {
        Ok(p) => {
            println!("      \x1b[32m✓\x1b[0m found {}", p.display());
            match intelnav_ggml::Loader::open(&p) {
                Ok(loader) => {
                    println!("      \x1b[32m✓\x1b[0m dlopen succeeded (loaded {} companion lib(s))",
                        // Companion count isn't exposed publicly; cheap
                        // proxy: number of ggml-* files next to libllama.
                        count_companions(&p));
                    // Try the full backend_load_all path — this is
                    // what the #29 fix actually wires up.
                    match intelnav_ggml::backend_load_all() {
                        Ok(()) => {
                            println!("      \x1b[32m✓\x1b[0m ggml backends loaded from {}",
                                loader.loaded_from.parent().map(|p| p.display().to_string())
                                    .unwrap_or_else(|| "(unknown dir)".into()));
                        }
                        Err(e) => {
                            println!("      \x1b[31m✗\x1b[0m backend_load_all failed: {e:#}");
                            actions.push(
                                "no ggml backends available — check that `libggml-*.so` files \
                                 ship alongside libllama.so in the same directory".into()
                            );
                        }
                    }
                }
                Err(e) => {
                    println!("      \x1b[31m✗\x1b[0m dlopen failed: {e:#}");
                    actions.push(dlopen_hint(&format!("{e:#}")));
                }
            }
        }
        Err(e) => {
            println!("      \x1b[31m✗\x1b[0m libllama not found: {e}");
            actions.push(
                "download a libllama tarball from https://github.com/IntelNav/llama.cpp/releases \
                 and set INTELNAV_LIBLLAMA_DIR to its bin/ directory (or wait for \
                 `intelnav install` in #31 which does this for you)".into()
            );
        }
    }

    // Local models.
    let scan = local::list_models(&cfg.models_dir);
    let usable = scan.iter().filter(|m| m.is_usable()).count();
    ok_or_warn(
        "models_dir",
        format!("{} ({} usable / {} total)", cfg.models_dir.display(), usable, scan.len()),
        usable > 0 || scan.is_empty(),
    );
    for m in &scan {
        let tag = if m.is_usable() { "·" } else { "!" };
        println!("      {tag} {}", m.status_line());
    }

    // Gateway reachable?
    let gw = cfg.gateway_url.clone();
    let gw_ok = reqwest::get(format!("{}/v1/network/health", gw.trim_end_matches('/')))
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false);
    ok_or_warn("gateway", format!("{} {}", gw, if gw_ok { "reachable" } else { "unreachable" }), gw_ok);

    // Upstream reachable?
    let up = cfg.upstream_url.clone();
    let up_ok = reqwest::get(format!("{}/v1/models", up.trim_end_matches('/')))
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false);
    ok_or_warn("upstream", format!("{} {}", up, if up_ok { "reachable" } else { "unreachable" }), up_ok);

    ok_or_warn("default tier", format!("{:?}", cfg.default_tier), true);
    ok_or_warn("allow WAN",    format!("{}",   cfg.allow_wan),    true);

    if usable == 0 && !gw_ok {
        actions.push(format!(
            "drop a .gguf into {} or start a gateway with `intelnav gateway`",
            cfg.models_dir.display()
        ));
    } else if !gw_ok {
        actions.push("no gateway yet — start one with `intelnav gateway`".into());
    }
    if !up_ok {
        actions.push("point INTELNAV_UPSTREAM_URL at a running Ollama/LM Studio/vLLM".into());
    }

    println!();
    if actions.is_empty() {
        println!("\x1b[32mAll green.\x1b[0m You're ready to run.");
    } else {
        println!("\x1b[33m{} thing{} to fix, in order:\x1b[0m",
            actions.len(), if actions.len() == 1 { "" } else { "s" });
        for (i, a) in actions.iter().enumerate() {
            println!("  {}. {a}", i + 1);
        }
    }
    Ok(())
}

/// Cheap companion-count proxy — just list `libggml-*` files next to
/// libllama. Purely diagnostic; the real loading work is already done
/// inside `Loader::open`.
fn count_companions(libllama: &std::path::Path) -> usize {
    let Some(dir) = libllama.parent() else { return 0; };
    std::fs::read_dir(dir)
        .map(|it| it.flatten()
            .filter(|e| {
                let Some(n) = e.file_name().to_str().map(String::from) else { return false; };
                let body = n.strip_prefix("lib").unwrap_or(&n).to_string();
                (body.starts_with("ggml-") || body.starts_with("ggml."))
                    && (n.ends_with(".so") || n.ends_with(".dylib")
                        || n.ends_with(".dll") || n.contains(".so."))
            })
            .count())
        .unwrap_or(0)
}

/// Interpret a dlopen error message and suggest a concrete fix.
/// Pattern-matches the errors we've actually seen users hit rather
/// than returning a generic "something's broken".
fn dlopen_hint(err: &str) -> String {
    let e = err.to_lowercase();
    if e.contains("no such file or directory") {
        "the library is not at INTELNAV_LIBLLAMA_DIR or its SONAME dependencies aren't \
         next to it — unpack a libllama tarball with `tar xzf libllama-*.tar.gz \
         --strip-components=1 -C $HOME/.cache/intelnav/libllama` and point \
         INTELNAV_LIBLLAMA_DIR at the resulting bin/ subdir".into()
    } else if e.contains("version `glibc_") || e.contains("symbol version") {
        "your libllama tarball was built against a newer glibc than your system — \
         grab the matching tarball for your distro (linux-x64 tarballs target \
         ubuntu-22.04 / glibc 2.35) or rebuild from source".into()
    } else if e.contains("undefined symbol") {
        "libllama and its companion ggml libs are from different builds — make sure you \
         extracted ONE tarball cleanly, and there are no stale .so files in \
         INTELNAV_LIBLLAMA_DIR from an older version".into()
    } else {
        format!("dlopen failed ({err}); check INTELNAV_LIBLLAMA_DIR points at a bin/ \
                 directory containing libllama + its companion libggml*.so files")
    }
}

fn ok_or_warn(label: &str, value: impl Into<String>, ok: bool) {
    let tag = if ok { "\x1b[32m✓\x1b[0m" } else { "\x1b[33m!\x1b[0m" };
    println!("  {tag} {label:<14} {}", value.into());
}

// ======================================================================
//  init — generate config + identity
// ======================================================================

pub async fn init(force: bool) -> Result<()> {
    let Some(cfg_path) = Config::config_path() else {
        anyhow::bail!("could not resolve XDG config directory");
    };
    if let Some(parent) = cfg_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let default = Config::default();

    if cfg_path.exists() && !force {
        println!("config exists at {} (use --force to overwrite)", cfg_path.display());
    } else {
        let toml_str = toml::to_string_pretty(&default)?;
        std::fs::write(&cfg_path, toml_str)?;
        println!("wrote config → {}", cfg_path.display());
    }

    std::fs::create_dir_all(&default.models_dir)?;
    println!("models dir   → {}  (drop .gguf + tokenizer.json here)", default.models_dir.display());

    let id_path = identity_path();
    if id_path.exists() && !force {
        println!("identity exists at {} (use --force to overwrite)", id_path.display());
    } else {
        if let Some(p) = id_path.parent() { std::fs::create_dir_all(p)?; }
        let id = Identity::generate();
        std::fs::write(&id_path, hex::encode(id.seed()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&id_path)?.permissions();
            perms.set_mode(0o600);
            std::fs::set_permissions(&id_path, perms)?;
        }
        println!("wrote identity → {}  (peer id: {})", id_path.display(), id.peer_id());
    }
    Ok(())
}

fn identity_path() -> PathBuf {
    directories::ProjectDirs::from("io", "intelnav", "intelnav")
        .map(|p| p.data_dir().join("peer.key"))
        .unwrap_or_else(|| PathBuf::from("./peer.key"))
}

// ======================================================================
//  node — bridge to the Python shard server
// ======================================================================

pub async fn node(_cfg: &Config, shard: &str) -> Result<()> {
    println!("intelnav node — contributor bridge");
    println!();
    println!("The contributor shard server runs as a Python process that embeds");
    println!("llama.cpp. It communicates with the CLI/gateway over:");
    println!();
    println!("    {shard}");
    println!();
    println!("To start it (once implemented, paper §12.1):");
    println!();
    println!("    cd python/intelnav_shard");
    println!("    python -m intelnav_shard.shard_server --socket {shard}");
    println!();
    println!("This command will bridge the shard to the gateway in M3.");
    Ok(())
}

