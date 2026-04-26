# Contributing to IntelNav

---

## First-time setup

```bash
git clone git@github.com:IntelNav/IntelNav.git
cd IntelNav
bash scripts/provision.sh     # system deps + rust + workspace check
```

Supports Debian/Ubuntu, Fedora, Arch, macOS. For other platforms, see
[`docs/QUICKSTART.md`](docs/QUICKSTART.md) §0 for the manual package
list.

**MSRV: Rust 1.88.** The repo's `rust-toolchain.toml` pins `channel = "stable"`,
so rustup users pick up a compatible version automatically.

---

## Dev loop

```bash
# fast check before every commit
cargo check --workspace --all-targets

# full test suite
cargo test --workspace --no-fail-fast

# format + lint
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
```

`just check` / `just test` / `just fmt` / `just lint` are aliases.

---

## Workspace rules

- **`#![forbid(unsafe_code)]`** in every crate except `intelnav-cli`,
  which drops to `libc::dup2` for the TUI's stderr redirect (marked
  `#[allow(unsafe_code)]` inline). Don't introduce `unsafe` elsewhere
  without explicit justification.
- **`core` has no heavy deps.** Shape-only types. Keep it that way.
- **`runtime` is the only crate that depends on `candle-*`.** The wire,
  gateway, and registry stay inference-backend agnostic.
- **Protocol messages are additive.** If you add a field to a `Msg`
  variant, use `#[serde(default, skip_serializing_if = "Option::is_none")]`
  so proto-v1 peers still decode.
- **Layer-split must stay bit-identical.** Any change to Qwen2 fork
  paths must be verified with the `layer_split_matches_full` test
  (max_abs_diff == 0 on q4_k_m).

---

## Logging

- Crates use `tracing` — never `println!`/`eprintln!` for non-CLI output.
- The CLI routes logs to `$XDG_STATE_HOME/intelnav/intelnav.log` when
  the TUI is active (otherwise stderr). Don't bypass this — native
  deps that write raw FD 2 will paint over the Ratatui canvas.
- Log level: `-v` = debug, `-vv` = trace.

---

## Commits / PRs

- Create a new commit per logical change. Don't amend merged commits.
- Never push `--force` to `main`. Never skip hooks (`--no-verify`).
- Don't commit build artifacts (`target/`, `*.aux`, `*.log`).
- `Cargo.lock` **is** committed (binary workspace).
- Append one line per session to [`docs/dev/PROGRESS.md`](docs/dev/PROGRESS.md):
  what moved, what's next. Newest first.

---

## Protocol + spec changes

- The wire protocol is normative. Changes to `Msg` must update
  [`specs/protocol-v1.md`](specs/protocol-v1.md).
- The registry is normative. Changes to routes / envelope / hysteresis
  must update [`specs/shard-registry-v1.md`](specs/shard-registry-v1.md).
- The threat model is normative. Security-relevant changes must
  update [`specs/security-v1.md`](specs/security-v1.md).

Paper/Petals references are treated as references, not contracts.
Deviations are fine when they produce cleaner UX or cross-device
reliability; log them under *Deviations from the paper* in PROGRESS.

---

## Adding a crate

1. `cargo new --lib crates/<name>`.
2. Add to `members = [...]` in the root `Cargo.toml`.
3. Add a workspace-level path dep (`intelnav-<name> = { path = "crates/<name>" }`).
4. Write a one-paragraph `crates/<name>/README.md` following the
   pattern in the existing crates.
5. Add `#![forbid(unsafe_code)]` unless you have a reason not to.
