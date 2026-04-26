#!/usr/bin/env bash
#
# provision.sh — one-shot contributor bootstrap.
#
# Detects the host distro, installs the system packages IntelNav needs,
# installs rustup if no Rust toolchain is present, then runs a workspace
# check. Idempotent: re-running is safe. Supports Debian/Ubuntu, Fedora,
# Arch, and macOS (Homebrew).
#
# Usage:
#     bash scripts/provision.sh              # interactive (sudo prompts)
#     bash scripts/provision.sh --yes        # auto-accept package installs
#     bash scripts/provision.sh --skip-check # install deps, don't build
#
set -euo pipefail

AUTO_YES=0
RUN_CHECK=1
for arg in "$@"; do
  case "$arg" in
    -y|--yes)         AUTO_YES=1 ;;
    --skip-check)     RUN_CHECK=0 ;;
    -h|--help)
      sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown flag: $arg"; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
#  pretty output
# ---------------------------------------------------------------------------
log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarn:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
#  distro detection
# ---------------------------------------------------------------------------
detect_os() {
  case "$(uname -s)" in
    Darwin) echo "macos"; return ;;
    Linux)
      if [[ -r /etc/os-release ]]; then
        . /etc/os-release
        case "${ID:-}${ID_LIKE:-}" in
          *ubuntu*|*debian*) echo "debian"; return ;;
          *fedora*|*rhel*|*centos*) echo "fedora"; return ;;
          *arch*|*manjaro*) echo "arch"; return ;;
        esac
      fi
      ;;
  esac
  die "unsupported OS; please install build-essential + pkg-config + openssl-dev manually"
}

OS="$(detect_os)"
log "detected: $OS"

# ---------------------------------------------------------------------------
#  sudo wrapper — transparent no-op when already root
# ---------------------------------------------------------------------------
maybe_sudo() {
  if [[ $EUID -eq 0 ]] || [[ "$OS" == "macos" ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

# ---------------------------------------------------------------------------
#  system packages
# ---------------------------------------------------------------------------
install_packages() {
  case "$OS" in
    debian)
      local pkgs=(build-essential pkg-config libssl-dev curl ca-certificates git)
      log "apt: installing ${pkgs[*]}"
      maybe_sudo apt-get update -qq
      if [[ $AUTO_YES -eq 1 ]]; then
        maybe_sudo apt-get install -y "${pkgs[@]}"
      else
        maybe_sudo apt-get install "${pkgs[@]}"
      fi
      ;;
    fedora)
      log "dnf: installing @development-tools, pkgconf, openssl-devel, curl, git"
      maybe_sudo dnf install ${AUTO_YES:+-y} -q \
        @development-tools pkgconf openssl-devel curl ca-certificates git
      ;;
    arch)
      log "pacman: installing base-devel, pkgconf, openssl, curl, git"
      maybe_sudo pacman -S --needed ${AUTO_YES:+--noconfirm} \
        base-devel pkgconf openssl curl git
      ;;
    macos)
      command -v brew >/dev/null 2>&1 || \
        die "Homebrew not found; install from https://brew.sh then rerun"
      log "brew: installing pkg-config, openssl, git"
      brew install pkg-config openssl git
      ;;
  esac
}

# ---------------------------------------------------------------------------
#  rust toolchain (via rustup)
# ---------------------------------------------------------------------------
install_rust() {
  if command -v cargo >/dev/null 2>&1 && command -v rustc >/dev/null 2>&1; then
    log "rust present: $(rustc --version)"
    # Honor the repo's rust-toolchain.toml — rustup will auto-install on
    # the next `cargo` invocation in the workspace.
    return
  fi

  log "installing rustup (non-interactive, default profile)"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain stable --profile default

  # shellcheck disable=SC1091
  [[ -r "$HOME/.cargo/env" ]] && . "$HOME/.cargo/env"
  log "rust installed: $(rustc --version)"
}

# ---------------------------------------------------------------------------
#  workspace check (catches toolchain / dep issues early)
# ---------------------------------------------------------------------------
run_workspace_check() {
  [[ $RUN_CHECK -eq 0 ]] && { log "--skip-check set; skipping cargo check"; return; }

  # shellcheck disable=SC1091
  [[ -r "$HOME/.cargo/env" ]] && . "$HOME/.cargo/env"

  local root; root="$(cd "$(dirname "$0")/.." && pwd)"
  log "cargo check --workspace (in $root) — first run downloads the dep tree, takes a few minutes"
  ( cd "$root" && cargo check --workspace --all-targets )
  log "workspace check passed"
}

# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------
install_packages
install_rust
run_workspace_check

cat <<'DONE'

provision complete.

  next:
    cargo build --release -p intelnav-cli               # build the binary
    ./target/release/intelnav doctor                    # preflight
    ./target/release/intelnav chat                      # start the TUI

  docs:
    docs/QUICKSTART.md — commands that run today
    docs/ARCHITECTURE.md — crate graph + data flow
    docs/STATUS.md — what works, what doesn't
DONE
