#!/usr/bin/env bash
#
# intelnav-cli-pack.sh <os-arch>
#
# Package the release binaries for `<os-arch>` into a tarball the
# installer + doctor know how to consume:
#
#   dist/intelnav-<os-arch>-<short_sha>.tar.gz
#     └─ intelnav-<os-arch>-<short_sha>/
#        ├─ bin/
#        │   ├─ intelnav (or intelnav.exe)
#        │   ├─ intelnav-chunk
#        │   ├─ pipe_peer
#        │   ├─ pipe_driver
#        │   └─ bench_chain
#        ├─ LICENSE
#        ├─ README.md
#        └─ INTELNAV_VERSION
#
# Expected cwd: the intelnav workspace root.
#
# Paired with `IntelNav/llama.cpp@intelnav-pack.sh` — together they
# produce the full offline-install bundle that #31's installer
# script stitches together.

set -euo pipefail

OS_ARCH="${1:?usage: intelnav-cli-pack.sh <os-arch>}"

if [[ ! -d target/release ]]; then
    echo "intelnav-cli-pack: target/release missing — did cargo build run?" >&2
    exit 1
fi

SHA="$(git rev-parse HEAD)"
SHORT_SHA="${SHA:0:12}"
DIST="dist"
PKG_NAME="intelnav-${OS_ARCH}-${SHORT_SHA}"
PKG_ROOT="${DIST}/${PKG_NAME}"

rm -rf "${DIST}"
mkdir -p "${PKG_ROOT}/bin"

# Windows emits `.exe`; Linux / macOS don't. Pick up either.
#
# Binaries split across `target/release/` (bins) and
# `target/release/examples/` (examples). Keep that layout abstract
# from the user by collapsing everything under `bin/`.
copy_one() {
    local src_dir="$1"
    local name="$2"
    for ext in "" ".exe"; do
        local src="${src_dir}/${name}${ext}"
        if [[ -f "$src" ]]; then
            cp -a "$src" "${PKG_ROOT}/bin/"
            return 0
        fi
    done
    echo "intelnav-cli-pack: missing binary ${name} (looked in ${src_dir})" >&2
    return 1
}

copy_one target/release          intelnav
copy_one target/release          intelnav-chunk
copy_one target/release/examples pipe_peer
copy_one target/release/examples pipe_driver
copy_one target/release/examples bench_chain

# Strip debug symbols on Unixes — binaries ship ~2× smaller, and a
# user hitting a crash posts a core dump + the release SHA rather
# than inline debug info. Windows binaries aren't stripped; the MSVC
# toolchain writes symbols to a separate .pdb (which we skip).
case "$OS_ARCH" in
    linux-*)  strip --strip-unneeded "${PKG_ROOT}/bin/"* 2>/dev/null || true ;;
    macos-*)  strip -u -r            "${PKG_ROOT}/bin/"* 2>/dev/null || true ;;
esac

cp LICENSE "${PKG_ROOT}/" 2>/dev/null || echo "no LICENSE file, skipping" >&2
echo "${SHA}" > "${PKG_ROOT}/INTELNAV_VERSION"

cat > "${PKG_ROOT}/README.md" <<EOF
# intelnav — prebuilt binaries

Built from commit \`${SHA}\` of
\`https://github.com/IntelNav/IntelNav\`.

Platform: **${OS_ARCH}**

## What's here

* \`bin/intelnav\` — the main CLI. Run \`intelnav doctor\` first.
* \`bin/intelnav-chunk\` — Path B model chunker / fetcher / serve.
* \`bin/pipe_peer\` — join a swarm as a peer serving a layer range.
* \`bin/pipe_driver\` — drive a pipeline manually (dev/debug).
* \`bin/bench_chain\` — local performance harness.

## Pairing with libllama

The Rust binaries dlopen a separate \`libllama.so\` (or \`.dylib\` /
\`.dll\`) built from the IntelNav-patched llama.cpp fork. Download the
matching tarball from the companion release:

  https://github.com/IntelNav/llama.cpp/releases

Pick the backend that matches your hardware — \`cpu\` is universal;
\`rocm\`, \`cuda\`, \`metal\`, \`vulkan\` are faster if you have the GPU.
Unpack it somewhere stable, then tell \`intelnav\` where to find it:

    export INTELNAV_LIBLLAMA_DIR=\$HOME/.cache/intelnav/libllama/bin
    intelnav doctor

\`intelnav doctor\` will verify that libllama loads, backends are
discoverable, and (optionally) that your GPU is usable.

This step goes away once \`intelnav install\` lands (task #31) — the
installer picks the right libllama and sets the env vars for you.
EOF

# ---------- tarball ----------
tar -czf "${DIST}/${PKG_NAME}.tar.gz" -C "${DIST}" "${PKG_NAME}"
rm -rf "${PKG_ROOT}"

ls -la "${DIST}"
echo "intelnav-cli-pack: wrote ${DIST}/${PKG_NAME}.tar.gz"
