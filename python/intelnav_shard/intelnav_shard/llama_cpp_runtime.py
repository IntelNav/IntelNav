"""Automatic llama.cpp runtime bootstrapper.

The goal is zero manual steps for a contributor. On the first start we:

1. Read the backend probe (``backend.py``) to know what the host can drive.
2. Query GitHub for the latest ``llama.cpp`` release tag.
3. Pick the asset name that matches ``(os, arch, backend)``. If the exact
   backend has no published asset for this OS, fall back through a short
   compatibility chain ending at CPU (which is always published).
4. Download + extract to ``~/.cache/intelnav/llama.cpp/<tag>/<backend>/``.
5. Return the path to the ``llama-server`` executable inside.

Subsequent starts hit the cache and bring up `llama-server` in a few ms.

Linux note: the upstream release page ships ``ubuntu-rocm-*`` and
``ubuntu-vulkan-*`` but **not** an explicit CUDA binary (they use Docker for
that). On NVIDIA + Linux we silently fall through to the Vulkan build, which
drives NVIDIA GPUs perfectly well via their Vulkan driver at ~80–90 % of
CUDA throughput. Same trick covers Intel Arc if SYCL isn't published.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import stat
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .backend import GpuBackend

log = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
GITHUB_RELEASES = "https://github.com/ggml-org/llama.cpp/releases/download"
USER_AGENT = "intelnav-shard/0.1 (+https://github.com/intelnav)"


# ---------------------------------------------------------------------------
#  asset matching
# ---------------------------------------------------------------------------


def _host_triple() -> tuple[str, str]:
    system = sys.platform
    if system.startswith("linux"):
        os_tag = "linux"
    elif system == "darwin":
        os_tag = "macos"
    elif system.startswith("win") or os.name == "nt":
        os_tag = "windows"
    else:
        os_tag = system

    mach = platform.machine().lower()
    if mach in {"x86_64", "amd64"}:
        arch = "x64"
    elif mach in {"aarch64", "arm64"}:
        arch = "arm64"
    else:
        arch = mach
    return os_tag, arch


# (os, arch, backend) → ordered list of *suffix patterns* to search for.
# Patterns are matched case-insensitively against the full asset name.
# Each entry is a regex fragment; anchoring is handled by `_match_asset`.
# First match wins — earlier entries are preferred.
_MATRIX: dict[tuple[str, str, GpuBackend], list[str]] = {
    # ---------------- Linux x64 ----------------
    ("linux", "x64", GpuBackend.CUDA):
        [r"ubuntu-cuda-[\d\.]+-x64", r"ubuntu-vulkan-x64"],        # fallback → Vulkan
    ("linux", "x64", GpuBackend.ROCM):
        [r"ubuntu-rocm-[\d\.]+-x64", r"ubuntu-hip-[\w-]*x64", r"ubuntu-vulkan-x64"],
    ("linux", "x64", GpuBackend.VULKAN):
        [r"ubuntu-vulkan-x64"],
    ("linux", "x64", GpuBackend.SYCL):
        [r"ubuntu-sycl-x64", r"ubuntu-openvino-[\w\.]+-x64", r"ubuntu-vulkan-x64"],
    ("linux", "x64", GpuBackend.CPU):
        [r"ubuntu-x64"],
    # ---------------- Linux arm64 ----------------
    ("linux", "arm64", GpuBackend.VULKAN):
        [r"ubuntu-vulkan-arm64"],
    ("linux", "arm64", GpuBackend.CPU):
        [r"ubuntu-arm64"],
    # ---------------- macOS ----------------
    ("macos", "arm64", GpuBackend.METAL):
        [r"macos-arm64(?!-kleidiai)"],
    ("macos", "arm64", GpuBackend.CPU):
        [r"macos-arm64(?!-kleidiai)"],     # same build — Metal is always on
    ("macos", "x64", GpuBackend.CPU):
        [r"macos-x64"],
    # ---------------- Windows x64 ----------------
    ("windows", "x64", GpuBackend.CUDA):
        [r"win-cuda-[\d\.]+-x64", r"win-vulkan-x64"],
    ("windows", "x64", GpuBackend.ROCM):
        [r"win-hip-[\w-]*x64", r"win-vulkan-x64"],
    ("windows", "x64", GpuBackend.VULKAN):
        [r"win-vulkan-x64"],
    ("windows", "x64", GpuBackend.SYCL):
        [r"win-sycl-x64", r"win-vulkan-x64"],
    ("windows", "x64", GpuBackend.CPU):
        [r"win-cpu-x64"],
}


def _match_asset(assets: list[dict], patterns: list[str]) -> Optional[dict]:
    """Return the first asset whose name matches any of `patterns`.

    Patterns are fullmatched against the asset name stripped of its
    `llama-<tag>-bin-` prefix and its archive extension.
    """
    for p in patterns:
        rx = re.compile(p, re.IGNORECASE)
        for a in assets:
            name = a["name"]
            core = re.sub(r"^llama-[\w\.]+-bin-", "", name)
            core = re.sub(r"\.(tar\.gz|tgz|zip)$", "", core, flags=re.IGNORECASE)
            if rx.fullmatch(core):
                return a
    return None


# ---------------------------------------------------------------------------
#  public API
# ---------------------------------------------------------------------------


@dataclass
class Runtime:
    tag: str                       # e.g. "b8851"
    backend: GpuBackend            # backend actually selected (may differ from requested if fallback)
    asset_name: str                # filename we downloaded
    root: Path                     # extracted root
    server_bin: Path               # absolute path to llama-server

    def summary(self) -> str:
        return f"llama.cpp {self.tag} ({self.backend.value}) → {self.server_bin}"


class RuntimeError_(RuntimeError):
    pass


def ensure(
    backend: GpuBackend,
    *,
    cache_dir: Optional[Path] = None,
    tag_override: Optional[str] = None,
) -> Runtime:
    """Return a ready-to-use Runtime, downloading if the cache is cold.

    Raises ``RuntimeError_`` on any unrecoverable failure — the caller
    should surface these to the user with enough context to fix (e.g. "no
    internet and nothing cached").
    """
    cache_dir = cache_dir or Path.home() / ".cache" / "intelnav" / "llama.cpp"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Fast path: cache already populated for this backend.
    hit = _find_cached(cache_dir, backend)
    if hit is not None:
        return hit

    os_tag, arch = _host_triple()
    tag = tag_override or _latest_tag()
    release = _release_metadata(tag)
    assets = release["assets"]

    patterns = _MATRIX.get((os_tag, arch, backend))
    if not patterns:
        raise RuntimeError_(
            f"no llama.cpp asset mapping for {os_tag}/{arch}/{backend.value}; "
            f"try forcing --backend cpu or vulkan."
        )

    asset = _match_asset(assets, patterns)
    if asset is None:
        # Last resort: CPU always exists.
        cpu_patterns = _MATRIX.get((os_tag, arch, GpuBackend.CPU), [])
        asset = _match_asset(assets, cpu_patterns)
        if asset is None:
            raise RuntimeError_(
                f"no matching asset for {os_tag}/{arch}/{backend.value} in release {tag}"
            )
        log.warning("no %s asset published for %s/%s — falling back to CPU",
                    backend.value, os_tag, arch)
        backend = GpuBackend.CPU

    target_dir = cache_dir / tag / backend.value
    target_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / asset["name"]

    if not archive.exists():
        _download(asset["browser_download_url"], archive, size_hint=asset.get("size"))
    _extract(archive, target_dir)

    server_bin = _find_server_bin(target_dir)
    if server_bin is None:
        raise RuntimeError_(
            f"llama-server binary not found in extracted archive at {target_dir}"
        )
    _ensure_executable(server_bin)

    return Runtime(
        tag=tag,
        backend=backend,
        asset_name=asset["name"],
        root=target_dir,
        server_bin=server_bin,
    )


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------


def _find_cached(cache_dir: Path, backend: GpuBackend) -> Optional[Runtime]:
    """Look for the newest cached tag that already has a server bin for
    this backend, so we don't hit the network every start-up."""
    if not cache_dir.exists():
        return None
    candidates = []
    for tag_dir in cache_dir.iterdir():
        if not tag_dir.is_dir() or not tag_dir.name.startswith("b"):
            continue
        b_dir = tag_dir / backend.value
        if not b_dir.exists():
            continue
        bin_path = _find_server_bin(b_dir)
        if bin_path is not None:
            candidates.append((tag_dir.name, b_dir, bin_path))
    if not candidates:
        return None
    # "b8851" > "b8123" lexicographically — good enough for release tags.
    candidates.sort(key=lambda x: x[0], reverse=True)
    tag, root, bin_path = candidates[0]
    return Runtime(
        tag=tag, backend=backend, asset_name="(cached)",
        root=root, server_bin=bin_path,
    )


def _latest_tag() -> str:
    meta = _gh_get_json(GITHUB_API)
    return meta["tag_name"]


def _release_metadata(tag: str) -> dict:
    url = f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}"
    return _gh_get_json(url)


def _gh_get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT,
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError_(f"GitHub API request failed: {url} → {e}") from e


def _download(url: str, dst: Path, size_hint: Optional[int] = None) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    log.info("downloading %s (%s MB)", dst.name,
             "?" if not size_hint else f"{size_hint/1e6:.1f}")

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        with open(tmp, "wb") as f:
            total = 0
            chunk = 1 << 16
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                total += len(buf)
                if size_hint and total and total % (chunk * 64) == 0:
                    pct = 100.0 * total / size_hint
                    sys.stderr.write(f"  {dst.name}: {pct:5.1f}% ({total/1e6:6.1f} / {size_hint/1e6:.1f} MB)\r")
                    sys.stderr.flush()
    sys.stderr.write("\n")
    tmp.rename(dst)


def _extract(archive: Path, into: Path) -> None:
    """Extract archive into `into/`, stripping a single top-level dir if
    present (so the binary lands at `into/bin/llama-server` regardless of
    how the zipfile was packed)."""
    if into.exists() and any(into.iterdir()):
        return
    log.info("extracting %s → %s", archive.name, into)
    tmp = into.with_suffix(".unpack")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    name_lower = archive.name.lower()
    if name_lower.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(tmp, filter="data")
    elif name_lower.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp)
    else:
        raise RuntimeError_(f"unknown archive type: {archive.name}")

    entries = list(tmp.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        src = entries[0]
    else:
        src = tmp
    into.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        shutil.move(str(p), str(into / p.name))
    shutil.rmtree(tmp, ignore_errors=True)


def _find_server_bin(root: Path) -> Optional[Path]:
    for name in ("llama-server", "llama-server.exe"):
        for p in root.rglob(name):
            if p.is_file():
                return p
    return None


def _ensure_executable(p: Path) -> None:
    if os.name == "nt":
        return
    mode = p.stat().st_mode
    p.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
