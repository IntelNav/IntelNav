"""Model weight resolver — turn a registry `weight_url` or user-supplied
reference into a local GGUF file.

Supported `weight_url` schemes:

  * ``file://…``   — a local absolute path (used in smoke tests and dev).
  * ``http://…`` / ``https://…`` — stream-download with sha256 verification.

Resolved files cache under ``~/.cache/intelnav/weights/<sha256>.gguf`` so
repeated (re-)claims don't re-download.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from urllib.parse import urlparse

import httpx

log = logging.getLogger(__name__)

CACHE_ROOT = Path.home() / ".cache" / "intelnav" / "weights"


def _cache_path(sha256_hex: str) -> Path:
    return CACHE_ROOT / f"{sha256_hex}.gguf"


def resolve(weight_url: str, sha256_hex: str) -> Path:
    """Return a local path to the weight file, downloading if necessary.

    Raises on sha256 mismatch after download.
    """
    cache = _cache_path(sha256_hex)
    if cache.exists() and _sha256_file(cache) == sha256_hex:
        log.info("weight cache hit: %s", cache)
        return cache

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(weight_url)

    if parsed.scheme == "file" or parsed.scheme == "":
        src = Path(parsed.path if parsed.scheme == "file" else weight_url)
        if not src.exists():
            raise FileNotFoundError(f"weight_url points to missing file: {src}")
        _copy_and_verify(src, cache, sha256_hex)
        return cache

    if parsed.scheme in ("http", "https"):
        _download_and_verify(weight_url, cache, sha256_hex)
        return cache

    raise ValueError(f"unsupported weight_url scheme: {weight_url!r}")


# ---------------------------------------------------------------------------

def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_and_verify(src: Path, dst: Path, expected: str) -> None:
    log.info("copying weights %s → %s", src, dst)
    tmp = dst.with_suffix(".partial")
    shutil.copyfile(src, tmp)
    got = _sha256_file(tmp)
    if got != expected:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"sha256 mismatch for {src}: got {got}, expected {expected}")
    tmp.replace(dst)


def _download_and_verify(url: str, dst: Path, expected: str) -> None:
    log.info("downloading weights %s → %s", url, dst)
    tmp = dst.with_suffix(".partial")
    h = hashlib.sha256()
    with httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                h.update(chunk)
    got = h.hexdigest()
    if got != expected:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"sha256 mismatch for {url}: got {got}, expected {expected}")
    tmp.replace(dst)
