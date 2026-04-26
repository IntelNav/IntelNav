"""Shard inference wrapper driving a bundled ``llama-server`` subprocess.

Design (refactor rationale): we don't embed llama.cpp via Python bindings
because prebuilt bindings are CPU-only on PyPI and asking a contributor to
build from source is an unacceptable UX. Instead we download an official
llama.cpp release binary for the host's best backend (see
``llama_cpp_runtime``) and spawn ``llama-server`` as a subprocess. The
shard talks HTTP to it on a loopback port; we forward tokens out to the
CBOR Unix socket.

Lifecycle:

    load()      → download runtime if needed, spawn subprocess, poll /health
    stream(…)   → POST /completion stream=true, yield (fake_id, piece)
    close()     → send SIGTERM, wait, fall back to SIGKILL on timeout

Token ids in streaming mode are synthesized from the piece bytes — llama
-server's SSE stream does not surface raw llama.cpp token ids per chunk,
and our wire protocol's ``sampled`` field isn't interpreted until M3
(quorum). The piece text is what the gateway eventually emits via SSE.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, Iterable, Optional

import httpx

from .backend import BackendProbe, GpuBackend, n_gpu_layers_for
from .llama_cpp_runtime import Runtime, ensure as ensure_runtime

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  config
# ---------------------------------------------------------------------------


@dataclass
class ShardConfig:
    model_path: Path
    n_ctx: int = 2048
    n_threads: Optional[int] = None
    gpu_layers: Optional[int] = None
    seed: int = 0xC0FFEE
    host: str = "127.0.0.1"
    port: Optional[int] = None            # None → pick an ephemeral port
    startup_timeout_s: float = 60.0       # how long to wait for /health
    tag_override: Optional[str] = None    # pin a specific llama.cpp release


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _synth_token_id(piece: str) -> int:
    # crc32 mapped away from the EOS sentinel (0xFFFFFFFF).
    if not piece:
        return 0
    h = zlib.crc32(piece.encode("utf-8", errors="replace")) & 0xFFFFFFFF
    return 0 if h == 0xFFFFFFFF else h


# ---------------------------------------------------------------------------
#  Shard
# ---------------------------------------------------------------------------


class Shard:
    """Owns one llama-server subprocess for the lifetime of the shard server."""

    def __init__(self, cfg: ShardConfig, probe: BackendProbe):
        self.cfg = cfg
        self.probe = probe
        self.runtime: Optional[Runtime] = None
        self._proc: Optional[subprocess.Popen] = None
        self._port: Optional[int] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._n_layers: int = 0

    # ------------------------------------------------------------------
    #  lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        if not self.cfg.model_path.exists():
            raise FileNotFoundError(f"model not found: {self.cfg.model_path}")

        self.runtime = ensure_runtime(
            self.probe.chosen, tag_override=self.cfg.tag_override,
        )
        log.info("runtime: %s", self.runtime.summary())

        self._port = self.cfg.port or _pick_free_port()
        n_gpu = n_gpu_layers_for(self.runtime.backend, self.cfg.gpu_layers)
        args = [
            str(self.runtime.server_bin),
            "--model", str(self.cfg.model_path),
            "--host", self.cfg.host,
            "--port", str(self._port),
            "--ctx-size", str(self.cfg.n_ctx),
            "--n-gpu-layers", str(n_gpu),
            "--threads", str(self.cfg.n_threads or os.cpu_count() or 4),
            "--log-disable",     # suppress llama-server's own colour logs
        ]
        log.info("spawning llama-server on :%d (n_gpu_layers=%d)", self._port, n_gpu)
        # Set LD_LIBRARY_PATH to the runtime root so the server can load its
        # bundled ggml / backend shared objects without a system install.
        env = os.environ.copy()
        lib_dirs = {str(self.runtime.root)}
        for sub in ("lib", "lib64", "bin"):
            d = self.runtime.root / sub
            if d.is_dir():
                lib_dirs.add(str(d))
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            list(lib_dirs) + ([existing] if existing else [])
        )

        self._proc = subprocess.Popen(
            args,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        self._await_ready()
        self._client = httpx.AsyncClient(
            base_url=f"http://{self.cfg.host}:{self._port}",
            timeout=httpx.Timeout(300.0, connect=5.0),
        )

    def _await_ready(self) -> None:
        assert self._proc is not None and self._port is not None
        deadline = time.monotonic() + self.cfg.startup_timeout_s
        url = f"http://{self.cfg.host}:{self._port}/health"
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                err = self._drain_stderr()
                raise RuntimeError(
                    f"llama-server exited during startup (rc={self._proc.returncode}). "
                    f"Last stderr:\n{err}"
                )
            try:
                with httpx.Client(timeout=1.0) as c:
                    r = c.get(url)
                    if r.status_code == 200:
                        # Health endpoint sometimes reports model still loading;
                        # wait for status == "ok".
                        status = r.json().get("status")
                        if status in (None, "ok"):
                            log.info("llama-server ready (%.2fs)",
                                     self.cfg.startup_timeout_s - (deadline - time.monotonic()))
                            return
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError):
                pass
            time.sleep(0.2)
        raise RuntimeError(
            f"llama-server did not become ready within {self.cfg.startup_timeout_s:.0f}s"
        )

    def _drain_stderr(self, limit: int = 4096) -> str:
        if self._proc is None or self._proc.stderr is None:
            return ""
        try:
            data = self._proc.stderr.read(limit)
            return (data or b"").decode("utf-8", errors="replace")
        except Exception:
            return ""

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._proc is not None:
            if self._proc.poll() is None:
                try:
                    self._proc.terminate()
                    try:
                        self._proc.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        log.warning("llama-server didn't terminate; sending SIGKILL")
                        self._proc.kill()
                        self._proc.wait(timeout=2.0)
                except Exception:
                    pass
            self._proc = None

    @property
    def n_layers(self) -> int:
        return self._n_layers      # populated post-load via /props; 0 until then

    # ------------------------------------------------------------------
    #  inference
    # ------------------------------------------------------------------

    async def stream(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: Optional[Iterable[str]] = None,
    ) -> AsyncGenerator[tuple[int, str], None]:
        """Stream `(synth_id, piece)` pairs from the running llama-server.

        This is an async generator — callers must `async for` over it."""
        assert self._client is not None, "shard.load() was not called"
        body = {
            "prompt":       prompt,
            "n_predict":    int(max_tokens),
            "temperature":  float(temperature),
            "top_p":        float(top_p),
            "stream":       True,
            "cache_prompt": True,
        }
        if stop:
            body["stop"] = list(stop)

        async with self._client.stream("POST", "/completion", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                # llama-server emits SSE-ish "data: {...}\n\n" framing.
                payload = line[6:] if line.startswith("data: ") else line
                if payload.strip() in ("", "[DONE]"):
                    continue
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    log.debug("non-json chunk: %r", payload[:120])
                    continue
                piece = chunk.get("content", "")
                if piece:
                    yield _synth_token_id(piece), piece
                if chunk.get("stop"):
                    return
