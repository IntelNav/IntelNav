"""Thin HTTP client for `intelnav-registry` (spec §3).

Drives the volunteer/cloud lifecycle:

    assign → claim → heartbeat (loop) → release (on shutdown)

The registry authenticates every mutating call with an Ed25519 signature
over `CBOR(envelope)`. The envelope binds `(peer_id, model_cid, part_id,
timestamp, nonce)` so replays and cross-part reuse both fail. Directives
(`standby`/`resume`) arrive inside the heartbeat response — the shard
just mirrors them locally.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import cbor2
import httpx

from . import crypto

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  envelope
# ---------------------------------------------------------------------------


def _build_signed_envelope(
    identity: crypto.Identity,
    model_cid: str,
    part_id: str,
) -> dict[str, Any]:
    env = {
        "peer_id":   identity.peer_id_hex(),
        "model_cid": model_cid,
        "part_id":   part_id,
        "timestamp": int(time.time()),
        "nonce":     os.urandom(16).hex(),
    }
    # The Rust side CBOR-encodes the envelope struct *as-is*. cbor2 emits
    # a text-keyed map which matches ciborium's default serialization of a
    # named struct.
    payload = cbor2.dumps(env)
    sig = identity.sign(payload)
    return {"envelope": env, "sig": sig.hex()}


# ---------------------------------------------------------------------------
#  assignment result + config
# ---------------------------------------------------------------------------


@dataclass
class Assignment:
    part_id:           str
    layer_range:       tuple[int, int]
    weight_url:        str
    sha256:            str
    size_bytes:        int
    reservation_ttl_s: int


@dataclass
class RegistryClientConfig:
    base_url:      str                  # e.g. http://127.0.0.1:8787
    model_cid:     str                  # path segment in /v1/shards/<cid>
    identity:      crypto.Identity
    role:          str = "volunteer"    # "volunteer" | "cloud"
    vram_bytes:    int = 0              # 0 → no VRAM filter
    http_timeout_s: float = 10.0


# ---------------------------------------------------------------------------
#  client
# ---------------------------------------------------------------------------


class RegistryClient:
    def __init__(self, cfg: RegistryClientConfig):
        self.cfg = cfg
        self._client = httpx.AsyncClient(
            base_url=cfg.base_url.rstrip("/"),
            timeout=cfg.http_timeout_s,
        )

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    #  endpoints
    # ------------------------------------------------------------------

    async def assign(self) -> Assignment:
        r = await self._client.post(
            f"/v1/shards/{self.cfg.model_cid}/assign",
            json={
                "peer_id":    self.cfg.identity.peer_id_hex(),
                "role":       self.cfg.role,
                "vram_bytes": self.cfg.vram_bytes,
            },
        )
        r.raise_for_status()
        data = r.json()
        lr = data["layer_range"]
        return Assignment(
            part_id=           data["part_id"],
            layer_range=       (int(lr[0]), int(lr[1])),
            weight_url=        data["weight_url"],
            sha256=            data["sha256"],
            size_bytes=        int(data["size_bytes"]),
            reservation_ttl_s= int(data["reservation_ttl_s"]),
        )

    async def claim(self, part_id: str) -> None:
        body = _build_signed_envelope(self.cfg.identity, self.cfg.model_cid, part_id)
        body["role"] = self.cfg.role
        r = await self._client.post(
            f"/v1/shards/{self.cfg.model_cid}/{part_id}/claim",
            json=body,
        )
        r.raise_for_status()

    async def heartbeat(self, part_id: str) -> Optional[str]:
        body = _build_signed_envelope(self.cfg.identity, self.cfg.model_cid, part_id)
        r = await self._client.post(
            f"/v1/shards/{self.cfg.model_cid}/{part_id}/heartbeat",
            json=body,
        )
        r.raise_for_status()
        return r.json().get("directive")

    async def release(self, part_id: str) -> None:
        body = _build_signed_envelope(self.cfg.identity, self.cfg.model_cid, part_id)
        r = await self._client.post(
            f"/v1/shards/{self.cfg.model_cid}/{part_id}/release",
            json=body,
        )
        r.raise_for_status()

    async def snapshot(self) -> dict[str, Any]:
        """GET /v1/shards/<cid> — used mainly by smoke tests."""
        r = await self._client.get(f"/v1/shards/{self.cfg.model_cid}")
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
#  heartbeat loop
# ---------------------------------------------------------------------------


async def run_heartbeat_loop(
    client: RegistryClient,
    part_id: str,
    interval_s: int,
    on_directive: Callable[[str], None],
    stop: asyncio.Event,
) -> None:
    """Send heartbeats until `stop` is set. Forwards any directive to
    `on_directive` on the shard side (typically pauses/resumes serving)."""
    while not stop.is_set():
        try:
            directive = await client.heartbeat(part_id)
            if directive:
                log.info("registry directive: %s", directive)
                on_directive(directive)
        except httpx.HTTPStatusError as e:
            log.warning("heartbeat HTTP %s: %s", e.response.status_code, e.response.text[:200])
            if e.response.status_code == 410:
                # evicted — re-claim on next cycle
                try:
                    await client.claim(part_id)
                    log.info("re-claimed after eviction")
                except Exception as ce:
                    log.error("re-claim failed: %s", ce)
        except Exception as e:
            log.warning("heartbeat failed: %s", e)
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass
