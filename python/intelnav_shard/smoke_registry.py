"""End-to-end registry smoke test.

Runs entirely against a live `intelnav-registry` binary over HTTP; does not
spawn llama-server. Proves the full lifecycle:

  1. A cloud peer claims every part → each part has live=1, counted=0
     (cloud doesn't count toward the volunteer quorum).
  2. A volunteer assigns → the registry steers it to the most under-served
     part, then it claims.
  3. Once the volunteer's joined_at ages past `min_live_seconds`, the
     cloud peer's next heartbeat receives a `standby` directive for that
     part (hysteresis: live_v ≥ desired_k).
  4. Volunteer releases → cloud's next heartbeat receives `resume`.
  5. Stop heartbeating the cloud peer → it gets evicted after
     `heartbeat_interval_s * heartbeat_miss_tolerance`.

Run manually:
    python smoke_registry.py --registry http://127.0.0.1:8787 --model-cid bafy-smoke-01
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from typing import Optional

import httpx

from intelnav_shard import crypto
from intelnav_shard.registry_client import RegistryClient, RegistryClientConfig


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def expect(cond: bool, msg: str) -> None:
    if not cond:
        print(f"FAIL: {msg}", file=sys.stderr)
        sys.exit(1)
    print(f"  ok: {msg}")


async def _snapshot(base: str, cid: str) -> dict:
    async with httpx.AsyncClient(timeout=5.0) as c:
        r = await c.get(f"{base.rstrip('/')}/v1/shards/{cid}")
        r.raise_for_status()
        return r.json()


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="http://127.0.0.1:8787")
    ap.add_argument("--model-cid", default="bafy-smoke-01")
    args = ap.parse_args()

    # identities
    cloud_id = crypto.Identity.generate()
    vol_id   = crypto.Identity.generate()

    log(f"cloud peer_id={cloud_id.peer_id_hex()[:16]}…")
    log(f"vol   peer_id={vol_id.peer_id_hex()[:16]}…")

    # clients
    def mk(identity: crypto.Identity, role: str) -> RegistryClient:
        return RegistryClient(RegistryClientConfig(
            base_url=  args.registry,
            model_cid= args.model_cid,
            identity=  identity,
            role=      role,
        ))
    cloud = mk(cloud_id, "cloud")
    vol   = mk(vol_id,   "volunteer")

    try:
        # 1. cloud claims every part directly (bypass /assign so it gets all 3)
        log("step 1: cloud claims p1, p2, p3")
        for part in ("p1", "p2", "p3"):
            await cloud.claim(part)
        snap = await _snapshot(args.registry, args.model_cid)
        parts = {p["id"]: p for p in snap["parts"]}
        for pid in ("p1", "p2", "p3"):
            expect(parts[pid]["live_total"] == 1,    f"{pid} live_total == 1")
            expect(parts[pid]["counted_volunteers"] == 0, f"{pid} counted_volunteers == 0 (cloud doesn't count)")

        # 2. volunteer assigns (registry picks a part) then claims it
        log("step 2: volunteer assigns + claims")
        assignment = await vol.assign()
        log(f"  volunteer assigned → {assignment.part_id}")
        await vol.claim(assignment.part_id)

        # 3. wait for min_live_seconds then heartbeat cloud — expect standby
        log("step 3: wait ≥ min_live_seconds, heartbeat cloud → expect 'standby'")
        await asyncio.sleep(2)  # manifest min_live_seconds = 1
        # heartbeat cloud on the part the volunteer claimed
        directive = await cloud.heartbeat(assignment.part_id)
        expect(directive == "standby",
               f"cloud heartbeat on {assignment.part_id} returned 'standby' (got {directive!r})")

        snap = await _snapshot(args.registry, args.model_cid)
        part = next(p for p in snap["parts"] if p["id"] == assignment.part_id)
        expect(part["counted_volunteers"] >= 1,
               f"{assignment.part_id} counted_volunteers ≥ 1 after age-up")

        # Volunteer-over-cloud ordering in the snapshot's peers list —
        # registry itself doesn't sort, but the gateway does. Here we just
        # sanity-check that both roles are present.
        roles = {p["role"] for p in part["peers"]}
        expect(roles == {"volunteer", "cloud"},
               f"{assignment.part_id} has both volunteer and cloud seeders")

        # 4. volunteer releases → cloud next heartbeat receives 'resume'
        log("step 4: volunteer releases → cloud heartbeat expects 'resume'")
        await vol.release(assignment.part_id)
        directive = await cloud.heartbeat(assignment.part_id)
        expect(directive == "resume",
               f"cloud heartbeat after volunteer release returned 'resume' (got {directive!r})")

        # 5. stop heartbeating cloud → expect eviction after 2*3 = 6s
        log("step 5: stop heartbeats, expect cloud eviction within ~10s")
        # pick an untouched part to watch (p1/p2/p3 minus the volunteer's one);
        # here we just check every part.
        deadline = time.monotonic() + 12.0
        evicted = False
        while time.monotonic() < deadline:
            snap = await _snapshot(args.registry, args.model_cid)
            any_live = any(
                any(pr["status"] == "live" and pr["role"] == "cloud"
                    for pr in p["peers"])
                for p in snap["parts"]
            )
            if not any_live:
                evicted = True
                break
            await asyncio.sleep(1.0)
        expect(evicted, "cloud evicted after missed heartbeats")

        log("ALL CHECKS PASSED")
        return 0

    finally:
        await cloud.close()
        await vol.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
