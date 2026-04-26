"""Command-line entry point: `python -m intelnav_shard` or `intelnav-shard`."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

from . import backend as backend_mod
from . import crypto
from .registry_client import RegistryClient, RegistryClientConfig, run_heartbeat_loop
from .model_resolver import resolve as resolve_weights
from .server import ServerConfig, ShardServer
from .shard import ShardConfig

log = logging.getLogger("intelnav_shard")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="intelnav-shard",
        description="IntelNav contributor shard server (paper §12.1).",
    )
    p.add_argument("--model", type=Path, default=None,
                   help="Path to a GGUF model file. Optional when --registry is set: "
                        "in that case the shard fetches the assigned shard via the "
                        "registry's weight_url.")
    p.add_argument("--model-name", default=None,
                   help="Advertised model id (defaults to file stem).")
    p.add_argument("--socket", type=Path, default=Path("/tmp/intelnav_shard.sock"),
                   help="Unix-domain socket path to bind.")
    p.add_argument("--identity", type=Path, default=None,
                   help="Hex-seed file for the peer's Ed25519 identity. "
                        "Generated ephemerally if omitted.")
    p.add_argument("--n-ctx", type=int, default=2048)
    p.add_argument("--n-threads", type=int, default=None)
    p.add_argument("--gpu-layers", type=int, default=None,
                   help="Layers to offload to GPU (default: all for non-CPU backends).")
    p.add_argument("--backend", default=None,
                   help="Force a backend: cuda | rocm | metal | vulkan | sycl | cpu.")
    p.add_argument("--max-tokens", type=int, default=512)

    # registry integration
    p.add_argument("--registry", default=None,
                   help="Registry base URL, e.g. http://127.0.0.1:8787. When set, "
                        "the shard runs assign→claim→heartbeat→release instead of "
                        "serving a user-picked GGUF file directly.")
    p.add_argument("--registry-model", default=None,
                   help="Model CID path segment, required with --registry.")
    p.add_argument("--role", choices=("volunteer", "cloud"), default="volunteer",
                   help="Peer role advertised to the registry.")
    p.add_argument("--vram-bytes", type=int, default=0,
                   help="Advertised VRAM; 0 disables the VRAM filter.")
    p.add_argument("--heartbeat-s", type=int, default=20,
                   help="Heartbeat interval (must be <= registry miss tolerance).")

    p.add_argument("-v", "--verbose", action="count", default=0)
    return p


def _load_identity(path: Optional[Path]) -> crypto.Identity:
    if path is None:
        return crypto.Identity.generate()
    seed_hex = path.read_text().strip()
    return crypto.Identity.from_seed(bytes.fromhex(seed_hex))


async def _registry_bootstrap(
    args: argparse.Namespace,
    identity: crypto.Identity,
) -> tuple[Path, RegistryClient, str]:
    """assign → download weights → claim. Returns (weights_path, client, part_id)."""
    if not args.registry_model:
        raise SystemExit("--registry requires --registry-model")

    client = RegistryClient(RegistryClientConfig(
        base_url=  args.registry,
        model_cid= args.registry_model,
        identity=  identity,
        role=      args.role,
        vram_bytes=args.vram_bytes,
    ))
    log.info("registry: assign (role=%s, vram=%d)", args.role, args.vram_bytes)
    assignment = await client.assign()
    log.info("assigned part=%s layers=%s size=%d",
             assignment.part_id, assignment.layer_range, assignment.size_bytes)

    weights = resolve_weights(assignment.weight_url, assignment.sha256)
    log.info("weights ready at %s", weights)

    await client.claim(assignment.part_id)
    log.info("registry: claim ok (part=%s)", assignment.part_id)
    return weights, client, assignment.part_id


async def _run(args: argparse.Namespace) -> int:
    identity = _load_identity(args.identity)
    log.info("peer_id=%s", identity.peer_id_hex())

    probe = (backend_mod.force(args.backend)
             if args.backend else backend_mod.detect())
    log.info(probe.summary())

    registry_client: Optional[RegistryClient] = None
    part_id: Optional[str] = None
    model_path: Optional[Path] = args.model

    if args.registry:
        model_path, registry_client, part_id = await _registry_bootstrap(args, identity)

    if model_path is None:
        raise SystemExit("either --model or --registry must be set")

    shard_cfg = ShardConfig(
        model_path=model_path,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        gpu_layers=args.gpu_layers,
    )
    srv_cfg = ServerConfig(
        socket_path=args.socket,
        identity=identity,
        shard_cfg=shard_cfg,
        probe=probe,
        model_advertised_name=args.model_name or (args.registry_model or model_path.stem),
        max_tokens_per_req=args.max_tokens,
    )

    server = ShardServer(srv_cfg)

    # heartbeat loop (registry mode only)
    hb_task: Optional[asyncio.Task] = None
    hb_stop = asyncio.Event()
    serving_paused = False

    def on_directive(d: str) -> None:
        nonlocal serving_paused
        if d == "standby":
            serving_paused = True
            log.info("serving paused (standby)")
        elif d == "resume":
            serving_paused = False
            log.info("serving resumed")

    if registry_client is not None and part_id is not None:
        hb_task = asyncio.create_task(run_heartbeat_loop(
            registry_client, part_id, args.heartbeat_s, on_directive, hb_stop,
        ))

    try:
        await server.run()
    finally:
        if hb_task is not None:
            hb_stop.set()
            try:
                await asyncio.wait_for(hb_task, timeout=3.0)
            except asyncio.TimeoutError:
                hb_task.cancel()
        if registry_client is not None and part_id is not None:
            try:
                await registry_client.release(part_id)
                log.info("registry: released part=%s", part_id)
            except Exception as e:
                log.warning("release failed: %s", e)
            await registry_client.close()

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    level = {0: logging.INFO, 1: logging.DEBUG}.get(args.verbose, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
