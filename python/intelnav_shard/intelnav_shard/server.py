"""Async Unix-socket server speaking the IntelNav CBOR protocol.

State machine per connection (paper §3.3 step 3+):

    client                               shard (this process)
    ──────                               ────────────────────
      │ Hello(peer_id, proto_ver, q*)   │
      │ ──────────────────────────────▶ │
      │                                 │ Hello(peer_id, proto_ver, q*)
      │ ◀────────────────────────────── │
      │                                 │
      │ SessionInit(sid, x25519_c, …)   │
      │ ──────────────────────────────▶ │
      │                                 │ SessionAck(sid, x25519_s)
      │ ◀────────────────────────────── │
      │                                 │
      │ Prompt(sid, ct, nonce)          │
      │ ──────────────────────────────▶ │
      │                                 │ Token(sid, seq=0, sampled=…) *
      │ ◀────────────────────────────── │
      │                                 │ Token(sid, seq=N, sampled=EOS)
      │ ◀────────────────────────────── │

If anything goes wrong at any point we send `AbortSession(sid, reason)` and
close the connection. There's no retry — the gateway re-opens a new session.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import crypto, wire
from .backend import BackendProbe
from .shard import Shard, ShardConfig

log = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    socket_path: Path
    identity: crypto.Identity
    shard_cfg: ShardConfig
    probe: BackendProbe
    model_advertised_name: str         # what we claim in capabilities
    max_tokens_per_req: int = 512


class ShardServer:
    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        self.shard = Shard(cfg.shard_cfg, cfg.probe)
        self._server: Optional[asyncio.AbstractServer] = None
        self._stop = asyncio.Event()

    # ------------------------------------------------------------------

    async def run(self) -> None:
        # shard.load() is sync (subprocess spawn + HTTP health poll) — run
        # it off the event loop so a slow first-time download doesn't
        # block asyncio setup.
        await asyncio.get_running_loop().run_in_executor(None, self.shard.load)
        sock = self.cfg.socket_path
        if sock.exists():
            sock.unlink()
        sock.parent.mkdir(parents=True, exist_ok=True)

        self._server = await asyncio.start_unix_server(
            self._handle_connection, path=str(sock),
        )
        log.info("intelnav-shard listening on %s", sock)

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._stop.set)

        try:
            async with self._server:
                await self._stop.wait()
        finally:
            await self.shard.close()
            if sock.exists():
                try:
                    sock.unlink()
                except OSError:
                    pass
            log.info("intelnav-shard stopped")

    # ------------------------------------------------------------------
    #  one client
    # ------------------------------------------------------------------

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername") or "<unix>"
        log.info("connection opened (%s)", peer)
        session_key: Optional[bytes] = None
        session_id: Optional[str] = None

        try:
            # ---- Hello exchange ----
            msg = await wire.read_frame(reader)
            if not isinstance(msg, wire.Hello):
                await self._abort(writer, "", f"expected Hello, got {type(msg).__name__}")
                return
            if msg.proto_ver != wire.PROTO_VER:
                await self._abort(writer, "", f"proto version mismatch: {msg.proto_ver} != {wire.PROTO_VER}")
                return
            log.info("hello from %s… (quants=%s)",
                     msg.peer_id[:12], [q.value for q in msg.supported_quants])

            await self._send(writer, wire.Hello(
                peer_id=self.cfg.identity.peer_id_hex(),
                proto_ver=wire.PROTO_VER,
                supported_quants=list(msg.supported_quants),   # echo as baseline
            ))

            # ---- SessionInit → SessionAck ----
            msg = await wire.read_frame(reader)
            if not isinstance(msg, wire.SessionInit):
                await self._abort(writer, "", f"expected SessionInit, got {type(msg).__name__}")
                return
            session_id = msg.session_id

            handshake = crypto.EphemeralHandshake.generate()
            shared = handshake.derive_shared(msg.client_x25519_pub)
            session_key = crypto.session_key(shared)

            log.info("session %s… init (model_cid=%s, layers=%d..%d)",
                     session_id[:12], msg.model_cid, msg.layer_range.start, msg.layer_range.end)

            await self._send(writer, wire.SessionAck(
                session_id=session_id,
                shard_x25519_pub=handshake.public,
            ))

            # ---- Prompt → Token* ----
            msg = await wire.read_frame(reader)
            if not isinstance(msg, wire.Prompt):
                await self._abort(writer, session_id, f"expected Prompt, got {type(msg).__name__}")
                return

            try:
                plaintext = crypto.decrypt(session_key, msg.ciphertext, msg.nonce)
            except Exception as e:
                await self._abort(writer, session_id, f"decrypt failed: {e}")
                return

            prompt_text = plaintext.decode("utf-8", errors="replace")
            log.info("session %s… prompt (%d bytes plaintext)",
                     session_id[:12], len(plaintext))

            seq = 0
            async for tok_id, _piece in self.shard.stream(
                prompt_text,
                max_tokens=self.cfg.max_tokens_per_req,
            ):
                await self._send(writer, wire.Token(
                    session_id=session_id,
                    seq=seq,
                    logits_top_k=None,
                    sampled=tok_id,
                ))
                seq += 1

            # EOS sentinel: seq = last, sampled = 0xFFFF_FFFF (convention for
            # "stream done" until we wire a proper end-of-stream marker into
            # the spec).
            await self._send(writer, wire.Token(
                session_id=session_id,
                seq=seq,
                logits_top_k=None,
                sampled=0xFFFFFFFF,
            ))
            log.info("session %s… streamed %d tokens", session_id[:12], seq)

        except asyncio.IncompleteReadError:
            log.info("client disconnected mid-frame")
        except Exception as e:
            log.exception("connection error")
            try:
                await self._abort(writer, session_id or "", f"internal: {e}")
            except Exception:
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    # ------------------------------------------------------------------

    async def _send(self, writer: asyncio.StreamWriter, msg: wire.Msg) -> None:
        writer.write(wire.encode_frame(msg))
        await writer.drain()

    async def _abort(
        self,
        writer: asyncio.StreamWriter,
        session_id: str,
        reason: str,
    ) -> None:
        log.warning("abort(session=%s): %s", session_id[:12] if session_id else "<none>", reason)
        # 32 zero hex chars so the message always encodes, even pre-session.
        sid = session_id or ("0" * 64)
        try:
            await self._send(writer, wire.AbortSession(session_id=sid, reason=reason))
        except Exception:
            pass
