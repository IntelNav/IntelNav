"""End-to-end CBOR protocol smoke test.

Drives a locally-running `intelnav-shard` through the full state machine:

    Hello → Hello, SessionInit → SessionAck, Prompt → Token*

Verifies the shard decrypts AES-GCM correctly (key derivation must match
the Rust `intelnav_crypto::session_key`) and streams token ids back until
EOS / sentinel.

Usage:  python -m intelnav_shard.smoke_client --socket /tmp/intelnav_shard.sock
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from intelnav_shard import crypto, wire


EOS_SENTINEL = 0xFFFFFFFF


async def run(socket_path: Path, prompt: str, max_tokens: int) -> int:
    reader, writer = await asyncio.open_unix_connection(path=str(socket_path))

    # 1. Hello
    client_id = crypto.Identity.generate()
    await _send(writer, wire.Hello(
        peer_id=client_id.peer_id_hex(),
        proto_ver=wire.PROTO_VER,
        supported_quants=[wire.Quant.Q4_K_M, wire.Quant.Q8_0],
    ))
    srv_hello = await wire.read_frame(reader)
    assert isinstance(srv_hello, wire.Hello), f"expected Hello, got {type(srv_hello).__name__}"
    print(f"[hello] shard peer_id={srv_hello.peer_id[:12]}…")

    # 2. SessionInit → SessionAck
    hs = crypto.EphemeralHandshake.generate()
    session_id_hex = os.urandom(32).hex()
    await _send(writer, wire.SessionInit(
        session_id=session_id_hex,
        client_x25519_pub=hs.public,
        model_cid="smoke-test",
        layer_range=wire.LayerRange(start=0, end=0),
        max_seq=max_tokens,
    ))
    ack = await wire.read_frame(reader)
    assert isinstance(ack, wire.SessionAck), f"expected SessionAck, got {type(ack).__name__}"
    assert ack.session_id == session_id_hex, "session id mismatch"

    shared = hs.derive_shared(ack.shard_x25519_pub)
    key = crypto.session_key(shared)
    print(f"[session] established, key prefix={key[:4].hex()}")

    # 3. Prompt → Token*
    ct, nonce = crypto.encrypt(key, prompt.encode("utf-8"))
    await _send(writer, wire.Prompt(
        session_id=session_id_hex, ciphertext=ct, nonce=nonce,
    ))
    print(f"[prompt] sent {len(prompt)} chars plaintext, {len(ct)} bytes ciphertext")

    got = 0
    while True:
        msg = await wire.read_frame(reader)
        if msg is None:
            print("[stream] connection closed without sentinel")
            break
        if isinstance(msg, wire.AbortSession):
            print(f"[stream] server aborted: {msg.reason}")
            return 1
        assert isinstance(msg, wire.Token), f"expected Token, got {type(msg).__name__}"
        if msg.sampled == EOS_SENTINEL:
            print(f"[stream] done — {got} tokens streamed (seq={msg.seq})")
            break
        got += 1
        print(f"  tok[{msg.seq}] id={msg.sampled}")
        if got > max_tokens + 4:
            print("[stream] too many tokens — bailing")
            return 2

    writer.close()
    await writer.wait_closed()
    return 0


async def _send(writer: asyncio.StreamWriter, msg: wire.Msg) -> None:
    writer.write(wire.encode_frame(msg))
    await writer.drain()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--socket", type=Path, default=Path("/tmp/intelnav_shard.sock"))
    p.add_argument("--prompt", default="Say hello in one short sentence.")
    p.add_argument("--max-tokens", type=int, default=24)
    args = p.parse_args()
    return asyncio.run(run(args.socket, args.prompt, args.max_tokens))


if __name__ == "__main__":
    sys.exit(main())
