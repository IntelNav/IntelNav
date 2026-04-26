"""CBOR codecs mirroring `crates/wire/src/lib.rs::Msg`.

Serialization rules — kept in lockstep with the Rust serde attributes:

* `Msg` is internally tagged with `kind`; variant tags use `snake_case`
  (e.g. `hello`, `session_init`, `forward_hidden`, `abort_session`).
* `Phase`, `Dtype` serialize as lowercase strings.
* `Quant` uses the exact Rust-side rename strings: `Q4_K_M`, `Q5_K_M`,
  `Q8_0`, `fp16`, `bf16`.
* `PeerId`, `SessionId` are `#[serde(transparent)]` with `hex::serde` —
  they travel as *hex text strings* (64 chars), NOT raw CBOR bytes.
* `[u8; 32]` fields (x25519 pubkeys, nonces) and `Vec<u8>` fields
  (ciphertext, payload, sig, kv_delta) are default serde `Vec<u8>` →
  **CBOR array of unsigned ints**, not CBOR byte strings. ciborium does
  not enable `serde_bytes` by default.

The framing layer prepends a big-endian `u32` byte-count; total frame is
capped at 16 MiB.
"""

from __future__ import annotations

import enum
import io
import struct
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import cbor2

MAX_FRAME_BYTES = 16 * 1024 * 1024
PROTO_VER = 1


# ---------------------------------------------------------------------------
#  scalar enums
# ---------------------------------------------------------------------------


class Phase(str, enum.Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class Dtype(str, enum.Enum):
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"


class Quant(str, enum.Enum):
    Q4_K_M = "Q4_K_M"
    Q5_K_M = "Q5_K_M"
    Q8_0 = "Q8_0"
    FP16 = "fp16"
    BF16 = "bf16"


class Backend(str, enum.Enum):
    LLAMA_CPP = "llama-cpp"
    VLLM = "vllm"
    MLX_LM = "mlx-lm"
    OLLAMA = "ollama"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
#  value types
# ---------------------------------------------------------------------------


@dataclass
class LayerRange:
    start: int
    end: int

    def to_wire(self) -> dict[str, int]:
        return {"start": self.start, "end": self.end}

    @classmethod
    def from_wire(cls, obj: dict[str, int]) -> "LayerRange":
        return cls(start=int(obj["start"]), end=int(obj["end"]))


@dataclass
class ShardRoute:
    cid: str
    start: int
    end: int

    def to_wire(self) -> dict[str, Any]:
        return {"cid": self.cid, "start": self.start, "end": self.end}

    @classmethod
    def from_wire(cls, obj: dict[str, Any]) -> "ShardRoute":
        return cls(cid=str(obj["cid"]), start=int(obj["start"]), end=int(obj["end"]))


@dataclass
class CapabilityV1:
    peer_id: str                   # 64-char hex (PeerId transparent+hex::serde)
    backend: Backend
    quants: list[Quant]
    vram_bytes: int
    ram_bytes: int
    tok_per_sec: float
    max_seq: int
    models: list[str]
    layers: list[ShardRoute] = field(default_factory=list)

    def to_wire(self) -> dict[str, Any]:
        return {
            "peer_id":     self.peer_id,
            "backend":     self.backend.value,
            "quants":      [q.value for q in self.quants],
            "vram_bytes":  self.vram_bytes,
            "ram_bytes":   self.ram_bytes,
            "tok_per_sec": float(self.tok_per_sec),
            "max_seq":     self.max_seq,
            "models":      list(self.models),
            "layers":      [l.to_wire() for l in self.layers],
        }


# ---------------------------------------------------------------------------
#  Msg — discriminated union
# ---------------------------------------------------------------------------


@dataclass
class Hello:
    peer_id: str                       # 64-char hex
    proto_ver: int
    supported_quants: list[Quant]


@dataclass
class SessionInit:
    session_id: str                    # 64-char hex
    client_x25519_pub: bytes           # 32 bytes (encoded on wire as array[u8;32])
    model_cid: str
    layer_range: LayerRange
    max_seq: int


@dataclass
class SessionAck:
    session_id: str
    shard_x25519_pub: bytes            # 32 bytes


@dataclass
class Prompt:
    session_id: str
    ciphertext: bytes
    nonce: bytes                       # 12 bytes


@dataclass
class ForwardHidden:
    session_id: str
    seq: int
    phase: Phase
    dtype: Dtype
    shape: tuple[int, int, int]
    payload: bytes
    kv_delta: Optional[bytes] = None


@dataclass
class Token:
    session_id: str
    seq: int
    logits_top_k: Optional[list[tuple[int, float]]]
    sampled: int


@dataclass
class Heartbeat:
    session_id: str
    last_seq: int
    health: int                        # 0..=100


@dataclass
class AbortSession:
    session_id: str
    reason: str


@dataclass
class Advertise:
    capability_v1: CapabilityV1
    sig: bytes


@dataclass
class Gossip:
    topic: str
    payload: bytes
    sig: bytes


Msg = Union[
    Hello, SessionInit, SessionAck, Prompt, ForwardHidden, Token,
    Heartbeat, AbortSession, Advertise, Gossip,
]


# ---------------------------------------------------------------------------
#  encode / decode
# ---------------------------------------------------------------------------


def _bytes_to_u8_array(b: bytes) -> list[int]:
    # serde default for Vec<u8>/[u8; N] via ciborium → CBOR array of u8
    return list(b)


def _u8_array_to_bytes(arr: Any) -> bytes:
    if isinstance(arr, (bytes, bytearray, memoryview)):
        return bytes(arr)                  # tolerant: accept both encodings
    return bytes(int(x) & 0xFF for x in arr)


def _encode_msg_to_map(m: Msg) -> dict[str, Any]:
    if isinstance(m, Hello):
        return {
            "kind": "hello",
            "peer_id": m.peer_id,
            "proto_ver": m.proto_ver,
            "supported_quants": [q.value for q in m.supported_quants],
        }
    if isinstance(m, SessionInit):
        return {
            "kind": "session_init",
            "session_id": m.session_id,
            "client_x25519_pub": _bytes_to_u8_array(m.client_x25519_pub),
            "model_cid": m.model_cid,
            "layer_range": m.layer_range.to_wire(),
            "max_seq": m.max_seq,
        }
    if isinstance(m, SessionAck):
        return {
            "kind": "session_ack",
            "session_id": m.session_id,
            "shard_x25519_pub": _bytes_to_u8_array(m.shard_x25519_pub),
        }
    if isinstance(m, Prompt):
        return {
            "kind": "prompt",
            "session_id": m.session_id,
            "ciphertext": _bytes_to_u8_array(m.ciphertext),
            "nonce": _bytes_to_u8_array(m.nonce),
        }
    if isinstance(m, ForwardHidden):
        return {
            "kind": "forward_hidden",
            "session_id": m.session_id,
            "seq": m.seq,
            "phase": m.phase.value,
            "dtype": m.dtype.value,
            "shape": list(m.shape),
            "payload": _bytes_to_u8_array(m.payload),
            "kv_delta": None if m.kv_delta is None else _bytes_to_u8_array(m.kv_delta),
        }
    if isinstance(m, Token):
        return {
            "kind": "token",
            "session_id": m.session_id,
            "seq": m.seq,
            "logits_top_k": None if m.logits_top_k is None else [
                [int(i), float(p)] for (i, p) in m.logits_top_k
            ],
            "sampled": m.sampled,
        }
    if isinstance(m, Heartbeat):
        return {
            "kind": "heartbeat",
            "session_id": m.session_id,
            "last_seq": m.last_seq,
            "health": m.health,
        }
    if isinstance(m, AbortSession):
        return {
            "kind": "abort_session",
            "session_id": m.session_id,
            "reason": m.reason,
        }
    if isinstance(m, Advertise):
        return {
            "kind": "advertise",
            "capability_v1": m.capability_v1.to_wire(),
            "sig": _bytes_to_u8_array(m.sig),
        }
    if isinstance(m, Gossip):
        return {
            "kind": "gossip",
            "topic": m.topic,
            "payload": _bytes_to_u8_array(m.payload),
            "sig": _bytes_to_u8_array(m.sig),
        }
    raise TypeError(f"unknown Msg type: {type(m).__name__}")


def _decode_map_to_msg(obj: dict[str, Any]) -> Msg:
    kind = obj.get("kind")
    if kind == "hello":
        return Hello(
            peer_id=str(obj["peer_id"]),
            proto_ver=int(obj["proto_ver"]),
            supported_quants=[Quant(q) for q in obj["supported_quants"]],
        )
    if kind == "session_init":
        return SessionInit(
            session_id=str(obj["session_id"]),
            client_x25519_pub=_u8_array_to_bytes(obj["client_x25519_pub"]),
            model_cid=str(obj["model_cid"]),
            layer_range=LayerRange.from_wire(obj["layer_range"]),
            max_seq=int(obj["max_seq"]),
        )
    if kind == "session_ack":
        return SessionAck(
            session_id=str(obj["session_id"]),
            shard_x25519_pub=_u8_array_to_bytes(obj["shard_x25519_pub"]),
        )
    if kind == "prompt":
        return Prompt(
            session_id=str(obj["session_id"]),
            ciphertext=_u8_array_to_bytes(obj["ciphertext"]),
            nonce=_u8_array_to_bytes(obj["nonce"]),
        )
    if kind == "forward_hidden":
        shape = tuple(int(x) for x in obj["shape"])
        if len(shape) != 3:
            raise ValueError(f"shape must be length 3, got {shape}")
        return ForwardHidden(
            session_id=str(obj["session_id"]),
            seq=int(obj["seq"]),
            phase=Phase(obj["phase"]),
            dtype=Dtype(obj["dtype"]),
            shape=shape,  # type: ignore[arg-type]
            payload=_u8_array_to_bytes(obj["payload"]),
            kv_delta=None if obj.get("kv_delta") is None else _u8_array_to_bytes(obj["kv_delta"]),
        )
    if kind == "token":
        raw_topk = obj.get("logits_top_k")
        topk = None if raw_topk is None else [(int(i), float(p)) for (i, p) in raw_topk]
        return Token(
            session_id=str(obj["session_id"]),
            seq=int(obj["seq"]),
            logits_top_k=topk,
            sampled=int(obj["sampled"]),
        )
    if kind == "heartbeat":
        return Heartbeat(
            session_id=str(obj["session_id"]),
            last_seq=int(obj["last_seq"]),
            health=int(obj["health"]),
        )
    if kind == "abort_session":
        return AbortSession(
            session_id=str(obj["session_id"]),
            reason=str(obj["reason"]),
        )
    raise ValueError(f"unknown or unsupported Msg kind: {kind!r}")


def encode(msg: Msg) -> bytes:
    """Encode a Msg to CBOR bytes (no length prefix)."""
    return cbor2.dumps(_encode_msg_to_map(msg))


def decode(buf: bytes) -> Msg:
    """Decode a Msg from CBOR bytes."""
    return _decode_map_to_msg(cbor2.loads(buf))


def encode_frame(msg: Msg) -> bytes:
    """Length-prefix + CBOR-encode a Msg. Raises if > MAX_FRAME_BYTES."""
    payload = encode(msg)
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError(f"frame too large: {len(payload)} > {MAX_FRAME_BYTES}")
    return struct.pack(">I", len(payload)) + payload


async def read_frame(reader) -> Optional[Msg]:
    """Read one framed Msg from an asyncio StreamReader. None on EOF."""
    header = await reader.readexactly(4) if not reader.at_eof() else b""
    if len(header) < 4:
        return None
    (length,) = struct.unpack(">I", header)
    if length > MAX_FRAME_BYTES:
        raise ValueError(f"incoming frame too large: {length}")
    payload = await reader.readexactly(length)
    return decode(payload)
