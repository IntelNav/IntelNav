"""Crypto primitives mirroring `crates/crypto/src/lib.rs`.

Exactly three things happen on the shard side:

1. Generate an ephemeral X25519 keypair and derive the shared secret with the
   client's advertised public key.
2. Turn the shared secret into a 32-byte AES key via
   ``blake3_xof("intelnav/v1/prompt" || shared)``. Domain separation string
   MUST match the Rust side or prompts will decrypt to garbage.
3. Decrypt the `Prompt.ciphertext` under AES-256-GCM with the 12-byte nonce.

We also expose minimal Ed25519 sign/verify for `Advertise` signatures.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import blake3
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey,
)
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey, X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import serialization as ser
from cryptography.exceptions import InvalidSignature

DOMAIN_TAG = b"intelnav/v1/prompt"


# ---------------------------------------------------------------------------
#  X25519 handshake
# ---------------------------------------------------------------------------


@dataclass
class EphemeralHandshake:
    """Fresh X25519 keypair, one shot.

    `secret` is consumed by ``derive_shared`` — never reuse.
    """
    _secret: X25519PrivateKey
    public: bytes                  # 32-byte raw pubkey (wire form)

    @classmethod
    def generate(cls) -> "EphemeralHandshake":
        sk = X25519PrivateKey.generate()
        pk = sk.public_key().public_bytes(
            encoding=ser.Encoding.Raw, format=ser.PublicFormat.Raw,
        )
        return cls(_secret=sk, public=pk)

    def derive_shared(self, peer_pub: bytes) -> bytes:
        if len(peer_pub) != 32:
            raise ValueError("peer X25519 pubkey must be 32 bytes")
        peer = X25519PublicKey.from_public_bytes(peer_pub)
        return self._secret.exchange(peer)


# ---------------------------------------------------------------------------
#  Key derivation (MUST match intelnav_crypto::session_key)
# ---------------------------------------------------------------------------


def session_key(shared: bytes) -> bytes:
    """blake3-XOF(domain_tag || shared) → 32-byte AES-256 key.

    Must stay byte-identical to `intelnav_crypto::session_key` or the Rust
    entry-shard and this Python shard cannot share a session.
    """
    if len(shared) != 32:
        raise ValueError("X25519 shared secret must be 32 bytes")
    h = blake3.blake3()
    h.update(DOMAIN_TAG)
    h.update(shared)
    return h.digest(length=32)


# ---------------------------------------------------------------------------
#  AES-256-GCM
# ---------------------------------------------------------------------------


def encrypt(key: bytes, plaintext: bytes) -> tuple[bytes, bytes]:
    if len(key) != 32:
        raise ValueError("AES-256 key must be 32 bytes")
    nonce = os.urandom(12)
    ct = AESGCM(key).encrypt(nonce, plaintext, associated_data=None)
    return ct, nonce


def decrypt(key: bytes, ciphertext: bytes, nonce: bytes) -> bytes:
    if len(key) != 32:
        raise ValueError("AES-256 key must be 32 bytes")
    if len(nonce) != 12:
        raise ValueError("GCM nonce must be 12 bytes")
    return AESGCM(key).decrypt(nonce, ciphertext, associated_data=None)


# ---------------------------------------------------------------------------
#  Ed25519 identity (for signing Advertise records)
# ---------------------------------------------------------------------------


@dataclass
class Identity:
    _signing: Ed25519PrivateKey
    public: bytes                  # 32-byte raw pubkey

    @classmethod
    def generate(cls) -> "Identity":
        sk = Ed25519PrivateKey.generate()
        pk = sk.public_key().public_bytes(
            encoding=ser.Encoding.Raw, format=ser.PublicFormat.Raw,
        )
        return cls(_signing=sk, public=pk)

    @classmethod
    def from_seed(cls, seed: bytes) -> "Identity":
        if len(seed) != 32:
            raise ValueError("Ed25519 seed must be 32 bytes")
        sk = Ed25519PrivateKey.from_private_bytes(seed)
        pk = sk.public_key().public_bytes(
            encoding=ser.Encoding.Raw, format=ser.PublicFormat.Raw,
        )
        return cls(_signing=sk, public=pk)

    def sign(self, msg: bytes) -> bytes:
        return self._signing.sign(msg)

    def peer_id_hex(self) -> str:
        return self.public.hex()


def verify(peer_pub: bytes, msg: bytes, sig: bytes) -> bool:
    if len(peer_pub) != 32:
        return False
    try:
        Ed25519PublicKey.from_public_bytes(peer_pub).verify(sig, msg)
        return True
    except InvalidSignature:
        return False
