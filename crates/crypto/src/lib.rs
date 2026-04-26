//! `intelnav-crypto` — identity + transport + prompt-confidentiality helpers.
//!
//! * **Identity** — Ed25519 keypair; [`PeerId`] is `multihash(public_key)`
//!   (paper §6.1). For now we store the raw 32-byte public key.
//! * **Handshake** — ephemeral X25519 exchange between the client and the
//!   *entry shard* (paper §3.3 step 4). The gateway never sees the key.
//! * **Prompt encryption** — AES-256-GCM over the user prompt, keyed from the
//!   X25519 shared secret (paper §8.2).

#![forbid(unsafe_code)]

use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Nonce};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::rngs::OsRng;
use rand::RngCore;
use thiserror::Error;
use x25519_dalek::{EphemeralSecret, PublicKey as X25519Pub, StaticSecret};

use intelnav_core::PeerId;

#[derive(Debug, Error)]
pub enum CryptoError {
    #[error("signature verification failed")]
    BadSignature,
    #[error("aes-gcm: {0}")]
    Aead(String),
    #[error("invalid key length")]
    BadKeyLen,
    #[error("ed25519: {0}")]
    Ed25519(String),
}
pub type Result<T> = std::result::Result<T, CryptoError>;

// ======================================================================
//  Identity
// ======================================================================

/// Long-lived per-peer Ed25519 signing identity.
///
/// Paper §6.1: "Every peer generates an Ed25519 keypair on first start. The
/// peer ID is multihash(pubkey)."
#[derive(Clone)]
pub struct Identity {
    signing: SigningKey,
}

impl Identity {
    /// Generate a fresh identity using the OS CSPRNG.
    pub fn generate() -> Self {
        Self { signing: SigningKey::generate(&mut OsRng) }
    }

    /// Load an identity from its 32-byte seed.
    pub fn from_seed(seed: &[u8; 32]) -> Self {
        Self { signing: SigningKey::from_bytes(seed) }
    }

    /// Raw 32-byte seed — caller should store this in a keychain.
    pub fn seed(&self) -> [u8; 32] {
        self.signing.to_bytes()
    }

    /// 32-byte public key (also the wire `peer_id`).
    pub fn public(&self) -> [u8; 32] {
        self.signing.verifying_key().to_bytes()
    }

    /// Public peer identifier.
    pub fn peer_id(&self) -> PeerId {
        PeerId::new(self.public())
    }

    /// Sign a message.
    pub fn sign(&self, msg: &[u8]) -> Vec<u8> {
        self.signing.sign(msg).to_bytes().to_vec()
    }
}

/// Verify an Ed25519 signature over `msg` produced by `peer_pub`.
pub fn verify(peer_pub: &[u8; 32], msg: &[u8], sig: &[u8]) -> Result<()> {
    let vk = VerifyingKey::from_bytes(peer_pub).map_err(|e| CryptoError::Ed25519(e.to_string()))?;
    let sig_arr: [u8; 64] = sig.try_into().map_err(|_| CryptoError::BadKeyLen)?;
    vk.verify(msg, &Signature::from_bytes(&sig_arr))
        .map_err(|_| CryptoError::BadSignature)
}

// ======================================================================
//  Handshake (X25519)
// ======================================================================

/// One half of a fresh X25519 handshake. Produces a public key the peer can
/// see, consumes itself on `derive_shared`.
pub struct EphemeralHandshake {
    secret: EphemeralSecret,
    public: [u8; 32],
}

impl EphemeralHandshake {
    pub fn new() -> Self {
        let secret = EphemeralSecret::random_from_rng(OsRng);
        let public = X25519Pub::from(&secret).to_bytes();
        Self { secret, public }
    }
    pub fn public(&self) -> [u8; 32] {
        self.public
    }
    /// Finish the handshake, yielding a 32-byte shared secret.
    pub fn derive_shared(self, peer_pub: &[u8; 32]) -> [u8; 32] {
        let peer = X25519Pub::from(*peer_pub);
        self.secret.diffie_hellman(&peer).to_bytes()
    }
}
impl Default for EphemeralHandshake {
    fn default() -> Self {
        Self::new()
    }
}

/// Static (long-term) X25519 keypair for shard nodes. Useful when the shard
/// wants its advertised public key to be stable across sessions.
#[derive(Clone)]
pub struct StaticHandshake {
    secret: StaticSecret,
    public: [u8; 32],
}

impl StaticHandshake {
    pub fn generate() -> Self {
        let secret = StaticSecret::random_from_rng(OsRng);
        let public = X25519Pub::from(&secret).to_bytes();
        Self { secret, public }
    }
    pub fn public(&self) -> [u8; 32] {
        self.public
    }
    pub fn derive_shared(&self, peer_pub: &[u8; 32]) -> [u8; 32] {
        let peer = X25519Pub::from(*peer_pub);
        self.secret.diffie_hellman(&peer).to_bytes()
    }
}

// ======================================================================
//  Prompt encryption (AES-256-GCM)
// ======================================================================

/// Derive the AES-256 session key from the raw X25519 shared secret, with a
/// domain-separation tag ("intelnav/v1/prompt").
pub fn session_key(shared: &[u8; 32]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(b"intelnav/v1/prompt");
    h.update(shared);
    let mut out = [0u8; 32];
    h.finalize_xof().fill(&mut out);
    out
}

/// Encrypt `plaintext` with a freshly-generated 96-bit nonce; returns
/// `(ciphertext, nonce)`.
pub fn encrypt(key: &[u8; 32], plaintext: &[u8]) -> Result<(Vec<u8>, [u8; 12])> {
    let cipher = Aes256Gcm::new_from_slice(key).map_err(|_| CryptoError::BadKeyLen)?;
    let mut nonce = [0u8; 12];
    OsRng.fill_bytes(&mut nonce);
    let ct = cipher
        .encrypt(Nonce::from_slice(&nonce), plaintext)
        .map_err(|e| CryptoError::Aead(e.to_string()))?;
    Ok((ct, nonce))
}

pub fn decrypt(key: &[u8; 32], ciphertext: &[u8], nonce: &[u8; 12]) -> Result<Vec<u8>> {
    let cipher = Aes256Gcm::new_from_slice(key).map_err(|_| CryptoError::BadKeyLen)?;
    cipher
        .decrypt(Nonce::from_slice(nonce), ciphertext)
        .map_err(|e| CryptoError::Aead(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_sign_verify() {
        let id = Identity::generate();
        let sig = id.sign(b"hello");
        verify(&id.public(), b"hello", &sig).unwrap();
        assert!(verify(&id.public(), b"wrong", &sig).is_err());
    }

    #[test]
    fn x25519_matches_both_sides() {
        let a = EphemeralHandshake::new();
        let b = EphemeralHandshake::new();
        let a_pub = a.public();
        let b_pub = b.public();
        let k_a = a.derive_shared(&b_pub);
        let k_b = b.derive_shared(&a_pub);
        assert_eq!(k_a, k_b);
    }

    #[test]
    fn aes_gcm_roundtrip() {
        let k = [9u8; 32];
        let (ct, nonce) = encrypt(&k, b"prompt text").unwrap();
        let pt = decrypt(&k, &ct, &nonce).unwrap();
        assert_eq!(pt, b"prompt text");
    }
}
