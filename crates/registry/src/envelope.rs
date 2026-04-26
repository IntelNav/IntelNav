//! Signed-envelope verification (spec §3.1).
//!
//! Every mutating request carries `{ envelope, sig }` where `sig` is an
//! Ed25519 signature by `envelope.peer_id` over `CBOR(envelope)`.

use serde::{Deserialize, Serialize};

use intelnav_crypto::verify as ed25519_verify;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Envelope {
    pub peer_id:   String,        // 64-char hex (PeerId wire form)
    pub model_cid: String,
    pub part_id:   String,
    pub timestamp: i64,           // unix seconds
    pub nonce:     String,        // 32-char hex (16 bytes)
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SignedEnvelope {
    pub envelope: Envelope,
    pub sig:      String,         // 128-char hex
}

#[derive(Debug, thiserror::Error)]
pub enum EnvelopeError {
    #[error("malformed peer_id")]
    BadPeerId,
    #[error("malformed signature")]
    BadSignature,
    #[error("timestamp out of replay window ({skew}s skew)")]
    Replay { skew: i64 },
    #[error("envelope context mismatch (expected {expected}, got {got})")]
    ContextMismatch { expected: String, got: String },
    #[error("signature verification failed")]
    VerifyFailed,
    #[error("cbor encode: {0}")]
    Cbor(String),
}

pub type EnvelopeResult<T> = std::result::Result<T, EnvelopeError>;

impl SignedEnvelope {
    /// Verify signature, replay window, and that the envelope matches the
    /// expected `(model_cid, part_id)` context of the HTTP route.
    pub fn verify(
        &self,
        expected_model_cid: &str,
        expected_part_id: &str,
        replay_window_s: u32,
        now_unix: i64,
    ) -> EnvelopeResult<[u8; 32]> {
        if self.envelope.model_cid != expected_model_cid {
            return Err(EnvelopeError::ContextMismatch {
                expected: expected_model_cid.to_owned(),
                got:      self.envelope.model_cid.clone(),
            });
        }
        if self.envelope.part_id != expected_part_id {
            return Err(EnvelopeError::ContextMismatch {
                expected: expected_part_id.to_owned(),
                got:      self.envelope.part_id.clone(),
            });
        }

        let skew = (now_unix - self.envelope.timestamp).abs();
        if skew > replay_window_s as i64 {
            return Err(EnvelopeError::Replay { skew });
        }

        let peer_bytes = hex_to_32(&self.envelope.peer_id)
            .ok_or(EnvelopeError::BadPeerId)?;
        let sig_bytes = hex::decode(&self.sig)
            .map_err(|_| EnvelopeError::BadSignature)?;

        let mut payload = Vec::with_capacity(256);
        ciborium::into_writer(&self.envelope, &mut payload)
            .map_err(|e| EnvelopeError::Cbor(e.to_string()))?;

        ed25519_verify(&peer_bytes, &payload, &sig_bytes)
            .map_err(|_| EnvelopeError::VerifyFailed)?;

        Ok(peer_bytes)
    }
}

pub fn hex_to_32(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 { return None; }
    let mut out = [0u8; 32];
    hex::decode_to_slice(s, &mut out).ok().map(|_| out)
}

/// Helper for unsigned timestamps (tests + callers).
pub fn now_unix() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use intelnav_crypto::Identity;

    #[test]
    fn roundtrip_sign_verify() {
        let id = Identity::generate();
        let env = Envelope {
            peer_id:   hex::encode(id.public()),
            model_cid: "bafy-test".into(),
            part_id:   "p1".into(),
            timestamp: now_unix(),
            nonce:     hex::encode([7u8; 16]),
        };
        let mut buf = Vec::new();
        ciborium::into_writer(&env, &mut buf).unwrap();
        let sig = id.sign(&buf);
        let se = SignedEnvelope { envelope: env, sig: hex::encode(&sig) };
        let pk = se.verify("bafy-test", "p1", 120, now_unix()).unwrap();
        assert_eq!(pk, id.public());
    }

    #[test]
    fn rejects_wrong_context() {
        let id = Identity::generate();
        let env = Envelope {
            peer_id:   hex::encode(id.public()),
            model_cid: "bafy-test".into(),
            part_id:   "p1".into(),
            timestamp: now_unix(),
            nonce:     hex::encode([7u8; 16]),
        };
        let mut buf = Vec::new();
        ciborium::into_writer(&env, &mut buf).unwrap();
        let sig = id.sign(&buf);
        let se = SignedEnvelope { envelope: env, sig: hex::encode(&sig) };
        assert!(se.verify("bafy-test", "p2", 120, now_unix()).is_err());
    }
}
