//! `intelnav-wire` — CBOR codecs for the normative protocol messages defined
//! in paper §A.
//!
//! Every message is length-prefixed (big-endian `u32`) and CBOR-encoded;
//! helpers [`encode_frame`] / [`decode_frame`] handle framing.

#![forbid(unsafe_code)]

use bytes::{BufMut, BytesMut};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use intelnav_core::{ModelId, PeerId, SessionId};
use intelnav_core::types::{CapabilityV1, LayerRange, Quant};

#[derive(Debug, Error)]
pub enum WireError {
    #[error("cbor encode: {0}")]
    Encode(String),
    #[error("cbor decode: {0}")]
    Decode(String),
    #[error("frame too large: {0} bytes (max {1})")]
    FrameTooLarge(usize, usize),
    #[error("truncated frame: got {0} bytes, need {1}")]
    Truncated(usize, usize),
}
pub type Result<T> = std::result::Result<T, WireError>;

/// 16 MiB safety cap on a single frame.
pub const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;

/// Phase of the inference pass the hidden-state belongs to.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    Prefill,
    Decode,
}

/// Tensor element type for wire transport. Paper §4.5 — "int8 by default."
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
    Fp16,
    Bf16,
    Int8,
}

/// Normative protocol messages. See paper §A for the authoritative spec.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Msg {
    // -------- session setup --------
    Hello {
        peer_id:          PeerId,
        proto_ver:        u32,
        supported_quants: Vec<Quant>,
    },
    SessionInit {
        session_id:       SessionId,
        client_x25519_pub: [u8; 32],
        model_cid:        String,
        layer_range:      LayerRange,
        max_seq:          u32,
    },
    SessionAck {
        session_id:       SessionId,
        shard_x25519_pub: [u8; 32],
    },

    // -------- inference --------
    Prompt {
        session_id: SessionId,
        /// AES-256-GCM ciphertext of the user prompt (paper §8.2).
        ciphertext: Vec<u8>,
        nonce:      [u8; 12],
    },
    ForwardHidden {
        session_id: SessionId,
        seq:        u64,
        phase:      Phase,
        dtype:      Dtype,
        shape:      [u32; 3],
        payload:    Vec<u8>,
        kv_delta:   Option<Vec<u8>>,
        /// Truncate the peer's KV cache to this sequence length before
        /// running the forward. Used by speculative decoding to roll
        /// back rejected draft tokens. Omitted on the wire when `None`
        /// so non-spec traffic stays byte-identical to proto v1.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        kv_truncate_to: Option<u32>,
    },
    Token {
        session_id:  SessionId,
        seq:         u64,
        logits_top_k: Option<Vec<(u32, f32)>>,
        sampled:     u32,
    },

    // -------- control --------
    Heartbeat {
        session_id: SessionId,
        last_seq:   u64,
        health:     u8,
    },
    AbortSession {
        session_id: SessionId,
        reason:     String,
    },

    // -------- network --------
    Advertise {
        capability_v1: CapabilityV1,
        sig:           Vec<u8>,
    },
    Gossip {
        topic:   String,
        payload: Vec<u8>,
        sig:     Vec<u8>,
    },
}

/// Encode a single `Msg` to CBOR bytes (no length prefix).
pub fn encode(msg: &Msg) -> Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(256);
    ciborium::into_writer(msg, &mut buf).map_err(|e| WireError::Encode(e.to_string()))?;
    Ok(buf)
}

/// Decode a `Msg` from CBOR bytes.
pub fn decode(bytes: &[u8]) -> Result<Msg> {
    ciborium::from_reader(bytes).map_err(|e| WireError::Decode(e.to_string()))
}

/// Length-prefix + CBOR-encode a message into `dst`.
pub fn encode_frame(dst: &mut BytesMut, msg: &Msg) -> Result<()> {
    let payload = encode(msg)?;
    if payload.len() > MAX_FRAME_BYTES {
        return Err(WireError::FrameTooLarge(payload.len(), MAX_FRAME_BYTES));
    }
    dst.put_u32(payload.len() as u32);
    dst.put_slice(&payload);
    Ok(())
}

/// Try to pull a single framed message from the front of `src`.
/// Returns `Ok(None)` when the buffer does not yet contain a complete frame.
pub fn decode_frame(src: &mut BytesMut) -> Result<Option<Msg>> {
    if src.len() < 4 {
        return Ok(None);
    }
    let len = u32::from_be_bytes([src[0], src[1], src[2], src[3]]) as usize;
    if len > MAX_FRAME_BYTES {
        return Err(WireError::FrameTooLarge(len, MAX_FRAME_BYTES));
    }
    if src.len() < 4 + len {
        return Ok(None);
    }
    let _ = src.split_to(4);
    let payload = src.split_to(len);
    let msg = decode(&payload)?;
    Ok(Some(msg))
}

// ------------------------------------------------------------------
//  DHT key derivation (paper §7.1 sample record)
// ------------------------------------------------------------------

/// Build the DHT provider-lookup key for a `(model, quant)` tuple.
/// Matches the paper's `/intelnav/v1/model/<CID>/<quant>` convention.
pub fn dht_provider_key(model: &ModelId, quant: Quant) -> String {
    format!("/intelnav/v1/model/{}/{}", model.as_str(), quant.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use intelnav_core::{PeerId, SessionId};

    #[test]
    fn roundtrip_hello() {
        let m = Msg::Hello {
            peer_id: PeerId::new([7u8; 32]),
            proto_ver: 1,
            supported_quants: vec![Quant::Q4KM, Quant::FP16],
        };
        let bytes = encode(&m).unwrap();
        let back = decode(&bytes).unwrap();
        if let Msg::Hello { peer_id, proto_ver, .. } = back {
            assert_eq!(peer_id.as_bytes(), &[7u8; 32]);
            assert_eq!(proto_ver, 1);
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn framing_is_reversible() {
        let m = Msg::Heartbeat {
            session_id: SessionId([1u8; 32]),
            last_seq: 42,
            health: 100,
        };
        let mut buf = BytesMut::new();
        encode_frame(&mut buf, &m).unwrap();
        encode_frame(&mut buf, &m).unwrap();
        let a = decode_frame(&mut buf).unwrap();
        let b = decode_frame(&mut buf).unwrap();
        assert!(matches!(a, Some(Msg::Heartbeat { .. })));
        assert!(matches!(b, Some(Msg::Heartbeat { .. })));
        assert!(buf.is_empty());
    }
}
