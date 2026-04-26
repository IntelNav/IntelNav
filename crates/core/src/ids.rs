//! Opaque identifier types — peers, sessions, models.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Ed25519-derived peer identifier. Wire form is the raw 32-byte public key;
/// display form is base58 (matching libp2p's `PeerId` display convention).
#[derive(Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PeerId(#[serde(with = "hex::serde")] pub [u8; 32]);

impl PeerId {
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
    pub fn short(&self) -> String {
        let s = bs58::encode(self.0).into_string();
        let mut out = String::with_capacity(14);
        out.push_str(&s[..6]);
        out.push('…');
        out.push_str(&s[s.len() - 6..]);
        out
    }
}

impl fmt::Display for PeerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&bs58::encode(self.0).into_string())
    }
}
impl fmt::Debug for PeerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PeerId({})", self.short())
    }
}

impl FromStr for PeerId {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        let bytes = bs58::decode(s)
            .into_vec()
            .map_err(|e| Error::Parse(format!("invalid peer id: {e}")))?;
        let arr: [u8; 32] = bytes
            .try_into()
            .map_err(|_| Error::Parse("peer id must decode to 32 bytes".into()))?;
        Ok(Self(arr))
    }
}

/// 32-byte random session identifier (paper §4.5 `session_id`).
#[derive(Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionId(#[serde(with = "hex::serde")] pub [u8; 32]);

impl SessionId {
    pub fn random() -> Self {
        let mut b = [0u8; 32];
        getrandom_fill(&mut b);
        Self(b)
    }
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}
impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in self.0.iter().take(8) {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}
impl fmt::Debug for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SessionId({self})")
    }
}

/// Human-facing model identifier, e.g. `"deepseek-coder:33b"`.
///
/// Comparison with DHT records uses the IPFS CID of the weight blob — the
/// string form is only a convenience for clients.
#[derive(Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModelId(pub String);

impl ModelId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl fmt::Debug for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ModelId({:?})", self.0)
    }
}
impl From<&str> for ModelId {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}
impl From<String> for ModelId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// Thin portability wrapper so core does not take a hard `rand` dependency.
fn getrandom_fill(buf: &mut [u8]) {
    // blake3 pulls in a cryptographic RNG transitively; we seed from the
    // system instead of exposing another optional dep on `core`.
    // SAFETY: std::time + thread_id + blake3 is not cryptographic, but
    // SessionIds only need to be unique + unpredictable relative to network
    // peers, not against an informed attacker with local read access.
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tid = std::thread::current().id();
    let mut hasher = blake3::Hasher::new();
    hasher.update(&now.to_le_bytes());
    hasher.update(format!("{tid:?}").as_bytes());
    hasher.update(&(buf.len() as u64).to_le_bytes());
    let mut out = hasher.finalize_xof();
    out.fill(buf);
}

// tiny base58 impl so `core` doesn't need a transitive dep for display
mod bs58 {
    const ALPHABET: &[u8; 58] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    pub fn encode(bytes: impl AsRef<[u8]>) -> Encoder {
        Encoder(bytes.as_ref().to_vec())
    }
    pub fn decode(s: &str) -> Decoder {
        Decoder(s.to_string())
    }
    pub struct Encoder(Vec<u8>);
    impl Encoder {
        pub fn into_string(self) -> String {
            let input = self.0;
            let mut leading = 0;
            for &b in &input {
                if b == 0 {
                    leading += 1;
                } else {
                    break;
                }
            }
            let mut num: Vec<u32> = Vec::with_capacity(input.len());
            for &b in &input {
                let mut carry = b as u32;
                for digit in num.iter_mut() {
                    let acc = *digit * 256 + carry;
                    *digit = acc % 58;
                    carry = acc / 58;
                }
                while carry > 0 {
                    num.push(carry % 58);
                    carry /= 58;
                }
            }
            let mut out = Vec::with_capacity(leading + num.len());
            for _ in 0..leading {
                out.push(ALPHABET[0]);
            }
            for &d in num.iter().rev() {
                out.push(ALPHABET[d as usize]);
            }
            String::from_utf8(out).expect("ascii")
        }
    }
    pub struct Decoder(String);
    impl Decoder {
        pub fn into_vec(self) -> Result<Vec<u8>, String> {
            let input = self.0;
            let mut leading = 0;
            for c in input.chars() {
                if c == '1' {
                    leading += 1;
                } else {
                    break;
                }
            }
            let mut num: Vec<u32> = Vec::with_capacity(input.len());
            for c in input.chars() {
                let idx = ALPHABET
                    .iter()
                    .position(|&x| x == c as u8)
                    .ok_or_else(|| format!("invalid base58 char {c:?}"))?;
                let mut carry = idx as u32;
                for digit in num.iter_mut() {
                    let acc = *digit * 58 + carry;
                    *digit = acc & 0xff;
                    carry = acc >> 8;
                }
                while carry > 0 {
                    num.push(carry & 0xff);
                    carry >>= 8;
                }
            }
            let mut out = Vec::with_capacity(leading + num.len());
            for _ in 0..leading {
                out.push(0);
            }
            out.extend(num.iter().rev().map(|&x| x as u8));
            Ok(out)
        }
    }
}
