//! CID construction: SHA-256 over raw bytes → CIDv1.
//!
//! Codec is `raw` (0x55); multihash code is `sha2-256` (0x12). This
//! matches what an IPFS node would produce for `ipfs add --raw-leaves
//! --cid-version=1 --hash=sha2-256 <file>` on a single-chunk file,
//! which is the property we want: anyone who fetches a chunk by CID
//! from any IPFS-compatible swarm and verifies the hash gets the
//! same bytes we wrote.

use cid::Cid;
use multihash::Multihash;
use sha2::{Digest, Sha256};

/// `raw` codec — the chunk bytes are stored verbatim, not wrapped in
/// DAG-CBOR or DAG-PB.
pub const RAW_CODEC: u64 = 0x55;

/// `sha2-256` multihash code, per the multicodec table.
pub const SHA2_256_CODE: u64 = 0x12;

/// Byte length of a SHA-256 digest.
pub const SHA2_256_LEN: usize = 32;

/// Compute the CIDv1 for the given bytes. The underlying `Cid` type
/// alias from the `cid` crate is `CidGeneric<64>` — a 64-byte digest
/// envelope. SHA-256 is only 32 bytes, but it fits cleanly.
pub fn cid_for(bytes: &[u8]) -> Cid {
    let digest = Sha256::digest(bytes);
    // Multihash::wrap copies the digest and stores (code, size, data);
    // it only fails if digest.len() > S, which can't happen here.
    let mh = Multihash::<64>::wrap(SHA2_256_CODE, digest.as_slice())
        .expect("SHA-256 digest is always 32 bytes, fits in Multihash<64>");
    Cid::new_v1(RAW_CODEC, mh)
}

/// Convenience wrapper returning the canonical base32 string
/// representation (the familiar `bafkrei...` form).
pub fn cid_string_for(bytes: &[u8]) -> String {
    cid_for(bytes).to_string()
}

/// Construct a CID directly from a pre-computed SHA-256 digest. Use
/// this when you've already streamed the bytes through a `Sha256`
/// hasher (e.g. in the chunker/stitcher/fetcher) and want to avoid
/// re-hashing them.
pub fn cid_string_from_sha256(digest: &[u8]) -> String {
    assert!(digest.len() == SHA2_256_LEN, "digest must be 32 bytes");
    let mh = Multihash::<64>::wrap(SHA2_256_CODE, digest)
        .expect("SHA-256 digest fits in Multihash<64>");
    Cid::new_v1(RAW_CODEC, mh).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference vector: SHA-256("hello world") = a948904f2f0f479b8f8197694b30184b
    /// 0d2ed1c1cd2a1ec0fb85d299a192a447. The CID wrapping that digest
    /// (raw codec, CIDv1, base32) is a well-known IPFS fixture.
    #[test]
    fn hello_world_matches_ipfs() {
        let cid = cid_string_for(b"hello world");
        // This is what `ipfs add --cid-version=1 --raw-leaves` emits
        // for a file containing exactly "hello world" (no newline).
        assert_eq!(
            cid,
            "bafkreifzjut3te2nhyekklss27nh3k72ysco7y32koao5eei66wof36n5e"
        );
    }

    #[test]
    fn same_bytes_same_cid() {
        let a = cid_for(b"intelnav");
        let b = cid_for(b"intelnav");
        assert_eq!(a, b);
    }

    #[test]
    fn different_bytes_different_cid() {
        let a = cid_for(b"intelnav");
        let b = cid_for(b"IntelNav");
        assert_ne!(a, b);
    }
}
