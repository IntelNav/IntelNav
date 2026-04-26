//! Wire-bytes ↔ ggml hidden-state conversion.
//!
//! The ggml-side sibling of `crates/runtime/src/hidden.rs`. Same wire
//! contract (little-endian, row-major, Fp16 or per-row symmetric Int8)
//! so a peer running the ggml path is on-wire-compatible with any peer
//! still on the candle path. The only shape change is that ggml has no
//! native Rust tensor type — hidden state lives as a flat `Vec<f32>`
//! plus an explicit `[batch, seq, hidden]` shape.
//!
//! Typical call sites:
//!
//!  * After `ctx.decode_layers(..., run_head=false)`, the caller pulls
//!    per-position hidden state out of the context into a flat
//!    `Vec<f32>` of length `n_tokens * n_embd` and calls
//!    [`encode_hidden`] to produce the `ForwardHidden` payload for the
//!    next peer.
//!
//!  * On the receiving side, the peer reconstructs a `HiddenPayload`
//!    from the wire, calls [`decode_hidden`] to get a flat `Vec<f32>`,
//!    and fills a [`crate::Batch::embeddings`] with it before calling
//!    `decode_layers` or `head_only`.
//!
//! For now we fix `batch == 1` because the IntelNav runtime never
//! drives >1 concurrent session through a single pipe. The wire format
//! still carries a 3-D shape so the protocol stays consistent with
//! the candle path; receivers that want B > 1 will bump a `proto_ver`.

use anyhow::{anyhow, Result};
use half::f16;
use intelnav_wire::Dtype;

/// In-memory hidden state (or logits) shared between Pipelined trait
/// calls. Row-major `f32`. Shape is `[batch, seq, hidden]` for
/// layer-path activations and either `[batch, vocab]` or
/// `[batch, seq, vocab]` for head outputs — encoded in `shape`.
#[derive(Clone, Debug, PartialEq)]
pub struct Hidden {
    pub data:  Vec<f32>,
    pub shape: Vec<usize>,
}

impl Hidden {
    /// Validates `data.len() == shape.iter().product()` and returns
    /// an error if not. Zero-rank shapes are rejected.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        if shape.is_empty() {
            return Err(anyhow!("Hidden::new: empty shape"));
        }
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(anyhow!(
                "Hidden::new: data len {} does not match shape {:?} ({} elems)",
                data.len(),
                shape,
                expected
            ));
        }
        Ok(Self { data, shape })
    }

    /// Number of elements (`shape.iter().product()`).
    pub fn n_elems(&self) -> usize {
        self.shape.iter().product()
    }

    /// `shape[dim]` or `None` if out of range.
    pub fn dim(&self, i: usize) -> Option<usize> {
        self.shape.get(i).copied()
    }

    /// Select a single sequence position from a rank-3
    /// `[batch, seq, inner]` Hidden. Returns shape `[batch, inner]`
    /// with a copy of the chosen row. Used by spec-dec to pull a
    /// specific position's logits out of a head_all output.
    pub fn select_position(&self, pos: usize) -> Result<Hidden> {
        if self.shape.len() != 3 {
            return Err(anyhow!(
                "select_position: need rank-3 hidden [batch, seq, inner], got {:?}",
                self.shape
            ));
        }
        let (b, seq, inner) = (self.shape[0], self.shape[1], self.shape[2]);
        if pos >= seq {
            return Err(anyhow!(
                "select_position: pos {pos} out of range (seq={seq})"
            ));
        }
        // Row-major: the slice for batch `i` at position `pos` lives at
        // `data[i*seq*inner + pos*inner .. i*seq*inner + (pos+1)*inner]`.
        let mut data = Vec::with_capacity(b * inner);
        for i in 0..b {
            let off = i * seq * inner + pos * inner;
            data.extend_from_slice(&self.data[off..off + inner]);
        }
        Hidden::new(data, vec![b, inner])
    }

    /// Argmax over the last axis, flattened across all others. Equivalent
    /// to "give me the vocab id with the highest logit", across any
    /// rank as long as the last axis is the one we're reducing. Spec-dec's
    /// `argmax_last` helper was this in Tensor flavour.
    pub fn argmax_last(&self) -> Result<u32> {
        if self.data.is_empty() {
            return Err(anyhow!("argmax_last: empty hidden"));
        }
        let (idx, _) = self
            .data
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                if x > bv { (i, x) } else { (bi, bv) }
            });
        // idx is a flat index; the last-axis size is shape[last].
        // We want "last-axis position within the final row" — so mod
        // by the last-axis length.
        let last = *self.shape.last().unwrap_or(&self.data.len());
        Ok((idx % last) as u32)
    }
}

/// A hidden-state tensor rendered for the wire. Mirrors the three
/// dynamic fields the `Msg::ForwardHidden` message carries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HiddenPayload {
    pub dtype: Dtype,
    pub shape: [u32; 3], // [batch, seq, hidden]
    pub bytes: Vec<u8>,
}

impl HiddenPayload {
    pub fn encoded_len(&self) -> usize {
        self.bytes.len()
    }

    pub fn n_elems(&self) -> usize {
        (self.shape[0] as usize) * (self.shape[1] as usize) * (self.shape[2] as usize)
    }
}

/// Encode a flat row-major `[batch, seq, hidden]` F32 slice to the Fp16
/// wire format. Thin wrapper around [`encode_hidden_with`] — the common
/// case reads trivially.
pub fn encode_hidden(hidden_f32: &[f32], shape: [u32; 3]) -> Result<HiddenPayload> {
    encode_hidden_with(hidden_f32, shape, Dtype::Fp16)
}

/// Encode a flat row-major `[batch, seq, hidden]` F32 slice using the
/// requested wire dtype.
///
/// * `Dtype::Fp16` — 2 bytes/elem; round-trips F32 with ≤ 2⁻¹⁰ relative
///   error (well below the Q4_K_M weight noise floor).
/// * `Dtype::Int8` — per-token symmetric int8. Layout is
///   `[f32 scale × B*S][i8 value × B*S*H]`: one scale per (batch, seq)
///   row, then i8 values row-major.
/// * `Dtype::Bf16` — rejected; we'd duplicate the Fp16 path with a
///   different cast, and no caller needs it today.
pub fn encode_hidden_with(
    hidden_f32: &[f32],
    shape: [u32; 3],
    dtype: Dtype,
) -> Result<HiddenPayload> {
    let n_elems = shape_n_elems(&shape)?;
    if hidden_f32.len() != n_elems {
        return Err(anyhow!(
            "encode_hidden: slice len {} does not match shape {:?} ({} elems)",
            hidden_f32.len(),
            shape,
            n_elems
        ));
    }

    let bytes = match dtype {
        Dtype::Fp16 => {
            let mut out = Vec::with_capacity(n_elems * 2);
            for &x in hidden_f32 {
                out.extend_from_slice(&f16::from_f32(x).to_le_bytes());
            }
            out
        }
        Dtype::Int8 => {
            let rows = (shape[0] as usize) * (shape[1] as usize);
            let h = shape[2] as usize;
            encode_int8_rows(hidden_f32, rows, h)
        }
        Dtype::Bf16 => {
            return Err(anyhow!("encode_hidden: Bf16 not yet supported"));
        }
    };

    Ok(HiddenPayload { dtype, shape, bytes })
}

/// Per-row symmetric int8 quantization. Matches the candle-side layout
/// byte-for-byte: `[scale_0 .. scale_{rows-1}][q_0 .. q_{rows*h-1}]`.
fn encode_int8_rows(flat: &[f32], rows: usize, h: usize) -> Vec<u8> {
    debug_assert_eq!(flat.len(), rows * h);
    let mut bytes = Vec::with_capacity(rows * 4 + rows * h);

    let mut scales: Vec<f32> = Vec::with_capacity(rows);
    for r in 0..rows {
        let row = &flat[r * h..(r + 1) * h];
        let absmax = row.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
        // All-zero row: any positive scale decodes back to zero; 1.0
        // keeps the math simple and avoids a divide-by-zero below.
        let scale = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
        scales.push(scale);
    }
    for s in &scales {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    for r in 0..rows {
        let row = &flat[r * h..(r + 1) * h];
        let inv = 1.0 / scales[r];
        for &x in row {
            let q = (x * inv).round().clamp(-127.0, 127.0) as i8;
            bytes.push(q as u8);
        }
    }
    bytes
}

/// Decode a wire `HiddenPayload` back into a flat row-major F32 slice
/// ready to be handed to [`crate::Batch::fill_embeddings`]. Returns
/// `(shape, hidden_f32)` so the caller doesn't need to re-validate.
///
/// The caller owns the returned `Vec<f32>`; on-wire bytes are not
/// retained.
pub fn decode_hidden(payload: &HiddenPayload) -> Result<(/*shape*/ [u32; 3], Vec<f32>)> {
    let n_elems = shape_n_elems(&payload.shape)?;
    let rows = (payload.shape[0] as usize) * (payload.shape[1] as usize);

    let expected_bytes = match payload.dtype {
        Dtype::Fp16 | Dtype::Bf16 => n_elems * 2,
        Dtype::Int8 => rows * 4 + n_elems,
    };
    if payload.bytes.len() != expected_bytes {
        return Err(anyhow!(
            "hidden payload size mismatch: shape={:?} dtype={:?} → {} bytes, got {}",
            payload.shape,
            payload.dtype,
            expected_bytes,
            payload.bytes.len()
        ));
    }

    let vals: Vec<f32> = match payload.dtype {
        Dtype::Fp16 => {
            let mut vals: Vec<f32> = Vec::with_capacity(n_elems);
            for chunk in payload.bytes.chunks_exact(2) {
                let v = f16::from_le_bytes([chunk[0], chunk[1]]);
                vals.push(v.to_f32());
            }
            vals
        }
        Dtype::Int8 => {
            let h = payload.shape[2] as usize;
            let (scale_bytes, q_bytes) = payload.bytes.split_at(rows * 4);
            let mut scales: Vec<f32> = Vec::with_capacity(rows);
            for chunk in scale_bytes.chunks_exact(4) {
                scales.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            let mut vals: Vec<f32> = Vec::with_capacity(n_elems);
            for r in 0..rows {
                let s = scales[r];
                let row = &q_bytes[r * h..(r + 1) * h];
                for &byte in row {
                    let q = byte as i8;
                    vals.push((q as f32) * s);
                }
            }
            vals
        }
        Dtype::Bf16 => {
            return Err(anyhow!("decode_hidden: Bf16 not yet supported"));
        }
    };

    Ok((payload.shape, vals))
}

fn shape_n_elems(shape: &[u32; 3]) -> Result<usize> {
    (shape[0] as usize)
        .checked_mul(shape[1] as usize)
        .and_then(|n| n.checked_mul(shape[2] as usize))
        .ok_or_else(|| anyhow!("hidden shape overflows usize: {:?}", shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic "random-ish" sample spread across ~[-2, 2] — typical
    /// post-residual activation range. Irrational scaling avoids lattice
    /// artifacts that would round-trip exactly in fp16.
    fn sample(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (i as f32).sin() * 1.7 + ((i * 31) as f32).cos() * 0.3)
            .collect()
    }

    #[test]
    fn roundtrip_shape_and_dtype() {
        let shape = [1u32, 4, 16];
        let n = 1 * 4 * 16;
        let data = sample(n);

        let p = encode_hidden(&data, shape).unwrap();
        assert_eq!(p.dtype, Dtype::Fp16);
        assert_eq!(p.shape, shape);
        assert_eq!(p.bytes.len(), n * 2);

        let (back_shape, back) = decode_hidden(&p).unwrap();
        assert_eq!(back_shape, shape);
        assert_eq!(back.len(), n);
    }

    #[test]
    fn roundtrip_values_within_fp16_precision() {
        let shape = [2u32, 8, 32];
        let n = 2 * 8 * 32;
        let orig = sample(n);

        let p = encode_hidden(&orig, shape).unwrap();
        let (_, got) = decode_hidden(&p).unwrap();

        let max_abs = orig
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        // fp16 carries ~10 bits of mantissa → max rel err ≈ 2⁻¹⁰ ≈ 1e-3.
        // With values in ~[-2, 2] the absolute bound stays well under 4e-3.
        assert!(max_abs < 4e-3, "fp16 round-trip exceeded tolerance: max_abs = {max_abs}");
    }

    #[test]
    fn bytes_are_stable_across_encode_calls() {
        // Identical bytes must come out of encode every call — the
        // network layer uses hash-based replay detection + cache
        // dedup on the serialized payload.
        let shape = [1u32, 3, 8];
        let data = sample(1 * 3 * 8);
        let a = encode_hidden(&data, shape).unwrap();
        let b = encode_hidden(&data, shape).unwrap();
        assert_eq!(a.bytes, b.bytes);
    }

    #[test]
    fn slice_len_shape_mismatch_is_error() {
        let shape = [1u32, 2, 4];
        let too_short = vec![0.0_f32; 7];
        assert!(encode_hidden(&too_short, shape).is_err());
    }

    #[test]
    fn truncated_payload_is_rejected() {
        let shape = [1u32, 2, 4];
        let data = sample(1 * 2 * 4);
        let mut p = encode_hidden(&data, shape).unwrap();
        p.bytes.pop();
        assert!(decode_hidden(&p).is_err());
    }

    #[test]
    fn int8_payload_size_is_rows_plus_values() {
        let shape = [2u32, 3, 16];
        let data = sample(2 * 3 * 16);
        let p = encode_hidden_with(&data, shape, Dtype::Int8).unwrap();
        assert_eq!(p.dtype, Dtype::Int8);
        assert_eq!(p.shape, shape);
        assert_eq!(p.bytes.len(), 2 * 3 * 4 + 2 * 3 * 16);
    }

    #[test]
    fn int8_roundtrip_within_per_row_tolerance() {
        let shape = [2u32, 8, 64];
        let rows = 2 * 8;
        let h = 64;
        let orig = sample(rows * h);

        let p = encode_hidden_with(&orig, shape, Dtype::Int8).unwrap();
        let (_, got) = decode_hidden(&p).unwrap();

        for r in 0..rows {
            let row_o = &orig[r * h..(r + 1) * h];
            let row_g = &got[r * h..(r + 1) * h];
            let absmax = row_o.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
            let tol = (absmax / 127.0) * 1.01 + 1e-6;
            let max_err = row_o
                .iter()
                .zip(row_g)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                max_err <= tol,
                "row {r}: max_err={max_err} > tol={tol} (absmax={absmax})"
            );
        }
    }

    #[test]
    fn int8_all_zero_row_decodes_to_zero() {
        let shape = [1u32, 1, 8];
        let data = vec![0.0_f32; 8];
        let p = encode_hidden_with(&data, shape, Dtype::Int8).unwrap();
        let (_, got) = decode_hidden(&p).unwrap();
        assert_eq!(got, vec![0.0_f32; 8]);
    }

    #[test]
    fn int8_bytes_are_stable_across_encode_calls() {
        let shape = [1u32, 4, 32];
        let data = sample(1 * 4 * 32);
        let a = encode_hidden_with(&data, shape, Dtype::Int8).unwrap();
        let b = encode_hidden_with(&data, shape, Dtype::Int8).unwrap();
        assert_eq!(a.bytes, b.bytes);
    }

    #[test]
    fn int8_truncated_payload_is_rejected() {
        let shape = [1u32, 2, 8];
        let data = sample(1 * 2 * 8);
        let mut p = encode_hidden_with(&data, shape, Dtype::Int8).unwrap();
        p.bytes.pop();
        assert!(decode_hidden(&p).is_err());
    }

    #[test]
    fn bf16_rejected_with_error() {
        let shape = [1u32, 2, 4];
        let data = sample(1 * 2 * 4);
        assert!(encode_hidden_with(&data, shape, Dtype::Bf16).is_err());
    }
}
