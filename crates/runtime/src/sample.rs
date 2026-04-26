//! Pure-Rust logits sampler — temperature, top-p, repeat penalty, seed.
//!
//! Replaces the old `candle_transformers::generation::LogitsProcessor` +
//! `candle_transformers::utils::apply_repeat_penalty` pair so the
//! runtime crate doesn't pull in candle just to sample one u32 per
//! decoded token. Math matches the candle implementations on the
//! greedy and top-p paths we exercise:
//!
//! * `temperature == 0.0` → argmax. Matches candle's `Sampling::ArgMax`
//!   (no division, no softmax).
//! * `temperature > 0.0`, `top_p == None` → straight multinomial after
//!   `softmax(logits / T)`.
//! * `temperature > 0.0`, `top_p == Some(p)` with `0 < p < 1` →
//!   nucleus sampling: sort by probability descending, keep the
//!   smallest prefix whose cumulative mass is `>= p`, renormalize, draw.
//!   `p >= 1.0` is treated as a no-op (full distribution).
//! * `repeat_penalty > 1.0` divides the logit at every previously-
//!   generated token id (in the configured context window) by the
//!   penalty, matching `candle_transformers::utils::apply_repeat_penalty`.

use anyhow::{anyhow, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Stateful sampler — owns the RNG, otherwise a transparent function
/// of its config. Cheap to construct; one per generation loop.
pub struct Sampler {
    temperature: f64,
    top_p:       Option<f64>,
    rng:         StdRng,
}

impl Sampler {
    pub fn new(seed: u64, temperature: f64, top_p: Option<f64>) -> Self {
        Self {
            temperature,
            top_p,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Sample one token id from a `[vocab]`-shaped logit slice. Mutates
    /// a working copy of the logits to apply repeat penalty; the
    /// caller's slice is not modified.
    pub fn sample(&mut self, logits: &[f32], ctx: &[u32], cfg: &SamplerCfg) -> Result<u32> {
        if logits.is_empty() {
            return Err(anyhow!("sample: empty logits"));
        }

        // Repeat penalty: in-place on a working copy so the caller's
        // logits stay clean. Apply *before* temperature scaling so the
        // penalty's magnitude is logit-space (matches candle).
        let mut work: Vec<f32>;
        let view: &[f32] = if cfg.repeat_penalty > 1.0 && !ctx.is_empty() {
            work = logits.to_vec();
            apply_repeat_penalty(&mut work, ctx, cfg.repeat_ctx, cfg.repeat_penalty);
            &work
        } else {
            logits
        };

        if self.temperature <= 0.0 {
            return Ok(argmax(view));
        }

        // Softmax with temperature, in-place into `probs`.
        let probs = softmax_temperature(view, self.temperature as f32);

        match self.top_p {
            Some(p) if p > 0.0 && p < 1.0 => sample_top_p(&probs, p as f32, &mut self.rng),
            _ => sample_multinomial(&probs, &mut self.rng),
        }
    }
}

/// Knobs that aren't part of the RNG state — shared with `SamplingCfg`
/// in the public API but kept separate here so the sampler doesn't
/// depend on the broader runtime types.
#[derive(Clone, Copy, Debug)]
pub struct SamplerCfg {
    pub repeat_penalty: f32,
    pub repeat_ctx:     usize,
}

fn argmax(logits: &[f32]) -> u32 {
    let (idx, _) = logits.iter().enumerate().fold(
        (0usize, f32::NEG_INFINITY),
        |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) },
    );
    idx as u32
}

/// `apply_repeat_penalty` — matches the candle helper exactly. For
/// each token id appearing in `ctx[ctx.len() - repeat_ctx ..]`, divide
/// the corresponding logit by `penalty` if positive, multiply if
/// negative. (Mirrors HF transformers' RepetitionPenaltyLogitsProcessor.)
fn apply_repeat_penalty(logits: &mut [f32], ctx: &[u32], repeat_ctx: usize, penalty: f32) {
    let start = ctx.len().saturating_sub(repeat_ctx);
    for &tok in &ctx[start..] {
        let i = tok as usize;
        if i >= logits.len() {
            continue;
        }
        let v = logits[i];
        logits[i] = if v >= 0.0 { v / penalty } else { v * penalty };
    }
}

/// Numerically-stable softmax with temperature. Subtracts max before
/// exponentiating so `exp` doesn't overflow on a wide vocab.
fn softmax_temperature(logits: &[f32], t: f32) -> Vec<f32> {
    let scale = 1.0 / t;
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter().map(|&x| ((x - max) * scale).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        for e in &mut exps {
            *e /= sum;
        }
    }
    exps
}

fn sample_multinomial(probs: &[f32], rng: &mut StdRng) -> Result<u32> {
    let r: f32 = rng.gen_range(0.0_f32..1.0_f32);
    let mut acc = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if r < acc {
            return Ok(i as u32);
        }
    }
    // Floating-point slop: every prob was 0 or the random hit the tail.
    Ok((probs.len() - 1) as u32)
}

fn sample_top_p(probs: &[f32], top_p: f32, rng: &mut StdRng) -> Result<u32> {
    // Sort indices by probability descending. We renormalize over the
    // smallest prefix whose cumulative mass crosses `top_p`.
    let mut idx: Vec<u32> = (0..probs.len() as u32).collect();
    idx.sort_unstable_by(|&a, &b| {
        probs[b as usize]
            .partial_cmp(&probs[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut cum = 0.0_f32;
    let mut cutoff = idx.len();
    for (k, &i) in idx.iter().enumerate() {
        cum += probs[i as usize];
        if cum >= top_p {
            cutoff = k + 1;
            break;
        }
    }

    let mass: f32 = idx[..cutoff].iter().map(|&i| probs[i as usize]).sum();
    if mass <= 0.0 {
        return Ok(idx[0]);
    }

    let r: f32 = rng.gen_range(0.0_f32..1.0_f32) * mass;
    let mut acc = 0.0;
    for &i in &idx[..cutoff] {
        acc += probs[i as usize];
        if r < acc {
            return Ok(i);
        }
    }
    Ok(idx[cutoff - 1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_largest_logit() {
        let logits = vec![0.1, 0.2, 5.0, -3.0, 4.99];
        let mut s = Sampler::new(0, 0.0, None);
        let cfg = SamplerCfg { repeat_penalty: 1.0, repeat_ctx: 0 };
        assert_eq!(s.sample(&logits, &[], &cfg).unwrap(), 2);
    }

    #[test]
    fn repeat_penalty_suppresses_recent_tokens() {
        let logits = vec![0.1, 0.2, 5.0, -3.0, 4.99];
        let mut s = Sampler::new(0, 0.0, None);
        let cfg = SamplerCfg { repeat_penalty: 100.0, repeat_ctx: 4 };
        let ctx = [2u32];
        assert_eq!(s.sample(&logits, &ctx, &cfg).unwrap(), 4);
    }

    #[test]
    fn repeat_penalty_inverts_for_negative_logits() {
        let logits = vec![-2.0, -0.1, -0.5];
        let mut s = Sampler::new(0, 0.0, None);
        let cfg = SamplerCfg { repeat_penalty: 10.0, repeat_ctx: 4 };
        let ctx = [1u32];
        assert_eq!(s.sample(&logits, &ctx, &cfg).unwrap(), 2);
    }

    #[test]
    fn temperature_zero_is_argmax_regardless_of_seed() {
        let logits = vec![1.0, 0.5, 0.0, 2.5, 0.1];
        let cfg = SamplerCfg { repeat_penalty: 1.0, repeat_ctx: 0 };
        for seed in 0..5 {
            let mut s = Sampler::new(seed, 0.0, Some(0.9));
            assert_eq!(s.sample(&logits, &[], &cfg).unwrap(), 3);
        }
    }

    #[test]
    fn softmax_temperature_sums_to_one() {
        let p = softmax_temperature(&[1.0, 2.0, 3.0, 4.0], 1.0);
        let total: f32 = p.iter().sum();
        assert!((total - 1.0).abs() < 1e-5, "softmax sum = {total}");
    }

    #[test]
    fn top_p_zero_falls_back_to_full_distribution() {
        let logits = vec![0.0; 32];
        let mut s = Sampler::new(42, 1.0, Some(0.0));
        let cfg = SamplerCfg { repeat_penalty: 1.0, repeat_ctx: 0 };
        let tok = s.sample(&logits, &[], &cfg).unwrap();
        assert!((tok as usize) < logits.len());
    }
}
