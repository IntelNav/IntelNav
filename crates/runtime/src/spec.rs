//! Speculative decoding v1 — greedy draft-and-verify.
//!
//! Each round, a cheap driver-local **draft** model proposes `k` tokens
//! autoregressively. The expensive **target** (distributed across the
//! peer chain) then verifies all `k` positions in a **single batched
//! forward**. We accept the longest prefix whose argmax matches the
//! draft; on the first mismatch we resample that position from the
//! target and discard the rest. If every draft matches, we take one
//! free bonus token from the target's last-position logits.
//!
//! Amortised cost: one chain forward (of `≤ k` tokens) produces up to
//! `k + 1` tokens whenever the draft is right. The draft's wall-clock
//! is nearly free compared to a pipelined target step, so the expected
//! speedup is the acceptance rate `α ∈ [0, 1]` × `k`.
//!
//! **Scope in this release.** Greedy only (`temperature == 0`, no
//! repeat penalty). Probabilistic accept/reject, coupled sampling, and
//! per-turn `k` adaptation all wait for v2.
//!
//! **Correctness contract**: with `temperature = 0`, `repeat_penalty =
//! 1.0`, and a draft whose tokenizer matches the target, this function
//! must emit the exact same token sequence as [`crate::chain::run_turn`]
//! for the same prompt. Any drift is a bug.
//!
//! **Compute/transfer overlap (§5.3).** The draft's "advance KV +
//! propose next round's k tokens" only depends on `last_committed` and
//! touches the draft model alone. The target's "commit chain step" only
//! touches the target + chain. Post-verify, the two run concurrently on
//! a multi-worker runtime — the draft on a `spawn_blocking` thread, the
//! chain step on the current worker — so the round's wall-clock is
//! `max(commit_roundtrip, draft_advance+propose)` instead of their sum.
//!
//! To spawn the draft side without `'static` gymnastics (the crate
//! forbids `unsafe`), `run_turn_spec` takes `draft` **by value** and
//! returns it. The blocking task moves the handle in, advances it,
//! and hands it back via its `JoinHandle`; callers rebind the returned
//! `ModelHandle` between turns.

use anyhow::{anyhow, Result};
use intelnav_ggml::Hidden;
use intelnav_wire::Phase;

use crate::chain::{front_forward, head_all_forward, head_forward, Chain};
use crate::generate::SamplingCfg;
use crate::model::ModelHandle;
use crate::tokenizer::Tok;

/// Tunable knobs for the speculative loop. Keeps the `run_turn_spec`
/// signature short and lets us grow (e.g. adaptive `k`, accept-rate
/// telemetry) without more function args.
#[derive(Clone, Debug)]
pub struct SpecCfg {
    /// Tokens the draft proposes per round. Typical range 3..8.
    /// `k = 1` collapses spec-dec into regular decode, so `< 2`
    /// is rejected at entry.
    pub k: usize,
}

impl Default for SpecCfg {
    fn default() -> Self {
        Self { k: 4 }
    }
}

/// End-to-end turn with speculative decoding over `chain`.
///
/// * `target`  — the chain-fronting model. Must be `Pipelined`.
/// * `tok`     — target's tokenizer (also used to decode the output).
///               Must match the draft's tokenizer.
/// * `draft`   — driver-local small model, moved in and handed back
///               in the return value. Ownership transfer is what lets
///               the overlap task be `'static` without unsafe.
/// * `chain`   — live peer chain over `target`.
/// * `cfg`     — sampling config. v1 requires greedy + no repeat penalty.
/// * `spec`    — spec-dec knobs (k).
///
/// Returns `(n_gen, draft)` — the number of **committed** tokens
/// (not including the prompt) and the draft handle with its KV cache
/// advanced to match the final committed state.
pub async fn run_turn_spec<F>(
    target:  &mut ModelHandle,
    tok:     &Tok,
    mut draft: ModelHandle,
    chain:   &mut Chain,
    prompt:  &str,
    cfg:     &SamplingCfg,
    spec:    &SpecCfg,
    mut on_token: F,
) -> Result<(usize, ModelHandle)>
where
    F: FnMut(&str) -> Result<()>,
{
    if spec.k < 2 {
        return Err(anyhow!("spec-dec needs k >= 2, got {}", spec.k));
    }
    if cfg.temperature != 0.0 {
        return Err(anyhow!(
            "spec-dec v1 is greedy-only (temperature must be 0.0, got {})",
            cfg.temperature
        ));
    }
    if cfg.repeat_penalty > 1.0 {
        return Err(anyhow!(
            "spec-dec v1 does not apply a repeat penalty; set repeat_penalty = 1.0"
        ));
    }

    let k = spec.k;
    let front_end = chain.front_range().1;

    // Reset both caches: target (front slice + every peer via the
    // chain's SessionInit) and draft.
    target.reset_cache();
    draft.reset_cache();

    let mut tokens = tok.encode(prompt)?;
    if tokens.is_empty() {
        return Err(anyhow!("prompt tokenized to zero tokens"));
    }
    let mut decoder = tok.incremental();

    // --- Prefill both models in parallel in terms of sequence state ---
    //
    // Target goes through the chain (front slice + peers + head).
    // Draft runs locally in full. After this block, both models have
    // KV cache of length `target_pos` and we hold the logits they
    // would produce for position `target_pos`.
    let (mut target_pending, draft_pending_initial, mut target_pos) = {
        let prompt_len = tokens.len();

        let front = front_forward(target, &tokens, 0, front_end)?;
        let tail = chain.step(front, Phase::Prefill).await
            .map_err(|e| anyhow!("{e}"))?;
        let tgt_pending = head_forward(target, &tail)?;

        // `ModelHandle::forward` returns last-position logits already
        // (shape `[batch, vocab]`) — same shape as head_forward.
        let d_logits = draft.forward(&tokens, 0)?;
        (tgt_pending, d_logits, prompt_len)
    };

    // Round-0 drafts: propose k tokens off the prefill's draft logits.
    let mut drafts_for_round = propose_k_drafts(
        &mut draft, draft_pending_initial, k, target_pos,
    )?;

    let mut n_gen = 0usize;
    'outer: loop {
        if n_gen >= cfg.max_new_tokens {
            break;
        }

        // ---- 1. Target verify: one batched forward of k tokens --
        let drafts: Vec<u32> = drafts_for_round.clone();
        let verify_front = front_forward(target, &drafts, target_pos, front_end)?;
        let verify_tail = chain.step(verify_front, Phase::Decode).await
            .map_err(|e| anyhow!("{e}"))?;
        // `[1, k, vocab]`
        let verify_logits = head_all_forward(target, &verify_tail)?;

        // ---- 2. Match prefix ------------------------------------
        //   pred[0]     = target_pending (prior to this round)
        //   pred[i>=1]  = verify_logits[.., i-1, ..]
        //
        // pred[i] predicts token at position target_pos+i, conditioned
        // on d_0 .. d_{i-1}. The draft's d_i matches if argmax(pred[i])
        // == d_i.
        let mut j = 0usize;
        for i in 0..k {
            let pred = if i == 0 {
                target_pending.clone()
            } else {
                verify_logits.select_position(i - 1)?
            };
            if pred.argmax_last()? == drafts[i] {
                j += 1;
            } else {
                break;
            }
        }

        // ---- 3. Commit tokens -----------------------------------
        let mut emitted: Vec<u32> = Vec::with_capacity(k + 1);
        let last_committed: u32;
        if j == k {
            // All k accepted — plus one free bonus from the last verify position.
            let bonus_logits = verify_logits.select_position(k - 1)?;
            let bonus = bonus_logits.argmax_last()?;
            emitted.extend_from_slice(&drafts);
            emitted.push(bonus);
            last_committed = bonus;
        } else {
            // Reject d_j: resample from target's prediction for that slot.
            let pred_at_j = if j == 0 {
                target_pending.clone()
            } else {
                verify_logits.select_position(j - 1)?
            };
            let resampled = pred_at_j.argmax_last()?;
            emitted.extend_from_slice(&drafts[..j]);
            emitted.push(resampled);
            last_committed = resampled;
        }

        // Respect max_new_tokens / EOS. Trim and emit in lock-step.
        let mut stop = false;
        for &t in &emitted {
            if n_gen >= cfg.max_new_tokens {
                stop = true;
                break;
            }
            tokens.push(t);
            n_gen += 1;
            if let Some(txt) = decoder.push(t)? {
                on_token(&txt)?;
            }
            if tok.is_eos(t) {
                stop = true;
                break;
            }
        }
        if stop {
            break 'outer;
        }

        // ---- 4. Overlap commit chain step with draft advance + propose --
        //
        // Both halves start from the same `last_committed` but touch
        // disjoint state: the commit side owns `target` + `chain`, the
        // draft side owns `draft`. We spawn the draft work onto the
        // blocking pool and drive the commit asynchronously on the
        // current worker; they complete in parallel.
        let target_pos_old = target_pos;
        let new_pos = target_pos_old + j;
        truncate_front_kv(target, new_pos)?;
        let commit_ids = [last_committed];
        let commit_front = front_forward(target, &commit_ids, new_pos, front_end)?;

        // Spawn the draft advance+propose on the blocking pool. Move
        // the handle in; receive it back via the JoinHandle so the
        // outer function can keep using it across iterations. No
        // aliasing — we're the sole owner until `draft_task.await`.
        let drafts_prev = drafts.clone();
        let k_c = k;
        let j_c = j;
        let last_c = last_committed;
        let draft_task = tokio::task::spawn_blocking(
            move || -> Result<(ModelHandle, Vec<u32>)> {
                let next = advance_and_propose(
                    &mut draft, j_c, k_c, &drafts_prev, last_c, target_pos_old,
                )?;
                Ok((draft, next))
            }
        );

        // Concurrently: advance target KV through the chain + head forward.
        // Chain.step_with_truncate rolls back each peer's KV to `new_pos`
        // (discarding rejected draft tokens) then appends last_committed.
        let commit_tail = chain
            .step_with_truncate(commit_front, Phase::Decode, Some(new_pos as u32))
            .await
            .map_err(|e| anyhow!("{e}"))?;
        let target_pending_new = head_forward(target, &commit_tail)?;

        // Join the draft task. If it panicked, surface as an error so
        // the outer stream doesn't silently hang on the next iteration.
        let (draft_back, drafts_next) = match draft_task.await {
            Ok(Ok(pair)) => pair,
            Ok(Err(e))   => return Err(e),
            Err(join_e)  => return Err(anyhow!("draft task failed: {join_e}")),
        };
        draft = draft_back;

        // Loop state update: both caches now cover position 0..target_pos,
        // the draft has already pre-proposed the next round's k tokens.
        target_pending = target_pending_new;
        target_pos = new_pos + 1;
        drafts_for_round = drafts_next;
    }

    Ok((n_gen, draft))
}

/// Autoregressive propose: starting from `d_cur_start` (logits for
/// position `target_pos`), emit `k` draft tokens and leave the draft
/// KV cache covering positions `0..target_pos + k - 1`.
///
/// We intentionally skip the final forward (after `d_{k-1}`) — those
/// logits are never consumed; the next round's first argmax comes off
/// the commit-forward's output, not off the propose chain.
fn propose_k_drafts(
    draft: &mut ModelHandle,
    d_cur_start: Hidden,
    k: usize,
    target_pos: usize,
) -> Result<Vec<u32>> {
    let mut drafts: Vec<u32> = Vec::with_capacity(k);
    let mut d_cur = d_cur_start;
    for i in 0..k {
        let d_i = d_cur.argmax_last()?;
        drafts.push(d_i);
        if i + 1 < k {
            d_cur = draft.forward(&[d_i], target_pos + i)?;
        }
    }
    Ok(drafts)
}

/// Draft-side round transition: roll draft KV to match the target's
/// committed state, then pre-compute the next round's `k` drafts.
///
/// Entering this function, the draft KV covers `0..target_pos_old +
/// k - 1` (from the previous round's propose). The target has just
/// committed `j` accepted drafts plus `last_committed`, so both models
/// now agree up to position `target_pos_old + j + 1` (or
/// `target_pos_old + k + 1` on full accept).
///
/// We:
///   1. advance the draft to that same position by feeding the
///      not-yet-seen committed tokens, then
///   2. call `propose_k_drafts` from the resulting logits.
fn advance_and_propose(
    draft: &mut ModelHandle,
    j: usize,
    k: usize,
    drafts_prev: &[u32],
    last_committed: u32,
    target_pos_old: usize,
) -> Result<Vec<u32>> {
    let (d_cur, target_pos_new) = if j == k {
        // Full accept. Draft KV has d_0..d_{k-2}; feed d_{k-1} and
        // last_committed in one batched forward to bring KV to
        // `target_pos_old + k + 1`.
        let logits = draft.forward(
            &[drafts_prev[k - 1], last_committed],
            target_pos_old + k - 1,
        )?;
        (logits, target_pos_old + k + 1)
    } else {
        // Reject at j. Roll draft KV back to target_pos_old + j, then
        // feed last_committed at that position → logits for
        // target_pos_old + j + 1.
        truncate_draft_kv(draft, target_pos_old + j)?;
        let logits = draft.forward(&[last_committed], target_pos_old + j)?;
        (logits, target_pos_old + j + 1)
    };
    propose_k_drafts(draft, d_cur, k, target_pos_new)
}

/// Truncate the front slice of the target model's KV cache. Peers
/// handle their own slice via `kv_truncate_to` on `ForwardHidden`.
fn truncate_front_kv(model: &mut ModelHandle, keep: usize) -> Result<()> {
    let pl = model.pipelined().ok_or_else(|| anyhow!(
        "spec-dec requires a pipelined model for the target (qwen2 today)"
    ))?;
    pl.truncate_kv_to(keep)
}

/// Truncate the (local, non-pipelined) draft model's KV cache.
///
/// We need a Pipelined draft too in v1 because `truncate_kv_to` is
/// only exposed on that trait. This is fine since the draft has to be
/// a small Qwen2 anyway to share the tokenizer — the set of models
/// that satisfy "Qwen2 vocab AND tiny enough to run alongside the
/// target" is narrow.
fn truncate_draft_kv(model: &mut ModelHandle, keep: usize) -> Result<()> {
    let pl = model.pipelined().ok_or_else(|| anyhow!(
        "spec-dec requires a pipelined model for the draft (qwen2 today)"
    ))?;
    pl.truncate_kv_to(keep)
}
