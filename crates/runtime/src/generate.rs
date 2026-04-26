//! Autoregressive generation loop around the layer-split runtime.
//!
//! Flow:
//!
//! 1. Encode the prompt to token ids.
//! 2. Feed the whole prompt in one forward pass (`index_pos = 0`);
//!    that warms every layer's KV cache and produces the logits for
//!    the last prompt position.
//! 3. Sample the next token, append it to the running context, call
//!    `forward` with just that one token at `index_pos = ctx_len`,
//!    sample again, repeat until EOS or `max_new_tokens`.
//!
//! The layer-split pieces (`embed / forward_range / head`) are
//! composed here; a pipelined implementation (two peers, wire
//! transport) reuses the same composition but runs the two
//! `forward_range` calls on different hosts.

use anyhow::{anyhow, Result};

use crate::model::ModelKind;
use crate::pipeline::Forwarding;
use crate::sample::{Sampler, SamplerCfg};
use crate::tokenizer::Tok;

#[derive(Clone, Debug)]
pub struct SamplingCfg {
    pub temperature:    f64,
    pub top_p:          Option<f64>,
    pub repeat_penalty: f32,
    pub repeat_ctx:     usize,
    pub seed:           u64,
    pub max_new_tokens: usize,
}

impl Default for SamplingCfg {
    fn default() -> Self {
        Self {
            temperature:    0.7,
            top_p:          Some(0.9),
            repeat_penalty: 1.1,
            repeat_ctx:     64,
            seed:           0,
            max_new_tokens: 256,
        }
    }
}

/// Wrap a user message in Qwen's chat template.
pub fn qwen_chat_prompt(user: &str, system: Option<&str>) -> String {
    let sys = system.unwrap_or("You are a helpful assistant.");
    format!(
        "<|im_start|>system\n{sys}<|im_end|>\n\
         <|im_start|>user\n{user}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

/// One turn in a chat conversation. `role` is `"system" | "user" | "assistant"`.
#[derive(Clone, Debug)]
pub struct ChatTurn<'a> {
    pub role:    &'a str,
    pub content: &'a str,
}

/// Render a multi-turn conversation into the model's native prompt
/// string. Defaults to the Qwen2 `<|im_start|>…<|im_end|>` template
/// since that's what our primary test models use.
///
/// TODO(task #14): read `tokenizer.chat_template` from the GGUF
/// metadata and render that directly — libllama stores the model's
/// native Jinja template and we're currently reinventing a coarse
/// approximation of it.
pub fn build_chat_prompt(_kind: ModelKind, turns: &[ChatTurn<'_>]) -> String {
    let mut out = String::new();
    let has_system = turns.first().map(|t| t.role == "system").unwrap_or(false);
    if !has_system {
        out.push_str("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n");
    }
    for t in turns {
        let role = match t.role {
            "system" | "user" | "assistant" => t.role,
            _ => "user",
        };
        out.push_str("<|im_start|>");
        out.push_str(role);
        out.push('\n');
        out.push_str(t.content);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

/// Greedy/sampled autoregressive generation. Calls `on_token(text)`
/// for each new decoded text chunk as it's produced. Returns the
/// total number of tokens generated.
pub fn generate<F: FnMut(&str) -> Result<()>>(
    model:  &mut dyn Forwarding,
    tok:    &Tok,
    prompt: &str,
    cfg:    &SamplingCfg,
    mut on_token: F,
) -> Result<usize> {
    model.reset_cache();

    let mut tokens = tok.encode(prompt)?;
    if tokens.is_empty() {
        return Err(anyhow!("prompt tokenized to zero tokens"));
    }

    let mut sampler = Sampler::new(cfg.seed, cfg.temperature, cfg.top_p);
    let scfg = SamplerCfg { repeat_penalty: cfg.repeat_penalty, repeat_ctx: cfg.repeat_ctx };
    let mut decoder = tok.incremental();

    // --- prompt pass ---
    let logits = model.forward(&tokens, 0)?;
    let mut next = sampler.sample(&logits.data, &tokens, &scfg)?;

    let mut n_gen = 0usize;
    loop {
        if tok.is_eos(next) || n_gen >= cfg.max_new_tokens {
            break;
        }
        tokens.push(next);
        n_gen += 1;
        if let Some(txt) = decoder.push(next)? {
            on_token(&txt)?;
        }

        let idx = tokens.len() - 1;
        let logits = model.forward(&[next], idx)?;
        next = sampler.sample(&logits.data, &tokens, &scfg)?;
    }

    Ok(n_gen)
}
