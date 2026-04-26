//! Streaming chat-completion client talking to the local gateway.
//!
//! Emits a stream of [`Delta`]s for the caller to render.

use anyhow::{Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

#[derive(Clone, Debug, Serialize)]
pub struct ChatMessage {
    pub role:    String,
    pub content: String,
}

#[derive(Debug)]
pub enum Delta {
    /// A fragment of assistant content.
    Token(String),
    /// Terminal — the assistant finished normally.
    Done,
    /// Terminal — transport / upstream error.
    Error(String),
}

#[derive(Clone, Debug)]
pub struct ChatRequest {
    pub gateway:   String,
    pub model:     String,
    pub messages:  Vec<ChatMessage>,
    pub quorum:    Option<u8>,
    pub allow_wan: bool,
}

/// Spawn the HTTP request + SSE parser; returns a receiver of deltas.
pub fn stream(req: ChatRequest) -> mpsc::UnboundedReceiver<Delta> {
    let (tx, rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        if let Err(e) = run_stream(req, tx.clone()).await {
            let _ = tx.send(Delta::Error(e.to_string()));
        }
    });
    rx
}

async fn run_stream(req: ChatRequest, tx: mpsc::UnboundedSender<Delta>) -> Result<()> {
    let url = format!("{}/v1/chat/completions", req.gateway.trim_end_matches('/'));
    let mut body = serde_json::json!({
        "model":       req.model,
        "messages":    req.messages,
        "stream":      true,
        "temperature": 0.2,
    });
    let mut ext = serde_json::Map::new();
    if let Some(q) = req.quorum   { ext.insert("quorum".into(),    serde_json::json!(q)); }
    if req.allow_wan              { ext.insert("allow_wan".into(), serde_json::json!(true)); }
    if !ext.is_empty() {
        body["intelnav"] = serde_json::Value::Object(ext);
    }

    let client = reqwest::Client::builder()
        .user_agent(concat!("intelnav-cli/", env!("CARGO_PKG_VERSION")))
        .build()
        .context("http client")?;

    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .with_context(|| format!("POST {url}"))?;

    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("gateway {status}: {text}");
    }

    let mut stream = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::with_capacity(4096);

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("upstream stream")?;
        buf.extend_from_slice(&chunk);

        while let Some(end) = find_event_boundary(&buf) {
            let raw: Vec<u8> = buf.drain(..end).collect();
            let text = String::from_utf8_lossy(&raw);
            for line in text.lines() {
                let Some(payload) = line.strip_prefix("data:") else { continue };
                let payload = payload.trim();
                if payload.is_empty() { continue; }
                if payload == "[DONE]" {
                    let _ = tx.send(Delta::Done);
                    return Ok(());
                }
                match serde_json::from_str::<StreamChunk>(payload) {
                    Ok(sc) => {
                        for choice in sc.choices {
                            if let Some(content) = choice.delta.content {
                                if !content.is_empty() {
                                    let _ = tx.send(Delta::Token(content));
                                }
                            }
                            if choice.finish_reason.is_some() {
                                let _ = tx.send(Delta::Done);
                                return Ok(());
                            }
                        }
                    }
                    Err(e) => {
                        tracing::debug!(?e, %payload, "sse decode skipped");
                    }
                }
            }
        }
    }
    let _ = tx.send(Delta::Done);
    Ok(())
}

fn find_event_boundary(buf: &[u8]) -> Option<usize> {
    let mut prev = 0u8;
    for (i, &b) in buf.iter().enumerate() {
        if prev == b'\n' && b == b'\n' {
            return Some(i + 1);
        }
        prev = b;
    }
    None
}

#[derive(Deserialize)]
struct StreamChunk {
    choices: Vec<Choice>,
}
#[derive(Deserialize)]
struct Choice {
    delta:         DeltaField,
    #[serde(default)]
    finish_reason: Option<String>,
}
#[derive(Deserialize, Default)]
struct DeltaField {
    #[serde(default)]
    content: Option<String>,
}
