//! Interactive chat REPL — the Claude-Code lookalike.
//!
//! Layout:
//!
//! ```text
//!  ┌ model · tier · quorum · peers ────────────────────────── session ┐
//!  │                                                                  │
//!  │  conversation (scrollable)                                       │
//!  │                                                                  │
//!  └──────────────────────────────────────────────────────────────────┘
//!  ┌ prompt ──────────────────────────────────────────────────────────┐
//!  │ > _                                                              │
//!  └──────────────────────────────────────────────────────────────────┘
//!   status bar — shimmer during streaming, static otherwise
//! ```
//!
//! Slash commands: `/help`, `/model <id>`, `/clear`, `/quit`, `/peers`,
//! `/doctor`, `/mode`, `/models`, `/quorum`, `/tier`.

use std::io;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{
    self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Terminal;
use tokio::sync::mpsc;

use intelnav_core::{Config, RunMode};
use intelnav_runtime::{DevicePref, Probe, SamplingCfg};
use unicode_width::UnicodeWidthChar;

use crate::banner::tagline;
use crate::browser::{self, BrowserAction, BrowserState, RowKind};
use crate::catalog::CatalogEntry;
use crate::chain_driver::{ChainDriver, ChainTarget, DraftTarget};
use crate::chat::{self, ChatMessage, ChatRequest, Delta};
use crate::download::{self, Event as DlEvent};
use crate::local::{self, human_bytes, LocalDriver, LocalModel};
use crate::shimmer;
use crate::slash::{self, SlashCmd};
use crate::theme;

pub async fn run(
    config: &Config,
    mode: RunMode,
    model: Option<String>,
    quorum: Option<u8>,
    allow_wan: bool,
) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableBracketedPaste)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let local_scan = local::list_models(&config.models_dir);
    let model_name = model
        .or_else(|| {
            if mode == RunMode::Local {
                local::pick_default(&local_scan).map(|m| m.name.clone())
            } else {
                None
            }
        })
        .unwrap_or_else(|| config.default_model.clone());

    let device_pref: DevicePref = config.device.parse().unwrap_or(DevicePref::Auto);
    let local_driver = LocalDriver::new(device_pref);
    let chain_driver = ChainDriver::new(device_pref);
    let initial_chain = ChainTarget::from_config(&config.peers, &config.splits).ok();
    if let Some(t) = initial_chain.as_ref() {
        chain_driver.set_target(Some(t.clone()));
    }
    if let (Some(path), k) = (config.draft_model.clone(), config.spec_k) {
        if k >= 2 {
            chain_driver.set_draft(Some(DraftTarget { path, k: k as usize }));
        }
    }
    {
        let (dtype, _name) = crate::chain_driver::parse_wire_dtype(&config.wire_dtype);
        chain_driver.set_wire_dtype(dtype);
    }

    let mut app = AppState::new(
        config.gateway_url.clone(),
        config.models_dir.clone(),
        mode,
        local_driver,
        chain_driver,
        local_scan,
        model_name,
        quorum.unwrap_or(config.quorum),
        allow_wan || config.allow_wan,
        config.default_tier.display().to_string(),
    );
    app.history.push(Turn::system(format!(
        "Welcome to {}. Type /help for commands. Ctrl+C to quit.",
        tagline()
    )));
    app.greet_mode();

    let result = run_loop(&mut terminal, &mut app).await;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), DisableBracketedPaste, LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    result
}

// ======================================================================
//  App state
// ======================================================================

#[derive(Clone, Debug)]
struct Turn {
    role:       Role,
    content:    String,
    /// Set once the turn is fully received (not streaming).
    complete:   bool,
}
#[derive(Clone, Copy, Debug, PartialEq)]
enum Role { System, User, Assistant }

impl Turn {
    fn user(s: impl Into<String>)      -> Self { Self { role: Role::User,      content: s.into(), complete: true  } }
    fn system(s: impl Into<String>)    -> Self { Self { role: Role::System,    content: s.into(), complete: true  } }
    fn assistant_open()                -> Self { Self { role: Role::Assistant, content: String::new(), complete: false } }
}

enum View {
    Chat,
    Browser(BrowserState),
    Doctor(DoctorView),
}

/// Preflight snapshot rendered as a dedicated screen rather than a
/// transcript dump. Captured at the moment `/doctor` was invoked so
/// the view doesn't flicker as the probe re-runs.
#[derive(Clone, Debug)]
struct DoctorView {
    runtime:    String,
    available:  Vec<String>,
    preferred:  String,
    gateway:    String,
    models_dir: String,
    model:      String,
    mode:       String,
}

#[derive(Clone, Debug)]
struct DownloadProgress {
    label: String,
    done:  u64,
    total: Option<u64>,
    bps:   f64,
}

struct AppState {
    gateway:     String,
    models_dir:  std::path::PathBuf,
    mode:        RunMode,
    model:       String,
    tier:        String,
    quorum:      u8,
    allow_wan:   bool,

    local:       LocalDriver,
    chain:       ChainDriver,
    local_scan:  Vec<LocalModel>,

    history:     Vec<Turn>,
    input:       String,
    cursor:      usize,

    /// Vertical scroll offset in *visible* rows from the top of the
    /// rendered transcript. When `follow_tail` is true, this is
    /// recomputed each frame to keep the last line pinned to the
    /// bottom; user scroll keys break that pin.
    scroll_off:  u16,
    follow_tail: bool,
    /// Last viewport height we rendered with — cached so PageUp/Down
    /// can step half-pages without having to draw first.
    last_viewport: u16,
    /// Last total wrapped-line count of the transcript; used for clamp
    /// + "am I at the bottom?" detection.
    last_total:    u16,
    /// Cached inner width of the input box — used by the ↑/↓ handlers
    /// to decide if the cursor is on the first/last visual row
    /// without needing access to the current frame.
    last_input_inner_w: u16,
    /// Currently-highlighted suggestion in the slash overlay. Reset
    /// whenever the suggestion list changes.
    sugg_idx: usize,

    view:        View,

    /// Prior submitted prompts; ↑/↓ when cursor is at the top/bottom
    /// row of the input box walks this list.
    prompt_history: Vec<String>,
    /// Current position in `prompt_history` while navigating. `None`
    /// means "editing a fresh prompt, not in history".
    history_idx:    Option<usize>,
    /// Paste placeholders: full pasted text keyed by id. Inline
    /// references like `[#pasted-1]` in the input get re-inflated at
    /// submit time.
    paste_stash:  std::collections::HashMap<u32, String>,
    next_paste_id: u32,

    streaming:   Option<mpsc::UnboundedReceiver<Delta>>,
    download:    Option<mpsc::UnboundedReceiver<DlEvent>>,
    dl_progress: Option<DownloadProgress>,
    start:       Instant,

    /// Set by `/quit` — the main loop reads this each tick and exits
    /// cleanly so terminal teardown runs in `run()`.
    should_quit: bool,

    /// First Ctrl+C arms a 1.5 s exit confirmation; a second Ctrl+C
    /// within the window exits. Resets when the user types anything
    /// else, or when the window expires on the main-loop tick.
    exit_pending: Option<Instant>,
}

impl AppState {
    #[allow(clippy::too_many_arguments)]
    fn new(
        gateway:    String,
        models_dir: std::path::PathBuf,
        mode:       RunMode,
        local:      LocalDriver,
        chain:      ChainDriver,
        local_scan: Vec<LocalModel>,
        model:      String,
        quorum:     u8,
        allow_wan:  bool,
        tier:       String,
    ) -> Self {
        Self {
            gateway, models_dir, mode, model, tier, quorum, allow_wan,
            local, chain, local_scan,
            history: Vec::new(),
            input:   String::new(),
            cursor:  0,
            scroll_off:    0,
            follow_tail:   true,
            last_viewport: 0,
            last_total:    0,
            last_input_inner_w: 0,
            sugg_idx: 0,
            view:    View::Chat,
            prompt_history: Vec::new(),
            history_idx:    None,
            paste_stash:    std::collections::HashMap::new(),
            next_paste_id:  1,
            streaming: None,
            download:    None,
            dl_progress: None,
            start:       Instant::now(),
            should_quit: false,
            exit_pending: None,
        }
    }

    fn is_streaming(&self) -> bool { self.streaming.is_some() }
    fn is_downloading(&self) -> bool { self.download.is_some() }
    fn in_browser(&self) -> bool { matches!(self.view, View::Browser(_)) }
    fn in_doctor(&self)  -> bool { matches!(self.view, View::Doctor(_)) }
    fn in_overlay(&self) -> bool { self.in_browser() || self.in_doctor() }

    fn backend_label(&self) -> String {
        match self.mode {
            RunMode::Local   => "local".into(),
            RunMode::Network => "network".into(),
            RunMode::Auto    => "auto".into(),
        }
    }

    fn greet_mode(&mut self) {
        match self.mode {
            RunMode::Local => {
                let usable: Vec<_> = self.local_scan.iter().filter(|m| m.is_usable()).collect();
                if usable.is_empty() {
                    self.history.push(Turn::system(format!(
                        "local mode — no usable models in {}. Drop a .gguf (+ tokenizer.json) there, or /mode network.",
                        self.models_dir.display()
                    )));
                } else {
                    self.history.push(Turn::system(format!(
                        "local mode — {} models in {}. Current: {}",
                        usable.len(), self.models_dir.display(), self.model,
                    )));
                }
            }
            RunMode::Network => {
                self.history.push(Turn::system(format!(
                    "network mode — gateway {}. /mode local to switch offline.",
                    self.gateway,
                )));
            }
            RunMode::Auto => {
                self.history.push(Turn::system(
                    "auto mode — first turn will pick local or network.",
                ));
            }
        }
    }

    fn insert_paste(&mut self, pasted: String) {
        const THRESHOLD: usize  = 10_000;
        const KEEP_HEAD: usize  = 500;
        const KEEP_TAIL: usize  = 500;

        let body = if pasted.chars().count() > THRESHOLD {
            let head: String = pasted.chars().take(KEEP_HEAD).collect();
            let tail: String = pasted.chars().rev().take(KEEP_TAIL).collect::<String>()
                .chars().rev().collect();
            let middle: String = pasted.chars()
                .skip(KEEP_HEAD)
                .take(pasted.chars().count().saturating_sub(KEEP_HEAD + KEEP_TAIL))
                .collect();
            let id = self.next_paste_id;
            self.next_paste_id += 1;
            let lines = middle.matches('\n').count() + 1;
            self.paste_stash.insert(id, middle);
            format!("{head}[#pasted-{id} +{lines} lines]{tail}")
        } else {
            pasted
        };

        // Insert at cursor in one shot — no per-char events, no re-renders
        // between characters.
        self.input.insert_str(self.cursor, &body);
        self.cursor += body.len();
        self.history_idx = None;
    }

    /// Re-inflate `[#pasted-N ...]` placeholders back into full text at
    /// submit time so the model sees the whole payload.
    fn inflate_paste_refs(&self, text: &str) -> String {
        if self.paste_stash.is_empty() { return text.to_string(); }
        let mut out = String::with_capacity(text.len());
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'[' && text[i..].starts_with("[#pasted-") {
                if let Some(end) = text[i..].find(']') {
                    let token = &text[i+9..i+end]; // "N +K lines"
                    let id: u32 = token.split_whitespace().next()
                        .and_then(|s| s.parse().ok()).unwrap_or(0);
                    if let Some(full) = self.paste_stash.get(&id) {
                        // middle was stored as raw text; glue it back in.
                        // Context: placeholder replaced [head..tail]; here
                        // we splice the full middle between the kept head
                        // and tail — text[i..i+end+1] is the placeholder.
                        out.push_str(full);
                        i += end + 1;
                        continue;
                    }
                }
            }
            // copy one char (utf-8 boundary-safe)
            let ch_end = text[i..].char_indices().nth(1).map(|(b,_)| i + b).unwrap_or(bytes.len());
            out.push_str(&text[i..ch_end]);
            i = ch_end;
        }
        out
    }

    fn suggestions(&self) -> Vec<&'static SlashCmd> {
        slash::suggest(&self.input)
    }
    fn accept_suggestion(&mut self) {
        let s = self.suggestions();
        if s.is_empty() { return; }
        let cmd = s[self.sugg_idx.min(s.len() - 1)];
        self.input = format!("/{}", cmd.name);
        if !cmd.args.is_empty() {
            self.input.push(' ');
        }
        self.cursor = self.input.len();
        self.sugg_idx = 0;
    }

    fn cursor_on_first_input_row(&self) -> bool {
        // If the viewport is unknown yet (first frame), treat as single
        // line and allow history nav.
        let w = self.last_input_inner_w.max(4) as usize;
        let (row, _) = cursor_visual_pos(&self.input, self.cursor, w);
        row == 0
    }
    fn cursor_on_last_input_row(&self) -> bool {
        let w = self.last_input_inner_w.max(4) as usize;
        let (row, _) = cursor_visual_pos(&self.input, self.cursor, w);
        let rows = input_visual_rows(&self.input, w).len() as u16;
        row + 1 >= rows
    }

    fn history_up(&mut self) {
        if self.prompt_history.is_empty() { return; }
        let next = match self.history_idx {
            None    => self.prompt_history.len() - 1,
            Some(0) => 0,
            Some(n) => n - 1,
        };
        self.history_idx = Some(next);
        self.input = self.prompt_history[next].clone();
        self.cursor = self.input.len();
    }

    fn history_down(&mut self) {
        let Some(cur) = self.history_idx else { return; };
        if cur + 1 >= self.prompt_history.len() {
            self.history_idx = None;
            self.input.clear();
            self.cursor = 0;
        } else {
            self.history_idx = Some(cur + 1);
            self.input = self.prompt_history[cur + 1].clone();
            self.cursor = self.input.len();
        }
    }

    fn submit(&mut self) {
        if self.in_overlay() { return; }
        let text = std::mem::take(&mut self.input);
        self.cursor = 0;
        self.history_idx = None;
        let text = text.trim().to_string();
        if text.is_empty() { return; }
        if !text.starts_with('/') && self.prompt_history.last() != Some(&text) {
            self.prompt_history.push(text.clone());
        }
        let text = self.inflate_paste_refs(&text);

        if let Some(cmd) = text.strip_prefix('/') {
            self.handle_slash(cmd);
            return;
        }

        self.history.push(Turn::user(text.clone()));
        self.history.push(Turn::assistant_open());
        self.follow_tail = true;

        let mut messages: Vec<ChatMessage> = self
            .history
            .iter()
            .filter(|t| t.role != Role::System)
            .filter(|t| !(t.role == Role::Assistant && !t.complete && t.content.is_empty()))
            .map(|t| ChatMessage {
                role: match t.role { Role::User => "user", Role::Assistant => "assistant", Role::System => "system" }.into(),
                content: t.content.clone(),
            })
            .collect();
        if matches!(messages.last(), Some(m) if m.role == "assistant" && m.content.is_empty()) {
            messages.pop();
        }

        let rx = match self.mode {
            RunMode::Local => {
                let Some(m) = local::resolve(&self.local_scan, &self.model) else {
                    self.emit_fatal(format!(
                        "no local model matches `{}`. /models to list. Drop a .gguf into {}",
                        self.model, self.models_dir.display()
                    ));
                    return;
                };
                if !m.is_usable() {
                    self.emit_fatal(m.status_line());
                    return;
                }
                let cfg = SamplingCfg::default();
                self.local.stream(m, messages, cfg)
            }
            // Network mode with a peer chain configured runs the turn
            // locally-front-half + chain-tail instead of the gateway.
            RunMode::Network if self.chain.target().is_some() => {
                let Some(m) = local::resolve(&self.local_scan, &self.model) else {
                    self.emit_fatal(format!(
                        "peer chain needs the same GGUF locally for the front slice; \
                         no match for `{}` in {}. /models to list.",
                        self.model, self.models_dir.display()
                    ));
                    return;
                };
                if !m.is_usable() {
                    self.emit_fatal(m.status_line());
                    return;
                }
                let cfg = SamplingCfg::default();
                self.chain.stream(m, messages, cfg)
            }
            RunMode::Network | RunMode::Auto => chat::stream(ChatRequest {
                gateway:   self.gateway.clone(),
                model:     self.model.clone(),
                messages,
                quorum:    Some(self.quorum),
                allow_wan: self.allow_wan,
            }),
        };
        self.streaming = Some(rx);
    }

    fn emit_fatal(&mut self, msg: String) {
        if let Some(last) = self.history.last_mut() {
            if last.role == Role::Assistant && !last.complete {
                last.content.push_str(&msg);
                last.complete = true;
                return;
            }
        }
        self.history.push(Turn::system(msg));
    }

    fn handle_slash(&mut self, cmd: &str) {
        let mut parts = cmd.split_whitespace();
        let head = parts.next().unwrap_or("");
        match head {
            "help" => {
                let mut msg = String::from("commands:\n");
                for c in slash::COMMANDS {
                    if c.args.is_empty() {
                        msg.push_str(&format!("  /{:<8}  {}\n", c.name, c.help));
                    } else {
                        msg.push_str(&format!("  /{:<8} {:<20}  {}\n", c.name, c.args, c.help));
                    }
                }
                self.history.push(Turn::system(msg.trim_end().to_string()));
            }
            "clear" => self.history.clear(),

            "mode" => match parts.next() {
                Some(m) => match m.parse::<RunMode>() {
                    Ok(new_mode) => {
                        self.mode = new_mode;
                        self.history.push(Turn::system(format!("mode → {}", self.backend_label())));
                    }
                    Err(e) => self.history.push(Turn::system(format!("{e}"))),
                },
                None => self.history.push(Turn::system(format!("current mode: {}", self.backend_label()))),
            },

            "model" => match parts.next() {
                Some(m) => {
                    self.model = m.to_string();
                    self.history.push(Turn::system(format!("model → {m}")));
                }
                None => self.history.push(Turn::system(format!("current model: {}", self.model))),
            },

            "models" => self.open_browser(),

            "doctor" => {
                let p = Probe::collect();
                self.view = View::Doctor(DoctorView {
                    runtime:    p.summary,
                    available:  p.backends.available.iter().map(|s| s.to_string()).collect(),
                    preferred:  p.backends.recommended.to_string(),
                    gateway:    self.gateway.clone(),
                    models_dir: self.models_dir.display().to_string(),
                    model:      self.model.clone(),
                    mode:       self.backend_label(),
                });
            }

            "quorum" => if let Some(n) = parts.next().and_then(|v| v.parse().ok()) {
                self.quorum = n;
                self.history.push(Turn::system(format!("quorum → {n}")));
            },
            "tier" => if let Some(t) = parts.next() {
                self.tier = t.to_ascii_uppercase();
                self.history.push(Turn::system(format!("tier → {t}")));
            },
            "peers" => self.handle_peers_cmd(parts),
            "draft" => self.handle_draft_cmd(parts),
            "wire"  => self.handle_wire_cmd(parts),
            "quit" | "exit" => { self.should_quit = true; }
            other => self.history.push(Turn::system(format!("unknown command: /{other}"))),
        }
    }

    /// `/peers` — zero args: show current target. Two args: parse
    /// `host:port,...` and `split,...` and install as the new chain
    /// target. `/peers clear` removes the chain and falls back to the
    /// gateway path in Network mode.
    fn handle_peers_cmd<'a, I: Iterator<Item = &'a str>>(&mut self, mut parts: I) {
        let a = parts.next();
        let b = parts.next();
        match (a, b) {
            (None, _) => {
                let msg = match self.chain.target() {
                    Some(t) => format!("peer chain: {}", t.summary()),
                    None    => "peer chain: (not configured) — /peers a:7717,b:7717 6,12".into(),
                };
                self.history.push(Turn::system(msg));
            }
            (Some("clear"), _) => {
                self.chain.set_target(None);
                self.history.push(Turn::system("peer chain cleared — network mode will use gateway"));
            }
            (Some(peers_s), Some(splits_s)) => {
                let peers: Vec<String> = peers_s.split(',')
                    .filter(|s| !s.is_empty()).map(String::from).collect();
                let splits: Result<Vec<u16>, _> = splits_s.split(',')
                    .filter(|s| !s.is_empty()).map(|s| s.parse::<u16>()).collect();
                let splits = match splits {
                    Ok(v)  => v,
                    Err(e) => {
                        self.history.push(Turn::system(format!("bad --splits value: {e}")));
                        return;
                    }
                };
                match ChainTarget::from_config(&peers, &splits) {
                    Ok(t) => {
                        let s = t.summary();
                        self.chain.set_target(Some(t));
                        self.history.push(Turn::system(format!(
                            "peer chain → {s}\n(use /mode network to route turns through it)"
                        )));
                    }
                    Err(e) => self.history.push(Turn::system(format!("peers: {e}"))),
                }
            }
            (Some(_), None) => {
                self.history.push(Turn::system(
                    "/peers <host:port,...> <split,...> — e.g. /peers 10.0.0.4:7717,10.0.0.5:7717 8,16",
                ));
            }
        }
    }

    /// `/draft` — zero args: show. `/draft clear`: disable. `/draft
    /// <path> [k]`: enable spec-dec with that GGUF as the draft (k
    /// defaults to 4).
    fn handle_draft_cmd<'a, I: Iterator<Item = &'a str>>(&mut self, mut parts: I) {
        let a = parts.next();
        let b = parts.next();
        match a {
            None => {
                let msg = match self.chain.draft() {
                    Some(d) => format!("draft: {}", d.summary()),
                    None    => "draft: (not configured) — /draft <path.gguf> [k=4]".into(),
                };
                self.history.push(Turn::system(msg));
            }
            Some("clear") => {
                self.chain.set_draft(None);
                self.history.push(Turn::system("draft cleared — spec-dec disabled"));
            }
            Some(path_s) => {
                let path = std::path::PathBuf::from(path_s);
                if !path.is_file() {
                    self.history.push(Turn::system(format!("draft: no such file: {path_s}")));
                    return;
                }
                let k = b.and_then(|s| s.parse::<usize>().ok()).unwrap_or(4);
                if k < 2 {
                    self.history.push(Turn::system(format!("draft: k must be ≥ 2, got {k}")));
                    return;
                }
                let d = DraftTarget { path, k };
                let s = d.summary();
                self.chain.set_draft(Some(d));
                self.history.push(Turn::system(format!(
                    "draft → {s}\n(spec-dec is greedy-only in v1: temperature is ignored)"
                )));
            }
        }
    }

    /// `/wire` — zero args: show current wire dtype. One arg: `fp16`,
    /// `int8` (or aliases) to switch. Affects the chain driver only
    /// (gateway path doesn't use ForwardHidden). Takes effect on the
    /// next turn; no reconnect needed.
    fn handle_wire_cmd<'a, I: Iterator<Item = &'a str>>(&mut self, mut parts: I) {
        use intelnav_runtime::Dtype;
        match parts.next() {
            None => {
                let name = match self.chain.wire_dtype() {
                    Dtype::Fp16 => "fp16",
                    Dtype::Int8 => "int8",
                    Dtype::Bf16 => "bf16",
                };
                self.history.push(Turn::system(format!(
                    "wire dtype: {name} (switch: /wire fp16|int8)"
                )));
            }
            Some(s) => {
                let (dtype, name) = crate::chain_driver::parse_wire_dtype(s);
                self.chain.set_wire_dtype(dtype);
                self.history.push(Turn::system(format!("wire dtype → {name}")));
            }
        }
    }

    fn open_browser(&mut self) {
        self.local_scan = local::list_models(&self.models_dir);
        let probe = Probe::collect();
        let remote = fetch_remote_models(&self.gateway);
        let rows = browser::build_rows(&self.local_scan, remote.as_ref(), &probe);
        self.view = View::Browser(BrowserState::new(rows));
    }

    fn close_browser(&mut self) {
        self.view = View::Chat;
    }

    fn commit_browser(&mut self) {
        let Some(row) = (match &self.view {
            View::Browser(s) => s.current().cloned(),
            _                => None,
        }) else { return; };
        if !row.enabled { return; }

        match row.kind {
            RowKind::Local { path } => {
                let stem = path.file_stem().and_then(|s| s.to_str())
                    .unwrap_or(&row.model).to_string();
                self.mode  = RunMode::Local;
                self.model = stem.clone();
                self.history.push(Turn::system(format!("→ local · {stem}")));
            }
            RowKind::Network { .. } => {
                self.mode  = RunMode::Network;
                self.model = row.model.clone();
                self.history.push(Turn::system(format!("→ network · {}", row.model)));
            }
            RowKind::Install { entry, .. } => {
                self.start_install(entry);
            }
        }
        self.close_browser();
    }

    fn start_install(&mut self, entry: &'static CatalogEntry) {
        if self.is_downloading() {
            self.history.push(Turn::system(
                "another download is in progress — wait for it to finish.",
            ));
            return;
        }
        self.history.push(Turn::system(format!(
            "installing {} → {}", entry.display_name, self.models_dir.display(),
        )));
        let rx = download::install_entry(entry, self.models_dir.clone());
        self.download    = Some(rx);
        self.dl_progress = Some(DownloadProgress {
            label: entry.display_name.to_string(),
            done: 0, total: Some(entry.size_bytes), bps: 0.0,
        });
    }

    fn drain_download(&mut self) -> bool {
        let Some(rx) = self.download.as_mut() else { return false };
        let mut dirty = false;
        while let Ok(ev) = rx.try_recv() {
            dirty = true;
            match ev {
                DlEvent::Progress { label, done, total, bps } => {
                    self.dl_progress = Some(DownloadProgress { label, done, total, bps });
                }
                DlEvent::Done { label, path } => {
                    let stem = path.file_stem().and_then(|s| s.to_str())
                        .unwrap_or(&label).to_string();
                    self.history.push(Turn::system(format!(
                        "✓ installed {label} — ready to use as `{stem}`",
                    )));
                    self.local_scan  = local::list_models(&self.models_dir);
                    self.mode        = RunMode::Local;
                    self.model       = stem;
                    self.download    = None;
                    self.dl_progress = None;
                    break;
                }
                DlEvent::Error { label, message } => {
                    self.history.push(Turn::system(format!("⚠ {label}: {message}")));
                    self.download    = None;
                    self.dl_progress = None;
                    break;
                }
            }
        }
        dirty
    }

    fn drain_stream(&mut self) -> bool {
        let Some(rx) = self.streaming.as_mut() else { return false };
        let mut dirty = false;
        while let Ok(delta) = rx.try_recv() {
            dirty = true;
            match delta {
                Delta::Token(t) => {
                    if let Some(last) = self.history.last_mut() {
                        if last.role == Role::Assistant && !last.complete {
                            last.content.push_str(&t);
                        }
                    }
                }
                Delta::Done => {
                    if let Some(last) = self.history.last_mut() {
                        if last.role == Role::Assistant { last.complete = true; }
                    }
                    self.streaming = None;
                    break;
                }
                Delta::Error(e) => {
                    if let Some(last) = self.history.last_mut() {
                        if last.role == Role::Assistant && !last.complete {
                            last.content.push_str(&format!("\n⚠ {e}"));
                            last.complete = true;
                        }
                    } else {
                        self.history.push(Turn::system(format!("error: {e}")));
                    }
                    self.streaming = None;
                    break;
                }
            }
        }
        dirty
    }
}

// ======================================================================
//  Main loop
// ======================================================================

async fn run_loop<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    app: &mut AppState,
) -> Result<()> {
    let tick_rate = Duration::from_millis(80);
    let mut last_tick = Instant::now();

    loop {
        terminal.draw(|f| draw(f, app))?;
        let _ = app.drain_stream();
        let _ = app.drain_download();
        // Expire an armed Ctrl+C window so the toast clears after 1.5s.
        if let Some(t) = app.exit_pending {
            if t.elapsed() >= Duration::from_millis(1500) { app.exit_pending = None; }
        }
        if app.should_quit { break; }

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_millis(0));

        if event::poll(timeout)? {
            match event::read()? {
                Event::Key(k) if k.kind == KeyEventKind::Press => {
                    if handle_key(app, k) { break; }
                }
                Event::Paste(s) => {
                    if !app.in_browser() { app.insert_paste(s); }
                }
                _ => {}
            }
        }

        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
        }
    }
    Ok(())
}

/// Returns `true` if the app should quit.
fn handle_key(app: &mut AppState, k: KeyEvent) -> bool {
    const EXIT_WINDOW: Duration = Duration::from_millis(1500);

    if k.modifiers.contains(KeyModifiers::CONTROL) {
        match k.code {
            KeyCode::Char('c') => {
                // First press: cancel active stream / clear input /
                // dismiss overlay. Second press within 1.5s: exit.
                let armed = app.exit_pending
                    .map(|t| t.elapsed() < EXIT_WINDOW)
                    .unwrap_or(false);
                if armed { return true; }
                if app.is_streaming() {
                    app.streaming = None;
                    if let Some(t) = app.history.last_mut() { t.complete = true; }
                } else if app.in_overlay() {
                    app.view = View::Chat;
                } else if !app.input.is_empty() {
                    app.input.clear();
                    app.cursor = 0;
                    app.sugg_idx = 0;
                    app.history_idx = None;
                }
                app.exit_pending = Some(Instant::now());
                return false;
            }
            KeyCode::Char('d') => {
                // Ctrl+D exits only when the input is empty (shell-like).
                if app.input.is_empty() && !app.in_overlay() { return true; }
                return false;
            }
            KeyCode::Char('l') => { app.history.clear(); return false; }
            _ => {}
        }
    }

    // Any other keystroke disarms a pending exit.
    app.exit_pending = None;

    if app.in_doctor() {
        // Doctor overlay: Esc / q / Enter close.
        match k.code {
            KeyCode::Esc | KeyCode::Enter | KeyCode::Char('q') => {
                app.view = View::Chat;
            }
            _ => {}
        }
        return false;
    }

    if app.in_browser() {
        if let View::Browser(state) = &mut app.view {
            match browser::handle_key(state, k) {
                BrowserAction::Commit      => app.commit_browser(),
                BrowserAction::Close       => app.close_browser(),
                BrowserAction::Consumed    => {}
                BrowserAction::Passthrough => {}
            }
        }
        return false;
    }

    // When the slash-autocomplete overlay is visible, it captures
    // ↑/↓/Tab/Enter/Esc. Everything else (typing) passes through and
    // will continue to update the suggestion list on the next frame.
    let suggs = app.suggestions();
    if !suggs.is_empty() {
        match k.code {
            KeyCode::Up   => {
                app.sugg_idx = if app.sugg_idx == 0 { suggs.len() - 1 } else { app.sugg_idx - 1 };
                return false;
            }
            KeyCode::Down => {
                app.sugg_idx = (app.sugg_idx + 1) % suggs.len();
                return false;
            }
            KeyCode::Tab => { app.accept_suggestion(); return false; }
            KeyCode::Enter => {
                // If the user has typed exactly one command's name and
                // pressed Enter, run it directly. Otherwise accept the
                // highlighted suggestion first.
                let raw = app.input.trim_start_matches('/').to_string();
                if suggs.iter().any(|c| c.name == raw) {
                    app.submit();
                } else {
                    app.accept_suggestion();
                }
                return false;
            }
            KeyCode::Esc  => {
                app.input.clear();
                app.cursor = 0;
                app.sugg_idx = 0;
                return false;
            }
            _ => {}
        }
    }

    match k.code {
        KeyCode::Enter => {
            if k.modifiers.contains(KeyModifiers::SHIFT) {
                app.input.insert(app.cursor, '\n');
                app.cursor += 1;
            } else {
                app.submit();
            }
        }
        KeyCode::Backspace => {
            if app.cursor > 0 {
                let prev = prev_char_boundary(&app.input, app.cursor);
                app.input.drain(prev..app.cursor);
                app.cursor = prev;
            }
        }
        KeyCode::Delete => {
            if app.cursor < app.input.len() {
                let next = next_char_boundary(&app.input, app.cursor);
                app.input.drain(app.cursor..next);
            }
        }
        KeyCode::Left  => app.cursor = prev_char_boundary(&app.input, app.cursor),
        KeyCode::Right => app.cursor = next_char_boundary(&app.input, app.cursor),
        KeyCode::Up => {
            // Only walk history when the cursor is visually on the
            // first row of the input box; otherwise let it be a no-op
            // (multiline in-box navigation lives in Home/End today).
            if app.cursor_on_first_input_row() {
                app.history_up();
            }
        }
        KeyCode::Down => {
            if app.cursor_on_last_input_row() {
                app.history_down();
            }
        }
        KeyCode::Home  => {
            if k.modifiers.contains(KeyModifiers::SHIFT) {
                let (off, follow) = transcript_scroll_to_top();
                app.scroll_off  = off;
                app.follow_tail = follow;
            } else {
                let w = app.last_input_inner_w.max(4) as usize;
                app.cursor = visual_row_start(&app.input, app.cursor, w);
            }
        }
        KeyCode::End   => {
            if k.modifiers.contains(KeyModifiers::SHIFT) {
                let (off, follow) =
                    transcript_scroll_to_bottom(app.last_total, app.last_viewport);
                app.scroll_off  = off;
                app.follow_tail = follow;
            } else {
                let w = app.last_input_inner_w.max(4) as usize;
                app.cursor = visual_row_end(&app.input, app.cursor, w);
            }
        }
        KeyCode::PageUp => {
            app.follow_tail = false;
            let step = (app.last_viewport / 2).max(1);
            app.scroll_off = app.scroll_off.saturating_sub(step);
        }
        KeyCode::PageDown => {
            let step = (app.last_viewport / 2).max(1);
            let max_off = app.last_total.saturating_sub(app.last_viewport);
            app.scroll_off = (app.scroll_off.saturating_add(step)).min(max_off);
            if app.scroll_off == max_off { app.follow_tail = true; }
        }
        KeyCode::Char(c) => {
            app.input.insert(app.cursor, c);
            app.cursor += c.len_utf8();
        }
        KeyCode::Esc => {
            if app.is_streaming() {
                app.streaming = None;
                if let Some(t) = app.history.last_mut() { t.complete = true; }
            }
        }
        _ => {}
    }
    false
}

// ======================================================================
//  Rendering
// ======================================================================

fn draw(f: &mut ratatui::Frame<'_>, app: &mut AppState) {
    let area = f.area();
    let overlay = app.in_overlay();
    let input_h = if overlay {
        0
    } else {
        let cap = (area.height / 2).max(3);
        input_height_visual(&app.input, area.width, cap)
    };
    let suggs = if overlay { Vec::new() } else { app.suggestions() };
    // Clamp the selected suggestion against the current list length.
    if !suggs.is_empty() && app.sugg_idx >= suggs.len() {
        app.sugg_idx = 0;
    }
    let sugg_h = if suggs.is_empty() { 0 } else { suggs.len() as u16 + 2 };

    let mut constraints: Vec<Constraint> = vec![
        Constraint::Length(2),  // header (title row + bottom rule)
        Constraint::Min(4),     // transcript / overlay
    ];
    if sugg_h > 0 { constraints.push(Constraint::Length(sugg_h)); }
    if input_h > 0 { constraints.push(Constraint::Length(input_h)); }
    constraints.push(Constraint::Length(1)); // status

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    render_header(f, chunks[0], app);
    match &app.view {
        View::Browser(state) => browser::render(f, chunks[1], state),
        View::Doctor(d)      => render_doctor(f, chunks[1], d),
        View::Chat           => render_conversation(f, chunks[1], app),
    }

    let mut idx = 2;
    if sugg_h > 0 {
        render_suggestions(f, chunks[idx], &suggs, app.sugg_idx);
        idx += 1;
    }
    if input_h > 0 {
        render_input(f, chunks[idx], app);
        idx += 1;
    }
    render_status(f, chunks[idx], app);
}

fn render_doctor(f: &mut ratatui::Frame<'_>, area: Rect, d: &DoctorView) {
    let t = theme::theme();
    let label = |s: &str| Span::styled(format!("  {s:<12}"), Style::default().fg(t.subtle));
    let value = |s: String| Span::styled(s, Style::default().fg(t.text));
    let ok    = |s: String| Span::styled(s, Style::default().fg(t.success));

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::raw("  "),
        Span::styled("doctor", Style::default().fg(t.intel).add_modifier(Modifier::BOLD)),
        Span::styled(" · preflight snapshot", Style::default().fg(t.subtle)),
    ]));
    lines.push(Line::from(""));
    lines.push(Line::from(vec![label("runtime"),  ok(d.runtime.clone())]));
    lines.push(Line::from(vec![
        label("available"),
        value(if d.available.is_empty() { "(none)".into() } else { d.available.join(", ") }),
    ]));
    lines.push(Line::from(vec![
        label("preferred"),
        value(d.preferred.clone()),
    ]));
    lines.push(Line::from(""));
    lines.push(Line::from(vec![label("mode"),       value(d.mode.clone())]));
    lines.push(Line::from(vec![label("model"),      value(d.model.clone())]));
    lines.push(Line::from(vec![label("gateway"),    value(d.gateway.clone())]));
    lines.push(Line::from(vec![label("models_dir"), value(d.models_dir.clone())]));
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  Esc · ↵ · q  close",
        Style::default().fg(t.inactive),
    )));

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(t.border_inactive))
        .title(Span::styled(" /doctor ", Style::default().fg(t.subtle)));
    f.render_widget(Paragraph::new(lines).block(block), area);
}

fn render_suggestions(
    f: &mut ratatui::Frame<'_>,
    area: Rect,
    items: &[&SlashCmd],
    selected: usize,
) {
    let t = theme::theme();
    let lines: Vec<Line> = items.iter().enumerate().map(|(i, c)| {
        let is_sel = i == selected;
        let marker = if is_sel { "›" } else { " " };
        let name_style = if is_sel {
            Style::default().fg(t.intel).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(t.text)
        };
        let mut spans = vec![
            Span::styled(format!(" {marker} "),
                Style::default().fg(if is_sel { t.intel } else { t.inactive })),
            Span::styled(format!("/{:<8}", c.name), name_style),
        ];
        if !c.args.is_empty() {
            spans.push(Span::raw(" "));
            spans.push(Span::styled(c.args.to_string(),
                Style::default().fg(t.inactive)));
        }
        spans.push(Span::raw("  "));
        spans.push(Span::styled(c.help.to_string(),
            Style::default().fg(t.subtle)));
        Line::from(spans)
    }).collect();

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(t.border_inactive))
        .title(Span::styled(" commands ",
            Style::default().fg(t.subtle)));
    f.render_widget(Paragraph::new(lines).block(block), area);
}

fn render_header(f: &mut ratatui::Frame<'_>, area: Rect, app: &AppState) {
    let t = theme::theme();
    let mode_color = match app.mode {
        RunMode::Local   => t.mode_local,
        RunMode::Network => t.mode_network,
        RunMode::Auto    => t.mode_auto,
    };
    let mut spans = vec![
        Span::styled("  intelnav  ", theme::accent_bold()),
        Span::styled(format!("· {} ", app.backend_label()), Style::default().fg(mode_color)),
        Span::styled(format!("· {} ", app.model),    theme::subtle()),
        Span::styled(format!("· {} ", app.tier),     theme::subtle()),
        Span::styled(format!("· q={} ", app.quorum), theme::subtle()),
        Span::styled(
            format!("· {}wan ", if app.allow_wan { "+" } else { "-" }),
            Style::default().fg(if app.allow_wan { t.warning } else { t.inactive }),
        ),
    ];
    if let Some(tgt) = app.chain.target() {
        spans.push(Span::styled(
            format!("· {}p chain ", tgt.peers.len()),
            Style::default().fg(t.mode_network),
        ));
    }
    if let Some(d) = app.chain.draft() {
        spans.push(Span::styled(
            format!("· spec k={} ", d.k),
            Style::default().fg(t.mode_network),
        ));
    }
    let title = Line::from(spans);
    let block = Block::default().borders(Borders::BOTTOM).title(title);
    f.render_widget(block, area);
}

fn render_conversation(f: &mut ratatui::Frame<'_>, area: Rect, app: &mut AppState) {
    let t = theme::theme();
    let viewport = area.height;
    let inner_w = area.width.saturating_sub(7).max(10) as usize;

    // Animation phase, used for the streaming shimmer on the active
    // assistant turn's tail + trailing cursor glyph.
    let phase = (app.start.elapsed().as_secs_f32() / 3.2) % 1.0;

    // The open (still-streaming) turn gets the shimmer treatment — any
    // earlier turn is rendered static.
    let streaming_idx = if app.is_streaming() {
        app.history.iter().rposition(|t| t.role == Role::Assistant && !t.complete)
    } else { None };

    let mut lines: Vec<Line<'static>> = Vec::with_capacity(app.history.len() * 3);
    for (turn_i, turn) in app.history.iter().enumerate() {
        let (tag, role_key) = match turn.role {
            Role::User      => ("you  ", theme::Role::You),
            Role::Assistant => ("intel", theme::Role::Intel),
            Role::System    => ("sys  ", theme::Role::System),
        };
        let tag_style  = theme::role(role_key);
        let body_style = theme::body(role_key);
        let shimmer_here = Some(turn_i) == streaming_idx;

        // Render fenced code blocks with a dimmed left gutter; prose
        // between fences wraps normally.
        let segments = split_code_fences(&turn.content);

        let mut first = true;
        for seg in segments {
            match seg {
                Segment::Prose(body) => {
                    for paragraph in body.split('\n') {
                        let rows = wrap_visual(paragraph, inner_w);
                        for row in rows {
                            let gutter_span = if first {
                                Span::styled(tag.to_string(), tag_style)
                            } else {
                                Span::raw("     ")
                            };
                            first = false;
                            lines.push(Line::from(vec![
                                gutter_span,
                                Span::raw(" "),
                                Span::styled(row, body_style),
                            ]));
                        }
                    }
                }
                Segment::Code { lang: _, body } => {
                    for raw in body.split('\n') {
                        let gutter_span = if first {
                            Span::styled(tag.to_string(), tag_style)
                        } else {
                            Span::raw("     ")
                        };
                        first = false;
                        // Trim code lines to the viewport width (no wrap)
                        // so long lines don't reflow jarringly mid-stream.
                        let visible: String = raw.chars()
                            .scan(0usize, |w, c| {
                                let cw = unicode_width::UnicodeWidthChar::width(c).unwrap_or(0);
                                if *w + cw > inner_w.saturating_sub(2) { return None; }
                                *w += cw; Some(c)
                            })
                            .collect();
                        lines.push(Line::from(vec![
                            gutter_span,
                            Span::raw(" "),
                            Span::styled("│ ", Style::default().fg(t.code_gutter)),
                            Span::styled(visible, Style::default().fg(t.code_fg)),
                        ]));
                    }
                }
            }
        }

        // If this is the streaming turn, sweep a gradient across the
        // tail of the last rendered row and append a cursor glyph so
        // the user sees activity. Don't touch earlier rows — recolor-
        // ing a whole paragraph every frame costs too much and distracts.
        if shimmer_here {
            if let Some(last) = lines.last_mut() {
                const TAIL_LEN: usize = 10;
                if let Some(body_span) = last.spans.pop() {
                    let content: String = body_span.content.to_string();
                    let base_style = body_span.style;
                    let total_chars = content.chars().count();
                    if total_chars > TAIL_LEN {
                        let split_at = content.char_indices()
                            .nth(total_chars - TAIL_LEN)
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                        let (head, tail) = content.split_at(split_at);
                        last.spans.push(Span::styled(head.to_string(), base_style));
                        let tail_len = tail.chars().count().max(1);
                        for (i, c) in tail.chars().enumerate() {
                            last.spans.push(Span::styled(
                                c.to_string(),
                                Style::default().fg(shimmer::char_color(i, tail_len, phase)),
                            ));
                        }
                    } else {
                        let n = total_chars.max(1);
                        for (i, c) in content.chars().enumerate() {
                            last.spans.push(Span::styled(
                                c.to_string(),
                                Style::default().fg(shimmer::char_color(i, n, phase)),
                            ));
                        }
                    }
                }
                last.spans.push(Span::styled(
                    "▍".to_string(),
                    Style::default().fg(shimmer::char_color(0, 1, phase)),
                ));
            }
        }
        lines.push(Line::from(""));
    }

    let total = lines.len() as u16;
    app.last_viewport = viewport;
    app.last_total    = total;

    // Follow-tail: pin the last line to the bottom of the viewport.
    // Otherwise keep the user's manual offset, clamped to the new
    // content length.
    let max_off = total.saturating_sub(viewport);
    if app.follow_tail {
        app.scroll_off = max_off;
    } else {
        app.scroll_off = app.scroll_off.min(max_off);
        // If the user manually scrolled down to the tail, re-engage
        // the pin so new tokens keep tracking.
        if app.scroll_off == max_off { app.follow_tail = true; }
    }

    let para = Paragraph::new(lines).scroll((app.scroll_off, 0));
    f.render_widget(para, area);
}

/// Walk backwards from `i` to the nearest UTF-8 char boundary.
/// Safe to call with any byte offset inside `s`.
fn prev_char_boundary(s: &str, i: usize) -> usize {
    if i == 0 { return 0; }
    let mut j = i.min(s.len()).saturating_sub(1);
    while j > 0 && !s.is_char_boundary(j) { j -= 1; }
    j
}

/// Walk forwards from `i` to the next UTF-8 char boundary.
fn next_char_boundary(s: &str, i: usize) -> usize {
    let n = s.len();
    if i >= n { return n; }
    let mut j = i + 1;
    while j < n && !s.is_char_boundary(j) { j += 1; }
    j
}

/// A chunk of assistant output — prose runs wrap, code runs get a
/// dimmed left rail and are rendered verbatim.
enum Segment {
    Prose(String),
    Code { lang: String, body: String },
}

/// Split assistant text on triple-backtick fences. An unclosed fence
/// at the end of the stream is still emitted as a `Code` segment so
/// the user sees the block grow during streaming instead of seeing
/// it flip between prose and code every time a newline arrives.
///
/// Fences must begin at column 0 of a line — mid-line backticks are
/// left alone so inline snippets in prose render as plain text.
fn split_code_fences(text: &str) -> Vec<Segment> {
    let mut out: Vec<Segment> = Vec::new();
    let mut prose = String::new();
    let mut code  = String::new();
    let mut lang  = String::new();
    let mut in_code = false;
    // Accept either `\n` or start-of-string as a valid fence anchor.
    let mut at_line_start = true;

    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if at_line_start && c == '`' {
            // Look ahead for the other two backticks.
            let mut ticks = 1;
            while ticks < 3 && chars.peek() == Some(&'`') {
                chars.next();
                ticks += 1;
            }
            if ticks == 3 {
                if in_code {
                    // Closing fence — flush the code segment.
                    out.push(Segment::Code {
                        lang: std::mem::take(&mut lang),
                        body: std::mem::take(&mut code),
                    });
                    in_code = false;
                    // Drop the rest of the fence line (usually empty).
                    while let Some(&nc) = chars.peek() {
                        chars.next();
                        if nc == '\n' { break; }
                    }
                    at_line_start = true;
                } else {
                    // Opening fence — flush pending prose.
                    if !prose.is_empty() {
                        out.push(Segment::Prose(std::mem::take(&mut prose)));
                    }
                    // Capture the language tag up to the newline.
                    while let Some(&nc) = chars.peek() {
                        chars.next();
                        if nc == '\n' { break; }
                        lang.push(nc);
                    }
                    let trimmed = lang.trim().to_string();
                    lang = trimmed;
                    in_code = true;
                    at_line_start = true;
                }
                continue;
            } else {
                // Literal backticks (1 or 2) — fall through.
                let buf = if in_code { &mut code } else { &mut prose };
                for _ in 0..ticks { buf.push('`'); }
                at_line_start = false;
                continue;
            }
        }

        let buf = if in_code { &mut code } else { &mut prose };
        buf.push(c);
        at_line_start = c == '\n';
    }

    if in_code && !code.is_empty() {
        out.push(Segment::Code { lang, body: code });
    } else if !prose.is_empty() {
        out.push(Segment::Prose(prose));
    }
    if out.is_empty() {
        out.push(Segment::Prose(String::new()));
    }
    out
}

/// Split `text` into visual rows that each fit within `width`
/// terminal columns. Breaks on whitespace when possible; falls back
/// to hard-wrap for long unbroken runs (URLs, code, identifiers).
/// Returns at least one row, even for empty input, so a blank line
/// in the source still consumes a visible row.
fn wrap_visual(text: &str, width: usize) -> Vec<String> {
    if width == 0 { return vec![text.to_string()]; }
    if text.is_empty() { return vec![String::new()]; }

    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut cur_w: usize = 0;
    let mut word = String::new();
    let mut word_w: usize = 0;

    let flush_word = |cur: &mut String, cur_w: &mut usize,
                      word: &mut String, word_w: &mut usize,
                      out: &mut Vec<String>| {
        if word.is_empty() { return; }
        if *cur_w + *word_w <= width {
            cur.push_str(word);
            *cur_w += *word_w;
        } else {
            // Word won't fit on the current row.
            if !cur.is_empty() {
                out.push(std::mem::take(cur));
                *cur_w = 0;
            }
            // If the word itself is longer than one row, hard-wrap it.
            if *word_w > width {
                let mut w_acc = 0usize;
                let mut piece = String::new();
                for c in word.chars() {
                    let cw = c.width().unwrap_or(0);
                    if w_acc + cw > width {
                        out.push(std::mem::take(&mut piece));
                        w_acc = 0;
                    }
                    piece.push(c);
                    w_acc += cw;
                }
                *cur = piece;
                *cur_w = w_acc;
            } else {
                *cur = std::mem::take(word);
                *cur_w = *word_w;
                *word_w = 0;
                return;
            }
            word.clear();
            *word_w = 0;
        }
        word.clear();
        *word_w = 0;
    };

    for c in text.chars() {
        if c == ' ' || c == '\t' {
            flush_word(&mut cur, &mut cur_w, &mut word, &mut word_w, &mut out);
            let cw = 1;
            if cur_w + cw > width {
                out.push(std::mem::take(&mut cur));
                cur_w = 0;
                // Drop leading whitespace at row start.
            } else if !cur.is_empty() {
                cur.push(c);
                cur_w += cw;
            }
        } else {
            let cw = c.width().unwrap_or(0);
            word.push(c);
            word_w += cw;
        }
    }
    flush_word(&mut cur, &mut cur_w, &mut word, &mut word_w, &mut out);
    if !cur.is_empty() || out.is_empty() { out.push(cur); }
    out
}

fn render_input(f: &mut ratatui::Frame<'_>, area: Rect, app: &mut AppState) {
    let t = theme::theme();
    // Muted at rest, brand-lit while you're actually typing or
    // streaming. Keeps the chrome quiet so transcript reads first.
    let border = if app.input.is_empty() && !app.is_streaming() {
        t.border_inactive
    } else {
        t.border_active
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border))
        .title(Span::styled(
            if app.is_streaming() { " streaming — Esc to cancel " } else { " ask intelnav " },
            Style::default().fg(t.subtle),
        ));

    // Usable content width inside the box: total width minus two
    // border columns minus the "▸ " gutter on the first visual row.
    let inner_w = area.width.saturating_sub(4).max(4) as usize;
    app.last_input_inner_w = inner_w as u16;
    let inner_h = area.height.saturating_sub(2);

    // Build visible rows of the input buffer. Each `\n` in the raw
    // buffer begins a new paragraph, and each paragraph is wrapped
    // to `inner_w` columns. On the very first row we render the
    // "▸ " gutter; continuation rows use blank indent to stay aligned.
    let rows = input_visual_rows(&app.input, inner_w);
    let (cursor_row, cursor_col) = cursor_visual_pos(&app.input, app.cursor, inner_w);
    // Scroll the box so the cursor row is always visible. Pins the
    // cursor to the last visible row when the input overflows; lets
    // earlier rows stay in view while still typing toward the bottom.
    let scroll_row = input_scroll_for_cursor(cursor_row, inner_h);

    let lines: Vec<Line> = rows.iter().enumerate().map(|(i, r)| {
        if i == 0 {
            Line::from(vec![
                Span::styled("▸ ", Style::default().fg(t.intel)),
                Span::styled(r.clone(), Style::default().fg(t.text)),
            ])
        } else {
            Line::from(vec![
                Span::raw("  "),
                Span::styled(r.clone(), Style::default().fg(t.text)),
            ])
        }
    }).collect();

    let para = Paragraph::new(lines).block(block).scroll((scroll_row, 0));
    f.render_widget(para, area);

    if !app.is_streaming() {
        let visible_row = cursor_row.saturating_sub(scroll_row);
        // border (1) + "▸ " (2) on row 0, or border (1) + "  " (2) elsewhere.
        let x = area.x + 1 + 2 + cursor_col;
        let y = area.y + 1 + visible_row;
        let max_x = area.x + area.width.saturating_sub(2);
        let max_y = area.y + area.height.saturating_sub(2);
        f.set_cursor_position((x.min(max_x), y.min(max_y)));
    }
}

/// Jump the transcript to its top: `(scroll_off, follow_tail)`. Tail
/// follow is released so streaming tokens don't yank the view back.
fn transcript_scroll_to_top() -> (u16, bool) { (0, false) }

/// Jump the transcript to its tail and re-engage bottom-pin so new
/// tokens continue to track.
fn transcript_scroll_to_bottom(total: u16, viewport: u16) -> (u16, bool) {
    (total.saturating_sub(viewport), true)
}

/// Vertical scroll offset for the input Paragraph that keeps `cursor_row`
/// inside the visible window of `inner_h` rows. Returns 0 when the
/// content fits; otherwise the smallest scroll that puts the cursor
/// on the last visible row.
fn input_scroll_for_cursor(cursor_row: u16, inner_h: u16) -> u16 {
    if inner_h == 0 { return 0; }
    cursor_row.saturating_sub(inner_h - 1)
}

/// Wrap the raw input buffer into the visual rows the box will show.
/// Each `\n` always starts a new visual row.
fn input_visual_rows(input: &str, width: usize) -> Vec<String> {
    if width == 0 { return vec![input.to_string()]; }
    let mut out: Vec<String> = Vec::new();
    for para in input.split('\n') {
        if para.is_empty() {
            out.push(String::new());
            continue;
        }
        let mut cur = String::new();
        let mut cur_w = 0usize;
        for c in para.chars() {
            let cw = c.width().unwrap_or(0);
            if cur_w + cw > width {
                out.push(std::mem::take(&mut cur));
                cur_w = 0;
            }
            cur.push(c);
            cur_w += cw;
        }
        out.push(cur);
    }
    if out.is_empty() { out.push(String::new()); }
    out
}

/// Translate a byte offset in the raw input into `(row, col)` within
/// the wrapped view. Matches the wrapping used by `input_visual_rows`.
fn cursor_visual_pos(input: &str, cursor: usize, width: usize) -> (u16, u16) {
    if width == 0 { return (0, 0); }
    let mut row: u16 = 0;
    let mut col_w: usize = 0;
    for (i, c) in input.char_indices() {
        if i >= cursor { break; }
        if c == '\n' {
            row += 1;
            col_w = 0;
            continue;
        }
        let cw = c.width().unwrap_or(0);
        if col_w + cw > width {
            row += 1;
            col_w = 0;
        }
        col_w += cw;
    }
    (row, col_w as u16)
}

/// Byte offset of the start of the visual row that `cursor` sits on.
/// Matches `cursor_visual_pos`'s row convention: when `cursor` lands
/// exactly on a wrap boundary, it's treated as still being at the end
/// of the previous row (not the start of the next one), so Home from
/// "the cursor position the box currently shows" is a no-op.
fn visual_row_start(input: &str, cursor: usize, width: usize) -> usize {
    if width == 0 { return 0; }
    let mut row_start = 0usize;
    let mut col_w: usize = 0;
    for (i, c) in input.char_indices() {
        if i >= cursor { break; }
        if c == '\n' {
            row_start = i + 1;
            col_w = 0;
            continue;
        }
        let cw = c.width().unwrap_or(0);
        if col_w + cw > width {
            row_start = i;
            col_w = 0;
        }
        col_w += cw;
    }
    row_start
}

/// Byte offset of the end of the visual row containing `cursor` —
/// i.e., the position just before the next `\n` or wrap point, or
/// `input.len()` if neither occurs.
fn visual_row_end(input: &str, cursor: usize, width: usize) -> usize {
    if width == 0 { return input.len(); }
    let start = visual_row_start(input, cursor, width);
    let mut col_w: usize = 0;
    for (i, c) in input[start..].char_indices() {
        let abs = start + i;
        if c == '\n' { return abs; }
        let cw = c.width().unwrap_or(0);
        if col_w + cw > width { return abs; }
        col_w += cw;
    }
    input.len()
}

/// Visual height the input box needs, clamped so it never eats more
/// than half the terminal (caller passes that cap). Always ≥ 3
/// (top border + one row + bottom border).
fn input_height_visual(input: &str, width: u16, max_h: u16) -> u16 {
    let inner_w = width.saturating_sub(4).max(4) as usize;
    let rows = input_visual_rows(input, inner_w).len() as u16;
    (rows + 2).clamp(3, max_h.max(3))
}

fn render_status(f: &mut ratatui::Frame<'_>, area: Rect, app: &AppState) {
    let t = theme::theme();
    if let Some(p) = &app.dl_progress {
        let pct = match p.total {
            Some(t) if t > 0 => (p.done as f64 / t as f64 * 100.0).min(100.0),
            _ => 0.0,
        };
        let mb_s = p.bps / (1024.0 * 1024.0);
        let text = match p.total {
            Some(t) => format!(
                "  ⇣ {} · {:.0}% ({} / {}) · {:.1} MB/s  ",
                p.label, pct, human_bytes(p.done), human_bytes(t), mb_s,
            ),
            None => format!(
                "  ⇣ {} · {} · {:.1} MB/s  ",
                p.label, human_bytes(p.done), mb_s,
            ),
        };
        let phase = (app.start.elapsed().as_secs_f32() / 3.2) % 1.0;
        let n = text.chars().count();
        let chars: Vec<Span> = text.chars().enumerate().map(|(i, c)| {
            Span::styled(c.to_string(),
                Style::default().fg(shimmer::char_color(i, n, phase)))
        }).collect();
        f.render_widget(Paragraph::new(Line::from(chars)), area);
        return;
    }

    // Exit-pending toast trumps other status hints so the user can
    // see the confirm window.
    if let Some(armed_at) = app.exit_pending {
        if armed_at.elapsed() < Duration::from_millis(1500) {
            let text = "  press ctrl+c again to exit  ";
            f.render_widget(
                Paragraph::new(Line::from(Span::styled(
                    text, Style::default().fg(t.warning).add_modifier(Modifier::BOLD),
                ))),
                area,
            );
            return;
        }
    }

    if app.in_browser() {
        let hint = "  ↑/↓ pick · ↵ select · esc back  ";
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(hint, Style::default().fg(t.subtle)))),
            area,
        );
        return;
    }

    if app.in_doctor() {
        let hint = "  esc · ↵ · q  close  ";
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(hint, Style::default().fg(t.subtle)))),
            area,
        );
        return;
    }

    if app.is_streaming() {
        let text = "  ▸ streaming… press Esc to cancel  ";
        let phase = (app.start.elapsed().as_secs_f32() / 3.2) % 1.0;
        let n = text.chars().count();
        let chars: Vec<Span> = text.chars().enumerate().map(|(i, c)| {
            Span::styled(c.to_string(),
                Style::default().fg(shimmer::char_color(i, n, phase)))
        }).collect();
        f.render_widget(Paragraph::new(Line::from(chars)), area);
    } else {
        let hint = "  ↵ send · shift+↵ newline · /models · /help · ctrl+c quit  ";
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(hint, Style::default().fg(t.subtle)))),
            area,
        );
    }
}

/// Fetch `/v1/models` with a tight timeout. Returns `None` if the
/// gateway isn't reachable — that's fine, the browser just won't show
/// network rows in that case. Bridges our sync call site into the
/// ambient multi-thread tokio runtime via `block_in_place`.
fn fetch_remote_models(gateway: &str) -> Option<serde_json::Value> {
    let url = format!("{}/v1/models", gateway.trim_end_matches('/'));
    tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(async move {
            let client = reqwest::Client::builder()
                .timeout(Duration::from_millis(500))
                .build()
                .ok()?;
            let resp = client.get(&url).send().await.ok()?;
            if !resp.status().is_success() { return None; }
            resp.json::<serde_json::Value>().await.ok()
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scroll_keeps_cursor_visible() {
        // Content fits — never scroll.
        assert_eq!(input_scroll_for_cursor(0, 5), 0);
        assert_eq!(input_scroll_for_cursor(4, 5), 0);
        // Cursor at first overflow row — scroll one row up.
        assert_eq!(input_scroll_for_cursor(5, 5), 1);
        // Cursor deep below — pin to last visible row.
        assert_eq!(input_scroll_for_cursor(29, 10), 20);
        // Degenerate viewport — no division by zero, just no scroll.
        assert_eq!(input_scroll_for_cursor(10, 0), 0);
    }

    #[test]
    fn cursor_pos_matches_wrap() {
        // The first char past the wrap boundary lands on row 1, col 1.
        let (row, col) = cursor_visual_pos("abcdef", 6, 5);
        assert_eq!((row, col), (1, 1));
        // Newline always advances a row regardless of fill.
        let (row, col) = cursor_visual_pos("ab\ncd", 4, 5);
        assert_eq!((row, col), (1, 1));
    }

    #[test]
    fn rows_match_cursor_pos() {
        // The wrapped view and the cursor mapping must agree on row count.
        let input = "the quick brown fox jumps over";
        let rows = input_visual_rows(input, 10);
        let (row, _) = cursor_visual_pos(input, input.len(), 10);
        assert_eq!(row as usize, rows.len() - 1);
    }

    #[test]
    fn home_end_visual_row_single_line() {
        // Single short row — Home → 0, End → len, regardless of cursor.
        assert_eq!(visual_row_start("hello", 3, 20), 0);
        assert_eq!(visual_row_end("hello", 3, 20), 5);
    }

    #[test]
    fn home_end_visual_row_under_wrap() {
        // "abcdef" wrapped at width 5 → rows ["abcde", "f"].
        // Cursor mid-first-row: Home→0, End→5 (just before wrap).
        assert_eq!(visual_row_start("abcdef", 2, 5), 0);
        assert_eq!(visual_row_end("abcdef", 2, 5), 5);
        // Cursor on second row (byte 6 = end of "f"): Home→5, End→6.
        assert_eq!(visual_row_start("abcdef", 6, 5), 5);
        assert_eq!(visual_row_end("abcdef", 6, 5), 6);
    }

    #[test]
    fn transcript_scroll_jumps() {
        // Top: offset zero, tail follow released so streaming tokens
        // don't yank the view.
        assert_eq!(transcript_scroll_to_top(), (0, false));
        // Bottom: offset clamped to max_off, pin re-engaged.
        assert_eq!(transcript_scroll_to_bottom(100, 20), (80, true));
        // Content fits entirely — no scroll, pin still engaged.
        assert_eq!(transcript_scroll_to_bottom(10, 20), (0, true));
        // Degenerate: total == viewport means nothing to scroll past.
        assert_eq!(transcript_scroll_to_bottom(20, 20), (0, true));
    }

    #[test]
    fn home_end_visual_row_across_newlines() {
        // "abc\ndef" — cursor on second row (byte 5, between 'd' and 'e'):
        // Home jumps to byte 4 (start of "def"), End jumps to 7 (eol).
        assert_eq!(visual_row_start("abc\ndef", 5, 20), 4);
        assert_eq!(visual_row_end("abc\ndef", 5, 20), 7);
        // Cursor on first row (byte 1): End must stop at the '\n', not run on.
        assert_eq!(visual_row_end("abc\ndef", 1, 20), 3);
        // Cursor immediately after a trailing newline (empty new row): both
        // collapse to the cursor position.
        assert_eq!(visual_row_start("abc\n", 4, 20), 4);
        assert_eq!(visual_row_end("abc\n", 4, 20), 4);
    }
}

