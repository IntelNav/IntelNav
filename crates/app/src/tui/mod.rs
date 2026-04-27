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
//!
//! Pure text/wrap helpers (cursor geometry, code-fence splitting,
//! transcript scrolling) live in the sibling [`wrap`] module so this
//! file stays focused on `AppState` and the render/key plumbing.

mod wrap;

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

use crate::banner::tagline;
use crate::browser::{self, BrowserAction, BrowserState, RowKind};
use crate::catalog::{self, CatalogEntry};
use crate::contribute::KeptRanges;
use intelnav_net::{SwarmIndex, SwarmModel};

use crate::chain_driver::{ChainDriver, ChainTarget, DraftTarget};
use crate::delta::{ChatMessage, Delta};
use crate::download::{self, Event as DlEvent};
use crate::local::{self, human_bytes, LocalDriver, LocalModel};
use crate::shimmer;
use crate::slash::{self, SlashCmd};
use crate::theme;

use wrap::{
    cursor_visual_pos, input_height_visual, input_scroll_for_cursor, input_visual_rows,
    next_char_boundary, prev_char_boundary, split_code_fences, transcript_scroll_to_bottom,
    transcript_scroll_to_top, visual_row_end, visual_row_start, wrap_visual, Segment,
};

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

    // Spawn a *client-only* libp2p host: routing table + DHT
    // queries, no announce loop. To host slices for the swarm,
    // run `intelnav-node` separately — closing this chat window
    // won't take your slices offline because the chat never
    // published them.
    let swarm_handle = match crate::swarm_node::spawn_client_only(config).await {
        Ok(h) => {
            app.history.push(Turn::system(format!(
                "swarm: peer {} reading the DHT (run `intelnav-node` to host).",
                h.node.peer_id,
            )));
            app.swarm_index = Some(h.index.clone());
            Some(h)
        }
        Err(e) => {
            app.history.push(Turn::system(format!(
                "swarm: offline ({e}). /models will show only local + hub rows.",
            )));
            None
        }
    };

    let result = run_loop(&mut terminal, &mut app).await;
    drop(swarm_handle);

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
    models_dir:  std::path::PathBuf,
    mode:        RunMode,
    model:       String,
    tier:        String,
    quorum:      u8,
    allow_wan:   bool,

    local:       LocalDriver,
    chain:       ChainDriver,
    local_scan:  Vec<LocalModel>,
    /// Live DHT handle. `None` if libp2p couldn't bind on startup —
    /// the TUI degrades to local-only in that case.
    swarm_index: Option<SwarmIndex>,
    /// Cached SwarmIndex snapshot. Populated each time `/models`
    /// refreshes against the DHT.
    swarm_models: Vec<SwarmModel>,
    /// Channel that delivers refreshed swarm snapshots from the
    /// background fan-out task.
    swarm_rx:    Option<mpsc::UnboundedReceiver<Vec<SwarmModel>>>,

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
    /// Active GGUF-split job (hub→split→host pipeline).
    splitting:   Option<mpsc::UnboundedReceiver<crate::contribute::SplitEvent>>,
    /// Active swarm slice-pull (pre-split contribute path).
    pulling:     Option<mpsc::UnboundedReceiver<crate::swarm_contribute::SwarmPullEvent>>,
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
            models_dir, mode, model, tier, quorum, allow_wan,
            local, chain, local_scan,
            swarm_index:  None,
            swarm_models: Vec::new(),
            swarm_rx:     None,
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
            splitting:   None,
            pulling:     None,
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
                let msg = match self.chain.target() {
                    Some(t) => format!("network mode — peer chain {}", t.summary()),
                    None    => "network mode — no peer chain configured. /peers a:7717,b:7717 6,12 to set one.".into(),
                };
                self.history.push(Turn::system(msg));
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
            // Network mode runs locally-front-half + chain-tail through
            // the configured peer pipeline.
            RunMode::Network => {
                if self.chain.target().is_none() {
                    self.emit_fatal(
                        "no peer chain configured. /peers <host:port,...> <split,...> to set one.".into(),
                    );
                    return;
                }
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
    /// target. `/peers clear` drops the chain.
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
                self.history.push(Turn::system("peer chain cleared"));
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
    /// `int8` (or aliases) to switch the chain driver's activation
    /// dtype. Takes effect on the next turn; no reconnect needed.
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
        let rows = browser::build_rows(&self.local_scan, &self.swarm_models, &probe);
        self.view = View::Browser(BrowserState::new(rows));
        self.kick_swarm_refresh();
    }

    /// Fire a fan-out DHT query for every catalog cid + every cid we
    /// already host locally. Results land on `self.swarm_rx` and the
    /// next tick rebuilds the browser rows. Drops if we don't have a
    /// live SwarmIndex (libp2p failed to bind).
    fn kick_swarm_refresh(&mut self) {
        let Some(index) = self.swarm_index.clone() else { return };
        let mut requests: Vec<(String, Vec<(u16, u16)>)> = catalog::catalog().iter()
            .map(|e| (e.model_cid(), e.swarm_ranges()))
            .collect();
        // Also probe any cid we host locally — it may have peers we
        // didn't catalog (custom HF models split by another user).
        for k in scan_local_kept(&self.models_dir) {
            if !requests.iter().any(|(c, _)| c == &k.model_cid) {
                requests.push((k.model_cid, k.kept));
            }
        }
        let (tx, rx) = mpsc::unbounded_channel();
        self.swarm_rx = Some(rx);
        tokio::spawn(async move {
            let models = index.refresh_many(&requests).await;
            let _ = tx.send(models);
        });
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
            RowKind::Swarm { cid, unique_peers, .. } => {
                let Some(sm) = self.swarm_models.iter().find(|m| m.cid == cid) else {
                    self.history.push(Turn::system(
                        "swarm cache lost this model — reopen /models and retry.",
                    ));
                    self.close_browser();
                    return;
                };
                let ranges: Vec<(u16, u16, Vec<intelnav_net::ProviderRecord>)> =
                    sm.ranges.iter()
                        .map(|r| (r.start, r.end, r.providers.clone()))
                        .collect();
                match ChainTarget::from_swarm(&ranges) {
                    Ok(target) => {
                        self.history.push(Turn::system(format!(
                            "→ swarm · {} ({} peers, cid={}) · chain {}",
                            row.model, unique_peers,
                            &cid[..cid.len().min(12)],
                            target.summary(),
                        )));
                        self.chain.set_target(Some(target));
                        self.mode  = RunMode::Network;
                        self.model = row.model.clone();
                    }
                    Err(e) => {
                        self.history.push(Turn::system(format!(
                            "couldn't assemble chain for {}: {e}", row.model,
                        )));
                    }
                }
            }
            RowKind::Install { entry, .. } => {
                self.start_install(entry);
            }
        }
        self.close_browser();
    }

    /// `c` on a row in the model picker. Hub rows trigger
    /// download+split+host (#21); swarm rows trigger pre-split
    /// pull+announce (#22). Both paths share the chunk-server
    /// + DHT announce backend; today we surface the user-visible
    /// "I want to contribute X" intent and stop short of running
    /// the actual job — those are the next two tasks.
    fn contribute_browser(&mut self) {
        let Some(row) = (match &self.view {
            View::Browser(s) => s.current().cloned(),
            _                => None,
        }) else { return; };
        if !row.contribute_ok { return; }

        match row.kind {
            RowKind::Local { .. } => {
                self.history.push(Turn::system(
                    "this model is already cached — nothing to contribute.",
                ));
            }
            RowKind::Swarm { cid, .. } => {
                let Some(sm) = self.swarm_models.iter().find(|m| m.cid == cid) else {
                    self.history.push(Turn::system(
                        "swarm cache lost this model — reopen /models and retry.",
                    ));
                    self.close_browser();
                    return;
                };
                let candidates: Vec<(u16, u16, Vec<intelnav_net::ProviderRecord>)> = sm.ranges.iter()
                    .map(|r| (r.start, r.end, r.providers.clone()))
                    .collect();
                let Some((start, end, provider)) =
                    crate::swarm_contribute::default_range(&cid, &candidates)
                else {
                    self.history.push(Turn::system(
                        "no provider on the DHT publishes a chunk-server URL for any slice yet.",
                    ));
                    self.close_browser();
                    return;
                };
                self.history.push(Turn::system(format!(
                    "pulling slice [{start}..{end}) of {} from {}",
                    &cid[..cid.len().min(12)],
                    &provider.peer_id[..provider.peer_id.len().min(12)],
                )));
                let rx = crate::swarm_contribute::start_pull(
                    cid.clone(),
                    (start, end),
                    provider,
                    self.models_dir.clone(),
                );
                self.pulling = Some(rx);
            }
            RowKind::Install { entry, .. } => {
                // If the GGUF is already cached, jump straight to the
                // split. Otherwise tell the user to download first
                // (Enter on the same row) and re-press `c` after.
                let stem = entry.gguf_file.trim_end_matches(".gguf");
                let cached = self.local_scan.iter().find(|m| m.name == stem).cloned();
                match cached {
                    Some(m) if m.is_usable() => {
                        self.history.push(Turn::system(format!(
                            "splitting {} → shards in {}/.shards/{}",
                            entry.display_name,
                            self.models_dir.display(),
                            entry.model_cid(),
                        )));
                        let rx = crate::contribute::start_split(
                            entry,
                            m,
                            self.models_dir.clone(),
                        );
                        self.splitting = Some(rx);
                    }
                    Some(m) => {
                        self.history.push(Turn::system(format!(
                            "{} is cached but not usable: {}", entry.display_name, m.status_line(),
                        )));
                    }
                    None => {
                        self.history.push(Turn::system(format!(
                            "{} isn't cached yet — press Enter on this row to download first, \
                             then `c` to split + host.",
                            entry.display_name,
                        )));
                    }
                }
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

    fn drain_swarm(&mut self) -> bool {
        let Some(rx) = self.swarm_rx.as_mut() else { return false };
        let mut dirty = false;
        while let Ok(models) = rx.try_recv() {
            self.swarm_models = models;
            dirty = true;
            // Rebuild the browser rows in place if the picker is
            // still open so the user sees the swarm rows pop in.
            if let View::Browser(_) = &self.view {
                let probe = Probe::collect();
                let rows = browser::build_rows(&self.local_scan, &self.swarm_models, &probe);
                self.view = View::Browser(BrowserState::new(rows));
            }
        }
        if dirty { self.swarm_rx = None; }
        dirty
    }

    fn drain_pull(&mut self) -> bool {
        use crate::swarm_contribute::SwarmPullEvent;
        let Some(rx) = self.pulling.as_mut() else { return false };
        let mut dirty = false;
        while let Ok(ev) = rx.try_recv() {
            dirty = true;
            match ev {
                SwarmPullEvent::Started { manifest_url, range } => {
                    self.history.push(Turn::system(format!(
                        "pull: GET {manifest_url} for slice [{}..{})", range.0, range.1,
                    )));
                }
                SwarmPullEvent::ManifestOk { manifest_cid, n_layers } => {
                    self.history.push(Turn::system(format!(
                        "pull: manifest ok ({} layers, cid {})",
                        n_layers,
                        &manifest_cid[..manifest_cid.len().min(12)],
                    )));
                }
                SwarmPullEvent::ChunksDone { bytes, n_chunks } => {
                    self.history.push(Turn::system(format!(
                        "pull: chunks ok ({n_chunks} bundles, {})",
                        crate::local::human_bytes(bytes),
                    )));
                }
                SwarmPullEvent::Done { kept_ranges, shard_root } => {
                    let kept_str = kept_ranges.iter()
                        .map(|(s, e)| format!("[{s}..{e})"))
                        .collect::<Vec<_>>().join(" ");
                    self.history.push(Turn::system(format!(
                        "✓ slice pulled — hosting {kept_str} in {}", shard_root.display(),
                    )));
                    self.pulling = None;
                    break;
                }
                SwarmPullEvent::Error(msg) => {
                    self.history.push(Turn::system(format!("⚠ pull failed: {msg}")));
                    self.pulling = None;
                    break;
                }
            }
        }
        dirty
    }

    fn drain_split(&mut self) -> bool {
        use crate::contribute::SplitEvent;
        let Some(rx) = self.splitting.as_mut() else { return false };
        let mut dirty = false;
        while let Ok(ev) = rx.try_recv() {
            dirty = true;
            match ev {
                SplitEvent::Started { gguf, output } => {
                    self.history.push(Turn::system(format!(
                        "split: chunking {} → {}", gguf.display(), output.display(),
                    )));
                }
                SplitEvent::Done { manifest_cid, n_bundles, bytes, kept_ranges, shard_root } => {
                    let kept_str = kept_ranges.iter()
                        .map(|(s, e)| format!("[{s}..{e})"))
                        .collect::<Vec<_>>().join(" ");
                    self.history.push(Turn::system(format!(
                        "✓ split done — {n_bundles} bundles · {} written · manifest {} · hosting {}",
                        crate::local::human_bytes(bytes),
                        &manifest_cid[..manifest_cid.len().min(12)],
                        kept_str,
                    )));
                    self.history.push(Turn::system(format!(
                        "shards live at {}", shard_root.display(),
                    )));
                    self.splitting = None;
                    break;
                }
                SplitEvent::Error(msg) => {
                    self.history.push(Turn::system(format!("⚠ split failed: {msg}")));
                    self.splitting = None;
                    break;
                }
            }
        }
        dirty
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
        let _ = app.drain_split();
        let _ = app.drain_pull();
        let _ = app.drain_swarm();
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

/// Walk `<models_dir>/.shards/<cid>/kept_ranges.json` for cids the
/// local peer hosts. Returns an empty Vec on any read / parse error
/// so a stale sidecar can't break the picker.
fn scan_local_kept(models_dir: &std::path::Path) -> Vec<KeptRanges> {
    let shards = models_dir.join(".shards");
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(&shards) else { return out; };
    for entry in rd.flatten() {
        let path = entry.path().join("kept_ranges.json");
        let Ok(bytes) = std::fs::read(&path) else { continue };
        if let Ok(k) = serde_json::from_slice::<KeptRanges>(&bytes) {
            out.push(k);
        }
    }
    out
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
                BrowserAction::Contribute  => app.contribute_browser(),
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

