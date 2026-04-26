//! The "pick a model" view.
//!
//! Flattens three sources — cached-local, network, catalog — into a
//! single arrow-pickable list. Enter commits: local/network switches
//! the active model, catalog starts a download.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use intelnav_runtime::Probe;

use crate::catalog::{catalog, CatalogEntry, Fit};
use crate::local::{human_bytes, LocalModel};
use crate::theme;

// ======================================================================
//  Rows
// ======================================================================

#[derive(Debug, Clone)]
pub enum RowKind {
    Local   { path: std::path::PathBuf },
    Network { peers: usize, tok_per_s: f64 },
    Install { entry: &'static CatalogEntry, fit: Fit },
}

#[derive(Debug, Clone)]
pub struct SourceRow {
    pub model:   String,
    pub tag:     &'static str,   // "local" | "network" | "install"
    pub detail:  String,         // the grey-text line suffix
    pub kind:    RowKind,
    pub enabled: bool,           // TooBig rows rendered but can't commit
}

/// Fold everything we know about the world into a single list of rows.
pub fn build_rows(
    local:  &[LocalModel],
    remote: Option<&serde_json::Value>,
    probe:  &Probe,
) -> Vec<SourceRow> {
    let mut out = Vec::new();

    // 1. Locally cached.
    for m in local {
        if !m.is_usable() { continue; }
        out.push(SourceRow {
            model:  m.name.clone(),
            tag:    "local",
            detail: format!("cached · {}", human_bytes(m.size_bytes)),
            kind:   RowKind::Local { path: m.path.clone() },
            enabled: true,
        });
    }

    // 2. Network.
    if let Some(body) = remote {
        if let Some(arr) = body.get("data").and_then(|v| v.as_array()) {
            for m in arr {
                let Some(id) = m.get("id").and_then(|v| v.as_str()) else { continue };
                let tps  = m.get("best_tok_per_s").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let prov = m.get("providers").and_then(|v| v.as_array())
                    .map(|a| a.len()).unwrap_or(0);
                out.push(SourceRow {
                    model: id.to_string(),
                    tag:   "network",
                    detail: format!("{prov} peers · {tps:.1} tok/s"),
                    kind:   RowKind::Network { peers: prov, tok_per_s: tps },
                    enabled: prov > 0,
                });
            }
        }
    }

    // 3. Catalog — skip entries we already have cached.
    let cached_names: std::collections::HashSet<&str> =
        local.iter().map(|m| m.name.as_str()).collect();
    for e in catalog() {
        let gguf_stem = e.gguf_file.trim_end_matches(".gguf");
        if cached_names.contains(gguf_stem) { continue; }
        let fit = e.fit(probe);
        let detail = match fit {
            Fit::Fits   => format!("{} · fits your RAM", human_bytes(e.size_bytes)),
            Fit::Tight  => format!("{} · tight — {} min free", human_bytes(e.size_bytes),
                                   human_bytes(e.ram_bytes_min)),
            Fit::TooBig => format!("{} · too big — needs {} free", human_bytes(e.size_bytes),
                                   human_bytes(e.ram_bytes_min)),
        };
        out.push(SourceRow {
            model:   e.display_name.to_string(),
            tag:     "install",
            detail,
            kind:    RowKind::Install { entry: e, fit },
            enabled: fit != Fit::TooBig,
        });
    }
    out
}

// ======================================================================
//  State + rendering
// ======================================================================

#[derive(Debug, Clone)]
pub struct BrowserState {
    pub rows:     Vec<SourceRow>,
    pub selected: usize,
}

impl BrowserState {
    pub fn new(rows: Vec<SourceRow>) -> Self {
        let selected = rows.iter().position(|r| r.enabled).unwrap_or(0);
        Self { rows, selected }
    }
    pub fn up(&mut self) {
        if self.rows.is_empty() { return; }
        let mut i = self.selected;
        for _ in 0..self.rows.len() {
            i = if i == 0 { self.rows.len() - 1 } else { i - 1 };
            if self.rows[i].enabled { self.selected = i; return; }
        }
    }
    pub fn down(&mut self) {
        if self.rows.is_empty() { return; }
        let mut i = self.selected;
        for _ in 0..self.rows.len() {
            i = (i + 1) % self.rows.len();
            if self.rows[i].enabled { self.selected = i; return; }
        }
    }
    pub fn current(&self) -> Option<&SourceRow> { self.rows.get(self.selected) }
}

pub enum BrowserAction {
    /// User pressed Enter on the current row.
    Commit,
    /// User pressed Esc — return to chat view.
    Close,
    /// Keystroke consumed, no-op for the parent.
    Consumed,
    /// Not a browser key — let the parent handle it.
    Passthrough,
}

pub fn handle_key(state: &mut BrowserState, k: KeyEvent) -> BrowserAction {
    match k.code {
        KeyCode::Up   | KeyCode::Char('k') => { state.up();   BrowserAction::Consumed }
        KeyCode::Down | KeyCode::Char('j') => { state.down(); BrowserAction::Consumed }
        KeyCode::Enter                     => BrowserAction::Commit,
        KeyCode::Esc  | KeyCode::Char('q') => BrowserAction::Close,
        _                                  => BrowserAction::Passthrough,
    }
}

pub fn render(f: &mut ratatui::Frame<'_>, area: Rect, state: &BrowserState) {
    let t = theme::theme();
    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(Span::styled(
        "  ↑/↓ pick · ↵ select · esc back · only highlighted rows are selectable",
        Style::default().fg(t.subtle).add_modifier(Modifier::ITALIC),
    )));
    lines.push(Line::from(""));

    if state.rows.is_empty() {
        lines.push(Line::from(Span::styled(
            "  no models visible — no local cache, no gateway, no catalog hits.",
            Style::default().fg(t.subtle),
        )));
    }

    for (i, row) in state.rows.iter().enumerate() {
        let selected = i == state.selected;
        let arrow = if selected { "›" } else { " " };

        let tag_color = match row.tag {
            "local"   => t.tag_local,
            "network" => t.tag_network,
            "install" => t.tag_install,
            _         => t.subtle,
        };
        let model_style = if selected {
            Style::default().fg(t.intel).add_modifier(Modifier::BOLD)
        } else if row.enabled {
            Style::default().fg(t.text)
        } else {
            Style::default().fg(t.inactive).add_modifier(Modifier::DIM)
        };

        lines.push(Line::from(vec![
            Span::styled(format!(" {arrow} "),
                Style::default().fg(if selected { t.intel } else { t.inactive })),
            Span::styled(format!("{:<8}", row.tag), Style::default().fg(tag_color)),
            Span::raw("  "),
            Span::styled(format!("{:<36}", row.model), model_style),
            Span::raw("  "),
            Span::styled(row.detail.clone(),
                Style::default().fg(if row.enabled { t.subtle } else { t.inactive })),
        ]));
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(t.border_active))
        .title(Span::styled(" models ",
            Style::default().fg(t.text).add_modifier(Modifier::BOLD)));
    f.render_widget(Paragraph::new(lines).block(block).wrap(Wrap { trim: false }), area);
}
