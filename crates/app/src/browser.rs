//! The "pick a model" view.
//!
//! Flattens three sources — Local cache, Swarm (DHT-discovered), Hub
//! catalog — into a single arrow-pickable list. `Enter` runs / joins
//! a model; `c` contributes (downloads a slice + announces to DHT).

use std::collections::HashMap;

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use intelnav_net::SwarmModel;
use intelnav_runtime::Probe;

use crate::catalog::{catalog, CatalogEntry, Fit};
use crate::local::{human_bytes, LocalModel};
use crate::theme;

// ======================================================================
//  Rows
// ======================================================================

/// Schema-complete enum: the renderer only consumes a subset today,
/// but the full per-variant metadata is captured at construction so
/// new render passes (e.g. fit-aware coloring) don't need a parallel
/// data path.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum RowKind {
    Local   { path: std::path::PathBuf },
    /// A model the swarm advertises slices for (DHT-discovered).
    /// The picker uses `cid` to look the model up against the
    /// SwarmIndex when the user commits.
    Swarm   {
        cid:           String,
        unique_peers:  usize,
        ranges_total:  usize,
        ranges_covered: usize,
        fully_served:  bool,
    },
    /// A catalog entry the user can install. `swarm_peers` is the
    /// peer count if the same cid is already on the DHT — selecting
    /// this row gives the user a choice between "use existing
    /// providers" and "download + host yourself".
    Install {
        entry:        &'static CatalogEntry,
        fit:          Fit,
        swarm_peers:  usize,
    },
}

#[derive(Debug, Clone)]
pub struct SourceRow {
    pub model:   String,
    pub tag:     &'static str,   // "local" | "swarm" | "hub"
    pub detail:  String,
    pub kind:    RowKind,
    /// Inference-runnable. Disabled rows (TooBig hub entries, swarm
    /// rows with gaps the user can't bridge) render but can't commit.
    pub enabled: bool,
    /// Contribute-able. `c` on this row pulls a slice + announces.
    pub contribute_ok: bool,
}

/// Fold everything we know about the world into a single list of rows.
///
/// `swarm` is whatever the SwarmIndex currently has cached for this
/// session — the picker is read-only over it; refreshing happens on
/// `/models` open from the TUI side.
pub fn build_rows(
    local:  &[LocalModel],
    swarm:  &[SwarmModel],
    probe:  &Probe,
) -> Vec<SourceRow> {
    let mut out = Vec::new();

    // Index swarm models by cid so catalog rows can annotate.
    let by_cid: HashMap<&str, &SwarmModel> = swarm.iter()
        .map(|m| (m.cid.as_str(), m))
        .collect();
    let catalog_cids: std::collections::HashSet<String> = catalog().iter()
        .map(|e| e.model_cid())
        .collect();

    // 1. Locally cached.
    for m in local {
        if !m.is_usable() { continue; }
        out.push(SourceRow {
            model:  m.name.clone(),
            tag:    "local",
            detail: format!("cached · {}", human_bytes(m.size_bytes)),
            kind:   RowKind::Local { path: m.path.clone() },
            enabled: true,
            contribute_ok: false,
        });
    }

    // 2. Swarm — every model the DHT advertises, *except* those the
    //    catalog also covers (those get merged into the catalog row).
    for sm in swarm {
        if catalog_cids.contains(&sm.cid) { continue; }
        let display = sm.envelope.as_ref()
            .map(|e| e.display_name.clone())
            .unwrap_or_else(|| sm.cid.clone());
        let unique = sm.unique_providers();
        let total  = sm.ranges.len();
        let cov    = sm.ranges.iter().filter(|r| !r.providers.is_empty()).count();
        let fully  = sm.fully_served();
        let detail = if fully {
            format!("{unique} peers · {cov}/{total} slices · ready")
        } else if cov == 0 {
            format!("{unique} peers · no slice fully covered yet")
        } else {
            format!("{unique} peers · {cov}/{total} slices · partial")
        };
        out.push(SourceRow {
            model: display,
            tag:   "swarm",
            detail,
            kind:  RowKind::Swarm {
                cid:           sm.cid.clone(),
                unique_peers:  unique,
                ranges_total:  total,
                ranges_covered: cov,
                fully_served:  fully,
            },
            enabled: fully,
            contribute_ok: true,
        });
    }

    // 3. Catalog. Skip entries we already have cached locally; merge
    //    swarm peer count for entries whose cid lives on the DHT.
    let cached_names: std::collections::HashSet<&str> =
        local.iter().map(|m| m.name.as_str()).collect();
    for e in catalog() {
        let gguf_stem = e.gguf_file.trim_end_matches(".gguf");
        if cached_names.contains(gguf_stem) { continue; }
        let fit = e.fit(probe);
        let cid = e.model_cid();
        let swarm_peers = by_cid.get(cid.as_str())
            .map(|m| m.unique_providers())
            .unwrap_or(0);
        let fit_blurb = match fit {
            Fit::Fits   => format!("{} · fits your RAM", human_bytes(e.size_bytes)),
            Fit::Tight  => format!("{} · tight — {} min free", human_bytes(e.size_bytes),
                                   human_bytes(e.ram_bytes_min)),
            Fit::TooBig => format!("{} · too big — needs {} free", human_bytes(e.size_bytes),
                                   human_bytes(e.ram_bytes_min)),
        };
        let detail = if swarm_peers > 0 {
            format!("{fit_blurb} · {swarm_peers} swarm peers")
        } else {
            fit_blurb
        };
        out.push(SourceRow {
            model:   e.display_name.to_string(),
            tag:     "hub",
            detail,
            kind:    RowKind::Install { entry: e, fit, swarm_peers },
            enabled: fit != Fit::TooBig,
            contribute_ok: fit != Fit::TooBig,
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
            if self.rows[i].enabled || self.rows[i].contribute_ok { self.selected = i; return; }
        }
    }
    pub fn down(&mut self) {
        if self.rows.is_empty() { return; }
        let mut i = self.selected;
        for _ in 0..self.rows.len() {
            i = (i + 1) % self.rows.len();
            if self.rows[i].enabled || self.rows[i].contribute_ok { self.selected = i; return; }
        }
    }
    pub fn current(&self) -> Option<&SourceRow> { self.rows.get(self.selected) }
}

pub enum BrowserAction {
    /// User pressed Enter on the current row — run / join the model.
    Commit,
    /// User pressed `c` on the current row — pull a slice + announce.
    Contribute,
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
        KeyCode::Char('c')                 => BrowserAction::Contribute,
        KeyCode::Esc  | KeyCode::Char('q') => BrowserAction::Close,
        _                                  => BrowserAction::Passthrough,
    }
}

pub fn render(f: &mut ratatui::Frame<'_>, area: Rect, state: &BrowserState) {
    let t = theme::theme();
    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(Span::styled(
        "  ↑/↓ pick · ↵ run · c contribute · esc back",
        Style::default().fg(t.subtle).add_modifier(Modifier::ITALIC),
    )));
    lines.push(Line::from(""));

    if state.rows.is_empty() {
        lines.push(Line::from(Span::styled(
            "  no models visible — local cache empty, swarm has nothing yet, no catalog hits.",
            Style::default().fg(t.subtle),
        )));
    }

    for (i, row) in state.rows.iter().enumerate() {
        let selected = i == state.selected;
        let arrow = if selected { "›" } else { " " };

        let tag_color = match row.tag {
            "local" => t.tag_local,
            "swarm" => t.tag_network,
            "hub"   => t.tag_install,
            _       => t.subtle,
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
            Span::styled(format!("{:<6}", row.tag), Style::default().fg(tag_color)),
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
