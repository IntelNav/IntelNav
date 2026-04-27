//! Pure text/wrap helpers for the chat REPL — no `AppState`, no
//! ratatui frames, no I/O. Carved out of the original `tui.rs` so the
//! rest of the TUI stays focused on `AppState` + render plumbing.
//!
//! Two clusters live here:
//!
//! * **Conversation rendering helpers** — [`Segment`],
//!   [`split_code_fences`], [`wrap_visual`]. Used by
//!   `render_conversation` to split assistant text into prose runs
//!   that wrap and code runs that don't.
//!
//! * **Input-box geometry** — [`input_visual_rows`],
//!   [`cursor_visual_pos`], [`visual_row_start`] / [`visual_row_end`],
//!   [`input_scroll_for_cursor`], [`input_height_visual`], plus
//!   [`prev_char_boundary`] / [`next_char_boundary`] for cursor
//!   movement. These together let the REPL handle multi-line input
//!   with proper Unicode width and wrap-aware Home/End/↑/↓.
//!
//! * **Transcript scroll jumps** — [`transcript_scroll_to_top`],
//!   [`transcript_scroll_to_bottom`]. Tiny but worth co-locating with
//!   the input-scroll helper above.

use unicode_width::UnicodeWidthChar;

// ---------------------------------------------------------------------
// Char-boundary helpers
// ---------------------------------------------------------------------

/// Walk backwards from `i` to the nearest UTF-8 char boundary.
/// Safe to call with any byte offset inside `s`.
pub(super) fn prev_char_boundary(s: &str, i: usize) -> usize {
    if i == 0 { return 0; }
    let mut j = i.min(s.len()).saturating_sub(1);
    while j > 0 && !s.is_char_boundary(j) { j -= 1; }
    j
}

/// Walk forwards from `i` to the next UTF-8 char boundary.
pub(super) fn next_char_boundary(s: &str, i: usize) -> usize {
    let n = s.len();
    if i >= n { return n; }
    let mut j = i + 1;
    while j < n && !s.is_char_boundary(j) { j += 1; }
    j
}

// ---------------------------------------------------------------------
// Code fences + prose wrap (conversation rendering)
// ---------------------------------------------------------------------

/// A chunk of assistant output — prose runs wrap, code runs get a
/// dimmed left rail and are rendered verbatim.
pub(super) enum Segment {
    Prose(String),
    Code {
        /// Captured for future syntax highlighting / filetype-aware
        /// rendering; currently dimmed verbatim so unused in render.
        #[allow(dead_code)]
        lang: String,
        body: String,
    },
}

/// Split assistant text on triple-backtick fences. An unclosed fence
/// at the end of the stream is still emitted as a `Code` segment so
/// the user sees the block grow during streaming instead of seeing
/// it flip between prose and code every time a newline arrives.
///
/// Fences must begin at column 0 of a line — mid-line backticks are
/// left alone so inline snippets in prose render as plain text.
pub(super) fn split_code_fences(text: &str) -> Vec<Segment> {
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
pub(super) fn wrap_visual(text: &str, width: usize) -> Vec<String> {
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

// ---------------------------------------------------------------------
// Transcript scroll jumps
// ---------------------------------------------------------------------

/// Jump the transcript to its top: `(scroll_off, follow_tail)`. Tail
/// follow is released so streaming tokens don't yank the view back.
pub(super) fn transcript_scroll_to_top() -> (u16, bool) { (0, false) }

/// Jump the transcript to its tail and re-engage bottom-pin so new
/// tokens continue to track.
pub(super) fn transcript_scroll_to_bottom(total: u16, viewport: u16) -> (u16, bool) {
    (total.saturating_sub(viewport), true)
}

// ---------------------------------------------------------------------
// Input-box geometry
// ---------------------------------------------------------------------

/// Vertical scroll offset for the input Paragraph that keeps `cursor_row`
/// inside the visible window of `inner_h` rows. Returns 0 when the
/// content fits; otherwise the smallest scroll that puts the cursor
/// on the last visible row.
pub(super) fn input_scroll_for_cursor(cursor_row: u16, inner_h: u16) -> u16 {
    if inner_h == 0 { return 0; }
    cursor_row.saturating_sub(inner_h - 1)
}

/// Wrap the raw input buffer into the visual rows the box will show.
/// Each `\n` always starts a new visual row.
pub(super) fn input_visual_rows(input: &str, width: usize) -> Vec<String> {
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
pub(super) fn cursor_visual_pos(input: &str, cursor: usize, width: usize) -> (u16, u16) {
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
pub(super) fn visual_row_start(input: &str, cursor: usize, width: usize) -> usize {
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
pub(super) fn visual_row_end(input: &str, cursor: usize, width: usize) -> usize {
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
pub(super) fn input_height_visual(input: &str, width: u16, max_h: u16) -> u16 {
    let inner_w = width.saturating_sub(4).max(4) as usize;
    let rows = input_visual_rows(input, inner_w).len() as u16;
    (rows + 2).clamp(3, max_h.max(3))
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
        let (row, col) = cursor_visual_pos("abcdef", 6, 5);
        assert_eq!((row, col), (1, 1));
        let (row, col) = cursor_visual_pos("ab\ncd", 4, 5);
        assert_eq!((row, col), (1, 1));
    }

    #[test]
    fn rows_match_cursor_pos() {
        let input = "the quick brown fox jumps over";
        let rows = input_visual_rows(input, 10);
        let (row, _) = cursor_visual_pos(input, input.len(), 10);
        assert_eq!(row as usize, rows.len() - 1);
    }

    #[test]
    fn home_end_visual_row_single_line() {
        assert_eq!(visual_row_start("hello", 3, 20), 0);
        assert_eq!(visual_row_end("hello", 3, 20), 5);
    }

    #[test]
    fn home_end_visual_row_under_wrap() {
        assert_eq!(visual_row_start("abcdef", 2, 5), 0);
        assert_eq!(visual_row_end("abcdef", 2, 5), 5);
        assert_eq!(visual_row_start("abcdef", 6, 5), 5);
        assert_eq!(visual_row_end("abcdef", 6, 5), 6);
    }

    #[test]
    fn transcript_scroll_jumps() {
        assert_eq!(transcript_scroll_to_top(), (0, false));
        assert_eq!(transcript_scroll_to_bottom(100, 20), (80, true));
        assert_eq!(transcript_scroll_to_bottom(10, 20), (0, true));
        assert_eq!(transcript_scroll_to_bottom(20, 20), (0, true));
    }

    #[test]
    fn home_end_visual_row_across_newlines() {
        assert_eq!(visual_row_start("abc\ndef", 5, 20), 4);
        assert_eq!(visual_row_end("abc\ndef", 5, 20), 7);
        assert_eq!(visual_row_end("abc\ndef", 1, 20), 3);
        assert_eq!(visual_row_start("abc\n", 4, 20), 4);
        assert_eq!(visual_row_end("abc\n", 4, 20), 4);
    }
}
