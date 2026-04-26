//! Gradient-shimmer text coloring — the Claude-Code-style orange pulse.
//!
//! Given a phase in `[0.0, 1.0)` and a character position, sample a 3-stop
//! gradient (dark orange → bright orange → dark orange) so the highlight
//! slides across the string as phase advances.

use ratatui::style::Color;

/// Claude-brand orange stops: `#b4542e` → `#d97757` → `#ffb089`.
const STOP_A: (u8, u8, u8) = (0xb4, 0x54, 0x2e);
const STOP_B: (u8, u8, u8) = (0xd9, 0x77, 0x57);
const STOP_C: (u8, u8, u8) = (0xff, 0xb0, 0x89);

/// Sample the gradient at `t ∈ [0, 1)` — triangle wave A → C → A.
fn sample(t: f32) -> (u8, u8, u8) {
    let t = t.rem_euclid(1.0);
    let (from, to, local) = if t < 0.5 {
        (STOP_A, STOP_C, t * 2.0)
    } else {
        (STOP_C, STOP_A, (t - 0.5) * 2.0)
    };
    // Pass through STOP_B implicitly via the midpoint of the linear blend.
    let mid = STOP_B;
    // Blend in two halves: from→mid (0..0.5) and mid→to (0.5..1)
    if local < 0.5 {
        lerp(from, mid, local * 2.0)
    } else {
        lerp(mid, to, (local - 0.5) * 2.0)
    }
}

fn lerp(a: (u8, u8, u8), b: (u8, u8, u8), t: f32) -> (u8, u8, u8) {
    let l = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t) as u8;
    (l(a.0, b.0), l(a.1, b.1), l(a.2, b.2))
}

/// Color for character `idx` of a string of length `len` at time-phase `phase`.
pub fn char_color(idx: usize, len: usize, phase: f32) -> Color {
    let pos = if len == 0 { 0.0 } else { idx as f32 / len.max(1) as f32 };
    let (r, g, b) = sample(pos + phase);
    Color::Rgb(r, g, b)
}

/// Solid base orange — for non-shimmering brand text.
pub fn base() -> Color {
    Color::Rgb(STOP_B.0, STOP_B.1, STOP_B.2)
}
