//! Semantic palette — every color the TUI uses lives here so swapping
//! themes is a one-file change and no renderer has to remember RGB
//! triples. Slots are named after *meaning* (text, inactive, success,
//! accent...) not appearance, and shimmer-capable slots come in a
//! base+lit pair the streaming sweep can interpolate between.
//!
//! Defaults below are tuned for a dark terminal. Light and high-contrast
//! variants can be added later without touching callers.

use ratatui::style::{Color, Modifier, Style};

/// Schema-complete palette. Several slots aren't read by today's
/// renderer (`inverse`, `intel_lit`, `error`, `suggestion`) but
/// they're part of the theme contract — hand-tuned alongside the
/// rest. Keeps the dark/light variants symmetric.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct Theme {
    // prose
    pub text:       Color,
    pub subtle:     Color,
    pub inactive:   Color,
    pub inverse:    Color,

    // roles
    pub you:        Color,
    pub intel:      Color, // active accent; primary brand color
    pub intel_lit:  Color, // shimmer partner for `intel`
    pub sys:        Color,

    // semantic
    pub success:    Color,
    pub warning:    Color,
    pub error:      Color,
    pub suggestion: Color,

    // chrome
    pub border_active:   Color,
    pub border_inactive: Color,

    // mode badges
    pub mode_local:   Color,
    pub mode_network: Color,
    pub mode_auto:    Color,

    // browser row tags
    pub tag_local:    Color,
    pub tag_network:  Color,
    pub tag_install:  Color,

    // code block rendering
    pub code_fg:      Color,
    pub code_gutter:  Color,
}

impl Theme {
    pub const fn dark() -> Self {
        Self {
            text:     Color::Rgb(0xe6, 0xe6, 0xe6),
            subtle:   Color::Rgb(0x88, 0x88, 0x88),
            inactive: Color::Rgb(0x55, 0x55, 0x55),
            inverse:  Color::Rgb(0x00, 0x00, 0x00),

            you:        Color::Rgb(0x7a, 0xb4, 0xe8),
            intel:      Color::Rgb(0xd9, 0x77, 0x57),
            intel_lit:  Color::Rgb(0xff, 0xb0, 0x89),
            sys:        Color::Rgb(0x80, 0x80, 0x80),

            success:    Color::Rgb(0x4e, 0xba, 0x65),
            warning:    Color::Rgb(0xff, 0xc1, 0x07),
            error:      Color::Rgb(0xff, 0x6b, 0x80),
            suggestion: Color::Rgb(0xb1, 0xb9, 0xf9),

            border_active:   Color::Rgb(0xd9, 0x77, 0x57),
            border_inactive: Color::Rgb(0x88, 0x88, 0x88),

            mode_local:   Color::Rgb(0x4e, 0xba, 0x65),
            mode_network: Color::Rgb(0x79, 0xd5, 0xe6),
            mode_auto:    Color::Rgb(0x88, 0x88, 0x88),

            tag_local:   Color::Rgb(0x4e, 0xba, 0x65),
            tag_network: Color::Rgb(0x79, 0xd5, 0xe6),
            tag_install: Color::Rgb(0xc8, 0x8d, 0xe8),

            code_fg:     Color::Rgb(0xe6, 0xe6, 0xe6),
            code_gutter: Color::Rgb(0x55, 0x55, 0x55),
        }
    }
}

pub fn theme() -> Theme { Theme::dark() }

// ---------------------------------------------------------------------
// Convenience builders — callers don't need to remember modifier combos.
// ---------------------------------------------------------------------

// `text`, `inactive`, `selected` mirror the role/body builders below
// so callers can reach a palette slot without remembering which
// modifier combo goes with it. Currently only `subtle` and
// `accent_bold` are wired into the live render; the rest are kept
// as the API surface for future widgets.
#[allow(dead_code)] pub fn text()     -> Style { Style::default().fg(theme().text) }
pub fn subtle()   -> Style { Style::default().fg(theme().subtle) }
#[allow(dead_code)] pub fn inactive() -> Style { Style::default().fg(theme().inactive).add_modifier(Modifier::DIM) }

pub fn accent_bold() -> Style {
    Style::default().fg(theme().intel).add_modifier(Modifier::BOLD)
}
#[allow(dead_code)]
pub fn selected() -> Style {
    Style::default().fg(theme().intel).add_modifier(Modifier::BOLD)
}
pub fn role(role: Role) -> Style {
    let t = theme();
    match role {
        Role::You    => Style::default().fg(t.you).add_modifier(Modifier::BOLD),
        Role::Intel  => Style::default().fg(t.intel).add_modifier(Modifier::BOLD),
        Role::System => Style::default().fg(t.sys).add_modifier(Modifier::ITALIC),
    }
}
pub fn body(role: Role) -> Style {
    let t = theme();
    match role {
        Role::System => Style::default().fg(t.subtle).add_modifier(Modifier::ITALIC),
        _            => Style::default().fg(t.text),
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Role { You, Intel, System }
