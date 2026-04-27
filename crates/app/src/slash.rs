//! Slash-command catalog + a tiny prefix-match engine for the input
//! autocomplete overlay.
//!
//! The catalog is static and hand-curated — we only want to suggest
//! commands that actually exist in the REPL's `handle_slash` dispatch.

#[derive(Debug, Clone, Copy)]
pub struct SlashCmd {
    pub name: &'static str,
    pub args: &'static str,   // e.g. "<local|network>", or ""
    pub help: &'static str,
}

pub const OVERLAY_MAX_ITEMS: usize = 5;

pub const COMMANDS: &[SlashCmd] = &[
    SlashCmd { name: "help",   args: "",                    help: "list commands" },
    SlashCmd { name: "clear",  args: "",                    help: "clear the transcript" },
    SlashCmd { name: "mode",   args: "<local|network>",     help: "pick backend mode" },
    SlashCmd { name: "model",  args: "<name>",              help: "switch active model" },
    SlashCmd { name: "models", args: "",                    help: "open the model picker" },
    SlashCmd { name: "doctor", args: "",                    help: "runtime snapshot" },
    SlashCmd { name: "quorum", args: "<n>",                 help: "set replication quorum" },
    SlashCmd { name: "tier",   args: "<lan|cont|wan>",      help: "set network tier" },
    SlashCmd { name: "peers",  args: "[host:port,... split,...]", help: "configure pipeline peer chain" },
    SlashCmd { name: "draft",  args: "[path.gguf k]",       help: "enable speculative decoding" },
    SlashCmd { name: "wire",   args: "<fp16|int8>",         help: "activation dtype on the chain" },
    SlashCmd { name: "quit",   args: "",                    help: "exit the app" },
];

/// Given the current input buffer, return up to `OVERLAY_MAX_ITEMS`
/// matching commands. Matching is case-insensitive prefix on the
/// command name. Returns empty if the input isn't in a state where
/// the overlay should be visible (not starting with `/`, or has a
/// space already — meaning the user has moved on to arguments).
pub fn suggest(input: &str) -> Vec<&'static SlashCmd> {
    let Some(rest) = input.strip_prefix('/') else { return Vec::new(); };
    if rest.contains(char::is_whitespace) { return Vec::new(); }
    let needle = rest.to_ascii_lowercase();
    COMMANDS.iter()
        .filter(|c| c.name.starts_with(&needle))
        .take(OVERLAY_MAX_ITEMS)
        .collect()
}
