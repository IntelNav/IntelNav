"use client";

import { ReactNode } from "react";
import { motion } from "framer-motion";

/** A faux-terminal card with macOS-style traffic-light chrome. The
 * `title` shows in the title bar, the `children` are the body. Use
 * `<TerminalLine>` rows inside for monospaced log-style content. */
export function TerminalMock({
    title,
    children,
    delay = 0,
}: {
    title: string;
    children: ReactNode;
    delay?: number;
}) {
    return (
        <motion.div
            initial={{ y: 16, opacity: 0 }}
            whileInView={{ y: 0, opacity: 1 }}
            viewport={{ once: true, margin: "-80px" }}
            transition={{ duration: 0.55, delay, ease: [0.16, 1, 0.3, 1] }}
            className="rounded-xl overflow-hidden shadow-xl"
            style={{
                background: "var(--panel)",
                border: "1px solid var(--line)",
                boxShadow:
                    "0 1px 0 0 var(--line) inset, 0 24px 60px -24px rgba(58, 53, 40, 0.18)",
            }}
        >
            {/* Title bar with macOS-style traffic lights. */}
            <div
                className="flex items-center gap-3 px-4 py-2.5"
                style={{
                    background: "var(--panel-2)",
                    borderBottom: "1px solid var(--line)",
                }}
            >
                <div className="flex gap-1.5">
                    <span className="w-3 h-3 rounded-full bg-[#ff5f57]" />
                    <span className="w-3 h-3 rounded-full bg-[#ffbd2e]" />
                    <span className="w-3 h-3 rounded-full bg-[#28c940]" />
                </div>
                <span
                    className="font-mono text-[12px] tracking-tight ml-2"
                    style={{ color: "var(--muted)" }}
                >
                    {title}
                </span>
            </div>
            <div className="px-5 py-4 font-mono text-[13px] leading-[1.65]">
                {children}
            </div>
        </motion.div>
    );
}

/** A single line in the mock terminal. Use `tag` to pin a colored
 * label on the left (e.g. `sys`, `INFO`, `peer`). */
export function TerminalLine({
    tag,
    tone = "default",
    children,
}: {
    tag?: string;
    tone?: "default" | "muted" | "accent" | "success" | "warn";
    children: ReactNode;
}) {
    const tagColor =
        tone === "accent"  ? "var(--accent)" :
        tone === "success" ? "#39a96b" :
        tone === "warn"    ? "#cf9237" :
        tone === "muted"   ? "var(--faint)" :
                             "var(--muted)";
    const bodyColor = tone === "muted" ? "var(--muted)" : "var(--fg)";
    return (
        <div className="flex gap-3">
            {tag && (
                <span
                    className="shrink-0 w-12 text-right select-none"
                    style={{ color: tagColor }}
                >
                    {tag}
                </span>
            )}
            <span style={{ color: bodyColor }}>{children}</span>
        </div>
    );
}
