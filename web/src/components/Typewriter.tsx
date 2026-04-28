"use client";

import { useEffect, useState } from "react";

/** Type out `text` one char at a time with a blinking cursor. The
 * delay defaults to 28 ms/char which feels like an LLM streaming
 * tokens, not a typewriter on amphetamines. Pause between strings
 * if `text` is an array. */
export function Typewriter({
    text,
    delay = 28,
    pause = 900,
    className,
    cursor = true,
}: {
    text: string | string[];
    delay?: number;
    pause?: number;
    className?: string;
    cursor?: boolean;
}) {
    const lines = Array.isArray(text) ? text : [text];
    const [lineIdx, setLineIdx] = useState(0);
    const [charIdx, setCharIdx] = useState(0);

    useEffect(() => {
        const current = lines[lineIdx];
        if (charIdx < current.length) {
            const id = window.setTimeout(() => setCharIdx((c) => c + 1), delay);
            return () => window.clearTimeout(id);
        }
        if (lines.length > 1) {
            const id = window.setTimeout(() => {
                setLineIdx((i) => (i + 1) % lines.length);
                setCharIdx(0);
            }, pause);
            return () => window.clearTimeout(id);
        }
    }, [charIdx, lineIdx, delay, pause, lines]);

    return (
        <span className={className}>
            {lines[lineIdx].slice(0, charIdx)}
            {cursor && (
                <span
                    className="animate-blink inline-block ml-[1px]"
                    style={{ color: "var(--accent)" }}
                >
                    ▍
                </span>
            )}
        </span>
    );
}
