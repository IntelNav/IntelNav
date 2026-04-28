"use client";

import Link from "next/link";
import { motion } from "framer-motion";

const links = [
    { href: "/how-it-works/", label: "How it works" },
    { href: "/install/",      label: "Install" },
    { href: "/docs/",         label: "Docs" },
    { href: "/community/",    label: "Community" },
];

export function Nav() {
    return (
        <motion.header
            initial={{ y: -20, opacity: 0 }}
            animate={{ y: 0,   opacity: 1 }}
            transition={{ duration: 0.4, ease: "easeOut" }}
            className="sticky top-0 z-50"
            style={{
                background: "color-mix(in srgb, var(--bg) 78%, transparent)",
                borderBottom: "1px solid var(--line)",
                backdropFilter: "blur(12px)",
                WebkitBackdropFilter: "blur(12px)",
            }}
        >
            <nav className="max-w-6xl mx-auto px-6 h-16 flex items-center gap-8">
                <Link
                    href="/"
                    className="font-serif text-lg tracking-tight hover:opacity-80 transition-opacity"
                    style={{ color: "var(--strong)" }}
                >
                    IntelNav
                </Link>
                <ul className="hidden sm:flex gap-6 text-[14px]" style={{ color: "var(--muted)" }}>
                    {links.map((l) => (
                        <li key={l.href}>
                            <Link
                                href={l.href}
                                className="transition-colors hover:text-[var(--strong)]"
                            >
                                {l.label}
                            </Link>
                        </li>
                    ))}
                </ul>
                <div className="flex-1" />
                <a
                    href="https://github.com/IntelNav/intelnav"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm px-3 py-1.5 rounded-full transition-all hover:scale-[1.02]"
                    style={{
                        color: "var(--strong)",
                        background: "var(--panel)",
                        border: "1px solid var(--line)",
                    }}
                >
                    GitHub →
                </a>
            </nav>
        </motion.header>
    );
}
