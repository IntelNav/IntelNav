"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { Logo } from "@/components/Logo";

const links = [
    { href: "/how-it-works/", label: "How it works" },
    { href: "/install/",      label: "Install" },
    { href: "/docs/",         label: "Docs" },
    { href: "/community/",    label: "Community" },
];

export function Nav() {
    return (
        <motion.header
            initial={{ y: -16, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
            className="sticky top-0 z-50"
            style={{
                background: "color-mix(in srgb, var(--bg) 70%, transparent)",
                borderBottom: "1px solid var(--line)",
                backdropFilter: "blur(14px) saturate(180%)",
                WebkitBackdropFilter: "blur(14px) saturate(180%)",
            }}
        >
            <nav className="max-w-6xl mx-auto px-6 h-16 flex items-center gap-8">
                <Link
                    href="/"
                    className="flex items-center gap-3 transition-opacity hover:opacity-80 group"
                >
                    <motion.span
                        animate={{ rotate: [-2, 2, -2] }}
                        transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
                        className="inline-block"
                    >
                        <Logo size={40} />
                    </motion.span>
                    <span
                        className="font-serif text-[20px] tracking-tight"
                        style={{ color: "var(--strong)" }}
                    >
                        IntelNav
                    </span>
                </Link>
                <ul className="hidden sm:flex gap-7 text-[14px]" style={{ color: "var(--muted)" }}>
                    {links.map((l) => (
                        <li key={l.href}>
                            <Link
                                href={l.href}
                                className="link-sweep transition-colors hover:text-[var(--strong)]"
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
                    className="text-sm px-3.5 py-1.5 rounded-full transition-all hover:scale-[1.03]"
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
