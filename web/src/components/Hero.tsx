"use client";

import Link from "next/link";
import { motion, useScroll, useTransform } from "framer-motion";
import { useRef } from "react";
import { Logo } from "@/components/Logo";
import { Typewriter } from "@/components/Typewriter";

export function Hero() {
    // Scroll-driven parallax. The logo rises slightly faster than the
    // text as the user scrolls — gives the section depth without
    // being distracting.
    const ref = useRef<HTMLDivElement>(null);
    const { scrollYProgress } = useScroll({
        target: ref,
        offset: ["start start", "end start"],
    });
    const logoY    = useTransform(scrollYProgress, [0, 1], ["0%", "-30%"]);
    const logoRot  = useTransform(scrollYProgress, [0, 1], [0, -12]);
    const textY    = useTransform(scrollYProgress, [0, 1], ["0%", "-15%"]);
    const opacity  = useTransform(scrollYProgress, [0, 0.6, 1], [1, 1, 0]);

    return (
        <section
            ref={ref}
            className="relative overflow-hidden"
            style={{ minHeight: "92vh" }}
        >
            {/* indigo grid + soft aurora behind everything */}
            <div className="grid-bg absolute inset-0" aria-hidden />
            <div className="aurora-indigo" aria-hidden />

            <div className="relative z-10 max-w-6xl mx-auto px-6 pt-24 pb-32 sm:pt-32 sm:pb-40 grid lg:grid-cols-[1.2fr_1fr] gap-12 items-center">
                {/* Text column — left on lg+, top on mobile. */}
                <motion.div style={{ y: textY, opacity }} className="relative z-10">
                    <motion.p
                        initial={{ opacity: 0, y: 8 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5, delay: 0.1 }}
                        className="text-[13px] tracking-[0.2em] uppercase mb-7 font-mono"
                        style={{ color: "var(--accent)" }}
                    >
                        IntelNav · v0.1
                    </motion.p>

                    <h1
                        className="font-serif text-5xl sm:text-6xl lg:text-7xl xl:text-[5.5rem] leading-[1.02] tracking-[-0.025em]"
                        style={{ color: "var(--strong)" }}
                    >
                        <SplitWord text="The model is the" delay={0.15} />{" "}
                        <motion.span
                            initial={{ opacity: 0, y: 14 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.7, delay: 0.85, ease: [0.16, 1, 0.3, 1] }}
                            className="italic relative inline-block"
                            style={{ color: "var(--accent-3)" }}
                        >
                            network.
                        </motion.span>
                    </h1>

                    <motion.p
                        initial={{ opacity: 0, y: 8 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5, delay: 1.0 }}
                        className="mt-8 text-lg sm:text-xl leading-relaxed max-w-xl"
                        style={{ color: "var(--muted)" }}
                    >
                        IntelNav splits a model into layer-range slices and
                        scatters them across volunteer hardware. Each peer holds
                        one slice. Hidden states stream through the chain —{" "}
                        <span style={{ color: "var(--strong)" }}>
                            no single peer holds the whole model.
                        </span>
                    </motion.p>

                    <motion.div
                        initial={{ opacity: 0, y: 8 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5, delay: 1.2 }}
                        className="mt-10 flex flex-wrap items-center gap-3"
                    >
                        <Link
                            href="/install/"
                            className="px-5 py-3 rounded-full text-[15px] font-medium transition-all hover:scale-[1.03] active:scale-[0.97]"
                            style={{
                                background: "var(--accent)",
                                color: "#ffffff",
                                boxShadow: "0 8px 24px -8px rgba(99, 102, 241, 0.55)",
                            }}
                        >
                            Install →
                        </Link>
                        <Link
                            href="/how-it-works/"
                            className="px-5 py-3 rounded-full text-[15px] font-medium transition-colors hover:bg-[var(--panel-2)]"
                            style={{
                                background: "transparent",
                                color: "var(--strong)",
                                border: "1px solid var(--line-2)",
                            }}
                        >
                            How it works
                        </Link>
                    </motion.div>

                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ duration: 0.5, delay: 1.4 }}
                        className="mt-10 flex flex-wrap items-center gap-x-5 gap-y-2 text-[13px]"
                        style={{ color: "var(--faint)" }}
                    >
                        <Spec>open source</Spec>
                        <Spec>apache-2.0</Spec>
                        <Spec>linux today</Spec>
                        <Spec>mac · win soon</Spec>
                    </motion.div>

                    <motion.p
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ duration: 0.5, delay: 1.6 }}
                        className="mt-10 text-sm font-mono"
                        style={{ color: "var(--faint)" }}
                    >
                        $&nbsp;<Typewriter
                            text={[
                                "curl -fsSL https://intelnav.net/install.sh | sh",
                                "intelnav    # opens the TUI",
                                "/models     # pick a slice to host",
                            ]}
                            delay={42}
                            pause={2200}
                        />
                    </motion.p>
                </motion.div>

                {/* Logo column. Big, animated, with halo. The mark
                 * needs to feel *alive* — three layered animations:
                 *   1. entrance: slow blur-and-scale-in.
                 *   2. continuous float: gentle Y oscillation +
                 *      tilt, runs forever.
                 *   3. scroll parallax: outer wrap rises faster than
                 *      the text column on page scroll. */}
                <motion.div
                    style={{ y: logoY, rotate: logoRot, opacity }}
                    className="relative flex justify-center lg:justify-end items-center"
                >
                    <motion.div
                        initial={{ opacity: 0, scale: 0.55, rotate: -16, filter: "blur(28px)" }}
                        animate={{ opacity: 1, scale: 1,    rotate: 0,   filter: "blur(0px)" }}
                        transition={{
                            duration: 1.6,
                            delay: 0.4,
                            ease: [0.22, 1, 0.36, 1],
                            opacity: { duration: 0.9, delay: 0.4 },
                        }}
                        className="logo-glow relative"
                    >
                        {/* Inner wrap drives the continuous float.
                         * Two transforms layered: gentle vertical
                         * float (~10px) and slight rotation drift
                         * (-1.8° ↔ 1.8°). Different periods so they
                         * stay un-synced, which reads as organic. */}
                        <motion.div
                            animate={{ y: [0, -12, 0] }}
                            transition={{
                                duration: 6,
                                repeat: Infinity,
                                ease: "easeInOut",
                                delay: 1.8,
                            }}
                        >
                            <motion.div
                                animate={{ rotate: [-1.8, 1.8, -1.8] }}
                                transition={{
                                    duration: 9,
                                    repeat: Infinity,
                                    ease: "easeInOut",
                                    delay: 1.8,
                                }}
                            >
                                <motion.div
                                    animate={{ scale: [1, 1.025, 1] }}
                                    transition={{
                                        duration: 5,
                                        repeat: Infinity,
                                        ease: "easeInOut",
                                        delay: 1.8,
                                    }}
                                >
                                    <Logo size={380} className="select-none" />
                                </motion.div>
                            </motion.div>
                        </motion.div>
                    </motion.div>
                </motion.div>
            </div>

            {/* Scroll cue. Subtle, fades when scrolled. */}
            <motion.div
                style={{ opacity }}
                className="absolute bottom-8 left-1/2 -translate-x-1/2 z-10 text-[11px] tracking-[0.3em] uppercase font-mono"
            >
                <span style={{ color: "var(--faint)" }}>scroll ↓</span>
            </motion.div>
        </section>
    );
}

/* Animate each word of a phrase up from below in a stagger. */
function SplitWord({ text, delay }: { text: string; delay: number }) {
    const words = text.split(" ");
    return (
        <>
            {words.map((w, i) => (
                <motion.span
                    key={i}
                    initial={{ y: 18, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{
                        duration: 0.6,
                        delay: delay + i * 0.08,
                        ease: [0.16, 1, 0.3, 1],
                    }}
                    className="inline-block mr-[0.18em]"
                >
                    {w}
                </motion.span>
            ))}
        </>
    );
}

function Spec({ children }: { children: React.ReactNode }) {
    return (
        <span className="flex items-center gap-2">
            <span
                className="w-1 h-1 rounded-full"
                style={{ background: "var(--accent)" }}
            />
            {children}
        </span>
    );
}
