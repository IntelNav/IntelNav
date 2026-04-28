"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { Aurora } from "@/components/Aurora";
import { Typewriter } from "@/components/Typewriter";
import { TerminalMock, TerminalLine } from "@/components/TerminalMock";
import { NetworkGraph } from "@/components/NetworkGraph";

export default function Home() {
    return (
        <>
            <Hero />
            <LiveDemo />
            <NetworkSection />
            <Features />
            <InstallStrip />
        </>
    );
}

/* ────────────────────────────────────────────────────────────────── */

function Hero() {
    return (
        <section className="relative overflow-hidden">
            <Aurora />
            <div className="relative z-10 max-w-6xl mx-auto px-6 pt-24 pb-20 sm:pt-32 sm:pb-28">
                <motion.div
                    initial={{ y: 12, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
                    className="max-w-2xl"
                >
                    <p
                        className="text-sm tracking-wider uppercase mb-6"
                        style={{ color: "var(--accent)" }}
                    >
                        Decentralized inference
                    </p>
                    <h1
                        className="font-serif text-5xl sm:text-6xl lg:text-7xl leading-[1.05] tracking-tight"
                        style={{ color: "var(--strong)" }}
                    >
                        Run language models on hardware <em className="italic">that's already running.</em>
                    </h1>
                    <p
                        className="mt-7 text-lg sm:text-xl leading-relaxed max-w-xl"
                        style={{ color: "var(--muted)" }}
                    >
                        IntelNav splits a model into layer-range slices, scatters them
                        across volunteer hardware, and streams hidden states through the
                        chain to answer a prompt. <span style={{ color: "var(--strong)" }}>
                        No single peer holds the whole model.</span>
                    </p>

                    <div className="mt-10 flex flex-wrap gap-3">
                        <Link
                            href="/install/"
                            className="px-5 py-3 rounded-full text-[15px] font-medium transition-all hover:scale-[1.02] active:scale-[0.98]"
                            style={{
                                background: "var(--strong)",
                                color: "var(--bg)",
                            }}
                        >
                            Install →
                        </Link>
                        <Link
                            href="/how-it-works/"
                            className="px-5 py-3 rounded-full text-[15px] font-medium transition-colors"
                            style={{
                                background: "var(--panel)",
                                color: "var(--strong)",
                                border: "1px solid var(--line-2)",
                            }}
                        >
                            How it works
                        </Link>
                    </div>

                    <p
                        className="mt-8 text-sm font-mono"
                        style={{ color: "var(--faint)" }}
                    >
                        $&nbsp;<Typewriter
                            text={[
                                "curl -fsSL https://intelnav.net/install.sh | sh",
                                "intelnav    # opens the TUI",
                                "/models     # pick a slice to host",
                            ]}
                            delay={42}
                            pause={2400}
                        />
                    </p>
                </motion.div>
            </div>
        </section>
    );
}

/* ────────────────────────────────────────────────────────────────── */

function LiveDemo() {
    return (
        <section className="relative max-w-6xl mx-auto px-6 py-24 sm:py-32">
            <div className="max-w-2xl mb-12">
                <p
                    className="text-sm tracking-wider uppercase mb-4"
                    style={{ color: "var(--accent)" }}
                >
                    Two binaries, one chain
                </p>
                <h2
                    className="font-serif text-3xl sm:text-4xl leading-tight tracking-tight"
                    style={{ color: "var(--strong)" }}
                >
                    Chat from one terminal. Host from another.
                </h2>
                <p
                    className="mt-4 text-lg leading-relaxed"
                    style={{ color: "var(--muted)" }}
                >
                    <code>intelnav</code> is the chat client. <code>intelnav-node</code>
                    {" "}is the host daemon. They share an identity but live in separate
                    processes — closing the chat doesn't drop your slices off the
                    network.
                </p>
            </div>

            <div className="grid lg:grid-cols-2 gap-6">
                <TerminalMock title="islam@laptop ~ — intelnav">
                    <TerminalLine tag="sys" tone="muted">
                        Welcome to intelnav 0.1.0 · proto v1.
                    </TerminalLine>
                    <TerminalLine tag="sys" tone="muted">
                        peer 12D3KooW…uizeh reading the DHT.
                    </TerminalLine>
                    <div className="h-2" />
                    <TerminalLine tag="you" tone="accent">
                        write a rust function that reverses a string
                    </TerminalLine>
                    <div className="h-2" />
                    <TerminalLine tag="qwen">
                        <Typewriter
                            text={`fn reverse(s: &str) -> String {\n    s.chars().rev().collect()\n}`}
                            delay={26}
                        />
                    </TerminalLine>
                </TerminalMock>

                <TerminalMock title="islam@laptop ~ — intelnav-node" delay={0.15}>
                    <TerminalLine tag="INFO" tone="success">
                        libp2p node up · /ip4/0.0.0.0/tcp/4001
                    </TerminalLine>
                    <TerminalLine tag="INFO" tone="success">
                        DHT announces published n=2
                    </TerminalLine>
                    <TerminalLine tag="INFO" tone="muted">
                        bootstrap: dialed seed1.intelnav.net:4001
                    </TerminalLine>
                    <div className="h-2" />
                    <TerminalLine tag="peer" tone="accent">
                        accepted chain · cid=AEaibo2n…  layers [0..7)
                    </TerminalLine>
                    <TerminalLine tag="peer" tone="muted">
                        forward 12 → 24 (28 ms)
                    </TerminalLine>
                    <TerminalLine tag="peer" tone="muted">
                        forward 24 → 48 (29 ms)
                    </TerminalLine>
                    <TerminalLine tag="peer" tone="muted">
                        <Typewriter
                            text={[
                                "forward 48 → 96 (28 ms)",
                                "forward 96 → 144 (30 ms)",
                                "forward 144 → 192 (29 ms)",
                                "forward 192 → 240 (28 ms)",
                            ]}
                            delay={20}
                            pause={1400}
                        />
                    </TerminalLine>
                </TerminalMock>
            </div>
        </section>
    );
}

/* ────────────────────────────────────────────────────────────────── */

function NetworkSection() {
    return (
        <section className="max-w-6xl mx-auto px-6 py-24 sm:py-32">
            <motion.div
                initial={{ y: 12, opacity: 0 }}
                whileInView={{ y: 0, opacity: 1 }}
                viewport={{ once: true, margin: "-100px" }}
                transition={{ duration: 0.6 }}
                className="max-w-2xl mb-10"
            >
                <p
                    className="text-sm tracking-wider uppercase mb-4"
                    style={{ color: "var(--accent)" }}
                >
                    The chain
                </p>
                <h2
                    className="font-serif text-3xl sm:text-4xl leading-tight tracking-tight"
                    style={{ color: "var(--strong)" }}
                >
                    A model is a chain of layers. So is the network.
                </h2>
                <p
                    className="mt-4 text-lg leading-relaxed"
                    style={{ color: "var(--muted)" }}
                >
                    Hidden states travel from your machine through every peer that
                    owns a slice of the model, in order, until the tail produces a
                    token. Each hop is TCP + Noise + yamux; each peer's place is
                    found via Kademlia.
                </p>
            </motion.div>
            <NetworkGraph />
        </section>
    );
}

/* ────────────────────────────────────────────────────────────────── */

function Features() {
    return (
        <section className="max-w-6xl mx-auto px-6 py-24 sm:py-28">
            <motion.h2
                initial={{ y: 12, opacity: 0 }}
                whileInView={{ y: 0, opacity: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5 }}
                className="font-serif text-3xl sm:text-4xl leading-tight tracking-tight max-w-xl mb-12"
                style={{ color: "var(--strong)" }}
            >
                Built so the cost of joining stays small.
            </motion.h2>
            <div className="grid md:grid-cols-2 gap-4">
                {features.map((f, i) => (
                    <FeatureCard key={f.title} feature={f} delay={i * 0.06} />
                ))}
            </div>
        </section>
    );
}

function FeatureCard({
    feature,
    delay,
}: {
    feature: { title: string; body: string; tag: string };
    delay: number;
}) {
    return (
        <motion.div
            initial={{ y: 16, opacity: 0 }}
            whileInView={{ y: 0, opacity: 1 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.5, delay, ease: [0.16, 1, 0.3, 1] }}
            className="rounded-2xl p-6 transition-colors"
            style={{
                background: "var(--panel)",
                border: "1px solid var(--line)",
            }}
        >
            <p
                className="text-xs tracking-wider uppercase mb-3"
                style={{ color: "var(--accent)" }}
            >
                {feature.tag}
            </p>
            <h3
                className="font-serif text-xl mb-2"
                style={{ color: "var(--strong)" }}
            >
                {feature.title}
            </h3>
            <p style={{ color: "var(--muted)" }}>{feature.body}</p>
        </motion.div>
    );
}

const features = [
    {
        tag: "01",
        title: "Pipeline-parallel",
        body: "Each peer owns a contiguous layer range. Hidden states forward through every hop until the head produces a token.",
    },
    {
        tag: "02",
        title: "Kademlia DHT",
        body: "Provider records advertise (model_cid, layer_range). Clients fan out per-range lookups to assemble a chain on demand.",
    },
    {
        tag: "03",
        title: "Mandatory contribution",
        body: "Every peer hosts a slice or relays DHT routing. No leech mode — the network only works because everyone commits.",
    },
    {
        tag: "04",
        title: "Two binaries",
        body: "intelnav is the chat client. intelnav-node is the host daemon. Closing chat doesn't drop your slices.",
    },
];

/* ────────────────────────────────────────────────────────────────── */

function InstallStrip() {
    return (
        <section
            className="max-w-6xl mx-auto px-6 py-20 my-12 rounded-3xl"
            style={{
                background: "var(--panel-2)",
                border: "1px solid var(--line)",
            }}
        >
            <div className="max-w-3xl mx-auto text-center">
                <h2
                    className="font-serif text-3xl sm:text-4xl leading-tight tracking-tight"
                    style={{ color: "var(--strong)" }}
                >
                    Three lines on Linux. Pick a slice. Chat.
                </h2>
                <pre
                    className="mt-8 mx-auto max-w-xl text-left rounded-md px-5 py-4 text-[13px] leading-relaxed font-mono overflow-x-auto"
                    style={{
                        background: "var(--bg)",
                        border: "1px solid var(--line)",
                        color: "var(--fg)",
                    }}
                >
                    <span style={{ color: "var(--faint)" }}>$ </span>curl -fsSL https://intelnav.net/install.sh | sh{"\n"}
                    <span style={{ color: "var(--faint)" }}>$ </span>intelnav{"\n"}
                    <span style={{ color: "var(--faint)" }}>$ </span><span style={{ color: "var(--accent)" }}>/models</span>   <span style={{ color: "var(--faint)" }}># press &apos;c&apos; on a row to host</span>
                </pre>
                <div className="mt-8 flex justify-center gap-3">
                    <Link
                        href="/install/"
                        className="px-5 py-3 rounded-full text-[15px] font-medium transition-all hover:scale-[1.02]"
                        style={{ background: "var(--strong)", color: "var(--bg)" }}
                    >
                        Full install guide
                    </Link>
                    <a
                        href="https://github.com/IntelNav/intelnav"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="px-5 py-3 rounded-full text-[15px] font-medium"
                        style={{
                            background: "transparent",
                            color: "var(--strong)",
                            border: "1px solid var(--line-2)",
                        }}
                    >
                        Source on GitHub →
                    </a>
                </div>
            </div>
        </section>
    );
}
