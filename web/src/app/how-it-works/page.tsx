import Link from "next/link";

export const metadata = { title: "How it works" };

export default function HowItWorks() {
    return (
        <article className="max-w-3xl mx-auto px-6 py-16 space-y-12">
            <header>
                <p className="text-sm tracking-wider uppercase mb-3" style={{ color: "var(--accent)" }}>
                    Walkthrough
                </p>
                <h1 className="font-serif text-4xl sm:text-5xl leading-tight tracking-tight" style={{ color: "var(--strong)" }}>
                    How it works
                </h1>
                <p className="mt-2" style={{ color: "var(--muted)" }}>
                    A single chat turn, end to end.
                </p>
            </header>

            <section className="space-y-4">
                <h2 className="font-serif text-2xl tracking-tight" style={{ color: "var(--strong)" }}>
                    1. The user types a prompt
                </h2>
                <p>
                    The chat client (<code>intelnav</code>) tokenizes the prompt
                    and embeds it into the first hidden state. It owns the
                    embedding layer and the front slice (layers <code>0..k</code>)
                    locally.
                </p>
            </section>

            <section className="space-y-4">
                <h2 className="font-serif text-2xl tracking-tight" style={{ color: "var(--strong)" }}>
                    2. The DHT chooses a chain
                </h2>
                <p>
                    Each peer that hosts a slice publishes a provider record on
                    Kademlia, keyed by{" "}
                    <code>blake3(&quot;intelnav/shard/v1|&lt;cid&gt;|&lt;start&gt;|&lt;end&gt;&quot;)</code>.
                    The chat client fans out one DHT lookup per range it needs,
                    then ranks providers by TCP probe latency. Up to two
                    candidates per hop are kept as backups.
                </p>
            </section>

            <section className="space-y-4">
                <h2 className="font-serif text-2xl tracking-tight" style={{ color: "var(--strong)" }}>
                    3. Hidden states flow through the chain
                </h2>
                <pre
                    className="overflow-x-auto px-5 py-4 rounded-md text-[13px] leading-relaxed font-mono"
                    style={{
                        background: "var(--panel)",
                        border: "1px solid var(--line)",
                        color: "var(--muted)",
                    }}
                >
{`User → TUI → Local pipeline (layers 0..k)
                       │
                       │  ForwardHidden (CBOR-framed)
                       ▼
              peer A · layers k..m
                       │
                       ▼
              peer B · layers m..N
                       │  hidden state
                       ▼
              tail peer · head + sample
                       │  token
                       ▼
                     stream`}
                </pre>
                <p>
                    Hidden states travel as length-prefixed CBOR{" "}
                    <code>ForwardHidden</code> messages. Each peer keeps its own
                    KV cache for the session. <code>SessionInit</code> resets
                    cache state at the start of each turn.
                </p>
            </section>

            <section className="space-y-4">
                <h2 className="font-serif text-2xl tracking-tight" style={{ color: "var(--strong)" }}>
                    4. Tokens stream back to the user
                </h2>
                <p>
                    The tail peer samples a token from the final logits and
                    sends it back upstream. The chat client renders it. Loop
                    until end-of-sequence.
                </p>
            </section>

            <section className="space-y-4">
                <h2 className="font-serif text-2xl tracking-tight" style={{ color: "var(--strong)" }}>
                    Failure handling
                </h2>
                <ul className="space-y-2 list-disc pl-5" style={{ color: "var(--fg)" }}>
                    <li>
                        A hop disconnects mid-turn → chain driver swaps in the
                        next-best candidate for that slot, retries the
                        connection, continues streaming.
                    </li>
                    <li>
                        A peer wants to stop hosting → flips the slice to
                        Draining, stops re-publishing the DHT record, refuses
                        new chains. In-flight ones keep streaming until they
                        finish or hit the 5-minute force-stop.
                    </li>
                    <li>
                        A peer crashes → its provider record ages out of the
                        DHT in 30 minutes. Re-announce interval is 5 minutes,
                        so a healthy peer re-claims its place quickly.
                    </li>
                </ul>
            </section>

            <p className="text-sm" style={{ color: "var(--muted)" }}>
                For implementation details, see the architecture diagram and
                runtime sequence diagram in{" "}
                <a
                    href="https://github.com/IntelNav/intelnav/blob/main/docs/architecture.md"
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{ color: "var(--accent)" }}
                >
                    docs/architecture.md
                </a>
                .
            </p>

            <div className="flex gap-6 text-sm pt-4">
                <Link href="/install/" className="hover:underline" style={{ color: "var(--accent)" }}>
                    Install →
                </Link>
                <Link href="/" className="hover:underline" style={{ color: "var(--accent)" }}>
                    ← back home
                </Link>
            </div>
        </article>
    );
}
