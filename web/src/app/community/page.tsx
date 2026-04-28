import Link from "next/link";

export const metadata = { title: "Community" };

export default function Community() {
    return (
        <article className="max-w-3xl mx-auto px-6 py-16 space-y-12">
            <header>
                <p className="text-sm tracking-wider uppercase mb-3" style={{ color: "var(--accent)" }}>
                    Join in
                </p>
                <h1 className="font-serif text-4xl sm:text-5xl leading-tight tracking-tight" style={{ color: "var(--strong)" }}>
                    Community
                </h1>
                <p className="mt-2" style={{ color: "var(--muted)" }}>
                    IntelNav lives or dies by the people who run nodes and the
                    people who fix bugs. There's no full-time team — every
                    contribution lands faster the more clearly it's scoped.
                </p>
            </header>

            <section className="space-y-4">
                <h2 className="font-serif text-2xl tracking-tight" style={{ color: "var(--strong)" }}>
                    Where to find us
                </h2>
                <ul className="space-y-3">
                    <Channel
                        href="https://github.com/IntelNav/intelnav/issues"
                        label="GitHub Issues"
                        body="Bug reports, feature proposals, design discussion. The fastest place to get a fix landed."
                    />
                    <Channel
                        href="https://github.com/IntelNav/intelnav/pulls"
                        label="Pull requests"
                        body="One commit per logical change, follow CONTRIBUTING.md conventions."
                    />
                    <Channel
                        href="https://stats.uptimerobot.com/rhi8AeGmhy"
                        label="Network status"
                        body="External uptime monitor for the seed peer. If it's red, new clients can't bootstrap until it's back."
                    />
                </ul>
            </section>

            <section className="space-y-4">
                <h2 className="font-serif text-2xl tracking-tight" style={{ color: "var(--strong)" }}>
                    How to contribute
                </h2>
                <p>
                    Three flavours of contribution, ordered by how much they help:
                </p>
                <ol className="space-y-2 list-decimal pl-5" style={{ color: "var(--fg)" }}>
                    <li>
                        <strong>Run a node.</strong> Hosting a slice is the
                        single most useful thing you can do. Even one well-
                        connected peer materially improves the network's
                        coverage.
                    </li>
                    <li>
                        <strong>File a real bug.</strong> The kind of bug
                        report that includes journal output, the exact command
                        run, and the expected vs actual behaviour. See{" "}
                        <Link href="/docs/" style={{ color: "var(--accent)" }}>
                            Docs
                        </Link>{" "}
                        for the canonical log location.
                    </li>
                    <li>
                        <strong>Send a focused PR.</strong> Read{" "}
                        <a
                            href="https://github.com/IntelNav/intelnav/blob/main/CONTRIBUTING.md"
                            target="_blank"
                            rel="noopener noreferrer"
                            style={{ color: "var(--accent)" }}
                        >
                            CONTRIBUTING.md
                        </a>{" "}
                        and{" "}
                        <a
                            href="https://github.com/IntelNav/intelnav/blob/main/docs/architecture.md"
                            target="_blank"
                            rel="noopener noreferrer"
                            style={{ color: "var(--accent)" }}
                        >
                            docs/architecture.md
                        </a>{" "}
                        first. Single-purpose PRs land orders of magnitude
                        faster than sprawling refactors.
                    </li>
                </ol>
            </section>

            <section className="space-y-4">
                <h2 className="font-serif text-2xl tracking-tight" style={{ color: "var(--strong)" }}>
                    License
                </h2>
                <p style={{ color: "var(--muted)" }}>
                    Apache-2.0. Use it however you want; we'd love to hear
                    what you build with it.
                </p>
            </section>

            <div className="flex gap-6 text-sm pt-4">
                <Link href="/" className="hover:underline" style={{ color: "var(--accent)" }}>
                    ← back home
                </Link>
            </div>
        </article>
    );
}

function Channel({ href, label, body }: { href: string; label: string; body: string }) {
    return (
        <li>
            <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="block rounded-md p-4 transition-opacity hover:opacity-90"
                style={{ background: "var(--panel)", border: "1px solid var(--line)" }}
            >
                <span className="font-semibold" style={{ color: "var(--strong)" }}>{label}</span>
                <p className="text-sm mt-1" style={{ color: "var(--muted)" }}>{body}</p>
            </a>
        </li>
    );
}
