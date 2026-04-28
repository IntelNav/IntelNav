import Link from "next/link";

export const metadata = { title: "Docs" };

export default function Docs() {
    return (
        <article className="max-w-3xl mx-auto px-6 py-16 space-y-12">
            <header>
                <p className="text-sm tracking-wider uppercase mb-3" style={{ color: "var(--accent)" }}>
                    Reference
                </p>
                <h1 className="font-serif text-4xl sm:text-5xl leading-tight tracking-tight" style={{ color: "var(--strong)" }}>
                    Docs
                </h1>
                <p className="mt-2" style={{ color: "var(--muted)" }}>
                    Authoritative documentation lives in the repo so it stays in
                    lockstep with the code. Quick index below.
                </p>
            </header>

            <section>
                <h2 className="font-serif text-2xl tracking-tight mb-4" style={{ color: "var(--strong)" }}>
                    Onboarding
                </h2>
                <ul className="space-y-3">
                    {onboarding.map((d) => (
                        <DocLink key={d.href} {...d} />
                    ))}
                </ul>
            </section>

            <section>
                <h2 className="font-serif text-2xl tracking-tight mb-4" style={{ color: "var(--strong)" }}>
                    Architecture &amp; specs
                </h2>
                <ul className="space-y-3">
                    {architecture.map((d) => (
                        <DocLink key={d.href} {...d} />
                    ))}
                </ul>
            </section>

            <section>
                <h2 className="font-serif text-2xl tracking-tight mb-4" style={{ color: "var(--strong)" }}>
                    Contributing
                </h2>
                <ul className="space-y-3">
                    {contributing.map((d) => (
                        <DocLink key={d.href} {...d} />
                    ))}
                </ul>
            </section>

            <p className="text-sm" style={{ color: "var(--muted)" }}>
                Something missing? <Link href="/community/" style={{ color: "var(--accent)" }}>
                    Open an issue
                </Link>.
            </p>
        </article>
    );
}

function DocLink({ href, title, blurb }: { href: string; title: string; blurb: string }) {
    return (
        <li>
            <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="block rounded-md p-4 transition-colors hover:opacity-90"
                style={{
                    background: "var(--panel)",
                    border: "1px solid var(--line)",
                }}
            >
                <div className="flex items-baseline gap-2">
                    <span className="font-semibold" style={{ color: "var(--strong)" }}>{title}</span>
                    <span className="text-xs" style={{ color: "var(--faint)" }}>↗</span>
                </div>
                <p className="text-sm mt-1" style={{ color: "var(--muted)" }}>{blurb}</p>
            </a>
        </li>
    );
}

const REPO = "https://github.com/IntelNav/intelnav/blob/main";

const onboarding = [
    {
        href: `${REPO}/docs/onboarding-user.md`,
        title: "First run",
        blurb: "What launching intelnav for the first time looks like — the contribution gate, hosting a slice, relay-only mode.",
    },
    {
        href: `${REPO}/docs/onboarding-host.md`,
        title: "Host a slice",
        blurb: "Deeper walkthrough for committing real hardware to the swarm: contribute flow, /service install, drain protocol.",
    },
];

const architecture = [
    {
        href: `${REPO}/docs/architecture.md`,
        title: "Architecture",
        blurb: "Workspace dependency diagram, two-binary split, runtime sequence diagram of a single chat turn, DHT shard index, identity model.",
    },
    {
        href: `${REPO}/specs/protocol-v1.md`,
        title: "Protocol v1",
        blurb: "Wire-level Msg envelope, framing, DHT key derivation. Normative — changes here require a proto_ver bump.",
    },
    {
        href: `${REPO}/specs/security-v1.md`,
        title: "Security model v1",
        blurb: "Identity, prompt confidentiality, quorum, transport, threat-model table marking what is and isn't protected today.",
    },
];

const contributing = [
    {
        href: `${REPO}/CONTRIBUTING.md`,
        title: "CONTRIBUTING.md",
        blurb: "First-time setup, dev loop, workspace rules, layer-split bit-identical guarantee, how to add a crate.",
    },
    {
        href: `${REPO}`,
        title: "Source code",
        blurb: "10-crate Rust workspace + Next.js site under web/. README at the root has the layout map.",
    },
];
