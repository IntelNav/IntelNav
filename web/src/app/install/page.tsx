import Link from "next/link";
import { Code } from "@/components/Code";

export const metadata = { title: "Install" };

export default function Install() {
    return (
        <article className="max-w-3xl mx-auto px-6 py-16 space-y-12">
            <header>
                <p className="text-sm tracking-wider uppercase mb-3" style={{ color: "var(--accent)" }}>
                    Get started
                </p>
                <h1 className="font-serif text-4xl sm:text-5xl leading-tight tracking-tight" style={{ color: "var(--strong)" }}>
                    Install
                </h1>
                <p className="mt-2" style={{ color: "var(--muted)" }}>
                    Linux for now. macOS and Windows after the Linux flow stabilises.
                </p>
            </header>

            <section className="space-y-4">
                <h2 className="font-serif text-2xl tracking-tight" style={{ color: "var(--strong)" }}>
                    One-liner
                </h2>
                <Code prompt>{`curl -fsSL https://intelnav.net/install.sh | sh`}</Code>
                <p style={{ color: "var(--muted)" }}>
                    Detects your OS, arch, and GPU vendor; pulls the matching{" "}
                    <code>intelnav</code> + <code>libllama</code> tarballs from
                    GitHub Releases; drops binaries into{" "}
                    <code>~/.local/intelnav/bin</code>; runs{" "}
                    <code>intelnav doctor</code> at the end. Re-run to upgrade.
                </p>
            </section>

            <section className="space-y-4">
                <h2 className="font-serif text-2xl tracking-tight" style={{ color: "var(--strong)" }}>
                    From source
                </h2>
                <p>
                    If you'd rather build it yourself:
                </p>
                <Code prompt>
{`git clone https://github.com/IntelNav/intelnav
cd intelnav
bash scripts/provision.sh
cargo build --release -p intelnav-cli -p intelnav-node
bash scripts/install-libllama.sh
./target/release/intelnav`}
                </Code>
                <p style={{ color: "var(--muted)" }}>
                    Provision installs system deps + Rust. install-libllama.sh
                    auto-detects your GPU and fetches a prebuilt tarball from
                    the latest{" "}
                    <a
                        href="https://github.com/IntelNav/llama.cpp/releases"
                        target="_blank"
                        rel="noopener noreferrer"
                        style={{ color: "var(--accent)" }}
                    >
                        IntelNav/llama.cpp release
                    </a>
                    .
                </p>
            </section>

            <section className="space-y-4">
                <h2 className="font-serif text-2xl tracking-tight" style={{ color: "var(--strong)" }}>
                    First launch
                </h2>
                <p>
                    Just run <code>intelnav</code>. The TUI:
                </p>
                <ol className="space-y-2 list-decimal pl-5" style={{ color: "var(--fg)" }}>
                    <li>Writes <code>config.toml</code> with auto-picked free ports.</li>
                    <li>Generates your peer identity at <code>~/.local/share/intelnav/peer.key</code>.</li>
                    <li>Auto-discovers libllama in <code>~/.cache/intelnav/libllama/bin</code>.</li>
                    <li>Fetches the bootstrap seed list from GitHub releases.</li>
                    <li>Probes your GPU and shows the contribution gate.</li>
                </ol>
                <p style={{ color: "var(--muted)" }}>
                    Pick a slice to host (recommended) or run as a DHT relay,
                    then chat. See the{" "}
                    <a
                        href="https://github.com/IntelNav/intelnav/blob/main/docs/onboarding-user.md"
                        target="_blank"
                        rel="noopener noreferrer"
                        style={{ color: "var(--accent)" }}
                    >
                        onboarding doc
                    </a>{" "}
                    for the full first-run walkthrough.
                </p>
            </section>

            <section className="space-y-4">
                <h2 className="font-serif text-2xl tracking-tight" style={{ color: "var(--strong)" }}>
                    Supported backends
                </h2>
                <div className="overflow-x-auto">
                    <table
                        className="text-sm w-full"
                        style={{ borderCollapse: "collapse" }}
                    >
                        <thead>
                            <tr style={{ borderBottom: "1px solid var(--line)" }}>
                                <th className="text-left py-2 px-2" style={{ color: "var(--muted)" }}>Platform</th>
                                <th className="text-left py-2 px-2" style={{ color: "var(--muted)" }}>Backend</th>
                                <th className="text-left py-2 px-2" style={{ color: "var(--muted)" }}>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {backends.map((b) => (
                                <tr key={`${b.platform}-${b.backend}`} style={{ borderBottom: "1px solid var(--line)" }}>
                                    <td className="py-2 px-2">{b.platform}</td>
                                    <td className="py-2 px-2"><code>{b.backend}</code></td>
                                    <td className="py-2 px-2" style={{ color: "var(--muted)" }}>{b.status}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </section>

            <div className="flex gap-6 text-sm pt-4">
                <Link href="/how-it-works/" className="hover:underline" style={{ color: "var(--accent)" }}>
                    ← How it works
                </Link>
                <Link href="/docs/" className="hover:underline" style={{ color: "var(--accent)" }}>
                    Docs →
                </Link>
            </div>
        </article>
    );
}

const backends = [
    { platform: "Linux x86_64", backend: "cpu",    status: "shipping" },
    { platform: "Linux x86_64", backend: "vulkan", status: "shipping" },
    { platform: "Linux x86_64", backend: "rocm",   status: "shipping (15 GPU archs native)" },
    { platform: "Linux x86_64", backend: "cuda",   status: "shipping (sm_86, sm_89)" },
    { platform: "macOS arm64",  backend: "metal",  status: "tarball ships, runtime path WIP" },
    { platform: "Windows x64",  backend: "vulkan", status: "tarball ships, runtime path WIP" },
];
