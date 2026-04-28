export function Footer() {
    return (
        <footer
            className="mt-32"
            style={{ borderTop: "1px solid var(--line)" }}
        >
            <div className="max-w-6xl mx-auto px-6 py-12 flex flex-wrap gap-x-8 gap-y-4 items-baseline">
                <span className="font-serif text-lg" style={{ color: "var(--strong)" }}>
                    IntelNav
                </span>
                <span className="text-sm" style={{ color: "var(--faint)" }}>
                    Apache-2.0
                </span>
                <div className="flex-1" />
                <div className="flex flex-wrap gap-x-6 gap-y-2 text-sm" style={{ color: "var(--muted)" }}>
                    <a
                        href="https://github.com/IntelNav/intelnav"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="hover:text-[var(--strong)] transition-colors"
                    >
                        GitHub
                    </a>
                    <a
                        href="https://stats.uptimerobot.com/rhi8AeGmhy"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="hover:text-[var(--strong)] transition-colors"
                    >
                        Status
                    </a>
                    <a
                        href="https://github.com/IntelNav/intelnav/blob/main/docs/architecture.md"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="hover:text-[var(--strong)] transition-colors"
                    >
                        Architecture
                    </a>
                </div>
            </div>
        </footer>
    );
}
