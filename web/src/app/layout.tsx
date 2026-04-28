import type { Metadata } from "next";
import { Inter, Crimson_Pro, JetBrains_Mono } from "next/font/google";
import { Nav } from "@/components/Nav";
import { Footer } from "@/components/Footer";
import "./globals.css";

// Self-host through next/font: no FOUT, no privacy-leaking external
// requests at runtime. Crimson Pro stands in for Tiempos on the
// headline, Inter Variable for body text, JetBrains Mono for code.
const inter = Inter({
    subsets: ["latin"],
    display: "swap",
    variable: "--font-sans",
});
const crimson = Crimson_Pro({
    subsets: ["latin"],
    display: "swap",
    variable: "--font-serif",
    weight: ["400", "500", "600"],
});
const mono = JetBrains_Mono({
    subsets: ["latin"],
    display: "swap",
    variable: "--font-mono",
    weight: ["400", "500"],
});

export const metadata: Metadata = {
    metadataBase: new URL("https://intelnav.net"),
    title: {
        default: "IntelNav — decentralized, pipeline-parallel LLM inference",
        template: "%s · IntelNav",
    },
    description:
        "IntelNav splits a model into layer-range slices, scatters them across volunteer hardware, and streams hidden states through the chain to answer a prompt. No single peer holds the whole model.",
    openGraph: {
        title: "IntelNav",
        description:
            "Decentralized, pipeline-parallel LLM inference. No single peer holds the whole model.",
        url: "https://intelnav.net",
        siteName: "IntelNav",
        type: "website",
    },
    twitter: { card: "summary_large_image", title: "IntelNav" },
    icons: { icon: "/favicon.svg" },
};

export const viewport = {
    themeColor: [
        { media: "(prefers-color-scheme: light)", color: "#faf9f5" },
        { media: "(prefers-color-scheme: dark)",  color: "#181612" },
    ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
    return (
        <html
            lang="en"
            className={`${inter.variable} ${crimson.variable} ${mono.variable}`}
        >
            <body className="font-sans antialiased">
                <Nav />
                <main>{children}</main>
                <Footer />
            </body>
        </html>
    );
}
