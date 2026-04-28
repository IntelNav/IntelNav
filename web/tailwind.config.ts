import type { Config } from "tailwindcss";

const config: Config = {
    content: ["./src/**/*.{ts,tsx,mdx}"],
    darkMode: "media",
    theme: {
        extend: {
            fontFamily: {
                sans:   ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
                serif:  ["var(--font-serif)", "ui-serif", "Georgia", "serif"],
                mono:   ["var(--font-mono)", "ui-monospace", "Menlo", "monospace"],
            },
            colors: {
                // Warm cream palette inspired by claude.ai's surface tones —
                // not a copy, just the same family of warm off-whites and
                // deep saturated browns rather than the cool github-grey we
                // had before.
                cream: {
                    50:  "#faf9f5",
                    100: "#f5f2e7",
                    200: "#ebe6d4",
                    300: "#d8d2bb",
                    400: "#a8a08a",
                    500: "#6e6856",
                    600: "#3a3528",
                    700: "#26221a",
                    800: "#181612",
                    900: "#0e0d09",
                },
                // Single accent — warm amber, used sparingly.
                amber: {
                    400: "#d97757",
                    500: "#c25e3f",
                    600: "#9a4527",
                },
            },
            keyframes: {
                "fade-in-up": {
                    "0%":   { opacity: "0", transform: "translateY(8px)" },
                    "100%": { opacity: "1", transform: "translateY(0)" },
                },
                "blink": {
                    "0%, 49%": { opacity: "1" },
                    "50%, 100%": { opacity: "0" },
                },
                "aurora": {
                    "0%, 100%": { transform: "translate3d(0,0,0) scale(1)" },
                    "50%":      { transform: "translate3d(2%, -1%, 0) scale(1.05)" },
                },
            },
            animation: {
                "fade-in-up": "fade-in-up 0.6s ease-out forwards",
                "blink":      "blink 1s step-start infinite",
                "aurora":     "aurora 14s ease-in-out infinite",
            },
        },
    },
    plugins: [],
};

export default config;
