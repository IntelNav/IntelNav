"use client";

import { useEffect, useRef, useState } from "react";
import { gsap } from "gsap";

/** Hero-sized inlined logo with a GSAP path-trace reveal.
 *
 * Why inlined (rather than `<img src="/logo.svg">`): we need direct
 * access to every <path> to animate `strokeDashoffset`, `fillOpacity`
 * etc. An <img> is opaque to JS.
 *
 * The reveal is a three-phase GSAP timeline:
 *   1. trace — each path's outline draws itself in as an indigo
 *      stroke (stagger 0.025 s/path × 65 paths ≈ 1.6 s tail).
 *   2. fill — the indigo body blooms in (fillOpacity 0 → 1).
 *   3. settle — the stroke fades, leaving the filled mark.
 *
 * After the reveal the mark sits still. No float, no breath, no spin
 * — it's already done its work.
 */
export function LogoSVG({
    size = 360,
    className,
    /** Skip the reveal animation (useful for thumbnails / og previews). */
    instant = false,
}: {
    size?: number;
    className?: string;
    instant?: boolean;
}) {
    const containerRef = useRef<HTMLDivElement>(null);
    const [svgText, setSvgText] = useState<string | null>(null);

    // Phase 1 — load the SVG. We could inline it as JSX but the file
    // is 41 KB / 65 paths; the build-time inline turns the home page
    // chunk into a beast for no benefit. Fetch is cheap; same origin.
    useEffect(() => {
        let cancelled = false;
        fetch("/logo.svg")
            .then((r) => r.text())
            .then((text) => { if (!cancelled) setSvgText(text); })
            .catch(() => { /* SVG just won't render — better than throwing */ });
        return () => { cancelled = true; };
    }, []);

    // Phase 2 — once injected, prepare each path and run the GSAP
    // timeline. Skipped on `instant` (renders the final state).
    useEffect(() => {
        if (!svgText || !containerRef.current) return;
        const svg = containerRef.current.querySelector("svg");
        if (!svg) return;

        // Make the SVG fill its container (it ships with width="1024mm").
        svg.setAttribute("width", "100%");
        svg.setAttribute("height", "100%");

        const paths = Array.from(svg.querySelectorAll<SVGPathElement>("path"));

        // Pre-set every path: full dasharray = path length, offset =
        // length (invisible), fill transparent. Stroke uses
        // currentColor so the parent's `color` controls the indigo.
        paths.forEach((p) => {
            let len = 0;
            try {
                len = p.getTotalLength();
            } catch {
                // Pathologically broken paths don't expose a length;
                // skip the animation for those — they'll just appear
                // when their fillOpacity hits 1 below.
            }
            gsap.set(p, {
                strokeDasharray:  len,
                strokeDashoffset: len,
                fillOpacity:      0,
                stroke:           "currentColor",
                strokeWidth:      1.5,
                strokeOpacity:    instant ? 0 : 1,
            });
            if (instant) {
                gsap.set(p, { fillOpacity: 1, strokeDashoffset: 0 });
            }
        });

        if (instant) return;

        const tl = gsap.timeline({ delay: 0.45 });
        // Trace each path's outline. `each: 0.025` means the next
        // path starts 25 ms after the previous — across 65 paths the
        // trail end is about 1.6 s after the head, which lands well
        // visually (head finishes near the end of the trail).
        tl.to(paths, {
            strokeDashoffset: 0,
            duration: 1.6,
            ease: "power2.inOut",
            stagger: { each: 0.025, from: "start" },
        }).to(paths, {
            fillOpacity: 1,
            duration: 0.55,
            ease: "power1.in",
            stagger: { each: 0.018, from: "start" },
        }, "-=0.75").to(paths, {
            strokeOpacity: 0,
            duration: 0.5,
            ease: "power1.out",
        }, "-=0.25");

        return () => { tl.kill(); };
    }, [svgText, instant]);

    return (
        <div
            ref={containerRef}
            className={className}
            style={{
                width: size,
                height: size,
                color: "var(--accent)", // drives `currentColor` on each path's stroke
            }}
            // SSR-safe: empty until svgText loads, then dangerouslySet
            // wires up the inline tree. We could also pre-fetch in a
            // server component; not worth the orchestration today.
            dangerouslySetInnerHTML={svgText ? { __html: svgText } : undefined}
        />
    );
}
