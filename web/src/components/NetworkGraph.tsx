"use client";

import { motion } from "framer-motion";

/** SVG visualization of a layer-range chain. Five peer nodes laid
 * out in an arc, with a flowing "hidden state" packet animating
 * along the chain edges. Pure SVG — no force graph dep, no canvas,
 * no WebGL. Animated with Framer Motion. */
export function NetworkGraph() {
    type Role = "client" | "peer" | "tail";
    type Node = { id: string; x: number; y: number; label: string; sub: string; role: Role };
    // Layout:  client (left) → A → B → C → tail (right)
    const nodes: Node[] = [
        { id: "client", x: 80,  y: 200, label: "you",    sub: "layers 0..6",   role: "client" },
        { id: "a",      x: 250, y: 110, label: "peer A", sub: "layers 6..12",  role: "peer" },
        { id: "b",      x: 460, y: 80,  label: "peer B", sub: "layers 12..18", role: "peer" },
        { id: "c",      x: 670, y: 110, label: "peer C", sub: "layers 18..24", role: "peer" },
        { id: "tail",   x: 840, y: 200, label: "peer D", sub: "layers 24..28", role: "tail" },
    ];
    const edges: [string, string][] = [
        ["client", "a"],
        ["a",      "b"],
        ["b",      "c"],
        ["c",      "tail"],
    ];
    const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));

    return (
        <div
            className="relative w-full overflow-hidden rounded-2xl"
            style={{
                background: "var(--panel)",
                border: "1px solid var(--line)",
            }}
        >
            <svg
                viewBox="0 0 920 320"
                className="w-full h-auto block"
                style={{ color: "var(--muted)" }}
            >
                <defs>
                    <linearGradient id="edge" x1="0" y1="0" x2="1" y2="0">
                        <stop offset="0%" stopColor="var(--line-2)" />
                        <stop offset="100%" stopColor="var(--accent)" stopOpacity="0.7" />
                    </linearGradient>
                    <radialGradient id="packet" cx="50%" cy="50%" r="50%">
                        <stop offset="0%"  stopColor="var(--accent-2)" stopOpacity="1" />
                        <stop offset="60%" stopColor="var(--accent)"   stopOpacity="0.6" />
                        <stop offset="100%" stopColor="var(--accent)"  stopOpacity="0" />
                    </radialGradient>
                </defs>

                {/* Edges. */}
                {edges.map(([from, to], i) => {
                    const a = byId[from], b = byId[to];
                    return (
                        <motion.line
                            key={`${from}-${to}`}
                            x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                            stroke="url(#edge)"
                            strokeWidth={2}
                            initial={{ pathLength: 0, opacity: 0 }}
                            whileInView={{ pathLength: 1, opacity: 1 }}
                            viewport={{ once: true }}
                            transition={{ duration: 0.6, delay: 0.15 * i, ease: "easeOut" }}
                        />
                    );
                })}

                {/* Flowing packet — a soft glowing dot that traces the chain
                 * end-to-end on a loop. We animate its (cx, cy) along each
                 * edge in sequence using SMIL via a single key-frames timeline
                 * realised through Framer's `animate` on the SVG attrs. */}
                <FlowingPacket nodes={nodes} edges={edges} />

                {/* Nodes. Render after edges so they sit on top. */}
                {nodes.map((n, i) => (
                    <NodeMarker key={n.id} {...n} delay={0.5 + i * 0.08} />
                ))}
            </svg>

            <div className="px-5 pb-4 pt-1 text-xs flex flex-wrap gap-x-6 gap-y-2"
                 style={{ color: "var(--muted)" }}>
                <span className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full" style={{ background: "var(--accent)" }} />
                    hidden state in flight
                </span>
                <span>edges = TCP + Noise + yamux</span>
                <span>nodes = libp2p peers · DHT-discovered</span>
            </div>
        </div>
    );
}

function NodeMarker({
    x, y, label, sub, role, delay,
}: {
    x: number; y: number; label: string; sub: string;
    role: "client" | "peer" | "tail"; delay: number;
}) {
    const fill = role === "client"
        ? "var(--accent)"
        : role === "tail"
            ? "var(--accent-2)"
            : "var(--panel-2)";
    const stroke = role === "peer" ? "var(--line-2)" : "var(--accent)";
    const labelColor = role === "peer" ? "var(--strong)" : "var(--strong)";

    return (
        <motion.g
            initial={{ scale: 0, opacity: 0 }}
            whileInView={{ scale: 1, opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay, ease: [0.16, 1, 0.3, 1] }}
            style={{ transformOrigin: `${x}px ${y}px` }}
        >
            {/* Pulse halo for the active client + tail. */}
            {role !== "peer" && (
                <motion.circle
                    cx={x} cy={y} r={28}
                    fill={fill} opacity={0.18}
                    animate={{ r: [26, 36, 26], opacity: [0.25, 0.05, 0.25] }}
                    transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut" }}
                />
            )}
            <circle cx={x} cy={y} r={20} fill={fill} stroke={stroke} strokeWidth={1.5} />
            <text
                x={x} y={y - 32}
                textAnchor="middle"
                fontSize={13}
                fontWeight={600}
                style={{ fill: labelColor, fontFamily: "var(--font-sans)" }}
            >
                {label}
            </text>
            <text
                x={x} y={y + 38}
                textAnchor="middle"
                fontSize={11}
                style={{ fill: "var(--muted)", fontFamily: "var(--font-mono)" }}
            >
                {sub}
            </text>
        </motion.g>
    );
}

function FlowingPacket({
    nodes,
    edges,
}: {
    nodes: { id: string; x: number; y: number }[];
    edges: [string, string][];
}) {
    const byId = Object.fromEntries(nodes.map((n) => [n.id, n] as const));
    // Walk every edge sequentially, then loop back. Each segment's
    // duration is proportional to its visual length so the packet
    // moves at a constant speed across a non-uniform layout.
    const segments = edges.map(([from, to]) => {
        const a = byId[from], b = byId[to];
        return { ax: a.x, ay: a.y, bx: b.x, by: b.y };
    });
    const lens = segments.map((s) => Math.hypot(s.bx - s.ax, s.by - s.ay));
    const total = lens.reduce((s, l) => s + l, 0);
    const SPEED = 280; // px / second
    const cycle = total / SPEED + 0.6; // little tail pause

    const xKeys: number[] = [];
    const yKeys: number[] = [];
    const tKeys: number[] = [];
    let acc = 0;
    segments.forEach((s) => {
        xKeys.push(s.ax); yKeys.push(s.ay); tKeys.push(acc / total);
        acc += Math.hypot(s.bx - s.ax, s.by - s.ay) / SPEED / cycle * total;
    });
    // Final point.
    const last = segments[segments.length - 1];
    xKeys.push(last.bx); yKeys.push(last.by); tKeys.push(1);

    return (
        <motion.circle
            r={10}
            fill="url(#packet)"
            initial={false}
            animate={{ cx: xKeys, cy: yKeys }}
            transition={{
                cx: { duration: cycle, repeat: Infinity, ease: "linear", times: tKeys },
                cy: { duration: cycle, repeat: Infinity, ease: "linear", times: tKeys },
            }}
        />
    );
}
