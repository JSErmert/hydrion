import React from "react";

export type SystemCutawayProps = {
  running?: boolean; // true = flow anim active, false = idle glow only
  flow?: number; // 0..1
  clog?: number; // 0..1
  eField?: number; // 0..1
  backflush?: number; // 0..1 (0 none, 1 active)
  storageFill?: number; // 0..1
  width?: number | string;
  height?: number | string;
};

function clamp01(x: number) {
  if (!Number.isFinite(x)) return 0;
  return Math.max(0, Math.min(1, x));
}

export default function SystemCutaway({
  running = false,
  flow = 0.25,
  clog = 0.05,
  eField = 0.5,
  backflush = 0.0,
  storageFill = 0.25,
  width = "100%",
  height = "100%",
}: SystemCutawayProps) {
  const FLOW = clamp01(flow);
  const CLOG = clamp01(clog);
  const EF = clamp01(eField);
  const BF = clamp01(backflush);
  const FILL = clamp01(storageFill);

  // Visual mapping (tuned for "industrial/navy blueprint")
  const idleGlow = 0.25 + 0.35 * EF; // subtle "powered" feel
  const ringGlow = 0.15 + 0.65 * EF; // polarization ring intensity
  const clogShade = 0.05 + 0.85 * CLOG; // mesh darkening
  const flowSpeed = 0.6 + 2.2 * FLOW; // animation speed scalar
  const backflushActive = BF > 0.2;

  // Storage fill height in local coordinate space (reservoir box from y=610..710)
  const fillTopY = 710 - 100 * FILL;

  return (
    <div
      style={{
        width,
        height,
        position: "relative",
        borderRadius: 16,
        overflow: "hidden",
        background:
          "radial-gradient(1200px 700px at 50% 35%, rgba(38, 94, 140, 0.18), rgba(5, 12, 22, 0.92) 70%)",
        border: "1px solid rgba(120,180,255,0.10)",
        boxShadow: "0 0 0 1px rgba(10,20,40,0.6) inset",
      }}
    >
      <svg
        viewBox="0 0 520 780"
        style={{ width: "100%", height: "100%", display: "block" }}
      >
        <defs>
          <linearGradient id="steel" x1="0" x2="1">
            <stop offset="0" stopColor="rgba(220,235,255,0.10)" />
            <stop offset="0.45" stopColor="rgba(220,235,255,0.04)" />
            <stop offset="1" stopColor="rgba(220,235,255,0.12)" />
          </linearGradient>

          <linearGradient id="glass" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0" stopColor="rgba(120,180,255,0.10)" />
            <stop offset="1" stopColor="rgba(120,180,255,0.03)" />
          </linearGradient>

          <linearGradient id="flowGrad" x1="0" x2="1">
            <stop offset="0" stopColor="rgba(70,200,255,0.0)" />
            <stop offset="0.5" stopColor="rgba(70,200,255,0.35)" />
            <stop offset="1" stopColor="rgba(70,200,255,0.0)" />
          </linearGradient>

          <filter id="softGlow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feColorMatrix
              in="blur"
              type="matrix"
              values="
                1 0 0 0 0
                0 1 0 0 0
                0 0 1 0 0
                0 0 0 0.9 0"
            />
            <feMerge>
              <feMergeNode />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>

          {/* Flow animation: dashed stroke moves "down" unless backflush is active */}
          <style>{`
            .blueprint-line { stroke: rgba(140, 200, 255, 0.20); stroke-width: 1; shape-rendering: geometricPrecision; }
            .blueprint-strong { stroke: rgba(170, 220, 255, 0.28); stroke-width: 1.25; }
            .label { fill: rgba(190, 230, 255, 0.75); font-size: 12px; font-family: ui-sans-serif, system-ui; letter-spacing: 0.10em; }
            .sub { fill: rgba(190, 230, 255, 0.50); font-size: 11px; font-family: ui-sans-serif, system-ui; }
            .hud { fill: rgba(80, 170, 255, 0.85); font-size: 13px; font-family: ui-sans-serif, system-ui; letter-spacing: 0.12em; }
            .panel { fill: rgba(10, 22, 40, 0.35); stroke: rgba(120, 180, 255, 0.12); stroke-width: 1; }
            .steel { fill: url(#steel); stroke: rgba(160, 210, 255, 0.18); stroke-width: 1.25; }
            .glass { fill: url(#glass); stroke: rgba(120, 180, 255, 0.18); stroke-width: 1; }

            .idlePulse {
              opacity: ${idleGlow.toFixed(3)};
              animation: idlePulse 2.8s ease-in-out infinite;
            }
            @keyframes idlePulse {
              0%, 100% { opacity: ${idleGlow.toFixed(3)}; }
              50% { opacity: ${(idleGlow + 0.1).toFixed(3)}; }
            }

            .ringGlow {
              opacity: ${ringGlow.toFixed(3)};
              filter: url(#softGlow);
            }

            .flowPath {
              stroke: url(#flowGrad);
              stroke-width: 5;
              stroke-linecap: round;
              stroke-dasharray: 10 14;
              opacity: ${running ? 0.55 : 0.18};
              animation: ${running ? "flowDash" : "none"} ${(
                2.4 / flowSpeed
              ).toFixed(3)}s linear infinite;
            }
            @keyframes flowDash {
              from { stroke-dashoffset: ${backflushActive ? "-260" : "260"}; }
              to { stroke-dashoffset: ${backflushActive ? "260" : "-260"}; }
            }

            .meshFill {
              fill: rgba(10, 20, 35, ${clogShade.toFixed(3)});
              stroke: rgba(140, 200, 255, 0.14);
              stroke-width: 1;
            }

            .warningFlash {
              opacity: ${backflushActive ? 0.9 : 0.0};
              animation: ${backflushActive ? "warn" : "none"} 0.65s ease-in-out infinite;
            }
            @keyframes warn {
              0%,100% { opacity: 0.1; }
              50% { opacity: 0.9; }
            }
          `}</style>
        </defs>

        {/* subtle grid */}
        {Array.from({ length: 13 }).map((_, i) => (
          <line
            key={`v${i}`}
            x1={40 + i * 35}
            y1={40}
            x2={40 + i * 35}
            y2={740}
            className="blueprint-line"
            opacity="0.18"
          />
        ))}
        {Array.from({ length: 16 }).map((_, i) => (
          <line
            key={`h${i}`}
            x1={40}
            y1={40 + i * 45}
            x2={480}
            y2={40 + i * 45}
            className="blueprint-line"
            opacity="0.14"
          />
        ))}

        {/* outer housing */}
        <rect x="140" y="90" width="240" height="600" rx="26" className="steel" />
        <rect x="162" y="120" width="196" height="540" rx="18" className="glass" />

        {/* inlet / outlet pipes */}
        <path d="M90 160 H140" className="blueprint-strong" />
        <path d="M380 630 H430" className="blueprint-strong" />
        <text x="70" y="150" className="sub">
          INLET
        </text>
        <text x="438" y="640" className="sub">
          OUTLET
        </text>

        {/* Flow line (center) */}
        <path
          d="M140 160 H168 C185 160 190 150 200 140
             C210 130 220 130 230 140
             C260 170 260 200 260 220
             V595
             C260 620 280 630 300 630
             H380"
          className="flowPath"
        />

        {/* polarization ring */}
        <g>
          <rect x="176" y="135" width="168" height="34" rx="10" className="panel" />
          <rect
            x="184"
            y="142"
            width="152"
            height="20"
            rx="9"
            fill="rgba(50,120,180,0.16)"
          />
          <rect
            x="184"
            y="142"
            width="152"
            height="20"
            rx="9"
            className="ringGlow"
            fill="rgba(70,200,255,0.25)"
          />
          <text x="182" y="128" className="label">
            POLARIZATION RING
          </text>
        </g>

        {/* stages */}
        <g>
          <text x="182" y="205" className="label">
            STAGE 1
          </text>
          <text x="182" y="220" className="sub">
            COARSE MESH
          </text>
          <rect x="182" y="235" width="156" height="85" rx="10" className="panel" />
          <rect x="190" y="243" width="140" height="69" rx="8" className="meshFill" />
          {Array.from({ length: 10 }).map((_, i) => (
            <line
              key={`m1${i}`}
              x1={195 + i * 14}
              y1={245}
              x2={195 + i * 14}
              y2={310}
              stroke="rgba(180,230,255,0.16)"
              strokeWidth="1"
            />
          ))}
        </g>

        <g>
          <text x="182" y="355" className="label">
            STAGE 2
          </text>
          <text x="182" y="370" className="sub">
            MEDIUM MESH
          </text>
          <rect x="182" y="385" width="156" height="90" rx="10" className="panel" />
          <rect x="190" y="393" width="140" height="74" rx="8" className="meshFill" />
          {Array.from({ length: 14 }).map((_, i) => (
            <line
              key={`m2${i}`}
              x1={192}
              y1={398 + i * 5}
              x2={328}
              y2={398 + i * 5}
              stroke="rgba(180,230,255,0.12)"
              strokeWidth="1"
            />
          ))}
        </g>

        <g>
          <text x="182" y="515" className="label">
            STAGE 3
          </text>
          <text x="182" y="530" className="sub">
            FINE PLEATED
          </text>
          <rect x="182" y="545" width="156" height="80" rx="10" className="panel" />
          {/* pleats */}
          {Array.from({ length: 18 }).map((_, i) => (
            <path
              key={`p${i}`}
              d={`M${190 + i * 8} 550
                 C${192 + i * 8} 560, ${192 + i * 8} 610, ${190 + i * 8} 620`}
              stroke={`rgba(180,230,255,${0.1 + 0.1 * (1 - CLOG)})`}
              strokeWidth="1"
              fill="none"
            />
          ))}
          <rect
            x="190"
            y="553"
            width="140"
            height="67"
            rx="8"
            fill={`rgba(10,20,35,${0.1 + 0.55 * CLOG})`}
          />
        </g>

        {/* backflush channel indicator */}
        <g className="warningFlash">
          <rect
            x="170"
            y="635"
            width="180"
            height="24"
            rx="8"
            fill="rgba(255,140,70,0.18)"
            stroke="rgba(255,160,90,0.35)"
          />
          <text x="184" y="652" className="sub" fill="rgba(255,200,150,0.85)">
            BACKFLUSH ACTIVE
          </text>
        </g>

        {/* storage reservoir */}
        <g>
          <text x="182" y="675" className="label">
            RESERVOIR
          </text>
          <rect x="182" y="610" width="156" height="100" rx="12" className="panel" />
          {/* fill */}
          <clipPath id="fillClip">
            <rect x="186" y="614" width="148" height="92" rx="10" />
          </clipPath>
          <g clipPath="url(#fillClip)">
            <rect
              x="186"
              y={fillTopY}
              width="148"
              height={710 - fillTopY}
              fill="rgba(70,200,255,0.15)"
            />
            {/* particulate texture */}
            {Array.from({ length: 40 }).map((_, i) => (
              <circle
                key={`b${i}`}
                cx={190 + (i * 7) % 140}
                cy={620 + (i * 17) % 88}
                r={0.8 + (i % 3) * 0.4}
                fill="rgba(180,230,255,0.12)"
                opacity={0.25}
              />
            ))}
          </g>
          <text x="300" y="700" className="hud">
            {Math.round(FILL * 100)}%
          </text>
        </g>

        {/* status header */}
        <text x="210" y="78" className="hud">
          {running ? "SYSTEM LIVE" : "SYSTEM READY"}
        </text>

        {/* idle glow overlay */}
        <rect
          x="162"
          y="120"
          width="196"
          height="540"
          rx="18"
          className="idlePulse"
          fill="rgba(70,200,255,0.06)"
        />
      </svg>
    </div>
  );
}
