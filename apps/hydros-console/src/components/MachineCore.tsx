/**
 * components/MachineCore.tsx
 *
 * SVG machine diagram: horizontal INLET → S1 → S2 → S3 → OUTLET flow path.
 * Visual state bound entirely to HydrosDisplayState (no internal state).
 *
 * Visual elements:
 *   - Three filter stage chambers with fouling fill (bottom-up, color-coded)
 *   - Main pipe conduit connecting stages
 *   - Flow direction arrows (visible when running)
 *   - Backflush return path (dashed when idle, solid + animated arrows when active)
 *   - E-field indicator band (visible when eField > threshold)
 */

import type { HydrosDisplayState } from '../scenarios/displayStateMapper';

const FONT = '"JetBrains Mono", "Fira Code", "Courier New", monospace';

// Stage geometry
const STAGES: Array<{ x: number; label: string }> = [
  { x: 185, label: 'S1' },
  { x: 375, label: 'S2' },
  { x: 565, label: 'S3' },
];
const SW = 110;  // stage width
const SY = 82;   // stage top
const SH = 136;  // stage height

// Pipe conduit
const PY = SY + SH / 2 - 9;  // pipe top (vertically centered in stage)
const PH = 18;                 // pipe height

// Overall canvas bounds
const LEFT_X = 60;
const RIGHT_X = 820;

function foulingColor(frac: number): string {
  if (frac < 0.25) return '#34D399';
  if (frac < 0.50) return '#FBBF24';
  if (frac < 0.75) return '#FB923C';
  return '#F87171';
}

// Flow arrow: small chevron pointing right
function FlowArrow({ x, y }: { x: number; y: number }) {
  return (
    <path
      d={`M${x - 6},${y - 4} L${x + 6},${y} L${x - 6},${y + 4}`}
      fill="none"
      stroke="#38BDF8"
      strokeWidth={1.5}
      opacity={0.55}
    />
  );
}

// Backflush arrow: chevron pointing left
function BfArrow({ x, y }: { x: number; y: number }) {
  return (
    <path
      d={`M${x},${y - 4} L${x - 12},${y} L${x},${y + 4} Z`}
      fill="#38BDF8"
      opacity={0.85}
    />
  );
}

interface MachineCoreProps {
  state: HydrosDisplayState | null;
}

export default function MachineCore({ state }: MachineCoreProps) {
  const f1 = state?.foulingS1 ?? 0;
  const f2 = state?.foulingS2 ?? 0;
  const f3 = state?.foulingS3 ?? 0;
  const fouling = [f1, f2, f3];

  const bf      = (state?.backflush ?? 0) > 0.5;
  const running = state?.running ?? false;
  const eField  = state?.eField ?? 0;

  const BF_Y = SY + SH + 38;

  // Pipe segment x-ranges between stages and end caps
  const pipeSegments: Array<{ x1: number; x2: number }> = [
    { x1: LEFT_X, x2: STAGES[0].x },
    { x1: STAGES[0].x + SW, x2: STAGES[1].x },
    { x1: STAGES[1].x + SW, x2: STAGES[2].x },
    { x1: STAGES[2].x + SW, x2: RIGHT_X },
  ];

  // Flow arrow x positions (midpoints of each gap)
  const flowArrowXs = pipeSegments.map(s => Math.round((s.x1 + s.x2) / 2));

  return (
    <svg
      viewBox="0 0 880 300"
      width="100%"
      height="100%"
      style={{ display: 'block' }}
      preserveAspectRatio="xMidYMid meet"
    >
      <defs>
        <linearGradient id="pipeGrad" x1="0" y1="0" x2="0" y2="1"
          gradientUnits="objectBoundingBox">
          <stop offset="0%"   stopColor="#0E1A2A" />
          <stop offset="25%"  stopColor="#1A2E48" />
          <stop offset="50%"  stopColor="#1E3858" />
          <stop offset="75%"  stopColor="#1A2E48" />
          <stop offset="100%" stopColor="#0E1A2A" />
        </linearGradient>

        <filter id="bfGlow" x="-10%" y="-40%" width="120%" height="180%">
          <feGaussianBlur in="SourceGraphic" stdDeviation="2.5" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Background */}
      <rect width={880} height={300} fill="#080D18" />

      {/* Subtle scanlines */}
      {Array.from({ length: 25 }, (_, i) => (
        <line key={i} x1={0} y1={i * 12} x2={880} y2={i * 12}
          stroke="#FFFFFF" strokeWidth={0.3} opacity={0.02} />
      ))}

      {/* ── E-FIELD BAND ─────────────────────────────────────── */}
      {eField > 0.05 && (
        <g>
          <line
            x1={STAGES[0].x} y1={48}
            x2={STAGES[2].x + SW} y2={48}
            stroke="#818CF8" strokeWidth={1}
            strokeDasharray="3 6"
            opacity={Math.min(1, eField * 0.9 + 0.2)}
          />
          <text x={(STAGES[0].x + STAGES[2].x + SW) / 2} y={42}
            textAnchor="middle" fill="#818CF8"
            fontSize={9} fontFamily={FONT} opacity={0.85}>
            E-FIELD&nbsp;{(eField * 100).toFixed(0)}%
          </text>
        </g>
      )}

      {/* ── PIPE SEGMENTS ────────────────────────────────────── */}
      {pipeSegments.map(({ x1, x2 }, i) => (
        <rect key={i} x={x1} y={PY} width={x2 - x1} height={PH}
          fill="url(#pipeGrad)" />
      ))}

      {/* ── FLOW ARROWS ──────────────────────────────────────── */}
      {running && flowArrowXs.map((x, i) => (
        <FlowArrow key={i} x={x} y={PY + PH / 2} />
      ))}

      {/* ── INLET / OUTLET LABELS ────────────────────────────── */}
      <text x={LEFT_X - 6} y={PY + PH / 2 + 4}
        textAnchor="end" fill="#475569"
        fontSize={9} fontFamily={FONT}>INLET</text>
      <text x={RIGHT_X + 6} y={PY + PH / 2 + 4}
        textAnchor="start" fill="#475569"
        fontSize={9} fontFamily={FONT}>OUTLET</text>

      {/* ── FILTER STAGES ────────────────────────────────────── */}
      {STAGES.map(({ x, label }, i) => {
        const frac   = fouling[i];
        const fillH  = SH * frac;
        const color  = foulingColor(frac);
        const pct    = Math.round(frac * 100);

        return (
          <g key={label}>
            {/* Stage housing background (covers pipe passthrough) */}
            <rect x={x} y={SY} width={SW} height={SH} rx={2} fill="#0A0F1C" />

            {/* Fouling fill (rises from bottom) */}
            {frac > 0.005 && (
              <rect
                x={x + 3} y={SY + SH - fillH + 3}
                width={SW - 6} height={fillH - 3}
                rx={2} fill={color} opacity={0.28}
              />
            )}

            {/* Fouling boundary line */}
            {frac > 0.005 && (
              <line
                x1={x + 5} y1={SY + SH - fillH + 3}
                x2={x + SW - 5} y2={SY + SH - fillH + 3}
                stroke={color} strokeWidth={1} opacity={0.65}
              />
            )}

            {/* Pipe passthrough within stage */}
            <rect x={x} y={PY} width={SW} height={PH}
              fill="#1A2540" opacity={0.7} />

            {/* Stage housing border */}
            <rect x={x} y={SY} width={SW} height={SH}
              rx={2} fill="none" stroke="#334155" strokeWidth={1.5} />

            {/* Stage label (below) */}
            <text x={x + SW / 2} y={SY + SH + 16}
              textAnchor="middle" fill="#475569"
              fontSize={10} fontFamily={FONT}>
              {label}
            </text>

            {/* Fouling % (above) */}
            <text x={x + SW / 2} y={SY - 10}
              textAnchor="middle"
              fill={frac > 0.005 ? color : '#2D3E56'}
              fontSize={10} fontFamily={FONT}>
              {pct}%
            </text>
          </g>
        );
      })}

      {/* ── BACKFLUSH PATH ───────────────────────────────────── */}
      <g filter={bf ? 'url(#bfGlow)' : undefined}>
        {/* Vertical connects from stages down to BF rail */}
        {STAGES.map(({ x }, i) => (
          <line key={i}
            x1={x + SW / 2} y1={SY + SH}
            x2={x + SW / 2} y2={BF_Y}
            stroke={bf ? '#1E4A6E' : '#151F2E'}
            strokeWidth={1}
            strokeDasharray={bf ? undefined : '3 4'}
          />
        ))}

        {/* BF horizontal rail */}
        <line
          x1={STAGES[2].x + SW / 2} y1={BF_Y}
          x2={STAGES[0].x + SW / 2} y2={BF_Y}
          stroke={bf ? '#38BDF8' : '#1E3A5F'}
          strokeWidth={bf ? 2 : 1.5}
          strokeDasharray={bf ? undefined : '5 4'}
        />

        {/* BF flow arrows (active only) */}
        {bf && [
          STAGES[1].x + SW + 20,
          STAGES[0].x + SW + 20,
        ].map((x, i) => (
          <BfArrow key={i} x={x} y={BF_Y} />
        ))}

        {/* BF label */}
        <text
          x={(STAGES[0].x + STAGES[2].x + SW) / 2} y={BF_Y + 16}
          textAnchor="middle"
          fill={bf ? '#38BDF8' : '#263650'}
          fontSize={9} fontFamily={FONT} letterSpacing="0.1em">
          {bf ? 'BACKFLUSH ACTIVE' : 'BACKFLUSH'}
        </text>
      </g>
    </svg>
  );
}
