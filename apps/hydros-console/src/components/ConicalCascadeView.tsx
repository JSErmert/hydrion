// apps/hydros-console/src/components/ConicalCascadeView.tsx
//
// M5 Conical Cascade machine view — Baseline 1 (locked 2026-04-11)
// Visual surface only. No metrics embedded here. η belongs to the N-layer panels.
//
// All geometric constants match machine-core-v4.html exactly.
// Dynamic layers are driven by HydrosDisplayState prop.

import type { HydrosDisplayState } from '../scenarios/displayStateMapper';

const FONT = '"JetBrains Mono", "Fira Code", "Courier New", monospace';

// ── Geometric constants (Baseline 1) ──────────────────────────────────────
const CY = 154;          // device centreline y
const BORE_TOP = 64;     // housing bore top wall y
const BORE_BOT = 244;    // housing bore bottom wall y
const BORE_H   = 180;    // BORE_BOT - BORE_TOP

void [BORE_TOP, BORE_BOT, BORE_H]; // geometric reference constants used in Tasks 5-10

// Stage x-ranges [xStart, xApexX, xApexY]
const STAGES = [
  { label: 'S1', xStart: 118, xEnd: 298, apexX: 296, apexY: 243,
    bezier: 'M 118,64 C 195,64 292,96 296,243',
    innerBezier: 'M 118,72 C 195,72 292,104 296,243',
    color: '#FB923C', weave: 'url(#weaveCoarse)', mult: 0.4,
    ejY: 252, chY: 252, chH: 16 },
  { label: 'S2', xStart: 306, xEnd: 486, apexX: 484, apexY: 243,
    bezier: 'M 306,64 C 383,64 480,96 484,243',
    innerBezier: 'M 306,72 C 383,72 480,104 484,243',
    color: '#FBBF24', weave: 'url(#weaveMedium)', mult: 0.7,
    ejY: 274, chY: 274, chH: 16 },
  { label: 'S3', xStart: 494, xEnd: 674, apexX: 672, apexY: 243,
    bezier: 'M 494,64 C 571,64 668,96 672,243',
    innerBezier: 'M 494,72 C 571,72 668,104 672,243',
    color: '#38BDF8', weave: 'url(#weaveFine)', mult: 1.0,
    ejY: 296, chY: 296, chH: 16 },
] as const;

const INLET_OVALS = [
  { cx: 118, stroke: '#2A5878' },
  { cx: 306, stroke: '#2A6878' },
  { cx: 494, stroke: '#306888' },
  { cx: 674, stroke: '#38BDF8' },
  { cx: 714, stroke: '#2A6080' },
] as const;

// ── Dynamic layer helpers ─────────────────────────────────────────────────

interface FieldLinesProps {
  xStart: number;
  xEnd:   number;
  eField: number;       // [0,1] normalised field strength
  mult:   number;       // stage multiplier: S1=0.4, S2=0.7, S3=1.0
  color:  string;
}

function RadialFieldLines({ xStart, xEnd, eField, mult, color }: FieldLinesProps) {
  const count = Math.round(eField * 14 * mult);
  if (count === 0) return null;
  const lines = [];
  for (let i = 0; i < count; i++) {
    const x    = xStart + (xEnd - xStart) * (i + 0.5) / count;
    // Lower half (toward bottom outer wall) — stronger, more physically significant
    const loOp = Math.min(0.55, 0.35 * mult * eField + 0.08);
    // Upper half (toward top outer wall) — weaker counterpart
    const upOp = loOp * 0.4;
    lines.push(
      <g key={i}>
        <line x1={x} y1={154} x2={x} y2={244}
          stroke={color} strokeWidth={0.75} opacity={loOp} />
        <line x1={x} y1={154} x2={x} y2={64}
          stroke={color} strokeWidth={0.55} opacity={upOp} />
      </g>
    );
  }
  return <>{lines}</>;
}

// ── Animated particle system ──────────────────────────────────────────────

interface AnimParticle {
  id: string;
  cx: number;      // absolute SVG x — particle start
  cy: number;      // absolute SVG y — particle start
  r: number;       // dot radius
  dx: number;      // CSS translate delta X (start → end)
  dy: number;      // CSS translate delta Y (start → end)
  opacity: number; // peak opacity during animation [0,1]
  duration: number; // loop duration in seconds
  delay: number;   // negative = staggered so stream is already in motion on mount
}

/**
 * Compute particle positions and trajectories from physics state.
 *
 * Captured particles curve toward (apexX, apexY) — nDEP deflection.
 * Escaped particles drift toward the exit edge near the centreline.
 * Buoyant escaped particles (PP-dominated escape) float slightly upward.
 * Backflush reverses dx so particles visibly move right-to-left.
 */
function buildParticles(
  stageIdx: number,
  xStart: number,
  xEnd: number,
  apexX: number,
  apexY: number,
  conc: number,
  etaStage: number,
  etaPP: number,
  etaPET: number,
  flow: number,
  backflush: boolean,
): AnimParticle[] {
  const n = Math.round(conc * 18);
  if (n === 0) return [];

  // Deterministic scatter per stage (same seed logic as static version)
  const rng = (i: number, off: number) =>
    Math.abs(Math.sin(stageIdx * 99.1 + i * 17.3 + off)) % 1;

  // Faster flow → shorter loop duration (particles move quicker)
  const duration = Math.max(0.9, 2.6 / Math.max(flow, 0.06));
  const buoyancyActive = etaPP < etaPET * 0.75 && etaPET > 0.1;

  const particles: AnimParticle[] = [];
  for (let i = 0; i < n; i++) {
    const t  = rng(i, 0);
    // Start x: left portion of stage (particles enter from left, travel right)
    const cx = xStart + (apexX - xStart) * (0.05 + t * 0.70);
    // Start y: concentration zone between centreline (154) and floor (apexY)
    const cy = 154 + (apexY - 154) * (0.05 + rng(i, 1) * 0.85);
    const r  = 1.5 + rng(i, 2) * 1.8;
    const op = 0.38 + conc * 0.42;

    const captureRng    = (Math.imul(stageIdx * 31 + 7 ^ ((stageIdx * 31 + 7) >>> 16), 0x45d9f3b) ^ Math.imul(i ^ (i >>> 11), 0x3da7f9f5)) >>> 0;
    const captured      = (captureRng / 0xffffffff) < etaStage;
    const buoyantEscape = !captured && buoyancyActive && i % 2 === 0;

    let dx: number, dy: number;
    if (backflush) {
      // Reverse flow: push particles back toward inlet
      dx = -(cx - xStart) - 20 - rng(i, 4) * 15;
      dy = (154 - cy) * 0.4;
    } else if (captured) {
      // nDEP deflection: curves down and right toward apex trap
      dx = apexX - cx + (rng(i, 4) - 0.5) * 8;
      dy = apexY - cy + (rng(i, 5) - 0.5) * 4;
    } else if (buoyantEscape) {
      // PP buoyant escape: drifts upward toward centreline while passing through
      dx = (xEnd - cx) * (0.3 + rng(i, 4) * 0.4);
      dy = (154 - cy) * (0.6 + rng(i, 5) * 0.4) - 8;
    } else {
      // Dense escaped: passes through near centreline
      dx = (xEnd - cx) * (0.35 + rng(i, 4) * 0.45);
      dy = (154 - cy) * (0.15 + rng(i, 5) * 0.25);
    }

    // Negative delay staggers particles so the stream is already mid-animation on mount
    const delay = -(rng(i, 6) * duration);

    particles.push({ id: `p${stageIdx}-${i}`, cx, cy, r, dx, dy, opacity: op, duration, delay });
  }
  return particles;
}

/** Serialise one particle's trajectory into a CSS @keyframes string. */
function particleKeyframes(p: AnimParticle): string {
  return (
    `@keyframes ${p.id}{` +
    `0%{transform:translate(0,0);opacity:0;}` +
    `8%{opacity:${p.opacity.toFixed(2)};}` +
    `88%{opacity:${p.opacity.toFixed(2)};}` +
    `100%{transform:translate(${p.dx.toFixed(1)}px,${p.dy.toFixed(1)}px);opacity:0;}}`
  );
}

interface AnimatedParticleStreamProps {
  stageIdx: number;
  xStart:   number;
  xEnd:     number;
  apexX:    number;
  apexY:    number;
  conc:     number;
  etaStage: number;  // this stage's capture efficiency (drives captured/escaped split)
  etaPP:    number;  // buoyant species — triggers buoyant escape cue
  etaPET:   number;  // dense species
  flow:     number;  // [0,1] controls animation speed; < 0.05 pauses animation
  color:    string;
  backflush: boolean;
}

function AnimatedParticleStream({
  stageIdx, xStart, xEnd, apexX, apexY,
  conc, etaStage, etaPP, etaPET, flow, color, backflush,
}: AnimatedParticleStreamProps) {
  const paused    = flow < 0.05 && !backflush;
  const particles = buildParticles(
    stageIdx, xStart, xEnd, apexX, apexY,
    conc, etaStage, etaPP, etaPET, flow, backflush,
  );
  if (particles.length === 0) return null;

  const css = particles.map(particleKeyframes).join('');

  return (
    <>
      <style>{css}</style>
      {particles.map(p => (
        <circle
          key={p.id}
          cx={p.cx}
          cy={p.cy}
          r={p.r}
          fill={color}
          style={{
            animation: `${p.id} ${p.duration.toFixed(2)}s ${p.delay.toFixed(2)}s linear infinite`,
            animationPlayState: paused ? 'paused' : 'running',
          }}
        />
      ))}
    </>
  );
}

function FlowArrow({ x, y, opacity }: { x: number; y: number; opacity: number }) {
  return (
    <path
      d={`M${x - 6},${y - 4} L${x + 6},${y} L${x - 6},${y + 4}`}
      fill="none" stroke="#38BDF8" strokeWidth={1.5} opacity={opacity}
    />
  );
}

interface ConicalCascadeViewProps {
  state: HydrosDisplayState | null;
}

export default function ConicalCascadeView({ state }: ConicalCascadeViewProps) {
  const s = state;

  return (
    <svg
      viewBox="0 0 1060 460"
      width="100%"
      height="100%"
      style={{ display: 'block' }}
      preserveAspectRatio="xMidYMid meet"
    >
      <defs>
        <radialGradient id="bg" cx="50%" cy="44%" r="62%">
          <stop offset="0%"   stopColor="#061428" />
          <stop offset="100%" stopColor="#010306" />
        </radialGradient>

        <linearGradient id="cleanFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%"   stopColor="#0C2240" stopOpacity={0.55} />
          <stop offset="100%" stopColor="#061828" stopOpacity={0.15} />
        </linearGradient>
        <linearGradient id="concFill" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%"   stopColor="#0A1C30" stopOpacity={0.6} />
          <stop offset="100%" stopColor="#040C18" stopOpacity={0.9} />
        </linearGradient>

        <radialGradient id="nodeG1" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#FB923C" stopOpacity={0.22} />
          <stop offset="100%" stopColor="#FB923C" stopOpacity={0} />
        </radialGradient>
        <radialGradient id="nodeG2" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#FBBF24" stopOpacity={0.28} />
          <stop offset="100%" stopColor="#FBBF24" stopOpacity={0} />
        </radialGradient>
        <radialGradient id="nodeG3" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#38BDF8" stopOpacity={0.45} />
          <stop offset="100%" stopColor="#38BDF8" stopOpacity={0} />
        </radialGradient>

        <radialGradient id="chamberGlow" cx="50%" cy="70%" r="60%">
          <stop offset="0%"   stopColor="#0A1C30" />
          <stop offset="100%" stopColor="#020810" />
        </radialGradient>

        <filter id="fxSoft" x="-20%" y="-30%" width="140%" height="160%">
          <feGaussianBlur in="SourceGraphic" stdDeviation={3} result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="fxStrong" x="-30%" y="-40%" width="160%" height="180%">
          <feGaussianBlur in="SourceGraphic" stdDeviation={5} result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>

        {/* Mesh weave patterns */}
        <pattern id="weaveCoarse" x="0" y="0" width="8" height="8" patternUnits="userSpaceOnUse">
          <line x1="0" y1="4" x2="8" y2="4" stroke="#2A5878" strokeWidth={0.9} opacity={0.7} />
          <line x1="4" y1="0" x2="4" y2="8" stroke="#2A5878" strokeWidth={0.9} opacity={0.7} />
        </pattern>
        <pattern id="weaveMedium" x="0" y="0" width="5" height="5" patternUnits="userSpaceOnUse">
          <line x1="0" y1="2.5" x2="5" y2="2.5" stroke="#306888" strokeWidth={0.7} opacity={0.75} />
          <line x1="2.5" y1="0" x2="2.5" y2="5" stroke="#306888" strokeWidth={0.7} opacity={0.75} />
        </pattern>
        <pattern id="weaveFine" x="0" y="0" width="3" height="3" patternUnits="userSpaceOnUse">
          <line x1="0" y1="1.5" x2="3" y2="1.5" stroke="#38BDF8" strokeWidth={0.5} opacity={0.8} />
          <line x1="1.5" y1="0" x2="1.5" y2="3" stroke="#38BDF8" strokeWidth={0.5} opacity={0.8} />
        </pattern>

        {/* Channel gradient fills */}
        <linearGradient id="tS1" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%"   stopColor="#FB923C" stopOpacity={0.08} />
          <stop offset="100%" stopColor="#FB923C" stopOpacity={0.35} />
        </linearGradient>
        <linearGradient id="tS2" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%"   stopColor="#FBBF24" stopOpacity={0.08} />
          <stop offset="100%" stopColor="#FBBF24" stopOpacity={0.3} />
        </linearGradient>
        <linearGradient id="tS3" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%"   stopColor="#38BDF8" stopOpacity={0.08} />
          <stop offset="100%" stopColor="#38BDF8" stopOpacity={0.3} />
        </linearGradient>

        <style>{`
          @keyframes bfSweep {
            0%   { transform: translateX(558px); opacity: 0.18; }
            85%  { transform: translateX(-60px);  opacity: 0.08; }
            100% { transform: translateX(-60px);  opacity: 0;    }
          }
          .bf-sweep { animation: bfSweep 1.4s linear infinite; }
          @keyframes swapPulse {
            0%, 100% { opacity: 1.0; }
            50%       { opacity: 0.45; }
          }
          .swap-warn { animation: swapPulse 1.8s ease-in-out infinite; }
        `}</style>
      </defs>

      {/* Background */}
      <rect width={1060} height={460} fill="url(#bg)" />

      {/* ── HOUSING WALLS ───────────────────────────────────────────── */}
      <line x1={36} y1={64}  x2={674} y2={64}  stroke="#1E3A5A" strokeWidth={2.2} />
      <line x1={36} y1={244} x2={674} y2={244} stroke="#1E3A5A" strokeWidth={2.2} />
      <line x1={36} y1={64}  x2={36}  y2={244} stroke="#1E3A5A" strokeWidth={2.2} />
      <text x={380} y={52} textAnchor="middle" fill="#3A6888"
        fontSize={8} fontFamily={FONT} letterSpacing={3}>
        OUTER HOUSING — CONSTANT DIAMETER
      </text>

      {/* ── IN TUBE ─────────────────────────────────────────────────── */}
      <rect x={4} y={64} width={32} height={180} fill="#030810" opacity={0.9} />
      <line x1={4} y1={64}  x2={36} y2={64}  stroke="#1E3A5A" strokeWidth={2.2} />
      <line x1={4} y1={244} x2={36} y2={244} stroke="#1E3A5A" strokeWidth={2.2} />
      <ellipse cx={4} cy={CY} rx={4} ry={90}
        fill="#060E1C" stroke="#2A5878" strokeWidth={1.5} opacity={0.9}
        filter="url(#fxSoft)" />
      <line x1={6} y1={CY} x2={34} y2={CY}
        stroke="#38BDF8" strokeWidth={1.2} opacity={0.35} strokeDasharray="4 3" />
      <text x={20} y={52} textAnchor="middle" fill="#5A90B0" fontSize={8} fontFamily={FONT}>IN</text>
      <text x={20} y={62} textAnchor="middle" fill="#3A7898" fontSize={6.5} fontFamily={FONT}>water</text>

      {/* ── FLOW VELOCITY INDICATORS ────────────────────────────────── */}
      {(() => {
        const flow = s?.flow ?? 0;
        if (flow < 0.05) return null;
        const op = 0.2 + flow * 0.5;
        const arrowXs = [77, 207, 395, 583];  // POL zone + S1/S2/S3 bore
        return arrowXs.map(x => (
          <FlowArrow key={x} x={x} y={CY} opacity={op} />
        ));
      })()}

      {/* ── POLARISATION ZONE x=36–118 ──────────────────────────────── */}
      <rect x={36} y={64} width={82} height={180} fill="#060F1C" />
      {[54, 68, 82, 96, 110].map(lx => (
        <line key={lx} x1={lx} y1={72} x2={lx} y2={236}
          stroke="#818CF8" strokeWidth={0.7} opacity={0.32} />
      ))}
      <text x={77} y={150} textAnchor="middle" fill="#38BDF8"
        fontSize={9} fontFamily={FONT} letterSpacing={1}>POL</text>
      <text x={77} y={163} textAnchor="middle" fill="#5A90B0"
        fontSize={7.5} fontFamily={FONT}>ZONE</text>
      <line x1={118} y1={64} x2={118} y2={244}
        stroke="#0D2030" strokeWidth={1} strokeDasharray="3 3" />

      {/* ── STAGE CONES ─────────────────────────────────────────────── */}
      {STAGES.map((stg) => (
        <g key={stg.label}>
          {/* Concentration zone (below outer bezier) */}
          <path d={`${stg.bezier} L ${stg.xStart},243 Z`} fill="url(#concFill)" />
          {/* Clean water zone (above outer bezier) */}
          <path d={`${stg.bezier} L ${stg.xEnd},64 Z`} fill="url(#cleanFill)" />
          {/* Mesh weave fabric */}
          <path
            d={`${stg.bezier} ${stg.innerBezier.replace('M', 'C').replace(/M \d+,\d+ /, '')} Z`}
            fill={stg.weave} opacity={0.85}
          />
          {/* Outer mesh wall — S3 opacity reduced under high flow (degradation cue) */}
          <path d={stg.bezier} stroke="#2A5878" strokeWidth={1.8} fill="none"
            opacity={stg.label === 'S3' ? Math.max(0.4, 0.9 - ((s?.flow ?? 0) - 0.8) * 1.5) : 0.9} />
          {/* Stage label above */}
          <text x={(stg.xStart + stg.xEnd) / 2} y={46}
            textAnchor="middle" fill="#5A90B0" fontSize={9} fontFamily={FONT} letterSpacing={2}>
            {stg.label === 'S1' ? 'S1 — COARSE' : stg.label === 'S2' ? 'S2 — MEDIUM' : 'S3 — FINE'}
          </text>
        </g>
      ))}

      {/* ── RADIAL E-FIELD LINES (radial, center → outer wall) ──────── */}
      {STAGES.map((stg) => (
        <RadialFieldLines
          key={`ef-${stg.label}`}
          xStart={stg.xStart}
          xEnd={stg.apexX}
          eField={s?.eField ?? 0}
          mult={stg.mult}
          color={stg.color}
        />
      ))}

      {/* ── ANIMATED PARTICLE STREAMS ───────────────────────────────── */}
      {STAGES.map((stg, i) => {
        const etaArr   = [s?.etaS1 ?? 0, s?.etaS2 ?? 0, s?.etaS3 ?? 0];
        const survival = etaArr.slice(0, i).reduce((acc, e) => acc * (1 - e), 1);
        const conc     = (s?.clog ?? 0.5) * survival;
        return (
          <AnimatedParticleStream
            key={`aps-${stg.label}`}
            stageIdx={i}
            xStart={stg.xStart}
            xEnd={stg.xEnd}
            apexX={stg.apexX}
            apexY={stg.apexY}
            conc={conc}
            etaStage={etaArr[i]}
            etaPP={s?.etaPP ?? 0}
            etaPET={s?.etaPET ?? 0}
            flow={s?.flow ?? 0}
            color={stg.color}
            backflush={(s?.backflush ?? 0) > 0.5}
          />
        );
      })}

      {/* ── INLET FACE OVALS ────────────────────────────────────────── */}
      {INLET_OVALS.map(({ cx, stroke }) => (
        <g key={cx}>
          <ellipse cx={cx} cy={CY} rx={6} ry={90}
            fill="#081C30" stroke={stroke} strokeWidth={1.8} opacity={0.9}
            filter="url(#fxSoft)" />
          <ellipse cx={cx} cy={CY} rx={2.5} ry={90}
            fill="#1A4060" opacity={0.4} />
        </g>
      ))}

      {/* ── TRANSITION BUFFER (x=674–714, FIXED) ────────────────────── */}
      <rect x={674} y={64} width={40} height={264} rx={2}
        fill="#060E1C" stroke="#2A6080" strokeWidth={1.2} />
      <rect x={676} y={64} width={36} height={180}
        fill="#030810" stroke="#1A4060" strokeWidth={0.8} />

      {/* ── SNAP-OFF SEAM (x=714) ───────────────────────────────────── */}
      <line x1={714} y1={64} x2={714} y2={328}
        stroke="#F59E0B" strokeWidth={1.5} strokeDasharray="5 3" opacity={0.8} />
      <polygon points="712,150 708,154 712,158" fill="#F59E0B" opacity={0.75} />
      <polygon points="716,150 720,154 716,158" fill="#F59E0B" opacity={0.75} />
      <text x={710} y={143} textAnchor="end" fill="#5A8AAA" fontSize={6} fontFamily={FONT}>FIXED ←</text>
      <text x={718} y={143} fill="#F59E0B" fontSize={6} fontFamily={FONT}>→ DETACH</text>

      {/* ── STORAGE CHAMBER (x=714–886, DETACHABLE) ─────────────────── */}
      <rect x={714} y={64} width={172} height={264} rx={2}
        fill="#060E1C" stroke="#38BDF8" strokeWidth={1.8} filter="url(#fxSoft)" />
      <rect x={718} y={68} width={164} height={256} rx={1} fill="url(#chamberGlow)" />
      <rect x={726} y={66} width={72} height={10} rx={2}
        fill="#0A1828" stroke="#F59E0B" strokeWidth={0.8} />
      <text x={762} y={74} textAnchor="middle" fill="#F59E0B"
        fontSize={6} fontFamily={FONT} letterSpacing={0.8}>DETACHABLE</text>
      <rect x={716} y={64} width={168} height={180}
        fill="#030810" stroke="#1A4060" strokeWidth={1} />
      <text x={800} y={146} textAnchor="middle" fill="#38BDF8"
        fontSize={8} fontFamily={FONT} letterSpacing={1}>CLEAN WATER</text>
      <text x={800} y={158} textAnchor="middle" fill="#38BDF8"
        fontSize={8} fontFamily={FONT} letterSpacing={1}>BORE</text>
      <text x={800} y={171} textAnchor="middle" fill="#5A90B0"
        fontSize={7} fontFamily={FONT}>continuous flow → outlet</text>
      <text x={770} y={292} textAnchor="middle" fill="#38BDF8"
        fontSize={7.5} fontFamily={FONT} letterSpacing={0.8}>STORAGE CHAMBER</text>
      <text x={770} y={304} textAnchor="middle" fill="#5A8AAA"
        fontSize={6} fontFamily={FONT}>particle accumulation</text>

      {/* ── STORAGE FILL SENSOR ─────────────────────────────────────── */}
      {(() => {
        const fill   = s?.storageFill ?? 0;
        const barH   = 66;
        const fillH  = Math.round(fill * barH);
        const atWarn = fill >= 0.8;
        const warnY  = 253 + barH * 0.2;   // 80% threshold marker (20% from top)
        return (
          <g>
            {/* Sensor housing */}
            <rect x={847} y={253} width={7} height={barH} rx={1.5}
              fill="#010608" stroke="#5A8AAA" strokeWidth={0.7} />
            {/* Green fill (rises from bottom) */}
            {fillH > 0 && (
              <rect x={848} y={253 + barH - fillH} width={5} height={fillH} rx={1}
                fill={atWarn ? '#F59E0B' : '#22C55E'} opacity={0.7} />
            )}
            {/* Threshold marker at 80% */}
            <line x1={843} y1={warnY} x2={856} y2={warnY}
              stroke="#F59E0B" strokeWidth={1}
              opacity={atWarn ? 1.0 : 0.55} />
            <circle cx={858} cy={warnY} r={2.5}
              fill="#F59E0B" opacity={atWarn ? 1.0 : 0.5} />
            {/* Labels */}
            <text x={851} y={251} textAnchor="middle" fill="#5A8AAA"
              fontSize={5.5} fontFamily={FONT}>FILL</text>
            <text x={841} y={warnY + 4} textAnchor="end" fill="#F59E0B"
              fontSize={5.5} fontFamily={FONT}
              className={atWarn ? 'swap-warn' : undefined}>
              ⚠ 80% swap
            </text>
          </g>
        );
      })()}

      {/* ── OUT TUBE ────────────────────────────────────────────────── */}
      <ellipse cx={886} cy={CY} rx={6} ry={90}
        fill="#060E1C" stroke="#38BDF8" strokeWidth={1.5} opacity={0.9} filter="url(#fxSoft)" />
      <ellipse cx={886} cy={CY} rx={2.5} ry={90} fill="#0A2840" opacity={0.5} />
      <rect x={886} y={64} width={48} height={180} fill="#030810" opacity={0.9} />
      <line x1={886} y1={64}  x2={934} y2={64}  stroke="#1E3A5A" strokeWidth={2.2} />
      <line x1={886} y1={244} x2={934} y2={244} stroke="#1E3A5A" strokeWidth={2.2} />
      <ellipse cx={934} cy={CY} rx={4} ry={90}
        fill="#060E1C" stroke="#38BDF8" strokeWidth={1.5} opacity={0.9} filter="url(#fxSoft)" />
      <line x1={890} y1={CY} x2={932} y2={CY}
        stroke="#38BDF8" strokeWidth={1.2} opacity={0.35} strokeDasharray="4 3" />
      <text x={910} y={52} textAnchor="middle" fill="#5A90B0" fontSize={8} fontFamily={FONT}>OUT</text>
      <text x={910} y={62} textAnchor="middle" fill="#3A7898" fontSize={6.5} fontFamily={FONT}>clean</text>

      {/* ── COLLECTION CHANNELS (fill driven by channelFillSx) ─────── */}
      <text x={395} y={249} textAnchor="middle" fill="#3A7898"
        fontSize={7} fontFamily={FONT} letterSpacing={1.2}>
        COLLECTION CHANNELS  →  BUFFER ZONE  →  STORAGE
      </text>
      {STAGES.map((stg, i) => {
        const fills   = [s?.channelFillS1 ?? 0, s?.channelFillS2 ?? 0, s?.channelFillS3 ?? 0];
        const fActive = [s?.flushActiveS1 ?? false, s?.flushActiveS2 ?? false, s?.flushActiveS3 ?? false];
        const fill    = fills[i];
        const flush   = fActive[i];
        const fillW   = Math.round(fill * 556);   // channel width = 556 (x=118 to x=674)
        const tIds    = ['tS1', 'tS2', 'tS3'];
        return (
          <g key={`ch-${stg.label}`}>
            {/* Channel tube structure */}
            <rect x={118} y={stg.chY} width={556} height={stg.chH} rx={2.5}
              fill="#030608"
              stroke={stg.color}
              strokeWidth={flush ? 1.6 : 1.0}
              opacity={flush ? 0.9 : 0.65} />
            {/* Particle fill — grows from left as accumulation increases */}
            {fillW > 2 && (
              <rect x={118} y={stg.chY + 1} width={fillW} height={stg.chH - 2} rx={2}
                fill={`url(#${tIds[i]})`} opacity={0.9} />
            )}
            {/* Channel label */}
            <text x={130} y={stg.chY + 11} fill={stg.color}
              fontSize={8} fontFamily={FONT} letterSpacing={0.5}>
              {stg.label}  COLLECTION  ·  {stg.label === 'S1' ? '500µm stage' : stg.label === 'S2' ? '100µm stage' : '5µm membrane'}
            </text>
          </g>
        );
      })}

      {/* Backflush sweep overlay — sweeps R→L through channel band when bf active */}
      {(s?.backflush ?? 0) > 0.5 && (
        <rect x={118} y={250} width={558} height={68}
          fill="#38BDF8" className="bf-sweep"
          style={{ transformOrigin: 'left center' }} />
      )}

      {/* ── FLUSH INLETS (highlight active) ─────────────────────────── */}
      {STAGES.map((stg, i) => {
        const fActive = [s?.flushActiveS1 ?? false, s?.flushActiveS2 ?? false, s?.flushActiveS3 ?? false];
        const flush   = fActive[i];
        const fy      = stg.chY + 2;
        return (
          <g key={`flush-${i}`}>
            <rect x={86} y={fy} width={18} height={12} rx={2}
              fill="#040810"
              stroke={stg.color}
              strokeWidth={flush ? 1.8 : 0.9}
              opacity={flush ? 1.0 : 0.85} />
            <text x={95} y={fy + 9} textAnchor="middle" fill={stg.color}
              fontSize={8} fontFamily={FONT}>→</text>
            <text x={83} y={fy} textAnchor="end" fill={stg.color}
              fontSize={6} fontFamily={FONT}>FLUSH</text>
            {/* Bridge: inlet → channel */}
            <line x1={104} y1={stg.chY + stg.chH / 2} x2={118} y2={stg.chY + stg.chH / 2}
              stroke="#030810" strokeWidth={7} strokeLinecap="butt" />
            <line x1={104} y1={stg.chY + stg.chH / 2} x2={118} y2={stg.chY + stg.chH / 2}
              stroke={stg.color} strokeWidth={3.5} opacity={flush ? 0.95 : 0.7} />
          </g>
        );
      })}

      {/* ── CONNECTOR PIPES (channel → seam) ────────────────────────── */}
      {STAGES.map((stg) => {
        const cy2 = stg.chY + stg.chH / 2;
        return (
          <g key={`conn-${stg.label}`}>
            <line x1={673} y1={cy2} x2={714} y2={cy2}
              stroke="#030810" strokeWidth={7} strokeLinecap="butt" />
            <line x1={673} y1={cy2} x2={714} y2={cy2}
              stroke={stg.color} strokeWidth={3.5} opacity={0.72} />
            <circle cx={714} cy={cy2} r={3.5} fill={stg.color} opacity={0.9} />
          </g>
        );
      })}

      {/* ── EJECTION PIPES (apex → channel) ─────────────────────────── */}
      {STAGES.map((stg) => (
        <g key={`ej-${stg.label}`}>
          <line x1={stg.apexX} y1={244} x2={stg.apexX} y2={stg.chY}
            stroke="#030810" strokeWidth={7} strokeLinecap="butt" />
          <line x1={stg.apexX} y1={244} x2={stg.apexX} y2={stg.chY}
            stroke={stg.color} strokeWidth={3.5} opacity={0.68} />
        </g>
      ))}

      {/* ── APEX NODES ──────────────────────────────────────────────── */}
      {(() => {
        const etas   = [s?.etaS1 ?? 0, s?.etaS2 ?? 0, s?.etaS3 ?? 0];
        const glows  = ['url(#nodeG1)', 'url(#nodeG2)', 'url(#nodeG3)'];
        const fxIds  = ['fxSoft', 'fxSoft', 'fxStrong'];
        const borders = ['#FB923C', '#FBBF24', '#38BDF8'];
        // Enforce hierarchy: S3 always >= S2 >= S1 in displayed glow
        const enforced = [
          etas[0],
          Math.max(etas[1], etas[0] * 1.1),
          Math.max(etas[2], etas[1] * 1.3, 0.2),  // S3 minimum glow 0.2
        ];
        return STAGES.map((stg, i) => {
          const sw = 1.8 + enforced[i] * 1.2;
          return (
            <g key={`node-${i}`}>
              <ellipse cx={stg.apexX} cy={stg.apexY}
                rx={32 * (0.7 + enforced[i] * 0.5)}
                ry={20 * (0.7 + enforced[i] * 0.5)}
                fill={glows[i]}
                filter={`url(#${fxIds[i]})`} />
              <circle cx={stg.apexX} cy={stg.apexY} r={9 + enforced[i] * 2}
                fill="#060E1C" stroke={borders[i]} strokeWidth={sw}
                filter={`url(#${fxIds[i]})`} />
              <text x={stg.apexX} y={stg.apexY + 5}
                textAnchor="middle" fill={borders[i]}
                fontSize={12} fontFamily={FONT}>⊕</text>
            </g>
          );
        });
      })()}

      {/* ── STAGE SPEC LABELS ───────────────────────────────────────── */}
      {STAGES.map((stg) => (
        <text key={`spec-${stg.label}`}
          x={(stg.xStart + stg.xEnd) / 2} y={322}
          textAnchor="middle" fill="#5A8AAA" fontSize={7} fontFamily={FONT}>
          {stg.label === 'S1' ? '500 µm · coarse weave · RT + nDEP'
            : stg.label === 'S2' ? '100 µm · medium weave · RT + nDEP'
            : '5 µm · microporous membrane · RT + nDEP'}
        </text>
      ))}

      {/* ── MODE ANNOTATION ─────────────────────────────────────────── */}
      <text x={800} y={340} textAnchor="middle"
        fill="#F59E0B" fontSize={6.5} fontFamily={FONT} letterSpacing={0.3}
        className={(s?.storageFill ?? 0) >= 0.8 ? 'swap-warn' : undefined}>
        M1: flush → detach  ·  M2: detach → install fresh → bf_cmd
      </text>
    </svg>
  );
}
