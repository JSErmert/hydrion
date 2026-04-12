# Particle Animation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the static `ParticleStream` component in `ConicalCascadeView.tsx` with a physics-driven animated version where particles flow, deflect toward the apex, and reverse on backflush.

**Architecture:** A pure CSS keyframe approach — `buildParticles()` computes each particle's start position and destination delta from display state, `particleKeyframes()` serialises those into CSS `@keyframes` strings injected via an inline `<style>` element. Duration is derived from `flow`, trajectory split (captured vs escaped) from `etaStage`, buoyancy split from `etaPP/etaPET`, and direction reversal from `backflush`. No external animation library, no `requestAnimationFrame`, no React state — all motion lives in CSS.

**Tech Stack:** React 18 / TypeScript 5 / SVG CSS animations / `npx tsc --noEmit` (type verification)

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Modify | `apps/hydros-console/src/components/ConicalCascadeView.tsx` | Replace static `ParticleStream` + `ParticleStreamProps` with animated equivalents; update JSX mount |

No other files change.

---

## Task 1: Replace ParticleStream with AnimatedParticleStream

Replaces the static dot renderer (lines 80–123) with a CSS-animated version. Every particle gets a unique `@keyframes` block encoding its trajectory from start position to captured apex or escaped exit. Duration is flow-speed-driven. The old `ParticleStream` function and interface are deleted.

**Files:**
- Modify: `apps/hydros-console/src/components/ConicalCascadeView.tsx`

- [ ] **Step 1: Confirm tsc is clean before touching anything**

```bash
cd C:/Users/JSEer/hydrOS/apps/hydros-console && npx tsc --noEmit
```

Expected: no output (zero errors).

- [ ] **Step 2: Delete the old ParticleStream interface and function**

In `ConicalCascadeView.tsx`, delete lines 80–123 in their entirety (the `ParticleStreamProps` interface and the `ParticleStream` function). Do not touch anything above line 80 or below line 123.

After deletion the file should jump from the closing `}` of `RadialFieldLines` directly to `function FlowArrow`.

- [ ] **Step 3: Insert the animated particle system in the same location**

At exactly where lines 80–123 were, insert:

```tsx
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

    const captured      = rng(i, 3) < etaStage;
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
```

- [ ] **Step 4: Update the JSX particle stream block**

Find the JSX block that starts with `{/* ── PARTICLE STREAMS + DENSITY SPLIT CUE */}` and runs through its closing `})}`. Replace the entire block (including the comment) with:

```tsx
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
```

- [ ] **Step 5: Run tsc — must be clean**

```bash
cd C:/Users/JSEer/hydrOS/apps/hydros-console && npx tsc --noEmit
```

Expected: no output (zero errors). If TypeScript complains about `JSX.Element` or namespace issues, change `const particles: AnimParticle[] = []` — the type annotation is correct and should not need changes.

- [ ] **Step 6: Commit**

```bash
cd C:/Users/JSEer/hydrOS
git add apps/hydros-console/src/components/ConicalCascadeView.tsx
git commit -m "$(cat <<'EOF'
feat(console): animated particle streams — flow-driven, nDEP capture, backflush reversal

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Rebuild frontend and redeploy

Rebuilds the production bundle (which is what FastAPI serves) and restarts the backend so the phone can see the animated version.

**Files:**
- No code changes — build + process management only

- [ ] **Step 1: Build production bundle**

```bash
cd C:/Users/JSEer/hydrOS/apps/hydros-console && npm run build 2>&1 | tail -8
```

Expected:
```
✓ 42 modules transformed.
dist/index.html                  0.32 kB │ gzip:  0.23 kB
dist/assets/index-*.css          ...
dist/assets/index-*.js           ...
✓ built in <N>ms
```

Zero errors.

- [ ] **Step 2: Kill the running backend**

```bash
python -c "
import subprocess
r = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
for line in r.stdout.splitlines():
    if ':8000' in line and 'LISTENING' in line:
        pid = line.split()[-1]
        print('killing PID', pid)
        subprocess.run(['taskkill', '/F', '/PID', pid])
        break
"
```

Expected: `killing PID <N>` followed by success message.

- [ ] **Step 3: Restart the backend**

```bash
cd C:/Users/JSEer/hydrOS && uvicorn hydrion.service.app:app --host 0.0.0.0 --port 8000 > /tmp/backend.log 2>&1 &
sleep 3 && cat /tmp/backend.log
```

Expected:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

- [ ] **Step 4: Confirm the frontend is served**

```bash
curl -s http://127.0.0.1:8000/ | head -3
```

Expected:
```
<!DOCTYPE html>
<html>
  <head>
```

- [ ] **Step 5: Verify ngrok is still running and get URL**

```bash
curl -s http://127.0.0.1:4040/api/tunnels | python -c "
import sys, json
t = json.load(sys.stdin)['tunnels']
print(t[0]['public_url'], '->', t[0]['config']['addr'])
"
```

Expected: `https://unwired-pasquale-conciliatory.ngrok-free.dev -> http://localhost:8000`

If ngrok is no longer running (tunnel died), restart it:
```bash
ngrok http 8000 --log=stdout > /tmp/ngrok.log 2>&1 &
sleep 5 && curl -s http://127.0.0.1:4040/api/tunnels | python -c "import sys,json; t=json.load(sys.stdin)['tunnels']; print(t[0]['public_url'])"
```

- [ ] **Step 6: Commit is already done in Task 1 — report complete**

Report the ngrok URL so the user can open it on their phone.

---

## Self-Review

### 1. Spec coverage

| Requirement | Covered by |
|---|---|
| Particles flow left→right driven by `flow` | Task 1 — `duration = 2.6 / flow`, dx positive for captured/escaped |
| Deflect toward apex (nDEP) | Task 1 — `captured` branch: `dx = apexX - cx`, `dy = apexY - cy` |
| Capture rate from `etaStage` | Task 1 — `captured = rng(i, 3) < etaStage` |
| Buoyancy split (PP escapes upward) | Task 1 — `buoyantEscape` branch: `dy` floats toward centreline |
| Reverse on backflush | Task 1 — `backflush` branch: `dx` negated |
| Pause when flow near zero | Task 1 — `animationPlayState: paused ? 'paused' : 'running'` |
| Cascade concentration attenuation | Task 1 JSX — `survival` reduces conc per stage |

### 2. Placeholder scan

None found. All code is complete and executable.

### 3. Type consistency

- `AnimParticle` interface defined in Task 1 Step 3, used in `buildParticles` return type, `particleKeyframes` param, and particle `.map()` — consistent.
- `AnimatedParticleStreamProps` defined and consumed in same step — consistent.
- JSX in Step 4 references `etaArr[i]` as `etaStage` prop — matches `AnimatedParticleStreamProps.etaStage: number`.
- `stageIdx` param in `buildParticles` used only as seed offset — no conflict with `i` in JSX map.
