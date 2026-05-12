// apps/hydros-console/src/components/WebGLCascadeView.tsx
//
// HydrOS WebGL 3D Cascade View — GPU-rendered alternative to ConicalCascadeView.
//
// Renders the three-stage conical cascade as 3D geometry using Three.js via
// React Three Fiber, with custom GLSL vertex/fragment shaders for particle
// dynamics and electrostatic-field visualization.
//
// Consumes the same HydrosDisplayState the SVG view consumes — particles, field
// strength, and capture status drive the GPU pipeline directly.
//
// Stages match ConicalCascadeView geometry exactly:
//   S1 (coarse, orange) → S2 (medium, yellow) → S3 (fine, blue)
//
// Coordinate convention here: x maps to world X (downstream flow), y maps to
// world Z (radial), all in normalised [-1, +1] device space converted from the
// SVG coordinate model used by the physics engine.

import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Environment } from '@react-three/drei';
import { useMemo, useRef } from 'react';
import * as THREE from 'three';
import type {
  HydrosDisplayState,
  ParticlePoint,
} from '../scenarios/displayStateMapper';

// ─────────────────────────────────────────────────────────────────────────
//  Geometric constants — derived from ConicalCascadeView SVG coordinates,
//  normalised to a [-1, +1] cube on each axis for a Three.js scene.
// ─────────────────────────────────────────────────────────────────────────

const SVG_BORE_TOP = 64;
const SVG_BORE_BOT = 244;
const SVG_CY = 154;

const SVG_STAGE_X = [
  { xStart: 118, xEnd: 298, apexX: 296, apexY: 243, color: '#FB923C', mult: 0.4 },
  { xStart: 306, xEnd: 486, apexX: 484, apexY: 243, color: '#FBBF24', mult: 0.7 },
  { xStart: 494, xEnd: 674, apexX: 672, apexY: 243, color: '#38BDF8', mult: 1.0 },
] as const;

// Total SVG x extent across all three stages
const SVG_X_MIN = 80;
const SVG_X_MAX = 730;
const SVG_X_RANGE = SVG_X_MAX - SVG_X_MIN;

// Normalise an SVG x coordinate to world X in [-1, +1].
function nx(xSvg: number): number {
  return ((xSvg - SVG_X_MIN) / SVG_X_RANGE) * 2 - 1;
}

// Normalise an SVG y coordinate (BORE_TOP..BORE_BOT) to world Z in [-1, +1].
function nz(ySvg: number): number {
  return ((ySvg - SVG_BORE_TOP) / (SVG_BORE_BOT - SVG_BORE_TOP)) * 2 - 1;
}

// ─────────────────────────────────────────────────────────────────────────
//  Reactor housing — outer cylinder enclosing all three stages
// ─────────────────────────────────────────────────────────────────────────

function ReactorHousing() {
  // Tube length spans full normalized X range; narrower radius for a more elongated reactor read.
  return (
    <group>
      {/* Main transparent tube */}
      <mesh rotation={[0, 0, Math.PI / 2]} position={[0, 0, 0]}>
        <cylinderGeometry args={[0.62, 0.62, 2.4, 64, 1, true]} />
        <meshPhysicalMaterial
          color="#0E1E33"
          metalness={0.3}
          roughness={0.25}
          transmission={0.7}
          thickness={0.4}
          ior={1.45}
          side={THREE.DoubleSide}
          transparent
          opacity={0.22}
        />
      </mesh>
      {/* End caps — inlet and outlet flanges */}
      <mesh rotation={[0, 0, Math.PI / 2]} position={[-1.2, 0, 0]}>
        <torusGeometry args={[0.62, 0.06, 16, 48]} />
        <meshStandardMaterial color="#475569" metalness={0.85} roughness={0.25} />
      </mesh>
      <mesh rotation={[0, 0, Math.PI / 2]} position={[1.2, 0, 0]}>
        <torusGeometry args={[0.62, 0.06, 16, 48]} />
        <meshStandardMaterial color="#475569" metalness={0.85} roughness={0.25} />
      </mesh>
    </group>
  );
}

// ─────────────────────────────────────────────────────────────────────────
//  Conical stage — one filtration cone (S1, S2, or S3)
// ─────────────────────────────────────────────────────────────────────────

interface StageProps {
  stageIdx: 0 | 1 | 2;
  eField: number;       // [0, 1] field strength
  clogLevel: number;    // [0, 1] clogging
}

function ConicalStage({ stageIdx, eField, clogLevel }: StageProps) {
  const stg = SVG_STAGE_X[stageIdx];
  const xMid = (nx(stg.xStart) + nx(stg.apexX)) / 2;
  const length = nx(stg.apexX) - nx(stg.xStart);
  const baseRadius = 0.55;
  // Apex radius reflects mesh fineness: S1 wide, S3 narrow
  const apexRadius = [0.32, 0.20, 0.10][stageIdx];

  // Color shifts with clogging
  const color = useMemo(() => {
    const base = new THREE.Color(stg.color);
    const clogged = new THREE.Color('#3a1f1f');
    return base.clone().lerp(clogged, clogLevel * 0.7);
  }, [stg.color, clogLevel]);

  const fieldGlowRef = useRef<THREE.MeshStandardMaterial>(null);

  useFrame(({ clock }) => {
    if (fieldGlowRef.current) {
      // Pulsing emissive intensity proportional to electrostatic field strength
      const pulse = 0.4 + 0.6 * Math.sin(clock.elapsedTime * 3 + stageIdx);
      fieldGlowRef.current.emissiveIntensity = eField * stg.mult * pulse;
    }
  });

  return (
    <group position={[xMid, 0, 0]}>
      {/* Outer mesh cone — the filter surface */}
      <mesh rotation={[0, 0, -Math.PI / 2]}>
        <coneGeometry args={[baseRadius, length, 48, 1, true]} />
        <meshStandardMaterial
          ref={fieldGlowRef}
          color={color}
          emissive={color}
          emissiveIntensity={eField * stg.mult}
          metalness={0.6}
          roughness={0.4}
          side={THREE.DoubleSide}
          wireframe={false}
          transparent
          opacity={0.55}
        />
      </mesh>
      {/* Inner mesh — denser wireframe shows the weave fineness */}
      <mesh rotation={[0, 0, -Math.PI / 2]} scale={[0.96, 0.96, 0.96]}>
        <coneGeometry args={[baseRadius, length, 32, 1, true]} />
        <meshBasicMaterial
          color={color}
          wireframe
          transparent
          opacity={0.55 - clogLevel * 0.25}
        />
      </mesh>
      {/* Apex collar — narrows to next stage */}
      <mesh position={[length / 2, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
        <cylinderGeometry args={[apexRadius, apexRadius, 0.04, 24]} />
        <meshStandardMaterial color="#475569" metalness={0.85} roughness={0.25} />
      </mesh>
    </group>
  );
}

// ─────────────────────────────────────────────────────────────────────────
//  Particle field — GPU-instanced particles driven by simulation state
// ─────────────────────────────────────────────────────────────────────────

// Custom vertex shader: instanced particles with per-instance size + color
// attributes; per-vertex sphere geometry scaled by particle diameter.
const particleVertexShader = /* glsl */ `
attribute vec3 iPosition;     // per-instance world position
attribute float iSize;        // per-instance scale (diameter in microns / 500)
attribute vec3 iColor;        // per-instance RGB color
attribute float iCaptured;    // 1.0 if captured (drives emissive glow)

varying vec3 vColor;
varying float vCaptured;
varying vec3 vNormal;
varying vec3 vViewPosition;

void main() {
  vColor = iColor;
  vCaptured = iCaptured;

  vec3 transformed = position * iSize + iPosition;
  vec4 mvPosition = modelViewMatrix * vec4(transformed, 1.0);
  vViewPosition = -mvPosition.xyz;
  vNormal = normalize(normalMatrix * normal);

  gl_Position = projectionMatrix * mvPosition;
}
`;

// Custom fragment shader: per-particle lighting + captured-particle emissive
// halo + species-color rim lighting.
const particleFragmentShader = /* glsl */ `
varying vec3 vColor;
varying float vCaptured;
varying vec3 vNormal;
varying vec3 vViewPosition;

void main() {
  vec3 N = normalize(vNormal);
  vec3 V = normalize(vViewPosition);

  // Lambertian-ish term against a fixed key light direction
  vec3 L = normalize(vec3(0.5, 1.0, 0.8));
  float diffuse = max(dot(N, L), 0.0);

  // Rim term — graphics-engineer-classic edge highlight
  float rim = 1.0 - max(dot(N, V), 0.0);
  rim = pow(rim, 2.5);

  vec3 base = vColor * (0.35 + 0.65 * diffuse);
  vec3 rimColor = vColor * rim * 1.8;

  // Captured particles glow — emissive boost
  vec3 emissive = vColor * vCaptured * 0.9;

  vec3 finalColor = base + rimColor + emissive;
  gl_FragColor = vec4(finalColor, 1.0);
}
`;

// Convert SVG-space ParticlePoint to world coords [-1, +1]
function particleToWorld(p: ParticlePoint): [number, number, number] {
  return [nx(p.x), 0, nz(p.y)];
}

// Species → color in RGB [0, 1]
function speciesColor(species: string): [number, number, number] {
  switch (species) {
    case 'PP':  return [0.95, 0.45, 0.20];  // polypropylene — orange-red
    case 'PE':  return [0.30, 0.85, 0.60];  // polyethylene — green
    case 'PET': return [0.30, 0.65, 1.00];  // PET — blue
    default:    return [0.85, 0.85, 0.85];
  }
}

interface ParticleFieldProps {
  particles: ParticlePoint[];
  capacity: number;     // Max instances; rendering clipped to this.
  syntheticMode: boolean;  // when true, generate demo particles in useFrame
}

function ParticleField({ particles, capacity, syntheticMode }: ParticleFieldProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);

  // Allocate per-instance attribute buffers once
  const { iPosition, iSize, iColor, iCaptured } = useMemo(() => {
    return {
      iPosition: new Float32Array(capacity * 3),
      iSize: new Float32Array(capacity),
      iColor: new Float32Array(capacity * 3),
      iCaptured: new Float32Array(capacity),
    };
  }, [capacity]);

  // Custom shader material — survives strict-mode remounts via useMemo
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: particleVertexShader,
        fragmentShader: particleFragmentShader,
        transparent: false,
      }),
    []
  );

  useFrame(({ clock }) => {
    if (!meshRef.current) return;
    const m = meshRef.current;

    // In synthetic mode, regenerate demo particles each frame so they flow
    const activeParticles = syntheticMode
      ? generateDemoParticles(450, clock.elapsedTime)
      : particles;

    const n = Math.min(activeParticles.length, capacity);

    for (let i = 0; i < n; i++) {
      const p = activeParticles[i];
      const [wx, wy, wz] = particleToWorld(p);
      iPosition[i * 3 + 0] = wx;
      // Add a small deterministic 3D dispersion in Y so particles aren't a flat sheet
      iPosition[i * 3 + 1] = wy + Math.sin(i * 7.31 + clock.elapsedTime * 0.8) * 0.12;
      iPosition[i * 3 + 2] = wz;

      // Size: clamp d_p_um to [5, 500] then map to world scale [0.012, 0.045]
      const d = Math.max(5, Math.min(500, p.d_p_um));
      iSize[i] = 0.012 + (d - 5) / 495 * 0.033;

      const [r, g, b] = speciesColor(p.species);
      iColor[i * 3 + 0] = r;
      iColor[i * 3 + 1] = g;
      iColor[i * 3 + 2] = b;

      iCaptured[i] = p.status === 'captured' ? 1.0 : 0.0;
    }

    // Hide unused instances by zeroing their size
    for (let i = n; i < capacity; i++) {
      iSize[i] = 0;
    }

    // Push attribute updates to GPU
    const geom = m.geometry;
    (geom.attributes.iPosition as THREE.InstancedBufferAttribute).needsUpdate = true;
    (geom.attributes.iSize as THREE.InstancedBufferAttribute).needsUpdate = true;
    (geom.attributes.iColor as THREE.InstancedBufferAttribute).needsUpdate = true;
    (geom.attributes.iCaptured as THREE.InstancedBufferAttribute).needsUpdate = true;
    m.count = n;
  });

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, capacity]}
      material={material}
      frustumCulled={false}
    >
      <sphereGeometry args={[1, 12, 12]}>
        <instancedBufferAttribute attach="attributes-iPosition" args={[iPosition, 3]} />
        <instancedBufferAttribute attach="attributes-iSize" args={[iSize, 1]} />
        <instancedBufferAttribute attach="attributes-iColor" args={[iColor, 3]} />
        <instancedBufferAttribute attach="attributes-iCaptured" args={[iCaptured, 1]} />
      </sphereGeometry>
    </instancedMesh>
  );
}

// ─────────────────────────────────────────────────────────────────────────
//  Top-level WebGL view
// ─────────────────────────────────────────────────────────────────────────

interface WebGLCascadeViewProps {
  state: HydrosDisplayState | null;
}

// ─────────────────────────────────────────────────────────────────────────
//  Synthetic particle fallback — drives a credible-looking demo when no
//  scenario data is available (backend not running, scenario not started).
//  Produces a stable seeded particle distribution flowing through the stages
//  with deterministic positions so the scene reads as "alive" in screenshots.
// ─────────────────────────────────────────────────────────────────────────

function generateDemoParticles(count: number, t: number): ParticlePoint[] {
  const particles: ParticlePoint[] = [];
  const species = ['PP', 'PE', 'PET'];
  const sizes = [500, 100, 5];
  for (let i = 0; i < count; i++) {
    // Walk particle through full X range with phase offset; wrap at end.
    const phase = (i / count + t * 0.05) % 1.0;
    const xSvg = SVG_X_MIN + phase * SVG_X_RANGE;
    // Determine which stage this x falls in
    let stageIdx = 0;
    if (xSvg > 306) stageIdx = 1;
    if (xSvg > 494) stageIdx = 2;
    const stg = SVG_STAGE_X[stageIdx];
    // Local progress within stage; radial position tapers toward apex
    const localPhase = (xSvg - stg.xStart) / (stg.xEnd - stg.xStart);
    const r = (Math.sin(i * 12.97 + phase * 6.28) * 0.6 + 0.4) * (0.4 + (1 - localPhase) * 0.6);
    const ySvg = SVG_CY + r * (SVG_BORE_BOT - SVG_CY) * (i % 2 === 0 ? 1 : -1);
    const sizeIdx = i % 3;
    // Captured if past apex and r > threshold (mimics filtration)
    const captured = localPhase > 0.85 && Math.abs(r) > 0.35;
    particles.push({
      x: xSvg,
      y: ySvg,
      species: species[sizeIdx],
      d_p_um: sizes[sizeIdx],
      status: captured ? 'captured' : 'in_transit',
    });
  }
  return particles;
}

export default function WebGLCascadeView({ state }: WebGLCascadeViewProps) {
  // Aggregate all particle streams into one buffer per render
  const realParticles = useMemo(() => {
    const streams = state?.particleStreams;
    if (!streams) return [];
    return [...streams.s1, ...streams.s2, ...streams.s3];
  }, [state?.particleStreams]);

  const hasRealData = realParticles.length > 0;

  // HydrosDisplayState exposes single eField + clog values; per-stage intensity
  // is derived by the stage's mesh multiplier (S1=0.4, S2=0.7, S3=1.0) — the same
  // convention ConicalCascadeView uses for its radial field-line rendering.
  const eField = state?.eField ?? 0;
  const clog = state?.clog ?? 0;
  const eFieldArr: number[] = [
    eField * SVG_STAGE_X[0].mult,
    eField * SVG_STAGE_X[1].mult,
    eField * SVG_STAGE_X[2].mult,
  ];
  const clogArr: number[] = [clog, clog, clog];

  return (
    <div style={{ width: '100%', height: '100%', background: '#080D18' }}>
      <Canvas
        camera={{ position: [1.6, 0.9, 2.0], fov: 36 }}
        gl={{ antialias: true, powerPreference: 'high-performance' }}
        dpr={[1, 2]}
      >
        {/* Lighting rig */}
        <ambientLight intensity={0.4} />
        <hemisphereLight color="#88AACC" groundColor="#1E293B" intensity={0.55} />
        <directionalLight position={[3, 4, 2]} intensity={1.3} color="#FFFFFF" castShadow />
        <directionalLight position={[-2, 1, -3]} intensity={0.5} color="#7DD3FC" />
        <pointLight position={[0, 0, 0]} intensity={0.6} color="#FBBF24" distance={2} />
        <Environment preset="warehouse" background={false} />

        {/* Reactor */}
        <ReactorHousing />
        <ConicalStage stageIdx={0} eField={eFieldArr[0] ?? 0} clogLevel={clogArr[0] ?? 0} />
        <ConicalStage stageIdx={1} eField={eFieldArr[1] ?? 0} clogLevel={clogArr[1] ?? 0} />
        <ConicalStage stageIdx={2} eField={eFieldArr[2] ?? 0} clogLevel={clogArr[2] ?? 0} />

        {/* Particles — synthetic demo when no real scenario data is available */}
        <ParticleField particles={realParticles} capacity={1500} syntheticMode={!hasRealData} />

        {/* Controls */}
        <OrbitControls
          enablePan={false}
          minDistance={1.5}
          maxDistance={6}
          maxPolarAngle={Math.PI / 1.6}
          autoRotate
          autoRotateSpeed={0.2}
        />
      </Canvas>
      {/* HUD overlay */}
      <div
        style={{
          position: 'absolute',
          top: 12,
          left: 16,
          padding: '6px 10px',
          background: 'rgba(8, 13, 24, 0.7)',
          border: '1px solid rgba(56, 189, 248, 0.4)',
          borderRadius: 4,
          color: '#7DD3FC',
          font: '11px/1.35 "JetBrains Mono", "Fira Code", monospace',
          pointerEvents: 'none',
        }}
      >
        WebGL 3D · Three.js / R3F · custom GLSL shaders<br />
        {hasRealData ? `${realParticles.length} particles · live` : '450 particles · synthetic demo'} · {Math.round(eFieldArr.reduce((a, b) => a + b, 0) * 100 / 3)}% mean field
      </div>
    </div>
  );
}
