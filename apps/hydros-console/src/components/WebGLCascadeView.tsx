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
  // Tube length spans full normalized X range; radius ~1 covers normalised Z extent.
  return (
    <mesh rotation={[0, 0, Math.PI / 2]} position={[0, 0, 0]}>
      <cylinderGeometry args={[1.05, 1.05, 2.4, 64, 1, true]} />
      <meshPhysicalMaterial
        color="#0E1E33"
        metalness={0.4}
        roughness={0.35}
        transmission={0.65}
        thickness={0.6}
        ior={1.4}
        side={THREE.DoubleSide}
        transparent
        opacity={0.18}
      />
    </mesh>
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
  const baseRadius = 0.9;
  // Apex radius reflects mesh fineness: S1 wide, S3 narrow
  const apexRadius = [0.42, 0.22, 0.08][stageIdx];

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
}

function ParticleField({ particles, capacity }: ParticleFieldProps) {
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

  useFrame(() => {
    if (!meshRef.current) return;
    const m = meshRef.current;

    const n = Math.min(particles.length, capacity);

    for (let i = 0; i < n; i++) {
      const p = particles[i];
      const [wx, wy, wz] = particleToWorld(p);
      iPosition[i * 3 + 0] = wx;
      iPosition[i * 3 + 1] = wy + (Math.random() - 0.5) * 0.3;
      iPosition[i * 3 + 2] = wz;

      // Size: clamp d_p_um to [5, 500] then map to world scale [0.008, 0.06]
      const d = Math.max(5, Math.min(500, p.d_p_um));
      iSize[i] = 0.008 + (d - 5) / 495 * 0.052;

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

export default function WebGLCascadeView({ state }: WebGLCascadeViewProps) {
  // Aggregate all particle streams into one buffer per render
  const allParticles = useMemo(() => {
    const streams = state?.particleStreams;
    if (!streams) return [];
    return [...streams.s1, ...streams.s2, ...streams.s3];
  }, [state?.particleStreams]);

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
        camera={{ position: [0, 1.2, 2.6], fov: 38 }}
        gl={{ antialias: true, powerPreference: 'high-performance' }}
        dpr={[1, 2]}
      >
        {/* Lighting rig */}
        <ambientLight intensity={0.35} />
        <hemisphereLight color="#88AACC" groundColor="#1E293B" intensity={0.5} />
        <directionalLight position={[3, 4, 2]} intensity={1.1} color="#FFFFFF" />
        <directionalLight position={[-2, 1, -3]} intensity={0.4} color="#7DD3FC" />
        <Environment preset="warehouse" background={false} />

        {/* Reactor */}
        <ReactorHousing />
        <ConicalStage stageIdx={0} eField={eFieldArr[0] ?? 0} clogLevel={clogArr[0] ?? 0} />
        <ConicalStage stageIdx={1} eField={eFieldArr[1] ?? 0} clogLevel={clogArr[1] ?? 0} />
        <ConicalStage stageIdx={2} eField={eFieldArr[2] ?? 0} clogLevel={clogArr[2] ?? 0} />

        {/* Particles */}
        <ParticleField particles={allParticles} capacity={1500} />

        {/* Controls */}
        <OrbitControls
          enablePan={false}
          minDistance={1.5}
          maxDistance={6}
          maxPolarAngle={Math.PI / 1.6}
          autoRotate
          autoRotateSpeed={0.4}
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
        {allParticles.length} particles · {Math.round(eFieldArr.reduce((a, b) => a + b, 0) * 100 / 3)}% mean field
      </div>
    </div>
  );
}
