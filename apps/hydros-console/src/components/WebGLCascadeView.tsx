// apps/hydros-console/src/components/WebGLCascadeView.tsx
//
// HydrOS WebGL 3D Cascade View — GPU-rendered alternative to ConicalCascadeView.
//
// Renders the three-stage conical cascade as 3D geometry using Three.js via
// React Three Fiber, with custom GLSL vertex/fragment shaders for particle
// dynamics, capture-flash effects, and electrostatic-field visualization.
//
// Stages match ConicalCascadeView geometry exactly:
//   S1 (coarse mesh, orange #FB923C) → S2 (medium mesh, yellow #FBBF24)
//                                   → S3 (fine mesh, blue #38BDF8)
//
// Each stage is a conical filter with apex bottom-biased (the 2D design's
// gravity-fed extraction model). Particles flow strictly left → right, with
// probabilistic capture at each stage's apex; captured particles accumulate
// visibly in extraction channels below each cone.
//
// Coordinate convention:
//   world +X = downstream flow direction
//   world +Z = bottom of the bore (matches SVG y increasing downward)
//   world +Y = depth into the scene (the rotational axis the user orbits around)

import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Environment, Text } from '@react-three/drei';
import { useMemo, useRef, useState } from 'react';
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
  { xStart: 118, xEnd: 298, apexX: 296, apexY: 243, color: '#FB923C', mult: 0.4, chY: 280, wireSegments: 16, label: 'S1' },
  { xStart: 306, xEnd: 486, apexX: 484, apexY: 243, color: '#FBBF24', mult: 0.7, chY: 302, wireSegments: 28, label: 'S2' },
  { xStart: 494, xEnd: 674, apexX: 672, apexY: 243, color: '#38BDF8', mult: 1.0, chY: 324, wireSegments: 40, label: 'S3' },
] as const;

const SVG_X_MIN = 80;
const SVG_X_MAX = 730;
const SVG_X_RANGE = SVG_X_MAX - SVG_X_MIN;

// Three side-by-side extraction pipes — same Y axis, staircased Z, different
// lengths.  S1 is the OUTERMOST (largest Z, furthest from bore wall) and the
// LONGEST (starts upstream at S1's xStart, extends furthest right to the
// storage convergence).  S3 is the INNERMOST (smallest Z, closest to bore)
// and the SHORTEST.  Channels now END at the same X as S3's downstream edge
// (nx(674) = 0.828) — matches the 2D canonical where channels span SVG
// x=118→674 exactly.
const CHANNEL_X_END = 0.828;
const CHANNEL_Z_BY_STAGE = [0.82, 0.76, 0.70];   // S1 outermost → S3 innermost

function nx(xSvg: number): number {
  return ((xSvg - SVG_X_MIN) / SVG_X_RANGE) * 2 - 1;
}

function nz(ySvg: number): number {
  // SVG y increases downward; world +Z = bottom of bore
  return ((ySvg - SVG_BORE_TOP) / (SVG_BORE_BOT - SVG_BORE_TOP)) * 2 - 1;
}

// Apex bottom-bias — matches 2D's `M xStart,64 ... apexX,243` where the cone
// arcs from the bore TOP at upstream all the way down to the bore BOTTOM at
// the apex.  We achieve this by applying a Z-shear to the revolved profile:
// base stays centered on the bore axis, apex shifts toward +Z (bore bottom).
// The bore radius is 0.62, so SHEAR_Z=0.50 puts the apex about 80% of the
// way to the bore bottom from the axis — visibly "curved down" without
// punching through the housing wall.
const APEX_TILT_RAD = 0;             // legacy — no longer used (replaced by shear)
const SHEAR_Z = 0.50;

// Build a Bezier-curve profile for the conical stage's revolved mesh.
// Inspired by the 2D canonical path
//   M xStart,64 C xStart+77,64 apexX-4,96 apexX,243
// where the wall hangs near the bore wall for the first ~40% of stage length
// and then arcs rapidly down to the apex.  Returns Vector2 points in
// (radius, axial) form for THREE.LatheGeometry, centered at local Y=0 so it
// composes cleanly with the existing group rotation/positioning.
function buildBezierConeProfile(
  baseR: number,
  apexR: number,
  length: number,
  steps: number = 24,
): THREE.Vector2[] {
  const halfLen = length / 2;
  const p0x = baseR,                  p0y = -halfLen;
  const p1x = baseR,                  p1y = -halfLen + length * 0.43;
  const p2x = (baseR + apexR) * 0.55, p2y = -halfLen + length * 0.70;
  const p3x = apexR,                  p3y = halfLen;
  const points: THREE.Vector2[] = [];
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const u = 1 - t;
    const x = u * u * u * p0x + 3 * u * u * t * p1x + 3 * u * t * t * p2x + t * t * t * p3x;
    const y = u * u * u * p0y + 3 * u * u * t * p1y + 3 * u * t * t * p2y + t * t * t * p3y;
    points.push(new THREE.Vector2(x, y));
  }
  return points;
}

// ─────────────────────────────────────────────────────────────────────────
//  Reactor housing — transparent tube with PROPERLY-ORIENTED end flanges
//  plus inlet/outlet nozzles to communicate flow direction visually.
// ─────────────────────────────────────────────────────────────────────────

function ReactorHousing() {
  return (
    <group>
      {/* Main transparent tube along X axis.  Uses plain transparent
          standardMaterial (no PBR transmission) so the cones / particles
          inside render correctly through the housing wall from any side
          angle — transmission was hiding everything behind the wall. */}
      <mesh rotation={[0, 0, Math.PI / 2]} position={[0, 0, 0]}>
        <cylinderGeometry args={[0.62, 0.62, 2.4, 64, 1, true]} />
        <meshStandardMaterial
          color="#1A3A5E"
          metalness={0.35}
          roughness={0.30}
          side={THREE.DoubleSide}
          transparent
          opacity={0.12}
          depthWrite={false}
        />
      </mesh>

      {/* End flange rings — rotated around Y axis so their plane is
          perpendicular to X (i.e., concentric with the housing).  Previous
          rotation around Z left them as vertical hoops crossing the tube. */}
      <mesh rotation={[0, Math.PI / 2, 0]} position={[-1.2, 0, 0]}>
        <torusGeometry args={[0.62, 0.06, 16, 48]} />
        <meshStandardMaterial color="#64748B" metalness={0.85} roughness={0.25} />
      </mesh>
      <mesh rotation={[0, Math.PI / 2, 0]} position={[1.2, 0, 0]}>
        <torusGeometry args={[0.62, 0.06, 16, 48]} />
        <meshStandardMaterial color="#64748B" metalness={0.85} roughness={0.25} />
      </mesh>

      {/* Inlet nozzle — short cylinder extruding leftward from inlet flange */}
      <mesh rotation={[0, 0, Math.PI / 2]} position={[-1.4, 0, 0]}>
        <cylinderGeometry args={[0.28, 0.28, 0.4, 32]} />
        <meshStandardMaterial color="#475569" metalness={0.75} roughness={0.3} />
      </mesh>

      {/* Outlet nozzle — short narrower cylinder extruding rightward */}
      <mesh rotation={[0, 0, Math.PI / 2]} position={[1.4, 0, 0]}>
        <cylinderGeometry args={[0.18, 0.18, 0.4, 32]} />
        <meshStandardMaterial color="#475569" metalness={0.75} roughness={0.3} />
      </mesh>

      {/* Flow direction arrow chevron at the inlet — subtle but visible */}
      <mesh position={[-1.6, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
        <coneGeometry args={[0.12, 0.18, 16]} />
        <meshStandardMaterial color="#7DD3FC" emissive="#1A3A5E" emissiveIntensity={0.6} />
      </mesh>
    </group>
  );
}

// ─────────────────────────────────────────────────────────────────────────
//  Conical stage — one filtration cone with forced color, wireframe-density
//  variation, asymmetric apex tilt, and a downstream gap disc.
// ─────────────────────────────────────────────────────────────────────────

interface StageProps {
  stageIdx: 0 | 1 | 2;
  eField: number;
  clogLevel: number;
}

function ConicalStage({ stageIdx, eField, clogLevel }: StageProps) {
  const stg = SVG_STAGE_X[stageIdx];
  const xMid = (nx(stg.xStart) + nx(stg.apexX)) / 2;
  const length = nx(stg.apexX) - nx(stg.xStart);
  const baseRadius = 0.52;
  // Pointy apex so each cone reads as a funnel with a clear "node" at the
  // tip — the node is where particles converge and drop through the ejection
  // pipe into the channel below.
  const apexRadius = [0.10, 0.075, 0.05][stageIdx];

  // Per-stage mesh thickness + wireframe opacity.  S1 = coarse + thick (largest
  // gap between outer surface and inner wireframe, most prominent wires),
  // S3 = fine + thin (tight gap, faint wires).
  const innerScale = [0.92, 0.96, 0.985][stageIdx];
  const wireOpacityBase = [0.85, 0.72, 0.58][stageIdx];

  // FORCE stage color — bypass clog tint so the three stages always read as
  // distinct orange/yellow/blue regardless of operational state.
  const stageColor = useMemo(() => new THREE.Color(stg.color), [stg.color]);
  const emissiveColor = useMemo(() => new THREE.Color(stg.color).multiplyScalar(0.4), [stg.color]);

  // Bezier-revolved profile for the curved cone wall — hangs near the bore
  // wall for the first ~40% then arcs to the apex, matching the 2D canonical
  // path style.
  const bezierProfile = useMemo(
    () => buildBezierConeProfile(baseRadius, apexRadius, length),
    [baseRadius, apexRadius, length],
  );

  // Sheared lathe geometry: revolve the profile, then shift each vertex in
  // local Z proportional to its axial position so the base stays centered on
  // the bore axis but the apex pulls toward +Z (bore bottom).  Apex node
  // ends up at world Z = SHEAR_Z ≈ 80% of the way to the bore wall.
  const shearedConeGeom = useMemo(() => {
    const geom = new THREE.LatheGeometry(bezierProfile, 48);
    const positions = geom.attributes.position.array as Float32Array;
    for (let v = 0; v < positions.length; v += 3) {
      const y = positions[v + 1];
      const yNorm = (y + length / 2) / length;   // 0 at base, 1 at apex
      positions[v + 2] += yNorm * SHEAR_Z;
    }
    geom.attributes.position.needsUpdate = true;
    geom.computeVertexNormals();
    return geom;
  }, [bezierProfile, length]);

  const shearedWireGeom = useMemo(() => {
    const geom = new THREE.LatheGeometry(bezierProfile, stg.wireSegments);
    const positions = geom.attributes.position.array as Float32Array;
    for (let v = 0; v < positions.length; v += 3) {
      const y = positions[v + 1];
      const yNorm = (y + length / 2) / length;
      positions[v + 2] += yNorm * SHEAR_Z;
    }
    geom.attributes.position.needsUpdate = true;
    return geom;
  }, [bezierProfile, length, stg.wireSegments]);

  // Clog-tinted shadow color for the inner cone — fouled stages darken
  const cloggedColor = useMemo(() => {
    const clogged = new THREE.Color('#3a1f1f');
    return stageColor.clone().lerp(clogged, clogLevel * 0.4);
  }, [stageColor, clogLevel]);

  const fieldGlowRef = useRef<THREE.MeshStandardMaterial>(null);

  useFrame(({ clock }) => {
    if (fieldGlowRef.current) {
      const pulse = 0.4 + 0.6 * Math.sin(clock.elapsedTime * 3 + stageIdx);
      fieldGlowRef.current.emissiveIntensity = 0.35 + eField * stg.mult * pulse * 0.6;
    }
  });

  return (
    // No group rotation — the cone's downward tilt is now achieved by
    // shearing the LatheGeometry directly (apex shifts to +Z = bore bottom),
    // which preserves the base at the bore axis instead of tilting the
    // whole stage.
    <group position={[xMid, 0, 0]}>
      {/* Outer curved filter wall — Bezier-revolved profile that's been
          sheared so the base sits centered on the bore axis but the apex
          arcs all the way down to ~80% of the bore radius toward +Z (bore
          bottom).  Matches the 2D canonical path
            M xStart,64 C xStart+77,64 apexX-4,96 apexX,243
          where the wall goes from bore TOP at upstream to bore BOTTOM at
          the apex. */}
      <mesh
        geometry={shearedConeGeom}
        rotation={[0, 0, -Math.PI / 2]}
      >
        <meshStandardMaterial
          ref={fieldGlowRef}
          color={stageColor}
          emissive={emissiveColor}
          emissiveIntensity={0.55 + eField * stg.mult * 0.4}
          metalness={0.55}
          roughness={0.45}
          side={THREE.DoubleSide}
          transparent
          opacity={0.75}
        />
      </mesh>

      {/* Inner wireframe overlay — same shear, scaled radially inward to
          create a visible mesh THICKNESS.  Stage-specific gap (1 − innerScale)
          and wireframe opacity encode filtration fineness: S1 widest gap +
          most prominent wires (coarse), S3 tightest + faintest (fine). */}
      <mesh
        geometry={shearedWireGeom}
        rotation={[0, 0, -Math.PI / 2]}
        scale={[innerScale, 1, innerScale]}
      >
        <meshBasicMaterial
          color={stageColor}
          wireframe
          transparent
          opacity={Math.max(0.1, wireOpacityBase - clogLevel * 0.20)}
        />
      </mesh>

      {/* Apex collar — narrow cylindrical neck at the cone tip */}
      <mesh position={[length / 2, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
        <cylinderGeometry args={[apexRadius, apexRadius, 0.04, 24]} />
        <meshStandardMaterial color={cloggedColor} metalness={0.7} roughness={0.4} />
      </mesh>

      {/* Inter-stage transition collar — thin metallic ring matching the
          housing flange material so it reads as a flow-restriction baffle,
          not a black artifact in the middle of the bore. */}
      {stageIdx < 2 && (
        <mesh position={[length / 2 + 0.04, 0, 0]} rotation={[0, Math.PI / 2, 0]}>
          <torusGeometry args={[apexRadius + 0.05, 0.018, 12, 32]} />
          <meshStandardMaterial color="#94A3B8" metalness={0.85} roughness={0.25} />
        </mesh>
      )}
    </group>
  );
}

// ─────────────────────────────────────────────────────────────────────────
//  Stage label — 3D floating text identifying each stage above its cone.
// ─────────────────────────────────────────────────────────────────────────

interface StageLabelProps {
  stageIdx: 0 | 1 | 2;
}

function StageLabel({ stageIdx }: StageLabelProps) {
  const stg = SVG_STAGE_X[stageIdx];
  const xMid = (nx(stg.xStart) + nx(stg.apexX)) / 2;
  return (
    <Text
      position={[xMid, -0.85, -0.05]}
      fontSize={0.13}
      color={stg.color}
      anchorX="center"
      anchorY="middle"
      outlineWidth={0.005}
      outlineColor="#0E1E33"
    >
      {stg.label}
    </Text>
  );
}

// ─────────────────────────────────────────────────────────────────────────
//  Extraction channel — trough below each cone collecting captured particles
//  staircase configuration: S1 highest, S3 lowest (matches 2D chY values).
// ─────────────────────────────────────────────────────────────────────────

interface ExtractionChannelProps {
  stageIdx: 0 | 1 | 2;
}

// World-coordinate position helper for an extraction channel.  Three pipes,
// side-by-side, all at Y=0.  Each starts at its stage's xStart and runs to
// CHANNEL_X_END — so S1 (most upstream stage) is LONGEST, S3 (downstream
// stage) is SHORTEST.  Z is staircased OUTWARDS: S1 furthest from bore,
// S3 closest, so the longest pipe also reads as the outermost in section.
function channelWorldPos(stageIdx: 0 | 1 | 2): {
  x: number;
  y: number;
  z: number;
  length: number;
} {
  const stg = SVG_STAGE_X[stageIdx];
  const xStart = nx(stg.xStart);
  const length = CHANNEL_X_END - xStart;
  const xMid = (xStart + CHANNEL_X_END) / 2;
  const z = CHANNEL_Z_BY_STAGE[stageIdx];
  return { x: xMid, y: 0, z, length };
}

function ExtractionChannel({ stageIdx }: ExtractionChannelProps) {
  const stg = SVG_STAGE_X[stageIdx];
  const pos = channelWorldPos(stageIdx);
  const colorObj = useMemo(() => new THREE.Color(stg.color), [stg.color]);

  return (
    <group position={[pos.x, pos.y, pos.z]}>
      {/* Outer trough — long rectangular collection pipe from this stage's
          xStart through to CHANNEL_X_END.  S1 is the longest box, S3 the
          shortest, all three sitting parallel at their staircased Z. */}
      <mesh>
        <boxGeometry args={[pos.length * 0.96, 0.05, 0.07]} />
        <meshStandardMaterial color="#334155" metalness={0.7} roughness={0.45} />
      </mesh>
      {/* Inner highlight strip — color-codes this pipe to its source stage */}
      <mesh position={[0, 0.028, 0]}>
        <boxGeometry args={[pos.length * 0.93, 0.005, 0.05]} />
        <meshStandardMaterial color={colorObj} emissive={colorObj} emissiveIntensity={0.55} />
      </mesh>
    </group>
  );
}

// ─────────────────────────────────────────────────────────────────────────
//  EjectionPipe — color-coded vertical pipe from each cone's apex through
//  the bore wall down to its corresponding extraction channel.  Matches the
//  2D canonical ejection line `<line x1=apexX y1=244 x2=apexX y2=chY />`.
//  Communicates that captured particles funnel through the cone's "node"
//  (the apex) and drop into the channel below.
// ─────────────────────────────────────────────────────────────────────────

function EjectionPipe({ stageIdx }: { stageIdx: 0 | 1 | 2 }) {
  const stg = SVG_STAGE_X[stageIdx];
  const colorObj = useMemo(() => new THREE.Color(stg.color), [stg.color]);

  // Apex world position after the cone's shear: at the literal apexX along
  // bore axis, sheared down to z=SHEAR_Z (near bore bottom).
  const apexX = nx(stg.apexX);
  const apexZ = SHEAR_Z;

  const pos = channelWorldPos(stageIdx);
  const zTop = apexZ;
  const zBot = pos.z;
  const pipeLength = zBot - zTop;
  if (pipeLength <= 0.01) return null;
  const pipeMidZ = (zTop + zBot) / 2;

  return (
    <mesh rotation={[Math.PI / 2, 0, 0]} position={[apexX, 0, pipeMidZ]}>
      <cylinderGeometry args={[0.05, 0.05, pipeLength, 16]} />
      <meshStandardMaterial
        color={colorObj}
        emissive={colorObj}
        emissiveIntensity={0.35}
        metalness={0.55}
        roughness={0.40}
        transparent
        opacity={0.78}
      />
    </mesh>
  );
}

// ─────────────────────────────────────────────────────────────────────────
//  POL ZONE — inlet polarization marker upstream of S1.  Communicates where
//  particles get pre-charged before they encounter the first capture stage.
// ─────────────────────────────────────────────────────────────────────────

function PolZone() {
  return (
    <group>
      <mesh rotation={[0, Math.PI / 2, 0]} position={[-1.05, 0, 0]}>
        <torusGeometry args={[0.62, 0.025, 16, 48]} />
        <meshStandardMaterial
          color="#C4B5FD"
          emissive="#7C3AED"
          emissiveIntensity={1.0}
          metalness={0.3}
          roughness={0.35}
        />
      </mesh>
      <mesh rotation={[0, Math.PI / 2, 0]} position={[-0.95, 0, 0]}>
        <torusGeometry args={[0.62, 0.020, 16, 48]} />
        <meshStandardMaterial
          color="#C4B5FD"
          emissive="#7C3AED"
          emissiveIntensity={0.65}
          metalness={0.3}
          roughness={0.35}
        />
      </mesh>
      <Text
        position={[-1.0, -0.85, -0.05]}
        fontSize={0.095}
        color="#A78BFA"
        anchorX="center"
        anchorY="middle"
        outlineWidth={0.005}
        outlineColor="#0E1E33"
      >
        POL ZONE
      </Text>
    </group>
  );
}

// ─────────────────────────────────────────────────────────────────────────
//  Flush port — perpendicular inlet attached to the upstream end of each
//  extraction channel.  Glows + shows an inflow chevron when the matching
//  flushActiveS{n} flag is true (backflush water entering reverses capture).
// ─────────────────────────────────────────────────────────────────────────

interface FlushPortProps {
  stageIdx: 0 | 1 | 2;
  active: boolean;
}

function FlushPort({ stageIdx, active }: FlushPortProps) {
  const stg = SVG_STAGE_X[stageIdx];
  const pos = channelWorldPos(stageIdx);
  // Each port sits just upstream of its stage and just below its own pipe —
  // so S1 (deepest pipe, largest Z) has the lowest port, S3 (shallowest)
  // has the highest port.
  const portX = nx(stg.xStart) - 0.03;
  const portZ = pos.z + 0.12;
  return (
    <group position={[portX, 0, portZ]}>
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.035, 0.035, 0.16, 16]} />
        <meshStandardMaterial
          color={active ? '#7DD3FC' : '#475569'}
          emissive={active ? '#38BDF8' : '#000000'}
          emissiveIntensity={active ? 1.4 : 0.0}
          metalness={0.7}
          roughness={0.3}
        />
      </mesh>
      <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, 0, -0.07]}>
        <cylinderGeometry args={[0.05, 0.05, 0.02, 16]} />
        <meshStandardMaterial color="#64748B" metalness={0.8} roughness={0.3} />
      </mesh>
      {active && (
        <mesh position={[0, 0, 0.16]} rotation={[-Math.PI / 2, 0, 0]}>
          <coneGeometry args={[0.05, 0.09, 12]} />
          <meshStandardMaterial color="#A5F3FC" emissive="#38BDF8" emissiveIntensity={1.6} />
        </mesh>
      )}
    </group>
  );
}

// ─────────────────────────────────────────────────────────────────────────
//  Storage Chamber — detachable downstream collection canister below the
//  device.  Captured particles flushed from channels accumulate here as the
//  visible "extracted microplastics" volume.  Fill level reads state.storageFill;
//  stored swarm reads state.particleStreams.storage.
// ─────────────────────────────────────────────────────────────────────────

interface StoredParticleLike {
  species: string;
  d_p_um: number;
}

interface StorageChamberProps {
  fill: number;
  storedParticles: ReadonlyArray<StoredParticleLike>;
}

// Storage chamber on the RIGHT side of the three extraction pipes.  Its left
// wall sits exactly at CHANNEL_X_END so each pipe terminates AT the chamber's
// open mouth and feeds in directly — no drop chutes needed.  The Z axis is
// centered through the middle of the pipe staircase so all three pipes
// poke through the open left face into the chamber volume.
const STORAGE_LENGTH = 0.50;
const STORAGE_X_LEFT = CHANNEL_X_END;                           // 0.828
const STORAGE_X_RIGHT = STORAGE_X_LEFT + STORAGE_LENGTH;        // 1.328
const STORAGE_X_CENTER = (STORAGE_X_LEFT + STORAGE_X_RIGHT) / 2; // 1.078
const STORAGE_Z_CENTER = 0.80;                                  // middle of
                                                                // pipe Z range
                                                                // (S3 z=0.70 →
                                                                //  S1 z=0.82)
const STORAGE_RADIUS = 0.15;
const STORAGE_TOP_Z = STORAGE_Z_CENTER - STORAGE_RADIUS;        // 0.65

function StorageChamber({ fill, storedParticles }: StorageChamberProps) {
  const clampedFill = Math.max(0, Math.min(1, fill));
  const fillLength = STORAGE_LENGTH * clampedFill;
  const fillCenterX = STORAGE_X_LEFT + fillLength / 2;

  const particleData = useMemo(() => {
    const N = Math.min(storedParticles.length, 80);
    const arr: Array<{ pos: [number, number, number]; rgb: [number, number, number]; size: number }> = [];
    for (let i = 0; i < N; i++) {
      const p = storedParticles[i];
      const a = ((i * 9301 + 49297) % 233280) / 233280;
      const b = ((i * 4271 + 12911) % 233280) / 233280;
      const c = ((i * 6781 + 21379) % 233280) / 233280;
      const angle = a * Math.PI * 2;
      const radius = Math.sqrt(b) * STORAGE_RADIUS * 0.82;
      const yOff = Math.cos(angle) * radius;
      const zOff = Math.abs(Math.sin(angle)) * radius;
      const x = STORAGE_X_LEFT + c * Math.max(fillLength, STORAGE_LENGTH * 0.15);
      arr.push({
        pos: [x, yOff, STORAGE_Z_CENTER + zOff * 0.7],
        rgb: speciesColor(p.species),
        size: diameterToWorldSize(p.d_p_um) * 0.6,
      });
    }
    return arr;
  }, [storedParticles, fillLength]);

  return (
    <group>
      {/* Transparent outer shell — horizontal cylinder right of the channels.
          No left cap: the three pipes enter through the open left mouth and
          terminate inside the chamber. */}
      <mesh rotation={[0, 0, Math.PI / 2]} position={[STORAGE_X_CENTER, 0, STORAGE_Z_CENTER]}>
        <cylinderGeometry args={[STORAGE_RADIUS, STORAGE_RADIUS, STORAGE_LENGTH, 32, 1, true]} />
        <meshStandardMaterial
          color="#1A3A5E"
          metalness={0.25}
          roughness={0.3}
          side={THREE.DoubleSide}
          transparent
          opacity={0.18}
          depthWrite={false}
        />
      </mesh>

      {/* Right cap (removable end — accented slightly to read as detachable) */}
      <mesh rotation={[0, 0, Math.PI / 2]} position={[STORAGE_X_RIGHT, 0, STORAGE_Z_CENTER]}>
        <cylinderGeometry args={[STORAGE_RADIUS * 1.05, STORAGE_RADIUS * 1.05, 0.05, 32]} />
        <meshStandardMaterial color="#64748B" metalness={0.85} roughness={0.25} />
      </mesh>

      {/* Fill volume — translucent water column inside the chamber */}
      {fillLength > 0.001 && (
        <mesh rotation={[0, 0, Math.PI / 2]} position={[fillCenterX, 0, STORAGE_Z_CENTER]}>
          <cylinderGeometry args={[STORAGE_RADIUS * 0.92, STORAGE_RADIUS * 0.92, fillLength, 32]} />
          <meshPhysicalMaterial
            color="#1A3A5E"
            transmission={0.55}
            thickness={0.4}
            ior={1.33}
            roughness={0.05}
            metalness={0.05}
            transparent
            opacity={0.55}
            emissive="#1A3A5E"
            emissiveIntensity={0.18}
          />
        </mesh>
      )}

      {/* Stored particles */}
      {particleData.map((pd, i) => (
        <mesh key={i} position={pd.pos}>
          <sphereGeometry args={[pd.size, 8, 8]} />
          <meshStandardMaterial
            color={new THREE.Color(pd.rgb[0], pd.rgb[1], pd.rgb[2])}
            emissive={new THREE.Color(pd.rgb[0], pd.rgb[1], pd.rgb[2])}
            emissiveIntensity={0.4}
          />
        </mesh>
      ))}

      {/* Storage label — at label-row Y, aligned with chamber X/Z */}
      <Text
        position={[STORAGE_X_CENTER, -0.85, STORAGE_Z_CENTER]}
        fontSize={0.085}
        color="#7DD3FC"
        anchorX="center"
        anchorY="middle"
        outlineWidth={0.005}
        outlineColor="#0E1E33"
      >
        {`STORAGE · ${Math.round(clampedFill * 100)}%`}
      </Text>
    </group>
  );
}

// ─────────────────────────────────────────────────────────────────────────
//  Outlet label — "CLEAN WATER BORE" marker above the downstream outlet.
// ─────────────────────────────────────────────────────────────────────────

function OutletLabel() {
  return (
    <Text
      position={[1.4, 0.85, 0]}
      fontSize={0.08}
      color="#A5F3FC"
      anchorX="center"
      anchorY="middle"
      outlineWidth={0.005}
      outlineColor="#0E1E33"
    >
      CLEAN WATER BORE →
    </Text>
  );
}

// ─────────────────────────────────────────────────────────────────────────
//  GLSL shaders — particle vertex + fragment with capture-flash + status
// ─────────────────────────────────────────────────────────────────────────

const particleVertexShader = /* glsl */ `
attribute vec3 iPosition;
attribute float iSize;
attribute vec3 iColor;
attribute float iCaptured;
attribute float iFlashAge;     // time since capture in seconds; <0 = not captured

varying vec3 vColor;
varying float vCaptured;
varying vec3 vNormal;
varying vec3 vViewPosition;
varying float vFlash;

void main() {
  vColor = iColor;
  vCaptured = iCaptured;

  // Capture-flash intensity: 1.0 at moment of capture, decays over ~0.5s
  vFlash = (iFlashAge >= 0.0 && iFlashAge < 0.5) ? (1.0 - iFlashAge * 2.0) : 0.0;

  vec3 transformed = position * iSize + iPosition;
  vec4 mvPosition = modelViewMatrix * vec4(transformed, 1.0);
  vViewPosition = -mvPosition.xyz;
  vNormal = normalize(normalMatrix * normal);

  gl_Position = projectionMatrix * mvPosition;
}
`;

const particleFragmentShader = /* glsl */ `
varying vec3 vColor;
varying float vCaptured;
varying vec3 vNormal;
varying vec3 vViewPosition;
varying float vFlash;

void main() {
  vec3 N = normalize(vNormal);
  vec3 V = normalize(vViewPosition);
  vec3 L = normalize(vec3(0.5, 1.0, 0.8));

  float diffuse = max(dot(N, L), 0.0);
  float rim = pow(1.0 - max(dot(N, V), 0.0), 2.5);

  // In-transit particles: standard lit appearance with rim
  // Captured particles: emissive boost + slight desaturation
  vec3 base = vColor * (0.35 + 0.65 * diffuse);
  vec3 rimColor = vColor * rim * 1.8;

  // Captured = persistent emissive halo
  vec3 capturedEmissive = vColor * vCaptured * 1.4;

  // Flash = brief warm white burst at the moment of capture
  vec3 flashColor = vec3(1.0, 0.95, 0.75) * vFlash * 2.2;

  vec3 finalColor = base + rimColor + capturedEmissive + flashColor;
  gl_FragColor = vec4(finalColor, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────
//  Particle utilities — coordinate conversion, species → color, size mapping
// ─────────────────────────────────────────────────────────────────────────

function particleToWorld(p: ParticlePoint): [number, number, number] {
  return [nx(p.x), 0, nz(p.y)];
}

// Cubic Bezier evaluator matching buildBezierConeProfile's control points —
// gives the cone wall radius at any localPhase t in [0, 1].
function bezierConeRadius(t: number, baseR: number, apexR: number): number {
  const u = 1 - t;
  return (
    u * u * u * baseR
    + 3 * u * u * t * baseR
    + 3 * u * t * t * (baseR + apexR) * 0.55
    + t * t * t * apexR
  );
}

// Position a captured particle ON THE MESH WALL near the apex.  Backend's
// _is_rt_captured rule pins oversized particles at r_norm>=1.0 and slides
// them to the apex zone where capture finalizes — visually this means the
// captured particle should sit ON the cone surface, clustered near (but not
// at) the apex tip, accumulating over time as fouling grows.  This is what
// backflush is meant to dislodge; without it, backflush has no visible
// reason to exist.
function onMeshWallPosition(
  stageIdx: 0 | 1 | 2,
  seed: number,
  particleSize: number,
): [number, number, number] {
  const stg = SVG_STAGE_X[stageIdx];
  const baseRCone = 0.52;
  const apexRCone = [0.10, 0.075, 0.05][stageIdx];

  // Cluster near apex (localPhase 0.82–0.98) with a slight upstream spread
  const localPhase = 0.82 + ((Math.sin(seed * 0.97) + 1) * 0.5) * 0.16;
  const coneR = bezierConeRadius(localPhase, baseRCone, apexRCone);

  // Sit ON the wall — radially at coneR * 0.85, just inside the visible
  // mesh so the particle reads as "stuck against the surface", not floating
  // in the cone interior.
  const r = Math.max(0.02, coneR * 0.85 - particleSize * 0.3);

  // Stable angular position around the cone axis (per particle)
  const angle = seed * 2.31;

  const xStartW = nx(stg.xStart);
  const xApexW = nx(stg.apexX);
  const wx = xStartW + localPhase * (xApexW - xStartW);

  // Cone axis is sheared toward +Z by SHEAR_Z at the apex
  const zConeAxis = localPhase * SHEAR_Z;

  return [
    wx,
    r * Math.cos(angle),
    r * Math.sin(angle) + zConeAxis,
  ];
}

// Rich real-data positioning — applies the same cone-envelope clamping,
// sheared-axis tracking, and pipe/channel placement that the synthetic
// generator does, but driven by the backend's per-particle state instead
// of generated values.  Without this, the RUN path collapses to a flat
// nx/nz mapping and loses every realism improvement we've built.
function computeRealParticleWorldPos(
  p: ParticlePoint,
  i: number,
): [number, number, number] {
  const xSvg = p.x;
  const wx = nx(xSvg);

  // Which stage cone is this particle inside?
  let stageIdx: 0 | 1 | 2 = 0;
  if (xSvg > 306) stageIdx = 1;
  if (xSvg > 494) stageIdx = 2;
  const stg = SVG_STAGE_X[stageIdx];
  const localPhase = Math.max(0, Math.min(1, (xSvg - stg.xStart) / (stg.xEnd - stg.xStart)));

  // Backend's r_norm derived from SVG y (CY=154 axis, ~244 bore wall).
  const rNorm = Math.max(0, Math.min(1, (p.y - 154) / (244 - 154)));

  // Captured particles → on mesh wall, in pipe transit, or in channel.
  if (p.status === 'captured') {
    const ch = channelWorldPos(stageIdx);
    const apexX = nx(stg.apexX);
    const seed = i * 7.91 + xSvg * 0.013;
    const transitSeed = (Math.sin(seed) + 1) * 0.5;
    const particleSize = diameterToWorldSize(p.d_p_um);

    if (transitSeed < 0.40) {
      // ON THE MESH WALL near apex — accumulated capture load that backflush
      // would dislodge.  This is the fouling layer visible on each stage.
      return onMeshWallPosition(stageIdx, seed, particleSize);
    }
    if (transitSeed < 0.55) {
      // In the ejection pipe between cone apex (z=SHEAR_Z) and channel z
      const pipeT = (Math.cos(seed * 0.5) + 1) * 0.5;
      return [
        apexX + (Math.cos(seed * 0.7) * 0.5) * 0.018,
        (Math.sin(seed * 0.8) * 0.5) * 0.018,
        SHEAR_Z + pipeT * (ch.z - SHEAR_Z),
      ];
    }
    // Settled in the long collection channel
    const inChannel = (Math.sin(seed * 0.34) + 1) * 0.5;
    return [
      ch.x - ch.length * 0.46 + inChannel * ch.length * 0.92,
      ch.y + (Math.sin(i * 0.66) * 0.5) * 0.030,
      ch.z + (Math.cos(seed * 0.51) * 0.5) * 0.040,
    ];
  }

  // In-transit or passed: flow through the cone interior, envelope-clamped.
  const baseRCone = 0.52;
  const apexRCone = [0.10, 0.075, 0.05][stageIdx];
  const u = 1 - localPhase;
  const coneEnvR =
    u * u * u * baseRCone
    + 3 * u * u * localPhase * baseRCone
    + 3 * u * localPhase * localPhase * (baseRCone + apexRCone) * 0.55
    + localPhase * localPhase * localPhase * apexRCone;
  const particleSize = diameterToWorldSize(p.d_p_um);
  const maxParticleR = Math.max(0.04, coneEnvR * 0.92 - particleSize);

  // Backend's r_norm scaled to the cone envelope (0=axis, 1=wall) and
  // clamped so the particle's center stays inside the mesh wall.
  const targetR = rNorm * coneEnvR * 0.9;
  const r = Math.min(targetR, maxParticleR);

  // Stable per-particle angle so the 2D-axisymmetric backend physics fans
  // out into a full 3D distribution around the cone axis.
  const angle = i * 0.97;

  // Cone axis tilt (matches the shear we applied to the cone geometry):
  // captured particles are already handled above, so this is for the
  // visible in-transit flow — they ride with the cone interior.
  const zConeAxis = localPhase * SHEAR_Z * (p.status === 'in_transit' ? 1.0 : 0.55);

  return [
    wx,
    r * Math.cos(angle),
    r * Math.sin(angle) + zConeAxis,
  ];
}

// PP / PE / PET species → RGB color [0, 1]
function speciesColor(species: string): [number, number, number] {
  switch (species) {
    case 'PP':  return [0.98, 0.50, 0.20];   // polypropylene — warm orange-red
    case 'PE':  return [0.30, 0.85, 0.55];   // polyethylene — green
    case 'PET': return [0.40, 0.70, 1.00];   // PET — blue
    default:    return [0.85, 0.85, 0.85];
  }
}

// Size mapping: sqrt of (d_p_um / 500) — gives ~10x visual range for 100x physical range.
// This is the standard "perceptually correct yet visible" scale used in
// real-time particle visualisation.  Smaller particles still visible, larger
// particles clearly dominant.
function diameterToWorldSize(d_p_um: number): number {
  const ratio = Math.max(5, Math.min(500, d_p_um)) / 500;
  return 0.006 + Math.sqrt(ratio) * 0.052;
}

// ─────────────────────────────────────────────────────────────────────────
//  Synthetic-particle generator — replaces the previous uniform-distribution
//  approach with a true entry → traverse → capture lifecycle.  Each particle
//  is deterministically assigned a capture-fate (S1, S2, S3, or pass-through)
//  from its index, so the system behaves as a continuous flow with stable
//  per-particle identity across frames.
// ─────────────────────────────────────────────────────────────────────────

interface SyntheticOutput {
  particles: ParticlePoint[];
  flashAges: Float32Array;
  worldPositions: Float32Array;      // pre-computed world (x, y, z) per particle — bypasses SVG mapping
  fateDistribution: [number, number, number, number];  // total particles assigned to S1, S2, S3, pass-through
}

// Apex phase positions (when each stage's apex is reached in the particle's
// normalised travel through the reactor):
const STAGE_APEX_PHASES = [0.31, 0.62, 0.93];

function generateRealisticParticles(count: number, t: number): SyntheticOutput {
  const particles: ParticlePoint[] = [];
  const flashAges = new Float32Array(count);
  const worldPositions = new Float32Array(count * 3);
  const fateDistribution: [number, number, number, number] = [0, 0, 0, 0];  // S1, S2, S3, passthrough
  const species = ['PET', 'PE', 'PP'];
  // Particle diameters chosen JUST ABOVE each stage's mesh opening
  // (S1=500µm, S2=100µm, S3=5µm — per hydrion/physics/m5/capture_rt.py).
  // Per the backend's _is_rt_captured rule, only particles LARGER than the
  // mesh opening get retained, so each species gets caught at its intended
  // stage and only nano-fines (< S3 opening) would pass through.
  //
  //   PP  800 µm > S1 (500) → caught at S1
  //   PE  200 µm > S2 (100) → passes S1 (200<500), caught at S2
  //   PET  25 µm > S3 (5)   → passes S1 + S2, caught at S3
  const sizes = [25, 200, 800];

  for (let i = 0; i < count; i++) {
    const seed = i * 7331.7;
    const speed = 0.06 + ((Math.sin(seed) + 1) * 0.5) * 0.04;
    const startOffset = i / count;

    // Size class FIRST — capture probability depends on it.
    const sizeClass = i % 3;                  // 0=PET 5µm, 1=PE 100µm, 2=PP 500µm
    const d_p_um = sizes[sizeClass];
    const spec = species[sizeClass];

    // Size-dependent capture fate, matching the backend cascade rule
    // (each mesh retains anything LARGER than its opening; smaller particles
    // pass through).  Each species is sized just above its target stage's
    // opening so it gets caught primarily at that stage, with a small
    // fraction of upstream + downstream leakage to mimic real-world
    // imperfection.
    const captureRoll = (Math.sin(seed * 1.3 + 0.7) + 1) * 0.5;
    let captureStage = -1;
    if (sizeClass === 2) {
      // PP 800µm — S1 catches it (only mesh with opening < 800)
      if (captureRoll < 0.93)      { captureStage = 0; fateDistribution[0]++; }
      else if (captureRoll < 0.97) { captureStage = 1; fateDistribution[1]++; }
      else if (captureRoll < 0.99) { captureStage = 2; fateDistribution[2]++; }
      else                          { fateDistribution[3]++; }
    } else if (sizeClass === 1) {
      // PE 200µm — passes S1 (200<500), caught at S2 (200>100)
      if (captureRoll < 0.02)      { captureStage = 0; fateDistribution[0]++; }
      else if (captureRoll < 0.94) { captureStage = 1; fateDistribution[1]++; }
      else if (captureRoll < 0.98) { captureStage = 2; fateDistribution[2]++; }
      else                          { fateDistribution[3]++; }
    } else {
      // PET 25µm — passes S1+S2, caught at S3 (25>5)
      if (captureRoll < 0.005)     { captureStage = 0; fateDistribution[0]++; }
      else if (captureRoll < 0.03) { captureStage = 1; fateDistribution[1]++; }
      else if (captureRoll < 0.96) { captureStage = 2; fateDistribution[2]++; }
      else                          { fateDistribution[3]++; }
    }

    const phase = (startOffset + t * speed) % 1.0;

    let wx: number, wy: number, wz: number;
    let status: ParticlePoint['status'];

    if (captureStage !== -1 && phase > STAGE_APEX_PHASES[captureStage]) {
      // CAPTURED — 40% on mesh wall (accumulated fouling load), 15% in
      // ejection pipe (transiting from apex to channel), 45% settled in the
      // long collection tube.  The on-mesh layer is what backflush would
      // dislodge — it's the visible reason the flush state exists.
      const ch = channelWorldPos(captureStage as 0 | 1 | 2);
      const stgCaptured = SVG_STAGE_X[captureStage as 0 | 1 | 2];
      const apexX = nx(stgCaptured.apexX);
      const transitSeed = (Math.sin(seed * 7.91) + 1) * 0.5;
      const particleSize = diameterToWorldSize(d_p_um);

      if (transitSeed < 0.40) {
        // ON MESH WALL near apex — accumulating fouling load
        const [mx, my, mz] = onMeshWallPosition(
          captureStage as 0 | 1 | 2,
          seed * 7.91,
          particleSize,
        );
        wx = mx; wy = my; wz = mz;
      } else if (transitSeed < 0.55) {
        // IN EJECTION PIPE — between cone apex (z=SHEAR_Z) and channel z
        const pipeT = (Math.cos(seed * 4.31) + 1) * 0.5;
        wx = apexX + (Math.cos(seed * 5.7) * 0.5) * 0.018;
        wy = (Math.sin(seed * 6.1) * 0.5) * 0.018;
        wz = SHEAR_Z + pipeT * (ch.z - SHEAR_Z);
      } else {
        // IN CHANNEL — distributed along the long collection tube
        const inChannel = (Math.sin(seed * 2.7) + 1) * 0.5;
        wx = ch.x - ch.length * 0.46 + inChannel * ch.length * 0.92;
        wy = ch.y + (Math.sin(i * 5.3) * 0.5) * 0.030;
        wz = ch.z + (Math.cos(seed * 4.1) * 0.5) * 0.040;
      }
      status = 'captured';

      const capturePhaseAge = phase - STAGE_APEX_PHASES[captureStage];
      flashAges[i] = capturePhaseAge / speed;
    } else {
      // IN TRANSIT — particles enter the stage distributed broadly across
      // the bore cross-section (some near the wall, some further in), then
      // funnel VISIBLY toward the cone's apex node as they advance through
      // the stage.  By localPhase=1.0 the radius has collapsed to ~5–8% of
      // the bore — matching the 2D where particles converge at the apex
      // and drop through the ejection pipe into the channel.
      const xSvg = SVG_X_MIN + phase * SVG_X_RANGE;
      let stageIdx = 0;
      if (xSvg > 306) stageIdx = 1;
      if (xSvg > 494) stageIdx = 2;
      const stg = SVG_STAGE_X[stageIdx];
      const localPhase = (xSvg - stg.xStart) / (stg.xEnd - stg.xStart);

      const angle = seed * 1.93 + i * 0.097;
      const baseR = 0.14 + (Math.sin(seed * 3.7) * 0.5 + 0.5) * 0.38;   // [0.14, 0.52]

      // Cone envelope radius at this localPhase — same cubic Bezier used by
      // buildBezierConeProfile, so the particle is constrained to stay
      // INSIDE the visible cone wall.  A particle's center can never be on
      // the downstream side of the mesh: it's clamped to coneEnvR minus its
      // own visual size minus a small padding.
      const baseRCone = 0.52;
      const apexRCone = [0.10, 0.075, 0.05][stageIdx];
      const uBz = 1 - localPhase;
      const coneEnvR =
        uBz * uBz * uBz * baseRCone
        + 3 * uBz * uBz * localPhase * baseRCone
        + 3 * uBz * localPhase * localPhase * (baseRCone + apexRCone) * 0.55
        + localPhase * localPhase * localPhase * apexRCone;
      const particleSize = diameterToWorldSize(d_p_um);
      const maxParticleR = Math.max(0.04, coneEnvR * 0.92 - particleSize);

      // Only converge if THIS stage is where the particle gets captured.
      // Pass-through particles flow straight without being pulled to the apex.
      const willCaptureHere = (captureStage === stageIdx);
      const apexPullStrength = willCaptureHere
        ? Math.max(0, (localPhase - 0.35) / 0.65)
        : 0;
      const constrainedR = Math.min(baseR, maxParticleR);
      const r = constrainedR * (1 - apexPullStrength * 0.92);
      // Captured particles also follow the sheared cone axis (apex offset to
      // +Z by SHEAR_Z), so they converge to the actual apex node — not the
      // bore axis center.  Pass-through particles stay on the bore axis
      // because they're slipping through the mesh openings, not following
      // the cone interior.
      const zConeAxis = willCaptureHere ? localPhase * SHEAR_Z : 0;

      wx = nx(xSvg);
      wy = r * Math.cos(angle);
      wz = r * Math.sin(angle) + zConeAxis;
      status = 'in_transit';
      flashAges[i] = -1;
    }

    worldPositions[i * 3 + 0] = wx;
    worldPositions[i * 3 + 1] = wy;
    worldPositions[i * 3 + 2] = wz;

    particles.push({
      x: 0,        // SVG coords unused now; world coords are authoritative
      y: 0,
      species: spec,
      d_p_um,
      status,
    });
  }

  return { particles, flashAges, worldPositions, fateDistribution };
}

// ─────────────────────────────────────────────────────────────────────────
//  Particle field — GPU-instanced rendering with custom shaders
// ─────────────────────────────────────────────────────────────────────────

interface ParticleFieldProps {
  realParticles: ParticlePoint[];
  capacity: number;
  syntheticMode: boolean;
  onFateUpdate?: (dist: [number, number, number, number]) => void;
}

function ParticleField({ realParticles, capacity, syntheticMode, onFateUpdate }: ParticleFieldProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);

  const { iPosition, iSize, iColor, iCaptured, iFlashAge } = useMemo(() => ({
    iPosition: new Float32Array(capacity * 3),
    iSize: new Float32Array(capacity),
    iColor: new Float32Array(capacity * 3),
    iCaptured: new Float32Array(capacity),
    iFlashAge: new Float32Array(capacity),
  }), [capacity]);

  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: particleVertexShader,
        fragmentShader: particleFragmentShader,
        transparent: false,
      }),
    []
  );

  // The fate distribution is deterministic and stable per generation; report
  // it once on first frame and skip per-frame updates to avoid React thrash.
  const fateReported = useRef(false);

  useFrame(({ clock }) => {
    if (!meshRef.current) return;
    const m = meshRef.current;

    let activeParticles: ParticlePoint[];
    let activeFlashAges: Float32Array;
    let activeWorldPositions: Float32Array | null;

    if (syntheticMode) {
      const out = generateRealisticParticles(capacity, clock.elapsedTime);
      activeParticles = out.particles;
      activeFlashAges = out.flashAges;
      activeWorldPositions = out.worldPositions;
      if (!fateReported.current && onFateUpdate) {
        onFateUpdate(out.fateDistribution);
        fateReported.current = true;
      }
    } else {
      activeParticles = realParticles;
      activeFlashAges = new Float32Array(activeParticles.length).fill(-1);
      activeWorldPositions = null;
    }

    const n = Math.min(activeParticles.length, capacity);

    for (let i = 0; i < n; i++) {
      const p = activeParticles[i];

      if (activeWorldPositions) {
        // Synthetic mode: use pre-computed world coords directly (handles both
        // in-transit AND captured particles, with captured placed in channels).
        iPosition[i * 3 + 0] = activeWorldPositions[i * 3 + 0];
        iPosition[i * 3 + 1] = activeWorldPositions[i * 3 + 1];
        iPosition[i * 3 + 2] = activeWorldPositions[i * 3 + 2];
      } else {
        // Real-data mode: route the backend particle through the same rich
        // positioning logic as synthetic — cone-envelope clamping,
        // sheared-axis tracking, status-based pipe/channel placement.
        const [wx, wy, wz] = computeRealParticleWorldPos(p, i);
        iPosition[i * 3 + 0] = wx;
        iPosition[i * 3 + 1] = wy;
        iPosition[i * 3 + 2] = wz;
      }

      iSize[i] = diameterToWorldSize(p.d_p_um);

      const [r, g, b] = speciesColor(p.species);
      iColor[i * 3 + 0] = r;
      iColor[i * 3 + 1] = g;
      iColor[i * 3 + 2] = b;

      iCaptured[i] = p.status === 'captured' ? 1.0 : 0.0;
      iFlashAge[i] = activeFlashAges[i] ?? -1;
    }
    for (let i = n; i < capacity; i++) {
      iSize[i] = 0;
    }

    const geom = m.geometry;
    (geom.attributes.iPosition as THREE.InstancedBufferAttribute).needsUpdate = true;
    (geom.attributes.iSize as THREE.InstancedBufferAttribute).needsUpdate = true;
    (geom.attributes.iColor as THREE.InstancedBufferAttribute).needsUpdate = true;
    (geom.attributes.iCaptured as THREE.InstancedBufferAttribute).needsUpdate = true;
    (geom.attributes.iFlashAge as THREE.InstancedBufferAttribute).needsUpdate = true;
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
        <instancedBufferAttribute attach="attributes-iFlashAge" args={[iFlashAge, 1]} />
      </sphereGeometry>
    </instancedMesh>
  );
}

// ─────────────────────────────────────────────────────────────────────────
//  Top-level WebGL view with HUD legend + live capture statistics
// ─────────────────────────────────────────────────────────────────────────

interface WebGLCascadeViewProps {
  state: HydrosDisplayState | null;
}

export default function WebGLCascadeView({ state }: WebGLCascadeViewProps) {
  const realParticles = useMemo(() => {
    const streams = state?.particleStreams;
    if (!streams) return [];
    // Include the per-stage deposited streams so each channel visibly carries
    // its accumulated capture load (previously these were dropped from the
    // 3D render and only the in-transit tracers were shown).  The mapDeposited
    // function places each deposited particle's xSvg inside its stage's
    // xStart→apexX range, so the stage inference in
    // computeRealParticleWorldPos picks them up correctly.
    const asCaptured = (deps: { x: number; species: string; d_p_um: number }[]): ParticlePoint[] =>
      deps.map(d => ({
        x: d.x,
        y: 280,           // placeholder; the captured branch uses channelWorldPos directly
        species: d.species,
        d_p_um: d.d_p_um,
        status: 'captured' as const,
      }));
    return [
      ...streams.s1,
      ...streams.s2,
      ...streams.s3,
      ...asCaptured(streams.s1Deposited),
      ...asCaptured(streams.s2Deposited),
      ...asCaptured(streams.s3Deposited),
    ];
  }, [state?.particleStreams]);

  const hasRealData = realParticles.length > 0;

  const eField = state?.eField ?? 0;
  const clog = state?.clog ?? 0;
  const eFieldArr: number[] = [
    eField * SVG_STAGE_X[0].mult,
    eField * SVG_STAGE_X[1].mult,
    eField * SVG_STAGE_X[2].mult,
  ];
  const clogArr: number[] = [clog, clog, clog];

  const [fateDist, setFateDist] = useState<[number, number, number, number]>([0, 0, 0, 0]);

  const storageFillPct = Math.round((state?.storageFill ?? 0) * 100);
  const flushFlags = [
    state?.flushActiveS1 ?? false,
    state?.flushActiveS2 ?? false,
    state?.flushActiveS3 ?? false,
  ];
  const activeFlushes = ['S1', 'S2', 'S3'].filter((_, i) => flushFlags[i]);
  const flushDisplay = activeFlushes.length > 0 ? activeFlushes.join(' + ') : 'idle';
  const anyFlush = activeFlushes.length > 0;

  return (
    <div style={{ width: '100%', height: '100%', background: '#080D18', position: 'relative' }}>
      <Canvas
        camera={{ position: [1.8, 0.55, 1.95], fov: 42 }}
        gl={{ antialias: true, powerPreference: 'high-performance' }}
        dpr={[1, 2]}
      >
        <ambientLight intensity={0.4} />
        <hemisphereLight color="#88AACC" groundColor="#1E293B" intensity={0.55} />
        <directionalLight position={[3, 4, 2]} intensity={1.3} color="#FFFFFF" />
        <directionalLight position={[-2, 1, -3]} intensity={0.5} color="#7DD3FC" />
        <pointLight position={[0, 0, 0]} intensity={0.7} color="#FBBF24" distance={2} />
        <Environment preset="warehouse" background={false} />

        <ReactorHousing />

        <PolZone />

        <ConicalStage stageIdx={0} eField={eFieldArr[0]} clogLevel={clogArr[0]} />
        <ConicalStage stageIdx={1} eField={eFieldArr[1]} clogLevel={clogArr[1]} />
        <ConicalStage stageIdx={2} eField={eFieldArr[2]} clogLevel={clogArr[2]} />

        <StageLabel stageIdx={0} />
        <StageLabel stageIdx={1} />
        <StageLabel stageIdx={2} />

        <ExtractionChannel stageIdx={0} />
        <ExtractionChannel stageIdx={1} />
        <ExtractionChannel stageIdx={2} />

        <EjectionPipe stageIdx={0} />
        <EjectionPipe stageIdx={1} />
        <EjectionPipe stageIdx={2} />

        <FlushPort stageIdx={0} active={state?.flushActiveS1 ?? false} />
        <FlushPort stageIdx={1} active={state?.flushActiveS2 ?? false} />
        <FlushPort stageIdx={2} active={state?.flushActiveS3 ?? false} />

        <StorageChamber
          fill={state?.storageFill ?? 0}
          storedParticles={state?.particleStreams?.storage ?? []}
        />

        <OutletLabel />

        <ParticleField
          realParticles={realParticles}
          capacity={1500}
          syntheticMode={!hasRealData}
          onFateUpdate={setFateDist}
        />

        <OrbitControls
          target={[0.2, 0, 0.5]}
          enablePan={false}
          minDistance={1.6}
          maxDistance={7}
          maxPolarAngle={Math.PI / 1.6}
          autoRotate
          autoRotateSpeed={0.2}
        />
      </Canvas>

      {/* Top-left HUD: stack identity + capture stats */}
      <div
        style={{
          position: 'absolute',
          top: 12,
          left: 16,
          padding: '8px 12px',
          background: 'rgba(8, 13, 24, 0.75)',
          border: '1px solid rgba(56, 189, 248, 0.4)',
          borderRadius: 4,
          color: '#7DD3FC',
          font: '11px/1.45 "JetBrains Mono", "Fira Code", monospace',
          pointerEvents: 'none',
          minWidth: 220,
        }}
      >
        <div style={{ color: '#A5F3FC', marginBottom: 4, fontWeight: 600 }}>
          WebGL 3D · Three.js / R3F · custom GLSL shaders
        </div>
        <div style={{ color: '#CBD5E1', opacity: 0.85 }}>
          {hasRealData ? `${realParticles.length} particles · live` : '1500 particles · synthetic demo'}
          {' · '}
          {Math.round(eFieldArr.reduce((a, b) => a + b, 0) * 100 / 3)}% mean field
        </div>
        <div style={{ marginTop: 6, borderTop: '1px solid rgba(56, 189, 248, 0.2)', paddingTop: 6 }}>
          <div style={{ color: '#94A3B8', fontSize: 10, marginBottom: 3 }}>FILTRATION TARGETS · STABLE FATE DISTRIBUTION</div>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            <span style={{ color: '#FB923C' }}>S1: {fateDist[0]}</span>
            <span style={{ color: '#FBBF24' }}>S2: {fateDist[1]}</span>
            <span style={{ color: '#38BDF8' }}>S3: {fateDist[2]}</span>
            <span style={{ color: '#94A3B8' }}>pass-through: {fateDist[3]}</span>
          </div>
        </div>
        <div style={{ marginTop: 6, borderTop: '1px solid rgba(56, 189, 248, 0.2)', paddingTop: 6 }}>
          <div style={{ color: '#94A3B8', fontSize: 10, marginBottom: 3 }}>SYSTEM STATE</div>
          <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap' }}>
            <span style={{ color: '#7DD3FC' }}>storage: {storageFillPct}%</span>
            <span style={{ color: anyFlush ? '#38BDF8' : '#94A3B8', fontWeight: anyFlush ? 600 : 400 }}>
              flush: {flushDisplay}
            </span>
          </div>
        </div>
      </div>

      {/* Bottom-left HUD: particle size legend (sqrt mapping, log-scale physical) */}
      <div
        style={{
          position: 'absolute',
          bottom: 12,
          left: 16,
          padding: '8px 12px',
          background: 'rgba(8, 13, 24, 0.75)',
          border: '1px solid rgba(148, 163, 184, 0.3)',
          borderRadius: 4,
          color: '#CBD5E1',
          font: '10px/1.5 "JetBrains Mono", "Fira Code", monospace',
          pointerEvents: 'none',
        }}
      >
        <div style={{ color: '#94A3B8', marginBottom: 3 }}>PARTICLE SIZE LEGEND</div>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <span style={{ color: '#66B3FF' }}>● 25 µm PET</span>
          <span style={{ color: '#4DD89D', fontSize: 12 }}>● 200 µm PE</span>
          <span style={{ color: '#FA8033', fontSize: 14 }}>● 800 µm PP</span>
        </div>
        <div style={{ color: '#64748B', fontSize: 9, marginTop: 2 }}>
          PP&gt;S1 mesh (500µm), PE&gt;S2 (100µm), PET&gt;S3 (5µm)
        </div>
      </div>
    </div>
  );
}
