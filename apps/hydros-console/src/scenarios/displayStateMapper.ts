/**
 * scenarios/displayStateMapper.ts
 *
 * Maps a ScenarioStepRecord (raw truth_state from hydrion physics engine)
 * to HydrosDisplayState — the full display-layer schema consumed by all
 * Machine View components.
 *
 * Rules:
 *   - Source: step.truthState only (physics truth, never sensor_state)
 *   - All numeric outputs explicitly bounded [0, 1] via clamp01 where appropriate
 *   - Fallback to 0 for any missing truth_state field
 *   - Derived labels computed from governed precedence chains
 */

import type { ScenarioStepRecord, ParticlePointRaw } from '../api/types';

// ---------------------------------------------------------------------------
// Particle stream types
// ---------------------------------------------------------------------------

export interface ParticlePoint {
  x: number;        // SVG coordinate (converted from x_norm)
  y: number;        // SVG coordinate (converted from r_norm)
  status: string;   // "captured" | "passed"
  species: string;  // "PP" | "PE" | "PET"
  trail?: Array<{ x: number; y: number }>;  // SVG-space trajectory path
}

export interface ParticleStreams {
  s1: ParticlePoint[];
  s2: ParticlePoint[];
  s3: ParticlePoint[];
}

// Stage geometry for (x_norm, r_norm) → SVG coordinate conversion.
// These constants must match ConicalCascadeView.tsx STAGES exactly.
// If the geometry changes, update both files.
const _CY = 154;  // device centreline y (matches ConicalCascadeView CY)
const _STAGE_GEOM = [
  { xStart: 118, apexX: 296, apexY: 243 },  // S1
  { xStart: 306, apexX: 484, apexY: 243 },  // S2
  { xStart: 494, apexX: 672, apexY: 243 },  // S3
] as const;

function coneToSVG(
  xNorm: number,
  rNorm: number,
  stageIdx: number,
): { x: number; y: number } {
  const stg = _STAGE_GEOM[stageIdx];
  // The cone wall narrows linearly from apexY at x=xStart to CY at x=apexX.
  // r_norm is the fraction of the LOCAL wall height from the centreline.
  // Without the (1−xNorm) taper factor, particles at x_norm>0.5 are placed
  // outside the visible cone boundary and render off-screen.
  return {
    x: stg.xStart + xNorm * (stg.apexX - stg.xStart),
    y: _CY + rNorm * (stg.apexY - _CY) * (1 - xNorm),
  };
}

function mapParticleStream(
  raw: ParticlePointRaw[] | undefined,
  stageIdx: number,
): ParticlePoint[] {
  if (!raw || raw.length === 0) return [];
  return raw.map(p => ({
    ...coneToSVG(p.x_norm, p.r_norm, stageIdx),
    status:  p.status,
    species: p.species,
    trail:   p.trail?.map(tp => coneToSVG(tp.x_norm, tp.r_norm, stageIdx)),
  }));
}

export type SystemStatus =
  | 'BACKFLUSH ACTIVE'
  | 'BYPASS ACTIVE'
  | 'MAINTENANCE REQUIRED'
  | 'RISING LOAD'
  | 'NORMAL OPERATION';

export type FiberLoadLabel = 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL';
export type PulseLabel = 'READY' | 'ACTIVE' | 'COOLDOWN';
export type BypassLabel = 'OFF' | 'READY' | 'ACTIVE';

export interface HydrosDisplayState {
  // MachineCore visual inputs
  running: boolean;
  flow: number;        // 0..1  (truth_state.flow)
  clog: number;        // 0..1  (truth_state.clog / mesh_loading_avg)
  eField: number;      // 0..1  (truth_state.E_norm)
  backflush: number;   // 0..1  (truth_state.bf_active)
  storageFill: number; // 0..1  storage chamber fill level

  // M5 ConicalCascade — per-stage accumulation and efficiency
  channelFillS1: number;   // [0,1] particle volume fraction in S1 collection channel
  channelFillS2: number;
  channelFillS3: number;
  etaS1: number;           // [0,1] S1 stage capture efficiency (PET representative)
  etaS2: number;
  etaS3: number;           // always >= etaS1 by design (asymmetric stages)
  vCritNorm: number;       // v_crit_s3 / OBS_VCRIT_MAX normalised to [0,1]
  etaPP: number;           // [0,1] buoyant species (PP) efficiency — drives density split cue
  etaPET: number;          // [0,1] dense species (PET) efficiency
  flushActiveS1: boolean;  // hydraulic flush active on S1 channel
  flushActiveS2: boolean;
  flushActiveS3: boolean;

  // Raw physical values (for display)
  q_processed_lmin: number;  // Q_out_Lmin or commanded fallback
  pressureKpa: number;       // dp_total_pa / 1000 (or scaled from normalized)
  q_bypass_lmin: number;     // truth_state.q_bypass_lmin
  bf_pulse_idx: number;      // truth_state.bf_pulse_idx
  bf_cooldown_remaining: number; // truth_state.bf_cooldown_remaining

  // Normalized metrics
  flowLmin: number;    // commanded flow (scenarioInputs.flowLmin)
  pressure: number;    // 0..1 normalized
  captureEff: number;  // 0..1

  // Per-stage fouling (0..1)
  foulingS1: number;
  foulingS2: number;
  foulingS3: number;

  // Status flags
  maintenanceRequired: boolean;
  bypassActive: boolean;

  // Derived display labels (governed precedence chains)
  systemStatus: SystemStatus;
  fiberLoadLabel: FiberLoadLabel;
  pulseLabel: PulseLabel;
  bypassLabel: BypassLabel;
  efficiencyPct: number;     // captureEff * 100
  nextActions: string[];

  // Step identity
  t: number;
  stepIndex: number;
  reward: number;

  // Per-stage particle positions from ParticleDynamicsEngine
  particleStreams?: ParticleStreams | null;
}

function clamp01(x: number | undefined): number {
  if (x === undefined || !Number.isFinite(x)) return 0;
  return Math.max(0, Math.min(1, x));
}

function deriveSystemStatus(
  backflush: number,
  bypassActive: boolean,
  maintenanceRequired: boolean,
  clog: number,
): SystemStatus {
  if (backflush > 0.5) return 'BACKFLUSH ACTIVE';
  if (bypassActive) return 'BYPASS ACTIVE';
  if (maintenanceRequired) return 'MAINTENANCE REQUIRED';
  if (clog > 0.7) return 'RISING LOAD';
  return 'NORMAL OPERATION';
}

function deriveFiberLoad(clog: number): FiberLoadLabel {
  if (clog < 0.25) return 'LOW';
  if (clog < 0.50) return 'MODERATE';
  if (clog < 0.75) return 'HIGH';
  return 'CRITICAL';
}

function derivePulse(bfActive: number, bfCooldown: number): PulseLabel {
  if (bfActive > 0.5) return 'ACTIVE';
  if (bfCooldown > 0) return 'COOLDOWN';
  return 'READY';
}

function deriveBypass(
  bypassActive: boolean,
  pressure: number,
  maintenanceRequired: boolean,
): BypassLabel {
  if (bypassActive) return 'ACTIVE';
  if (pressure > 0.8 || maintenanceRequired) return 'READY';
  return 'OFF';
}

function deriveNextActions(
  systemStatus: SystemStatus,
  foulingS3: number,
): string[] {
  switch (systemStatus) {
    case 'BACKFLUSH ACTIVE':
      return ['WAIT — BACKFLUSH IN PROGRESS'];
    case 'BYPASS ACTIVE':
      return ['ISOLATE STAGES', 'CLEAR BLOCKAGE'];
    case 'MAINTENANCE REQUIRED': {
      const actions = ['SCHEDULE MAINTENANCE'];
      if (foulingS3 > 0.8) actions.push('INSPECT STAGE 3');
      return actions;
    }
    case 'RISING LOAD':
      return ['INITIATE BACKFLUSH', 'REDUCE FLOW'];
    default:
      return ['MONITOR — NO ACTION'];
  }
}

export function mapStepRecordToDisplayState(step: ScenarioStepRecord): HydrosDisplayState {
  const ts = step.truthState;

  const backflush = clamp01(ts['bf_active']);
  const bypassActive = (ts['bypass_active'] ?? 0) > 0.5;
  const maintenanceRequired = (ts['maintenance_required'] ?? 0) > 0.5;
  const clog = clamp01(ts['clog'] ?? ts['mesh_loading_avg']);
  const pressure = clamp01(ts['pressure']);
  const foulingS3 = clamp01(ts['fouling_frac_s3']);
  const bfCooldown = ts['bf_cooldown_remaining'] ?? 0;

  const systemStatus = deriveSystemStatus(backflush, bypassActive, maintenanceRequired, clog);
  const fiberLoadLabel = deriveFiberLoad(clog);
  const pulseLabel = derivePulse(backflush, bfCooldown);
  const bypassLabel = deriveBypass(bypassActive, pressure, maintenanceRequired);
  const efficiencyPct = clamp01(ts['capture_eff']) * 100;
  const nextActions = deriveNextActions(systemStatus, foulingS3);

  // Physical pressure: dp_total_pa if available, else scale normalized → ~0–100 kPa
  const dpPa = ts['dp_total_pa'] ?? (pressure * 100000);
  const pressureKpa = dpPa / 1000;

  // Actual processed flow: Q_out_Lmin if available, else commanded
  const q_processed_lmin =
    ts['Q_out_Lmin'] ?? ts['q_processed_lmin'] ?? step.scenarioInputs.flowLmin;

  return {
    running: !step.done,

    flow: clamp01(ts['flow']),
    clog,
    eField: clamp01(ts['E_norm']),
    backflush,
    storageFill: clamp01(ts['storage_fill']),

    channelFillS1: clamp01(ts['channel_fill_s1']),
    channelFillS2: clamp01(ts['channel_fill_s2']),
    channelFillS3: clamp01(ts['channel_fill_s3']),
    etaS1: clamp01(ts['eta_s1']),
    etaS2: clamp01(ts['eta_s2']),
    etaS3: clamp01(ts['eta_s3']),
    vCritNorm: clamp01((ts['v_crit_s3'] ?? 0) / 0.10),
    etaPP:  clamp01(ts['eta_PP']),
    etaPET: clamp01(ts['eta_PET']),
    flushActiveS1: (ts['flush_active_s1'] ?? 0) > 0.5,
    flushActiveS2: (ts['flush_active_s2'] ?? 0) > 0.5,
    flushActiveS3: (ts['flush_active_s3'] ?? 0) > 0.5,

    q_processed_lmin,
    pressureKpa,
    q_bypass_lmin: ts['q_bypass_lmin'] ?? 0,
    bf_pulse_idx: ts['bf_pulse_idx'] ?? 0,
    bf_cooldown_remaining: bfCooldown,

    flowLmin: step.scenarioInputs.flowLmin,
    pressure,
    captureEff: clamp01(ts['capture_eff']),

    foulingS1: clamp01(ts['fouling_frac_s1']),
    foulingS2: clamp01(ts['fouling_frac_s2']),
    foulingS3,

    maintenanceRequired,
    bypassActive,

    systemStatus,
    fiberLoadLabel,
    pulseLabel,
    bypassLabel,
    efficiencyPct,
    nextActions,

    t: step.t,
    stepIndex: step.stepIndex,
    reward: step.reward,

    particleStreams: step.particleStreams
      ? {
          s1: mapParticleStream(step.particleStreams.s1, 0),
          s2: mapParticleStream(step.particleStreams.s2, 1),
          s3: mapParticleStream(step.particleStreams.s3, 2),
        }
      : null,
  };
}
