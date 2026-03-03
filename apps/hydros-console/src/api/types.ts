/**
 * api/types.ts
 *
 * Type definitions for all HydrOS artifact structures consumed by the console.
 *
 * GOVERNANCE:
 *   - Every field here maps 1:1 to a named field produced by the backend.
 *   - No fields may be added that are not present in run_artifact_v1.
 *   - No client-side derived metrics. No computed fields. No UI convenience
 *     aliases that obscure artifact provenance.
 *   - Truth and sensor namespaces are kept strictly separate in all types.
 *     They must never be merged or co-typed.
 *
 * Source of record:
 *   hydrion/logging/artifacts.py  — append_spine_step(), write_manifest(),
 *                                   compute_metrics()
 *   hydrion/service/app.py        — route response shapes
 */

// ---------------------------------------------------------------------------
// RunRequest — payload for POST /api/run
// ---------------------------------------------------------------------------

export interface RunRequest {
  /** "random" | "ppo" | "baseline" — determines action selection policy */
  policy_type: string;
  seed: number;
  /** Filename only, resolved by backend under configs/ e.g. "default.yaml" */
  config_name: string;
  max_steps: number;
  noise_enabled: boolean;
}

// ---------------------------------------------------------------------------
// RunManifest — manifest.json
// ---------------------------------------------------------------------------

export interface RunManifest {
  /** Canonical run identifier: run_<unix_timestamp>_<seed> */
  run_id: string;
  /** Always "run_artifact_v1" — frozen schema version */
  artifact_schema: 'run_artifact_v1';
  engine_version: string;
  /** "obs12_v1" — 12D observation vector schema */
  obs_schema: string;
  /** "act4_v1" — 4-channel action schema */
  act_schema: string;
  /** SHA hash of the YAML config used for this run */
  config_hash: string;
  seed: number;
  policy_type: string;
  noise_enabled: boolean;
  /** Physics timestep in seconds */
  dt: number;
  created_at_utc: string;
}

// ---------------------------------------------------------------------------
// SpineStep — one row from spine.jsonl
// Represents a single simulation step. All values are normalized [0, 1]
// unless otherwise noted.
// ---------------------------------------------------------------------------

/**
 * Physics truth state for one step.
 * These values originate from env.truth_state — they are authoritative.
 * They are NOT measurements. They are the ground-truth physics simulation output.
 */
export interface SpineTruth {
  flow_norm: number;
  pressure_norm: number;
  clog_norm: number;
  E_norm: number;
  C_out: number;
  particle_capture_eff: number;
}

/**
 * Sensor state for one step.
 * These values originate from env.sensor_state — they are observational.
 * They may contain noise. They are NOT truth values.
 * Never promote sensor readings to ground truth in any UI component.
 */
export interface SpineSensors {
  turbidity: number;
  scatter: number;
}

/**
 * Actuator command state for one step.
 * Reflects the safe_action vector after shield pre-processing.
 * All values normalized [0, 1].
 */
export interface SpineActions {
  valve_cmd: number;
  pump_cmd: number;
  bf_cmd: number;
  node_voltage_cmd: number;
}

/** Safety layer state for one step. */
export interface SpineSafety {
  /** True if the shield projected (modified) the action before env.step */
  shield_intervened: boolean;
  /** True if any soft or hard violation occurred this step */
  violation: boolean;
  /** Human-readable violation category or null if no violation */
  violation_kind: string | null;
}

/** Event flags for one step. */
export interface SpineEvents {
  anomaly_active: boolean;
}

/** Complete representation of one simulation step from spine.jsonl */
export interface SpineStep {
  step_idx: number;
  /** Simulation wall-clock time in seconds: step_idx * dt */
  sim_time_s: number;
  truth: SpineTruth;
  sensors: SpineSensors;
  actions: SpineActions;
  reward: number;
  done: boolean;
  truncated: boolean;
  events: SpineEvents;
  safety: SpineSafety;
}

// ---------------------------------------------------------------------------
// RunMetrics — metrics.json (computed by artifacts.compute_metrics)
// ---------------------------------------------------------------------------

export interface RunMetrics {
  run_id: string;
  total_steps: number;
  total_reward: number;
  mean_reward: number;
  /** True if system held stable for >= 50 consecutive steps */
  stability_reached: boolean;
  /** Step index at which stability was first achieved, or null */
  time_to_stability_steps: number | null;
  shield_intervention_count: number;
  violation_count: number;
  mean_flow: number;
  mean_pressure: number;
  mean_clog: number;
  mean_E_norm: number;
  mean_particle_capture_eff: number;
}

// ---------------------------------------------------------------------------
// ApiError — structured error returned by all client functions
// ---------------------------------------------------------------------------

export interface ApiError {
  /** HTTP status code, or 0 for network/timeout failures */
  status: number;
  /** Human-readable message safe to surface in diagnostics panels */
  message: string;
  /** Original endpoint path that failed — for logging and debugging */
  endpoint: string;
}

// ---------------------------------------------------------------------------
// Result<T> — discriminated union returned by all client functions.
// Forces the caller to handle both success and failure paths explicitly.
// No client function throws. All errors are returned as values.
// ---------------------------------------------------------------------------

export type Result<T> =
  | { ok: true; data: T }
  | { ok: false; error: ApiError };
