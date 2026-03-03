/**
 * api/USAGE_EXAMPLES.ts
 *
 * Reference examples for correct API client consumption.
 *
 * This file is for developer reference only. It is not imported anywhere.
 * It documents the intended usage patterns for Step 2 (run library sidebar)
 * and Step 3 (ConsoleView orchestration) implementors.
 *
 * Every example demonstrates:
 *   - Result<T> destructuring (ok/error path handling is mandatory)
 *   - No metric recomputation
 *   - No artifact transformation
 *   - No direct env.step calls
 */

import {
  startRun,
  listRuns,
  getManifest,
  getSpine,
  getMetrics,
} from './index';

import type { SpineStep, RunManifest, RunMetrics } from './index';

// ---------------------------------------------------------------------------
// Example 1: Triggering a new run (user presses "Run Simulation")
// ---------------------------------------------------------------------------

async function exampleStartRun() {
  const result = await startRun({
    policy_type: 'random',
    seed: 42,
    config_name: 'default.yaml',
    max_steps: 200,
    noise_enabled: true,
  });

  if (!result.ok) {
    // Surface in a status indicator — do not throw, do not crash the console
    console.error(`[HydrOS] Run failed — ${result.error.message}`);
    return;
  }

  // result.data.run_id is the canonical identifier for this run
  // Pass it to listRuns() or directly to getSpine() for immediate loading
  const runId = result.data.run_id;
  console.log(`[HydrOS] Run complete: ${runId}`);
}

// ---------------------------------------------------------------------------
// Example 2: Populating the run library sidebar
// ---------------------------------------------------------------------------

async function exampleListRuns(): Promise<string[]> {
  const result = await listRuns();

  if (!result.ok) {
    console.error(`[HydrOS] Could not fetch run list — ${result.error.message}`);
    return [];
  }

  // result.data is string[] — run_ids in filesystem order (ascending by timestamp)
  // The sidebar renders these directly; no sorting or transformation needed
  return result.data;
}

// ---------------------------------------------------------------------------
// Example 3: Loading a run for display (ConsoleView)
// Manifest and spine are fetched independently — they serve different
// panel concerns and should not be merged into a single object.
// ---------------------------------------------------------------------------

async function exampleLoadRun(runId: string): Promise<{
  manifest: RunManifest;
  spine: SpineStep[];
  metrics: RunMetrics;
} | null> {
  // Fetch all three in parallel — they are independent reads
  const [manifestResult, spineResult, metricsResult] = await Promise.all([
    getManifest(runId),
    getSpine(runId),
    getMetrics(runId),
  ]);

  if (!manifestResult.ok) {
    console.error(`[HydrOS] Manifest fetch failed — ${manifestResult.error.message}`);
    return null;
  }
  if (!spineResult.ok) {
    console.error(`[HydrOS] Spine fetch failed — ${spineResult.error.message}`);
    return null;
  }
  if (!metricsResult.ok) {
    console.error(`[HydrOS] Metrics fetch failed — ${metricsResult.error.message}`);
    return null;
  }

  return {
    manifest: manifestResult.data,
    spine: spineResult.data,
    metrics: metricsResult.data,
  };
}

// ---------------------------------------------------------------------------
// Example 4: Correct field access from a SpineStep
// Demonstrates truth vs sensor separation at point of use.
// ---------------------------------------------------------------------------

function exampleConsumeStep(step: SpineStep) {
  // Truth fields — physics ground truth, authoritative
  const flow    = step.truth.flow_norm;        // drives particle velocity in animated core
  const clog    = step.truth.clog_norm;        // drives mesh overlay opacity
  const eNorm   = step.truth.E_norm;           // drives electrostatic intensity overlay
  const bfCmd   = step.actions.bf_cmd;         // drives backflush pulse animation (threshold: 0.5)

  // Sensor fields — observational only, may contain noise
  const turbidity = step.sensors.turbidity;   // displayed in sensor panel, NOT used for physics viz
  const scatter   = step.sensors.scatter;     // same

  // Safety fields — drive indicator states
  const shieldActive = step.safety.shield_intervened;  // shield flash indicator
  const hasViolation = step.safety.violation;           // violation marker

  // Events
  const anomalyActive = step.events.anomaly_active;    // anomaly indicator

  // INCORRECT — never mix truth and sensor namespaces:
  // const mixed = step.truth.flow_norm + step.sensors.turbidity; // ← reject
}

// ---------------------------------------------------------------------------
// Example 5: Accessing metrics — no recomputation
// ---------------------------------------------------------------------------

function exampleConsumeMetrics(metrics: RunMetrics) {
  // These values come from artifacts.compute_metrics() on the backend.
  // Never recompute stability_reached, shield counts, or mean values
  // from the spine array on the client side.
  const stable          = metrics.stability_reached;
  const timeToStability = metrics.time_to_stability_steps; // null if never reached
  const interventions   = metrics.shield_intervention_count;
  const meanClog        = metrics.mean_clog;
}
