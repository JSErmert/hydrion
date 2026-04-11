/**
 * api/client.ts
 *
 * Typed fetch wrappers for all HydrOS backend routes.
 *
 * GOVERNANCE:
 *   - No function here calls env.step or mutates simulation state.
 *   - No function performs client-side metric computation.
 *   - No function transforms or re-shapes artifact data beyond JSON parsing.
 *   - All functions return Result<T> — never throw. Callers must handle
 *     both ok and error paths. This is non-negotiable for research stability.
 *   - AbortController is used for timeout enforcement on every request.
 *   - Artifact reads (GET) use ARTIFACT_TIMEOUT_MS.
 *   - Run execution (POST) uses DEFAULT_TIMEOUT_MS to accommodate full
 *     synchronous simulation execution on the backend.
 *
 * Error handling strategy:
 *   Three failure categories are distinguished:
 *   1. Network failure / timeout (status: 0) — backend unreachable
 *   2. HTTP error response (status: 4xx/5xx) — backend reachable, request rejected
 *   3. Parse failure (status: 0, message prefixed "parse:") — malformed response
 *
 *   All three produce an ApiError and return { ok: false, error }. No
 *   category is treated as fatal at this layer. Callers decide severity.
 */

import {
  API_BASE_URL,
  DEFAULT_TIMEOUT_MS,
  ARTIFACT_TIMEOUT_MS,
} from './config';

import type {
  RunRequest,
  RunManifest,
  SpineStep,
  RunMetrics,
  ScenarioInfo,
  ScenarioExecutionHistory,
  ApiError,
  Result,
} from './types';

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Build an AbortController that fires after `ms` milliseconds.
 * Returns both the controller (for signal) and the timer id (for cleanup).
 */
function makeTimeout(ms: number): { signal: AbortSignal; clear: () => void } {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  return {
    signal: controller.signal,
    clear: () => clearTimeout(timer),
  };
}

/**
 * Core fetch executor. Handles timeout, HTTP status checking, and JSON parsing.
 * Returns a Result<T>. Never throws.
 */
async function fetchJson<T>(
  endpoint: string,
  options: RequestInit,
  timeoutMs: number,
): Promise<Result<T>> {
  const { signal, clear } = makeTimeout(timeoutMs);

  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      signal,
    });

    clear(); // request completed — cancel the abort timer

    if (!response.ok) {
      // Attempt to extract a detail message from FastAPI's error body
      let detail = `HTTP ${response.status}`;
      try {
        const body = await response.json();
        if (typeof body?.detail === 'string') {
          detail = body.detail;
        }
      } catch {
        // error body was not JSON — use status text
        detail = response.statusText || detail;
      }

      const error: ApiError = {
        status: response.status,
        message: detail,
        endpoint,
      };
      return { ok: false, error };
    }

    try {
      const data: T = await response.json();
      return { ok: true, data };
    } catch (parseErr) {
      const error: ApiError = {
        status: 0,
        message: `parse: response from ${endpoint} was not valid JSON`,
        endpoint,
      };
      return { ok: false, error };
    }
  } catch (networkErr) {
    clear();

    const isTimeout =
      networkErr instanceof DOMException && networkErr.name === 'AbortError';

    const error: ApiError = {
      status: 0,
      message: isTimeout
        ? `timeout: no response from ${endpoint} within ${timeoutMs}ms`
        : `network: failed to reach ${endpoint} — ${(networkErr as Error).message}`,
      endpoint,
    };
    return { ok: false, error };
  }
}

// ---------------------------------------------------------------------------
// Public API surface
// Five functions. One per backend route. No extras.
// ---------------------------------------------------------------------------

/**
 * POST /api/run
 *
 * Triggers a full simulation run on the backend. Execution is synchronous
 * on the backend — this call blocks until the run completes and the artifact
 * is written. Returns the run_id of the newly created run.
 *
 * The console must not call this more than once per user-initiated trigger.
 * There is no debounce at this layer — that is the caller's responsibility.
 */
export async function startRun(
  req: RunRequest,
): Promise<Result<{ run_id: string }>> {
  return fetchJson<{ run_id: string }>(
    '/api/run',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    },
    DEFAULT_TIMEOUT_MS,
  );
}

/**
 * GET /api/runs
 *
 * Returns the list of all run_ids present in the runs/ directory, sorted
 * by creation order (ascending). Each entry is a string of the form
 * "run_<timestamp>_<seed>".
 */
export async function listRuns(): Promise<Result<string[]>> {
  return fetchJson<string[]>(
    '/api/runs',
    { method: 'GET' },
    ARTIFACT_TIMEOUT_MS,
  );
}

/**
 * GET /api/runs/{run_id}/manifest
 *
 * Returns the manifest.json for the specified run. Contains run configuration,
 * schema versions, seed, policy_type, dt, and creation timestamp.
 */
export async function getManifest(
  runId: string,
): Promise<Result<RunManifest>> {
  return fetchJson<RunManifest>(
    `/api/runs/${encodeURIComponent(runId)}/manifest`,
    { method: 'GET' },
    ARTIFACT_TIMEOUT_MS,
  );
}

/**
 * GET /api/runs/{run_id}/spine
 *
 * Returns the complete spine.jsonl for the specified run as an ordered array
 * of SpineStep objects. This is the authoritative data source for all
 * animated core rendering and telemetry.
 *
 * The array is ordered by step_idx ascending. The caller must not reorder it.
 * No filtering, slicing, or transformation is performed here.
 */
export async function getSpine(
  runId: string,
): Promise<Result<SpineStep[]>> {
  return fetchJson<SpineStep[]>(
    `/api/runs/${encodeURIComponent(runId)}/spine`,
    { method: 'GET' },
    ARTIFACT_TIMEOUT_MS,
  );
}

/**
 * GET /api/runs/{run_id}/metrics
 *
 * Returns the metrics.json for the specified run. Contains computed aggregate
 * statistics: stability flag, intervention counts, mean physics values, etc.
 * These are computed server-side by artifacts.compute_metrics() — the client
 * must not recompute them.
 */
export async function getMetrics(
  runId: string,
): Promise<Result<RunMetrics>> {
  return fetchJson<RunMetrics>(
    `/api/runs/${encodeURIComponent(runId)}/metrics`,
    { method: 'GET' },
    ARTIFACT_TIMEOUT_MS,
  );
}

/**
 * GET /api/scenarios
 *
 * Returns metadata for all available scenario YAML files discovered under
 * hydrion/scenarios/examples/. Used to populate the scenario selector.
 */
export async function listScenarios(): Promise<Result<ScenarioInfo[]>> {
  return fetchJson<ScenarioInfo[]>(
    '/api/scenarios',
    { method: 'GET' },
    ARTIFACT_TIMEOUT_MS,
  );
}

/**
 * POST /api/scenarios/run
 *
 * Executes the named scenario end-to-end on the backend and returns the
 * complete ScenarioExecutionHistory.  Execution is synchronous — this call
 * blocks until the simulation completes.
 */
export async function runScenario(
  scenarioId: string,
): Promise<Result<ScenarioExecutionHistory>> {
  return fetchJson<ScenarioExecutionHistory>(
    '/api/scenarios/run',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scenario_id: scenarioId }),
    },
    DEFAULT_TIMEOUT_MS,
  );
}
