/**
 * api/index.ts
 *
 * Barrel export for the HydrOS API client module.
 *
 * All console components import from 'api/' — never from individual files.
 * This enforces a single controlled surface and allows internal file
 * reorganization without touching consumers.
 *
 * Usage:
 *   import { startRun, listRuns, getSpine } from '../api';
 *   import type { SpineStep, RunManifest, RunMetrics } from '../api';
 */

// Configuration constants (exported for use in diagnostics/debug panels)
export { API_BASE_URL, DEFAULT_TIMEOUT_MS, ARTIFACT_TIMEOUT_MS } from './config';

// Types — all artifact shapes and the Result<T> contract
export type {
  RunRequest,
  RunManifest,
  SpineTruth,
  SpineSensors,
  SpineActions,
  SpineSafety,
  SpineEvents,
  SpineStep,
  RunMetrics,
  ApiError,
  Result,
} from './types';

// Client functions — one per backend route
export {
  startRun,
  listRuns,
  getManifest,
  getSpine,
  getMetrics,
} from './client';
