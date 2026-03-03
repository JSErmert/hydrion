/**
 * api/config.ts
 *
 * Single source of truth for backend origin resolution.
 *
 * Resolution priority:
 *   1. VITE_API_BASE_URL environment variable (set in .env.local for dev,
 *      injected at build time for production)
 *   2. Empty string fallback — assumes the React app is served from the same
 *      origin as the FastAPI backend (standard production deployment pattern)
 *
 * Never hardcode localhost here. All environment-specific values live in
 * .env.local (gitignored) or in the deployment environment.
 *
 * Usage:
 *   import { API_BASE_URL } from './config';
 *   fetch(`${API_BASE_URL}/api/runs`)
 */
export const API_BASE_URL: string =
  (import.meta as any).env?.VITE_API_BASE_URL ?? '';

/**
 * Default fetch timeout in milliseconds.
 * POST /api/run executes a full simulation synchronously — allow sufficient
 * time for long runs. Individual GET requests are fast (artifact reads).
 */
export const DEFAULT_TIMEOUT_MS = 60_000;  // 60 s covers most simulation runs
export const ARTIFACT_TIMEOUT_MS = 10_000; // 10 s for read-only artifact fetches
