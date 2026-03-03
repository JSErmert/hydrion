/**
 * useRunLibrary.ts
 *
 * Single source of state for the Run Library Sidebar.
 *
 * GOVERNANCE:
 *   - This hook owns all sidebar state. No sidebar child component holds
 *     its own loading or error state. State flows down as props.
 *   - startRun() is the only user-initiated write operation. It is
 *     debounced against double-fire via the `isExecuting` flag.
 *   - listRuns() is called on mount and after every successful startRun().
 *     It is never called on a timer. Refresh is always explicit or
 *     consequence of a completed run.
 *   - selectedRunId propagates upward through onSelectRun callback.
 *     This hook does not know what the parent does with the selection —
 *     it only manages the identity of the selected run.
 *   - All errors are structured ApiError objects. The sidebar renders them
 *     in a status band. Errors do not crash the component tree.
 *
 * State shape:
 *   runIds          — ordered list of run identifiers from GET /api/runs
 *   selectedRunId   — currently selected run, or null
 *   isLoading       — true while GET /api/runs is in flight
 *   isExecuting     — true while POST /api/run is in flight
 *   listError       — last error from listRuns(), or null
 *   execError       — last error from startRun(), or null
 *   execRunId       — run_id returned by the most recent successful startRun()
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { listRuns, startRun } from '../../api';
import type { RunRequest, ApiError } from '../../api';

// ---------------------------------------------------------------------------
// RunConfig — the minimal form state needed to build a RunRequest.
// Kept separate from RunRequest so the sidebar can hold partial/draft state
// before the user commits to launching.
// ---------------------------------------------------------------------------

export interface RunConfig {
  policy_type: 'random' | 'ppo' | 'baseline';
  seed: number;
  config_name: string;
  max_steps: number;
  noise_enabled: boolean;
}

export const DEFAULT_RUN_CONFIG: RunConfig = {
  policy_type: 'random',
  seed: 42,
  config_name: 'default.yaml',
  max_steps: 200,
  noise_enabled: true,
};

// ---------------------------------------------------------------------------
// Hook interface
// ---------------------------------------------------------------------------

export interface UseRunLibraryReturn {
  /** Ordered list of run IDs from the backend */
  runIds: string[];
  /** Currently selected run ID, or null if no selection */
  selectedRunId: string | null;
  /** True while GET /api/runs is in flight */
  isLoading: boolean;
  /** True while POST /api/run is in flight — blocks re-trigger */
  isExecuting: boolean;
  /** Error from the most recent listRuns() call, or null */
  listError: ApiError | null;
  /** Error from the most recent startRun() call, or null */
  execError: ApiError | null;
  /** run_id returned by the last successful run execution, or null */
  lastExecutedRunId: string | null;
  /** Draft run configuration — controlled by the run config form */
  runConfig: RunConfig;
  /** Update one or more fields of the run config */
  updateRunConfig: (patch: Partial<RunConfig>) => void;
  /** Select a run by ID. Clears execError. Propagates to parent via onSelectRun. */
  selectRun: (runId: string) => void;
  /** Manually refresh the run list */
  refreshRunList: () => void;
  /** Trigger a new simulation run using the current runConfig */
  executeRun: () => void;
  /** Dismiss the current execError */
  clearExecError: () => void;
  /** Dismiss the current listError */
  clearListError: () => void;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useRunLibrary(
  onSelectRun: (runId: string) => void,
): UseRunLibraryReturn {
  const [runIds, setRunIds] = useState<string[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isExecuting, setIsExecuting] = useState(false);
  const [listError, setListError] = useState<ApiError | null>(null);
  const [execError, setExecError] = useState<ApiError | null>(null);
  const [lastExecutedRunId, setLastExecutedRunId] = useState<string | null>(null);
  const [runConfig, setRunConfig] = useState<RunConfig>(DEFAULT_RUN_CONFIG);

  // Guard against calling setState on an unmounted component
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  // ---------------------------------------------------------------------------
  // fetchRunList — internal, called on mount and after successful execution
  // ---------------------------------------------------------------------------

  const fetchRunList = useCallback(async () => {
    if (!mountedRef.current) return;
    setIsLoading(true);
    setListError(null);

    const result = await listRuns();

    if (!mountedRef.current) return;
    setIsLoading(false);

    if (!result.ok) {
      setListError(result.error);
      return;
    }

    setRunIds(result.data);
  }, []);

  // Fetch on mount
  useEffect(() => {
    fetchRunList();
  }, [fetchRunList]);

  // ---------------------------------------------------------------------------
  // selectRun — updates local selection and propagates to parent
  // ---------------------------------------------------------------------------

  const selectRun = useCallback((runId: string) => {
    setSelectedRunId(runId);
    setExecError(null); // a new selection clears any prior exec error
    onSelectRun(runId);
  }, [onSelectRun]);

  // ---------------------------------------------------------------------------
  // updateRunConfig — patch the draft config
  // ---------------------------------------------------------------------------

  const updateRunConfig = useCallback((patch: Partial<RunConfig>) => {
    setRunConfig(prev => ({ ...prev, ...patch }));
  }, []);

  // ---------------------------------------------------------------------------
  // executeRun — POST /api/run, then refresh list and auto-select new run
  // ---------------------------------------------------------------------------

  const executeRun = useCallback(async () => {
    // Hard guard: never fire if already executing
    if (isExecuting) return;

    setIsExecuting(true);
    setExecError(null);

    const req: RunRequest = { ...runConfig };
    const result = await startRun(req);

    if (!mountedRef.current) return;
    setIsExecuting(false);

    if (!result.ok) {
      setExecError(result.error);
      return;
    }

    const newRunId = result.data.run_id;
    setLastExecutedRunId(newRunId);

    // Refresh the run list — the new run is now on disk
    await fetchRunList();

    // Auto-select the newly created run so the parent ConsoleView
    // can begin loading it immediately.
    if (mountedRef.current) {
      selectRun(newRunId);
    }
  }, [isExecuting, runConfig, fetchRunList, selectRun]);

  // ---------------------------------------------------------------------------
  // Error dismissal
  // ---------------------------------------------------------------------------

  const clearExecError = useCallback(() => setExecError(null), []);
  const clearListError = useCallback(() => setListError(null), []);

  return {
    runIds,
    selectedRunId,
    isLoading,
    isExecuting,
    listError,
    execError,
    lastExecutedRunId,
    runConfig,
    updateRunConfig,
    selectRun,
    refreshRunList: fetchRunList,
    executeRun,
    clearExecError,
    clearListError,
  };
}
