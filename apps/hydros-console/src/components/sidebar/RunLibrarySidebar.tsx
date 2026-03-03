/**
 * RunLibrarySidebar.tsx
 *
 * Sidebar orchestrator. Composes useRunLibrary, RunConfigForm, and RunEntry
 * into the complete left panel of the HydrOS Research Console.
 *
 * Layout (top to bottom):
 *   ┌─────────────────────┐
 *   │  HEADER + REFRESH   │  — "RUN LIBRARY" label, run count, refresh button
 *   ├─────────────────────┤
 *   │  RunConfigForm      │  — launch parameter controls + EXECUTE RUN button
 *   ├─────────────────────┤
 *   │  STATUS BAND        │  — error display (exec or list), executing indicator
 *   ├─────────────────────┤
 *   │  RUN LIST           │  — scrollable list of RunEntry rows
 *   │  (scrollable)       │
 *   └─────────────────────┘
 *
 * GOVERNANCE:
 *   - This component calls useRunLibrary and passes onSelectRun from props
 *     directly into the hook. The parent receives selection via callback.
 *   - No state is held here. All state lives in useRunLibrary.
 *   - Error messages are shown inline in the status band, never in alerts
 *     or modals. They are dismissable.
 *   - Loading states use a skeleton shimmer for the list area.
 *   - The component does not know what happens when a run is selected —
 *     it only emits the run_id.
 *
 * Props:
 *   onSelectRun(runId: string) — called when user selects a run from the list
 *                                or when a new run completes (auto-select)
 *   width?: number             — sidebar width in px (default: 260)
 */

import React, { memo } from 'react';
import { useRunLibrary } from './useRunLibrary';
import { RunConfigForm } from './RunConfigForm';
import { RunEntry } from './RunEntry';
import type { ApiError } from '../../api';

// ---------------------------------------------------------------------------
// Style constants
// ---------------------------------------------------------------------------

const MONO: React.CSSProperties = {
  fontFamily: '"JetBrains Mono", "Fira Code", "Courier New", monospace',
};

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

/** Status band — shows errors and dismissal controls */
function StatusBand({
  error,
  label,
  onDismiss,
}: {
  error: ApiError;
  label: string;
  onDismiss: () => void;
}) {
  return (
    <div style={{
      margin: '0 10px 8px',
      padding: '7px 10px',
      background: 'rgba(239, 68, 68, 0.06)',
      border: '1px solid rgba(239,68,68,0.25)',
      borderRadius: 2,
      display: 'flex',
      flexDirection: 'column',
      gap: 4,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <span style={{
          ...MONO,
          fontSize: 9,
          fontWeight: 700,
          letterSpacing: '0.1em',
          color: '#f87171',
          textTransform: 'uppercase',
        }}>
          {label}
        </span>
        <button
          onClick={onDismiss}
          style={{
            ...MONO,
            background: 'none',
            border: 'none',
            color: '#475569',
            fontSize: 10,
            cursor: 'pointer',
            padding: '0 0 0 8px',
            lineHeight: 1,
          }}
        >
          ✕
        </button>
      </div>
      <span style={{
        ...MONO,
        fontSize: 10,
        color: '#94a3b8',
        lineHeight: 1.4,
        wordBreak: 'break-word',
      }}>
        {error.message}
      </span>
      <span style={{
        ...MONO,
        fontSize: 9,
        color: '#334155',
        letterSpacing: '0.04em',
      }}>
        {error.status !== 0 ? `HTTP ${error.status}` : 'network/timeout'} · {error.endpoint}
      </span>
    </div>
  );
}

/** Skeleton row for loading state */
function SkeletonRow({ opacity }: { opacity: number }) {
  return (
    <div style={{
      padding: '8px 12px',
      borderBottom: '1px solid rgba(255,255,255,0.04)',
      opacity,
    }}>
      <div style={{
        height: 10,
        width: '68%',
        background: 'rgba(71,85,105,0.2)',
        borderRadius: 2,
        marginBottom: 6,
      }} />
      <div style={{
        height: 8,
        width: '45%',
        background: 'rgba(71,85,105,0.12)',
        borderRadius: 2,
      }} />
    </div>
  );
}

/** Empty state when no runs exist yet */
function EmptyState() {
  return (
    <div style={{
      padding: '24px 12px',
      textAlign: 'center',
    }}>
      <div style={{
        ...MONO,
        fontSize: 9,
        color: '#1e293b',
        letterSpacing: '0.12em',
        textTransform: 'uppercase',
        marginBottom: 8,
      }}>
        NO RUNS FOUND
      </div>
      <div style={{
        ...MONO,
        fontSize: 10,
        color: '#334155',
        lineHeight: 1.5,
      }}>
        Execute a run to begin
        <br />
        recording artifacts.
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

interface RunLibrarySidebarProps {
  onSelectRun: (runId: string) => void;
  width?: number;
}

export const RunLibrarySidebar = memo(function RunLibrarySidebar({
  onSelectRun,
  width = 260,
}: RunLibrarySidebarProps) {
  const {
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
    refreshRunList,
    executeRun,
    clearExecError,
    clearListError,
  } = useRunLibrary(onSelectRun);

  return (
    <aside
      aria-label="Run Library"
      style={{
        width,
        minWidth: width,
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        background: 'rgba(8, 14, 26, 0.95)',
        borderRight: '1px solid rgba(30, 41, 59, 0.8)',
        overflow: 'hidden',
        flexShrink: 0,
      }}
    >
      {/* ── HEADER ──────────────────────────────────────────────── */}
      <div style={{
        padding: '12px 12px 10px',
        borderBottom: '1px solid rgba(30,41,59,0.8)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexShrink: 0,
      }}>
        <div>
          <span style={{
            ...MONO,
            fontSize: 9,
            fontWeight: 700,
            letterSpacing: '0.18em',
            color: '#334155',
            textTransform: 'uppercase',
            display: 'block',
          }}>
            RUN LIBRARY
          </span>
          <span style={{
            ...MONO,
            fontSize: 10,
            color: '#1e293b',
            letterSpacing: '0.04em',
          }}>
            {isLoading ? '—' : `${runIds.length} artifact${runIds.length !== 1 ? 's' : ''}`}
          </span>
        </div>

        {/* Refresh button */}
        <button
          onClick={refreshRunList}
          disabled={isLoading || isExecuting}
          title="Refresh run list"
          style={{
            ...MONO,
            background: 'none',
            border: '1px solid rgba(71,85,105,0.3)',
            borderRadius: 2,
            color: isLoading ? '#1e293b' : '#334155',
            fontSize: 10,
            padding: '4px 8px',
            cursor: isLoading || isExecuting ? 'not-allowed' : 'pointer',
            letterSpacing: '0.06em',
            transition: 'color 0.15s, border-color 0.15s',
          }}
          onMouseEnter={e => {
            if (!isLoading && !isExecuting)
              (e.currentTarget as HTMLButtonElement).style.color = '#94a3b8';
          }}
          onMouseLeave={e => {
            (e.currentTarget as HTMLButtonElement).style.color = isLoading ? '#1e293b' : '#334155';
          }}
        >
          ↺
        </button>
      </div>

      {/* ── RUN CONFIG FORM ─────────────────────────────────────── */}
      <div style={{ flexShrink: 0, borderBottom: '1px solid rgba(30,41,59,0.8)' }}>
        <RunConfigForm
          config={runConfig}
          isExecuting={isExecuting}
          onChange={updateRunConfig}
          onExecute={executeRun}
        />
      </div>

      {/* ── STATUS BAND (errors) ─────────────────────────────────── */}
      {execError && (
        <div style={{ flexShrink: 0 }}>
          <StatusBand
            error={execError}
            label="EXEC ERROR"
            onDismiss={clearExecError}
          />
        </div>
      )}
      {listError && (
        <div style={{ flexShrink: 0 }}>
          <StatusBand
            error={listError}
            label="LIST ERROR"
            onDismiss={clearListError}
          />
        </div>
      )}

      {/* ── RUN LIST HEADER ──────────────────────────────────────── */}
      <div style={{
        padding: '8px 12px 6px',
        flexShrink: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        <span style={{
          ...MONO,
          fontSize: 9,
          color: '#1e293b',
          letterSpacing: '0.12em',
          textTransform: 'uppercase',
        }}>
          ARTIFACTS
        </span>
        {selectedRunId && (
          <span style={{
            ...MONO,
            fontSize: 9,
            color: '#38bdf8',
            letterSpacing: '0.06em',
            opacity: 0.7,
          }}>
            1 SELECTED
          </span>
        )}
      </div>

      {/* ── RUN LIST ────────────────────────────────────────────── */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        overflowX: 'hidden',
        // Subtle scrollbar styling
        scrollbarWidth: 'thin',
        scrollbarColor: 'rgba(71,85,105,0.3) transparent',
      }}>
        {isLoading && runIds.length === 0 ? (
          // Skeleton loading state on first fetch
          <>
            {[1, 0.7, 0.45].map((opacity, i) => (
              <SkeletonRow key={i} opacity={opacity} />
            ))}
          </>
        ) : runIds.length === 0 ? (
          <EmptyState />
        ) : (
          // Render newest-first: reverse a copy so original order is preserved
          [...runIds].reverse().map(runId => (
            <RunEntry
              key={runId}
              runId={runId}
              isSelected={selectedRunId === runId}
              isNew={runId === lastExecutedRunId}
              onSelect={selectRun}
            />
          ))
        )}
      </div>

      {/* ── FOOTER ──────────────────────────────────────────────── */}
      <div style={{
        padding: '6px 12px',
        borderTop: '1px solid rgba(30,41,59,0.6)',
        flexShrink: 0,
      }}>
        <span style={{
          ...MONO,
          fontSize: 9,
          color: '#1e293b',
          letterSpacing: '0.06em',
        }}>
          {isExecuting
            ? 'EXECUTING — AWAITING ARTIFACT…'
            : `ARTIFACT SCHEMA: run_artifact_v1`}
        </span>
      </div>
    </aside>
  );
});
