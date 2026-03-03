/**
 * RunEntry.tsx
 *
 * A single row in the run library list.
 *
 * Renders the run_id parsed into its semantic components:
 *   run_<unix_timestamp>_<seed>  →  timestamp formatted as UTC datetime + seed label
 *
 * Selection state is reflected visually via CSS class. No internal state.
 * This component is purely presentational — all interaction flows upward.
 *
 * Design doctrine: Industrial mission-control. Monospace. Dense but legible.
 * No icons that don't carry information. No decorative elements.
 */

import React, { memo } from 'react';

// ---------------------------------------------------------------------------
// Run ID parsing
// ---------------------------------------------------------------------------

interface ParsedRunId {
  timestamp: string; // formatted UTC datetime e.g. "2025-03-01 14:22:07"
  seed: string;
  raw: string;
}

function parseRunId(runId: string): ParsedRunId {
  // Expected format: run_<unix_timestamp>_<seed>
  const match = runId.match(/^run_(\d+)_(\d+)$/);
  if (!match) {
    return { timestamp: '—', seed: '—', raw: runId };
  }

  const unixSec = parseInt(match[1], 10);
  const seed = match[2];
  const date = new Date(unixSec * 1000);

  const timestamp = date.toISOString().replace('T', ' ').slice(0, 19);

  return { timestamp, seed, raw: runId };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface RunEntryProps {
  runId: string;
  isSelected: boolean;
  isNew: boolean; // true if this is the most recently executed run
  onSelect: (runId: string) => void;
}

export const RunEntry = memo(function RunEntry({
  runId,
  isSelected,
  isNew,
  onSelect,
}: RunEntryProps) {
  const parsed = parseRunId(runId);

  return (
    <button
      onClick={() => onSelect(runId)}
      title={parsed.raw}
      style={{
        display: 'flex',
        flexDirection: 'column',
        width: '100%',
        padding: '8px 12px',
        background: isSelected
          ? 'rgba(56, 189, 248, 0.08)'
          : 'transparent',
        border: 'none',
        borderLeft: isSelected
          ? '2px solid #38bdf8'
          : '2px solid transparent',
        borderBottom: '1px solid rgba(255,255,255,0.04)',
        cursor: 'pointer',
        textAlign: 'left',
        transition: 'background 0.1s ease, border-color 0.1s ease',
      }}
      onMouseEnter={e => {
        if (!isSelected) {
          (e.currentTarget as HTMLButtonElement).style.background =
            'rgba(255,255,255,0.03)';
        }
      }}
      onMouseLeave={e => {
        if (!isSelected) {
          (e.currentTarget as HTMLButtonElement).style.background = 'transparent';
        }
      }}
    >
      {/* Row 1: timestamp + NEW badge */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 3,
      }}>
        <span style={{
          fontFamily: '"JetBrains Mono", "Fira Code", "Courier New", monospace',
          fontSize: 11,
          color: isSelected ? '#e0f2fe' : '#94a3b8',
          letterSpacing: '0.02em',
        }}>
          {parsed.timestamp}
        </span>
        {isNew && (
          <span style={{
            fontFamily: '"JetBrains Mono", "Fira Code", monospace',
            fontSize: 9,
            fontWeight: 700,
            color: '#38bdf8',
            letterSpacing: '0.1em',
            padding: '1px 5px',
            border: '1px solid rgba(56,189,248,0.4)',
            borderRadius: 2,
          }}>
            NEW
          </span>
        )}
      </div>

      {/* Row 2: run id fragment + seed */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      }}>
        <span style={{
          fontFamily: '"JetBrains Mono", "Fira Code", monospace',
          fontSize: 10,
          color: isSelected ? '#7dd3fc' : '#475569',
          letterSpacing: '0.04em',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
          maxWidth: '75%',
        }}>
          {/* Show abbreviated run ID so it doesn't overflow */}
          {parsed.raw.length > 28 ? `…${parsed.raw.slice(-24)}` : parsed.raw}
        </span>
        <span style={{
          fontFamily: '"JetBrains Mono", "Fira Code", monospace',
          fontSize: 9,
          color: '#334155',
          letterSpacing: '0.06em',
        }}>
          seed:{parsed.seed}
        </span>
      </div>
    </button>
  );
});
