/**
 * components/PlaybackBar.tsx
 *
 * Scenario selector, transport controls, and event marker navigation.
 * Receives all state and callbacks from App — no internal state.
 */

import React from 'react';
import type { ScenarioInfo, ScenarioEventMarker } from '../api/types';

export interface PlaybackBarProps {
  // Scenario selection
  scenarios: ScenarioInfo[];
  selectedId: string;
  onSelectScenario: (id: string) => void;
  onRun: () => void;
  isRunning: boolean;
  loadError: string | null;

  // Playback state
  hasHistory: boolean;
  stepIndex: number;
  totalSteps: number;
  currentTime: number;
  scenarioDuration: number;
  isPlaying: boolean;
  speedMultiplier: number;
  eventMarkers: ScenarioEventMarker[];

  // Controls
  onPlay: () => void;
  onPause: () => void;
  onNextStep: () => void;
  onPrevStep: () => void;
  onJumpToStep: (idx: number) => void;
  onJumpToMarker: (marker: ScenarioEventMarker) => void;
  onSetSpeed: (n: number) => void;
}

const SPEEDS = [1, 5, 10, 30, 100];

// Marker types surfaced as jump buttons (excludes bookend types)
const NAVIGABLE_MARKER_TYPES = new Set([
  'threshold_crossing',
  'backflush_start',
  'backflush_end',
  'bypass_start',
  'bypass_end',
  'disturbance_start',
  'disturbance_end',
]);

const MARKER_LABEL: Record<string, string> = {
  threshold_crossing: 'MAINT',
  backflush_start:    'BF+',
  backflush_end:      'BF-',
  bypass_start:       'BYP+',
  bypass_end:         'BYP-',
  disturbance_start:  'DIST+',
  disturbance_end:    'DIST-',
};

const mono = '"JetBrains Mono", "Fira Code", "Courier New", monospace';

const base: React.CSSProperties = {
  fontFamily: mono,
  fontSize: 12,
  color: 'var(--text-secondary)',
};

const btn: React.CSSProperties = {
  fontFamily: mono,
  fontSize: 12,
  background: 'var(--bg-panel)',
  border: '1px solid var(--border-subtle)',
  color: 'var(--text-primary)',
  padding: '4px 10px',
  cursor: 'pointer',
  borderRadius: 4,
};

const sel: React.CSSProperties = {
  fontFamily: mono,
  fontSize: 12,
  background: 'var(--bg-panel)',
  border: '1px solid var(--border-subtle)',
  color: 'var(--text-primary)',
  padding: '4px 8px',
  borderRadius: 4,
};

export default function PlaybackBar({
  scenarios, selectedId, onSelectScenario, onRun, isRunning, loadError,
  hasHistory, stepIndex, totalSteps, currentTime, scenarioDuration,
  isPlaying, speedMultiplier, eventMarkers,
  onPlay, onPause, onNextStep, onPrevStep, onJumpToStep, onJumpToMarker, onSetSpeed,
}: PlaybackBarProps) {
  const navigable = eventMarkers.filter(m => NAVIGABLE_MARKER_TYPES.has(m.type));

  return (
    <div style={{ padding: '12px 16px', display: 'flex', flexDirection: 'column', gap: 8 }}>

      {/* Row 1 — scenario selector + run trigger */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <span style={base}>SCENARIO</span>
        <select
          style={sel}
          value={selectedId}
          onChange={e => onSelectScenario(e.target.value)}
          disabled={isRunning}
        >
          {scenarios.length === 0 ? (
            <option value={selectedId}>{selectedId}</option>
          ) : (
            scenarios.map(s => <option key={s.id} value={s.id}>{s.name}</option>)
          )}
        </select>
        <button style={btn} onClick={onRun} disabled={isRunning}>
          {isRunning ? 'RUNNING...' : 'RUN'}
        </button>
        {loadError && (
          <span style={{ ...base, color: '#f87171', fontSize: 11 }}>{loadError}</span>
        )}
      </div>

      {/* Row 2 — transport controls (visible only after history is loaded) */}
      {hasHistory && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          <button style={btn} onClick={onPrevStep} disabled={stepIndex === 0}>◀</button>
          <button style={btn} onClick={isPlaying ? onPause : onPlay}>
            {isPlaying ? '⏸' : '▶'}
          </button>
          <button style={btn} onClick={onNextStep} disabled={stepIndex >= totalSteps - 1}>▶▶</button>
          <input
            type="range"
            min={0}
            max={totalSteps - 1}
            value={stepIndex}
            onChange={e => onJumpToStep(Number(e.target.value))}
            style={{ flex: 1, minWidth: 100, accentColor: 'var(--accent-electric)' }}
          />
          <select
            style={sel}
            value={speedMultiplier}
            onChange={e => onSetSpeed(Number(e.target.value))}
          >
            {SPEEDS.map(s => <option key={s} value={s}>{s}x</option>)}
          </select>
          <span style={base}>
            t={currentTime.toFixed(1)}s&nbsp;/&nbsp;{scenarioDuration.toFixed(1)}s
            &nbsp;&nbsp;
            {stepIndex + 1}/{totalSteps}
          </span>
        </div>
      )}

      {/* Row 3 — event marker navigation (visible when markers exist) */}
      {navigable.length > 0 && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
          <span style={base}>JUMP</span>
          {navigable.map((m, i) => (
            <button
              key={i}
              style={{ ...btn, fontSize: 11, padding: '3px 8px' }}
              onClick={() => onJumpToMarker(m)}
            >
              {MARKER_LABEL[m.type] ?? m.type} {m.time.toFixed(1)}s
            </button>
          ))}
        </div>
      )}

    </div>
  );
}
