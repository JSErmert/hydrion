/**
 * RunConfigForm.tsx
 *
 * Compact launch configuration panel at the top of the sidebar.
 *
 * Renders controls for the fields in RunConfig: policy_type, seed,
 * config_name, max_steps, noise_enabled. The "EXECUTE RUN" trigger button
 * is the primary action. It is disabled while a run is in flight.
 *
 * GOVERNANCE:
 *   - This component holds no state of its own. All values flow in as props.
 *   - The execute button must be visually locked during isExecuting — this
 *     prevents double-fire at the UI layer as a second defense behind the
 *     hook's isExecuting guard.
 *   - No validation logic here — the backend validates config_name existence
 *     and returns an HTTP 400 if invalid. The sidebar surfaces that as execError.
 *
 * Design: Industrial control panel aesthetic. Input fields styled as
 * hardware parameter readouts. Execute button is high-contrast and unambiguous.
 */

import React, { memo } from 'react';
import type { RunConfig } from './useRunLibrary';

// ---------------------------------------------------------------------------
// Shared style tokens — keep consistent with sidebar chrome
// ---------------------------------------------------------------------------

const MONO: React.CSSProperties = {
  fontFamily: '"JetBrains Mono", "Fira Code", "Courier New", monospace',
};

const LABEL_STYLE: React.CSSProperties = {
  ...MONO,
  fontSize: 9,
  fontWeight: 600,
  letterSpacing: '0.12em',
  color: '#475569',
  textTransform: 'uppercase' as const,
  marginBottom: 4,
  display: 'block',
};

const INPUT_STYLE: React.CSSProperties = {
  ...MONO,
  width: '100%',
  padding: '5px 8px',
  background: 'rgba(15, 23, 42, 0.8)',
  border: '1px solid rgba(71, 85, 105, 0.5)',
  borderRadius: 2,
  color: '#cbd5e1',
  fontSize: 11,
  letterSpacing: '0.03em',
  outline: 'none',
  boxSizing: 'border-box' as const,
};

const SELECT_STYLE: React.CSSProperties = {
  ...INPUT_STYLE,
  cursor: 'pointer',
  appearance: 'none' as const,
  WebkitAppearance: 'none' as const,
};

const FIELD_WRAP: React.CSSProperties = {
  marginBottom: 10,
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface RunConfigFormProps {
  config: RunConfig;
  isExecuting: boolean;
  onChange: (patch: Partial<RunConfig>) => void;
  onExecute: () => void;
}

export const RunConfigForm = memo(function RunConfigForm({
  config,
  isExecuting,
  onChange,
  onExecute,
}: RunConfigFormProps) {
  return (
    <div style={{ padding: '12px 12px 8px' }}>

      {/* Section header */}
      <div style={{
        ...MONO,
        fontSize: 9,
        fontWeight: 700,
        letterSpacing: '0.16em',
        color: '#334155',
        marginBottom: 12,
        paddingBottom: 6,
        borderBottom: '1px solid rgba(71,85,105,0.25)',
        textTransform: 'uppercase',
      }}>
        RUN CONFIGURATION
      </div>

      {/* Policy type */}
      <div style={FIELD_WRAP}>
        <label style={LABEL_STYLE}>Policy</label>
        <select
          style={SELECT_STYLE}
          value={config.policy_type}
          onChange={e => onChange({ policy_type: e.target.value as RunConfig['policy_type'] })}
          disabled={isExecuting}
        >
          <option value="random">random</option>
          <option value="ppo">ppo</option>
          <option value="baseline">baseline</option>
        </select>
      </div>

      {/* Config file */}
      <div style={FIELD_WRAP}>
        <label style={LABEL_STYLE}>Config</label>
        <input
          type="text"
          style={INPUT_STYLE}
          value={config.config_name}
          onChange={e => onChange({ config_name: e.target.value })}
          disabled={isExecuting}
          placeholder="default.yaml"
          spellCheck={false}
        />
      </div>

      {/* Seed + max_steps on one row */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 10 }}>
        <div>
          <label style={LABEL_STYLE}>Seed</label>
          <input
            type="number"
            style={INPUT_STYLE}
            value={config.seed}
            min={0}
            max={999999}
            onChange={e => onChange({ seed: parseInt(e.target.value, 10) || 0 })}
            disabled={isExecuting}
          />
        </div>
        <div>
          <label style={LABEL_STYLE}>Max Steps</label>
          <input
            type="number"
            style={INPUT_STYLE}
            value={config.max_steps}
            min={1}
            max={5000}
            onChange={e => onChange({ max_steps: parseInt(e.target.value, 10) || 1 })}
            disabled={isExecuting}
          />
        </div>
      </div>

      {/* Noise toggle */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: 14,
      }}>
        <span style={{ ...MONO, fontSize: 10, color: '#475569', letterSpacing: '0.06em' }}>
          Sensor Noise
        </span>
        <button
          onClick={() => onChange({ noise_enabled: !config.noise_enabled })}
          disabled={isExecuting}
          style={{
            ...MONO,
            fontSize: 9,
            fontWeight: 700,
            letterSpacing: '0.1em',
            padding: '3px 10px',
            border: `1px solid ${config.noise_enabled ? 'rgba(56,189,248,0.5)' : 'rgba(71,85,105,0.4)'}`,
            borderRadius: 2,
            background: config.noise_enabled
              ? 'rgba(56,189,248,0.08)'
              : 'transparent',
            color: config.noise_enabled ? '#38bdf8' : '#475569',
            cursor: isExecuting ? 'not-allowed' : 'pointer',
            transition: 'all 0.15s ease',
          }}
        >
          {config.noise_enabled ? 'ON' : 'OFF'}
        </button>
      </div>

      {/* Execute button */}
      <button
        onClick={onExecute}
        disabled={isExecuting}
        style={{
          ...MONO,
          width: '100%',
          padding: '9px 0',
          fontSize: 11,
          fontWeight: 700,
          letterSpacing: '0.14em',
          textTransform: 'uppercase',
          cursor: isExecuting ? 'not-allowed' : 'pointer',
          border: `1px solid ${isExecuting ? 'rgba(71,85,105,0.3)' : 'rgba(56,189,248,0.6)'}`,
          borderRadius: 2,
          background: isExecuting
            ? 'rgba(15,23,42,0.5)'
            : 'rgba(56,189,248,0.07)',
          color: isExecuting ? '#334155' : '#38bdf8',
          transition: 'all 0.15s ease',
          position: 'relative',
        }}
      >
        {isExecuting ? (
          <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
            <ExecutingSpinner />
            EXECUTING…
          </span>
        ) : (
          'EXECUTE RUN'
        )}
      </button>
    </div>
  );
});

// ---------------------------------------------------------------------------
// ExecutingSpinner — minimal CSS animation, no library dependency
// ---------------------------------------------------------------------------

function ExecutingSpinner() {
  return (
    <span
      style={{
        display: 'inline-block',
        width: 8,
        height: 8,
        border: '1px solid rgba(71,85,105,0.6)',
        borderTopColor: '#38bdf8',
        borderRadius: '50%',
        animation: 'hydros-spin 0.7s linear infinite',
      }}
    />
  );
}

// Inject the keyframe once. This is safe in a module context.
if (typeof document !== 'undefined') {
  const styleId = 'hydros-spinner-keyframes';
  if (!document.getElementById(styleId)) {
    const style = document.createElement('style');
    style.id = styleId;
    style.textContent = `@keyframes hydros-spin { to { transform: rotate(360deg); } }`;
    document.head.appendChild(style);
  }
}
