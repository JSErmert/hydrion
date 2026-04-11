/**
 * components/TopTelemetryBand.tsx
 *
 * Horizontal strip under the header. Displays three primary instruments:
 * FLOW RATE, PRESSURE, BYPASS — all sourced from HydrosDisplayState.
 */

import type { HydrosDisplayState, BypassLabel } from '../scenarios/displayStateMapper';

const FONT = '"JetBrains Mono", "Fira Code", "Courier New", monospace';

function bypassColor(label: BypassLabel): string {
  if (label === 'ACTIVE') return '#F87171';
  if (label === 'READY') return '#FBBF24';
  return '#64748B';
}

interface TileProps {
  label: string;
  value: string;
  color?: string;
  borderRight?: boolean;
}

function TelemetryTile({ label, value, color = '#E2E8F0', borderRight = true }: TileProps) {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      padding: '0 28px',
      borderRight: borderRight ? '1px solid #1E293B' : 'none',
      minWidth: 140,
    }}>
      <span style={{
        fontFamily: FONT,
        fontSize: 9,
        color: '#475569',
        letterSpacing: '0.14em',
        marginBottom: 4,
      }}>
        {label}
      </span>
      <span style={{
        fontFamily: FONT,
        fontSize: 18,
        color,
        fontWeight: 600,
        letterSpacing: '0.04em',
      }}>
        {value}
      </span>
    </div>
  );
}

export default function TopTelemetryBand({ state }: { state: HydrosDisplayState | null }) {
  const flowStr = state != null ? `${state.q_processed_lmin.toFixed(1)} L/min` : '\u2014';
  const pressStr = state != null ? `${state.pressureKpa.toFixed(1)} kPa` : '\u2014';
  const bypassLabel = state?.bypassLabel ?? 'OFF';

  return (
    <div style={{
      display: 'flex',
      alignItems: 'stretch',
      background: '#080D18',
      borderBottom: '1px solid #1E293B',
      height: 60,
      flexShrink: 0,
    }}>
      <TelemetryTile label="FLOW RATE" value={flowStr} />
      <TelemetryTile label="PRESSURE" value={pressStr} />
      <TelemetryTile
        label="BYPASS"
        value={bypassLabel}
        color={bypassColor(bypassLabel)}
      />

      {/* Spacer */}
      <div style={{ flex: 1 }} />

      {/* Step clock (right side) */}
      {state != null && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          padding: '0 24px',
          borderLeft: '1px solid #1E293B',
        }}>
          <span style={{
            fontFamily: FONT,
            fontSize: 10,
            color: '#475569',
            letterSpacing: '0.06em',
          }}>
            t&nbsp;=&nbsp;{state.t.toFixed(1)}s
            &nbsp;&nbsp;|&nbsp;&nbsp;
            step&nbsp;{state.stepIndex + 1}
          </span>
        </div>
      )}
    </div>
  );
}
