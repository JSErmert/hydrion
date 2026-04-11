/**
 * components/LeftMetricsPanel.tsx
 *
 * Left column metrics matching the reference image layout:
 *   Row 1: [bars] FLOW RATE 13.5 L/min  |  PRESSURE DROP
 *   Row 2: [bars] PRESSURE 45.2 kPa     |  ⊙ BYPASS
 *   Then:  RUN CYCLE timer
 */

import type { HydrosDisplayState } from '../scenarios/displayStateMapper';

const FONT = '"JetBrains Mono", "Fira Code", "Courier New", monospace';

// LED bar-graph indicator (horizontal bars, increasing height)
function LedBars({ value, color }: { value: number; color: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'flex-end', gap: 2, flexShrink: 0 }}>
      {[0.18, 0.38, 0.58, 0.78, 0.95].map((t, i) => (
        <div key={i} style={{
          width: 3,
          height: 4 + i * 2.2,
          background: value >= t ? color : '#1A2840',
          borderRadius: 1,
        }} />
      ))}
    </div>
  );
}

interface MetricCellProps {
  label: string;
  value: string;
  barValue?: number;
  barColor?: string;
  valueColor?: string;
}

function MetricCell({ label, value, barValue, barColor = '#38BDF8', valueColor = '#CBD5E1' }: MetricCellProps) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
        {barValue !== undefined && <LedBars value={barValue} color={barColor} />}
        <span style={{
          fontFamily: FONT,
          fontSize: 8,
          color: '#3A5570',
          letterSpacing: '0.14em',
        }}>
          {label}
        </span>
      </div>
      <div style={{
        fontFamily: FONT,
        fontSize: 14,
        color: valueColor,
        fontWeight: 600,
        letterSpacing: '0.04em',
        paddingLeft: barValue !== undefined ? 24 : 0,
      }}>
        {value}
      </div>
    </div>
  );
}

// Bypass status with dot indicator
function BypassCell({ label }: { label: string }) {
  const active = label === 'ACTIVE';
  const ready  = label === 'READY';
  const color  = active ? '#F87171' : ready ? '#FBBF24' : '#34D399';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <span style={{
        fontFamily: FONT,
        fontSize: 8,
        color: '#3A5570',
        letterSpacing: '0.14em',
      }}>
        BYPASS
      </span>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <div style={{
          width: 7,
          height: 7,
          borderRadius: '50%',
          background: color,
          boxShadow: `0 0 6px ${color}`,
          flexShrink: 0,
        }} />
        <span style={{
          fontFamily: FONT,
          fontSize: 12,
          color,
          fontWeight: 600,
          letterSpacing: '0.08em',
        }}>
          {label}
        </span>
      </div>
    </div>
  );
}

export default function LeftMetricsPanel({ state }: { state: HydrosDisplayState | null }) {
  const na  = '\u2014';
  const flow   = state ? `${state.q_processed_lmin.toFixed(1)} L/min` : na;
  const press  = state ? `${state.pressureKpa.toFixed(1)} kPa` : na;
  const dp     = state ? `${(state.pressureKpa * 0.42).toFixed(1)} kPa` : na;
  const flowN  = state ? Math.min(1, state.q_processed_lmin / 20) : 0;
  const pressN = state ? state.pressure : 0;

  return (
    <div style={{
      width: 190,
      flexShrink: 0,
      background: '#060A14',
      borderRight: '1px solid #1E293B',
      padding: '16px 14px',
      display: 'flex',
      flexDirection: 'column',
      gap: 0,
      overflowY: 'auto',
    }}>

      {/* ── 2×2 metric grid ─────────────────────────────────── */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gridTemplateRows: 'auto auto',
        gap: '16px 10px',
        marginBottom: 18,
      }}>
        <MetricCell
          label="FLOW RATE"
          value={flow}
          barValue={flowN}
          barColor="#38BDF8"
        />
        <MetricCell
          label="PRESSURE DROP"
          value={dp}
          barValue={pressN * 0.55}
          barColor="#7DD3FC"
        />
        <MetricCell
          label="PRESSURE"
          value={press}
          barValue={pressN}
          barColor={pressN > 0.8 ? '#FB923C' : '#38BDF8'}
        />
        <BypassCell label={state?.bypassLabel ?? 'OFF'} />
      </div>

      {/* Divider */}
      <div style={{ height: 1, background: '#1A2840', marginBottom: 16 }} />

      {/* ── RUN CYCLE ───────────────────────────────────────── */}
      <div>
        <div style={{
          fontFamily: FONT,
          fontSize: 8,
          color: '#3A5570',
          letterSpacing: '0.14em',
          marginBottom: 4,
        }}>
          RUN CYCLE
        </div>
        <div style={{
          fontFamily: FONT,
          fontSize: 18,
          color: state ? '#7FB8D8' : '#1A2E48',
          fontWeight: 600,
          letterSpacing: '0.08em',
        }}>
          {state ? formatTime(state.t) : '00:00:00'}
        </div>
      </div>

    </div>
  );
}

function formatTime(t: number): string {
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const s = Math.floor(t % 60);
  return `${pad(h)}:${pad(m)}:${pad(s)}`;
}
function pad(n: number): string { return String(n).padStart(2, '0'); }
