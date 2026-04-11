/**
 * components/BottomNarrativeBand.tsx
 *
 * Horizontal strip below the machine core.
 * Displays: RUN CYCLE timer | per-stage fouling | FIBER LOAD | PULSE | NEXT ACTION
 */

import type { HydrosDisplayState, FiberLoadLabel, PulseLabel } from '../scenarios/displayStateMapper';

const FONT = '"JetBrains Mono", "Fira Code", "Courier New", monospace';

function fiberLoadColor(label: FiberLoadLabel | undefined): string {
  switch (label) {
    case 'LOW':      return '#34D399';
    case 'MODERATE': return '#FBBF24';
    case 'HIGH':     return '#FB923C';
    case 'CRITICAL': return '#F87171';
    default:         return '#475569';
  }
}

function pulseColor(label: PulseLabel | undefined): string {
  switch (label) {
    case 'ACTIVE':   return '#38BDF8';
    case 'COOLDOWN': return '#7DD3FC';
    default:         return '#475569';
  }
}

function foulingColor(frac: number): string {
  if (frac < 0.25) return '#34D399';
  if (frac < 0.50) return '#FBBF24';
  if (frac < 0.75) return '#FB923C';
  return '#F87171';
}

interface NarrativeTileProps {
  label: string;
  value: string;
  color?: string;
  grow?: boolean;
}

function NarrativeTile({ label, value, color = '#E2E8F0', grow = false }: NarrativeTileProps) {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      padding: '0 20px',
      borderRight: '1px solid #1E293B',
      flexShrink: grow ? 0 : 1,
      flexGrow: grow ? 1 : 0,
      minWidth: grow ? 140 : 80,
    }}>
      <span style={{
        fontFamily: FONT,
        fontSize: 8,
        color: '#475569',
        letterSpacing: '0.14em',
        marginBottom: 4,
      }}>
        {label}
      </span>
      <span style={{
        fontFamily: FONT,
        fontSize: 13,
        color,
        fontWeight: 600,
        letterSpacing: '0.04em',
        whiteSpace: 'nowrap',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
      }}>
        {value}
      </span>
    </div>
  );
}

export default function BottomNarrativeBand({ state }: { state: HydrosDisplayState | null }) {
  const na = '\u2014';

  const cycleVal = state != null ? `${state.t.toFixed(1)}s` : na;
  const s1Val    = state != null ? `${Math.round(state.foulingS1 * 100)}% FOULING` : na;
  const s2Val    = state != null ? `${Math.round(state.foulingS2 * 100)}% FOULING` : na;
  const s3Val    = state != null ? `${Math.round(state.foulingS3 * 100)}% FOULING` : na;
  const fiberVal = state?.fiberLoadLabel ?? na;
  const pulseVal = state?.pulseLabel ?? na;
  const nextVal  = state != null ? (state.nextActions[0] ?? na) : na;

  return (
    <div style={{
      display: 'flex',
      alignItems: 'stretch',
      background: '#080D18',
      borderTop: '1px solid #1E293B',
      height: 72,
      flexShrink: 0,
    }}>
      <NarrativeTile
        label="RUN CYCLE"
        value={cycleVal}
        color="#94A3B8"
      />
      <NarrativeTile
        label="S1"
        value={s1Val}
        color={state ? foulingColor(state.foulingS1) : '#475569'}
      />
      <NarrativeTile
        label="S2"
        value={s2Val}
        color={state ? foulingColor(state.foulingS2) : '#475569'}
      />
      <NarrativeTile
        label="S3"
        value={s3Val}
        color={state ? foulingColor(state.foulingS3) : '#475569'}
      />
      <NarrativeTile
        label="FIBER LOAD"
        value={fiberVal}
        color={fiberLoadColor(state?.fiberLoadLabel)}
      />
      <NarrativeTile
        label="PULSE"
        value={pulseVal}
        color={pulseColor(state?.pulseLabel)}
      />
      <NarrativeTile
        label="NEXT ACTION"
        value={nextVal}
        color="#94A3B8"
        grow
      />
    </div>
  );
}
