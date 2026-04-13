/**
 * components/BottomInstruments.tsx
 *
 * Bottom instrument row matching the reference image:
 *   [CALIBRATION TIERS] | [FIBER LOAD 90px gauge] | [PULSE 01 90px gauge + arrows] | [NEXT ACTION]
 */

import CircularGauge from './CircularGauge';
import type { HydrosDisplayState } from '../scenarios/displayStateMapper';

const FONT = '"JetBrains Mono", "Fira Code", "Courier New", monospace';

function SectionLabel({ children }: { children: string }) {
  return (
    <div style={{
      fontFamily: FONT,
      fontSize: 8,
      color: '#334155',
      letterSpacing: '0.16em',
      marginBottom: 8,
    }}>
      {children}
    </div>
  );
}

function ColDivider() {
  return (
    <div style={{ width: 1, background: '#1A2840', flexShrink: 0, alignSelf: 'stretch' }} />
  );
}

// ── CALIBRATION TIERS ─────────────────────────────────────────────────────

function CalibrationTiers() {
  const tiers = [
    { label: 'GROUNDED',    color: '#34D399', active: true  },
    { label: 'PROXY',       color: '#FB923C', active: false },
    { label: 'PLACEHOLDER', color: '#FB923C', active: false },
  ];
  return (
    <div style={{
      padding: '12px 16px',
      minWidth: 148,
      flexShrink: 0,
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
    }}>
      <SectionLabel>CALIBRATION TIERS</SectionLabel>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
        {tiers.map(t => (
          <div key={t.label} style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
            <svg width={10} height={10} style={{ flexShrink: 0 }}>
              <path d="M0,0 L10,5 L0,10 Z"
                fill={t.active ? t.color : '#1A2840'}
                stroke={t.active ? undefined : '#283A52'}
                strokeWidth={t.active ? 0 : 1}
              />
            </svg>
            <span style={{
              fontFamily: FONT,
              fontSize: 9,
              color: t.active ? t.color : '#283A52',
              letterSpacing: '0.06em',
            }}>
              {t.label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── FIBER LOAD GAUGE ──────────────────────────────────────────────────────

function fiberColor(label: string): string {
  switch (label) {
    case 'LOW':      return '#34D399';
    case 'MODERATE': return '#FBBF24';
    case 'HIGH':     return '#FB923C';
    case 'CRITICAL': return '#F87171';
    default:         return '#475569';
  }
}

function fiberValue(label: string): number {
  switch (label) {
    case 'LOW':      return 20;
    case 'MODERATE': return 45;
    case 'HIGH':     return 72;
    case 'CRITICAL': return 93;
    default:         return 0;
  }
}

function FiberLoadGauge({ state }: { state: HydrosDisplayState | null }) {
  const label = state?.fiberLoadLabel ?? 'LOW';
  const color = fiberColor(label);
  const val   = state ? fiberValue(label) : 0;

  return (
    <div style={{
      padding: '12px 20px',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      flexShrink: 0,
    }}>
      <SectionLabel>FIBER LOAD</SectionLabel>
      <CircularGauge
        value={val}
        label=""
        size={90}
        color={color}
        centerText={state ? label : '\u2014'}
      />
    </div>
  );
}

// ── PULSE GAUGE ───────────────────────────────────────────────────────────

function PulseGauge({ state }: { state: HydrosDisplayState | null }) {
  const pulseLabel = state?.pulseLabel ?? 'READY';
  const pulseIdx   = state ? (state.bf_pulse_idx || 1) : 1;
  const isActive   = pulseLabel === 'ACTIVE';
  const isCooldown = pulseLabel === 'COOLDOWN';

  // Pulse value: 0 when ready, 42% of clog level when active/post-active
  const clogPct = state ? Math.round(state.clog * 100) : 0;
  const val     = isActive ? 100 : isCooldown ? 65 : 0;
  const color   = isActive ? '#38BDF8' : isCooldown ? '#7DD3FC' : '#1E3858';

  return (
    <div style={{
      padding: '12px 14px',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      flexShrink: 0,
    }}>
      <SectionLabel>PULSE</SectionLabel>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <CircularGauge
          value={val}
          label={`PULSE ${String(pulseIdx).padStart(2, '0')}`}
          size={90}
          color={color}
          centerText={state ? `${clogPct}%` : '\u2014'}
        />
        {/* Right arrows (pulse direction) */}
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: 4,
          paddingTop: 4,
        }}>
          {[0, 1, 2].map(i => (
            <svg key={i} width={14} height={10}>
              <path d="M0,5 L10,0 L10,10 Z"
                fill={isActive ? '#FB923C' : '#1E3858'}
                opacity={0.3 + i * 0.25}
              />
            </svg>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── NEXT ACTION ───────────────────────────────────────────────────────────

function NextAction({ state }: { state: HydrosDisplayState | null }) {
  const actions = state?.nextActions ?? [];

  return (
    <div style={{
      flex: 1,
      padding: '12px 18px',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      minWidth: 170,
    }}>
      <SectionLabel>NEXT ACTION</SectionLabel>
      {state ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {actions.map((a, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: 7 }}>
              <svg width={8} height={10} style={{ flexShrink: 0, marginTop: 1 }}>
                <path d="M0,5 L6,0 L6,10 Z" fill="#334155" />
              </svg>
              <span style={{
                fontFamily: FONT,
                fontSize: 9,
                color: '#607080',
                letterSpacing: '0.04em',
                lineHeight: 1.6,
              }}>
                {a}
              </span>
            </div>
          ))}
        </div>
      ) : (
        <span style={{ fontFamily: FONT, fontSize: 9, color: '#1E293B' }}>
          RUN A SCENARIO
        </span>
      )}
    </div>
  );
}

// ── Main ──────────────────────────────────────────────────────────────────

export default function BottomInstruments({ state }: { state: HydrosDisplayState | null }) {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'stretch',
      background: '#060A14',
      borderTop: '1px solid #1A2840',
      height: 120,
      flexShrink: 0,
    }}>
      <CalibrationTiers />
      <ColDivider />
      <FiberLoadGauge state={state} />
      <ColDivider />
      <PulseGauge state={state} />
      <ColDivider />
      <NextAction state={state} />
    </div>
  );
}
