/**
 * components/RightAdvisoryPanel.tsx
 *
 * Fixed-width right panel. Displays:
 *   - SYSTEM STATUS badge (color-coded precedence chain)
 *   - EFFICIENCY %
 *   - AGENT INSIGHTS list (state-driven next action strings)
 */

import type { HydrosDisplayState, SystemStatus } from '../scenarios/displayStateMapper';

const FONT = '"JetBrains Mono", "Fira Code", "Courier New", monospace';

function statusColor(status: SystemStatus | undefined): string {
  switch (status) {
    case 'BACKFLUSH ACTIVE':      return '#38BDF8';
    case 'BYPASS ACTIVE':         return '#F87171';
    case 'MAINTENANCE REQUIRED':  return '#FB923C';
    case 'RISING LOAD':           return '#FBBF24';
    default:                      return '#34D399';
  }
}

function SectionLabel({ children }: { children: string }) {
  return (
    <div style={{
      fontFamily: FONT,
      fontSize: 8,
      color: '#475569',
      letterSpacing: '0.16em',
      marginBottom: 8,
    }}>
      {children}
    </div>
  );
}

export default function RightAdvisoryPanel({ state }: { state: HydrosDisplayState | null }) {
  const color      = statusColor(state?.systemStatus);
  const statusLabel = state?.systemStatus ?? 'NO DATA';
  const effStr      = state != null ? `${state.efficiencyPct.toFixed(1)}%` : '\u2014';

  return (
    <div style={{
      width: 220,
      flexShrink: 0,
      background: '#080D18',
      borderLeft: '1px solid var(--border-subtle)',
      display: 'flex',
      flexDirection: 'column',
      padding: '20px 18px',
      gap: 24,
      overflowY: 'auto',
    }}>

      {/* SYSTEM STATUS */}
      <div>
        <SectionLabel>SYSTEM STATUS</SectionLabel>
        <div style={{
          background: `${color}18`,
          border: `1px solid ${color}55`,
          borderRadius: 4,
          padding: '8px 12px',
        }}>
          <span style={{
            fontFamily: FONT,
            fontSize: 11,
            color,
            fontWeight: 700,
            letterSpacing: '0.06em',
          }}>
            {statusLabel}
          </span>
        </div>
      </div>

      {/* EFFICIENCY */}
      <div>
        <SectionLabel>EFFICIENCY</SectionLabel>
        <span style={{
          fontFamily: FONT,
          fontSize: 26,
          color: state != null ? '#E2E8F0' : '#2D3E56',
          fontWeight: 600,
          letterSpacing: '0.02em',
        }}>
          {effStr}
        </span>
        {state != null && (
          <div style={{
            marginTop: 6,
            height: 4,
            background: '#1E293B',
            borderRadius: 2,
          }}>
            <div style={{
              width: `${Math.min(100, state.efficiencyPct)}%`,
              height: '100%',
              background: state.efficiencyPct > 70 ? '#34D399' :
                          state.efficiencyPct > 40 ? '#FBBF24' : '#F87171',
              borderRadius: 2,
              transition: 'width 0.2s ease',
            }} />
          </div>
        )}
      </div>

      {/* AGENT INSIGHTS */}
      <div>
        <SectionLabel>AGENT INSIGHTS</SectionLabel>
        {state != null ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {state.nextActions.map((action, i) => (
              <div key={i} style={{
                fontFamily: FONT,
                fontSize: 10,
                color: '#94A3B8',
                letterSpacing: '0.04em',
                paddingLeft: 8,
                borderLeft: '2px solid #1E293B',
              }}>
                {action}
              </div>
            ))}
          </div>
        ) : (
          <div style={{
            fontFamily: FONT,
            fontSize: 10,
            color: '#2D3E56',
            letterSpacing: '0.04em',
          }}>
            RUN A SCENARIO
          </div>
        )}
      </div>

      {/* REWARD */}
      {state != null && (
        <div>
          <SectionLabel>REWARD</SectionLabel>
          <span style={{
            fontFamily: FONT,
            fontSize: 13,
            color: state.reward >= 0 ? '#34D399' : '#F87171',
            fontWeight: 600,
          }}>
            {state.reward.toFixed(4)}
          </span>
        </div>
      )}

    </div>
  );
}
