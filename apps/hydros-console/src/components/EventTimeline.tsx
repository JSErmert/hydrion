/**
 * components/EventTimeline.tsx
 *
 * Horizontal event chain: ⊕ START → marker1 → marker2 → ... → END
 * Past events are bright; future events are dimmed.
 */

import type { ScenarioEventMarker } from '../api/types';

const FONT = '"JetBrains Mono", "Fira Code", "Courier New", monospace';

const MARKER_LABEL: Record<string, string> = {
  threshold_crossing:  'FOULING RISE',
  backflush_start:     'PULSE 01',
  backflush_end:       'PULSE END',
  bypass_start:        'BYPASS START',
  bypass_end:          'BYPASS END',
  disturbance_start:   'DIST+',
  disturbance_end:     'DIST-',
};

// Skip purely bookend types we don't want on the timeline
const SKIP_TYPES = new Set(['scenario_start', 'scenario_end']);

interface Props {
  markers: ScenarioEventMarker[];
  currentTime: number;
}

export default function EventTimeline({ markers, currentTime }: Props) {
  const visible = markers.filter(m => !SKIP_TYPES.has(m.type));

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      padding: '0 16px',
      height: 36,
      borderTop: '1px solid #1E293B',
      background: '#060A14',
      overflowX: 'auto',
      flexShrink: 0,
      gap: 0,
    }}>

      {/* START node */}
      <EventNode label="START" time={0} currentTime={currentTime} first />

      {visible.map((m, i) => (
        <EventNode
          key={i}
          label={MARKER_LABEL[m.type] ?? m.type.replace('_', ' ').toUpperCase()}
          time={m.time}
          currentTime={currentTime}
        />
      ))}
    </div>
  );
}

interface NodeProps {
  label: string;
  time: number;
  currentTime: number;
  first?: boolean;
}

function EventNode({ label, time, currentTime, first = false }: NodeProps) {
  const past  = currentTime >= time;
  const exact = Math.abs(currentTime - time) < 0.5;

  const dotColor   = past  ? '#38BDF8' : '#1A2E48';
  const textColor  = past  ? '#64748B' : '#2A3E56';
  const arrowColor = past  ? '#2A4060' : '#161E2A';

  return (
    <div style={{ display: 'flex', alignItems: 'center', flexShrink: 0 }}>
      {/* Arrow between nodes */}
      {!first && (
        <div style={{
          width: 18,
          height: 1,
          background: arrowColor,
          position: 'relative',
        }}>
          <div style={{
            position: 'absolute',
            right: -2,
            top: -3,
            width: 0,
            height: 0,
            borderTop: '3px solid transparent',
            borderBottom: '3px solid transparent',
            borderLeft: `4px solid ${arrowColor}`,
          }} />
        </div>
      )}

      {/* Node */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 4,
        flexShrink: 0,
      }}>
        {/* Dot */}
        <div style={{
          width: 6,
          height: 6,
          borderRadius: '50%',
          background: dotColor,
          boxShadow: exact ? `0 0 6px #38BDF8` : undefined,
          flexShrink: 0,
        }} />
        {/* Label */}
        <span style={{
          fontFamily: FONT,
          fontSize: 8,
          color: textColor,
          letterSpacing: '0.08em',
          whiteSpace: 'nowrap',
        }}>
          {label}
        </span>
      </div>
    </div>
  );
}
