const FONT = '"JetBrains Mono", "Fira Code", "Courier New", monospace';

export default function HeaderBar() {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 24px',
      background: '#0A0F1C',
      borderBottom: '1px solid #1E293B',
      height: 48,
      flexShrink: 0,
    }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
        <span style={{
          fontFamily: FONT,
          fontSize: 15,
          fontWeight: 700,
          color: '#E2E8F0',
          letterSpacing: '0.08em',
        }}>
          HydrOS
        </span>
        <span style={{
          fontFamily: FONT,
          fontSize: 11,
          color: '#475569',
          letterSpacing: '0.06em',
        }}>
          Machine View
        </span>
      </div>

      <div style={{
        fontFamily: FONT,
        fontSize: 10,
        color: '#38BDF8',
        border: '1px solid #1E3A5F',
        background: '#0C1F35',
        padding: '3px 10px',
        borderRadius: 3,
        letterSpacing: '0.12em',
      }}>
        SCENARIO PLAYBACK
      </div>
    </div>
  );
}
