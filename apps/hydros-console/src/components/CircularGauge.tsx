/**
 * components/CircularGauge.tsx
 *
 * Reusable 270° sweep arc gauge. Renders a track arc and a value arc.
 * Center label shows numeric value; sub-label below shows identifier.
 */

const FONT = '"JetBrains Mono", "Fira Code", "Courier New", monospace';

export interface CircularGaugeProps {
  value: number;         // 0–100
  label: string;         // text below center value
  size: number;          // px
  color: string;         // filled arc color
  trackColor?: string;   // unfilled arc color
  centerText?: string;   // override center text (default: "${value}%")
}

export default function CircularGauge({
  value, label, size, color,
  trackColor = '#1A2535',
  centerText,
}: CircularGaugeProps) {
  const cx = size / 2;
  const cy = size / 2;
  const r = size / 2 - 9;
  const circ = 2 * Math.PI * r;
  const arcLen = circ * 0.75;          // 270° = 75% of full circle
  const gap = circ - arcLen;
  const filled = arcLen * Math.max(0, Math.min(100, value)) / 100;

  const rotate = `rotate(135 ${cx} ${cy})`;
  const fontSize = Math.round(size * 0.20);
  const subSize  = Math.round(size * 0.09);

  return (
    <svg width={size} height={size} style={{ display: 'block', overflow: 'visible' }}>
      {/* Track arc */}
      <circle
        cx={cx} cy={cy} r={r}
        fill="none"
        stroke={trackColor}
        strokeWidth={5}
        strokeDasharray={`${arcLen} ${gap}`}
        strokeLinecap="round"
        transform={rotate}
      />
      {/* Value arc */}
      {value > 0 && (
        <circle
          cx={cx} cy={cy} r={r}
          fill="none"
          stroke={color}
          strokeWidth={5}
          strokeDasharray={`${filled} ${circ - filled}`}
          strokeLinecap="round"
          transform={rotate}
        />
      )}
      {/* Center value */}
      <text
        x={cx} y={cy + fontSize * 0.36}
        textAnchor="middle"
        fill="#E2E8F0"
        fontSize={fontSize}
        fontWeight={700}
        fontFamily={FONT}
        letterSpacing="0.02em"
      >
        {centerText ?? `${Math.round(value)}%`}
      </text>
      {/* Sub-label */}
      <text
        x={cx} y={cy + fontSize * 0.36 + subSize + 4}
        textAnchor="middle"
        fill="#475569"
        fontSize={subSize}
        fontFamily={FONT}
        letterSpacing="0.12em"
      >
        {label}
      </text>
    </svg>
  );
}
