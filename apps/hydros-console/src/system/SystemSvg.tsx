import { useEffect, useState } from "react";

export default function SystemSvg() {
  const [pulse, setPulse] = useState(0.4);

  useEffect(() => {
    const interval = setInterval(() => {
      setPulse((prev) => (prev > 0.6 ? 0.4 : prev + 0.02));
    }, 80);
    return () => clearInterval(interval);
  }, []);

  return (
    <svg width="1000" height="500" viewBox="0 0 1000 500">
      {/* Inlet */}
      <rect x="120" y="220" width="150" height="50" fill="#1E293B" rx="4" />

      {/* Stage Cylinders */}
      <rect x="350" y="140" width="300" height="70" fill="#1E293B" rx="6" />
      <rect x="350" y="225" width="300" height="70" fill="#1E293B" rx="6" />
      <rect x="350" y="310" width="300" height="70" fill="#1E293B" rx="6" />

      {/* Electric Ring (Idle Pulse) */}
      <circle
        cx="500"
        cy="260"
        r="170"
        stroke="var(--accent-electric)"
        strokeWidth="4"
        fill="none"
        opacity={pulse}
      />

      {/* Storage Tank */}
      <rect x="450" y="410" width="100" height="50" fill="#1E293B" rx="4" />

      {/* Status */}
      <text
        x="430"
        y="80"
        fill="var(--accent-electric)"
        fontSize="22"
        style={{ letterSpacing: "2px" }}
      >
        SYSTEM READY
      </text>
    </svg>
  );
}
