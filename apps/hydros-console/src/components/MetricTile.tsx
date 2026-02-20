export default function MetricTile({
  label,
  value
}: {
  label: string;
  value: string;
}) {
  return (
    <div
      style={{
        padding: "14px 18px",
        background: "var(--bg-panel)",
        border: "1px solid var(--border-subtle)",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center"
      }}
    >
      <div
        style={{
          fontSize: 12,
          color: "var(--text-secondary)",
          textTransform: "uppercase",
          letterSpacing: "1px"
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontSize: 28,
          fontWeight: 500
        }}
      >
        {value}
      </div>
    </div>
  );
}
