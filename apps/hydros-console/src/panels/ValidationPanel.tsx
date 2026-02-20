import MetricTile from "../components/MetricTile";

export default function ValidationPanel() {
  return (
    <div style={{ display: "grid", gap: 6 }}>
      <MetricTile label="Shield Status" value="Idle" />
      <MetricTile label="Violations" value="0" />
      <MetricTile label="Recovery Timer" value="—" />
      <MetricTile label="Anomaly Count" value="0" />
      <MetricTile label="Mass Balance" value="OK" />
    </div>
  );
}