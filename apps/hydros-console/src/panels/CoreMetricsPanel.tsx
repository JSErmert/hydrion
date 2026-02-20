import MetricTile from "../components/MetricTile";

export default function CoreMetricsPanel() {
  return (
    <div style={{ display: "grid", gap: 6 }}>
      <MetricTile label="Flow" value="0.18" />
      <MetricTile label="Pressure" value="0.09" />
      <MetricTile label="Clog" value="0.03" />
      <MetricTile label="Capture Eff" value="0.98" />
      <MetricTile label="Electric Field" value="0.52" />
    </div>
  );
}