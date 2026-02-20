import AppLayout from "./layout/AppLayout";
import HeaderBar from "./layout/HeaderBar";
import SystemCutaway from "./components/SystemView/SystemCutaway";
import CoreMetricsPanel from "./panels/CoreMetricsPanel";
import ValidationPanel from "./panels/ValidationPanel";

export default function App() {
  return (
    <AppLayout>
      <HeaderBar />
      <div
        style={{
          borderTop: "1px solid var(--border-subtle)",
          borderBottom: "1px solid var(--border-subtle)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          minHeight: "70vh",
          height: "70vh",
          padding: "32px 0",
          background: "linear-gradient(to bottom, #0F172A, #0E1624)",
        }}
      >
        <SystemCutaway
          running={false}
          flow={0.18}
          clog={0.03}
          eField={0.52}
          backflush={0.0}
          storageFill={0.25}
          width="100%"
          height="100%"
        />
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, padding: 16 }}>
        <CoreMetricsPanel />
        <ValidationPanel />
      </div>
      <div style={{
        borderTop: "1px solid var(--border-subtle)",
        padding: 16,
        color: "var(--text-secondary)"
      }}>
        TABS: Hydraulics | Clogging | Particles | Electro | Sensors | RL
      </div>
    </AppLayout>
  );
}