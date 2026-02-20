import SystemSvg from "./SystemSvg";

export default function SystemView() {
  return (
    <div
      style={{
        borderTop: "1px solid var(--border-subtle)",
        borderBottom: "1px solid var(--border-subtle)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        minHeight: "70vh",
        padding: "32px 0",
        background: "linear-gradient(to bottom, #0F172A, #0E1624)"
      }}
    >
      <SystemSvg />
    </div>
  );
}
