export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ display: "grid", gridTemplateRows: "64px 1fr auto auto", height: "100vh" }}>
      {children}
    </div>
  );
}