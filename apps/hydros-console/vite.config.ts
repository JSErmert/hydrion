import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    // Allow ngrok tunnel hostnames so the dev server doesn't reject the
    // forwarded request (Vite 5 blocks unknown hosts by default).
    allowedHosts: [".ngrok-free.app", ".ngrok-free.dev", ".ngrok.app", ".ngrok.io"],
    // Proxy /api/* to the FastAPI backend on localhost:8000.  This routes
    // all backend traffic through Vite (and therefore through the existing
    // ngrok tunnel), so the phone view can hit /api endpoints without a
    // second tunnel or any CORS plumbing.  Backend must be running locally
    // on port 8000 for this to resolve.
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});