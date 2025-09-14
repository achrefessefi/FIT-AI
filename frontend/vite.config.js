// frontend/vite.config.js
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  // Load env from the repo root (where your shared .env lives)
  envDir: "..",

  plugins: [react()],

  server: {
    port: 5173,
    strictPort: true,
    open: true, // auto-open browser
  },
  preview: {
    port: 5174,
    strictPort: true,
  },
});
