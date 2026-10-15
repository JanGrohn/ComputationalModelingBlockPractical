import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  // relative asset paths, so the build works wherever it is served from
  base: "./",
  plugins: [react()],
  build: { outDir: "dist", sourcemap: true },
});
