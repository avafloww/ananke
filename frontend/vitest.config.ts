import { defineConfig } from "vitest/config";

// Minimal Vitest config. Component tests run in jsdom; jsx/esm transform is
// provided by vite-jsx/esbuild via `vitest`, and the existing
// `vite.config.ts` carries the react/tailwind plugins for the app build.
// Keeping this file free of the React + rolldown plugins avoids a type-space
// conflict between vite 8 and the vendored vite vitest links against.
export default defineConfig({
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
