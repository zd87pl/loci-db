import path from "node:path";
import { cloudflareTest, readD1Migrations } from "@cloudflare/vitest-plugin";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [
    cloudflareTest(async () => ({
      // No calls to Cloudflare from tests: Workers AI is stubbed (test/ai-stub.ts).
      remoteBindings: false,
      wrangler: { configPath: "./wrangler.jsonc" },
      miniflare: {
        bindings: {
          ADMIN_TOKEN: "test-admin-token",
          DEV_ALLOW_UNPINNED: "1",
          TEST_MIGRATIONS: await readD1Migrations(path.join(__dirname, "migrations")),
        },
      },
    })),
  ],
  test: { setupFiles: ["./test/apply-migrations.ts"] },
});
