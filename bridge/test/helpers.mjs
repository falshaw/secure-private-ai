// Shared test helpers: start the real bridge on a free loopback port with a
// temporary allowlist, state file and (optionally) request log.
import { spawn } from "node:child_process";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const BRIDGE = join(dirname(fileURLToPath(import.meta.url)), "..", "claude-bridge.mjs");

export function freePort() {
  return new Promise((resolve, reject) => {
    const s = createServer();
    s.unref();
    s.on("error", reject);
    s.listen(0, "127.0.0.1", () => {
      const { port } = s.address();
      s.close(() => resolve(port));
    });
  });
}

async function waitForHealth(url, ms = 15000) {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    try {
      const r = await fetch(`${url}/health`);
      if (r.status === 200 || r.status === 503) return;
    } catch {}
    await new Promise((r) => setTimeout(r, 150));
  }
  throw new Error("bridge did not start");
}

/**
 * Start a bridge. No Claude CLI is needed: CLAUDE_BIN points at a binary that
 * does not exist, so generation fails with "spawn failed" after the token and
 * policy checks have run.
 */
export async function startBridge({ token, allowlist, env = {}, requestLog = false }) {
  const dir = mkdtempSync(join(tmpdir(), "bridge-test-"));
  const allowlistFile = join(dir, "allowlist.json");
  writeFileSync(allowlistFile, JSON.stringify(allowlist));
  const port = await freePort();
  const base = `http://127.0.0.1:${port}`;
  const proc = spawn(process.execPath, [BRIDGE], {
    env: {
      ...process.env,
      CLAUDE_BRIDGE_HOST: "127.0.0.1",
      CLAUDE_BRIDGE_PORT: String(port),
      CLAUDE_BRIDGE_TOKEN: token,
      CLAUDE_BRIDGE_ALLOWLIST_FILE: allowlistFile,
      CLAUDE_BIN: "claude-not-installed-for-tests",
      QDRANT_URL: "http://127.0.0.1:9",
      OLLAMA_URL: "",
      EMBED_PROVIDER: "none",
      BRIDGE_STATE_FILE: join(dir, "state.json"),
      BRIDGE_REQUEST_LOG: requestLog ? "true" : "false",
      BRIDGE_REQUEST_LOG_DIR: join(dir, "logs"),
      ...env,
    },
    stdio: "ignore",
    windowsHide: true,
  });
  await waitForHealth(base);
  return {
    base,
    dir,
    stop() {
      proc.kill();
      try {
        rmSync(dir, { recursive: true, force: true });
      } catch {} // Windows may still hold a file handle for a moment
    },
  };
}
