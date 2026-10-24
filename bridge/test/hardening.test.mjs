// Startup and access hardening: weak tokens refused, the optional admin token
// for the cross-app routes, and a clean 404 for a collection that does not exist.
// Run from the repo root:  node --test "bridge/test/*.test.mjs"
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { startBridge, freePort } from "./helpers.mjs";

const BRIDGE = join(dirname(fileURLToPath(import.meta.url)), "..", "claude-bridge.mjs");
const TOKEN = "test-token-hardening-0123456789";
const ADMIN = "admin-token-hardening-0123456789";

/** Start the bridge with the given env and resolve with its exit code (or null if it keeps running). */
async function exitCodeWith(env) {
  const port = await freePort();
  const proc = spawn(process.execPath, [BRIDGE], {
    env: { ...process.env, CLAUDE_BRIDGE_HOST: "127.0.0.1", CLAUDE_BRIDGE_PORT: String(port), EMBED_PROVIDER: "none", ...env },
    stdio: "ignore",
    windowsHide: true,
  });
  return new Promise((resolve) => {
    const t = setTimeout(() => {
      proc.kill();
      resolve(null);
    }, 4000);
    proc.on("exit", (code) => {
      clearTimeout(t);
      resolve(code);
    });
  });
}

test("refuses to start with the .env.example placeholder or a short token", async () => {
  assert.equal(await exitCodeWith({ CLAUDE_BRIDGE_TOKEN: "change-me" }), 1);
  assert.equal(await exitCodeWith({ CLAUDE_BRIDGE_TOKEN: "short-but-random-91x" }), 1);
  assert.equal(await exitCodeWith({ CLAUDE_BRIDGE_TOKEN: TOKEN, CLAUDE_BRIDGE_ADMIN_TOKEN: "replace-me-please-0123456789" }), 1);
});

let bridge;
let qdrant;

before(async () => {
  // A fake Qdrant that knows no collections.
  const qPort = await freePort();
  qdrant = createServer((req, res) => {
    req.resume();
    req.on("end", () => {
      if (req.url === "/collections") {
        res.writeHead(200, { "Content-Type": "application/json" });
        return res.end(JSON.stringify({ result: { collections: [] } }));
      }
      res.writeHead(404, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ status: { error: "Not found: Collection `nope` doesn't exist!" } }));
    });
  });
  await new Promise((r) => qdrant.listen(qPort, "127.0.0.1", r));
  bridge = await startBridge({
    token: TOKEN,
    allowlist: { "*": { models: ["haiku"], maxRequestsPerHour: 1000 } },
    env: { QDRANT_URL: `http://127.0.0.1:${qPort}`, CLAUDE_BRIDGE_ADMIN_TOKEN: ADMIN },
    requestLog: true,
  });
});

after(() => {
  bridge?.stop();
  qdrant?.close();
});

test("with an admin token set, /log/tail and /admin/reload also need X-Admin-Token", async () => {
  const H = { "X-Bridge-Token": TOKEN };
  assert.equal((await fetch(`${bridge.base}/log/tail?n=1`, { headers: H })).status, 403);
  assert.equal((await fetch(`${bridge.base}/admin/reload`, { method: "POST", headers: H })).status, 403);
  const ok = await fetch(`${bridge.base}/log/tail?n=1`, { headers: { ...H, "X-Admin-Token": ADMIN } });
  assert.equal(ok.status, 200);
  // the admin token alone is not enough: the shared token is still checked first
  assert.equal((await fetch(`${bridge.base}/log/tail?n=1`, { headers: { "X-Admin-Token": ADMIN } })).status, 401);
});

// /rag/search takes the same path once the query is embedded; the tests run
// without an embedder, so they cover the two delete routes.
test("deleting from a collection that does not exist is a 404 without Qdrant's raw body", async () => {
  const H = { "X-Bridge-Token": TOKEN };
  const point = await fetch(`${bridge.base}/rag/nope/doc-1`, { method: "DELETE", headers: H });
  assert.equal(point.status, 404);
  assert.equal((await point.json()).error, 'collection "nope" not found');
  const coll = await fetch(`${bridge.base}/rag/nope`, { method: "DELETE", headers: H });
  assert.equal(coll.status, 404);
});
