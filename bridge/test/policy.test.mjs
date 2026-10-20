// Auth + allowlist + rate-limit tests for the bridge.
// Run from the repo root:  node --test "bridge/test/*.test.mjs"
//
// Starts the real bridge on a free loopback port with a temporary allowlist.
// No Claude CLI, Qdrant or Ollama is needed: the tests hit GET /rag/collections,
// which passes through the same token and policy checks as /generate before the
// (unreachable) vector store is contacted.
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { startBridge } from "./helpers.mjs";

const TOKEN = "test-token-0123456789abcdef";
const SHARED_LIMIT = 3;
const APP_LIMIT = 2;

let bridge;
let base;

function call(app, token = TOKEN) {
  const headers = {};
  if (token !== null) headers["X-Bridge-Token"] = token;
  if (app !== null) headers["X-App-Name"] = app;
  return fetch(`${base}/rag/collections`, { headers });
}

before(async () => {
  bridge = await startBridge({
    token: TOKEN,
    allowlist: {
      _comment: "strings must not act as an app policy",
      listed: { models: ["haiku"], maxRequestsPerHour: APP_LIMIT },
      "*": { models: ["haiku"], maxRequestsPerHour: SHARED_LIMIT },
    },
  });
  base = bridge.base;
});

after(() => bridge?.stop());

test("missing or wrong token is rejected with 401", async () => {
  assert.equal((await call("listed", null)).status, 401);
  assert.equal((await call("listed", "wrong-token")).status, 401);
});

test("unlisted callers share one bucket: rotating X-App-Name does not escape the * quota", async () => {
  const statuses = [];
  for (let i = 0; i < SHARED_LIMIT + 3; i++) {
    statuses.push((await call(`rotating-${i}`)).status);
  }
  const denied = statuses.filter((s) => s === 403).length;
  assert.equal(statuses.slice(0, SHARED_LIMIT).includes(403), false, `first ${SHARED_LIMIT} allowed: ${statuses}`);
  assert.equal(denied, 3, `requests past the shared limit are denied: ${statuses}`);
  // Missing header, "*", "_comment" and inherited keys all land in the same bucket.
  for (const app of [null, "*", "_comment", "__proto__", "constructor"]) {
    assert.equal((await call(app)).status, 403, `app ${app} must hit the exhausted shared bucket`);
  }
});

test("allowlisted apps keep their own quota", async () => {
  const statuses = [];
  for (let i = 0; i < APP_LIMIT + 1; i++) statuses.push((await call("listed")).status);
  assert.notEqual(statuses[0], 403);
  assert.notEqual(statuses[APP_LIMIT - 1], 403);
  assert.equal(statuses[APP_LIMIT], 403, `listed app limited at ${APP_LIMIT}: ${statuses}`);
});

test("model policy is enforced on /generate", async () => {
  // A fresh listed name would be needed for a clean bucket; the 403 here comes
  // from the model check, which runs before the quota check.
  const r = await fetch(`${base}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Bridge-Token": TOKEN, "X-App-Name": "listed" },
    body: JSON.stringify({ prompt: "hi", model: "opus" }),
  });
  assert.equal(r.status, 403);
  assert.match((await r.json()).error, /not allowed model "opus"/);
});
