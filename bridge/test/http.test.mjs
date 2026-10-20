// Request handling tests: body validation, routing, metrics labels, model
// resolution, the /v1/chat quota and RAG point ids.
// Run from the repo root:  node --test "bridge/test/*.test.mjs"
//
// No Claude CLI is needed (generation fails with "spawn failed" once the
// policy checks have passed). A small fake Qdrant records what the bridge
// sends it, so the id mapping can be checked without a real vector store.
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { startBridge, freePort } from "./helpers.mjs";

const TOKEN = "test-token-http-0123456789";
const H = { "Content-Type": "application/json", "X-Bridge-Token": TOKEN };

let bridge;
let base;
let qdrant;
const qdrantCalls = [];

function post(path, body, headers = {}) {
  return fetch(`${base}${path}`, {
    method: "POST",
    headers: { ...H, ...headers },
    body: typeof body === "string" ? body : JSON.stringify(body),
  });
}

async function lastLogRecord(app) {
  const r = await fetch(`${base}/log/tail?n=1&app=${encodeURIComponent(app)}`, { headers: H });
  const j = await r.json();
  return j.records[0];
}

before(async () => {
  const qPort = await freePort();
  qdrant = createServer((req, res) => {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      qdrantCalls.push({ method: req.method, url: req.url, body: body ? JSON.parse(body) : null });
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ result: req.url === "/collections" ? { collections: [] } : true }));
    });
  });
  await new Promise((r) => qdrant.listen(qPort, "127.0.0.1", r));

  // Request log on, so tests can read the model a call ran with.
  bridge = await startBridge({
    token: TOKEN,
    allowlist: {
      limited: { models: ["sonnet"], maxRequestsPerHour: 2 },
      "*": { models: ["haiku"], maxRequestsPerHour: 1000 },
    },
    env: { QDRANT_URL: `http://127.0.0.1:${qPort}` },
    requestLog: true,
  });
  base = bridge.base;
});

after(() => {
  bridge?.stop();
  qdrant?.close();
});

test("a JSON body that is not an object gets 400 instead of hanging", async () => {
  for (const body of ["null", "[]", "42", '"text"']) {
    const r = await post("/generate", body, {}).catch((e) => e);
    assert.equal(r.status, 400, `body ${body}`);
    assert.match((await r.json()).error, /must be an object/);
  }
});

test("an oversized body gets 413, not a connection reset", async () => {
  const big = JSON.stringify({ prompt: "a".repeat(1024 * 1024 + 10) });
  const r = await post("/generate", big);
  assert.equal(r.status, 413);
  assert.match((await r.json()).error, /exceeds/);
});

test("query strings do not break routing", async () => {
  assert.equal((await fetch(`${base}/health?probe=1`)).status, 200);
  assert.equal((await fetch(`${base}/v1/models?x=1`, { headers: H })).status, 200);
});

test("metrics labels are route names, never raw URLs", async () => {
  for (let i = 0; i < 5; i++) await fetch(`${base}/random-${i}?q=${i}`);
  const text = await (await fetch(`${base}/metrics`, { headers: H })).text();
  assert.doesNotMatch(text, /random-/);
  assert.match(text, /path="other",status="401"/);
});

test("an unknown model name is rejected", async () => {
  const r = await post("/generate", { prompt: "hi", model: "gpt-5" });
  assert.equal(r.status, 400);
  assert.match((await r.json()).error, /unknown model/);
});

test("an omitted model runs on the first model the policy allows", async () => {
  const r = await post("/generate", { prompt: "hi" }, { "X-App-Name": "anyone" });
  assert.equal(r.status, 500); // no CLI in tests; the policy check has passed
  assert.equal((await lastLogRecord("anyone")).model, "haiku");
});

test("/v1/chat/completions allows exactly maxRequestsPerHour calls", async () => {
  const chat = () =>
    post("/v1/chat/completions", { messages: [{ role: "user", content: "a" }] }, { "X-App-Name": "limited" });
  const statuses = [(await chat()).status, (await chat()).status, (await chat()).status];
  // 502 = policy passed, CLI missing. The third call is over the limit.
  assert.deepEqual(statuses, [502, 502, 403]);
});

test("RAG delete maps ids the same way upsert does", async () => {
  qdrantCalls.length = 0;
  const del = (id) =>
    fetch(`${base}/rag/docs/${encodeURIComponent(id)}`, { method: "DELETE", headers: H });
  assert.equal((await del("42")).status, 200);
  assert.equal((await del("doc-gdpr")).status, 200);
  const uuid = "41b2533d-3a30-5031-8d5a-d6264d0938de";
  assert.equal((await del(uuid)).status, 200);
  const ids = qdrantCalls.filter((c) => c.url.includes("/points/delete")).map((c) => c.body.points[0]);
  assert.equal(ids[0], 42);
  assert.match(String(ids[1]), /^[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
  assert.notEqual(ids[1], "doc-gdpr");
  assert.equal(ids[2], uuid);
});

test("a non-object RAG payload is rejected", async () => {
  const r = await post("/rag/upsert", { collection: "docs", text: "x", payload: "abc" });
  assert.equal(r.status, 400);
  const b = await post("/rag/upsert-batch", { collection: "docs", items: [{ text: "x", payload: [1] }] });
  assert.equal(b.status, 400);
});
