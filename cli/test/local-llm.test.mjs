// Tests for cli/local-llm.mjs against a fake Ollama (no model needed).
// Run from the repo root:  node --test "cli/test/*.test.mjs"
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { mkdtempSync, writeFileSync, readFileSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const CLI = join(dirname(fileURLToPath(import.meta.url)), "..", "local-llm.mjs");

let ollama;
let url;
let chats = [];
let reply = [{ message: { content: "Two errors." } }, { done: true, prompt_eval_count: 42, eval_count: 3 }];

before(async () => {
  ollama = createServer((req, res) => {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      if (req.url === "/api/tags") {
        res.writeHead(200, { "Content-Type": "application/json" });
        return res.end(JSON.stringify({ models: [{ name: "qwen2.5:7b" }, { name: "llama3:latest" }] }));
      }
      chats.push(JSON.parse(body));
      res.writeHead(200, { "Content-Type": "application/x-ndjson" });
      res.end(reply.map((c) => JSON.stringify(c)).join("\n") + "\n");
    });
  });
  await new Promise((r) => ollama.listen(0, "127.0.0.1", r));
  url = `http://127.0.0.1:${ollama.address().port}`;
});

after(() => ollama?.close());

/** Run the CLI; resolves { code, stdout, stderr, dir }. */
function run(args, env = {}, stdin) {
  const dir = mkdtempSync(join(tmpdir(), "local-llm-test-"));
  return new Promise((resolve) => {
    const p = spawn(process.execPath, [CLI, ...args], {
      env: { ...process.env, OLLAMA_URL: url, LOCAL_LLM_DATA_DIR: dir, LOCAL_LLM_PREFERRED: "gemma4:31b,qwen2.5:7b", LOCAL_LLM_MODEL: "", ...env },
      windowsHide: true,
    });
    let stdout = "";
    let stderr = "";
    p.stdout.on("data", (d) => (stdout += d));
    p.stderr.on("data", (d) => (stderr += d));
    if (stdin !== undefined) p.stdin.end(stdin);
    else p.stdin.end();
    p.on("exit", (code) => resolve({ code, stdout, stderr, dir }));
  });
}

test("uses the first pulled preferred model, wraps files and logs the call", async () => {
  chats = [];
  const tmp = mkdtempSync(join(tmpdir(), "local-llm-in-"));
  const log = join(tmp, "app.log");
  writeFileSync(log, "ERROR one\nERROR two\n");
  const r = await run(["List the errors", "-f", log]);

  assert.equal(r.code, 0, r.stderr);
  assert.equal(r.stdout, "Two errors.\n");
  assert.match(r.stderr, /qwen2\.5:7b · ctx 8\d{3} · in 42 tok · out 3 tok/);
  const sent = chats[0];
  assert.equal(sent.model, "qwen2.5:7b"); // gemma4:31b is not pulled
  assert.equal(sent.think, false);
  assert.ok(sent.options.num_ctx >= 8192);
  assert.match(sent.messages[1].content, /<file name="app\.log">\nERROR one/);
  assert.match(sent.messages[0].content, /^Today's date is \d{4}-\d\d-\d\d\./);

  const logged = JSON.parse(readFileSync(join(r.dir, "calls.jsonl"), "utf8").trim());
  assert.equal(logged.model, "qwen2.5:7b");
  assert.equal(logged.output, "Two errors.\n");
  assert.equal(existsSync(join(r.dir, ".lock")), false); // the lock is released
  rmSync(tmp, { recursive: true, force: true });
});

test("LOCAL_LLM_THINK accepts only explicit true values", async () => {
  chats = [];
  await run(["x", "-m", "qwen2.5:7b"], { LOCAL_LLM_THINK: "0" });
  await run(["x", "-m", "qwen2.5:7b"], { LOCAL_LLM_THINK: "false" });
  await run(["x", "-m", "qwen2.5:7b"], { LOCAL_LLM_THINK: "1" });
  assert.deepEqual(chats.map((c) => c.think), [false, false, true]);
});

test("-o writes the answer to a file, stdin works with -f -, NOLOG skips the log", async () => {
  chats = [];
  const tmp = mkdtempSync(join(tmpdir(), "local-llm-out-"));
  const out = join(tmp, "answer.txt");
  const r = await run(["List the action items", "-f", "-", "-o", out, "-m", "llama3"], { LOCAL_LLM_NOLOG: "1" }, "buy milk");

  assert.equal(r.code, 0, r.stderr);
  assert.equal(readFileSync(out, "utf8"), "Two errors.\n");
  assert.match(r.stdout, /→ .*answer\.txt \(12 bytes\)/);
  assert.equal(chats[0].model, "llama3:latest"); // an untagged name means :latest
  assert.match(chats[0].messages[1].content, /<stdin>\nbuy milk\n<\/stdin>/);
  assert.equal(existsSync(join(r.dir, "calls.jsonl")), false);
  rmSync(tmp, { recursive: true, force: true });
});

test("exit codes: 2 when Ollama is unreachable, 1 for a missing model, file or an Ollama error", async () => {
  assert.equal((await run(["x"], { OLLAMA_URL: "http://127.0.0.1:9" })).code, 2);

  const notPulled = await run(["x", "-m", "mistral:7b"]);
  assert.equal(notPulled.code, 1);
  assert.match(notPulled.stderr, /not pulled\. Installed: qwen2\.5:7b, llama3:latest/);

  const missing = await run(["x", "-m", "qwen2.5:7b", "-f", join(tmpdir(), "does-not-exist-41.txt")]);
  assert.equal(missing.code, 1);
  assert.match(missing.stderr, /cannot read .*does-not-exist-41\.txt: ENOENT/);

  reply = [{ error: "model requires more system memory" }];
  const failed = await run(["x", "-m", "qwen2.5:7b"]);
  reply = [{ message: { content: "Two errors." } }, { done: true, prompt_eval_count: 42, eval_count: 3 }];
  assert.equal(failed.code, 1);
  assert.match(failed.stderr, /more system memory/);
});
