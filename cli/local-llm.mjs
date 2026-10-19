#!/usr/bin/env node
/**
 * local-llm.mjs — hand bulk text work (summaries, extraction, drafts, translation) to a
 * local Ollama model instead of a hosted API.
 *
 * Inputs go file → Ollama directly; with -o the result lands on disk, so a calling agent
 * or script only reads what it needs.
 *
 * Usage:
 *   node cli/local-llm.mjs "prompt" -f a.md -f b.log          # files appended as context
 *   node cli/local-llm.mjs "translate to German" -f post.md -o post.de.md
 *   node cli/local-llm.mjs -s "system" "prompt"
 *   cat x.txt | node cli/local-llm.mjs "summarise this" -f -      # piped input only with -f -
 *   node cli/local-llm.mjs -m gemma4:31b "..."                # pin a model
 *
 * Model: -m, else LOCAL_LLM_MODEL, else the first pulled model in LOCAL_LLM_PREFERRED (comma-separated).
 * Exit codes: 0 ok · 1 error · 2 Ollama not running
 */

import { readFileSync, writeFileSync, renameSync, appendFileSync, mkdirSync, openSync, closeSync, statSync, unlinkSync } from 'node:fs';
import { request } from 'node:http';
import { basename, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const OLLAMA = process.env.OLLAMA_URL ?? 'http://127.0.0.1:11434';
const DATASET_DIR = process.env.LOCAL_LLM_DATA_DIR ?? join(dirname(fileURLToPath(import.meta.url)), '..', 'data', 'local-llm');
// Models tried in order when neither -m nor LOCAL_LLM_MODEL is given (first one that is pulled wins).
const PREFERRED = (process.env.LOCAL_LLM_PREFERRED ?? 'gemma4:31b').split(',').map((m) => m.trim()).filter(Boolean);
const MAX_CTX = 65536;

const args = process.argv.slice(2);
const USAGE = 'usage: local-llm.mjs "prompt" [-f file]... [-s system] [-m model] [-o out]   (env: OLLAMA_URL, LOCAL_LLM_MODEL, LOCAL_LLM_PREFERRED, LOCAL_LLM_DATA_DIR, LOCAL_LLM_THINK=1, LOCAL_LLM_NOLOG=1)';
if (args.length === 1 && ['-h', '--help'].includes(args[0])) {
  console.log(USAGE);
  process.exit(0);
}
const files = [];
let system, model, out, prompt = '';
for (let i = 0; i < args.length; i++) {
  const a = args[i];
  if (['-f', '-s', '-m', '-o'].includes(a) && i + 1 >= args.length) {
    console.error(`${a} needs a value`);
    process.exit(1);
  }
  if (a === '-f') files.push(args[++i]);
  else if (a === '-s') system = args[++i];
  else if (a === '-m') model = args[++i];
  else if (a === '-o') out = args[++i];
  else prompt += (prompt ? ' ' : '') + a;
}
if (!prompt) {
  console.error(USAGE);
  process.exit(1);
}
// The model otherwise assumes its training-time clock and treats current dates as "in the future".
const today = new Date().toISOString().slice(0, 10);
system = `Today's date is ${today}.` + (system ? `\n\n${system}` : '');

let tags;
try {
  tags = await (await fetch(`${OLLAMA}/api/tags`, { signal: AbortSignal.timeout(5000) })).json();
} catch {
  console.error(`Ollama not reachable at ${OLLAMA} — start Ollama or set OLLAMA_URL.`);
  process.exit(2);
}
const installed = tags.models.map((m) => m.name);
model ??= process.env.LOCAL_LLM_MODEL ?? PREFERRED.find((m) => installed.includes(m));
if (model && !model.includes(':')) model += ':latest'; // Ollama lists untagged pulls as name:latest
if (!model || !installed.includes(model)) {
  console.error(`Model ${model ?? PREFERRED.join(' / ')} not pulled. Installed: ${installed.join(', ')}`);
  process.exit(1);
}

// Piped input only on explicit `-f -`: auto-reading stdin whenever it is not a TTY hangs forever in shells
// that hand the process an open pipe that never closes.
const context = files
  .map((f) => (f === '-' ? `<stdin>\n${readFileSync(0, 'utf8')}\n</stdin>` : `<file name="${basename(f)}">\n${readFileSync(f, 'utf8')}\n</file>`))
  .join('\n\n');
const content = context ? `${context}\n\n${prompt}` : prompt;

// Ollama defaults to a 4k window and silently truncates — size it to the input (~3.5 chars/token)
// plus room for the answer.
const numCtx = Math.min(MAX_CTX, Math.max(8192, Math.ceil(content.length / 3.5) + 8192));

// One local job at a time: parallel callers would otherwise load several large contexts at once
// (RAM/VRAM blow-up). A lock older than 40 min is stale.
mkdirSync(DATASET_DIR, { recursive: true });
const LOCK = join(DATASET_DIR, '.lock');
for (let waited = 0; ; waited += 5) {
  try {
    closeSync(openSync(LOCK, 'wx'));
    break;
  } catch {
    try {
      if (Date.now() - statSync(LOCK).mtimeMs > 40 * 60 * 1000) unlinkSync(LOCK);
    } catch {}
    if (waited % 60 === 0) console.error(`[local-llm] waiting for another local job (${waited} s)`);
    await new Promise((r) => setTimeout(r, 5000));
  }
}
process.on('exit', () => {
  try {
    unlinkSync(LOCK);
  } catch {}
});
process.on('SIGINT', () => process.exit(130));

// Streamed over node:http, not fetch: fetch (undici) drops a request after 300 s without response
// headers, and Ollama sends none before the first token, so a long answer or a queued request died at 5 min.
const t0 = Date.now();
const body = JSON.stringify({
  model,
  stream: true,
  // LOCAL_LLM_THINK=1 lets the model reason first (slower, fewer slips); the reasoning is not part of the output.
  think: !!process.env.LOCAL_LLM_THINK,
  options: { num_ctx: numCtx },
  messages: [{ role: 'system', content: system }, { role: 'user', content }],
});
const data = await new Promise((resolve) => {
  const acc = { status: 0, content: '', error: undefined, prompt_eval_count: undefined, eval_count: undefined };
  const req = request(
    `${OLLAMA}/api/chat`,
    { method: 'POST', headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) } },
    (res) => {
      acc.status = res.statusCode;
      res.setEncoding('utf8');
      let buffer = '';
      const take = (line) => {
        if (!line.trim()) return;
        let chunk;
        try {
          chunk = JSON.parse(line);
        } catch {
          acc.error ??= 'non-JSON response';
          return;
        }
        if (chunk.error) acc.error = chunk.error;
        acc.content += chunk.message?.content ?? '';
        if (chunk.done) Object.assign(acc, { prompt_eval_count: chunk.prompt_eval_count, eval_count: chunk.eval_count });
      };
      res.on('data', (d) => {
        buffer += d;
        let nl;
        while ((nl = buffer.indexOf('\n')) >= 0) {
          take(buffer.slice(0, nl));
          buffer = buffer.slice(nl + 1);
        }
      });
      res.on('end', () => {
        take(buffer);
        resolve(acc);
      });
    },
  );
  req.setTimeout(30 * 60 * 1000, () => req.destroy(new Error('no data for 30 min')));
  req.on('error', (e) => resolve({ ...acc, error: e.message }));
  req.end(body);
});
if (data.status !== 200 || data.error || !data.content.trim()) {
  console.error(`Ollama ${data.status}: ${data.error ?? 'empty response'}`);
  process.exit(1);
}

const text = data.content.trim() + '\n';

// Every call is appended to <data dir>/calls.jsonl (local-only, may hold private data) so it can be
// reviewed or reused as fine-tuning data. LOCAL_LLM_NOLOG=1 skips it.
if (!process.env.LOCAL_LLM_NOLOG) {
  appendFileSync(
    join(DATASET_DIR, 'calls.jsonl'),
    JSON.stringify({ ts: new Date().toISOString(), model, system, prompt, files: files.map((f) => basename(f)), input: content, output: text }) + '\n',
  );
}
const stats = `[local-llm] ${model} · ctx ${numCtx} · in ${data.prompt_eval_count} tok · out ${data.eval_count} tok · ${((Date.now() - t0) / 1000).toFixed(0)} s`;
if (out) {
  writeFileSync(out + '.tmp', text, 'utf8');
  renameSync(out + '.tmp', out);
  console.log(`${stats} → ${out} (${Buffer.byteLength(text)} bytes)`);
} else {
  process.stdout.write(text);
  console.error(stats);
}
