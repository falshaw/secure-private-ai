#!/usr/bin/env node
/**
 * LLM bridge — authenticated gateway in front of the Claude Code CLI + shared RAG.
 *
 * A single HTTP service that:
 *  1. Routes LLM calls from containers to `claude -p` (the CLI's own login is
 *     the only credential; callers never hold a model API key).
 *  2. Provides a shared vector store interface on top of Qdrant for any
 *     app to upsert/search/delete embedded chunks.
 *
 * Embeddings: in-process ONNX model via @huggingface/transformers (default),
 * or Ollama (EMBED_PROVIDER=ollama). 768-dim by default.
 *
 * Run: `node bridge/claude-bridge.mjs` (or via bridge/docker-compose.yml)
 *
 * Endpoints:
 *   GET    /health            → { ok, claude, qdrant, ollama, ... }   (the ONLY unauthenticated route)
 *   GET    /metrics           → Prometheus text format   (requires X-Bridge-Token)
 *   POST   /generate          → body { prompt, system?, model?, rag?, cache?, fallbackOnQuotaExhausted? }
 *   POST   /rag/upsert        → body { collection, id?, text, payload?, chunk? }
 *   POST   /rag/upsert-batch  → body { collection, items: [{id?, text, payload?}], chunk? }
 *   POST   /rag/search        → body { collection, query, topK?, filter? }
 *   DELETE /rag/:collection/:id → delete a single point
 *   GET    /rag/collections   → list collections (with embed model metadata)
 *   POST   /admin/reload      → reload allowlist + state from disk
 *   GET    /log/tail?n=&app=  → last N request-log records (newest first)
 *
 * Headers (optional but recommended on every call):
 *   X-App-Name:      calling app name (e.g. "crm") — appears in logs + metrics
 *   X-Request-Id:    caller-supplied id; bridge generates one otherwise; echoed back
 *   X-Bridge-Token:  shared secret if CLAUDE_BRIDGE_TOKEN env is set
 *
 * Request log (local-only usage/audit trail of /generate calls):
 *   BRIDGE_REQUEST_LOG=true|false           on by default
 *   BRIDGE_REQUEST_LOG_DIR=<path>           default bridge/.bridge-logs/ (gitignored)
 *   BRIDGE_REQUEST_LOG_RETENTION_DAYS=30    daily files pruned past this
 *   BRIDGE_REQUEST_LOG_CONTENT=false        true = also store truncated prompt/response text
 *   BRIDGE_REQUEST_LOG_CONTENT_MAX=500      truncation length when content logging is on
 */

import { createServer } from "node:http";
import { spawn, execFile } from "node:child_process";
import { promisify } from "node:util";
import { randomUUID, createHash } from "node:crypto";
import {
  existsSync,
  readFileSync,
  writeFileSync,
  appendFileSync,
  mkdirSync,
  readdirSync,
  unlinkSync,
} from "node:fs";
import { dirname, resolve as pathResolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createRequire } from "node:module";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const execFileP = promisify(execFile);

// =============================================================================
// Config (env)
// =============================================================================

const HOST = process.env.CLAUDE_BRIDGE_HOST ?? "127.0.0.1";
const PORT = Number(process.env.CLAUDE_BRIDGE_PORT ?? "7777");
const CLAUDE_BIN = process.env.CLAUDE_BIN ?? "claude";
const MAX_BODY_BYTES = 1024 * 1024;
const SPAWN_TIMEOUT_MS = Number(process.env.SPAWN_TIMEOUT_MS ?? "120000");

// Shared RAG infra
const OLLAMA_URL = process.env.OLLAMA_URL ?? "http://127.0.0.1:11434";
const QDRANT_URL = process.env.QDRANT_URL ?? "http://127.0.0.1:6333";
const QDRANT_API_KEY = process.env.QDRANT_API_KEY ?? ""; // when set, sent as `api-key` header on every Qdrant request
const EMBED_MODEL = process.env.EMBED_MODEL ?? "nomic-embed-text";
const EMBED_DIM = Number(process.env.EMBED_DIM ?? "768");

// Embedder provider. Default is "local": an in-process ONNX embedder via
// @huggingface/transformers - no server, no GPU, no cost.
//   local  - @huggingface/transformers feature-extraction (default)
//   ollama - HTTP embeddings from an Ollama server at OLLAMA_URL
//   none   - disabled; /rag/* and /v1/embeddings fail with a clear message
const EMBED_PROVIDER = (process.env.EMBED_PROVIDER ?? "local").toLowerCase();
// Must stay in the same model family as whatever built the existing Qdrant
// collections. Ollama's `nomic-embed-text` tag is nomic-embed-text-v1.5, so
// that is the default here (vectors from either provider stay comparable) - a different model yields vectors of the right
// LENGTH that are not comparable with the stored ones, which degrades search
// silently rather than erroring. See the dim guard in getEmbedding().
const EMBED_MODEL_LOCAL =
  process.env.EMBED_MODEL_LOCAL ?? "nomic-ai/nomic-embed-text-v1.5";
// Where transformers.js caches downloaded weights. Defaults beside the bridge
// state so the one-off ~500 MB download survives container recreation.
const EMBED_CACHE_DIR =
  process.env.EMBED_CACHE_DIR ?? pathResolve(__dirname, ".embed-cache");
// Directory the embedder package is installed into (the bridge's /app mount is
// read-only and carries no node_modules of its own).
const EMBED_PKG_DIR = process.env.EMBED_PKG_DIR ?? "/opt/embedder";
// The model that ACTUALLY produced a vector. EMBED_MODEL alone is not it: with
// EMBED_PROVIDER=local the vector comes from EMBED_MODEL_LOCAL, so recording or
// comparing EMBED_MODEL would report "no change" across an embedder swap and the
// collection.embed.mismatch warning below would never fire - which is the exact
// failure it exists to catch (same dim, different model, silent nonsense).
const EMBED_MODEL_EFFECTIVE =
  EMBED_PROVIDER === "ollama"
    ? EMBED_MODEL
    : EMBED_PROVIDER === "local"
      ? `local:${EMBED_MODEL_LOCAL}`
      : "none";

// Optional features
const BRIDGE_TOKEN = process.env.CLAUDE_BRIDGE_TOKEN || null; // shared secret; empty/unset => disabled (|| not ?? so "" isn't a 0-length enforced token)
const RAG_ENV = (process.env.RAG_ENV ?? "").toLowerCase(); // "" = no prefix
const ALLOWLIST_FILE =
  process.env.CLAUDE_BRIDGE_ALLOWLIST_FILE ??
  pathResolve(__dirname, "bridge-allowlist.json");
const STATE_FILE =
  process.env.BRIDGE_STATE_FILE ??
  pathResolve(__dirname, ".bridge-state.json");
const CACHE_MAX_ENTRIES = Number(process.env.CACHE_MAX_ENTRIES ?? "256");
const CACHE_TTL_MS = Number(process.env.CACHE_TTL_MS ?? String(60 * 60_000));
const RETRY_TRANSIENT =
  (process.env.RETRY_TRANSIENT ?? "true").toLowerCase() !== "false";
const RETRY_DELAY_MS = Number(process.env.RETRY_DELAY_MS ?? "2000");
const OLLAMA_FALLBACK_MODEL =
  process.env.OLLAMA_FALLBACK_MODEL ?? "qwen2.5:7b";
const CHUNK_TARGET_CHARS = Number(process.env.CHUNK_TARGET_CHARS ?? "2000"); // ~500 tokens
const CHUNK_OVERLAP_CHARS = Number(process.env.CHUNK_OVERLAP_CHARS ?? "200"); // ~50 tokens

// Request log — append-only JSONL, daily-rotated, LOCAL-ONLY usage/audit trail.
// Metadata-first by design: prompt/response TEXT is NOT stored unless
// BRIDGE_REQUEST_LOG_CONTENT=true (and even then it's truncated). Lives under
// bridge/.bridge-logs/ (gitignored) — never committed, never leaves the host.
const REQUEST_LOG_ENABLED =
  (process.env.BRIDGE_REQUEST_LOG ?? "true").toLowerCase() !== "false";
const REQUEST_LOG_DIR =
  process.env.BRIDGE_REQUEST_LOG_DIR ?? pathResolve(__dirname, ".bridge-logs");
const REQUEST_LOG_RETENTION_DAYS = Number(
  process.env.BRIDGE_REQUEST_LOG_RETENTION_DAYS ?? "30",
);
const REQUEST_LOG_CONTENT =
  (process.env.BRIDGE_REQUEST_LOG_CONTENT ?? "false").toLowerCase() === "true";
const REQUEST_LOG_CONTENT_MAX = Number(
  process.env.BRIDGE_REQUEST_LOG_CONTENT_MAX ?? "500",
);

const MINIMAL_FLAGS = [
  "-p",
  "--output-format", "text",
  "--tools", "",
  "--strict-mcp-config",
  "--mcp-config", '{"mcpServers":{}}',
  "--disable-slash-commands",
  "--no-session-persistence",
  "--exclude-dynamic-system-prompt-sections",
];
const VALID_MODELS = new Set(["haiku", "sonnet", "opus"]);

// =============================================================================
// Logging
// =============================================================================

function log(level, msg, extra = {}) {
  process.stdout.write(
    JSON.stringify({ t: new Date().toISOString(), level, msg, ...extra }) + "\n",
  );
}

// =============================================================================
// Request log (append-only JSONL — local-only usage/audit trail)
// One line per resolved /generate call. Metadata-first: prompt/response are
// stored only as char-count + a short sha (BRIDGE_REQUEST_LOG_CONTENT=true adds
// truncated text). Daily file requests-YYYY-MM-DD.jsonl, pruned past retention.
// =============================================================================

function sha12(s) {
  return createHash("sha256").update(s).digest("hex").slice(0, 12);
}

let lastLogPrune = 0;

function pruneOldLogs() {
  try {
    if (!existsSync(REQUEST_LOG_DIR)) return;
    const cutoff = Date.now() - REQUEST_LOG_RETENTION_DAYS * 86_400_000;
    for (const f of readdirSync(REQUEST_LOG_DIR)) {
      const m = /^requests-(\d{4}-\d{2}-\d{2})\.jsonl$/.exec(f);
      if (!m) continue;
      const day = Date.parse(`${m[1]}T00:00:00Z`);
      if (!Number.isNaN(day) && day < cutoff) {
        unlinkSync(pathResolve(REQUEST_LOG_DIR, f));
      }
    }
  } catch (err) {
    log("warn", "requestlog.prune.failed", { error: err.message });
  }
}

function buildGenRecord(ctx, o) {
  const rec = {
    ts: new Date().toISOString(),
    requestId: ctx.requestId,
    route: "/generate",
    app: ctx.appName,
    model: o.model ?? "default",
    status: o.status, // ok | error | cache
    durationMs: o.durationMs ?? 0,
    promptChars: o.prompt?.length ?? 0,
    promptSha12: o.prompt ? sha12(o.prompt) : null,
    responseChars: o.response?.length ?? 0,
    ragChunks: o.ragChunksUsed ?? 0,
    collection: o.usedCollection ?? null,
    cached: o.cached ?? false,
  };
  if (o.fellback) rec.fellback = true;
  if (o.error) rec.error = String(o.error).slice(0, 300);
  if (REQUEST_LOG_CONTENT) {
    rec.prompt = (o.prompt ?? "").slice(0, REQUEST_LOG_CONTENT_MAX);
    rec.response = (o.response ?? "").slice(0, REQUEST_LOG_CONTENT_MAX);
  }
  return rec;
}

function writeRequestLog(rec) {
  if (!REQUEST_LOG_ENABLED) return;
  try {
    mkdirSync(REQUEST_LOG_DIR, { recursive: true });
    const file = pathResolve(REQUEST_LOG_DIR, `requests-${rec.ts.slice(0, 10)}.jsonl`);
    appendFileSync(file, JSON.stringify(rec) + "\n", "utf8");
    if (Date.now() - lastLogPrune > 3_600_000) {
      lastLogPrune = Date.now();
      pruneOldLogs();
    }
  } catch (err) {
    log("warn", "requestlog.write.failed", { error: err.message });
  }
}

function tailRequestLog(n, appFilter) {
  const out = [];
  try {
    if (!existsSync(REQUEST_LOG_DIR)) return out;
    const files = readdirSync(REQUEST_LOG_DIR)
      .filter((f) => /^requests-\d{4}-\d{2}-\d{2}\.jsonl$/.test(f))
      .sort()
      .reverse(); // newest day first
    for (const f of files) {
      const content = readFileSync(pathResolve(REQUEST_LOG_DIR, f), "utf8").trim();
      if (!content) continue;
      const recs = content.split("\n");
      for (let i = recs.length - 1; i >= 0; i--) {
        let obj;
        try {
          obj = JSON.parse(recs[i]);
        } catch {
          continue;
        }
        if (appFilter && obj.app !== appFilter) continue;
        out.push(obj);
        if (out.length >= n) return out;
      }
    }
  } catch (err) {
    log("warn", "requestlog.tail.failed", { error: err.message });
  }
  return out;
}

// =============================================================================
// State file (collection embedding metadata)
// =============================================================================

let state = { collections: {} };

function loadState() {
  try {
    if (existsSync(STATE_FILE)) {
      state = JSON.parse(readFileSync(STATE_FILE, "utf8"));
      if (!state.collections) state.collections = {};
    }
  } catch (err) {
    log("warn", "state.load.failed", { error: err.message });
  }
}
function saveState() {
  try {
    writeFileSync(STATE_FILE, JSON.stringify(state, null, 2), "utf8");
  } catch (err) {
    log("warn", "state.save.failed", { error: err.message });
  }
}
loadState();

async function backfillCollectionState(attempt = 1) {
  // On startup, find any pre-existing Qdrant collections that have no
  // metadata in our state file and record the current embed model/dim.
  // We cannot know what model originally embedded them — but recording the
  // current values means /rag/collections returns consistent metadata and
  // any later mismatch (after the embedder changes) gets flagged via the
  // collection.embed.mismatch warning.
  // Qdrant may be a separate stack that comes up alongside this one after a
  // Docker daemon restart, so the first attempt can lose the race. Retry a
  // few times before giving up.
  const MAX_ATTEMPTS = 6;
  const RETRY_MS = 15_000;
  try {
    const cols = await qdrantListCollections();
    let added = 0;
    for (const c of cols) {
      const name = c?.name;
      if (!name || state.collections[name]) continue;
      state.collections[name] = {
        embedModel: EMBED_MODEL_EFFECTIVE,
        embedDim: EMBED_DIM,
        createdAt: new Date().toISOString(),
        note: "pre-existing collection — embedding model assumed at backfill",
      };
      added += 1;
    }
    if (added > 0) saveState();
    log("info", "state.backfill", {
      newCollections: added,
      total: cols.length,
    });
  } catch (err) {
    if (attempt < MAX_ATTEMPTS) {
      log("info", "state.backfill.retry", {
        error: err.message,
        attempt,
        nextInMs: RETRY_MS,
      });
      setTimeout(() => void backfillCollectionState(attempt + 1), RETRY_MS).unref();
      return;
    }
    log("warn", "state.backfill.failed", { error: err.message, attempts: attempt });
  }
}

// =============================================================================
// Allowlist file (per-app model + rate limit policy)
// Format: { "crm":        { models: ["haiku","sonnet","opus"], maxRequestsPerHour: 100 },
//          "*":          { models: ["haiku","sonnet"] } }
// =============================================================================

let allowlist = {};
function loadAllowlist() {
  try {
    if (existsSync(ALLOWLIST_FILE)) {
      allowlist = JSON.parse(readFileSync(ALLOWLIST_FILE, "utf8"));
      log("info", "allowlist.loaded", {
        entries: Object.keys(allowlist).length,
        path: ALLOWLIST_FILE,
      });
    } else {
      log("info", "allowlist.absent", {
        path: ALLOWLIST_FILE,
        note: "no per-app policy enforced",
      });
    }
  } catch (err) {
    log("warn", "allowlist.load.failed", { error: err.message });
    allowlist = {};
  }
}
loadAllowlist();

// =============================================================================
// Per-app request counters (sliding 1-hour window)
// =============================================================================

const requestWindows = new Map(); // app -> [timestamp...]

function recordAppRequest(appName) {
  const now = Date.now();
  const cutoff = now - 60 * 60_000;
  const arr = requestWindows.get(appName) ?? [];
  while (arr.length && arr[0] < cutoff) arr.shift();
  arr.push(now);
  requestWindows.set(appName, arr);
  return arr.length;
}
function getAppCount(appName) {
  const now = Date.now();
  const cutoff = now - 60 * 60_000;
  const arr = requestWindows.get(appName) ?? [];
  while (arr.length && arr[0] < cutoff) arr.shift();
  return arr.length;
}
function checkAllowlist(appName, model) {
  const entry = allowlist[appName] ?? allowlist["*"];
  if (!entry) return { ok: true };
  if (entry.models && model && !entry.models.includes(model)) {
    return {
      ok: false,
      error: `app "${appName}" not allowed model "${model}". Allowed: ${entry.models.join(", ")}`,
    };
  }
  if (
    typeof entry.maxRequestsPerHour === "number" &&
    entry.maxRequestsPerHour > 0
  ) {
    const count = getAppCount(appName);
    if (count >= entry.maxRequestsPerHour) {
      return {
        ok: false,
        error: `app "${appName}" exceeded rate limit ${count}/${entry.maxRequestsPerHour} per hour`,
      };
    }
  }
  return { ok: true };
}

// =============================================================================
// LRU cache (opt-in via cache:true in body)
// =============================================================================

const cache = new Map(); // key -> { value, expiresAt }

function cacheKey({ model, system, prompt, rag }) {
  const h = createHash("sha256");
  h.update(model ?? "default");
  h.update("\x1f");
  h.update(system ?? "");
  h.update("\x1f");
  h.update(prompt);
  h.update("\x1f");
  if (rag) {
    h.update(rag.collection ?? "");
    h.update("\x1f");
    h.update(String(rag.topK ?? 5));
  }
  return h.digest("hex");
}
function cacheGet(key) {
  const ent = cache.get(key);
  if (!ent) return null;
  if (Date.now() > ent.expiresAt) {
    cache.delete(key);
    return null;
  }
  cache.delete(key);
  cache.set(key, ent); // LRU bump
  return ent.value;
}
function cacheSet(key, value) {
  cache.set(key, { value, expiresAt: Date.now() + CACHE_TTL_MS });
  if (cache.size > CACHE_MAX_ENTRIES) {
    const oldest = cache.keys().next().value;
    cache.delete(oldest);
  }
}

// =============================================================================
// Metrics (Prometheus text format via /metrics)
// =============================================================================

const metrics = {
  requestsTotal: new Map(), // "method,path,status" -> count
  generateTotal: new Map(), // "app,model,status" -> count
  generateDurationMs: new Map(), // "app,model" -> [sumMs, count]
  cacheHits: 0,
  cacheMisses: 0,
  ragSearchesTotal: 0,
  ragSearchesFailed: 0,
  ragAugmentations: 0,
  ragChunksTotal: 0,
  ragUpsertsTotal: 0,
  ragReranks: 0,
  ragReranksFailed: 0,
  fallbacksOllama: 0,
  retriesTransient: 0,
  startedAt: Date.now(),
};

function incMap(m, key, n = 1) {
  m.set(key, (m.get(key) ?? 0) + n);
}
function recordGenerate(appName, model, status, durationMs) {
  incMap(metrics.generateTotal, `${appName},${model},${status}`);
  const dKey = `${appName},${model}`;
  const cur = metrics.generateDurationMs.get(dKey) ?? [0, 0];
  cur[0] += durationMs;
  cur[1] += 1;
  metrics.generateDurationMs.set(dKey, cur);
}

function escapeLabel(s) {
  return String(s).replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, "\\n");
}
function renderPromText() {
  const out = [];
  const push = (l) => out.push(l);
  push("# HELP claude_bridge_uptime_seconds Bridge uptime");
  push("# TYPE claude_bridge_uptime_seconds gauge");
  push(`claude_bridge_uptime_seconds ${(Date.now() - metrics.startedAt) / 1000}`);

  push("# HELP claude_bridge_requests_total Total HTTP requests");
  push("# TYPE claude_bridge_requests_total counter");
  for (const [k, v] of metrics.requestsTotal) {
    const [method, path, status] = k.split(",");
    push(
      `claude_bridge_requests_total{method="${escapeLabel(method)}",path="${escapeLabel(path)}",status="${escapeLabel(status)}"} ${v}`,
    );
  }

  push("# HELP claude_bridge_generate_total /generate calls");
  push("# TYPE claude_bridge_generate_total counter");
  for (const [k, v] of metrics.generateTotal) {
    const [app, model, status] = k.split(",");
    push(
      `claude_bridge_generate_total{app="${escapeLabel(app)}",model="${escapeLabel(model)}",status="${escapeLabel(status)}"} ${v}`,
    );
  }

  push("# HELP claude_bridge_generate_duration_ms_avg Avg /generate duration ms");
  push("# TYPE claude_bridge_generate_duration_ms_avg gauge");
  for (const [k, [sum, count]] of metrics.generateDurationMs) {
    const [app, model] = k.split(",");
    const avg = count > 0 ? sum / count : 0;
    push(
      `claude_bridge_generate_duration_ms_avg{app="${escapeLabel(app)}",model="${escapeLabel(model)}"} ${avg.toFixed(2)}`,
    );
  }

  push("# HELP claude_bridge_cache_hits_total Cache hits");
  push("# TYPE claude_bridge_cache_hits_total counter");
  push(`claude_bridge_cache_hits_total ${metrics.cacheHits}`);
  push("# HELP claude_bridge_cache_misses_total Cache misses");
  push("# TYPE claude_bridge_cache_misses_total counter");
  push(`claude_bridge_cache_misses_total ${metrics.cacheMisses}`);

  push("# HELP claude_bridge_rag_searches_total RAG searches");
  push("# TYPE claude_bridge_rag_searches_total counter");
  push(`claude_bridge_rag_searches_total ${metrics.ragSearchesTotal}`);
  push("# HELP claude_bridge_rag_searches_failed_total RAG search failures");
  push("# TYPE claude_bridge_rag_searches_failed_total counter");
  push(`claude_bridge_rag_searches_failed_total ${metrics.ragSearchesFailed}`);
  push("# HELP claude_bridge_rag_upserts_total RAG upserts");
  push("# TYPE claude_bridge_rag_upserts_total counter");
  push(`claude_bridge_rag_upserts_total ${metrics.ragUpsertsTotal}`);
  push("# HELP claude_bridge_rag_chunks_emitted_total Chunks emitted by auto-chunker");
  push("# TYPE claude_bridge_rag_chunks_emitted_total counter");
  push(`claude_bridge_rag_chunks_emitted_total ${metrics.ragChunksTotal}`);

  push("# HELP claude_bridge_rag_reranks_total Reranks performed via Claude");
  push("# TYPE claude_bridge_rag_reranks_total counter");
  push(`claude_bridge_rag_reranks_total ${metrics.ragReranks}`);
  push("# HELP claude_bridge_rag_reranks_failed_total Rerank failures (fell back to dense order)");
  push("# TYPE claude_bridge_rag_reranks_failed_total counter");
  push(`claude_bridge_rag_reranks_failed_total ${metrics.ragReranksFailed}`);

  push("# HELP claude_bridge_ollama_fallbacks_total Quota-fallback to ollama");
  push("# TYPE claude_bridge_ollama_fallbacks_total counter");
  push(`claude_bridge_ollama_fallbacks_total ${metrics.fallbacksOllama}`);
  push("# HELP claude_bridge_retries_total Transient retries triggered");
  push("# TYPE claude_bridge_retries_total counter");
  push(`claude_bridge_retries_total ${metrics.retriesTransient}`);

  push("# HELP claude_bridge_app_requests_hour Requests in last hour per app");
  push("# TYPE claude_bridge_app_requests_hour gauge");
  for (const [app] of requestWindows) {
    push(`claude_bridge_app_requests_hour{app="${escapeLabel(app)}"} ${getAppCount(app)}`);
  }

  push("# HELP claude_bridge_cache_size Current LRU cache size");
  push("# TYPE claude_bridge_cache_size gauge");
  push(`claude_bridge_cache_size ${cache.size}`);

  return out.join("\n") + "\n";
}

// =============================================================================
// UUID / ID helpers
// =============================================================================

const UUID_NS_BYTES = Buffer.from(process.env.RAG_UUID_NAMESPACE ?? "llm-bridge-rag", "utf8");
function stringToUuid(s) {
  const h = createHash("sha256").update(UUID_NS_BYTES).update(s).digest();
  const hex = h.subarray(0, 16).toString("hex");
  return [
    hex.slice(0, 8),
    hex.slice(8, 12),
    "5" + hex.slice(13, 16),
    ((parseInt(hex.slice(16, 18), 16) & 0x3f) | 0x80).toString(16).padStart(2, "0") +
      hex.slice(18, 20),
    hex.slice(20, 32),
  ].join("-");
}
function contentHashId(text) {
  return stringToUuid(`content:${text}`);
}
function normalizePointId(externalId, text) {
  if (!externalId) {
    // Dedup-by-content-hash so re-upsert of same text doesn't duplicate
    return { qdrantId: contentHashId(text ?? ""), externalId: null };
  }
  if (
    /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(externalId)
  ) {
    return { qdrantId: externalId, externalId: null };
  }
  if (/^\d+$/.test(externalId)) {
    return { qdrantId: Number(externalId), externalId: null };
  }
  return { qdrantId: stringToUuid(externalId), externalId };
}

// =============================================================================
// Auto-chunker (sentence-window splitter)
// =============================================================================

function chunkText(text, targetChars = CHUNK_TARGET_CHARS, overlapChars = CHUNK_OVERLAP_CHARS) {
  if (text.length <= targetChars) return [text];
  const chunks = [];
  let i = 0;
  while (i < text.length) {
    let end = Math.min(i + targetChars, text.length);
    if (end < text.length) {
      const lookback = 200;
      const winStart = Math.max(end - lookback, i);
      const window = text.slice(winStart, end);
      // last sentence boundary in window
      const matches = [...window.matchAll(/[.!?]\s+(?=[A-Z])/g)];
      if (matches.length > 0) {
        const m = matches[matches.length - 1];
        const cut = winStart + m.index + m[0].length;
        if (cut > i + Math.floor(targetChars / 2)) end = cut;
      }
    }
    const piece = text.slice(i, end).trim();
    if (piece) chunks.push(piece);
    if (end >= text.length) break;
    i = Math.max(end - overlapChars, i + 1);
  }
  return chunks;
}

// =============================================================================
// Claude CLI invocation
// =============================================================================

async function getClaudeVersion() {
  try {
    const { stdout } = await execFileP(CLAUDE_BIN, ["--version"], { timeout: 5000 });
    return stdout.trim();
  } catch (err) {
    return `unavailable: ${err instanceof Error ? err.message : String(err)}`;
  }
}

function isTransientError(stderr) {
  const lc = (stderr ?? "").toLowerCase();
  return /\b(429|529|503|overload|rate.?limit|temporarily unavail|service unavail)\b/.test(lc);
}

const ALLOWED_RESEARCH_TOOLS = new Set(["WebSearch", "WebFetch"]);
const RESEARCH_SPAWN_TIMEOUT_MS = Number(
  process.env.RESEARCH_SPAWN_TIMEOUT_MS ?? "300000",
);

function callClaudeOnce({ prompt, system, model, tools }) {
  return new Promise((resolve) => {
    const args = [...MINIMAL_FLAGS];
    const safeTools = Array.isArray(tools)
      ? tools.filter((t) => ALLOWED_RESEARCH_TOOLS.has(t))
      : [];
    if (safeTools.length > 0) {
      const i = args.indexOf("--tools");
      if (i >= 0) args[i + 1] = safeTools.join(",");
      // --tools makes the tools available; --allowedTools grants permission to
      // actually use them in non-interactive (-p) mode. Without this the CLI
      // silently skips WebSearch/WebFetch whenever no ambient .claude/ project
      // permissions are inherited from the CWD — e.g. when the bridge runs
      // containerized, outside the repo tree.
      args.push("--allowedTools", safeTools.join(","));
    }
    if (system) args.push("--system-prompt", system);
    if (model && VALID_MODELS.has(model)) args.push("--model", model);
    const spawnTimeoutMs = safeTools.length > 0
      ? RESEARCH_SPAWN_TIMEOUT_MS
      : SPAWN_TIMEOUT_MS;

    const t0 = Date.now();
    let stdout = "";
    let stderr = "";
    let settled = false;

    const proc = spawn(CLAUDE_BIN, args, {
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
    });

    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      try { proc.kill("SIGTERM"); } catch {}
      resolve({
        ok: false,
        transient: false,
        error: `claude timed out after ${spawnTimeoutMs}ms`,
        durationMs: Date.now() - t0,
      });
    }, spawnTimeoutMs);

    proc.stdout.on("data", (b) => { stdout += b.toString("utf8"); });
    proc.stderr.on("data", (b) => { stderr += b.toString("utf8"); });

    proc.on("error", (err) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve({
        ok: false,
        transient: false,
        error: `spawn failed: ${err.message}`,
        durationMs: Date.now() - t0,
      });
    });

    proc.on("close", (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      const durationMs = Date.now() - t0;
      if (code !== 0) {
        const transient = isTransientError(stderr);
        resolve({
          ok: false,
          transient,
          error: `claude exited ${code}: ${stderr.trim().slice(0, 800) || "(no stderr)"}`,
          durationMs,
        });
        return;
      }
      resolve({
        ok: true,
        response: stdout,
        model: model ?? "default",
        durationMs,
      });
    });

    try {
      proc.stdin.end(prompt, "utf8");
    } catch (err) {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve({
        ok: false,
        transient: false,
        error: `stdin write failed: ${err.message}`,
        durationMs: Date.now() - t0,
      });
    }
  });
}

async function callClaude({ prompt, system, model, tools }) {
  const r = await callClaudeOnce({ prompt, system, model, tools });
  if (r.ok || !r.transient || !RETRY_TRANSIENT) return r;
  metrics.retriesTransient += 1;
  log("info", "claude.retry.transient", { afterMs: RETRY_DELAY_MS });
  await new Promise((res) => setTimeout(res, RETRY_DELAY_MS));
  const r2 = await callClaudeOnce({ prompt, system, model, tools });
  if (r2.ok) return r2;
  return { ...r2, error: `retry-after-transient: ${r2.error}` };
}

async function claudeRerank(query, candidates, topK) {
  // candidates: array of objects each carrying either `text` (top-level)
  // or `payload.text`. Returns { reranked: [...], ok: bool }.
  // On failure (claude error, malformed response) the caller's dense order
  // is preserved — never throws.
  if (!Array.isArray(candidates) || candidates.length <= topK) {
    return { reranked: candidates ?? [], ok: true };
  }
  const getText = (c) => c?.payload?.text ?? c?.text ?? "";
  const numbered = candidates
    .map((c, i) => `[${i}] ${getText(c).slice(0, 600)}`)
    .join("\n\n");
  const prompt =
    `You are a relevance ranker. Below is a user query and ${candidates.length} ` +
    `numbered candidate passages. Return ONLY a JSON array of the ${topK} ` +
    `most relevant candidate indices, in order from best to worst.\n\n` +
    `Query: ${query}\n\nCandidates:\n${numbered}\n\n` +
    `Output a single JSON array like [3,1,7]. No prose, no markdown, no explanation.`;
  const r = await callClaudeOnce({ prompt, model: "haiku" });
  if (!r.ok) return { reranked: candidates.slice(0, topK), ok: false };
  const m = (r.response ?? "").match(/\[\s*\d+(?:\s*,\s*\d+)*\s*\]/);
  if (!m) return { reranked: candidates.slice(0, topK), ok: false };
  try {
    const indices = JSON.parse(m[0]);
    if (!Array.isArray(indices)) {
      return { reranked: candidates.slice(0, topK), ok: false };
    }
    const seen = new Set();
    const picked = [];
    for (const i of indices) {
      if (!Number.isInteger(i)) continue;
      if (i < 0 || i >= candidates.length) continue;
      if (seen.has(i)) continue;
      seen.add(i);
      picked.push(candidates[i]);
      if (picked.length >= topK) break;
    }
    if (picked.length === 0) {
      return { reranked: candidates.slice(0, topK), ok: false };
    }
    return { reranked: picked, ok: true };
  } catch {
    return { reranked: candidates.slice(0, topK), ok: false };
  }
}

async function callOllamaFallback({ prompt, system }) {
  // Without this guard, a quota fallback with no Ollama configured burns a
  // full SPAWN_TIMEOUT_MS connecting to nothing and masks the real 429 behind
  // a connection error.
  if (EMBED_PROVIDER !== "ollama" && !process.env.OLLAMA_URL) {
    return {
      ok: false,
      error:
        "ollama fallback unavailable - no OLLAMA_URL is set; " +
        "the original quota error stands",
    };
  }
  try {
    const messages = [];
    if (system) messages.push({ role: "system", content: system });
    messages.push({ role: "user", content: prompt });
    const t0 = Date.now();
    const res = await fetch(`${OLLAMA_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: OLLAMA_FALLBACK_MODEL,
        messages,
        stream: false,
      }),
      signal: AbortSignal.timeout(SPAWN_TIMEOUT_MS),
    });
    if (!res.ok) {
      const t = await res.text().catch(() => "");
      return {
        ok: false,
        error: `ollama fallback HTTP ${res.status}: ${t.slice(0, 200)}`,
        durationMs: Date.now() - t0,
      };
    }
    const data = await res.json();
    return {
      ok: true,
      response: data?.message?.content ?? "",
      model: `ollama-${OLLAMA_FALLBACK_MODEL}`,
      durationMs: Date.now() - t0,
      fellbackToOllama: true,
    };
  } catch (err) {
    return { ok: false, error: `ollama fallback failed: ${err.message}` };
  }
}

// =============================================================================
// RAG helpers (Ollama embeddings + Qdrant)
// =============================================================================

function resolveCollectionName(name) {
  if (RAG_ENV && RAG_ENV !== "default") return `${RAG_ENV}-${name}`;
  return name;
}

// Lazily loaded in-process ONNX embedder. Resolved from EMBED_PKG_DIR because
// the bridge code carries no node_modules of its own; the package is installed
// into the image instead (see bridge/Dockerfile and bridge/embedder/).
let localEmbedderPromise = null;
async function getLocalEmbedder() {
  if (localEmbedderPromise) return localEmbedderPromise;
  localEmbedderPromise = (async () => {
    let mod;
    try {
      const requireFrom = createRequire(import.meta.url);
      const entry = requireFrom.resolve("@huggingface/transformers", {
        paths: [EMBED_PKG_DIR],
      });
      mod = await import(pathToFileURL(entry).href);
    } catch (err) {
      throw new Error(
        "local embedder unavailable: @huggingface/transformers not resolvable from " +
          `${EMBED_PKG_DIR} (${err.message}). Rebuild the bridge image ` +
          "(cd bridge && docker compose build), or set EMBED_PROVIDER=none.",
      );
    }
    mod.env.cacheDir = EMBED_CACHE_DIR;
    log("info", "embed.local.loading", {
      model: EMBED_MODEL_LOCAL,
      cacheDir: EMBED_CACHE_DIR,
    });
    const extractor = await mod.pipeline("feature-extraction", EMBED_MODEL_LOCAL);
    log("info", "embed.local.ready", { model: EMBED_MODEL_LOCAL });
    return extractor;
  })().catch((err) => {
    localEmbedderPromise = null; // let a later call retry instead of caching the failure
    throw err;
  });
  return localEmbedderPromise;
}

async function embedViaLocal(text) {
  const extractor = await getLocalEmbedder();
  const out = await extractor(text, { pooling: "mean", normalize: true });
  const v = Array.from(out?.data ?? []);
  if (v.length === 0) throw new Error("local embedder returned empty vector");
  return v;
}

async function embedViaOllama(text) {
  const res = await fetch(`${OLLAMA_URL}/api/embed`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: EMBED_MODEL, input: text }),
    signal: AbortSignal.timeout(30_000),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`embed HTTP ${res.status}: ${body.slice(0, 200)}`);
  }
  const data = await res.json();
  const v = data?.embeddings?.[0];
  if (!Array.isArray(v) || v.length === 0) {
    throw new Error("embed returned empty vector");
  }
  return v;
}

async function getEmbedding(text) {
  if (EMBED_PROVIDER === "none") {
    throw new Error(
      "embeddings are disabled (EMBED_PROVIDER=none) - /rag/* cannot run",
    );
  }
  const v =
    EMBED_PROVIDER === "ollama"
      ? await embedViaOllama(text)
      : await embedViaLocal(text);
  // A wrong-LENGTH vector is rejected by Qdrant anyway; a right-length vector
  // from the WRONG MODEL is accepted and then retrieves nonsense forever. Catch
  // the first case loudly here and name the second in the message.
  if (v.length !== EMBED_DIM) {
    throw new Error(
      `embedding dim ${v.length} != EMBED_DIM ${EMBED_DIM} (provider=${EMBED_PROVIDER}, ` +
        `model=${EMBED_PROVIDER === "ollama" ? EMBED_MODEL : EMBED_MODEL_LOCAL}). ` +
        `Existing collections were built at ${EMBED_DIM}-dim: either restore a matching ` +
        "model or re-index every collection.",
    );
  }
  return v;
}

async function qdrantRequest(method, path, body) {
  const headers = {};
  if (body) headers["Content-Type"] = "application/json";
  if (QDRANT_API_KEY) headers["api-key"] = QDRANT_API_KEY; // auth-on when key configured
  const res = await fetch(`${QDRANT_URL}${path}`, {
    method,
    headers: Object.keys(headers).length ? headers : undefined,
    body: body ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(30_000),
  });
  const text = await res.text();
  if (!res.ok) {
    throw new Error(`qdrant ${method} ${path} HTTP ${res.status}: ${text.slice(0, 300)}`);
  }
  try { return JSON.parse(text); } catch { return { raw: text }; }
}

// Warn once per collection per process when the vectors in Qdrant were written
// by a different embedder than the one running now. Called from BOTH the upsert
// and the search path, because querying a stale collection is the case where
// wrong vectors actually mislead someone.
const _embedMetaWarned = new Set();
function checkCollectionEmbedMeta(name) {
  const meta = state.collections[name];
  if (!meta) return;
  if (meta.embedModel === EMBED_MODEL_EFFECTIVE && meta.embedDim === EMBED_DIM) return;
  if (_embedMetaWarned.has(name)) return;
  _embedMetaWarned.add(name);
  log("warn", "collection.embed.mismatch", {
    collection: name,
    recorded: { model: meta.embedModel, dim: meta.embedDim },
    current: { model: EMBED_MODEL_EFFECTIVE, dim: EMBED_DIM },
    note: "queries may return nonsense — re-index required",
  });
}

const _ensuredCollections = new Set();
async function qdrantEnsureCollection(name) {
  if (_ensuredCollections.has(name)) return;
  let exists = false;
  try {
    await qdrantRequest("GET", `/collections/${encodeURIComponent(name)}`);
    exists = true;
  } catch { /* not found */ }
  if (exists) {
    _ensuredCollections.add(name);
    const meta = state.collections[name];
    if (meta) {
      checkCollectionEmbedMeta(name);
    } else {
      state.collections[name] = {
        embedModel: EMBED_MODEL_EFFECTIVE,
        embedDim: EMBED_DIM,
        createdAt: new Date().toISOString(),
        note: "pre-existing collection — embedding model assumed",
      };
      saveState();
    }
    return;
  }
  await qdrantRequest("PUT", `/collections/${encodeURIComponent(name)}`, {
    vectors: { size: EMBED_DIM, distance: "Cosine" },
  });
  state.collections[name] = {
    embedModel: EMBED_MODEL_EFFECTIVE,
    embedDim: EMBED_DIM,
    createdAt: new Date().toISOString(),
  };
  saveState();
  _ensuredCollections.add(name);
  log("info", "qdrant.collection.created", { name, dim: EMBED_DIM, model: EMBED_MODEL_EFFECTIVE });
}

async function qdrantUpsertBatch(collection, points) {
  return qdrantRequest(
    "PUT",
    `/collections/${encodeURIComponent(collection)}/points?wait=true`,
    { points },
  );
}
async function qdrantSearch(collection, vector, topK = 5, filter) {
  return qdrantRequest(
    "POST",
    `/collections/${encodeURIComponent(collection)}/points/search`,
    {
      vector,
      limit: topK,
      with_payload: true,
      ...(filter ? { filter } : {}),
    },
  );
}
async function qdrantDeletePoint(collection, id) {
  return qdrantRequest(
    "POST",
    `/collections/${encodeURIComponent(collection)}/points/delete?wait=true`,
    { points: [id] },
  );
}
async function qdrantDeleteCollection(name) {
  return qdrantRequest("DELETE", `/collections/${encodeURIComponent(name)}`);
}
async function qdrantListCollections() {
  const data = await qdrantRequest("GET", "/collections");
  return data?.result?.collections ?? [];
}
function isValidCollectionName(name) {
  return typeof name === "string" && /^[a-z0-9_-]{1,80}$/.test(name);
}

// =============================================================================
// HTTP helpers
// =============================================================================

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let len = 0;
    req.on("data", (c) => {
      len += c.length;
      if (len > MAX_BODY_BYTES) {
        reject(new Error(`body exceeds ${MAX_BODY_BYTES} bytes`));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

function send(res, status, payload, reqId) {
  const merged = { ...(reqId ? { requestId: reqId } : {}), ...payload };
  const body = JSON.stringify(merged);
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(body),
    ...(reqId ? { "X-Request-Id": reqId } : {}),
  });
  res.end(body);
}
function sendText(res, status, text, contentType = "text/plain; version=0.0.4; charset=utf-8") {
  res.writeHead(status, {
    "Content-Type": contentType,
    "Content-Length": Buffer.byteLength(text),
  });
  res.end(text);
}

async function parseJsonBody(req, res, reqId) {
  let bodyText;
  try {
    bodyText = await readBody(req);
  } catch (err) {
    send(res, 413, { ok: false, error: err.message }, reqId);
    return null;
  }
  if (bodyText === "") return {};
  try {
    return JSON.parse(bodyText);
  } catch {
    send(res, 400, { ok: false, error: "invalid JSON body" }, reqId);
    return null;
  }
}

function getRequestContext(req) {
  const hdr = (n) => {
    const v = req.headers[n];
    return Array.isArray(v) ? v[0] : v ?? null;
  };
  const incoming = hdr("x-request-id");
  // OpenAI-compatible clients can only set an API key — accept it as the
  // bridge token via Authorization: Bearer so /v1/* can be token-gated too.
  const auth = hdr("authorization");
  const bearer =
    typeof auth === "string" && auth.startsWith("Bearer ")
      ? auth.slice(7)
      : null;
  return {
    requestId: incoming || randomUUID(),
    appName: (hdr("x-app-name") || "unknown").toString().slice(0, 64),
    token: hdr("x-bridge-token") ?? bearer,
  };
}

function checkAuth(ctx) {
  if (BRIDGE_TOKEN === null) return true;
  if (typeof ctx.token !== "string") return false;
  if (ctx.token.length !== BRIDGE_TOKEN.length) return false;
  let mismatch = 0;
  for (let i = 0; i < ctx.token.length; i++) {
    mismatch |= ctx.token.charCodeAt(i) ^ BRIDGE_TOKEN.charCodeAt(i);
  }
  return mismatch === 0;
}

// =============================================================================
// Handlers
// =============================================================================

async function handleGenerate(req, res, ctx) {
  const body = await parseJsonBody(req, res, ctx.requestId);
  if (body === null) return;

  const prompt = typeof body.prompt === "string" ? body.prompt : "";
  if (prompt.length === 0) {
    return send(res, 400, { ok: false, error: "prompt is required" }, ctx.requestId);
  }
  const system = typeof body.system === "string" ? body.system : undefined;
  const modelRaw = typeof body.model === "string" ? body.model.toLowerCase() : undefined;
  const model = modelRaw && VALID_MODELS.has(modelRaw) ? modelRaw : undefined;
  const useCache = body.cache === true;
  const fallbackOnQuota = body.fallbackOnQuotaExhausted === true;
  const requestedTools = Array.isArray(body.tools)
    ? body.tools.filter(
        (t) => typeof t === "string" && ALLOWED_RESEARCH_TOOLS.has(t),
      )
    : [];

  const policy = checkAllowlist(ctx.appName, model);
  if (!policy.ok) {
    log("warn", "generate.denied.policy", {
      requestId: ctx.requestId,
      app: ctx.appName,
      model,
      reason: policy.error,
    });
    return send(res, 403, { ok: false, error: policy.error }, ctx.requestId);
  }
  recordAppRequest(ctx.appName);

  // RAG augmentation
  let ragChunksUsed = 0;
  let augmentedPrompt = prompt;
  let usedCollection = null;
  const rag = body.rag && typeof body.rag === "object" ? body.rag : null;
  if (rag && isValidCollectionName(rag.collection)) {
    usedCollection = resolveCollectionName(rag.collection);
    const topK = Number.isInteger(rag.topK)
      ? Math.min(Math.max(rag.topK, 1), 20)
      : 5;
    const rerankReq = rag.rerank === true;
    const fetchK = rerankReq ? Math.min(topK * 4, 50) : topK;
    try {
      const qVec = await getEmbedding(prompt);
      const sr = await qdrantSearch(usedCollection, qVec, fetchK, rag.filter);
      let hits = Array.isArray(sr?.result) ? sr.result : [];
      if (rerankReq && hits.length > topK) {
        const rr = await claudeRerank(prompt, hits, topK);
        hits = rr.reranked;
        if (rr.ok) {
          metrics.ragReranks += 1;
          log("info", "generate.rag.reranked", {
            requestId: ctx.requestId,
            app: ctx.appName,
            collection: usedCollection,
            candidates: fetchK,
            picked: hits.length,
          });
        } else {
          metrics.ragReranksFailed += 1;
        }
      } else if (hits.length > topK) {
        hits = hits.slice(0, topK);
      }
      ragChunksUsed = hits.length;
      metrics.ragSearchesTotal += 1;
      if (hits.length > 0) {
        metrics.ragAugmentations += 1;
        const block = hits
          .map((h, i) => {
            const text = h?.payload?.text ?? "";
            const meta =
              h?.payload &&
              Object.keys(h.payload).filter((k) => k !== "text").length > 0
                ? ` (${Object.entries(h.payload)
                    .filter(([k]) => k !== "text")
                    .map(
                      ([k, v]) =>
                        `${k}=${typeof v === "string" ? v : JSON.stringify(v)}`,
                    )
                    .join(", ")})`
                : "";
            return `[${i + 1}]${meta}\n${text}`;
          })
          .join("\n\n---\n\n");
        augmentedPrompt = `Knowledge (retrieved from "${usedCollection}", ${hits.length} chunks):\n\n${block}\n\n---\n\nUser question:\n${prompt}`;
        log("info", "generate.rag.augmented", {
          requestId: ctx.requestId,
          app: ctx.appName,
          collection: usedCollection,
          chunks: hits.length,
        });
      }
    } catch (err) {
      metrics.ragSearchesFailed += 1;
      log("warn", "generate.rag.failed", {
        requestId: ctx.requestId,
        app: ctx.appName,
        collection: usedCollection,
        error: err.message,
      });
    }
  }

  // Cache lookup
  const cKey = useCache
    ? cacheKey({ model, system, prompt: augmentedPrompt, rag })
    : null;
  if (useCache) {
    const cached = cacheGet(cKey);
    if (cached) {
      metrics.cacheHits += 1;
      recordGenerate(ctx.appName, model ?? "default", "cache", 0);
      log("info", "generate.cache.hit", {
        requestId: ctx.requestId,
        app: ctx.appName,
      });
      writeRequestLog(
        buildGenRecord(ctx, {
          model,
          status: "cache",
          durationMs: 0,
          prompt,
          ragChunksUsed,
          usedCollection,
          cached: true,
          response: cached.response,
        }),
      );
      return send(
        res,
        200,
        {
          ok: true,
          response: cached.response,
          model: cached.model,
          ...(ragChunksUsed > 0 ? { ragChunksUsed } : {}),
          durationMs: 0,
          cached: true,
        },
        ctx.requestId,
      );
    }
    metrics.cacheMisses += 1;
  }

  log("info", "generate.start", {
    requestId: ctx.requestId,
    app: ctx.appName,
    promptLen: augmentedPrompt.length,
    systemLen: system?.length ?? 0,
    model: model ?? "default",
    ragChunksUsed,
    cache: useCache,
  });

  let result = await callClaude({
    prompt: augmentedPrompt,
    system,
    model,
    tools: requestedTools,
  });

  if (
    !result.ok &&
    fallbackOnQuota &&
    /quota|429|529|overload|rate.?limit/i.test(result.error ?? "")
  ) {
    log("info", "generate.fallback.ollama", {
      requestId: ctx.requestId,
      app: ctx.appName,
      reason: (result.error ?? "").slice(0, 200),
    });
    metrics.fallbacksOllama += 1;
    result = await callOllamaFallback({ prompt: augmentedPrompt, system });
  }

  if (ragChunksUsed > 0) result.ragChunksUsed = ragChunksUsed;
  recordGenerate(
    ctx.appName,
    model ?? "default",
    result.ok ? "ok" : "error",
    result.durationMs ?? 0,
  );

  log(result.ok ? "info" : "warn", "generate.done", {
    requestId: ctx.requestId,
    app: ctx.appName,
    ok: result.ok,
    durationMs: result.durationMs,
    responseLen: result.response?.length ?? 0,
    ragChunksUsed,
    error: result.error,
    fellback: result.fellbackToOllama ?? false,
  });

  if (result.ok && useCache && cKey) {
    cacheSet(cKey, { response: result.response, model: result.model });
  }

  writeRequestLog(
    buildGenRecord(ctx, {
      model,
      status: result.ok ? "ok" : "error",
      durationMs: result.durationMs ?? 0,
      prompt,
      ragChunksUsed,
      usedCollection,
      cached: false,
      response: result.response,
      error: result.error,
      fellback: result.fellbackToOllama ?? false,
    }),
  );

  send(res, result.ok ? 200 : 500, result, ctx.requestId);
}

async function handleRagUpsert(req, res, ctx) {
  const body = await parseJsonBody(req, res, ctx.requestId);
  if (body === null) return;
  const { collection: collName, id, text, payload, chunk } = body;
  if (!isValidCollectionName(collName)) {
    return send(
      res,
      400,
      { ok: false, error: "invalid collection name (a-z 0-9 _ - only, max 80)" },
      ctx.requestId,
    );
  }
  if (typeof text !== "string" || text.length === 0) {
    return send(res, 400, { ok: false, error: "text is required" }, ctx.requestId);
  }
  if (text.length > 200_000) {
    return send(res, 400, { ok: false, error: "text exceeds 200KB" }, ctx.requestId);
  }
  const collection = resolveCollectionName(collName);
  try {
    await qdrantEnsureCollection(collection);

    if (chunk === true) {
      const chunks = chunkText(text);
      const points = [];
      const ids = [];
      for (let idx = 0; idx < chunks.length; idx++) {
        const c = chunks[idx];
        const extInput =
          typeof id === "string" && id.length > 0 ? `${id}#${idx}` : null;
        const { qdrantId: pid, externalId: extId } = normalizePointId(extInput, c);
        const vector = await getEmbedding(c);
        points.push({
          id: pid,
          vector,
          payload: {
            ...(payload ?? {}),
            text: c,
            ...(extId ? { _externalId: extId } : {}),
            _chunkIndex: idx,
            _chunkOf: chunks.length,
          },
        });
        ids.push(extId ?? pid);
      }
      await qdrantUpsertBatch(collection, points);
      metrics.ragUpsertsTotal += chunks.length;
      metrics.ragChunksTotal += chunks.length;
      log("info", "rag.upsert.chunked", {
        requestId: ctx.requestId,
        app: ctx.appName,
        collection,
        chunks: chunks.length,
      });
      return send(
        res,
        200,
        { ok: true, collection, chunked: true, count: chunks.length, ids },
        ctx.requestId,
      );
    }

    const externalIdInput = typeof id === "string" && id.length > 0 ? id : null;
    const { qdrantId, externalId } = normalizePointId(externalIdInput, text);
    const vector = await getEmbedding(text);
    const fullPayload = {
      ...(payload ?? {}),
      text,
      ...(externalId ? { _externalId: externalId } : {}),
    };
    await qdrantUpsertBatch(collection, [
      { id: qdrantId, vector, payload: fullPayload },
    ]);
    metrics.ragUpsertsTotal += 1;
    log("info", "rag.upsert", {
      requestId: ctx.requestId,
      app: ctx.appName,
      collection,
      id: qdrantId,
      externalId,
      textLen: text.length,
    });
    return send(
      res,
      200,
      {
        ok: true,
        id: qdrantId,
        ...(externalId ? { externalId } : {}),
        collection,
      },
      ctx.requestId,
    );
  } catch (err) {
    const msg = err.message || String(err);
    log("warn", "rag.upsert.failed", {
      requestId: ctx.requestId,
      app: ctx.appName,
      collection,
      error: msg,
    });
    return send(res, 500, { ok: false, error: msg }, ctx.requestId);
  }
}

async function handleRagUpsertBatch(req, res, ctx) {
  const body = await parseJsonBody(req, res, ctx.requestId);
  if (body === null) return;
  const { collection: collName, items, chunk } = body;
  if (!isValidCollectionName(collName)) {
    return send(res, 400, { ok: false, error: "invalid collection name" }, ctx.requestId);
  }
  if (!Array.isArray(items) || items.length === 0 || items.length > 100) {
    return send(
      res,
      400,
      { ok: false, error: "items must be an array of 1..100 entries" },
      ctx.requestId,
    );
  }
  const collection = resolveCollectionName(collName);
  try {
    await qdrantEnsureCollection(collection);
    const points = [];
    const ids = [];
    let chunksEmitted = 0;
    for (const it of items) {
      if (typeof it?.text !== "string" || it.text.length === 0) continue;
      const pieces = chunk === true ? chunkText(it.text) : [it.text];
      for (let i = 0; i < pieces.length; i++) {
        const piece = pieces[i];
        const extInput =
          typeof it.id === "string" && it.id.length > 0
            ? pieces.length > 1
              ? `${it.id}#${i}`
              : it.id
            : null;
        const { qdrantId: pid, externalId: extId } = normalizePointId(extInput, piece);
        const vector = await getEmbedding(piece);
        points.push({
          id: pid,
          vector,
          payload: {
            ...(it.payload ?? {}),
            text: piece,
            ...(extId ? { _externalId: extId } : {}),
            ...(pieces.length > 1
              ? { _chunkIndex: i, _chunkOf: pieces.length }
              : {}),
          },
        });
        ids.push(extId ?? pid);
        if (pieces.length > 1) chunksEmitted += 1;
      }
    }
    if (points.length === 0) {
      return send(res, 400, { ok: false, error: "no valid items in batch" }, ctx.requestId);
    }
    await qdrantUpsertBatch(collection, points);
    metrics.ragUpsertsTotal += points.length;
    metrics.ragChunksTotal += chunksEmitted;
    log("info", "rag.upsert-batch", {
      requestId: ctx.requestId,
      app: ctx.appName,
      collection,
      count: points.length,
      chunksEmitted,
    });
    return send(
      res,
      200,
      { ok: true, count: points.length, ids, chunksEmitted },
      ctx.requestId,
    );
  } catch (err) {
    const msg = err.message || String(err);
    log("warn", "rag.upsert-batch.failed", {
      requestId: ctx.requestId,
      app: ctx.appName,
      collection,
      error: msg,
    });
    return send(res, 500, { ok: false, error: msg }, ctx.requestId);
  }
}

async function handleRagSearch(req, res, ctx) {
  const body = await parseJsonBody(req, res, ctx.requestId);
  if (body === null) return;
  const { collection: collName, query, topK, filter } = body;
  const rerank = body.rerank === true;
  if (!isValidCollectionName(collName)) {
    return send(res, 400, { ok: false, error: "invalid collection name" }, ctx.requestId);
  }
  if (typeof query !== "string" || query.length === 0) {
    return send(res, 400, { ok: false, error: "query is required" }, ctx.requestId);
  }
  const k = Number.isInteger(topK) ? Math.min(Math.max(topK, 1), 50) : 5;
  // When reranking, pull a wider candidate pool (4x, capped at 50) so the
  // Claude pass has something to reorder.
  const fetchK = rerank ? Math.min(k * 4, 50) : k;
  const collection = resolveCollectionName(collName);
  try {
    checkCollectionEmbedMeta(collection);
    const qVec = await getEmbedding(query);
    const sr = await qdrantSearch(collection, qVec, fetchK, filter);
    metrics.ragSearchesTotal += 1;
    let hits = (Array.isArray(sr?.result) ? sr.result : []).map((h) => ({
      id: h.id,
      score: h.score,
      text: h?.payload?.text ?? "",
      payload: Object.fromEntries(
        Object.entries(h?.payload ?? {}).filter(([k]) => k !== "text"),
      ),
    }));
    let didRerank = false;
    if (rerank && hits.length > k) {
      const rr = await claudeRerank(query, hits, k);
      hits = rr.reranked;
      if (rr.ok) {
        metrics.ragReranks += 1;
        didRerank = true;
        log("info", "rag.rerank.ok", {
          requestId: ctx.requestId,
          app: ctx.appName,
          collection,
          candidates: fetchK,
          picked: hits.length,
        });
      } else {
        metrics.ragReranksFailed += 1;
        log("warn", "rag.rerank.failed", {
          requestId: ctx.requestId,
          app: ctx.appName,
          collection,
        });
      }
    } else if (hits.length > k) {
      hits = hits.slice(0, k);
    }
    return send(
      res,
      200,
      {
        ok: true,
        hits,
        collection,
        count: hits.length,
        ...(didRerank ? { reranked: true } : {}),
      },
      ctx.requestId,
    );
  } catch (err) {
    metrics.ragSearchesFailed += 1;
    return send(res, 500, { ok: false, error: err.message }, ctx.requestId);
  }
}

async function handleRagDelete(req, res, ctx, collName, id) {
  if (!isValidCollectionName(collName)) {
    return send(res, 400, { ok: false, error: "invalid collection name" }, ctx.requestId);
  }
  if (!id) return send(res, 400, { ok: false, error: "id required" }, ctx.requestId);
  const collection = resolveCollectionName(collName);
  try {
    await qdrantDeletePoint(collection, id);
    return send(res, 200, { ok: true, id, collection }, ctx.requestId);
  } catch (err) {
    return send(res, 500, { ok: false, error: err.message }, ctx.requestId);
  }
}

async function handleRagDeleteCollection(req, res, ctx, collName) {
  if (!isValidCollectionName(collName)) {
    return send(res, 400, { ok: false, error: "invalid collection name" }, ctx.requestId);
  }
  const collection = resolveCollectionName(collName);
  try {
    await qdrantDeleteCollection(collection);
    if (state.collections[collection]) {
      delete state.collections[collection];
      saveState();
    }
    _ensuredCollections.delete(collection);
    log("info", "rag.collection.deleted", {
      requestId: ctx.requestId,
      app: ctx.appName,
      collection,
    });
    return send(res, 200, { ok: true, collection, deleted: true }, ctx.requestId);
  } catch (err) {
    return send(res, 500, { ok: false, error: err.message }, ctx.requestId);
  }
}

async function handleRagCollections(_req, res, ctx) {
  try {
    const cols = await qdrantListCollections();
    const enriched = cols.map((c) => {
      const meta = state.collections[c.name];
      return meta
        ? {
            ...c,
            embedModel: meta.embedModel,
            embedDim: meta.embedDim,
            createdAt: meta.createdAt,
          }
        : c;
    });
    return send(
      res,
      200,
      { ok: true, collections: enriched },
      ctx.requestId,
    );
  } catch (err) {
    return send(res, 500, { ok: false, error: err.message }, ctx.requestId);
  }
}

async function handleHealth(_req, res, ctx) {
  const claudeV = await getClaudeVersion();
  let qdrantOk = false;
  try {
    await qdrantRequest("GET", "/healthz");
    qdrantOk = true;
  } catch {
    try {
      const r = await fetch(`${QDRANT_URL}/healthz`, {
        signal: AbortSignal.timeout(2_000),
      });
      qdrantOk = r.ok;
    } catch { qdrantOk = false; }
  }
  let ollamaOk = false;
  try {
    const r = await fetch(`${OLLAMA_URL}/api/tags`, {
      signal: AbortSignal.timeout(2_000),
    });
    ollamaOk = r.ok;
  } catch {}
  send(
    res,
    200,
    {
      ok: true,
      claude: claudeV,
      qdrant: qdrantOk ? "ok" : "unreachable",
      ollama: ollamaOk ? "ok" : "unreachable",
      embedProvider: EMBED_PROVIDER,
      embedModel: EMBED_MODEL_EFFECTIVE,
      embedDim: EMBED_DIM,
      ragEnv: RAG_ENV || null,
      authRequired: BRIDGE_TOKEN !== null,
      allowlistLoaded: Object.keys(allowlist).length > 0,
      cacheSize: cache.size,
      requestLog: REQUEST_LOG_ENABLED
        ? { enabled: true, content: REQUEST_LOG_CONTENT, retentionDays: REQUEST_LOG_RETENTION_DAYS }
        : { enabled: false },
      uptimeSec: Math.round((Date.now() - metrics.startedAt) / 1000),
    },
    ctx.requestId,
  );
}

async function handleMetrics(_req, res) {
  sendText(res, 200, renderPromText());
}

async function handleReload(_req, res, ctx) {
  loadAllowlist();
  loadState();
  return send(
    res,
    200,
    {
      ok: true,
      reloaded: true,
      allowlistEntries: Object.keys(allowlist).length,
    },
    ctx.requestId,
  );
}

// GET /log/tail?n=50&app=crm — last N request-log records (newest first).
async function handleLogTail(req, res, ctx) {
  if (!REQUEST_LOG_ENABLED) {
    return send(res, 200, { ok: true, enabled: false, records: [] }, ctx.requestId);
  }
  const u = new URL(req.url, "http://localhost");
  const n = Math.min(Math.max(Number(u.searchParams.get("n") ?? "50") || 50, 1), 1000);
  const app = u.searchParams.get("app") || null;
  const records = tailRequestLog(n, app);
  return send(
    res,
    200,
    { ok: true, enabled: true, count: records.length, records },
    ctx.requestId,
  );
}

// =============================================================================
// OpenAI-compatible chat shim  (POST /v1/chat/completions)
// Lets OpenAI/Ollama-protocol clients route their LLM calls through this
// bridge to the Claude Code CLI. Translates the OpenAI request into
// callClaude() and wraps the reply in OpenAI chat-completion shape. Supports
// fake-streaming for stream:true clients. Tool-calling is NOT supported: any
// `tools` are ignored and a plain assistant message is returned (clients that
// expect function-calling will just get a direct answer).
// =============================================================================
async function handleOpenAIChat(req, res, ctx) {
  const body = await parseJsonBody(req, res, ctx.requestId);
  if (body === null) return;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const toText = (c) =>
    Array.isArray(c)
      ? c.map((p) => (typeof p === "string" ? p : (p?.text ?? ""))).join("")
      : typeof c === "string"
        ? c
        : "";
  const sysParts = messages
    .filter((m) => m && m.role === "system")
    .map((m) => toText(m.content))
    .filter((s) => s.trim().length > 0);
  const convo = messages.filter((m) => m && m.role !== "system");
  const system = sysParts.length ? sysParts.join("\n\n") : undefined;
  let prompt;
  if (convo.length === 1 && convo[0].role === "user") {
    prompt = toText(convo[0].content);
  } else {
    prompt = convo
      .map((m) => {
        const who =
          m.role === "assistant" ? "Assistant" : m.role === "tool" ? "Tool" : "User";
        return `${who}: ${toText(m.content)}`;
      })
      .join("\n\n");
  }
  if (!prompt || !prompt.trim()) prompt = "(no content)";

  const model = "sonnet"; // map any client model name → a bridge model
  const policy = checkAllowlist(ctx.appName, model);
  if (!policy.ok) {
    log("warn", "v1.chat.denied.policy", {
      requestId: ctx.requestId,
      app: ctx.appName,
      reason: policy.error,
    });
    return sendText(
      res,
      403,
      JSON.stringify({
        error: { message: policy.error, type: "policy_violation" },
      }),
      "application/json; charset=utf-8",
    );
  }
  const t0 = Date.now();
  const r = await callClaude({ prompt, system, model });
  const ms = Date.now() - t0;
  const respModel = typeof body.model === "string" && body.model ? body.model : "bridge";
  const id = `chatcmpl-${ctx.requestId}`;
  const created = Math.floor(Date.now() / 1000);

  if (!r.ok) {
    recordGenerate(ctx.appName, model, "error", ms);
    log("warn", "v1.chat.error", {
      requestId: ctx.requestId,
      app: ctx.appName,
      error: r.error,
    });
    return sendText(
      res,
      502,
      JSON.stringify({
        error: { message: r.error || "bridge generate failed", type: "bridge_error" },
      }),
      "application/json; charset=utf-8",
    );
  }
  const text = r.response ?? "";
  recordGenerate(ctx.appName, model, "claude", ms);
  log("info", "v1.chat.ok", {
    requestId: ctx.requestId,
    app: ctx.appName,
    ms,
    stream: body.stream === true,
  });

  if (body.stream === true) {
    res.writeHead(200, {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });
    const emit = (delta, finish) =>
      res.write(
        `data: ${JSON.stringify({
          id,
          object: "chat.completion.chunk",
          created,
          model: respModel,
          choices: [{ index: 0, delta, finish_reason: finish ?? null }],
        })}\n\n`,
      );
    emit({ role: "assistant" });
    emit({ content: text });
    emit({}, "stop");
    res.write("data: [DONE]\n\n");
    res.end();
    return;
  }

  sendText(
    res,
    200,
    JSON.stringify({
      id,
      object: "chat.completion",
      created,
      model: respModel,
      choices: [
        { index: 0, message: { role: "assistant", content: text }, finish_reason: "stop" },
      ],
      usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
    }),
    "application/json; charset=utf-8",
  );
}

// =============================================================================
// OpenAI-compatible embeddings shim  (POST /v1/embeddings)
// Routes a client's embedding requests through the bridge's embedder (see
// EMBED_PROVIDER - in-process ONNX by default). Keeps OpenAI-shaped clients working unchanged.
// =============================================================================
async function handleOpenAIEmbeddings(req, res, ctx) {
  const body = await parseJsonBody(req, res, ctx.requestId);
  if (body === null) return;
  const inputs = Array.isArray(body.input)
    ? body.input
    : body.input != null
      ? [body.input]
      : [];
  try {
    const data = [];
    for (let i = 0; i < inputs.length; i++) {
      const vec = await getEmbedding(String(inputs[i] ?? ""));
      data.push({ object: "embedding", index: i, embedding: vec });
    }
    sendText(
      res,
      200,
      JSON.stringify({
        object: "list",
        data,
        model:
          typeof body.model === "string" && body.model
            ? body.model
            : EMBED_PROVIDER === "ollama"
              ? EMBED_MODEL
              : EMBED_MODEL_LOCAL,
        usage: { prompt_tokens: 0, total_tokens: 0 },
      }),
      "application/json; charset=utf-8",
    );
  } catch (err) {
    log("warn", "v1.embeddings.error", {
      requestId: ctx.requestId,
      app: ctx.appName,
      error: err.message,
    });
    sendText(
      res,
      502,
      JSON.stringify({ error: { message: err.message, type: "embedding_error" } }),
      "application/json; charset=utf-8",
    );
  }
}

// =============================================================================
// Server
// =============================================================================

const server = createServer(async (req, res) => {
  const ctx = getRequestContext(req);
  // /health stays open — Docker health checks cannot carry a token.
  // /metrics is NOT exempt: it leaks per-app request counts and model tiers,
  // so a Prometheus job must send X-Bridge-Token.
  const isPublic = req.method === "GET" && req.url === "/health";
  // /v1/* is token-gated too (otherwise it would be an unauthenticated path to
  // generation). OpenAI-compatible clients send the bridge token as their API
  // key (Authorization: Bearer <token>).
  if (!isPublic && !checkAuth(ctx)) {
    incMap(metrics.requestsTotal, `${req.method},${req.url ?? ""},401`);
    return send(
      res,
      401,
      { ok: false, error: "unauthorized — set X-Bridge-Token header" },
      ctx.requestId,
    );
  }
  // Allowlist + rate limiting for the /rag/* and /v1/* surfaces (/rag/search
  // rerank and /v1/chat both spawn generation). /generate keeps its own model-aware
  // check inside handleGenerate; /v1/chat additionally model-checks "sonnet".
  if (
    typeof req.url === "string" &&
    (req.url.startsWith("/rag/") ||
      req.url === "/v1/chat/completions" ||
      req.url === "/v1/embeddings")
  ) {
    const policy = checkAllowlist(ctx.appName, null);
    if (!policy.ok) {
      incMap(metrics.requestsTotal, `${req.method},${req.url ?? ""},403`);
      log("warn", "policy.denied", {
        requestId: ctx.requestId,
        app: ctx.appName,
        url: req.url,
        reason: policy.error,
      });
      return send(res, 403, { ok: false, error: policy.error }, ctx.requestId);
    }
    recordAppRequest(ctx.appName);
  }
  let status = 200;
  try {
    if (req.method === "GET" && req.url === "/health") {
      await handleHealth(req, res, ctx);
      status = res.statusCode;
      return;
    }
    if (req.method === "GET" && req.url === "/metrics") {
      await handleMetrics(req, res);
      status = res.statusCode;
      return;
    }
    if (req.method === "POST" && req.url === "/generate") {
      await handleGenerate(req, res, ctx);
      status = res.statusCode;
      return;
    }
    if (req.method === "POST" && req.url === "/v1/chat/completions") {
      await handleOpenAIChat(req, res, ctx);
      status = res.statusCode;
      return;
    }
    if (req.method === "POST" && req.url === "/v1/embeddings") {
      await handleOpenAIEmbeddings(req, res, ctx);
      status = res.statusCode;
      return;
    }
    if (req.method === "GET" && req.url === "/v1/models") {
      sendText(
        res,
        200,
        JSON.stringify({
          object: "list",
          data: [{ id: "bridge", object: "model", owned_by: "llm-bridge" }],
        }),
        "application/json; charset=utf-8",
      );
      status = res.statusCode;
      return;
    }
    if (req.method === "POST" && req.url === "/rag/upsert") {
      await handleRagUpsert(req, res, ctx);
      status = res.statusCode;
      return;
    }
    if (req.method === "POST" && req.url === "/rag/upsert-batch") {
      await handleRagUpsertBatch(req, res, ctx);
      status = res.statusCode;
      return;
    }
    if (req.method === "POST" && req.url === "/rag/search") {
      await handleRagSearch(req, res, ctx);
      status = res.statusCode;
      return;
    }
    if (req.method === "GET" && req.url === "/rag/collections") {
      await handleRagCollections(req, res, ctx);
      status = res.statusCode;
      return;
    }
    if (req.method === "POST" && req.url === "/admin/reload") {
      await handleReload(req, res, ctx);
      status = res.statusCode;
      return;
    }
    if (req.method === "GET" && req.url?.startsWith("/log/tail")) {
      await handleLogTail(req, res, ctx);
      status = res.statusCode;
      return;
    }
    if (req.method === "DELETE" && req.url?.startsWith("/rag/")) {
      const parts = req.url
        .slice(5)
        .split("/")
        .filter((p) => p.length > 0);
      const collection = decodeURIComponent(parts[0] ?? "");
      if (parts.length === 1) {
        await handleRagDeleteCollection(req, res, ctx, collection);
      } else {
        const id = decodeURIComponent(parts[1] ?? "");
        await handleRagDelete(req, res, ctx, collection, id);
      }
      status = res.statusCode;
      return;
    }
    status = 404;
    return send(res, 404, { ok: false, error: "not found" }, ctx.requestId);
  } catch (err) {
    status = 500;
    log("error", "handler.crashed", {
      requestId: ctx.requestId,
      app: ctx.appName,
      url: req.url,
      error: err.message,
    });
    return send(res, 500, { ok: false, error: err.message }, ctx.requestId);
  } finally {
    incMap(
      metrics.requestsTotal,
      `${req.method ?? "?"},${req.url ?? "?"},${status}`,
    );
  }
});

server.listen(PORT, HOST, () => {
  log("info", "claude-bridge listening", {
    host: HOST,
    port: PORT,
    bin: CLAUDE_BIN,
    authRequired: BRIDGE_TOKEN !== null,
    allowlistLoaded: Object.keys(allowlist).length > 0,
    ragEnv: RAG_ENV || null,
    cacheMax: CACHE_MAX_ENTRIES,
    retryTransient: RETRY_TRANSIENT,
    requestLog: REQUEST_LOG_ENABLED
      ? `on (content=${REQUEST_LOG_CONTENT}, retentionDays=${REQUEST_LOG_RETENTION_DAYS})`
      : "off",
  });
  // Fire-and-forget — server is already accepting requests; backfill is
  // best-effort metadata that should not block listen.
  void backfillCollectionState();
  if (REQUEST_LOG_ENABLED) pruneOldLogs();
});

function shutdown(signal) {
  log("info", "shutdown", { signal });
  saveState();
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(1), 2000).unref();
}
process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));
