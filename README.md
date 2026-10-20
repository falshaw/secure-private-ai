# secure-private-ai

Building blocks for running AI features in self-hosted, containerised apps without scattering model credentials across them.

## The pattern

1. **Containers never hold model credentials.** App containers call one HTTP gateway on the host. The gateway is the only component that can reach a hosted model; it uses the Claude Code CLI's own login, so no API key is stored in any app, image or compose file.
2. **One authenticated gateway with policy.** Every call carries a shared token (`X-Bridge-Token`) and an app name (`X-App-Name`). A per-app allowlist decides which model tiers an app may use and how many requests per hour it gets. Each call is logged as metadata (sizes and hashes, not text, unless you opt in).
3. **Local models for bulk text.** Large reading and drafting jobs go to a local Ollama model through a small CLI, so the data stays on the machine and costs nothing per token.
4. **Shared auth primitives.** Apps use the same small JWT/bcrypt core, so a security fix (algorithm pin, hash cost) lands everywhere at once.

```
 app container ──┐   X-Bridge-Token + X-App-Name
 app container ──┼──> bridge (127.0.0.1:7777) ──> claude -p  (CLI login, no tools, no MCP)
 app container ──┘        │  token check, allowlist, rate limit, request log
                          ├──> Qdrant  (/rag/*, optional)
                          └──> embeddings (in-process ONNX, or Ollama)

 host scripts ─────> cli/local-llm.mjs ──> Ollama (local model)
```

## Layout

| Path | What it is |
|---|---|
| `bridge/claude-bridge.mjs` | The gateway: a dependency-free Node HTTP server |
| `bridge/Dockerfile`, `bridge/docker-compose.yml` | Container build and run config for the gateway |
| `bridge/embedder/package.json` | In-process embedding backend (`@huggingface/transformers`) installed into the image |
| `bridge/bridge-allowlist.example.json` | Example per-app model and rate-limit policy |
| `cli/local-llm.mjs` | CLI that sends a prompt plus files to a local Ollama model |
| `packages/auth` | `@falshaw/auth`: bcrypt + HS256 JWT core built on `jose` |
| `packages/rag-module` | `@falshaw/rag-module`: embedded RAG on LanceDB with Ollama embeddings |

## Bridge

### How a call is handled

1. **Auth.** Every route except `GET /health` requires `X-Bridge-Token` (or `Authorization: Bearer <token>` on the OpenAI-compatible `/v1/*` routes). Comparison is constant-time. `/metrics` is gated too.
2. **Policy.** The allowlist entry for `X-App-Name` (or the `"*"` fallback) limits the models (`haiku`, `sonnet`, `opus`) and requests per hour (sliding one-hour window). Violations return `403`. This applies to `/generate`, `/rag/*` and `/v1/*`. A `/generate` call without `model` runs on the first model in the caller's entry (so a `["haiku"]` app cannot reach the CLI's default model by leaving the field out); an unknown model name returns `400`. Apps listed in the allowlist each get their own hourly bucket. Every other caller (no header, an unlisted name, `*`) shares one `"*"` bucket, because `X-App-Name` is chosen by the caller: inventing a new name per request does not reset the limit.
3. **Generation.** The prompt is piped to `claude -p` with tools disabled, an empty MCP config, slash commands disabled and no session persistence. A caller may request only `WebSearch` and `WebFetch` as tools. Transient errors (429/529/503/overload) are retried once.
4. **Optional extras per call:** RAG augmentation from a Qdrant collection (with optional model-based reranking), an in-memory LRU cache (`cache: true`), and a fallback to a local Ollama model when the hosted quota is exhausted (`fallbackOnQuotaExhausted: true`, needs `OLLAMA_URL`).
5. **Logging.** One JSONL record per `/generate` call: app, model, status, duration, prompt/response character counts and a 12-char SHA-256 prefix of the prompt. Prompt and response text are stored only with `BRIDGE_REQUEST_LOG_CONTENT=true`, truncated. Daily files, pruned after 30 days by default.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Status of the CLI, Qdrant, Ollama, embedder, auth and allowlist (unauthenticated) |
| GET | `/metrics` | Prometheus metrics |
| POST | `/generate` | `{ prompt, system?, model?, rag?, cache?, tools?, fallbackOnQuotaExhausted? }` |
| POST | `/v1/chat/completions` | OpenAI-compatible chat (non-streaming, or a single-chunk stream); no function calling. `model` `haiku`/`sonnet`/`opus` is honoured, any other name runs on the caller's first allowed model (`sonnet` without an allowlist) |
| POST | `/v1/embeddings` | OpenAI-compatible embeddings |
| GET | `/v1/models` | Lists one model, `bridge` |
| POST | `/rag/upsert`, `/rag/upsert-batch` | Embed and store text (optional sentence-window chunking) |
| POST | `/rag/search` | Vector search, optional `rerank: true` |
| GET | `/rag/collections` | Collections with the embedding model recorded for each |
| DELETE | `/rag/:collection[/:id]` | Delete a point (by the same `id` you upserted: UUID, integer or your own string, URL-encoded) or a whole collection |
| POST | `/admin/reload` | Re-read allowlist and state from disk |
| GET | `/log/tail?n=&app=` | Last N request-log records |

The bridge records which embedding model built each collection and logs a warning when the running embedder differs, because vectors of the right length from a different model are accepted by Qdrant but return poor matches.

### Run it

Requirements: Docker, a Claude Code login on the host (`claude login`, which creates `~/.claude/.credentials.json`). Qdrant and Ollama are optional.

```bash
cd bridge
cp .env.example .env              # set CLAUDE_BRIDGE_TOKEN and CLAUDE_CREDENTIALS_FILE
cp bridge-allowlist.example.json bridge-allowlist.json   # optional: edit the policy, then
                                  # add BRIDGE_ALLOWLIST_FILE=./bridge-allowlist.json to .env
docker compose up -d --build
curl -s http://127.0.0.1:7777/health    # or the BRIDGE_PORT you set
```

Without `BRIDGE_ALLOWLIST_FILE` in `.env`, compose mounts `bridge-allowlist.example.json`, so edits to your copy have no effect.

On Windows (PowerShell) use `Copy-Item` instead of `cp`, and write `CLAUDE_CREDENTIALS_FILE` with forward slashes, e.g. `C:/Users/you/.claude/.credentials.json`.

**Project name.** The compose file sets `name: llm-bridge`, so the stack does not take its name from the `bridge/` directory. If you already run a compose project with the same name, or want a second copy, pass a distinct name and port, otherwise compose treats the two as one project and recreates the other stack's containers:

```bash
BRIDGE_PORT=17777 docker compose -p my-bridge up -d --build    # PowerShell: $env:BRIDGE_PORT=17777; docker compose -p my-bridge up -d --build
docker compose -p my-bridge down                                 # stop it with the same -p
```

**Qdrant.** `/rag/*` needs a Qdrant. By default `QDRANT_URL` is `http://host.docker.internal:6333`, i.e. a Qdrant already published on the host's port 6333. Without one, `/health` reports Qdrant as unreachable and `/rag/*` returns errors; `/generate` works regardless. To use the Qdrant bundled in the compose file instead (no host port published, data in a named volume):

```bash
# in bridge/.env
QDRANT_URL=http://qdrant:6333
# then
docker compose --profile qdrant up -d --build
```

The port is published on `127.0.0.1` only. Containers reach it at `http://host.docker.internal:7777` (add `extra_hosts: ["host.docker.internal:host-gateway"]` on Linux). The container drops all capabilities, sets `no-new-privileges`, runs as a non-root user and has memory/CPU/PID limits. The credentials file is mounted read-only and copied into the container's home on start.

Calling it from an app:

```bash
curl -s http://127.0.0.1:7777/generate \
  -H "Content-Type: application/json" \
  -H "X-Bridge-Token: $CLAUDE_BRIDGE_TOKEN" \
  -H "X-App-Name: crm" \
  -d '{"prompt":"Summarise this ticket in one line: ...","model":"haiku"}'
```

Without Docker: `node bridge/claude-bridge.mjs` (Node 20+, `claude` on `PATH`). It binds to `127.0.0.1` by default. It does not read `bridge/.env`: export the variables yourself, otherwise it starts with auth disabled (it logs an `auth.disabled` warning). `claude -p` then runs as your own user with your Claude Code login and user settings.

**`/generate` returns `500` with "claude exited 1 ... not logged in or its credentials are missing/expired".** The CLI inside the container could not authenticate. Run `claude login` on the host, check that `CLAUDE_CREDENTIALS_FILE` points at the resulting `.credentials.json`, and recreate the container (`docker compose up -d --force-recreate`) so the fresh file is copied in. `/health` shows the CLI version but cannot tell whether the login is valid.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `CLAUDE_BRIDGE_TOKEN` | unset | Shared secret. **If unset, auth is disabled**; the compose file refuses to start without it |
| `CLAUDE_BRIDGE_HOST` / `CLAUDE_BRIDGE_PORT` | `127.0.0.1` / `7777` | Bind address |
| `CLAUDE_BRIDGE_ALLOWLIST_FILE` | `bridge/bridge-allowlist.json` | Policy file. **If missing, no per-app policy is enforced** |
| `CLAUDE_BIN` | `claude` | CLI binary |
| `SPAWN_TIMEOUT_MS` / `RESEARCH_SPAWN_TIMEOUT_MS` | `120000` / `300000` | CLI timeout without / with web tools |
| `RETRY_TRANSIENT` / `RETRY_DELAY_MS` | `true` / `2000` | One retry on transient errors |
| `CACHE_MAX_ENTRIES` / `CACHE_TTL_MS` | `256` / 1 h | LRU cache |
| `QDRANT_URL` / `QDRANT_API_KEY` | `http://127.0.0.1:6333` / empty | Vector store for `/rag/*` |
| `RAG_ENV` | empty | Prefix for collection names (e.g. `dev`) |
| `RAG_UUID_NAMESPACE` | `llm-bridge-rag` | Namespace for deterministic point IDs |
| `EMBED_PROVIDER` | `local` | `local` (ONNX in-process), `ollama`, or `none` |
| `EMBED_MODEL_LOCAL` / `EMBED_MODEL` / `EMBED_DIM` | `nomic-ai/nomic-embed-text-v1.5` / `nomic-embed-text` / `768` | Embedding model (local / Ollama) and dimension |
| `EMBED_PKG_DIR` / `EMBED_CACHE_DIR` | `/opt/embedder` / `bridge/.embed-cache` | Embedder package and weight cache |
| `OLLAMA_URL` / `OLLAMA_FALLBACK_MODEL` | `http://127.0.0.1:11434` / `qwen2.5:7b` | Ollama for embeddings and quota fallback |
| `CHUNK_TARGET_CHARS` / `CHUNK_OVERLAP_CHARS` | `2000` / `200` | Chunker |
| `BRIDGE_STATE_FILE` | `bridge/.bridge-state.json` | Collection metadata |
| `BRIDGE_REQUEST_LOG`, `_DIR`, `_RETENTION_DAYS`, `_CONTENT`, `_CONTENT_MAX` | `true`, `bridge/.bridge-logs`, `30`, `false`, `500` | Request log |

Compose-only: `CLAUDE_CREDENTIALS_FILE` (host path of the credentials file), `BRIDGE_ALLOWLIST_FILE` (host path of the allowlist) and `BRIDGE_PORT` (host port, default `7777`). In compose, `QDRANT_URL` defaults to `http://host.docker.internal:6333` (the host's Qdrant); set it to `http://qdrant:6333` with `--profile qdrant` for the bundled one.

## Local model CLI

`cli/local-llm.mjs` sends a prompt and any number of files to a local Ollama model and prints the answer or writes it to a file. No dependencies beyond Node 20+.

```bash
node cli/local-llm.mjs "Summarise the errors in this log" -f app.log
node cli/local-llm.mjs -s "You are a translator." "Translate to German" -f post.md -o post.de.md
cat notes.txt | node cli/local-llm.mjs "List the action items" -f -
node cli/local-llm.mjs -m qwen2.5:14b "..."
```

- Sizes the context window to the input (8k to 64k tokens) instead of Ollama's 4k default, which silently truncates.
- Streams over `node:http`, so long generations are not cut off by `fetch`'s header timeout.
- Runs one job at a time via a lock file, so parallel callers do not load several large models at once.
- Appends every call (prompt, inputs, output) to `data/local-llm/calls.jsonl` for review or later fine-tuning. This file can contain private data and is git-ignored. `LOCAL_LLM_NOLOG=1` disables it.
- Exit codes: `0` ok, `1` error, `2` Ollama not reachable.

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama server |
| `LOCAL_LLM_MODEL` | unset | Model to use (same as `-m`) |
| `LOCAL_LLM_PREFERRED` | `gemma4:31b` | Comma-separated fallback list; first pulled model wins |
| `LOCAL_LLM_DATA_DIR` | `data/local-llm` | Lock file and call log |
| `LOCAL_LLM_THINK` | unset | `1` (or `true`) enables the model's reasoning mode |
| `LOCAL_LLM_NOLOG` | unset | `1` skips the call log |

## packages/auth

`@falshaw/auth`: `hashPassword` / `verifyPassword` (bcrypt, cost 12), `encodeJwtSecret` (with optional minimum length), `signHs256` and `verifyHs256` (algorithm pinned to HS256). Peer dependencies: `jose`, `bcryptjs`. Apps that build in an isolated Docker context can vendor the single file. See [packages/auth/README.md](packages/auth/README.md).

## packages/rag-module

`@falshaw/rag-module`: `LocalRAG` stores documents in an embedded LanceDB table (no server) and embeds them with Ollama (`nomic-embed-text`, 768-dim). Methods: `init`, `ingest`, `search`, `getContext` (prompt-ready context string), `count`, `clear`.

```bash
cd packages/rag-module
npm install
ollama pull nomic-embed-text
npm test                        # smoke test against a throwaway table
npm run ingest                  # load every knowledge/*.json into ./data/lancedb
npm run ingest -- network-security
```

`knowledge/` contains a synthetic network-security sample and a German summary of GDPR, NIS2, EU AI Act, GoBD and BSI C5 provisions. `RAG_DB_PATH` and `OLLAMA_URL` override the defaults.

This package is independent of the bridge's `/rag/*` endpoints: use it for an embedded per-app store, or the bridge for a shared Qdrant-backed one.

## Tests

- `packages/rag-module`: `npm test` (needs Ollama with `nomic-embed-text`).
- `packages/rag-module`: `npm run build` type-checks and compiles without Ollama.
- Bridge: `node --test "bridge/test/*.test.mjs"` from the repo root (Node 20+, no dependencies, no CLI or Qdrant needed). Covers token auth, the per-app and shared rate limits (including the name-rotation case), the model allowlist and model defaulting, body validation (non-object JSON, oversized bodies), routing with query strings, metrics labels and RAG point-id mapping.
- CLI: no automated tests. `node --check bridge/claude-bridge.mjs && node --check cli/local-llm.mjs` checks syntax (`node --check` reads one file; a second path is passed to it as an argument and not checked). `/health` checks a running bridge.

## Security notes

- Keep the bridge on loopback. Anyone holding the token can generate text on your account, within the allowlist limits.
- Always set `CLAUDE_BRIDGE_TOKEN` and ship an allowlist with a restrictive `"*"` entry; the bridge itself runs open if either is missing.
- There is one shared token, not per-app credentials: `X-App-Name` is chosen by the caller, so the allowlist limits what a name can do, not who may use it. Every token holder can also call `/admin/reload` and read every app's `/log/tail`.
- Prompts sent to `/generate` leave the machine (to the hosted model). Use the local CLI or Ollama for data that must not.

## License

MIT, see [LICENSE](LICENSE).
