# LLM bridge — containerized.
# Wraps the logged-in `claude` CLI behind the bridge HTTP shim
# (claude-bridge.mjs, copied into /app).
#
# Auth: a Claude Code credentials file is mounted read-only at /seed and
# copied into the container's writable $HOME/.claude on startup. No API key
# is stored in the image or handed to calling containers.
FROM node:24.22-bookworm-slim@sha256:a5f4366e834695d5f363b920c104953a535d3a9864d1bdf74566a0ebe01daee5

# ca-certificates: TLS to api.anthropic.com.  ripgrep: used by the claude CLI.
RUN rm -rf /var/lib/apt/lists/* && apt-get -o Acquire::Retries=5 -o Acquire::http::Pipeline-Depth=0 update \
 && apt-get install -y --no-install-recommends ca-certificates ripgrep \
 && rm -rf /var/lib/apt/lists/*

# Claude Code CLI — pinned version. Puts `claude` on PATH.
RUN npm install -g @anthropic-ai/claude-code@2.1.145 \
 && npm cache clean --force

# Local embedding backend for /rag/* and /v1/embeddings. transformers.js runs
# the ONNX model in-process on CPU - no server, no GPU, no cost. Installed into
# its own prefix (the bridge code has no node_modules); the bridge resolves it
# from EMBED_PKG_DIR (default /opt/embedder).
#
# The dependency list lives in embedder/package.json rather than being written
# inline: npm 11 blocks install scripts by default and refuses --allow-scripts
# on a project-scoped install (EALLOWSCRIPTS), so the approvals have to be an
# "allowScripts" field in the manifest. Without them onnxruntime-node never
# fetches its native runtime and the import fails at runtime, not at build.
#
# transformers@4.2.0 pins sharp ^0.34.5 and onnxruntime-node pins adm-zip <0.6.0,
# which npm reports as "no fix available" - but fixed releases exist, the
# parents just have not moved. The "overrides" block in embedder/package.json
# forces sharp >=0.35.4 (GHSA-f88m-g3jw-g9cj, libvips) and adm-zip >=0.6.0
# (GHSA-xcpc-8h2w-3j85). If a future bump breaks the override, re-check before
# dropping it.
COPY embedder/package.json /opt/embedder/package.json
RUN cd /opt/embedder \
 && npm install --omit=dev \
 && npm cache clean --force

# Non-root user with a writable HOME for credential copy + CLI token refresh.
RUN useradd --create-home --uid 1001 claude \
 && mkdir -p /home/claude/.claude /app /state /config \
 && chown -R claude:claude /home/claude /state

# Bridge code: root-owned, read-only for the runtime user.
COPY claude-bridge.mjs /app/claude-bridge.mjs

ENV HOME=/home/claude
ENV CLAUDE_BIN=claude

USER claude
WORKDIR /app
EXPOSE 7777

# Seed credentials into the writable HOME, then exec the bridge. $HOME is
# expanded at runtime by sh -c (not at build time), so this is correct under
# both `docker run` and docker compose.
ENTRYPOINT ["/bin/sh","-c","mkdir -p \"$HOME/.claude\"; if [ -f /seed/.credentials.json ]; then cp /seed/.credentials.json \"$HOME/.claude/.credentials.json\"; chmod 600 \"$HOME/.claude/.credentials.json\"; else echo 'WARN: /seed/.credentials.json not mounted - claude auth will fail' >&2; fi; exec node /app/claude-bridge.mjs"]
