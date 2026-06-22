<!-- SPDX-License-Identifier: Apache-2.0 -->

# Sizing & deployment

How much disk and RAM Artesian needs, and how to run a shared vector backend as a service.
All figures are **order-of-magnitude estimates** — actual usage depends on the embedding model,
the average note/chunk size, and the payload you store. Use them to provision, then measure with
`artesian doctor` and your backend's own metrics.

## What consumes space

A unit of memory is a **chunk** (Markdown is section-chunked on store), embedded into one vector.
Per chunk:

- **Vector** — `dim × 4 bytes` (f32). The pinned default `intfloat/multilingual-e5-small` is
  **384-d → ~1.5 KB**. `multilingual-e5-large` is 1024-d → ~4 KB (≈2.7×).
- **ANN graph** (Qdrant/HNSW, `m=16`) — roughly `2 × m × 4 bytes` of links → **~0.1–0.5 KB**.
- **Payload** — the chunk text + metadata (tags, scope, ids, timestamps) → **~0.5–2 KB** typical.
- **OKF Markdown copy** (Files backend, or the OKF mirror) — the human-readable source on disk.

So budget **~2–4 KB per chunk on disk** for a vector backend, and a few hundred bytes for the
Files backend (text only, no vectors).

## Sizing by scale

Reference: `e5-small` (384-d), ~1.5 chunks per note, ~1 KB payload/chunk. "Records" = chunks.

| Scenario | Chunks | Files (md only) | sqlite-vec (local) | Qdrant (shared) |
|---|---|---|---|---|
| **1 user** | 10⁴–10⁵ | ~5–50 MB disk, trivial RAM | ~30–300 MB disk, tens of MB RAM | ~50–400 MB disk, ~0.2–0.5 GB RAM |
| **Several users** (≈5, own collections) | ~5×10⁴–5×10⁵ | ~25–250 MB | ~150 MB–1.5 GB disk | ~0.3–2 GB disk, ~0.5–1.5 GB RAM |
| **Team** (≈20, shared projects) | ~2×10⁵–2×10⁶ | ~0.1–1 GB | not recommended past ~10⁶ | ~1–8 GB disk, ~2–6 GB RAM |

Notes:

- **RAM is the Qdrant lever, not disk.** Qdrant keeps vectors + HNSW resident for low-latency
  search; with `mmap` it can spill cold data to disk. Scalar quantization (int8) cuts the resident
  vector footprint ~4× at a small recall cost — turn it on for large collections.
- **Embeddings are local and token-free.** The only hot-path cost is CPU embedding of the *query*
  (~10–50 ms on a laptop CPU); writes embed once. No external API, no token spend.
- **sqlite-vec scales to ~10⁵–10⁶** comfortably for a single user / small team and needs no server.
  Past that, or for true multi-writer sharing, move to Qdrant.
- **Files backend has no vectors** — it is for zero-infra, fully inspectable memory; search is
  keyword/lexical, so size it only for the Markdown.

Rule of thumb: **per active user, plan ~50–400 MB disk and ~0.3 GB RAM on Qdrant**; multiply by
users for a shared deployment and add headroom for HNSW + payload growth.

## Running Qdrant as a service (shared / cloud)

A shared Qdrant lets several machines and agents read/write one memory. Generic recipe — no
host-specific values; keep endpoints and the API key in your local config, never in a repo.

### 1. Compose

```yaml
# docker-compose.yml  (Docker / OrbStack / Podman)
services:
  qdrant:
    image: qdrant/qdrant:latest
    restart: unless-stopped
    ports:
      - "6333:6333"   # REST
      - "6334:6334"   # gRPC
    volumes:
      - ./qdrant-storage:/qdrant/storage   # bind to a real disk you back up
    environment:
      QDRANT__SERVICE__API_KEY: "${QDRANT_API_KEY}"   # from your env, not committed
```

```shell
QDRANT_API_KEY="$(openssl rand -hex 32)" docker compose up -d
curl -fsS -H "api-key: $QDRANT_API_KEY" http://localhost:6333/healthz && echo " ok"
```

### 2. Point Artesian at it

```shell
export QDRANT_API_KEY="…"   # also persist to your shell rc; init's wrapper sources it
artesian init --backend qdrant --project my-project \
  --qdrant-url http://HOST:6333 --qdrant-api-key-env QDRANT_API_KEY
```

`init` registers the MCP server (Claude Code user scope, Codex, Zed) via the
`run-artesian-mcp.sh` wrapper so the API key reaches the server even though MCP clients launch it
without your login shell. See [onboarding.md](onboarding.md).

### 3. Share over the network

- Several agents on a LAN can share one **MCP** server over streamable HTTP:
  `artesian-mcp --transport http --bind 0.0.0.0:8080 --backend qdrant --collection my-project`
  (build with `--features http`). Bind to a trusted interface only.
- Off-LAN access: front Qdrant/MCP with your own reverse proxy + TLS, or ride a private overlay
  (Tailscale/WireGuard). **Do not expose Qdrant publicly without auth + transport security** — the
  API key is the only gate at the Qdrant layer.

### 4. Security & ops

- **Auth:** always set `QDRANT__SERVICE__API_KEY`. Keep it in env / a `0600` file, never in git.
- **Isolation:** one collection per project (default) gives hard tenant isolation and simple
  routing; `user_id` payload metadata adds per-user scoping inside a shared collection.
- **Backups:** schedule the Qdrant snapshot API to a backed-up volume; the storage bind-mount
  should already live on durable disk.
- **Upgrades:** the binary upgrades independently of the data — `artesian update` swaps the binary;
  config, MCP registrations, and stored vectors persist. Run `artesian doctor` after.
- **Capacity guardrail:** enable scalar quantization and `mmap` for large collections; watch
  Qdrant RAM, not disk, as the first limit.

See also [backends.md](backends.md), [concurrency.md](concurrency.md), [upgrades.md](upgrades.md).
