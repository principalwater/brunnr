<!-- SPDX-License-Identifier: Apache-2.0 -->

# Onboarding

Two ways to bring Artesian up: **a human follows the Quickstart**, or **an AI agent follows the
agent recipe** below. Both are non-destructive and idempotent — running them twice changes
nothing the second time, and they never delete existing memory or overwrite unrelated config.

Artesian runs with **zero configuration** in `memory` mode on the Files (OKF) backend. Add a vector
backend or orchestration only when you want them. Sensible defaults everywhere.

---

## A. Human path (Quickstart)

Install the `artesian` CLI (see the [README](../README.md#install)), or run from source by prefixing
any command with `cargo run -p artesian-cli -- ` instead of `artesian`.

```shell
# memory mode, zero-infra Files (OKF) backend — the default
artesian init
artesian memory store "Artesian keeps durable context" --tag bootstrap
artesian memory find durable
```

Pick a backend (config choice, not a code change):

```shell
artesian init --backend sqlite-vec          # local hybrid, zero infra
artesian init --backend qdrant \            # shared / multi-user
  --project my-project --qdrant-url http://HOST:6333
```

For Qdrant, one URL is enough on default ports: `:6333` is treated as REST and the gRPC sibling
`:6334` is derived; `:6334` derives REST `:6333`. If you use custom ports, pass both
`--qdrant-url` and `--qdrant-rest-url`. `init` and import commands preflight both endpoints
(gRPC health + REST `/healthz` + auth) and fail with the exact endpoint that is wrong.

`artesian init` detects installed agent CLIs and writes the MCP registration for each (Claude Code,
Codex, Zed) pointing at `artesian-mcp` with the pinned embedding model and behavior-guiding tool
descriptions. Then drive your agent exactly as before — it now has `memory.find` / `memory.store`.

Backfill existing notes (idempotent), and explore modes:

```shell
artesian backfill ./memory-export   # md/json + task md -> OKF/Headrace
artesian memory context "what matters now"
```

`backfill` is robust: a bad file is skipped and reported, not fatal. Markdown is section-chunked
by heading, an OKF `index.md` catalog is generated, task/status markdown is routed into Headrace, and
the command prints a JSON summary with `{scanned, imported, skipped_duplicates, failed}` counts.

**Linked memory by default.** `backfill` and `onboard` now run deterministic entity-relation
extraction on every chunk at import time — no LLM required. Each chunk gets lightweight `mentions`
relations derived from camelCase/PascalCase identifiers, backtick-quoted terms, ALL-CAPS acronyms,
and tags. This means `memory neighbors` and `memory by_entity` return links immediately after
import. Use `--no-link` to opt out (e.g. for a corpus with no named entities).

**Recommended import flow for fully linked memory:**

```shell
# Step 1 — import with relations on by default (entity links extracted automatically):
artesian backfill ./memory-export

# Step 2 — optional LLM semantic pass: groups near-duplicates and builds higher-tier structure.
# Requires [acc.compressor] or [acc.judge] in artesian.toml (any Claude / local Ollama endpoint).
artesian consolidate --allow-llm

# Or combine both steps in one command (consolidate prints a note and skips if no LLM is found):
artesian backfill ./memory-export --consolidate
```

The `consolidate` step is always additive and safe to re-run. Without it, entity-relation links are
still present and `memory.find` + `neighbors` work — consolidate only adds LLM semantic grouping.

For a non-expert second project/user on the same Qdrant, use the wrapper:

```shell
artesian onboard my-project ./memory-export \
  --qdrant-url http://HOST:6333 --user-id user-a
```

Each project gets its own collection. `user_id` is also written as payload tenancy metadata inside
the project collection.

**Many agents on one project (Flume teams).** A team of agents shares the project collection and
reads each other's `shared` knowledge while keeping per-teammate `agent`/`task` scratch isolated —
no extra setup beyond the shared backend. See [teams.md](teams.md) and
[concurrency.md](concurrency.md).

### Daily workflow (once onboarded)

After `init`/`onboard`, every task can ride the same memory:

```shell
# Run a task to completion against a goal, memory-first each turn.
# The worker gets goal-relevant recall in $ARTESIAN_RECALL; each turn's
# outcome is committed run-scoped so it never clogs durable memory.
artesian loop --goal "cargo test" \
  --worker-cmd "codex exec 'fix the failing tests, using $ARTESIAN_RECALL'" \
  --max-turns 10 \
  --max-wall-secs 3600

# Emergency stop: create ~/.artesian/STOP (or set ARTESIAN_STOP_FILE to another
# path). Run logs are JSONL under ~/.artesian/runs/ unless ARTESIAN_RUNS_DIR is set.

# Reclaim any orphaned / runaway / hung teammate processes (also runs
# automatically before each new spawn). Over MCP: the team.gc tool.
artesian team gc --ttl-secs 3600 --heartbeat-timeout-secs 600
```

See [loop-engineering.md](loop-engineering.md) for the recall→act→verify→commit cycle and
[teams.md](teams.md) for orchestration.

**Working offline (laptop leaves the LAN).** Mirror a shared Qdrant collection into a local one
before you go, work against the local copy, then mirror back on return — one command each way, so an
agent can do it over MCP. Endpoints live in your local config, never in the repo:

```shell
# before leaving: LAN -> local docker qdrant
artesian replicate --from-url http://HOST:6333 --to-url http://localhost:6333 --collection my-project
# on return: local -> LAN (merges by point id)
artesian replicate --from-url http://localhost:6333 --to-url http://HOST:6333 --collection my-project
# health-only check, no copy
artesian replicate --from-url http://HOST:6333 --to-url http://localhost:6333 --collection my-project --status
```

**Shared memory over the network.** By default `artesian-mcp` speaks MCP over stdio (one local
client). Built with `--features http`, it can also serve **streamable HTTP** so several machines on
a LAN share one memory server:

```shell
artesian-mcp --transport http --bind 0.0.0.0:8080 --backend qdrant --collection my-project
# clients connect to http://HOST:8080/mcp
```

Bind to a trusted interface only — no auth is enforced at this layer; front it with your own
reverse proxy / network policy if exposed.

**More vector engines.** `qdrant` and `sqlite-vec` are the wired-in `--backend` choices; any other
vector store (e.g. PostgreSQL + `pgvector`) is a thin feature-gated `VectorStore` adapter that
inherits the same chunk-on-store, hybrid RRF, and tenancy. See
[backends.md](backends.md#adding-a-vector-backend-the-vectorstore-adapter-pattern).

See [modes.md](modes.md), [memory.md](memory.md), [concurrency.md](concurrency.md),
[teams.md](teams.md), [backends.md](backends.md), [upgrades.md](upgrades.md),
[sizing-and-deployment.md](sizing-and-deployment.md).

---

## B. AI-agent path (recipe any agent can follow)

Any agent (Codex, Claude Code, Gemini CLI, opencode, …) can bring Artesian up from these minimal,
deterministic steps. Ask the human operator for the four inputs, then execute — **idempotently and
without breaking anything**.

**Inputs to collect from the operator:**
- `mode`: `memory` (default) | `orchestrate` | `full` | `advanced`
- `backend`: `files` (default) | `sqlite-vec` | `qdrant`
- `qdrant_url` and API key — only if `backend = qdrant`; `qdrant_rest_url` is optional on default
  ports
- `project`: the project name (becomes the collection / OKF bundle scope) and the path to any
  existing memory to backfill

**Steps:**
1. Build or locate the binary: `cargo build --workspace` (or use a prebuilt `artesian`).
2. `artesian init --project <project> --backend <backend> [--qdrant-url …]`. This is
   idempotent and only writes Artesian's own MCP entry; it must NOT touch unrelated config.
3. If `backend = qdrant`: verify the server is reachable (`/healthz`) and that the collection's
   compat metadata (model + dim) matches the pinned model; if it mismatches, STOP and ask — run
   `artesian migrate` rather than mixing vector spaces.
4. Backfill the project's existing memory/tasks into the OKF bundle and selected backend:
   `artesian backfill <path>` (idempotent, content-hash dedup; never deletes the originals).
   Relations are extracted automatically — no LLM needed. To opt out: `--no-link`.
   To also run LLM consolidation in one step (requires [acc.compressor/judge]): `--consolidate`.
5. Verify: `artesian memory store "<probe>"` then `artesian memory find "<probe>"` returns it; report
   the backend, collection, and counts back to the operator. Optionally verify link quality with
   `artesian memory neighbors <node_id>` or `artesian memory by_entity <entity>`.
6. Report what changed (config entries added, records backfilled) and what was left untouched.

**Hard guardrails for the agent (do NOT violate):**
- Never delete or overwrite existing memory or unrelated MCP/config; `init` and `backfill` are
  additive/idempotent.
- Keep secrets (API keys) out of git; store them where the operator specifies.
- Do not change the pinned embedding model for an existing collection — that needs `migrate`
  (rebuild from OKF), not an in-place switch.
- `orchestrate`/`full` only when the operator asked for it; `memory` mode must not change how the
  operator already drives the agent.
- Do not `git push` or perform outward-facing actions without explicit operator approval.

This recipe is the canonical bring-up; `AGENTS.md` points here so any agent picks it up.

### Ready-to-paste setup prompt

Hand this to an agent (Claude Code, Codex, …) on the target machine. The user fills only the
`CONFIGURE` block — collection name, the note directories, and which backend (local by default, or a
shared Qdrant). Everything else runs autonomously.

```text
You are an agent setting up this machine (macOS/Linux). Install and configure Artesian
(https://github.com/aquifer-labs/artesian) as durable, cross-session memory for AI agents, import
the user's existing Markdown notes, and wire agents to use it. Act autonomously through the steps
below; the user fills only the CONFIGURE block.

==================== CONFIGURE BEFORE RUNNING ====================
# Collection / project name (your choice, lowercase):
COLLECTION="personal"

# Root directories with .md notes (one per line; nested subfolders are scanned RECURSIVELY, so
# list only the roots):
DIRS=(
  "/absolute/path/to/notes"
  "/absolute/path/to/another"
)

# Backend — pick ONE:
#   Local (default; private, no server, works offline):
BACKEND_ARGS=(--backend sqlite-vec)
#   OR a shared Qdrant (several machines share one store): comment the line above, uncomment the
#   three lines below, and fill the URL + API key.
# QDRANT_URL="http://your-qdrant-host:6333"
# QDRANT_API_KEY="your-qdrant-api-key"
# BACKEND_ARGS=(--backend qdrant --qdrant-url "$QDRANT_URL" --qdrant-api-key-env QDRANT_API_KEY)
=================================================================

Rules:
- Non-destructive: never delete or modify the user's .md files — only read/index them. onboard and
  backfill are idempotent and recursive (re-runs add no duplicates).
- Secrets only in the shell env / local config, never in git.
- Do not change the pinned embedding model of an existing collection (run `artesian migrate` if it
  ever mismatches).

Steps:

1) Install (if Homebrew is missing, install it first):
   brew install aquifer-labs/tap/artesian
   artesian --help

2) Qdrant variant only — export the API key for this session and persist it:
   export QDRANT_API_KEY="$QDRANT_API_KEY"
   grep -q QDRANT_API_KEY ~/.zshrc 2>/dev/null || echo "export QDRANT_API_KEY=\"$QDRANT_API_KEY\"" >> ~/.zshrc
   curl -fsS -H "api-key: $QDRANT_API_KEY" "$QDRANT_URL/healthz" && echo " qdrant ok"

3) Set up + import the first directory (writes ~/artesian/artesian.toml, registers the
   artesian-memory MCP server for Codex/Zed/Claude Code, and recursively imports the directory).
   Entity relations are extracted automatically (no LLM); pass --no-link to opt out:
   mkdir -p ~/artesian && cd ~/artesian
   artesian onboard "$COLLECTION" "${DIRS[0]}" --collection "$COLLECTION" "${BACKEND_ARGS[@]}"

4) Import the remaining directories (recursive, idempotent, entity relations on by default):
   cd ~/artesian
   for d in "${DIRS[@]:1}"; do artesian backfill "$d"; done

   Optional — run the LLM consolidation pass after all directories are imported.
   Requires ANTHROPIC_API_KEY or another LLM configured in artesian.toml.
   Without it, entity links are still present; consolidate only adds semantic grouping:
   artesian consolidate --allow-llm

5) Verify:
   artesian memory find "a topic from the notes" --limit 5
   artesian doctor

6) Wire agents and retire the old Markdown workflow. The `artesian-memory` MCP server is now
   registered at USER scope for Claude Code, Codex, and Zed (tools: memory.find / memory.context /
   memory.store / memory.anchor.*). With the operator's permission (or, if none is specified, in
   every AGENTS.md / CLAUDE.md you judge relevant), update those files so memory work goes through
   Artesian and the Markdown notes become a reference archive, not the mandatory workflow:

   a) Insert this block ABOVE any older memory/notes rules, so it is the primary instruction the
      agent reads first:

      ## Memory — Artesian (primary)
      This project uses the `artesian-memory` MCP server for durable, cross-session memory and task
      state. Do all memory/context work through it; the Markdown notes below are LEGACY reference
      only, not a workflow you must maintain.
      - Before non-trivial work, recall the relevant slice with `memory.context` (or `memory.find`)
        — do not re-read whole files.
      - For a goal/task, call `memory.context` with the goal to get a bounded packet (the goal, the
        invariants that must hold, the last failed check, and the most relevant memory).
      - After a durable decision or learning, `memory.store` it (concise, reusable). Record durable
        project rules once with tag `invariant` — they are always injected into goal packets.
      - Track task lifecycle (done / blocked / handoff / decisions) in Artesian, not in Markdown.

   b) Demote any pre-existing MANDATORY Markdown rules that sit above this block (e.g. "always read
      STATUS.md / handoff.md / memory/*.md first", "record decisions in decisions.md"): rewrite them
      as "legacy / historical reference" and drop the "mandatory" / "always" framing, so agents stop
      maintaining memory in Markdown in parallel. Do NOT delete the Markdown files — they remain an
      archive; only their authority changes.

   (Registration is user-scoped, so all three agents pick it up automatically; the project
   `.mcp.json` is an extra per-project copy for repos that prefer it.)

7) Report: what was installed, the collection, how many records were imported per directory, that
   the MCP server is registered, and where to paste the memory block.
```

Run it once per project — change `COLLECTION` and `DIRS` (and the backend) each time.

### Updating & health

Artesian installs via Homebrew, so a binary update keeps your config, MCP registrations, and stored
memory untouched:

```shell
artesian update    # convenience wrapper around `brew upgrade aquifer-labs/tap/artesian`
artesian doctor    # verify binary, config, backend reachability, collection compat, MCP registrations
```

`artesian doctor` prints the exact fix for anything that drifted (e.g. `artesian memory rebuild` or
`artesian migrate` if an embedding model ever changes) and exits non-zero on a critical problem, so
it is safe to run in scripts after an upgrade.
