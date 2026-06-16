<!-- SPDX-License-Identifier: Apache-2.0 -->

# Onboarding

Two ways to bring Brunnr up: **a human follows the Quickstart**, or **an AI agent follows the
agent recipe** below. Both are non-destructive and idempotent — running them twice changes
nothing the second time, and they never delete existing memory or overwrite unrelated config.

Brunnr runs with **zero configuration** in `memory` mode on the Files (OKF) backend. Add a vector
backend or orchestration only when you want them. Sensible defaults everywhere.

---

## A. Human path (Quickstart)

Install the `brunnr` CLI (see the [README](../README.md#install)), or run from source by prefixing
any command with `cargo run -p brunnr-cli -- ` instead of `brunnr`.

```shell
# memory mode, zero-infra Files (OKF) backend — the default
brunnr init
brunnr memory store "Brunnr keeps durable context" --tag bootstrap
brunnr memory find durable
```

Pick a backend (config choice, not a code change):

```shell
brunnr init --backend sqlite-vec          # local hybrid, zero infra
brunnr init --backend qdrant \            # shared / multi-user
  --project my-project --qdrant-url http://HOST:6333
```

For Qdrant, one URL is enough on default ports: `:6333` is treated as REST and the gRPC sibling
`:6334` is derived; `:6334` derives REST `:6333`. If you use custom ports, pass both
`--qdrant-url` and `--qdrant-rest-url`. `init` and import commands preflight both endpoints
(gRPC health + REST `/healthz` + auth) and fail with the exact endpoint that is wrong.

`brunnr init` detects installed agent CLIs and writes the MCP registration for each (Claude Code,
Codex, Zed) pointing at `brunnr-mcp` with the pinned embedding model and behavior-guiding tool
descriptions. Then drive your agent exactly as before — it now has `memory.find` / `memory.store`.

Backfill existing notes (idempotent), and explore modes:

```shell
brunnr backfill ./memory-export   # md/json + task md -> OKF/Thingr
brunnr memory context "what matters now"
```

`backfill` is robust: a bad file is skipped and reported, not fatal. Markdown is section-chunked
by heading, an OKF `index.md` catalog is generated, task/status markdown is routed into Thingr, and
the command prints a JSON summary with `{scanned, imported, skipped_duplicates, failed}` counts.
After import it prints the next opt-in step, `brunnr consolidate`, for LLM semantic consolidation.

For a non-expert second project/user on the same Qdrant, use the wrapper:

```shell
brunnr onboard my-project ./memory-export \
  --qdrant-url http://HOST:6333 --user-id user-a
```

Each project gets its own collection. `user_id` is also written as payload tenancy metadata inside
the project collection.

**Many agents on one project (Hirð teams).** A team of agents shares the project collection and
reads each other's `shared` knowledge while keeping per-teammate `agent`/`task` scratch isolated —
no extra setup beyond the shared backend. See [teams.md](teams.md) and
[concurrency.md](concurrency.md).

**More vector engines.** `qdrant` and `sqlite-vec` are the wired-in `--backend` choices; any other
vector store (e.g. PostgreSQL + `pgvector`) is a thin feature-gated `VectorStore` adapter that
inherits the same chunk-on-store, hybrid RRF, and tenancy. See
[backends.md](backends.md#adding-a-vector-backend-the-vectorstore-adapter-pattern).

See [modes.md](modes.md), [memory.md](memory.md), [concurrency.md](concurrency.md),
[teams.md](teams.md), [backends.md](backends.md), [upgrades.md](upgrades.md).

---

## B. AI-agent path (recipe any agent can follow)

Any agent (Codex, Claude Code, Gemini CLI, opencode, …) can bring Brunnr up from these minimal,
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
1. Build or locate the binary: `cargo build --workspace` (or use a prebuilt `brunnr`).
2. `brunnr init --project <project> --backend <backend> [--qdrant-url …]`. This is
   idempotent and only writes Brunnr's own MCP entry; it must NOT touch unrelated config.
3. If `backend = qdrant`: verify the server is reachable (`/healthz`) and that the collection's
   compat metadata (model + dim) matches the pinned model; if it mismatches, STOP and ask — run
   `brunnr migrate` rather than mixing vector spaces.
4. Backfill the project's existing memory/tasks into the OKF bundle and selected backend:
   `brunnr backfill <path>` (idempotent, content-hash dedup; never deletes the originals).
5. Verify: `brunnr memory store "<probe>"` then `brunnr memory find "<probe>"` returns it; report
   the backend, collection, and counts back to the operator.
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
