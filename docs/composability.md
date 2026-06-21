<!-- SPDX-License-Identifier: Apache-2.0 -->

# Composability — Artesian is built like LEGO

Artesian is not a monolith you must adopt whole. Every component is a **separately usable brick**:
take only the piece you need, and bring your own for the rest. If all you want is the ACC control
plane over your existing vector database, take exactly that — keep your store, your orchestration,
and your pipeline.

This is deliberate. The agent stack is converging on a small set of composable primitives
(scheduling, isolation, skills, connectors, verification, and **memory**). You should be able to slot
in *just the memory-control brick* without rewriting how you drive your agent.

## The bricks

| Brick (crate) | Role | Standalone? | Bring-your-own / swap with |
|---|---|---|---|
| `aquifer` | Token-efficient memory + retrieval (`VectorStore`: sqlite-vec / Qdrant / pgvector; chunking, small-to-big, hybrid RRF, reranking, semantic cache) | Yes | your own vector DB, or files / mem0 by implementing `RecallStore` |
| `headgate` | **ACC control plane** — bounded CCS, qualify-gate, commit-loop, pluggable judge + compressor | Yes — over *any* `RecallStore` | the unique brick; pair it with any data plane |
| `gauge` | Evaluation — footprint / drift / hallucination, plus recall and agentic (memory-guides-action) benches | Yes | your own eval harness |
| `basin` / `wellfield` / `headrace` | Orchestration / teams / task queue | Optional | your own multi-agent system |
| `sandbox` | Optional Docker isolation for workers | Optional | your own isolation |
| `artesian-mcp` | MCP server exposing a *selectable subset* of tools | Yes | any MCP client (Codex, Claude Code, Zed) |
| `artesian-cli` | CLI + the `artesiand` daemon | Yes | — |

The seams are trait-based — `RecallStore`, `VectorStore`, `Compressor`, `Agent` — so each brick is
swappable without touching the others.

## Composition recipes

- **Just ACC over my own store.** Use `headgate`. Implement `RecallStore` for your database (files,
  mem0, or any vector DB). Over MCP, enable only the commit / qualify tools.
- **Just token-efficient memory.** Use `aquifer` + the `artesian-mcp` memory tools. Bring your own
  orchestration and agent loop.
- **ACC + memory, my own multi-agent system.** Use `headgate` + `aquifer`; skip
  `basin` / `wellfield`.
- **Everything.** The full stack: memory + control plane + orchestration + sandbox + eval.

## How you compose

- **Cargo features** — depend on only the crates and features you need.
- **MCP tool subset** — enable only the specific tools (memory-only, or ACC-commit-only) in your
  MCP client config.
- **Python bindings** (planned) — import only the functions / decorators you use; the native core
  runs in-process. See [Why Rust](why-rust.md).
- **Daemon API** — drive `artesiand` from any language over its local interface.

## Composes *with*, not against

Artesian is a **control plane**; bring your own **data plane**. It is designed to sit over — not
replace — what you already run:

- **Stores:** Qdrant, pgvector, sqlite-vec, plain files, or mem0 — behind `RecallStore`.
- **Compression:** an external byte-level compressor can plug in as an optional `Compressor` under
  the control plane — Artesian governs the committed *state*; the compressor can shrink the *bytes*
  of a large artifact before it is qualified.
- **Agents:** any MCP-speaking agent (Codex, Claude Code, Zed, opencode).

## Portability — resume, not just recall

Artesian can export its committed *working context* (what the agent holds in force now) plus a
lifecycle log as a portable [working-context bundle](kit-format.md), so another runtime can *resume*
the loop instead of re-deriving it. The bundle owns only that layer and references your unit store —
it composes with the memory-unit formats you already use rather than replacing them.

```sh
artesian kit export --format bundle --output ./wc
artesian kit import ./wc
```

Take one brick or the whole set. That is the point.
