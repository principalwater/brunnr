<!-- SPDX-License-Identifier: Apache-2.0 -->

# Why Artesian — and how it compares

## Memory CONTROL: the wedge

Most agent memory systems solve retrieval: surface relevant records so the agent has more context.
That is necessary, but not sufficient. Retrieval alone does not answer three questions that
matter at scale:

1. **What is the agent's committed view of the world right now?** Retrieval gives a ranked list;
   it does not define a bounded, authoritative state.
2. **What qualifies to enter that view?** Without a gate, every write is trusted equally — drift,
   hallucination, and footprint inflate unchecked.
3. **How do you measure and control that drift?** Without a bench, memory quality degrades silently.

Bousetouane (arXiv:2601.11653, *"AI Agents Need Memory Control Over More Context"*, Jan 2026)
formalises this as the **Agent Cognitive Compressor (ACC)**: a control loop that separates the
*recall* channel (read from any retrieval store) from the *commit* channel (what is written into a
bounded, schema-governed **Committed Context State / CCS**). A **qualify-gate** sits between them:
only information that passes the gate — verified, relevant, non-redundant — enters the CCS.

Artesian is a ground-up implementation of this model, layered as a **control plane over any
retrieval store** (Aquifer/OKF, sqlite-vec, Qdrant, mem0, Anthropic, or any `MemoryBackend`
adapter):

```
         ┌─────────────────────────────────────────────────────────┐
         │                   Artesian control plane                 │
         │                                                           │
  recall │  memory.find   ──►  qualify-gate  ──►  CCS (bounded)    │  commit
  ───────┼──────────────────────────────────────────────────────────┼────────►
         │  any VectorStore/                  headgate (Step 4)     │  memory.store
         │  FilesBackend                                             │
         └─────────────────────────────────────────────────────────┘
```

The qualify-gate is today approximated by the **judge** role in `orchestrate` mode (verifiers +
accept/reject loop); `headgate` — the dedicated CCS controller with schema-state tracking and
drift/hallucination/footprint metrics — is the Step 4 build. No OSS system implements ACC fully
today; Artesian is the first-mover.

### Why this matters beyond RAG

| | Pure RAG | Artesian ACC |
|---|---|---|
| State model | stateless query → ranked list | **bounded CCS** — authoritative committed state |
| Write path | append-only, all equal | qualify-gate: only verified, non-redundant, non-drifted entries enter CCS |
| Drift control | none | judge-eval of drift / hallucination / footprint per cycle |
| Recall cost | ~6–10 k tokens/query (mem0 benchmark) | **~1 k tokens/query** (chunked OKF + small-to-big + adaptive budget) |
| Composes with | your retrieval store | **any retrieval store** (including mem0, Anthropic memory, existing Qdrant) |

## What Artesian is — and is not

Artesian is, first, **durable, semantic memory your agents own**: the decisions, facts, and context
they accumulate across sessions, kept in portable Open Knowledge Format markdown you can read,
commit, and carry anywhere. That is the flagship — use *only* memory and nothing else about how
you run your agent changes. Optionally, the same store is also an orchestration and agent-team
layer (composable components you opt into, never required).

It is **not**:

- a **cloud memory service you rent** — Artesian runs locally; writes are free (no per-write LLM
  call) and your data never leaves your machine;
- a **code-structure index** like [Codebase-Memory](https://github.com/DeusData/codebase-memory-mcp)
  — that is a parsed graph of *what your code is*; Artesian stores *what your agent learns*, and
  the two compose;
- **just a conversation log** — it is consolidated, retrievable, tiered knowledge with a qualify
  gate.

## How it compares

Against TencentDB Agent Memory specifically, the key differences:

| | TencentDB Agent Memory | Artesian |
|---|---|---|
| Scope | Memory only (capture → extract → recall) | Memory **+ ACC control plane + task tracking + master/worker/judge orchestration + sandbox** |
| Integration | OpenClaw plugin / Hermes provider (framework-coupled) | **MCP-first, agent-agnostic** (Claude Code, Codex, Zed, opencode, …) + pluggable `Agent` adapters |
| Runtime | Node ≥22.16 + TypeScript | **Rust** — single static binary, no runtime |
| Vector store | SQLite + sqlite-vec (local-first; remote on roadmap) | **Pluggable `VectorStore`**: Files(OKF) / sqlite-vec / Qdrant (+ TencentDB-style adapter possible) |
| On-disk format | bespoke markdown/JSONL layout | **Open Knowledge Format (OKF)** — vendor-neutral, portable, interop with the OKF ecosystem |
| Concurrency | single-user, local-first | **multi-project + multi-user + parallel** (collection-per-project + payload tenancy) — see [concurrency.md](concurrency.md) |
| Cross-tool memory | within its host framework | **neutral shared store both Claude Code and Codex read** (their native memories are siloed) |
| Upgrades | — | **upgrade-survivable**: OKF = source of truth, Qdrant = rebuildable index, `migrate` + version metadata ([upgrades.md](upgrades.md)) |
| Orchestration safety | n/a (not an orchestrator) | **verifiers-as-trust-boundary, judge-sole-committer, task DAG, worker workspace isolation** |

What Artesian reuses from TencentDB (with credit): the L0–L3 tiering, hybrid+RRF retrieval, the
markdown white-box principle, `node_id` drill-down, and the benchmark-rigor mindset. A
TencentDB-style symbolic Mermaid "task canvas" for short-term memory is a natural future addition
on top of Artesian's WorkingMemory + session anchor.

**One-line positioning:** Artesian is a **memory controller** — an ACC control plane that layers
bounded committed context and a qualify gate over any retrieval store — with local-first Rust
storage, MCP-first integration, and optional master/worker/judge orchestration sharing the same
store. Use as little (just `memory` mode, ~1 k tokens/query retrieval) or as much (`full`, with
ACC qualify-gate and orchestration) as you want.

## Direct competitors (general agent memory)

These solve the same core problem — durable memory for agents — and are the honest comparison set:

- **[mem0](https://github.com/mem0ai/mem0)** (Apache-2.0) — the most prominent. An LLM **extracts
  facts on every write**, stored with entity linking; hybrid semantic + BM25 + temporal retrieval;
  strong published numbers (LoCoMo 91.6, LongMemEval 94.8), broad vector-DB and LLM support, and a
  hosted cloud. **Artesian's wedge:** writes are **free and local** (no per-write LLM call), memory
  is **white-box OKF markdown you own** (not an opaque or rented store), it runs **zero-infra**,
  and it is **MCP-first / integrate-anything**. Additionally: Artesian composes *with* mem0 as a
  retrieval backend under the ACC control plane — they are not mutually exclusive. We aim to match
  mem0's retrieval quality (opt-in LLM consolidation, entity/temporal signals are on the roadmap)
  while keeping the zero-cost, own-your-data default.
- **[Zep / Graphiti](https://github.com/getzep/graphiti)** — temporal knowledge-graph memory with
  strong LongMemEval / DMR numbers; graph-centric and service-oriented. Artesian stays files-first
  and vendor-neutral (graph relations are a roadmap addition, not a required backend).
- **[Letta / MemGPT](https://github.com/letta-ai/letta)** — an agent "memory OS" with a server and
  its own agent runtime. Artesian is lighter and non-intrusive: memory you add to *your* agent over
  MCP, not a runtime you adopt.

Honest take: these are well-funded and benchmark-strong. Artesian does not try to out-platform
them — it wins on **ownership, simplicity, zero-cost local writes, freedom to integrate**, and
on being the first to implement the ACC control-plane model: a qualify gate, bounded committed
state, and drift/hallucination/footprint measurement as first-class features. A standardized
comparison on LongMemEval / LoCoMo is planned (see [benchmarks](../benchmarks/README.md)).

## Adjacent / related projects

- **[open-engram](https://github.com/Open-Nucleus/open-engram)** — a brain-inspired memory
  *library* (TypeScript): sensory → working → episodic → semantic stores, a multi-stage
  consolidation pipeline, and RFR-scored demotion. Memory-only, framework-SDK (Mastra/LangChain.js),
  no MCP, no orchestration. Strongly validates the **consolidation** direction Artesian is taking;
  Artesian differs by being Rust + MCP-first + pluggable backends + orchestration, not a
  single-runtime library.
- **[openrelay](https://github.com/romgX/openrelay)** — a model **quota aggregator / router** with
  a web dashboard that bridges credentials and routes requests from any tool to any provider. This
  is a **different layer** (model access), not memory or orchestration; it is *complementary* —
  Artesian could sit above an openrelay-style router. Its clean "connect any agent" UX is a good
  presentation model to learn from.
- **[h5i](https://github.com/h5i-dev/h5i)** — an **AI-aware Git sidecar** (Rust): per-commit agent
  context/reasoning in dedicated `refs/h5i/*`, **Agent Radio** typed inter-agent messaging with
  union-merge, output **token-reduction** (collapse tool output, keep recoverable raw), and
  **progressive sandbox isolation** (workspace → process → supervised → container). A **different
  layer** from Artesian — provenance, comms, and confinement over Git, not semantic retrieval — and
  **complementary**: they could compose. We borrow ideas, with credit: its typed agent-handoff
  protocol informs Artesian's orchestration handoffs, its isolation tiers inform `sandbox`, and its
  "collapse but never discard, recover by id" mirrors Artesian's L0–L3 + `node_id` drill-down.
- **[Codebase-Memory](https://github.com/DeusData/codebase-memory-mcp)** (MIT) — a Tree-Sitter
  **structural code graph** (who-calls-what, routes, impact) over MCP, single C binary, zero-infra.
  A **different kind of memory** — *what your code is*, parsed from source — versus Artesian's
  *what your agent learns*. Explicitly **complementary**: an agent can use Codebase-Memory for repo
  structure and Artesian for durable knowledge. Its local-first, deterministic, single-binary,
  commit-the-artifact philosophy mirrors and validates Artesian's own.

## Converging evidence shaping the roadmap

Karpathy's [LLM-wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f),
TencentDB's L0–L3, and open-engram's episodic→semantic consolidation all point the same way:
**curated, consolidated memory (atomic facts + entity/concept/scenario pages + an `index.md`
read-first catalog) beats flat record dumps** — for both retrieval precision and token cost. Below
~50–100 k tokens a curated wiki/index-first context can even beat vector RAG; vector retrieval
wins at larger scale. Artesian's plan is to do **both**: index-first + targeted `memory.find`,
with consolidation populating the tiers — see the memory roadmap in [memory.md](memory.md).

The ACC model (arXiv:2601.11653) provides the formal frame for why *control* over that
consolidation — not just retrieval — is the right level of abstraction. Bounded committed state
does not replace vector search; it governs what enters it.

## Acknowledgements

The memory-control framing in this document builds on Bousetouane's analysis in
**arXiv:2601.11653** (*"AI Agents Need Memory Control Over More Context"*, Jan 2026). We are
grateful for the clear formalisation of the ACC/CCS model; Artesian aims to be a concrete,
open-source implementation of those ideas.
