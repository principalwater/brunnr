<!-- SPDX-License-Identifier: Apache-2.0 -->

# Artesian Roadmap — Memory Control for Agent Loops

> Status: planning anchor for the next implementation phase. This document is the **single source
> of truth** for *where we are going and why*. Verify every change against it. Keep it current.

## Vision (one paragraph)

Long-running agent loops do not fail because the model is not smart enough — they fail because the
agent **forgets**. Every well-known failure mode (context rot, goal drift, re-ingesting one's own
mistakes, repeating finished work) is a memory failure. Artesian is the **memory control plane for
agent loops**: it keeps the agent's *working* context small, high-signal, and survivable across
compaction and disconnects, so the memory **guides the next action** (not just "recalls a fact"),
**compounds** across runs, and is **owned by you** — portable across any model and any retrieval
store. We are not "another memory store"; we are the control plane that sits over one.

## The problem we are actually solving

Three gaps the field has named but no open-source system closes together:

1. **Recall ≠ use.** Systems that *saturate* recall benchmarks (LoCoMo/LongMemEval) still fail when
   memory must *guide action* — "not *can you recall attempt 12*, but *given attempts 1–46, what do
   you do on 47*" (MemoryArena, arXiv:2602.16313). Recall is necessary, not sufficient.
2. **Shared state for multi-agent × multi-operator without corruption.** Naive file coordination
   silently corrupts under concurrent writes and collapses at fleet scale; the moment you add
   locking/atomic-writes/indexing/metadata discipline "you are no longer just using files — you are
   rebuilding a database" (Oracle, *File Systems vs Databases for agent memory*). Cursor's flat
   file-lock model degraded "20 agents to the throughput of 2–3"; they moved to optimistic
   concurrency. The owned/self-hosted version of this does not exist yet.
3. **Context survival across auto-compaction *and* disconnect.** Everyone calls memory "the durable
   spine" of a loop, then implements it as "write to a markdown file." No system does real
   self-repair (detect the compaction/reconnect boundary → re-anchor + targeted recall *before* the
   next action).

## The architectural principle: interface ≠ substrate

Do **not** pick "vector DB" *or* "files." They are different axes (Oracle):

- **Filesystem wins as the *interface*** — LLMs are pretrained on repos/markdown/grep; humans can
  read, edit, diff, and `git` it. This is our **OKF** layer (human-readable md/json).
- **Database wins as the *substrate*** — concurrency, ACID, semantic retrieval, auditability. This
  is required the instant memory is *shared* across agents/operators.
- **Avoid polyglot persistence** — separate vector + doc + graph + SQL services = four failure
  modes and a coordination tax. Converge them.

Artesian's differentiated shape: **a human-readable file *interface* over a transactional,
semantically-indexed, multi-writer *substrate*, in one Rust binary**, optionally sandboxed. Files
ergonomics + database guarantees + an ACC control plane — owned and self-hosted.

## Why Rust (earned, not assumed)

- **Single static binary, zero-dep** → drop into any loop / worktree / Docker sandbox without a
  Python runtime; lets us *converge* the substrate in-process (vector + FTS + transactions in one
  engine) instead of orchestrating four services.
- **Concurrency without a GIL** → multi-agent / multi-operator shared state *is* a concurrency
  problem; this is a structural advantage over Python memory layers.
- **Microsecond hot-path + embed-on-write** → token economy and latency are not ran­time-bound.
- **Memory-safe long-running daemon** → reliability for a process that runs for days in a loop.
- **Storage moat to build** → LEANN-style *computable* embeddings (recompute on a pruned graph
  instead of storing all vectors; ~97% less storage) + a transactional multi-writer log, both in
  one binary — hard to match from a stitched Python stack.

## Where we are today (honest baseline)

- `aquifer` — `VectorStore` trait (sqlite-vec / Qdrant / pgvector), chunking, small-to-big +
  adaptive budget, hybrid RRF, **reranking (BGE)**, semantic cache. **Footprint is strong** (committed
  ≈ 3.7% / 34% of full context). Accuracy: LoCoMo ≈ 0.475 (vector+rerank), LongMemEval-oracle ≈ 0.70.
  Reranking is the accuracy lever; HyDE/multi-query were an honest negative.
- `headgate` — ACC control plane: bounded CCS, qualify-gate, commit-loop, pluggable LLM judge +
  compressor. **First ACC implementation in OSS.**
- `gauge` — eval harness (LoCoMo/LongMemEval, LLM-as-judge, footprint/accuracy/tokens).
- `basin`/`flotilla`/`headrace`/`sandbox` — orchestration, teams, queue, Docker isolation.
- `artesian-mcp`/`artesian-cli` — MCP-first + CLI.

**Two gaps to close:** (a) the public framing still reads "memory store," not "control plane for
loops"; (b) we measure *recall* (LoCoMo/LongMemEval), not *use* (action-guiding).

---

## The roadmap (execute sequentially)

Each step lists **Goal / Where / Acceptance**. Steps 1–2 are cheap and differentiating; Step 4 is
the moat. Optional steps (6–7) are in scope. Step 8 is the final documentation sweep.

### Step 1 — Reposition the framing (docs only, cheap, do first) ✓ DONE

- **Goal:** close the witness↔code gap. Lead with *memory control for agent loops*: "recall ≠ use,"
  "own your learning loop" (swap the model, keep the company veteran), "interface ≠ substrate."
- **Where:** `README.md` (top), `docs/positioning.md` (extend the existing ACC wedge with the loop /
  recall≠use / ownership framing). Do **not** rewrite every doc here — that is Step 8.
- **Acceptance:** README headline states the control-plane-for-loops thesis; positioning cites the
  three gaps; no benchmark numbers fabricated.
- **Status:** README headline → "Memory control plane for agent loops"; positioning.md leads with
  the three gaps (recall≠use / shared-state-corruption / context-survival) before the ACC section.

### Step 2 — Measure *use*, not just recall (agentic benchmark) ✓ DONE

- **Goal:** become the only OSS memory system that benchmarks *memory-guides-action*. Keep
  LoCoMo/LongMemEval as the recall floor; add a MemoryArena-style interdependent multi-session task
  where success = correct *action* given accumulated memory. Add an honest scale lane (1M–10M tokens
  — the regime where memory is weakest and "almost nobody benchmarks").
- **Where:** `gauge` (eval), `benchmarks/comparison/`.
- **Acceptance:** a reproducible agentic-task score reported alongside recall; methodology +
  honesty notes in `benchmarks/comparison/README.md`; mem0/competitors cited only from their papers.
- **Status:** `gauge/src/agentic.rs` — `AgentTask`, `TaskSession`, `ScaleLane`, `run_agentic_eval`;
  `gauge-agent` binary; fixture at `benchmarks/comparison/samples/agent-smoke.json`;
  `benchmarks/comparison/README.md` extended with Part 2 agentic methodology + scale lane + honesty
  notes. 15/15 tests green with `--features llm`.

### Step 3 — Self-repair (survive compaction *and* disconnect) ✓ DONE

- **Goal:** detect a compaction / reconnect / session-restart boundary, then auto re-anchor
  (deterministic session anchor + targeted recall) *before* the next action. Make it a first-class,
  demoable feature.
- **Where:** `headgate` (anchor + replay), `artesian-cli`/`artesian-mcp` (hook), `docs/self-repair.md`.
- **Acceptance:** a demo that interrupts a loop mid-task (e.g. "turn 47") and resumes with the plan,
  decisions, and next step intact — no human "re-read the md" step.
- **Status:** `AnchorAnchorStore` + `recover_after_compaction` in aquifer; 2 passing tests in
  `aquifer/tests/anchor.rs`; CLI `artesian memory anchor get|set|recover`; MCP `memory.anchor.get` /
  `memory.anchor.set`; `docs/self-repair.md` updated with status table and demo recipe.

### Step 4 — Transactional multi-writer substrate + file interface (the moat)

- **Goal:** unify the OKF file *interface* and the vector *substrate* under one transactional
  commit-log (no polyglot, no flat file-locks). Per-scope isolation (operator / agent / run)
  enforced *transactionally*, not by convention. Optimistic concurrency (read free, write fails if
  state changed) per Cursor's lesson. Human edits to files are transactions (watch → reindex).
- **Where:** `aquifer` (substrate + commit-log), `sandbox` (isolation), `docs/concurrency.md`.
- **Acceptance:** N agents + M operators write the shared memory concurrently with zero corruption
  and correct isolation; a human-edited markdown file is reflected in retrieval; integrity proven by
  a concurrency stress test (the failure mode Oracle/Cursor document does not occur).

### Step 5 — Loop-native packaging (portable across agents)

- **Goal:** ship a "loop memory kit" — the stabilized anchor set (vision / per-iteration prompt /
  accumulated memory / skills) + MCP wiring, **portable across Codex and Claude Code** (the vendor
  that makes loop memory portable wins). One-command integration into any flow.
- **Where:** `artesian-cli` (`init`/kit), `artesian-mcp`, `docs/modes.md`.
- **Acceptance:** a single command wires Artesian memory into a Codex *and* a Claude Code loop with
  identical behavior; the loop's run N reads what runs 1..N-1 committed.

### Step 6 — Rust storage moat + local/council compressor (optional, in scope)

- **Goal:** (a) LEANN-style *computable* embeddings (pruned-graph recompute, ~97% less storage) as a
  `VectorStore` option; (b) pluggable **local** compressor/judge via LM Studio / `mlx_lm.server` /
  Ollama (zero token cost, private); (c) pluggable **council/judge** (panel + arbiter) for the ACC
  compressor — "the council decides, a cheaper agent executes."
- **Where:** `aquifer` (computable-embeddings store), `headgate` (compressor/judge providers).
- **Acceptance:** storage-savings benchmark vs the standard vector store at equal recall; a local
  compressor runs the ACC loop with no API token spend.

### Step 7 — headroom as an optional compressor (complementarity, optional, in scope)

- **Goal:** integrate headroom (the data-plane compression layer) as **one optional pluggable
  `Compressor`/transform** under the ACC control plane, and document the complementarity: Artesian
  governs the committed *state*; headroom can shrink the *bytes* of a large artifact before it is
  qualified. Stays optional; default build does not depend on it.
- **Where:** `headgate` (Compressor adapter, feature-gated), `docs/backends.md` + README
  "Composes with" section.
- **Acceptance:** headroom can be enabled as a compressor via config; with it off, behavior is
  unchanged; the README clearly frames "control plane (us) over compression (headroom)" as a
  supported, optional path.

### Step 8 — Documentation sweep (refine everything, do not bloat)

- **Goal:** after Steps 1–7 ship, align *all* docs with the delivered features — refine, clarify,
  de-duplicate, remove drift. No padding.
- **Where:** `README.md`, all of `docs/`, per-crate doc-comments, diagrams.
- **Acceptance:** every doc matches shipped behavior; no stale claims; concise.

---

## References (research basis for this roadmap)

- **ACC / Memory Control** — Bousetouane, *AI Agents Need Memory Control Over More Context*,
  arXiv:2601.11653 (the model `headgate` implements).
- **Recall ≠ use** — *MemoryArena*, arXiv:2602.16313.
- **Recall benchmarks** — *LongMemEval*, arXiv:2410.10813; LoCoMo (snap-research/locomo).
- **Atomic-facts memory (prior art)** — *AtomMem*, arXiv:2606.19847.
- **Loop engineering = memory engineering** — mem0, *Loop Engineering Works On Memory* (2026);
  A. Osmani, *Loop Engineering*; The New Stack, *Loop Engineering*.
- **Interface ≠ substrate; concurrency** — Oracle, *Comparing File Systems and Databases for
  Effective AI Agent Memory Management*; Cursor, *Scaling Agents* (optimistic concurrency).
- **Own the learning loop** — S. Nadella, *A frontier without an ecosystem is not stable* (2026).
- **Context rot** — *Context Rot in AI Coding Agents* (MindStudio); larger windows do not fix it.
- **Storage efficiency** — *LEANN* (computable embeddings, ~97% less storage).
- **Complementary compression** — *headroom* (data-plane compression layer, Apache-2.0).
