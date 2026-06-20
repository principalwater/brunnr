<!-- SPDX-License-Identifier: Apache-2.0 -->

# Why Rust — the concrete wins

Artesian is written in Rust on purpose, not for fashion. Here is what you actually *get* because the
core is Rust. Each point is a user-visible benefit, not a language preference.

## One core, many surfaces (write once, run many)

The same audited core ships as a **CLI**, an **MCP server**, a **daemon**, and (planned) a native
**Python wheel** and a **WASM** module — all from one codebase, with no second implementation to
drift out of sync. Python and TypeScript memory layers re-implement their logic per language;
Artesian binds thin language adapters to a single core. (Write-once is bounded by what Rust allows:
native wheels are built per platform via maturin / cibuildwheel; WASM covers the browser and edge —
but the *logic* is written once.)

## In-process, no-GIL hot path

Retrieval and the ACC qualify-gate run **inside your process** at microsecond latency, with
embed-on-write and no inter-process serialization tax. A Python memory layer pays the GIL plus
serialization on every call; a networked memory service pays a round-trip. The control loop runs on
every turn of a long-running loop, so this is the difference between memory that is effectively free
and memory that is a per-call line item.

## A converged substrate in one process

Vector search, full-text search, and a transactional log live in **one embedded engine** — not four
networked services. This avoids *polyglot persistence* (separate vector + document + graph + SQL
stores = four failure modes and a coordination tax). Shipping a converged substrate as a single
binary is only practical in a systems language.

## Fearless concurrency = the multi-writer moat

Sharing memory across many agents and many operators **without corruption** is a concurrency
problem. Naive file coordination silently corrupts under concurrent writes; bolting locks onto it
rebuilds a database badly. Rust's ownership model lets us build a **transactional multi-writer log**
with optimistic concurrency safely — the exact thing stitched-together stacks tend to get wrong.

## Tiny static binary, runs anywhere

A single dependency-free binary drops into any loop, CI job, git worktree, or Docker sandbox — no
runtime to install, no container required. That is what makes "just add the memory brick" actually
cheap.

## A storage moat to build on

Because the substrate is ours and in-process, we can implement **computable embeddings** (recompute
on a pruned graph instead of storing every vector — far less disk) and the transactional log **in
the same binary** — hard to match from a stack that wires together separate services.

## Memory-safe, long-running

A control plane runs for days inside a loop. Memory safety without a garbage collector means
predictable latency and no whole-class crashes — reliability that matters when the process is the
thing keeping your agent from forgetting.

---

**We keep it honest.** Every claim here is meant to be measurable: footprint, latency, and
tokens-per-iteration are tracked by `gauge` and the benchmark suite, so the Rust advantage is
provable rather than asserted.
