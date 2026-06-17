<!-- SPDX-License-Identifier: Apache-2.0 -->

# ARTESIAN

Artesian is a **memory controller for AI agents**: a control plane that layers bounded committed
context, a qualify gate, and pluggable retrieval over any storage backend — so agents accumulate
knowledge they actually own, without flooding the prompt or silently drifting.

It is designed to be useful in the smallest mode first: `memory.find` / `memory.store` over MCP,
zero workflow change, zero infrastructure.

## Concept names (hydro family)

The naming family draws from hydrology — the physics of water moving through rock under pressure.

| Name | Role |
|---|---|
| **Artesian** | the product; named for artesian wells, where confined aquifer pressure drives water to the surface without pumping — the right context surfaces automatically |
| **Aquifer** | the memory store crate — the saturated layer that holds and yields knowledge |
| **headgate** | the ACC control plane (Step 4): qualify-gate + bounded Committed Context State (CCS) |
| **headrace** | the task queue crate — a headrace is the channel that delivers water to a mill |
| **Basin** | the orchestration crate — a basin collects and routes flow |
| **Flotilla** | agent teams — a flotilla is a fleet that moves together |
| **gauge** | the metrics/TUI crate — a gauge reads flow and pressure |
| **sandbox** | the Docker isolation crate — kept verbatim for clarity |
| **recharge** | consolidation: writing new knowledge back into the aquifer |
| **saturation** | the bounded-budget concept: a saturated layer cannot absorb more without overflow |
| **tap / spout** | auto-surfacing of relevant context — what flows out at the surface |
| **well / draw** | recall: drawing from the aquifer |

Canonical role strings: `master` / `worker` / `judge`. Accepted aliases: `odin` / `thor` / `tyr`.

## Origin

An *artesian well* taps a confined aquifer — a saturated rock layer sealed above and below by
impermeable rock. The hydraulic pressure of the surrounding rock, rather than pumping, drives
the water to the surface. The well only needs to reach the layer; pressure does the rest.

The same principle governs agent memory: the agent should not have to pump — flood the context
with everything — to reach what it needs. A bounded, pressurised knowledge layer should surface
the right information automatically, at low cost, in any session.

**Daniel Bernoulli** (1700–1782) gave fluid mechanics the equation that describes this pressure
(Bernoulli's principle, *Hydrodynamica*, 1738). He was the son of **Johann Bernoulli** and the
nephew of **Jacob Bernoulli** — the Jacob who discovered the *Bernoulli numbers*, the sequence
of rational constants that recur in analytic number theory. **Ada Lovelace's** Note G (1843),
appended to her translation of Menabrea's paper on Babbage's Analytical Engine, contains the
first published algorithm — and is widely regarded as the first computer program — whose purpose
was to compute Bernoulli numbers. That lineage — Bernoulli fluid dynamics → artesian well →
Bernoulli numbers → Lovelace's first algorithm → modern AI agents reading from a pressurised
knowledge layer — is the origin story of this project.

*Sources: MacTutor History of Mathematics (St Andrews), "The Bernoulli Family";
Lovelace, A. (1843). "Sketch of the Analytical Engine invented by Charles Babbage…
with Notes by the Translator", Note G.*

## Principles

- **Memory control first.** The ACC qualify-gate (arXiv:2601.11653) governs what enters the
  committed context state; retrieval alone is not enough.
- **MCP-first integration.** One universal surface for every agent tool.
- **Non-intrusive by default.** `memory` mode adds only `memory.find` / `memory.store`; the
  agent workflow is unchanged.
- **Pluggable at every seam.** Agent adapters, memory backends, vector stores, verifiers — all
  are traits, not hard-coded choices.
- **Files-first bootstrap, vector-ready architecture.** OKF markdown is the source of truth;
  vector indexes are rebuildable from it.
- **Deterministic drill-down via `node_id`.** Any record reachable by a stable, portable id.
- **No private infrastructure assumptions.** Runs fully local; no cloud dependency.
- **Open (Apache-2.0).** You own your data and your stack.
