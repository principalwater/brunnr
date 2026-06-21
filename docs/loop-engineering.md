<!-- SPDX-License-Identifier: Apache-2.0 -->

# Loop Engineering — autonomous, memory-first agent loops

> You stop typing the next prompt and instead design the loop that prompts the agent — a system
> that finds work, does it, checks it, and writes down what happened, until a goal holds. The hard
> part of a long loop is not intelligence; it is **memory**: a loop fails when the agent *forgets*.
> Artesian is the memory layer that keeps such a loop on track.

This is the concept and a mini-guide for running autonomous, multi-agent loops on top of Artesian.
It composes primitives Artesian already ships; it is **not** a separate runtime.

## The loop, in one picture

Every iteration runs the same memory-first cycle (after mem0's six-stage loop and the Claude Agent
SDK loop):

```
            ┌──────────────────────────────────────────────────────────┐
            │  recall ─► assemble ─► decide ─► act ─► observe ─► commit │
            │   (find)    (CCS)     (model)  (tools)  (verify)  (gate)  │
            └─────────────────────────────▲────────────────────────────┘
                                          └── repeat until the goal holds
```

- **recall** — pull only the high-signal slice for this step (`memory.find`), not the whole history.
- **assemble** — the bounded **Committed Context State** (CCS) is what the agent actually reads.
- **decide / act** — the model calls tools.
- **observe / verify** — a separate **judge** checks the result (the maker never grades itself).
- **commit** — the **qualify-gate** decides what durable learning enters memory.

The agent forgets between turns; the repository does not. State that must survive a turn lives
outside the context window — in Artesian's memory and the [self-repair anchor](self-repair.md), so a
loop survives compaction and disconnects.

## Three modules of orchestration

A multi-agent loop is built from three decisions (after **Skill-MAS**, arXiv:2606.18837):

1. **Task decomposition** (*the what*) — break the goal into evaluable sub-tasks with success
   criteria. Lands on the [headrace](task-tracking.md) task board.
2. **Agent engineering** (*the who*) — instantiate specialized teammates (lead / workers / judge),
   each a role + tools, possibly different models. This is a [wellfield](teams.md).
3. **Workflow orchestration** (*the how*) — choose a topology: **sequential**, **hierarchical**, or
   **loop**, with a verifier gate at each step.

## The five harness building blocks → Artesian

Loop engineering sits on a reliable *harness* (after [Learn Harness Engineering]). Each block maps to
an Artesian crate:

| Harness block | What it does | Artesian |
|---|---|---|
| **Loop** | the run-until-done control loop | `basin` orchestration + `/goal`-style stop condition |
| **Memory** | durable state across turns/sessions | `aquifer` + `headgate` (CCS) + the self-repair anchor |
| **Verification** | catch premature "done" | the **judge** role (qualify-gate / a second model) |
| **Isolation** | clean state per teammate | `sandbox` (optional Docker) + per-scope memory |
| **Tools** | observable actions | MCP tools served by `artesian-mcp` |

## Autonomy controls

Autonomous does not mean unbounded. A loop is governed by:

- a **stop condition** — run until a verifiable goal holds (tests pass, a check returns true), not
  forever;
- **budget caps** — max turns / max spend, so an open-ended prompt cannot run away;
- the **verifier gate** — accepted outcomes pass the judge before they count as done;
- **periodic fresh starts** — reset the working context to the anchor + targeted recall to fight
  drift on very long runs (the loop's memory, not its prompt, is reset);
- **per-scope memory** — `user` / `agent` / `run` scopes keep a fleet from cross-contaminating while
  still sharing a coordination memory (after mem0's memory scopes).

## Mini-guide: run a loop with different agents and models

Today, the loop is driven over MCP by a lead agent (e.g. Claude Code, Codex) using Artesian's tools.
The shape:

1. **Bind roles to agents/models.** `artesian init` detects installed agent CLIs; map lead / worker
   / judge to any of Claude / Codex / Gemini / opencode / a local model. See [modes](modes.md).
2. **Start a wellfield.** Over MCP: `agents.list` → `team.create` → `team.spawn` the teammates.
3. **Decompose + dispatch.** `team.task.add` the sub-tasks; workers `team.task.claim` and execute;
   coordinate via `team.message`.
4. **Verify before done.** The judge reviews; only judge-accepted work is marked complete.
5. **Recall + commit each turn.** Workers `memory.find` before acting and `memory.commit` durable
   learnings after — so run *N* reads what runs *1..N-1* learned.
6. **Resume anything.** On compaction/disconnect, `memory.anchor.recover` restores the plan and next
   step; export/import the working state as an [OCF](https://github.com/aquifer-labs/ocf) bundle to
   move the loop to another runtime.

For a single bounded subtask you do not need a wellfield — `orchestrate.delegate(worker)` runs one
worker under the judge gate.

> **`artesian loop` (available now).** A convenience command drives this cycle directly — it repeats
> the worker action until the goal command exits 0 (the verifier gate), writing a resume anchor to
> memory each turn, bounded by `--max-turns`:
>
> `artesian loop --goal "cargo test" --worker-cmd "codex exec 'fix the failing tests'" --max-turns 10`
>
> The worker is any shell command — a script or an agent CLI (`codex exec`, `claude -p`, …), so you
> can drive a different model per loop. `--poll` re-checks the goal each turn without a worker.

## Why memory-first

Long loops fail in documented ways — context rot (coherence decays after ~20–30 turns), goal drift,
re-ingesting one's own early mistakes as truth, repeating finished work. Every one is a memory
failure. A loop with durable, curated, semantic memory turns the circle into a spiral: each pass
writes something the next pass builds on. That memory layer is exactly what Artesian provides.

## References (prior art this builds on)

- **Claude Agent SDK — the agent loop** — <https://code.claude.com/docs/en/agent-sdk/agent-loop>
  (receive → evaluate → execute → repeat; compaction boundary; subagents; resume).
- **mem0 — Loop Engineering for AI Agents (memory-first design)** —
  <https://mem0.ai/blog/loop-engineering-for-ai-agents-memory-first-design> (the six-stage loop;
  user / agent / global memory scopes).
- **Addy Osmani — Loop Engineering** — <https://addyosmani.com/blog/loop-engineering/> (the six loop
  primitives; "the agent forgets, the repo doesn't").
- **Learn Harness Engineering** — <https://walkinglabs.github.io/learn-harness-engineering/en/>
  (Loop / Memory / Verification / Isolation / Tools; repository as the system of record).
- **Skill-MAS — Evolving Meta-Skill for Automatic Multi-Agent Systems** —
  [arXiv:2606.18837](https://arxiv.org/abs/2606.18837) (orchestration as decompose / agent-engineer /
  orchestrate, evolved by reflection).
- **ACC / Committed Context State** — [arXiv:2601.11653](https://arxiv.org/abs/2601.11653) (the
  bounded committed state the loop reads).

Related Artesian docs: [modes](modes.md) · [teams (wellfield)](teams.md) · [self-repair](self-repair.md) ·
[task-tracking (headrace)](task-tracking.md) · [orchestration (basin)](orchestration.md).
