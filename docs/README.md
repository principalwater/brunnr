<!-- SPDX-License-Identifier: Apache-2.0 -->

# Artesian Documentation

Artesian is a multi-agent context orchestration system: pluggable agent **memory**, optional
**master / worker / judge** orchestration, and optional task tracking — all non-intrusive, so you
keep driving your agent the way you already do and simply gain speed and lower token cost.

New here? Start with **[onboarding](onboarding.md)**, then **[positioning](positioning.md)** and
**[architecture](architecture.md)**. The rest is grouped by component below.

### Concepts & overview

| Doc | What it covers |
|---|---|
| [architecture.md](architecture.md) | Top-level system architecture and crates |
| [positioning.md](positioning.md) | Why Artesian, and how it relates to adjacent projects |
| [modes.md](modes.md) | Operating modes: `memory`, `orchestrate`, `full`, `advanced` |
| [context-tree.md](context-tree.md) | Layered, priority-ordered context-md tree |

### Memory — Aquifer (the flagship)

| Doc | What it covers |
|---|---|
| [memory.md](memory.md) | Short/long-term memory, retrieval math, L0–L3 tiers |
| [backends.md](backends.md) | Backends (Files/OKF, sqlite-vec, Qdrant) and RRF per backend |
| [upgrades.md](upgrades.md) | Rebuild-from-OKF migration, Qdrant snapshots, compatibility guards |
| [self-repair.md](self-repair.md) | Surviving context auto-compaction (session anchor) |

### Orchestration — Basin

| Doc | What it covers |
|---|---|
| [orchestration.md](orchestration.md) | Master/worker/judge, topologies, model-aware bindings, router |
| [teams.md](teams.md) | Agent teams (Wellfield): vendor-neutral lead + teammates over shared memory |
| [task-tracking.md](task-tracking.md) | headrace task tracker: DAG; md / vector / external (Jira, Linear) |
| [concurrency.md](concurrency.md) | Multi-tenancy: many agents/users, parallel access, session lanes |

### Guides

| Doc | What it covers |
|---|---|
| [onboarding.md](onboarding.md) | Bring Artesian up: human Quickstart **and** an AI-agent recipe |
| [development.md](development.md) | Build, test, and contribution workflow |

### Reference

| Doc | What it covers |
|---|---|
| [../benchmarks/README.md](../benchmarks/README.md) | Retrieval benchmark: method, results, reproduce |
| [diagrams/README.md](diagrams/README.md) | Diagram sources and rendering |

---

Role names: `master` / `worker` / `judge`.
