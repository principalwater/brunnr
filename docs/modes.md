<!-- SPDX-License-Identifier: Apache-2.0 -->

# Modes

## memory

Memory mode exposes durable memory through CLI and MCP. It does not require orchestration or sandboxing.

`brunnr.toml` selects the backend with `memory.backend`. Supported memory-mode backends are
`files`, `sqlite-vec`, and feature-gated `qdrant`.

## orchestrate

Orchestrate mode adds optional master, worker, and judge roles. Role aliases are available as `master`/`odin`, `worker`/`thor`, and `judge`/`tyr`.

## full

Full mode combines memory, orchestration, and sandboxing.

## advanced

Advanced mode adapts to an existing markdown tree or vector collection without forcing Brunnr to own the schema.
