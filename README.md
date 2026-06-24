<!-- SPDX-License-Identifier: Apache-2.0 -->

# Artesian

**Memory control plane for agent loops.**

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![CI](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/aquifer-labs/artesian/actions)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org)
[![Docs](https://img.shields.io/badge/docs-aquifer--labs.github.io-1f6feb.svg)](https://aquifer-labs.github.io/artesian/)

Artesian keeps agent memory small, high-signal, and survivable across compaction — so the agent **acts on what it knows**, not re-reads everything it has ever stored.

## What it does

- **Bounded committed context (ACC)** — a qualify-gate (drift / novelty / relevance) decides what enters; the controller evicts under saturation and compresses to fit; never silently loses critical state.
- **Survives compaction** — deterministic session anchor + targeted recall; resumes correctly after any context reset, auto-compaction, or disconnect.
- **Token-efficient retrieval** — ~1,000 tokens per query regardless of memory size (99.9% saving vs full-context replay at 1M tokens; see [Proof](#proof)).
- **Private, zero-cost writes** — no per-write LLM call; local by default; LLM judge, consolidation, and compression are opt-in.
- **Composable** — ACC over any vector store; Qdrant / pgvector / sqlite-vec / files / mem0; bring your own judge and compressor; any MCP agent.

## How it works

```
recall candidates (from durable memory)
        │
        ▼  qualify-gate: drift / novelty / relevance
┌───────────────────────────────────────┐
│   Committed Context State (bounded)   │  ← what the agent sees
│   ┌───────────────────────────────┐   │
│   │ decision:  chose Rust+tokio   │   │
│   │ plan:      shard the embedder │   │
│   │ blocker:   GPU quota at limit │   │
│   └───────────────────────────────┘   │
└───────────────────────────────────────┘
        │ evict / compress under saturation
        ▼
durable memory (sqlite-vec / Qdrant / pgvector / files)
        ↑ anchor + targeted recall on any compaction
```

## Get started

```shell
# Install — pre-built binary, no Rust toolchain (macOS + Linux):
brew install aquifer-labs/tap/artesian
# (or build from source: cargo install --git https://github.com/aquifer-labs/artesian artesian-cli)

artesian init --backend sqlite-vec        # zero infrastructure
artesian memory store "chose Rust+tokio" --tag decision
artesian memory find "which language"
artesian tokens                           # how many tokens recall has saved so far
```

MCP drop-in (Claude Code, Codex, opencode — any MCP client):

```jsonc
// claude_desktop_config.json or mcp settings
{ "artesian-memory": { "command": "artesian-mcp", "args": ["--config", "artesian.toml"] } }
```

`artesian init` writes the config. `artesian perf` prints live proof that memory is working.

## Proof

### Context efficiency

Artesian's per-query cost stays flat at ~1,000 tokens while full-context replay grows 81×.

| Memory / history | Full-context replay | Artesian | Saving | Retrieved |
|---|---:|---:|---:|---:|
| ~13k tokens (180 docs) | 12,902 | 876 tokens | 93% | 100% |
| ~119k tokens (1,600 docs) | 118,566 | 974 tokens | 99.2% | 100% |
| ~478k tokens (6,400 docs) | 477,740 | 992 tokens | 99.8% | 100% |
| ~1M tokens (14,000 docs) | 1,046,431 | 1,046 tokens | **99.9%** | 100% |

```
just bench-check   # reproduce all tiers; fails if committed results differ
```

→ Full methodology, charts, large-source retrieval: [benchmarks/README.md](benchmarks/README.md)

### Retrieval quality

| Benchmark | Score | Method |
|---|---:|---|
| LoCoMo | 0.475 | vector + BGE reranking (vector-only baseline: 0.37) |
| LongMemEval (oracle) | 0.70 | vector retrieval |

### Agentic "use" (memory-guides-action)

The `gauge` agentic eval harness ships in this repo: multi-session tasks where success = correct next action given accumulated memory, not just recall of a fact (MemoryArena framing, arXiv:2602.16313). Published scores vary by model and task set — run `just bench-agentic` to measure on your own setup.

→ Methodology: [benchmarks/comparison/README.md](benchmarks/comparison/README.md)

## Composability (LEGO)

Artesian is built like LEGO — take only the pieces you need, bring your own for the rest.

→ **[docs/composability.md](docs/composability.md)**

## Composes with

| Store / layer | Role | How to use |
|---|---|---|
| **sqlite-vec** | Local vector store — zero infrastructure | `backend = "sqlite-vec"` (default `init`) |
| **Qdrant** | Shared vector store — multi-agent / multi-operator | `backend = "qdrant"` + `QDRANT_URL` |
| **pgvector** | PostgreSQL-native vectors | `features = ["pgvector"]` |
| **plain files** | Human-readable OKF markdown — no DB needed | `backend = "files"` |
| **mem0** | Existing mem0 store | Implement `headgate::RecallStore` |
| **headroom** | Data-plane compression — shrinks artifact bytes | `headroom` feature + `HeadroomCompressor` |
| **Ollama / LM Studio / mlx** | Local LLM for judge or compressor | `provider: "ollama"` / `"lm-studio"` / `"mlx"` |
| **Any OpenAI-compatible endpoint** | Cloud or self-hosted LLM | `provider: "openai-compatible"` + `base_url` |
| **Any agent CLI** | Codex / Claude Code / Gemini / opencode | `provider: "command"` — shelled out via stdin/stdout |

## Compared to

Qualitative only — architectures differ enough that cross-system benchmark numbers mislead.

| | Artesian | mem0 | LangMem | plain markdown |
|---|---|---|---|---|
| **What it is** | ACC control plane + memory | Managed memory API | LangGraph memory layer | Files + prompting |
| **Self-hosted, zero infra** | ✓ sqlite-vec or files | Cloud-first | Requires LangGraph runtime | ✓ |
| **No per-write LLM call** | ✓ | ✗ | ✗ | ✓ |
| **ACC qualify-gate** | ✓ drift/novelty/relevance | ✗ | ✗ | ✗ |
| **Compaction survival** | ✓ anchor + recall | ✗ | ✗ | ✗ |
| **Multi-writer, isolated** | ✓ optimistic CAS | partial | partial | ✗ |
| **Agentic eval harness** | ✓ ships in repo | ✗ | ✗ | ✗ |

## Docs

📖 **Full documentation site: <https://aquifer-labs.github.io/artesian/>** (rendered from `docs/`).

| | |
|---|---|
| Architecture | [docs/architecture.md](docs/architecture.md) |
| Memory internals | [docs/memory.md](docs/memory.md) |
| Backends + quantization | [docs/backends.md](docs/backends.md) |
| Self-repair after compaction | [docs/self-repair.md](docs/self-repair.md) |
| Concurrency + multi-writer | [docs/concurrency.md](docs/concurrency.md) |
| Composability | [docs/composability.md](docs/composability.md) |
| Why Rust | [docs/why-rust.md](docs/why-rust.md) |
| Positioning / prior art | [docs/positioning.md](docs/positioning.md) |
| Onboarding | [docs/onboarding.md](docs/onboarding.md) |
| Benchmarks | [benchmarks/README.md](benchmarks/README.md) |

## Contributing

All commits require a DCO `Signed-off-by:` line. All code in English. Apache-2.0 SPDX headers on new files. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) and [docs/development.md](docs/development.md).

```shell
cargo fmt --all --check
cargo test --workspace
cargo build --workspace
```

## Community

Issues and discussions: [GitHub Issues](https://github.com/aquifer-labs/artesian/issues). Pull requests welcome.

## License

Apache-2.0. See [LICENSE](LICENSE).

---

## Acknowledgments

Artesian stands on the shoulders of prior work. Artesian reuses ideas and APIs where appropriate, not third-party source code.

- **Andrej Karpathy — the "LLM wiki" pattern** — the LLM-maintained markdown knowledge base (`index.md` + `log.md` + entity pages; ingest/query/lint) that directly informs Artesian's Files/OKF backend and consolidation roadmap.
- **TencentDB Agent Memory** — L0–L3 tiering, hybrid BM25+vector RRF, node_id drill-down, sqlite-vec local-first; informs `SqliteVecBackend`.
- **Qdrant** — vector store; `QdrantBackend` via `QdrantVectorStore`.
- **Open Knowledge Format (OKF)** — Google Cloud `knowledge-catalog` (Apache-2.0) — the portable markdown+YAML knowledge-bundle format Artesian's `files` backend aligns with.
- **ACC model** — Bousetouane, *AI Agents Need Memory Control Over More Context*, arXiv:2601.11653 — the model `headgate` implements.
- **MemoryArena** — arXiv:2602.16313 — the "recall ≠ use" framing and agentic eval methodology.
- **OpenAI / Anthropic** — Codex Memories, Claude Code Agent Memory, and agent loop documentation — memory model and loop framing.
- **ApX Machine Learning — Agentic LLM Systems & Memory Architectures** — educational reference; underlying techniques grounded in primary sources cited per-topic in `docs/`.
- **h5i** — AI-aware Git sidecar; its typed agent-handoff messaging informs Artesian's orchestration handoff protocol.

Prior art also includes Cursor scaling-agents, OpenAI Symphony, and self-correcting long-running agent patterns. All acknowledgments are references to public ideas/APIs; no third-party content is reproduced.
