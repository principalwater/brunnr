<!-- SPDX-License-Identifier: Apache-2.0 -->

# Brunnr Retrieval Benchmark

This benchmark measures the question agent-memory systems are judged on: **as the durable memory (or
conversation history) grows, how many tokens does each query actually cost, and is the right context
still retrieved?** It follows the framing used by long-term-memory benchmarks such as LoCoMo and
LongMemEval — bounded retrieval versus full-context replay over realistically large histories — and
is fully reproducible (`just bench-check`).

## Headline

Brunnr keeps per-query context cost roughly **constant (~1,000 tokens)** while the memory grows into
the hundreds of thousands of tokens. Full-context replay grows with the history and quickly becomes
the dominant cost; a realistic multi-session workload is already 100k+ tokens of accumulated
reasoning, tool output, and messages.

![Per-query tokens stay flat as memory grows](results/scaling.png)

| Memory / history | Full-context replay | Brunnr (`memory.context`) | Saving | Answer doc retrieved |
|---|---:|---:|---:|---:|
| ~13k tokens (180 docs) | 12,902 | 876 | 93.2% | 100% |
| ~119k tokens (1,600 docs) | 118,566 | 974 | 99.2% | 100% |
| ~478k tokens (6,400 docs) | 477,740 | 992 | 99.8% | 100% |

Brunnr sends a compact index slice plus a top-k retrieval slice regardless of how large the memory
is, so its per-query cost barely moves (876 → 974 → 992 tokens) while replay grows ~37×. This is the
same property memory systems like Mem0 report (near-constant tokens per query as history scales);
here it is measured end-to-end against the real retrieval path.

## Methodology

The harness indexes a corpus's `memory/` and `distractors/` directories through the real
`mimisbrunnr::backfill_directory` path into `SqliteVecVectorStore`, then calls `VectorMemoryBackend.find`
for each retrieval strategy. The retriever sees one undifferentiated corpus; `tasks.json`
`relevant_docs` is used only *after* retrieval, to score precision and recall. A task succeeds when
its relevant source document is in the retrieved set.

Two families of corpora run the identical harness:

- **Scaling** (procedural, deterministic — `tools/generate_corpus.py`): `xl` (~13k), `session`
  (~119k), `mid` (~478k tokens). Each doc is a distinct fact, so these isolate the cost-vs-size
  curve above.
- **Retrieval quality** (hand-authored prose with plausible near-miss distractors): `seed` (13 docs)
  and `large` (41 docs), where retrieval is genuinely hard and recall can drop — see below.

Assumptions: backend `SqliteVecVectorStore`; embedding `intfloat/multilingual-e5-small` (384-d);
hybrid SQLite-FTS/BM25 + dense search fused with RRF, then a local lexical reranker where enabled;
tokenizer `cl100k_base` via `tiktoken-rs`.

## Retrieval quality

Where documents are semantically confusable (the `large` tier), retrieval is a real trade-off:

| Strategy | Success | Tokens/query | Precision | Recall |
|---|---:|---:|---:|---:|
| Full replay | 100% | 2,988 | 0.02 | 1.00 |
| **Brunnr default** | 80% | 861 | 0.27 | 0.80 |
| Brunnr + reflection | 95% | 1,059 | 0.32 | 0.95 |
| Brunnr + multi-query | 55% | 905 | 0.18 | 0.55 |
| Brunnr + HyDE | 45% | 909 | 0.15 | 0.45 |
| Built-in memory (top-1) | 75% | 120 | 0.75 | 0.75 |
| No memory | 0% | 52 | 0.00 | 0.00 |

Brunnr's default cuts tokens while recovering the answer document in most tasks; a larger or more
confusable corpus lowers recall (the trade-off Brunnr manages). Opt-in methods stay **off by
default** — they do not help here, and a weak strategy (HyDE) genuinely fails — enable one only if a
target corpus shows a measured gain. `Tokens/query` is the context cost; `tokens/success` (in the
raw results) additionally penalizes lower success.

## Reproduce

```sh
just bench           # seed (retrieval quality)
just bench-large     # large (retrieval quality)
just bench-xl        # xl    (~13k scaling)
just bench-session   # session (~119k scaling)
just bench-mid       # mid   (~478k scaling)
just bench-check     # rerun all tiers and fail if committed results differ
python3 benchmarks/tools/plot_scaling.py   # regenerate results/scaling.svg
```

Procedural corpora regenerate deterministically with
`python3 benchmarks/tools/generate_corpus.py --out <tier> --docs N --tasks T`. A live-Qdrant smoke
proves the same retrieval path against `QdrantVectorStore`:
`cargo test -p brunnr-bench --test qdrant -- --ignored` with `QDRANT_URL` set. Each tier keeps small,
byte-reproducible artifacts (`aggregate.json`, `summary.csv`, `charts.txt`, `checksums.txt`); the
bulky `raw.jsonl` and machine-dependent `timing.jsonl` are gitignored.

## Reproducibility and integrity

1. `tasks.json` `relevant_docs` is the ground truth and is never passed to the retriever; precision
   and recall are scored only against what `memory.find` returns.
2. Retrieval arms call `MemoryBackend.find` (`crates/brunnr-bench/src/main.rs`) — there are no
   hardcoded, label-derived result sets.
3. `just bench-check` reruns every tier and fails if any committed result changes; `aggregate.json`
   records the per-tier retrieval misses, so weak strategies are visible rather than hidden.

## Scope

This measures retrieval quality and tokenizer footprint, not end-to-end answer quality. Brunnr helps
when a bounded retrieval slice can surface the answer source from a larger durable context; it does
not help if the query is underspecified, the corpus lacks the answer, or the host agent already
retrieves the right document cheaply.
