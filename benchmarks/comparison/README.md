<!-- SPDX-License-Identifier: Apache-2.0 -->

# Competitor-comparable QA benchmark (LoCoMo / LongMemEval, vs mem0)

This harness produces the **Artesian** side of a head-to-head on the two public agent-memory
QA datasets the literature reports — **LoCoMo** and **LongMemEval** — in the same shape mem0
publishes (answer accuracy via LLM-as-judge, and **tokens/query**). It is intentionally *not* a
re-run of mem0: mem0 is a Python system that needs cloud LLM calls and its own infra, so its
numbers are cited from the published paper and compared against, rather than reproduced here.

## What it measures

For each question the runner stores the conversation as recall candidates, runs one **ACC
cycle** to build a bounded committed context, asks the LLM to answer **only from that context**,
then grades the answer against gold with the same LLM (LLM-as-judge — the standard protocol):

- **accuracy** — graded-correct fraction (comparable to LoCoMo / LongMemEval "J" scores).
- **mean tokens/query** — committed-context tokens fed to the answerer; the token-efficiency
  number directly comparable to mem0's reported per-query token budget.
- **footprint_ratio** — committed tokens ÷ raw recall dump.

Retrieval defaults to lexical (term-overlap) recall over the case facts: deterministic and
dependency-free. Swap in a vector backend for a production-grade retrieval run.

## Honesty notes

- mem0 numbers must be quoted from its **paper** (Chhikara et al., *"Mem0: Building
  Production-Ready AI Agents with Scalable Long-Term Memory"*, arXiv:2504.19413, 2025) and read
  off its tables — do **not** trust second-hand figures (including any earlier draft numbers in
  `docs/positioning.md`, which should be re-verified against the source before publication).
- Cross-system comparison is only fair under a matched protocol: same dataset split, same judge
  model, same retrieval budget. State the judge model and budget with any published result.
- The Artesian numbers below were produced by this harness; the mem0 column is left to be filled
  from the cited paper under the same judge/budget you run Artesian with.

## Datasets (download separately)

Not vendored here (size + licensing). Fetch into `benchmarks/comparison/data/`:

- **LoCoMo** — <https://github.com/snap-research/locomo> (`locomo10.json`).
- **LongMemEval** — <https://github.com/xiaowu0162/LongMemEval> (`longmemeval_s.json` /
  `longmemeval_oracle.json`; also on Hugging Face).

The loaders are tolerant of the public schemas (numbered `session_N` turns for LoCoMo;
`haystack_sessions` for LongMemEval) and report how many malformed entries were skipped.

## Running

The answering/grading LLM is reached through a command. The default wraps `codex exec`
(`benchmarks/comparison/codex-complete`, model `gpt-5.5`, reasoning `xhigh`; override with
`CODEX_MODEL` / `CODEX_REASONING`). Any OpenAI-compatible endpoint works too — point
`--llm-command` at your own wrapper.

```shell
# vector recall (real embedding retrieval) needs the `vector` feature; lexical needs only `llm`.
cargo build -p gauge --features "llm vector" --bin gauge-eval

# --recall lexical (default, deterministic) | vector (embedding + RRF). Scale up with --limit.
./target/debug/gauge-eval locomo      benchmarks/comparison/data/locomo10.json          --limit 50 --recall vector
./target/debug/gauge-eval longmemeval benchmarks/comparison/data/longmemeval_oracle.json --limit 50 --recall vector --json
```

## Results

Judge = `codex` gpt-5.5 (reasoning `xhigh`), Artesian **vector recall** (real
`VectorMemoryBackend`: embedding + small-to-big + RRF), LongMemEval on the **oracle** split.
(`graded` < n where a `codex` call errored and that case was skipped.)

### Retrieval tuning: higher accuracy at equal-or-better token economy

The goal was to raise accuracy **without** spending more committed tokens. Reranking a larger
candidate pool down to a slightly smaller recall limit does exactly that — same datasets,
n = 200 (LoCoMo), `--rerank 100 --recall-limit 12 --signals`:

| dataset | config | accuracy | tokens/query | footprint vs full |
|---|---|---|---|---|
| LoCoMo | vector (baseline) | 0.370 (74/200) | 534 | 0.039 |
| LoCoMo | + rerank | 0.475 (95/200) | 662 | 0.049 |
| LoCoMo | **+ rerank, tuned** | **0.475** (94/198) | **505** | **0.037** |
| LongMemEval (oracle) | vector (baseline, n=500) | 0.699 (348/498) | 1944 | 0.343 |
| LongMemEval (oracle) | + rerank (n=500) | 0.691 (344/498) | 1948 | 0.343 |
| LongMemEval (oracle) | **+ rerank, tuned** (n=200) | **0.698** (139/199) | 2027 | — |

- **LoCoMo: +28 % accuracy *and* better economy.** 0.370 → 0.475 while the committed footprint
  drops to **3.7 %** of the full conversation (505 vs 534 baseline tokens). Reranking (a BGE
  cross-encoder over the hybrid-RRF pool) surfaces the right evidence into a slightly tighter
  budget — the lexical/RRF top-k was missing it on this noisy, multi-session haystack.
- **LongMemEval-oracle is saturated.** Reranking neither helps nor hurts (0.699 ≈ 0.698) because
  the oracle split already pre-filters the haystack to evidence sessions — there is little to
  re-rank. (The tuned LongMemEval row is an n=200 subset, so its `footprint vs full` is not
  directly comparable to the n=500 rows; committed tokens/query are.)
- The free retrieval signals (`--signals`: entity-linking + episode-context) did not move
  accuracy here; the win is reranking plus the recall-limit trim.

### Recall ablation (n = 30, shows the lexical→vector lift)

| dataset | recall | accuracy | tokens/query | footprint |
|---|---|---|---|---|
| LoCoMo | lexical | 0.103 (3/29) | 671 | 0.047 |
| LoCoMo | **vector** | **0.276 (8/29)** | 524 | 0.037 |
| LongMemEval (oracle) | lexical | 0.621 (18/29) | 2064 | 0.288 |
| LongMemEval (oracle) | **vector** | **0.867 (26/30)** | 2052 | 0.286 |

Lexical (term-overlap) recall is a floor — it misses paraphrased evidence; embedding +
small-to-big + RRF (and then reranking, above) is what works.

**mem0** (arXiv:2504.19413, cited — not re-run here). The paper reports *relative* figures only:
**+26 %** LLM-as-judge over OpenAI memory on LoCoMo, **91 %** lower p95 latency, **> 90 %** token
savings vs. a full-context baseline. A same-protocol head-to-head (running mem0 on these splits
under this judge/budget) remains future work.

**Caveats:** n = 200 / 500 samples (some noise); strict LLM-as-judge; LoCoMo answers are often
exact dates/values; oracle split for LongMemEval.

### Pipeline smoke (not a benchmark result)

A 2-question hand-written LoCoMo-shaped fixture (`samples/locomo-smoke.json`), graded by
`codex` gpt-5.5 at `low` reasoning, validates the full load → ACC → answer → grade path:

```
dataset:             locomo
cases:               2
graded:              2
accuracy:            1.000
mean tokens/query:   59.0
footprint_ratio:     0.797
```

This only proves the harness runs end to end; real numbers come from the full datasets above.
