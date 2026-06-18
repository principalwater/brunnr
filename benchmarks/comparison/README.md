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

A **30-question sample per dataset**, judge = `codex` gpt-5.5 (reasoning `xhigh`), comparing the
two recall strategies. LongMemEval on the **oracle** split. (`graded` < 30 where a `codex` call
errored and the case was skipped.)

| dataset | recall | accuracy | tokens/query | footprint vs full |
|---|---|---|---|---|
| LoCoMo | lexical | 0.103 (3/29) | 671 | 0.047 |
| LoCoMo | **vector** | **0.276 (8/29)** | 524 | 0.037 |
| LongMemEval (oracle) | lexical | 0.621 (18/29) | 2064 | 0.288 |
| LongMemEval (oracle) | **vector** | **0.867 (26/30)** | 2052 | 0.286 |

**mem0** (arXiv:2504.19413, cited — not re-run here; the paper reports *relative* figures, no
absolute accuracy/tokens in the abstract): **+26 %** LLM-as-judge over OpenAI memory on LoCoMo,
**91 %** lower p95 latency and **> 90 %** token savings vs. a full-context baseline. A
same-protocol head-to-head (run mem0 on these splits with this judge/budget) is the remaining
work to put exact numbers side by side.

**Reading these honestly:**

- **Vector recall is the accuracy win.** Switching from lexical (term-overlap) to the real
  `VectorMemoryBackend` (embedding + small-to-big + RRF) lifts LoCoMo **0.10 → 0.28** (+167 %)
  and LongMemEval **0.62 → 0.87** (+40 %) — the lexical default was a floor, exactly as expected,
  because it misses paraphrased evidence. LongMemEval-oracle vector (0.87) is a competitive
  result; LoCoMo stays harder (long multi-session temporal reasoning, exact-value answers).
- **Token efficiency holds throughout.** Committed context is **3.7–4.7 %** of the full LoCoMo
  conversation and **~29 %** of the LongMemEval-oracle history — the bounded-footprint property
  a memory *controller* is for, and the axis comparable to mem0's "vs full-context" savings.
  Vector recall even *reduces* committed tokens (524 vs 671 on LoCoMo) by surfacing fewer, more
  relevant facts.
- **Caveats:** n = 30 (still a sample, some noise); strict LLM-as-judge; LoCoMo answers are often
  exact dates/values; oracle split for LongMemEval. Not a tuned, full-dataset result.

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
