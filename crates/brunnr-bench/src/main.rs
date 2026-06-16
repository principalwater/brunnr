// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{bail, Context, Result};
use clap::Parser;
use mimisbrunnr::{
    backfill_directory, FastembedTextEmbedder, LocalLexicalReranker, MemoryBackend, MemoryQuery,
    Reranker, SearchHit, SqliteVecVectorStore, TextEmbedder, VectorMemoryBackend,
    VectorMemoryConfig,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use tiktoken_rs::{cl100k_base, CoreBPE};

const SUITE_VERSION: &str = "seed-honest-v1";
const BRUNNR_VERSION: &str = env!("CARGO_PKG_VERSION");
const TOKENIZER_ID: &str = "cl100k_base";
const TOKENIZER_PACKAGE: &str = "tiktoken-rs";
const EMBEDDING_MODEL: &str = "intfloat/multilingual-e5-small/384";
const DEFAULT_REPS: usize = 2;
const DEFAULT_TOP_M: usize = 8;
const DEFAULT_TOP_K: usize = 3;

#[derive(Debug, Parser)]
#[command(about = "Run the Brunnr public retrieval benchmark")]
struct Args {
    #[arg(long, default_value = "benchmarks/seed-corpus")]
    seed_corpus: PathBuf,
    #[arg(long, default_value = "benchmarks/results/sample-run")]
    results: PathBuf,
    #[arg(long, default_value_t = DEFAULT_REPS)]
    reps: usize,
    /// Include signal-arm backends (entity-overlap, temporal-decay, supersession,
    /// episode-context). Off by default so scaling tiers (xl/session/mid/mega) skip the
    /// extra backfill cost. Enable for quality/ability tiers.
    #[arg(long, default_value_t = false)]
    signal_arms: bool,
}

#[derive(Debug, Deserialize)]
struct TaskSuite {
    suite: String,
    tasks: Vec<TaskSpec>,
}

#[derive(Debug, Clone, Deserialize)]
struct TaskSpec {
    id: String,
    difficulty: String,
    #[serde(default)]
    ability: Option<String>,
    question: String,
    relevant_docs: Vec<String>,
    #[serde(default)]
    distractor_docs: Vec<String>,
}

#[derive(Debug, Clone)]
struct CorpusDoc {
    id: String,
    title: String,
    full_text: String,
    body: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArmKind {
    FullReplay,
    FullReplayCold,
    BuiltInAgentMemory,
    MdOkfIndexFirst,
    DefaultBrunnr,
    DefaultBrunnrCold,
    Hyde,
    MultiQuery,
    Reflection,
    Debate,
    NoMemory,
    // D1: entity-overlap channel
    EntityOverlap,
    // D2: temporal recency decay
    TemporalDecay,
    // D2: knowledge-update supersession
    Supersession,
    // D3: episodic context expansion
    EpisodeContext,
}

impl ArmKind {
    /// Core arms: run on every tier including large scaling tiers.
    fn core() -> &'static [Self] {
        &[
            Self::FullReplay,
            Self::FullReplayCold,
            Self::BuiltInAgentMemory,
            Self::MdOkfIndexFirst,
            Self::DefaultBrunnr,
            Self::DefaultBrunnrCold,
            Self::Hyde,
            Self::MultiQuery,
            Self::Reflection,
            Self::Debate,
            Self::NoMemory,
        ]
    }

    /// All arms including signal arms. Signal arms only run with `--signal-arms` and only
    /// produce meaningful deltas on ability-tagged tasks.
    fn all() -> &'static [Self] {
        &[
            Self::FullReplay,
            Self::FullReplayCold,
            Self::BuiltInAgentMemory,
            Self::MdOkfIndexFirst,
            Self::DefaultBrunnr,
            Self::DefaultBrunnrCold,
            Self::Hyde,
            Self::MultiQuery,
            Self::Reflection,
            Self::Debate,
            Self::NoMemory,
            Self::EntityOverlap,
            Self::TemporalDecay,
            Self::Supersession,
            Self::EpisodeContext,
        ]
    }

    fn id(self) -> &'static str {
        match self {
            Self::FullReplay => "A-full-replay",
            Self::FullReplayCold => "A-full-replay-cold-session",
            Self::BuiltInAgentMemory => "C-built-in-agent-memory",
            Self::MdOkfIndexFirst => "E-md-okf-index-first",
            Self::DefaultBrunnr => "B-default-brunnr",
            Self::DefaultBrunnrCold => "B-default-brunnr-cold-session",
            Self::Hyde => "B-plus-hyde",
            Self::MultiQuery => "B-plus-multi-query",
            Self::Reflection => "B-reflection-consolidated",
            Self::Debate => "B-plus-debate",
            Self::NoMemory => "D-no-memory",
            Self::EntityOverlap => "B-plus-entity-overlap",
            Self::TemporalDecay => "B-plus-temporal-decay",
            Self::Supersession => "B-plus-supersession",
            Self::EpisodeContext => "B-plus-episode-context",
        }
    }

    fn cache_state(self) -> &'static str {
        match self {
            Self::FullReplayCold | Self::DefaultBrunnrCold => "cold",
            _ => "warm",
        }
    }

    fn strategy(self) -> &'static str {
        match self {
            Self::FullReplay | Self::FullReplayCold => "full-corpus-replay",
            Self::BuiltInAgentMemory => "real-memory-find-top1-no-index",
            Self::MdOkfIndexFirst => "md-okf-full-index-plus-whole-file-retrieval",
            Self::DefaultBrunnr | Self::DefaultBrunnrCold => {
                "real-memory-context-index-slice-plus-find-rrf-rerank"
            }
            Self::Hyde => "real-memory-context-with-hypothetical-query",
            Self::MultiQuery => "real-memory-context-with-multiple-find-queries",
            Self::Reflection => "real-memory-context-over-derived-reflection-index",
            Self::Debate => "real-memory-context-plus-answer-time-critique-accounting",
            Self::NoMemory => "no-context-negative-control",
            Self::EntityOverlap => "real-memory-rrf-keyword-vector-entity-overlap",
            Self::TemporalDecay => "real-memory-rrf-with-recency-decay-lambda-0.01",
            Self::Supersession => "real-memory-rrf-with-entity-linking-and-supersession",
            Self::EpisodeContext => "real-memory-rrf-with-episode-context-window-2",
        }
    }
}

struct BenchState {
    raw_backend: Box<dyn MemoryBackend>,
    reflection_backend: Box<dyn MemoryBackend>,
    // Signal backends — None when --signal-arms is not set (saves backfill cost on
    // scaling tiers). Ability-gated at retrieve time: each arm falls through to
    // raw_backend for tasks that don't carry the matching ability tag.
    entity_backend: Option<Box<dyn MemoryBackend>>,
    temporal_backend: Option<Box<dyn MemoryBackend>>,
    supersession_backend: Option<Box<dyn MemoryBackend>>,
    episode_backend: Option<Box<dyn MemoryBackend>>,
    raw_docs: Vec<CorpusDoc>,
    reflection_docs: Vec<CorpusDoc>,
    raw_index: String,
    reflection_index: String,
    tokenizer: CoreBPE,
    reflection_method_usage: TokenUsage,
}

#[derive(Debug, Clone, Serialize)]
struct RawRow {
    suite_version: String,
    brunnr_version: String,
    embedding_model: String,
    tokenizer: String,
    arm: String,
    strategy: String,
    cache_state: String,
    rep: usize,
    task_id: String,
    difficulty: String,
    ability: Option<String>,
    prompt: String,
    output: String,
    token_usage: TokenUsage,
    success: bool,
    retrieval: RetrievalReport,
}

#[derive(Debug, Clone, Serialize)]
struct TimingRow {
    suite_version: String,
    arm: String,
    rep: usize,
    task_id: String,
    difficulty: String,
    wall_clock_ms: f64,
    memory_find_latency_ms: f64,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
struct TokenUsage {
    input: usize,
    output: usize,
    total: usize,
    answer_input: usize,
    answer_output: usize,
    method_input: usize,
    method_output: usize,
}

impl TokenUsage {
    fn answer(input: usize, output: usize) -> Self {
        Self {
            input,
            output,
            total: input + output,
            answer_input: input,
            answer_output: output,
            method_input: 0,
            method_output: 0,
        }
    }

    fn with_method(mut self, method: TokenUsage) -> Self {
        self.input += method.input;
        self.output += method.output;
        self.total += method.total;
        self.method_input = method.input;
        self.method_output = method.output;
        self
    }

    fn method(input: usize, output: usize) -> Self {
        Self {
            input,
            output,
            total: input + output,
            answer_input: 0,
            answer_output: 0,
            method_input: input,
            method_output: output,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct RetrievalReport {
    relevant_docs: Vec<String>,
    distractor_docs: Vec<String>,
    retrieved_docs: Vec<String>,
    precision: f64,
    recall: f64,
    trace: Vec<TraceHit>,
}

#[derive(Debug, Clone, Serialize)]
struct TraceHit {
    doc_id: String,
    score: f32,
    source: String,
    content_preview: String,
}

#[derive(Debug)]
struct RetrievalOutput {
    hits: Vec<TraceHit>,
    latency_ms: f64,
    index_slice: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct AggregateGroup {
    runs: usize,
    success_rate: f64,
    success_rate_ci95: f64,
    mean_total_tokens: f64,
    total_tokens_variance: f64,
    tokens_per_success: Option<f64>,
    mean_retrieval_precision: f64,
    mean_retrieval_recall: f64,
}

#[derive(Debug, Clone, Serialize)]
struct AggregateArm {
    #[serde(flatten)]
    group: AggregateGroup,
    by_difficulty: BTreeMap<String, AggregateGroup>,
}

#[derive(Debug, Serialize)]
struct AggregateOutput {
    suite_version: String,
    tokenizer: String,
    embedding_model: String,
    backend: String,
    aggregate: BTreeMap<String, AggregateArm>,
    retrieval_misses: Vec<RetrievalMiss>,
    marginal_verdicts: BTreeMap<String, MarginalVerdict>,
    /// Per-signal focused verdict: recall/precision delta on ability-targeted tasks only.
    /// Omitted from JSON when empty (i.e., --signal-arms was not passed).
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    focused_signal_verdicts: BTreeMap<String, FocusedSignalVerdict>,
}

#[derive(Debug, Clone, Serialize)]
struct FocusedSignalVerdict {
    ability: String,
    targeted_task_count: usize,
    recall_delta: f64,
    precision_delta: f64,
    default_decision: String,
}

#[derive(Debug, Clone, Serialize)]
struct RetrievalMiss {
    arm: String,
    task_id: String,
    difficulty: String,
    relevant_docs: Vec<String>,
    retrieved_docs: Vec<String>,
    recall: f64,
    runs: usize,
}

#[derive(Debug, Clone, Serialize)]
struct MarginalVerdict {
    tokens_per_success_delta_vs_b: Option<f64>,
    success_rate_delta_vs_b: f64,
    precision_delta_vs_b: f64,
    recall_delta_vs_b: f64,
    verdict: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    if args.reps == 0 {
        bail!("--reps must be greater than zero");
    }

    let repo_root = std::env::current_dir().context("resolve current directory")?;
    let seed_root = normalize_path(&repo_root, &args.seed_corpus);
    let results_root = normalize_path(&repo_root, &args.results);
    fs::create_dir_all(&results_root)
        .with_context(|| format!("create {}", results_root.display()))?;

    let suite = load_suite(&seed_root)?;
    if suite.suite != SUITE_VERSION {
        bail!(
            "unsupported suite {}; expected {}",
            suite.suite,
            SUITE_VERSION
        );
    }
    let state = prepare_state(&repo_root, &seed_root, &suite, args.signal_arms).await?;

    let arms: &[ArmKind] = if args.signal_arms {
        ArmKind::all()
    } else {
        ArmKind::core()
    };

    let raw_path = results_root.join("raw.jsonl");
    let timing_path = results_root.join("timing.jsonl");
    let mut rows = Vec::new();
    let mut raw = String::new();
    let mut timing = String::new();
    for rep in 0..args.reps {
        for arm in arms {
            for task in &suite.tasks {
                let output = run_row(*arm, rep, task, &state).await?;
                raw.push_str(&serde_json::to_string(&output.row)?);
                raw.push('\n');
                timing.push_str(&serde_json::to_string(&output.timing)?);
                timing.push('\n');
                rows.push(output.row);
            }
        }
    }
    fs::write(&raw_path, raw).with_context(|| format!("write {}", raw_path.display()))?;
    fs::write(&timing_path, timing).with_context(|| format!("write {}", timing_path.display()))?;

    let aggregate = aggregate_rows(&rows);
    let aggregate_path = results_root.join("aggregate.json");
    fs::write(
        &aggregate_path,
        serde_json::to_string_pretty(&aggregate)? + "\n",
    )
    .with_context(|| format!("write {}", aggregate_path.display()))?;
    fs::write(results_root.join("summary.csv"), render_csv(&aggregate))
        .with_context(|| format!("write {}", results_root.join("summary.csv").display()))?;
    fs::write(results_root.join("charts.txt"), render_chart(&aggregate))
        .with_context(|| format!("write {}", results_root.join("charts.txt").display()))?;
    fs::write(
        results_root.join("checksums.txt"),
        render_checksums(&repo_root, &seed_root)?,
    )
    .with_context(|| format!("write {}", results_root.join("checksums.txt").display()))?;

    println!(
        "{}",
        serde_json::json!({
            "raw": raw_path,
            "aggregate": aggregate_path,
            "timing": timing_path,
            "rows": rows.len(),
        })
    );
    Ok(())
}

/// Create a [`VectorMemoryBackend`] with the shared embedder and the given config.
fn make_backend(
    collection: &str,
    config: VectorMemoryConfig,
    embedder: Arc<dyn TextEmbedder>,
) -> Result<VectorMemoryBackend<SqliteVecVectorStore>> {
    let store = SqliteVecVectorStore::in_memory()
        .with_context(|| format!("open {collection} sqlite-vec"))?;
    VectorMemoryBackend::with_embedder(store, config, embedder)
        .with_context(|| format!("create {collection} backend"))
}

async fn backfill_both(backend: &dyn MemoryBackend, import_root: &Path) -> Result<()> {
    backfill_directory(backend, import_root.join("memory"))
        .await
        .context("backfill memory")?;
    backfill_directory(backend, import_root.join("distractors"))
        .await
        .context("backfill distractors")?;
    Ok(())
}

async fn prepare_state(
    repo_root: &Path,
    seed_root: &Path,
    suite: &TaskSuite,
    signal_arms: bool,
) -> Result<BenchState> {
    let raw_docs = load_corpus(seed_root)?;
    let raw_index = render_index("Benchmark Corpus Index", &raw_docs);
    let reflection_docs = reflection_docs(&raw_docs);
    let reflection_index = render_index("Reflection Corpus Index", &reflection_docs);
    let tokenizer = cl100k_base().context("load cl100k_base tokenizer")?;
    let reflection_method_usage =
        reflection_method_usage(&tokenizer, &raw_docs, &reflection_docs, suite.tasks.len());

    let work_root = repo_root.join("target/brunnr-bench");
    if work_root.exists() {
        fs::remove_dir_all(&work_root).with_context(|| format!("clean {}", work_root.display()))?;
    }
    fs::create_dir_all(&work_root).with_context(|| format!("create {}", work_root.display()))?;

    // Shared embedder — loaded once, referenced by all backends.
    let embedder: Arc<dyn TextEmbedder> =
        Arc::new(FastembedTextEmbedder::new().context("initialize fastembed embedder")?);

    let raw_import = work_root.join("raw-import");
    copy_seed_corpus(seed_root, &raw_import)?;

    // Baseline backend (entity=off, all signals off).
    let raw_backend = make_backend(
        "bench_raw",
        VectorMemoryConfig::new("bench_raw"),
        embedder.clone(),
    )?;
    backfill_both(&raw_backend, &raw_import).await?;

    // Signal backends — only built when --signal-arms is set to avoid the backfill
    // overhead on large scaling tiers.
    let (entity_backend, temporal_backend, supersession_backend, episode_backend) = if signal_arms
    {
        let eb = make_backend(
            "bench_entity",
            VectorMemoryConfig::new("bench_entity").with_entity_linking(true),
            embedder.clone(),
        )?;
        backfill_both(&eb, &raw_import).await?;

        let tb = make_backend(
            "bench_temporal",
            VectorMemoryConfig::new("bench_temporal").with_temporal_decay(0.01),
            embedder.clone(),
        )?;
        backfill_both(&tb, &raw_import).await?;

        let sb = make_backend(
            "bench_supersession",
            VectorMemoryConfig::new("bench_supersession")
                .with_entity_linking(true)
                .with_knowledge_update_supersession(true),
            embedder.clone(),
        )?;
        backfill_both(&sb, &raw_import).await?;

        let epb = make_backend(
            "bench_episode",
            VectorMemoryConfig::new("bench_episode")
                .with_entity_linking(true)
                .with_episode_context_window(2),
            embedder.clone(),
        )?;
        backfill_both(&epb, &raw_import).await?;

        (
            Some(Box::new(eb) as Box<dyn MemoryBackend>),
            Some(Box::new(tb) as Box<dyn MemoryBackend>),
            Some(Box::new(sb) as Box<dyn MemoryBackend>),
            Some(Box::new(epb) as Box<dyn MemoryBackend>),
        )
    } else {
        (None, None, None, None)
    };

    // Reflection backend (existing arm).
    let reflection_import = work_root.join("reflection-import");
    write_reflection_corpus(&reflection_docs, &reflection_import)?;
    let reflection_backend = make_backend(
        "bench_reflection",
        VectorMemoryConfig::new("bench_reflection"),
        embedder.clone(),
    )?;
    backfill_directory(&reflection_backend, reflection_import.join("memory"))
        .await
        .context("backfill reflection memory docs")?;
    backfill_directory(&reflection_backend, reflection_import.join("distractors"))
        .await
        .context("backfill reflection distractor docs")?;

    Ok(BenchState {
        raw_backend: Box::new(raw_backend),
        reflection_backend: Box::new(reflection_backend),
        entity_backend,
        temporal_backend,
        supersession_backend,
        episode_backend,
        raw_docs,
        reflection_docs,
        raw_index,
        reflection_index,
        tokenizer,
        reflection_method_usage,
    })
}

struct BenchRun {
    row: RawRow,
    timing: TimingRow,
}

async fn run_row(
    arm: ArmKind,
    rep: usize,
    task: &TaskSpec,
    state: &BenchState,
) -> Result<BenchRun> {
    let started = Instant::now();
    let retrieval = retrieve(arm, task, state).await?;
    let prompt = build_prompt(task, arm, &retrieval);
    let success = task
        .relevant_docs
        .iter()
        .all(|doc| retrieval.hits.iter().any(|hit| &hit.doc_id == doc));
    let output = if success {
        "Retrieval surfaced every relevant document."
    } else {
        "Retrieval missed at least one relevant document."
    }
    .to_string();
    let answer_usage = TokenUsage::answer(
        token_count(&state.tokenizer, &prompt),
        token_count(&state.tokenizer, &output),
    );
    let method_usage = method_usage(arm, task, state);
    let token_usage = answer_usage.with_method(method_usage);
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let retrieval_report = score_retrieval(task, &retrieval.hits);

    let row = RawRow {
        suite_version: SUITE_VERSION.to_string(),
        brunnr_version: BRUNNR_VERSION.to_string(),
        embedding_model: EMBEDDING_MODEL.to_string(),
        tokenizer: format!("{TOKENIZER_ID} ({TOKENIZER_PACKAGE})"),
        arm: arm.id().to_string(),
        strategy: arm.strategy().to_string(),
        cache_state: arm.cache_state().to_string(),
        rep,
        task_id: task.id.clone(),
        difficulty: task.difficulty.clone(),
        ability: task.ability.clone(),
        prompt,
        output,
        token_usage,
        success,
        retrieval: retrieval_report,
    };
    let timing = TimingRow {
        suite_version: SUITE_VERSION.to_string(),
        arm: arm.id().to_string(),
        rep,
        task_id: task.id.clone(),
        difficulty: task.difficulty.clone(),
        wall_clock_ms: round6(elapsed_ms),
        memory_find_latency_ms: round6(retrieval.latency_ms),
    };
    Ok(BenchRun { row, timing })
}

async fn retrieve(arm: ArmKind, task: &TaskSpec, state: &BenchState) -> Result<RetrievalOutput> {
    match arm {
        ArmKind::FullReplay | ArmKind::FullReplayCold => {
            // Use the full body (not preview) so the token count reflects the actual
            // cost of including every document in context. For small docs (< 500 chars)
            // this is identical to preview; for large docs it shows the true scaling cost,
            // proving that full-replay token usage grows with doc size while Brunnr
            // retrieval stays bounded at top_k × chunk_size.
            let hits = state
                .raw_docs
                .iter()
                .map(|doc| TraceHit {
                    doc_id: doc.id.clone(),
                    score: 1.0,
                    source: "full-replay".to_string(),
                    content_preview: doc.body.clone(),
                })
                .collect();
            Ok(RetrievalOutput {
                hits,
                latency_ms: 0.0,
                index_slice: None,
            })
        }
        ArmKind::NoMemory => Ok(RetrievalOutput {
            hits: Vec::new(),
            latency_ms: 0.0,
            index_slice: None,
        }),
        ArmKind::BuiltInAgentMemory => {
            retrieve_find(
                state.raw_backend.as_ref(),
                &state.raw_docs,
                &task.question,
                1,
                1,
                false,
                None,
            )
            .await
        }
        ArmKind::MdOkfIndexFirst => {
            // md/OKF index-first: load the FULL index (one line per doc, grows with the
            // corpus) plus the retrieved whole file(s) — the cost of a markdown memory that
            // lists everything in a MEMORY.md before reading the relevant file.
            let mut output = retrieve_find(
                state.raw_backend.as_ref(),
                &state.raw_docs,
                &task.question,
                DEFAULT_TOP_M,
                DEFAULT_TOP_K,
                true,
                None,
            )
            .await?;
            output.index_slice = Some(state.raw_index.clone());
            Ok(output)
        }
        ArmKind::DefaultBrunnr | ArmKind::DefaultBrunnrCold => {
            retrieve_find(
                state.raw_backend.as_ref(),
                &state.raw_docs,
                &task.question,
                DEFAULT_TOP_M,
                DEFAULT_TOP_K,
                true,
                Some(&state.raw_index),
            )
            .await
        }
        ArmKind::Hyde => {
            let hypothetical = hypothetical_query(task);
            retrieve_find(
                state.raw_backend.as_ref(),
                &state.raw_docs,
                &hypothetical,
                DEFAULT_TOP_M,
                DEFAULT_TOP_K,
                true,
                Some(&state.raw_index),
            )
            .await
        }
        ArmKind::MultiQuery => {
            retrieve_multi_query(
                state.raw_backend.as_ref(),
                &state.raw_docs,
                task,
                Some(&state.raw_index),
            )
            .await
        }
        ArmKind::Reflection => {
            retrieve_find(
                state.reflection_backend.as_ref(),
                &state.reflection_docs,
                &task.question,
                DEFAULT_TOP_M,
                DEFAULT_TOP_K,
                true,
                Some(&state.reflection_index),
            )
            .await
        }
        ArmKind::Debate => {
            retrieve_find(
                state.raw_backend.as_ref(),
                &state.raw_docs,
                &task.question,
                DEFAULT_TOP_M,
                DEFAULT_TOP_K,
                true,
                Some(&state.raw_index),
            )
            .await
        }
        ArmKind::EntityOverlap => {
            // Only use signal backend on entity-disambiguation tasks; fall through to
            // baseline for all others so non-targeted tasks don't dilute the verdict.
            let backend: &dyn MemoryBackend =
                if task.ability.as_deref() == Some("entity-disambiguation") {
                    state
                        .entity_backend
                        .as_deref()
                        .unwrap_or(state.raw_backend.as_ref())
                } else {
                    state.raw_backend.as_ref()
                };
            retrieve_find(
                backend,
                &state.raw_docs,
                &task.question,
                DEFAULT_TOP_M,
                DEFAULT_TOP_K,
                true,
                Some(&state.raw_index),
            )
            .await
        }
        ArmKind::TemporalDecay => {
            let backend: &dyn MemoryBackend =
                if task.ability.as_deref() == Some("temporal-ordering") {
                    state
                        .temporal_backend
                        .as_deref()
                        .unwrap_or(state.raw_backend.as_ref())
                } else {
                    state.raw_backend.as_ref()
                };
            retrieve_find(
                backend,
                &state.raw_docs,
                &task.question,
                DEFAULT_TOP_M,
                DEFAULT_TOP_K,
                true,
                Some(&state.raw_index),
            )
            .await
        }
        ArmKind::Supersession => {
            let backend: &dyn MemoryBackend =
                if task.ability.as_deref() == Some("knowledge-update") {
                    state
                        .supersession_backend
                        .as_deref()
                        .unwrap_or(state.raw_backend.as_ref())
                } else {
                    state.raw_backend.as_ref()
                };
            retrieve_find(
                backend,
                &state.raw_docs,
                &task.question,
                DEFAULT_TOP_M,
                DEFAULT_TOP_K,
                true,
                Some(&state.raw_index),
            )
            .await
        }
        ArmKind::EpisodeContext => {
            let backend: &dyn MemoryBackend =
                if task.ability.as_deref() == Some("multi-session-synthesis") {
                    state
                        .episode_backend
                        .as_deref()
                        .unwrap_or(state.raw_backend.as_ref())
                } else {
                    state.raw_backend.as_ref()
                };
            retrieve_find(
                backend,
                &state.raw_docs,
                &task.question,
                DEFAULT_TOP_M,
                DEFAULT_TOP_K,
                true,
                Some(&state.raw_index),
            )
            .await
        }
    }
}

async fn retrieve_find(
    backend: &dyn MemoryBackend,
    docs: &[CorpusDoc],
    query: &str,
    top_m: usize,
    top_k: usize,
    rerank: bool,
    index: Option<&str>,
) -> Result<RetrievalOutput> {
    let started = Instant::now();
    let mut hits = backend
        .find(MemoryQuery::new(query).with_limit(top_m))
        .await
        .with_context(|| format!("memory.find for {query:?}"))?;
    if rerank {
        hits = LocalLexicalReranker
            .rerank(query, hits, top_k)
            .context("local lexical rerank")?;
    } else {
        hits.truncate(top_k);
    }
    let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
    Ok(RetrievalOutput {
        hits: trace_hits(hits, docs),
        latency_ms,
        index_slice: index.map(index_slice),
    })
}

async fn retrieve_multi_query(
    backend: &dyn MemoryBackend,
    docs: &[CorpusDoc],
    task: &TaskSpec,
    index: Option<&str>,
) -> Result<RetrievalOutput> {
    let started = Instant::now();
    let mut merged = BTreeMap::new();
    for query in multi_queries(task) {
        for hit in backend
            .find(MemoryQuery::new(query.clone()).with_limit(DEFAULT_TOP_M))
            .await
            .with_context(|| format!("memory.find for {query:?}"))?
        {
            let key = hit.record.id.to_string();
            merged
                .entry(key)
                .and_modify(|existing: &mut SearchHit| {
                    if hit.score > existing.score {
                        *existing = hit.clone();
                    }
                })
                .or_insert(hit);
        }
    }
    let hits = LocalLexicalReranker
        .rerank(
            &task.question,
            merged.into_values().collect::<Vec<_>>(),
            DEFAULT_TOP_K,
        )
        .context("local lexical rerank")?;
    let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
    Ok(RetrievalOutput {
        hits: trace_hits(hits, docs),
        latency_ms,
        index_slice: index.map(index_slice),
    })
}

fn trace_hits(hits: Vec<SearchHit>, docs: &[CorpusDoc]) -> Vec<TraceHit> {
    let mut seen = BTreeSet::new();
    let mut output = Vec::new();
    for hit in hits {
        let doc_id = doc_id_for_hit(&hit, docs).unwrap_or_else(|| hit.record.node_id.clone());
        if !seen.insert(doc_id.clone()) {
            continue;
        }
        output.push(TraceHit {
            doc_id,
            score: hit.score,
            source: format!("{:?}", hit.source),
            content_preview: preview(&hit.record.content),
        });
    }
    output
}

fn doc_id_for_hit(hit: &SearchHit, docs: &[CorpusDoc]) -> Option<String> {
    // 1. source_path metadata set by backfill — survives into chunk records at all
    //    nesting levels. Match when the absolute path ends with the doc's relative id.
    if let Some(raw_path) = hit.record.metadata.get("source_path") {
        let normalized = raw_path.replace('\\', "/");
        if let Some(doc) = docs
            .iter()
            .find(|doc| normalized.ends_with(&format!("/{}", doc.id)))
        {
            return Some(doc.id.clone());
        }
    }
    // 2. Full body equality — exact match for small single-chunk docs (unchanged behavior).
    let hit_body = normalize_body(&hit.record.content);
    if let Some(doc) = docs
        .iter()
        .find(|doc| normalize_body(&doc.body) == hit_body)
    {
        return Some(doc.id.clone());
    }
    // 3. Content containment — chunk text is a proper substring of the parent doc body.
    //    Handles large docs that were split into multiple chunks.
    if !hit_body.is_empty() {
        return docs
            .iter()
            .find(|doc| normalize_body(&doc.body).contains(&hit_body))
            .map(|doc| doc.id.clone());
    }
    None
}

fn score_retrieval(task: &TaskSpec, hits: &[TraceHit]) -> RetrievalReport {
    let relevant = task.relevant_docs.iter().cloned().collect::<BTreeSet<_>>();
    let retrieved = hits
        .iter()
        .map(|hit| hit.doc_id.clone())
        .collect::<BTreeSet<_>>();
    let true_positive = relevant.intersection(&retrieved).count();
    let precision = if retrieved.is_empty() {
        0.0
    } else {
        true_positive as f64 / retrieved.len() as f64
    };
    let recall = if relevant.is_empty() {
        1.0
    } else {
        true_positive as f64 / relevant.len() as f64
    };
    RetrievalReport {
        relevant_docs: task.relevant_docs.clone(),
        distractor_docs: task.distractor_docs.clone(),
        retrieved_docs: retrieved.into_iter().collect(),
        precision: round6(precision),
        recall: round6(recall),
        trace: hits.to_vec(),
    }
}

fn method_usage(arm: ArmKind, task: &TaskSpec, state: &BenchState) -> TokenUsage {
    match arm {
        ArmKind::Hyde => {
            let prompt = format!(
                "Generate a hypothetical answer passage for retrieval only.\nQuestion: {}",
                task.question
            );
            let output = hypothetical_query(task);
            TokenUsage::method(
                token_count(&state.tokenizer, &prompt),
                token_count(&state.tokenizer, &output),
            )
        }
        ArmKind::MultiQuery => {
            let prompt = format!(
                "Generate alternate search queries for retrieval only.\nQuestion: {}",
                task.question
            );
            let output = multi_queries(task).join("\n");
            TokenUsage::method(
                token_count(&state.tokenizer, &prompt),
                token_count(&state.tokenizer, &output),
            )
        }
        ArmKind::Reflection => state.reflection_method_usage,
        ArmKind::Debate => {
            let prompt = format!(
                "Proposer and critic evaluate whether the retrieved evidence answers: {}",
                task.question
            );
            let output = "Critique verdict placeholder over actually retrieved evidence.";
            TokenUsage::method(
                token_count(&state.tokenizer, &prompt),
                token_count(&state.tokenizer, output),
            )
        }
        _ => TokenUsage::default(),
    }
}

fn build_prompt(task: &TaskSpec, arm: ArmKind, retrieval: &RetrievalOutput) -> String {
    let mut prompt = format!("Question: {}\n\n", task.question);
    if let Some(index) = &retrieval.index_slice {
        prompt.push_str("Index-first slice:\n");
        prompt.push_str(index);
        prompt.push_str("\n\n");
    }
    prompt.push_str("Retrieved context:\n");
    if retrieval.hits.is_empty() {
        prompt.push_str("(none)\n");
    } else {
        for hit in &retrieval.hits {
            prompt.push_str(&format!(
                "[{} score={:.6} source={}]\n{}\n\n",
                hit.doc_id, hit.score, hit.source, hit.content_preview
            ));
        }
    }
    prompt.push_str(
        "Verifier rule: success is scored after retrieval by comparing retrieved source IDs to hidden labels.\n",
    );
    prompt.push_str(&format!("Arm: {}\n", arm.id()));
    prompt
}

fn aggregate_rows(rows: &[RawRow]) -> AggregateOutput {
    // Build arm list from rows in ArmKind::all() order, including only arms that ran.
    // This is data-driven: works whether signal arms were included or not.
    let seen: BTreeSet<&str> = rows.iter().map(|r| r.arm.as_str()).collect();
    let ordered_ids: Vec<String> = ArmKind::all()
        .iter()
        .map(|a| a.id().to_string())
        .filter(|id| seen.contains(id.as_str()))
        .collect();

    let mut aggregate = BTreeMap::new();
    for arm_id in &ordered_ids {
        let arm_rows = rows
            .iter()
            .filter(|row| &row.arm == arm_id)
            .cloned()
            .collect::<Vec<_>>();
        if arm_rows.is_empty() {
            continue;
        }
        let mut by_difficulty = BTreeMap::new();
        let difficulties = arm_rows
            .iter()
            .map(|row| row.difficulty.clone())
            .collect::<BTreeSet<_>>();
        for difficulty in difficulties {
            let group_rows = arm_rows
                .iter()
                .filter(|row| row.difficulty == difficulty)
                .cloned()
                .collect::<Vec<_>>();
            by_difficulty.insert(difficulty, aggregate_group(&group_rows));
        }
        aggregate.insert(
            arm_id.clone(),
            AggregateArm {
                group: aggregate_group(&arm_rows),
                by_difficulty,
            },
        );
    }
    let mut miss_counts: BTreeMap<(String, String), RetrievalMiss> = BTreeMap::new();
    for row in rows.iter().filter(|row| row.retrieval.recall < 1.0) {
        let key = (row.arm.clone(), row.task_id.clone());
        miss_counts
            .entry(key)
            .and_modify(|miss| miss.runs += 1)
            .or_insert_with(|| RetrievalMiss {
                arm: row.arm.clone(),
                task_id: row.task_id.clone(),
                difficulty: row.difficulty.clone(),
                relevant_docs: row.retrieval.relevant_docs.clone(),
                retrieved_docs: row.retrieval.retrieved_docs.clone(),
                recall: row.retrieval.recall,
                runs: 1,
            });
    }
    let retrieval_misses = miss_counts.into_values().collect::<Vec<_>>();
    let marginal_verdicts = marginal_verdicts(&aggregate);
    let focused_signal_verdicts = focused_signal_verdicts(rows, &aggregate);
    AggregateOutput {
        suite_version: SUITE_VERSION.to_string(),
        tokenizer: format!("{TOKENIZER_ID} ({TOKENIZER_PACKAGE})"),
        embedding_model: EMBEDDING_MODEL.to_string(),
        backend: "SqliteVecVectorStore via VectorMemoryBackend".to_string(),
        aggregate,
        retrieval_misses,
        marginal_verdicts,
        focused_signal_verdicts,
    }
}

fn aggregate_group(rows: &[RawRow]) -> AggregateGroup {
    let successes = rows
        .iter()
        .map(|row| if row.success { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
    let totals = rows
        .iter()
        .map(|row| row.token_usage.total as f64)
        .collect::<Vec<_>>();
    let precision = rows
        .iter()
        .map(|row| row.retrieval.precision)
        .collect::<Vec<_>>();
    let recall = rows
        .iter()
        .map(|row| row.retrieval.recall)
        .collect::<Vec<_>>();
    let success_count = successes.iter().filter(|value| **value > 0.0).count();
    AggregateGroup {
        runs: rows.len(),
        success_rate: mean(&successes),
        success_rate_ci95: ci95(&successes),
        mean_total_tokens: mean(&totals),
        total_tokens_variance: variance(&totals),
        tokens_per_success: (success_count > 0)
            .then(|| round6(totals.iter().sum::<f64>() / success_count as f64)),
        mean_retrieval_precision: mean(&precision),
        mean_retrieval_recall: mean(&recall),
    }
}

fn marginal_verdicts(
    aggregate: &BTreeMap<String, AggregateArm>,
) -> BTreeMap<String, MarginalVerdict> {
    let Some(baseline) = aggregate.get("B-default-brunnr") else {
        return BTreeMap::new();
    };
    [
        "B-plus-hyde",
        "B-plus-multi-query",
        "B-reflection-consolidated",
        "B-plus-debate",
        "B-plus-entity-overlap",
        "B-plus-temporal-decay",
        "B-plus-supersession",
        "B-plus-episode-context",
    ]
    .into_iter()
    .filter_map(|arm| {
        let row = aggregate.get(arm)?;
        let token_delta = match (
            row.group.tokens_per_success,
            baseline.group.tokens_per_success,
        ) {
            (Some(row_tokens), Some(base_tokens)) => Some(round6(row_tokens - base_tokens)),
            _ => None,
        };
        let success_delta = round6(row.group.success_rate - baseline.group.success_rate);
        let precision_delta =
            round6(row.group.mean_retrieval_precision - baseline.group.mean_retrieval_precision);
        let recall_delta =
            round6(row.group.mean_retrieval_recall - baseline.group.mean_retrieval_recall);
        let verdict = if success_delta > 0.0 || recall_delta > 0.0 {
            "recommend for the task classes where the measured recall gain appears"
        } else if token_delta.is_some_and(|delta| delta < 0.0)
            && success_delta >= 0.0
            && recall_delta >= 0.0
        {
            "recommend only when the measured token reduction repeats on the target corpus"
        } else {
            "skip by default; no measured quality gain over B-default in this sample"
        };
        Some((
            arm.to_string(),
            MarginalVerdict {
                tokens_per_success_delta_vs_b: token_delta,
                success_rate_delta_vs_b: success_delta,
                precision_delta_vs_b: precision_delta,
                recall_delta_vs_b: recall_delta,
                verdict: verdict.to_string(),
            },
        ))
    })
    .collect()
}

fn render_csv(aggregate: &AggregateOutput) -> String {
    let mut lines = vec![
        "arm,runs,success_rate,mean_total_tokens,tokens_per_success,precision,recall".to_string(),
    ];
    for (arm, row) in &aggregate.aggregate {
        lines.push(format!(
            "{},{},{},{},{},{},{}",
            arm,
            row.group.runs,
            row.group.success_rate,
            row.group.mean_total_tokens,
            row.group
                .tokens_per_success
                .map_or_else(String::new, |value| value.to_string()),
            row.group.mean_retrieval_precision,
            row.group.mean_retrieval_recall
        ));
    }
    lines.join("\n") + "\n"
}

/// Focused verdict for each signal arm: compare recall/precision on ability-targeted
/// tasks only (not diluted by the non-targeted baseline fall-through rows).
fn focused_signal_verdicts(
    rows: &[RawRow],
    aggregate: &BTreeMap<String, AggregateArm>,
) -> BTreeMap<String, FocusedSignalVerdict> {
    let signal_abilities: &[(&str, &str, &str)] = &[
        ("B-plus-entity-overlap", "entity-disambiguation", "B-default-brunnr"),
        ("B-plus-temporal-decay", "temporal-ordering", "B-default-brunnr"),
        ("B-plus-supersession", "knowledge-update", "B-default-brunnr"),
        ("B-plus-episode-context", "multi-session-synthesis", "B-default-brunnr"),
    ];
    let mut out = BTreeMap::new();
    for (arm_id, ability, baseline_id) in signal_abilities {
        // Only report if the arm actually ran.
        if !aggregate.contains_key(*arm_id) {
            continue;
        }
        let targeted: Vec<&RawRow> = rows
            .iter()
            .filter(|r| r.ability.as_deref() == Some(ability) && r.arm == *arm_id)
            .collect();
        let baseline: Vec<&RawRow> = rows
            .iter()
            .filter(|r| r.ability.as_deref() == Some(ability) && r.arm == *baseline_id)
            .collect();
        if targeted.is_empty() || baseline.is_empty() {
            out.insert(
                arm_id.to_string(),
                FocusedSignalVerdict {
                    ability: ability.to_string(),
                    targeted_task_count: 0,
                    recall_delta: 0.0,
                    precision_delta: 0.0,
                    default_decision: "no ability-tagged tasks in this corpus; keeping OFF"
                        .to_string(),
                },
            );
            continue;
        }
        let recall_signal = mean(
            &targeted
                .iter()
                .map(|r| r.retrieval.recall)
                .collect::<Vec<_>>(),
        );
        let recall_base = mean(
            &baseline
                .iter()
                .map(|r| r.retrieval.recall)
                .collect::<Vec<_>>(),
        );
        let prec_signal = mean(
            &targeted
                .iter()
                .map(|r| r.retrieval.precision)
                .collect::<Vec<_>>(),
        );
        let prec_base = mean(
            &baseline
                .iter()
                .map(|r| r.retrieval.precision)
                .collect::<Vec<_>>(),
        );
        let recall_delta = round6(recall_signal - recall_base);
        let precision_delta = round6(prec_signal - prec_base);
        // Turn default ON only if recall improves by ≥5pp without hurting precision.
        let default_decision = if recall_delta >= 0.05 && precision_delta >= -0.02 {
            format!("ENABLE DEFAULT: +{recall_delta:.2} recall on {ability} tasks, precision={precision_delta:+.2}")
        } else if recall_delta > 0.0 {
            format!("keep OFF (marginal gain {recall_delta:+.2} recall; run on larger corpus before enabling)")
        } else {
            format!("keep OFF (delta recall={recall_delta:+.2} precision={precision_delta:+.2} on {ability} tasks)")
        };
        out.insert(
            arm_id.to_string(),
            FocusedSignalVerdict {
                ability: ability.to_string(),
                targeted_task_count: targeted.len(),
                recall_delta,
                precision_delta,
                default_decision,
            },
        );
    }
    out
}

fn render_chart(aggregate: &AggregateOutput) -> String {
    let mut lines = vec!["Tokens per success (lower is better; tokenizer=cl100k_base)".to_string()];
    for (arm, row) in &aggregate.aggregate {
        let value = row.group.tokens_per_success.unwrap_or(0.0);
        let bar = "#".repeat(((value / 25.0).round() as usize).max(1));
        lines.push(format!("{arm:36} {value:10.2} {bar}"));
    }
    lines.push("\nRetrieval misses (recall < 1.0)".to_string());
    for miss in &aggregate.retrieval_misses {
        lines.push(format!(
            "{} {} recall={} relevant={:?} retrieved={:?}",
            miss.arm, miss.task_id, miss.recall, miss.relevant_docs, miss.retrieved_docs
        ));
    }
    lines.push("\nOpt-in marginal verdicts".to_string());
    for (arm, verdict) in &aggregate.marginal_verdicts {
        lines.push(format!(
            "{arm}: delta_tokens={:?} delta_success={} delta_precision={} delta_recall={} {}",
            verdict.tokens_per_success_delta_vs_b,
            verdict.success_rate_delta_vs_b,
            verdict.precision_delta_vs_b,
            verdict.recall_delta_vs_b,
            verdict.verdict
        ));
    }
    if !aggregate.focused_signal_verdicts.is_empty() {
        lines.push("\nSignal arm focused evaluation (ability-tagged tasks only)".to_string());
        for (arm, verdict) in &aggregate.focused_signal_verdicts {
            lines.push(format!(
                "{arm} [{ability}] n={n} delta_recall={dr:+.3} delta_precision={dp:+.3} → {dec}",
                ability = verdict.ability,
                n = verdict.targeted_task_count,
                dr = verdict.recall_delta,
                dp = verdict.precision_delta,
                dec = verdict.default_decision,
            ));
        }
    }
    lines.join("\n") + "\n"
}

fn load_suite(seed_root: &Path) -> Result<TaskSuite> {
    let tasks_path = seed_root.join("tasks.json");
    let text = fs::read_to_string(&tasks_path)
        .with_context(|| format!("read {}", tasks_path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", tasks_path.display()))
}

fn load_corpus(seed_root: &Path) -> Result<Vec<CorpusDoc>> {
    let mut docs = Vec::new();
    for subdir in ["memory", "distractors"] {
        let root = seed_root.join(subdir);
        let mut paths = fs::read_dir(&root)
            .with_context(|| format!("read {}", root.display()))?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<std::io::Result<Vec<_>>>()
            .with_context(|| format!("read {}", root.display()))?;
        paths.sort();
        for path in paths {
            if path.extension().and_then(|value| value.to_str()) != Some("md") {
                continue;
            }
            let relative = path
                .strip_prefix(seed_root)
                .with_context(|| format!("relativize {}", path.display()))?;
            let id = relative.to_string_lossy().replace('\\', "/");
            let full_text =
                fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
            let body = markdown_body(&full_text).to_string();
            let title = markdown_title(&body).unwrap_or_else(|| id.clone());
            docs.push(CorpusDoc {
                id,
                title,
                full_text,
                body,
            });
        }
    }
    Ok(docs)
}

fn copy_seed_corpus(seed_root: &Path, destination: &Path) -> Result<()> {
    for doc in load_corpus(seed_root)? {
        let target = destination.join(&doc.id);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        }
        fs::write(&target, doc.full_text).with_context(|| format!("write {}", target.display()))?;
    }
    Ok(())
}

fn reflection_docs(raw_docs: &[CorpusDoc]) -> Vec<CorpusDoc> {
    raw_docs
        .iter()
        .map(|doc| {
            let summary = structural_summary(doc);
            CorpusDoc {
                id: doc.id.clone(),
                title: format!("Reflection summary: {}", doc.title),
                full_text: render_okf_doc("reference", &doc.title, &summary),
                body: summary,
            }
        })
        .collect()
}

fn write_reflection_corpus(docs: &[CorpusDoc], destination: &Path) -> Result<()> {
    for doc in docs {
        let target = destination.join(&doc.id);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        }
        fs::write(&target, &doc.full_text)
            .with_context(|| format!("write {}", target.display()))?;
    }
    Ok(())
}

fn structural_summary(doc: &CorpusDoc) -> String {
    let first_sentences = doc
        .body
        .lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join(" ")
        .split('.')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .take(2)
        .collect::<Vec<_>>()
        .join(". ");
    format!("# {}\n\n{}.", doc.title, first_sentences)
}

fn render_okf_doc(kind: &str, title: &str, body: &str) -> String {
    format!(
        "---\ntype: {kind}\ntags: [benchmark, structural-summary]\ntitle: \"{}\"\nlicense: Apache-2.0\n---\n\n{body}\n",
        title.replace('"', "\\\"")
    )
}

fn render_index(title: &str, docs: &[CorpusDoc]) -> String {
    let mut output = format!("# {title}\n\n");
    for doc in docs {
        output.push_str(&format!("- `{}` — {}\n", doc.id, doc.title));
    }
    output
}

fn index_slice(index: &str) -> String {
    index.chars().take(2_500).collect()
}

fn hypothetical_query(task: &TaskSpec) -> String {
    format!(
        "A project memory document that directly answers: {}",
        task.question
    )
}

fn multi_queries(task: &TaskSpec) -> Vec<String> {
    let normalized = task
        .question
        .replace("What is", "")
        .replace("What was", "")
        .replace("Which", "")
        .replace("How long does", "")
        .replace("?", "")
        .trim()
        .to_string();
    vec![
        task.question.clone(),
        normalized,
        format!("{} decision record", task.difficulty),
    ]
}

fn reflection_method_usage(
    tokenizer: &CoreBPE,
    raw_docs: &[CorpusDoc],
    reflection_docs: &[CorpusDoc],
    task_count: usize,
) -> TokenUsage {
    let input = raw_docs
        .iter()
        .map(|doc| token_count(tokenizer, &doc.body))
        .sum::<usize>()
        / task_count.max(1);
    let output = reflection_docs
        .iter()
        .map(|doc| token_count(tokenizer, &doc.body))
        .sum::<usize>()
        / task_count.max(1);
    TokenUsage::method(input, output)
}

fn token_count(tokenizer: &CoreBPE, text: &str) -> usize {
    tokenizer.encode_with_special_tokens(text).len()
}

fn markdown_body(text: &str) -> &str {
    if let Some(rest) = text.strip_prefix("---\n") {
        if let Some((_, body)) = rest.split_once("\n---\n") {
            return body.trim();
        }
    }
    if let Some(rest) = text.strip_prefix("+++\n") {
        if let Some((_, body)) = rest.split_once("\n+++\n") {
            return body.trim();
        }
    }
    text.trim()
}

fn markdown_title(body: &str) -> Option<String> {
    body.lines()
        .find_map(|line| line.strip_prefix("# ").map(str::trim))
        .map(ToString::to_string)
}

fn normalize_body(body: &str) -> String {
    body.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn preview(content: &str) -> String {
    let mut preview = content.chars().take(500).collect::<String>();
    if content.chars().count() > 500 {
        preview.push_str("...");
    }
    preview
}

fn normalize_path(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

fn render_checksums(repo_root: &Path, seed_root: &Path) -> Result<String> {
    let mut paths = Vec::new();
    collect_files(seed_root, &mut paths)?;
    paths.sort();
    let mut lines = Vec::new();
    for path in paths {
        let digest =
            Sha256::digest(fs::read(&path).with_context(|| format!("read {}", path.display()))?);
        let relative = path
            .strip_prefix(repo_root)
            .with_context(|| format!("relativize {}", path.display()))?;
        lines.push(format!(
            "{:x}  {}",
            digest,
            relative.to_string_lossy().replace('\\', "/")
        ));
    }
    Ok(lines.join("\n") + "\n")
}

fn collect_files(root: &Path, output: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(root).with_context(|| format!("read {}", root.display()))? {
        let path = entry?.path();
        if path.is_dir() {
            collect_files(&path, output)?;
        } else if path.extension().and_then(|value| value.to_str()) == Some("md")
            || path.file_name().and_then(|value| value.to_str()) == Some("tasks.json")
        {
            output.push(path);
        }
    }
    Ok(())
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        round6(values.iter().sum::<f64>() / values.len() as f64)
    }
}

fn variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    round6(
        values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (values.len() - 1) as f64,
    )
}

fn ci95(values: &[f64]) -> f64 {
    if values.len() < 2 {
        0.0
    } else {
        round6(1.96 * (variance(values) / values.len() as f64).sqrt())
    }
}

fn round6(value: f64) -> f64 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoring_records_real_retrieval_miss() {
        let task = TaskSpec {
            id: "hard-task".to_string(),
            difficulty: "hard".to_string(),
            ability: None,
            question: "What is the hidden answer?".to_string(),
            relevant_docs: vec!["memory/right.md".to_string()],
            distractor_docs: vec!["distractors/near-miss.md".to_string()],
        };
        let hits = vec![TraceHit {
            doc_id: "distractors/near-miss.md".to_string(),
            score: 1.0,
            source: "Hybrid".to_string(),
            content_preview: "near miss".to_string(),
        }];

        let report = score_retrieval(&task, &hits);

        assert_eq!(report.precision, 0.0);
        assert_eq!(report.recall, 0.0);
        assert_eq!(report.retrieved_docs, vec!["distractors/near-miss.md"]);
    }

    #[test]
    fn aggregation_keeps_hard_task_miss_evidence() {
        let row = RawRow {
            suite_version: SUITE_VERSION.to_string(),
            brunnr_version: BRUNNR_VERSION.to_string(),
            embedding_model: EMBEDDING_MODEL.to_string(),
            tokenizer: TOKENIZER_ID.to_string(),
            arm: "C-built-in-agent-memory".to_string(),
            strategy: "real-memory-find-top1-no-index".to_string(),
            cache_state: "warm".to_string(),
            rep: 0,
            task_id: "hard-task".to_string(),
            difficulty: "hard".to_string(),
            ability: None,
            prompt: "Question only".to_string(),
            output: "Retrieval missed at least one relevant document.".to_string(),
            token_usage: TokenUsage::answer(10, 4),
            success: false,
            retrieval: RetrievalReport {
                relevant_docs: vec!["memory/right.md".to_string()],
                distractor_docs: vec!["distractors/near-miss.md".to_string()],
                retrieved_docs: vec!["distractors/near-miss.md".to_string()],
                precision: 0.0,
                recall: 0.0,
                trace: Vec::new(),
            },
        };

        let aggregate = aggregate_rows(&[row]);

        assert_eq!(aggregate.retrieval_misses.len(), 1);
        assert_eq!(aggregate.retrieval_misses[0].difficulty, "hard");
        assert_eq!(aggregate.retrieval_misses[0].runs, 1);
    }
}
