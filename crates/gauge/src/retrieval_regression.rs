// SPDX-License-Identifier: Apache-2.0

//! Deterministic retrieval regression lane for CI.
//!
//! This module keeps the release gate local and reproducible: it runs a fixed
//! LoCoMo/LongMemEval-shaped question set through the files and sqlite-vec
//! backends, scores retrieval with Recall@K plus extractive answer metrics, and
//! validates the project/shared partition invariant on the files backend.

use std::{
    collections::BTreeSet,
    fmt,
    path::{Path, PathBuf},
    sync::Arc,
};

use aquifer::{
    FilesBackend, MemoryBackend, MemoryQuery, MemoryResult, MemoryTier, SearchHit,
    SqliteVecVectorStore, StoreMemory, TextEmbedder, VectorMemoryBackend, VectorMemoryConfig,
    PINNED_FASTEMBED_DIMENSIONS,
};
use serde::{Deserialize, Serialize};

pub const DEFAULT_K: usize = 3;
pub const DEFAULT_TOLERANCE: f32 = 0.02;
const SCHEMA_VERSION: u32 = 1;
const SPDX_LICENSE: &str = "Apache-2.0";

#[derive(Debug)]
pub enum RegressionError {
    Backend(String),
    Io(std::io::Error),
    Json(serde_json::Error),
    Gate(String),
}

impl fmt::Display for RegressionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backend(error) => write!(formatter, "backend error: {error}"),
            Self::Io(error) => write!(formatter, "io error: {error}"),
            Self::Json(error) => write!(formatter, "json error: {error}"),
            Self::Gate(error) => write!(formatter, "gate failed: {error}"),
        }
    }
}

impl std::error::Error for RegressionError {}

impl From<std::io::Error> for RegressionError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for RegressionError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

type RegressionResult<T> = Result<T, RegressionError>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegressionReport {
    #[serde(rename = "_spdx_license")]
    pub spdx_license: String,
    pub schema_version: u32,
    pub k: usize,
    pub tolerance: f32,
    pub cases: usize,
    pub backends: Vec<BackendMetrics>,
    pub leak_gate: LeakGateReport,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendMetrics {
    pub backend: String,
    pub cases: usize,
    pub recall_at_k: f32,
    pub rouge_l_f1: f32,
    pub task_success: f32,
    pub case_metrics: Vec<CaseMetrics>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseMetrics {
    pub id: String,
    pub dataset: String,
    pub category: String,
    pub gold_fact_ids: Vec<String>,
    pub retrieved_fact_ids: Vec<String>,
    pub recall_at_k: f32,
    pub rouge_l_f1: f32,
    pub task_success: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeakGateReport {
    pub backend: String,
    pub project: String,
    pub project_hits: Vec<String>,
    pub default_hits: Vec<String>,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BaselineComparison {
    pub passed: bool,
    pub tolerance: f32,
    pub failures: Vec<String>,
}

#[derive(Debug, Clone)]
struct RetrievalFact {
    id: &'static str,
    content: &'static str,
}

#[derive(Debug, Clone)]
struct RetrievalCase {
    id: &'static str,
    dataset: &'static str,
    category: &'static str,
    question: &'static str,
    gold_answer: &'static str,
    gold_fact_ids: &'static [&'static str],
    facts: Vec<RetrievalFact>,
}

pub async fn run_regression_suite(k: usize, tolerance: f32) -> RegressionResult<RegressionReport> {
    let cases = fixed_cases();
    let backends = vec![
        run_files_backend(&cases, k).await?,
        run_sqlite_vec_backend(&cases, k).await?,
    ];
    let leak_gate = run_files_leak_gate().await?;
    if !leak_gate.passed {
        return Err(RegressionError::Gate(format!(
            "files partition leak gate failed: project_hits={:?} default_hits={:?}",
            leak_gate.project_hits, leak_gate.default_hits
        )));
    }
    Ok(RegressionReport {
        spdx_license: SPDX_LICENSE.to_string(),
        schema_version: SCHEMA_VERSION,
        k,
        tolerance,
        cases: cases.len(),
        backends,
        leak_gate,
    })
}

pub fn load_report(path: impl AsRef<Path>) -> RegressionResult<RegressionReport> {
    let raw = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&raw)?)
}

pub fn write_report(path: impl AsRef<Path>, report: &RegressionReport) -> RegressionResult<()> {
    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent)?;
    }
    let raw = serde_json::to_string_pretty(report)?;
    std::fs::write(path, format!("{raw}\n"))?;
    Ok(())
}

pub fn compare_to_baseline(
    current: &RegressionReport,
    baseline: &RegressionReport,
) -> BaselineComparison {
    let tolerance = baseline.tolerance;
    let mut failures = Vec::new();
    if current.schema_version != baseline.schema_version {
        failures.push(format!(
            "schema_version changed: current={} baseline={}",
            current.schema_version, baseline.schema_version
        ));
    }
    if current.k != baseline.k {
        failures.push(format!(
            "k changed: current={} baseline={}",
            current.k, baseline.k
        ));
    }
    if current.cases != baseline.cases {
        failures.push(format!(
            "case count changed: current={} baseline={}",
            current.cases, baseline.cases
        ));
    }
    for baseline_backend in &baseline.backends {
        match current
            .backends
            .iter()
            .find(|backend| backend.backend == baseline_backend.backend)
        {
            Some(current_backend) => {
                if current_backend.cases != baseline_backend.cases {
                    failures.push(format!(
                        "{} case count changed: current={} baseline={}",
                        current_backend.backend, current_backend.cases, baseline_backend.cases
                    ));
                }
                if current_backend.recall_at_k + tolerance < baseline_backend.recall_at_k {
                    failures.push(format!(
                        "{} recall@{} regressed: current={:.3} baseline={:.3} tolerance={:.3}",
                        current_backend.backend,
                        baseline.k,
                        current_backend.recall_at_k,
                        baseline_backend.recall_at_k,
                        tolerance
                    ));
                }
            }
            None => failures.push(format!(
                "missing backend in current report: {}",
                baseline_backend.backend
            )),
        }
    }
    BaselineComparison {
        passed: failures.is_empty(),
        tolerance,
        failures,
    }
}

pub fn render_regression_markdown(
    report: &RegressionReport,
    comparison: Option<&BaselineComparison>,
) -> String {
    let mut output = String::new();
    output.push_str("# Gauge Retrieval Regression\n\n");
    output.push_str(&format!(
        "Fixed cases: {} | Recall@K: @{} | tolerance: {:.3}\n\n",
        report.cases, report.k, report.tolerance
    ));
    output.push_str(
        "| backend | cases | recall@k | rouge-l f1 | task-success |\n|---|---:|---:|---:|---:|\n",
    );
    for backend in &report.backends {
        output.push_str(&format!(
            "| {} | {} | {:.3} | {:.3} | {:.3} |\n",
            backend.backend,
            backend.cases,
            backend.recall_at_k,
            backend.rouge_l_f1,
            backend.task_success
        ));
    }
    output.push('\n');
    output.push_str(&format!(
        "Leak gate: {} (project hits: {:?}; no-project hits: {:?})\n",
        if report.leak_gate.passed {
            "passed"
        } else {
            "failed"
        },
        report.leak_gate.project_hits,
        report.leak_gate.default_hits
    ));
    if let Some(comparison) = comparison {
        output.push_str(&format!(
            "\nBaseline gate: {} (tolerance {:.3})\n",
            if comparison.passed {
                "passed"
            } else {
                "failed"
            },
            comparison.tolerance
        ));
        for failure in &comparison.failures {
            output.push_str(&format!("- {failure}\n"));
        }
    }
    output
}

async fn run_files_backend(cases: &[RetrievalCase], k: usize) -> RegressionResult<BackendMetrics> {
    let mut case_metrics = Vec::with_capacity(cases.len());
    for case in cases {
        let temp = TempEvalDir::new("gauge-regression-files")?;
        let backend = FilesBackend::new(temp.path()).with_track_access(false);
        case_metrics.push(evaluate_case(&backend, case, k).await?);
    }
    Ok(aggregate_backend("files", case_metrics))
}

async fn run_sqlite_vec_backend(
    cases: &[RetrievalCase],
    k: usize,
) -> RegressionResult<BackendMetrics> {
    let embedder: Arc<dyn TextEmbedder> = Arc::new(HashingEmbedder);
    let mut case_metrics = Vec::with_capacity(cases.len());
    for (index, case) in cases.iter().enumerate() {
        let store = SqliteVecVectorStore::in_memory().map_err(memory_error)?;
        let config =
            VectorMemoryConfig::new(format!("gauge_ci_eval_{index}")).with_track_access(false);
        let backend = VectorMemoryBackend::with_embedder(store, config, embedder.clone())
            .map_err(memory_error)?;
        case_metrics.push(evaluate_case(&backend, case, k).await?);
    }
    Ok(aggregate_backend("sqlite-vec", case_metrics))
}

async fn evaluate_case<B: MemoryBackend>(
    backend: &B,
    case: &RetrievalCase,
    k: usize,
) -> RegressionResult<CaseMetrics> {
    for fact in &case.facts {
        backend
            .store(memory_from_fact(fact))
            .await
            .map_err(memory_error)?;
    }
    let hits = backend
        .find(MemoryQuery::new(case.question).with_limit(k))
        .await
        .map_err(memory_error)?;
    Ok(score_case(case, &hits))
}

fn score_case(case: &RetrievalCase, hits: &[SearchHit]) -> CaseMetrics {
    let gold_ids: BTreeSet<&str> = case.gold_fact_ids.iter().copied().collect();
    let retrieved_fact_ids: Vec<String> =
        hits.iter().map(|hit| hit.record.node_id.clone()).collect();
    let retrieved: BTreeSet<&str> = retrieved_fact_ids.iter().map(String::as_str).collect();
    let recalled = gold_ids
        .iter()
        .filter(|gold_id| retrieved.contains(**gold_id))
        .count();
    let recall_at_k = ratio(recalled, gold_ids.len());
    let answer = extractive_answer(case.gold_answer, hits);
    let rouge_l_f1 = rouge_l_f1(&answer, case.gold_answer);
    let task_success = recall_at_k >= 1.0 && answer_matches_gold(&answer, case.gold_answer);
    CaseMetrics {
        id: case.id.to_string(),
        dataset: case.dataset.to_string(),
        category: case.category.to_string(),
        gold_fact_ids: case
            .gold_fact_ids
            .iter()
            .map(|id| (*id).to_string())
            .collect(),
        retrieved_fact_ids,
        recall_at_k,
        rouge_l_f1,
        task_success,
    }
}

async fn run_files_leak_gate() -> RegressionResult<LeakGateReport> {
    let temp = TempEvalDir::new("gauge-partition-leak")?;
    let backend = FilesBackend::new(temp.path()).with_track_access(false);
    for (node_id, project, content) in [
        (
            "node:partition-a",
            Some("A"),
            "partition sentinel belongs to project A private memory",
        ),
        (
            "node:partition-shared",
            Some("shared"),
            "partition sentinel belongs to shared memory",
        ),
        (
            "node:partition-b",
            Some("B"),
            "partition sentinel belongs to project B private memory",
        ),
    ] {
        let mut memory = StoreMemory::atom(content);
        memory.node_id = Some(node_id.to_string());
        memory.project = project.map(str::to_string);
        backend.store(memory).await.map_err(memory_error)?;
    }

    let project_hits = backend
        .find(
            MemoryQuery::new("partition sentinel")
                .with_limit(10)
                .with_project("A"),
        )
        .await
        .map_err(memory_error)?;
    let default_hits = backend
        .find(MemoryQuery::new("partition sentinel").with_limit(10))
        .await
        .map_err(memory_error)?;
    validate_partition_hits(&project_hits, &default_hits)
}

fn validate_partition_hits(
    project_hits: &[SearchHit],
    default_hits: &[SearchHit],
) -> RegressionResult<LeakGateReport> {
    let project_nodes = hit_nodes(project_hits);
    let default_nodes = hit_nodes(default_hits);
    let project_leaks_b = project_nodes.iter().any(|node| node == "node:partition-b");
    let project_has_a = project_nodes.iter().any(|node| node == "node:partition-a");
    let project_has_shared = project_nodes
        .iter()
        .any(|node| node == "node:partition-shared");
    let default_has_private = default_nodes
        .iter()
        .any(|node| node == "node:partition-a" || node == "node:partition-b");
    let default_bounded = default_nodes.len() < 3;
    let passed = !project_leaks_b
        && project_has_a
        && project_has_shared
        && !default_has_private
        && default_bounded;

    let report = LeakGateReport {
        backend: "files".to_string(),
        project: "A".to_string(),
        project_hits: project_nodes,
        default_hits: default_nodes,
        passed,
    };
    if passed {
        Ok(report)
    } else {
        Err(RegressionError::Gate(format!(
            "project partition invariant failed: {report:?}"
        )))
    }
}

fn aggregate_backend(backend: &str, case_metrics: Vec<CaseMetrics>) -> BackendMetrics {
    let cases = case_metrics.len();
    let recall_total = case_metrics
        .iter()
        .map(|case| case.recall_at_k)
        .sum::<f32>();
    let rouge_total = case_metrics.iter().map(|case| case.rouge_l_f1).sum::<f32>();
    let success_count = case_metrics.iter().filter(|case| case.task_success).count();
    BackendMetrics {
        backend: backend.to_string(),
        cases,
        recall_at_k: mean(recall_total, cases),
        rouge_l_f1: mean(rouge_total, cases),
        task_success: ratio(success_count, cases),
        case_metrics,
    }
}

fn memory_from_fact(fact: &RetrievalFact) -> StoreMemory {
    let mut memory = StoreMemory::atom(fact.content);
    memory.tier = MemoryTier::L1Atom;
    memory.node_id = Some(fact.id.to_string());
    memory
}

fn extractive_answer(gold_answer: &str, hits: &[SearchHit]) -> String {
    let gold = normalize_text(gold_answer);
    hits.iter()
        .find(|hit| normalize_text(&hit.record.content).contains(&gold))
        .map(|_| gold_answer.to_string())
        .or_else(|| hits.first().map(|hit| hit.record.content.clone()))
        .unwrap_or_else(|| "unknown".to_string())
}

fn answer_matches_gold(answer: &str, gold_answer: &str) -> bool {
    let answer = normalize_text(answer);
    let gold = normalize_text(gold_answer);
    !gold.is_empty() && answer.contains(&gold)
}

fn rouge_l_f1(predicted: &str, gold: &str) -> f32 {
    let predicted = normalized_tokens(predicted);
    let gold = normalized_tokens(gold);
    if predicted.is_empty() || gold.is_empty() {
        return 0.0;
    }
    let lcs = lcs_len(&predicted, &gold);
    if lcs == 0 {
        return 0.0;
    }
    let precision = lcs as f32 / predicted.len() as f32;
    let recall = lcs as f32 / gold.len() as f32;
    2.0 * precision * recall / (precision + recall)
}

fn lcs_len(left: &[String], right: &[String]) -> usize {
    let mut previous = vec![0usize; right.len() + 1];
    let mut current = vec![0usize; right.len() + 1];
    for left_token in left {
        for (index, right_token) in right.iter().enumerate() {
            current[index + 1] = if left_token == right_token {
                previous[index] + 1
            } else {
                previous[index + 1].max(current[index])
            };
        }
        std::mem::swap(&mut previous, &mut current);
        current.fill(0);
    }
    previous[right.len()]
}

fn normalize_text(input: &str) -> String {
    normalized_tokens(input).join(" ")
}

fn normalized_tokens(input: &str) -> Vec<String> {
    input
        .to_ascii_lowercase()
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_string)
        .collect()
}

fn mean(total: f32, count: usize) -> f32 {
    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

fn ratio(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f32 / denominator as f32
    }
}

fn hit_nodes(hits: &[SearchHit]) -> Vec<String> {
    hits.iter().map(|hit| hit.record.node_id.clone()).collect()
}

fn memory_error(error: impl fmt::Display) -> RegressionError {
    RegressionError::Backend(error.to_string())
}

#[derive(Debug)]
struct HashingEmbedder;

impl TextEmbedder for HashingEmbedder {
    fn embed_query(&self, text: &str) -> MemoryResult<Vec<f32>> {
        Ok(hash_embedding(text))
    }

    fn embed_passage(&self, text: &str) -> MemoryResult<Vec<f32>> {
        Ok(hash_embedding(text))
    }
}

fn hash_embedding(text: &str) -> Vec<f32> {
    let mut vector = vec![0.0f32; PINNED_FASTEMBED_DIMENSIONS];
    for token in normalized_tokens(text) {
        let index = fnv1a(&token) as usize % vector.len();
        vector[index] += 1.0;
    }
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    } else {
        vector[0] = 1.0;
    }
    vector
}

fn fnv1a(input: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in input.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[derive(Debug)]
struct TempEvalDir {
    path: PathBuf,
}

impl TempEvalDir {
    fn new(prefix: &str) -> std::io::Result<Self> {
        let path = std::env::temp_dir().join(format!(
            "{prefix}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&path)?;
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempEvalDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn fixed_cases() -> Vec<RetrievalCase> {
    vec![
        RetrievalCase {
            id: "locomo-single-hop-language",
            dataset: "locomo",
            category: "single-hop",
            question: "what language is used for core crates by Mira",
            gold_answer: "Rust",
            gold_fact_ids: &["locomo-language-gold"],
            facts: vec![
                RetrievalFact {
                    id: "locomo-language-gold",
                    content: "Mira said the Rust language is used for the core crates",
                },
                RetrievalFact {
                    id: "locomo-language-distractor-1",
                    content: "Mira said the tea shelf is stocked for late reviews",
                },
                RetrievalFact {
                    id: "locomo-language-distractor-2",
                    content: "Old core office crates were recycled after the move",
                },
            ],
        },
        RetrievalCase {
            id: "locomo-temporal-update-ood",
            dataset: "locomo",
            category: "temporal-update-ood",
            question: "where is the updated invoice destination for Noor",
            gold_answer: "finance vault",
            gold_fact_ids: &["locomo-invoice-gold"],
            facts: vec![
                RetrievalFact {
                    id: "locomo-invoice-old",
                    content: "On Monday, Noor sent invoices to the old billing mailbox",
                },
                RetrievalFact {
                    id: "locomo-invoice-gold",
                    content: "On Friday, Noor updated the invoice destination to the finance vault",
                },
                RetrievalFact {
                    id: "locomo-invoice-distractor",
                    content: "The travel vault code was rotated by another team",
                },
            ],
        },
        RetrievalCase {
            id: "locomo-preference-ood",
            dataset: "locomo",
            category: "preference-ood",
            question: "which breakfast pastry should be ordered for Kai",
            gold_answer: "almond croissant",
            gold_fact_ids: &["locomo-pastry-gold"],
            facts: vec![
                RetrievalFact {
                    id: "locomo-pastry-gold",
                    content: "Kai prefers almond croissant as the breakfast pastry for orders",
                },
                RetrievalFact {
                    id: "locomo-pastry-distractor-1",
                    content: "Kai ordered a blueberry muffin during last winter travel",
                },
                RetrievalFact {
                    id: "locomo-pastry-distractor-2",
                    content: "Breakfast service starts after the release standup",
                },
            ],
        },
        RetrievalCase {
            id: "longmemeval-multi-hop",
            dataset: "longmemeval",
            category: "multi-session",
            question: "which cluster is used for the Tuesday migration rehearsal",
            gold_answer: "staging cluster",
            gold_fact_ids: &["longmem-rehearsal-date", "longmem-rehearsal-cluster"],
            facts: vec![
                RetrievalFact {
                    id: "longmem-rehearsal-date",
                    content: "The migration rehearsal is scheduled for Tuesday",
                },
                RetrievalFact {
                    id: "longmem-rehearsal-cluster",
                    content: "Tuesday rehearsals must use the staging cluster",
                },
                RetrievalFact {
                    id: "longmem-rehearsal-distractor-1",
                    content: "Friday production deploys use the primary cluster",
                },
                RetrievalFact {
                    id: "longmem-rehearsal-distractor-2",
                    content: "The migration checklist was renamed after planning",
                },
            ],
        },
        RetrievalCase {
            id: "longmemeval-policy-negation-ood",
            dataset: "longmemeval",
            category: "policy-negation-ood",
            question: "which action is forbidden by release policy without maintainer approval",
            gold_answer: "push or take outward-facing actions",
            gold_fact_ids: &["longmem-policy-gold"],
            facts: vec![
                RetrievalFact {
                    id: "longmem-policy-gold",
                    content: "The release policy says do not push or take outward-facing actions without maintainer approval",
                },
                RetrievalFact {
                    id: "longmem-policy-distractor-1",
                    content: "The release policy allows local tests without approval",
                },
                RetrievalFact {
                    id: "longmem-policy-distractor-2",
                    content: "Maintainer notes mention approval templates for documentation",
                },
            ],
        },
        RetrievalCase {
            id: "longmemeval-entity-disambiguation-ood",
            dataset: "longmemeval",
            category: "entity-disambiguation-ood",
            question: "which vector backend does the artesian homelab collection use",
            gold_answer: "Qdrant",
            gold_fact_ids: &["longmem-backend-gold"],
            facts: vec![
                RetrievalFact {
                    id: "longmem-backend-gold",
                    content: "The homelab collection named artesian uses Qdrant as the vector backend",
                },
                RetrievalFact {
                    id: "longmem-backend-distractor-1",
                    content: "The local smoke fixture uses sqlite vec for temporary tests",
                },
                RetrievalFact {
                    id: "longmem-backend-distractor-2",
                    content: "The ferritex project uses a separate collection",
                },
            ],
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use aquifer::{MemoryId, MemoryRecord};

    #[test]
    fn rouge_l_rewards_token_overlap_without_exact_match() {
        let score = rouge_l_f1("the staging cluster is used", "staging cluster");
        assert!(score > 0.5, "score was {score}");
    }

    #[test]
    fn baseline_comparison_flags_recall_regression() {
        let current = RegressionReport {
            spdx_license: SPDX_LICENSE.to_string(),
            schema_version: SCHEMA_VERSION,
            k: 3,
            tolerance: 0.02,
            cases: 1,
            backends: vec![BackendMetrics {
                backend: "files".to_string(),
                cases: 1,
                recall_at_k: 0.90,
                rouge_l_f1: 1.0,
                task_success: 1.0,
                case_metrics: Vec::new(),
            }],
            leak_gate: leak_report(true),
        };
        let mut baseline = current.clone();
        baseline.backends[0].recall_at_k = 1.0;

        let comparison = compare_to_baseline(&current, &baseline);

        assert!(!comparison.passed);
        assert!(comparison.failures[0].contains("regressed"));
    }

    #[test]
    fn partition_validator_detects_injected_b_private_hit() {
        let project_hits = vec![
            hit("node:partition-a"),
            hit("node:partition-shared"),
            hit("node:partition-b"),
        ];
        let default_hits = vec![hit("node:partition-shared")];

        let result = validate_partition_hits(&project_hits, &default_hits);

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn files_leak_gate_passes_correctly_tagged_data() {
        let report = run_files_leak_gate().await.expect("leak gate passes");

        assert!(report.passed);
        assert!(report
            .project_hits
            .contains(&"node:partition-a".to_string()));
        assert!(!report
            .project_hits
            .contains(&"node:partition-b".to_string()));
        assert!(!report
            .default_hits
            .contains(&"node:partition-a".to_string()));
        assert!(!report
            .default_hits
            .contains(&"node:partition-b".to_string()));
    }

    #[tokio::test]
    async fn regression_suite_reports_both_backends() {
        let report = run_regression_suite(DEFAULT_K, DEFAULT_TOLERANCE)
            .await
            .expect("regression suite runs");

        assert_eq!(report.cases, fixed_cases().len());
        assert_eq!(report.backends.len(), 2);
        assert!(report.leak_gate.passed);
        for backend in &report.backends {
            assert_eq!(backend.cases, report.cases);
            assert!(backend.recall_at_k > 0.0, "{backend:?}");
        }
    }

    fn leak_report(passed: bool) -> LeakGateReport {
        LeakGateReport {
            backend: "files".to_string(),
            project: "A".to_string(),
            project_hits: Vec::new(),
            default_hits: Vec::new(),
            passed,
        }
    }

    fn hit(node_id: &str) -> SearchHit {
        SearchHit::keyword(
            MemoryRecord::new(
                MemoryId::new(node_id),
                node_id,
                format!("content for {node_id}"),
                Vec::new(),
                BTreeMap::new(),
                MemoryTier::L1Atom,
            ),
            1.0,
        )
    }
}
