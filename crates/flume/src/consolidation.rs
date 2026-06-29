// SPDX-License-Identifier: Apache-2.0

//! Opt-in offline skill consolidation for completed agent sessions.
//!
//! The cycle is deliberately pull-based and disabled by default. It harvests completed-session
//! signals read-only, replays candidate skills through an injected replayer, and writes a governed
//! OCF session only after the caller's qualify gate admits the candidate.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    future::Future,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    pin::Pin,
    sync::Arc,
    time::{Duration, Instant},
};

use aquifer::{MemoryBackend, SessionKey, SessionListFilter, SessionStore, SessionSummary};
use headgate::{
    count_tokens, CcsSchema, CommittedContextState, CommittedEntry, LifecycleEntry,
    QualifyDecision, QualifyGate, RecallItem, SnapshotEntry, WorkingContextBundle,
    WorkingContextSnapshot,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const CONSOLIDATED_SKILL_SESSION_ID: &str = "offline-skill-consolidation";
pub const CONSOLIDATED_SKILL_PRODUCER: &str = "flume-offline-consolidation";
const MAX_SIGNAL_TEXT_CHARS: usize = 4_000;

pub type SkillReplayFuture<'a> =
    Pin<Box<dyn Future<Output = anyhow::Result<SkillReplayOutcome>> + Send + 'a>>;

/// Replay hook used by the offline cycle.
///
/// Production callers can wire this to a dry-run harness or agent runtime; tests use deterministic
/// stubs. The trait keeps flume's consolidation logic independent of live agent CLIs.
pub trait SkillReplayer: Send {
    fn replay<'a>(
        &'a mut self,
        candidate: &'a SkillCandidate,
        attempt: usize,
    ) -> SkillReplayFuture<'a>;
}

/// Offline consolidation is opt-in. `Default` keeps the whole cycle disabled.
#[derive(Debug, Clone, PartialEq)]
pub struct OfflineConsolidationConfig {
    pub enabled: bool,
    pub harvest: HarvestConfig,
    pub budget: EditBudget,
    pub variance: VarianceGateConfig,
    pub max_wall: Duration,
    /// Maximum replay attempts across the whole cycle.
    pub max_turns: usize,
    pub min_occurrences: usize,
    pub project: Option<String>,
}

impl Default for OfflineConsolidationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            harvest: HarvestConfig::default(),
            budget: EditBudget::default(),
            variance: VarianceGateConfig::default(),
            max_wall: Duration::from_secs(60),
            max_turns: 12,
            min_occurrences: 2,
            project: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarvestConfig {
    pub home_dir: Option<PathBuf>,
    pub claude_history_path: Option<PathBuf>,
    pub codex_session_roots: Vec<PathBuf>,
    pub max_ocf_sessions: usize,
    pub max_history_lines: usize,
    pub max_codex_files: usize,
    pub max_file_bytes: u64,
}

impl Default for HarvestConfig {
    fn default() -> Self {
        Self {
            home_dir: std::env::var_os("HOME").map(PathBuf::from),
            claude_history_path: None,
            codex_session_roots: Vec::new(),
            max_ocf_sessions: 128,
            max_history_lines: 2_000,
            max_codex_files: 64,
            max_file_bytes: 2 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EditBudget {
    pub max_edits: usize,
    pub max_estimated_tokens: usize,
    pub max_estimated_cost_micros: u64,
}

impl Default for EditBudget {
    fn default() -> Self {
        Self {
            max_edits: 2,
            max_estimated_tokens: 4_096,
            max_estimated_cost_micros: 50_000,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VarianceGateConfig {
    pub runs: usize,
    pub disagreement_threshold: f32,
}

impl Default for VarianceGateConfig {
    fn default() -> Self {
        Self {
            runs: 3,
            disagreement_threshold: 0.5,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionSignalSource {
    OcfSession,
    ClaudeHistory,
    CodexJsonl,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletedSessionSignal {
    pub id: String,
    pub source: SessionSignalSource,
    pub task: String,
    pub text: String,
    pub occurred_at: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillCandidate {
    pub id: String,
    pub task: String,
    pub evidence: Vec<CompletedSessionSignal>,
}

impl SkillCandidate {
    pub fn occurrences(&self) -> usize {
        self.evidence.len()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillReplayOutcome {
    pub improved_skill: String,
    /// Label-free ACC/replay verdict. Variance compares this with qualify-gate admission.
    pub acc_accepted: bool,
    pub score: f32,
    pub estimated_cost_micros: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VarianceGateReport {
    pub runs: usize,
    pub acc_accepted: usize,
    pub qualify_admitted: usize,
    pub disagreements: usize,
    pub disagreement_rate: f32,
    pub high_variance: bool,
    pub outcomes: Vec<SkillReplayOutcome>,
    pub decisions: Vec<QualifyDecision>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SkillEditKind {
    Reuse,
    Rewrite,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillEdit {
    pub id: String,
    pub candidate_id: String,
    pub task: String,
    pub content: String,
    pub kind: SkillEditKind,
    pub priority: f32,
    pub estimated_tokens: usize,
    pub estimated_cost_micros: u64,
    pub variance: VarianceGateReport,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OfflineConsolidationReport {
    pub enabled: bool,
    pub signals_harvested: usize,
    pub candidates_mined: usize,
    pub replay_attempts: usize,
    pub variance_flagged: usize,
    pub edits_ranked: usize,
    pub edits_selected: usize,
    pub written: usize,
    pub gate_rejected: usize,
    pub stopped_reason: Option<String>,
    pub sessions_written: Vec<SessionKey>,
}

impl OfflineConsolidationReport {
    fn disabled() -> Self {
        Self {
            enabled: false,
            signals_harvested: 0,
            candidates_mined: 0,
            replay_attempts: 0,
            variance_flagged: 0,
            edits_ranked: 0,
            edits_selected: 0,
            written: 0,
            gate_rejected: 0,
            stopped_reason: Some("disabled".to_string()),
            sessions_written: Vec::new(),
        }
    }
}

pub async fn run_offline_consolidation_cycle(
    backend: Arc<dyn MemoryBackend>,
    gate: &dyn QualifyGate,
    replayer: &mut dyn SkillReplayer,
    config: OfflineConsolidationConfig,
) -> anyhow::Result<OfflineConsolidationReport> {
    if !config.enabled {
        return Ok(OfflineConsolidationReport::disabled());
    }

    let started = Instant::now();
    let signals = harvest_completed_session_signals(backend.clone(), &config.harvest).await?;
    let ccs = committed_state_from_signals(&signals, config.budget.max_estimated_tokens);
    let candidates = mine_recurring_skill_candidates(&signals, config.min_occurrences);
    let mut report = OfflineConsolidationReport {
        enabled: true,
        signals_harvested: signals.len(),
        candidates_mined: candidates.len(),
        replay_attempts: 0,
        variance_flagged: 0,
        edits_ranked: 0,
        edits_selected: 0,
        written: 0,
        gate_rejected: 0,
        stopped_reason: None,
        sessions_written: Vec::new(),
    };

    let mut edits = Vec::new();
    for candidate in candidates {
        if report.replay_attempts >= config.max_turns {
            report.stopped_reason = Some("max-turns".to_string());
            break;
        }
        if started.elapsed() >= config.max_wall {
            report.stopped_reason = Some("max-wall".to_string());
            break;
        }
        let remaining_runs = config.max_turns.saturating_sub(report.replay_attempts);
        let runs = config.variance.runs.max(1).min(remaining_runs);
        let variance = variance_gate(
            &candidate,
            replayer,
            gate,
            &ccs,
            VarianceGateConfig {
                runs,
                ..config.variance
            },
        )
        .await?;
        report.replay_attempts += variance.runs;
        if variance.high_variance {
            report.variance_flagged += 1;
        }
        if let Some(edit) = edit_from_variance(&candidate, variance) {
            edits.push(edit);
        }
    }

    report.edits_ranked = edits.len();
    let selected = rank_and_select(edits, config.budget);
    report.edits_selected = selected.len();

    for edit in selected {
        if started.elapsed() >= config.max_wall {
            report.stopped_reason = Some("max-wall".to_string());
            break;
        }
        let item = RecallItem::new(edit.id.clone(), edit.content.clone(), edit.priority)
            .with_source(CONSOLIDATED_SKILL_PRODUCER);
        let decision = gate.qualify(&item, &ccs).await;
        if !decision.admitted {
            report.gate_rejected += 1;
            continue;
        }
        let key = write_skill_ocf_session(backend.clone(), &edit, decision).await?;
        report.written += 1;
        report.sessions_written.push(key);
    }

    Ok(report)
}

pub async fn harvest_completed_session_signals(
    backend: Arc<dyn MemoryBackend>,
    config: &HarvestConfig,
) -> anyhow::Result<Vec<CompletedSessionSignal>> {
    let mut signals = harvest_ocf_session_signals(backend, config.max_ocf_sessions).await?;
    if let Some(path) = claude_history_path(config) {
        signals.extend(read_jsonl_signals(
            &path,
            SessionSignalSource::ClaudeHistory,
            config.max_history_lines,
            config.max_file_bytes,
        )?);
    }
    for path in codex_jsonl_paths(config)? {
        signals.extend(read_jsonl_signals(
            &path,
            SessionSignalSource::CodexJsonl,
            config.max_history_lines,
            config.max_file_bytes,
        )?);
    }
    Ok(signals)
}

pub fn mine_recurring_skill_candidates(
    signals: &[CompletedSessionSignal],
    min_occurrences: usize,
) -> Vec<SkillCandidate> {
    let mut grouped: BTreeMap<String, Vec<CompletedSessionSignal>> = BTreeMap::new();
    for signal in signals {
        let key = task_key(&signal.task);
        if key.is_empty() {
            continue;
        }
        grouped.entry(key).or_default().push(signal.clone());
    }

    let mut candidates = grouped
        .into_iter()
        .filter_map(|(key, evidence)| {
            (evidence.len() >= min_occurrences.max(1)).then(|| {
                let task = evidence
                    .iter()
                    .max_by_key(|signal| signal.text.len())
                    .map(|signal| signal.task.clone())
                    .unwrap_or_else(|| key.clone());
                SkillCandidate {
                    id: format!("skill-candidate-{}", stable_hash(&key)),
                    task,
                    evidence,
                }
            })
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        right
            .occurrences()
            .cmp(&left.occurrences())
            .then_with(|| left.id.cmp(&right.id))
    });
    candidates
}

pub async fn variance_gate(
    candidate: &SkillCandidate,
    replayer: &mut dyn SkillReplayer,
    gate: &dyn QualifyGate,
    ccs: &CommittedContextState,
    config: VarianceGateConfig,
) -> anyhow::Result<VarianceGateReport> {
    let runs = config.runs.max(1);
    let mut outcomes = Vec::with_capacity(runs);
    let mut decisions = Vec::with_capacity(runs);
    let mut acc_accepted = 0usize;
    let mut qualify_admitted = 0usize;
    let mut disagreements = 0usize;

    for attempt in 0..runs {
        let outcome = replayer.replay(candidate, attempt).await?;
        let item = RecallItem::new(
            format!("{}:replay-{attempt}", candidate.id),
            outcome.improved_skill.clone(),
            outcome.score,
        )
        .with_source(CONSOLIDATED_SKILL_PRODUCER);
        let decision = gate.qualify(&item, ccs).await;
        if outcome.acc_accepted {
            acc_accepted += 1;
        }
        if decision.admitted {
            qualify_admitted += 1;
        }
        if outcome.acc_accepted != decision.admitted {
            disagreements += 1;
        }
        outcomes.push(outcome);
        decisions.push(decision);
    }

    let disagreement_rate = disagreements as f32 / runs as f32;
    Ok(VarianceGateReport {
        runs,
        acc_accepted,
        qualify_admitted,
        disagreements,
        disagreement_rate,
        high_variance: disagreement_rate >= config.disagreement_threshold.clamp(0.0, 1.0),
        outcomes,
        decisions,
    })
}

pub fn rank_and_select(mut edits: Vec<SkillEdit>, budget: EditBudget) -> Vec<SkillEdit> {
    edits.sort_by(|left, right| {
        right
            .priority
            .partial_cmp(&left.priority)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.id.cmp(&right.id))
    });

    let mut selected = Vec::new();
    let mut tokens = 0usize;
    let mut cost = 0u64;
    for edit in edits {
        if selected.len() >= budget.max_edits {
            break;
        }
        if tokens.saturating_add(edit.estimated_tokens) > budget.max_estimated_tokens {
            continue;
        }
        if cost.saturating_add(edit.estimated_cost_micros) > budget.max_estimated_cost_micros {
            continue;
        }
        tokens += edit.estimated_tokens;
        cost += edit.estimated_cost_micros;
        selected.push(edit);
    }
    selected
}

async fn harvest_ocf_session_signals(
    backend: Arc<dyn MemoryBackend>,
    limit: usize,
) -> anyhow::Result<Vec<CompletedSessionSignal>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let store = SessionStore::new(backend);
    let mut summaries = store.list(SessionListFilter::default()).await?;
    summaries.sort_by_key(|summary| std::cmp::Reverse(summary.updated_at));
    summaries.truncate(limit);

    let mut signals = Vec::new();
    for summary in summaries {
        if let Some(session) = store.load(&summary.key).await? {
            signals.extend(signals_from_session(&summary, &session.snapshot));
        }
    }
    Ok(signals)
}

fn signals_from_session(summary: &SessionSummary, snapshot: &Value) -> Vec<CompletedSessionSignal> {
    snapshot
        .get("entries")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
        .filter_map(|(index, entry)| {
            let text = entry.get("content").and_then(Value::as_str)?;
            let text = compact(redact_secrets(text), MAX_SIGNAL_TEXT_CHARS);
            if text.is_empty() {
                return None;
            }
            let slot = entry
                .get("slot")
                .and_then(Value::as_str)
                .unwrap_or("fact")
                .to_string();
            let id = entry
                .get("id")
                .and_then(Value::as_str)
                .map(ToString::to_string)
                .unwrap_or_else(|| format!("{}:{index}", summary.key.node_id()));
            Some(CompletedSessionSignal {
                id,
                source: SessionSignalSource::OcfSession,
                task: task_from_text(&slot, &text),
                text,
                occurred_at: Some(summary.updated_at.to_rfc3339()),
            })
        })
        .collect()
}

fn read_jsonl_signals(
    path: &Path,
    source: SessionSignalSource,
    max_lines: usize,
    max_file_bytes: u64,
) -> anyhow::Result<Vec<CompletedSessionSignal>> {
    let Ok(metadata) = fs::metadata(path) else {
        return Ok(Vec::new());
    };
    if !metadata.is_file() || metadata.len() > max_file_bytes {
        return Ok(Vec::new());
    }
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut signals = Vec::new();
    for (line_index, line) in reader.lines().take(max_lines).enumerate() {
        let line = line?;
        let Ok(value) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let fragments = json_text_fragments(&value);
        if fragments.is_empty() {
            continue;
        }
        let text = compact(redact_secrets(&fragments.join("\n")), MAX_SIGNAL_TEXT_CHARS);
        if text.is_empty() {
            continue;
        }
        signals.push(CompletedSessionSignal {
            id: format!("{}:{line_index}", path.display()),
            source: source.clone(),
            task: task_from_json(&value).unwrap_or_else(|| task_from_text("session", &text)),
            text,
            occurred_at: timestamp_from_json(&value),
        });
    }
    Ok(signals)
}

fn claude_history_path(config: &HarvestConfig) -> Option<PathBuf> {
    config.claude_history_path.clone().or_else(|| {
        config
            .home_dir
            .as_ref()
            .map(|home| home.join(".claude/history.jsonl"))
    })
}

fn codex_jsonl_paths(config: &HarvestConfig) -> anyhow::Result<Vec<PathBuf>> {
    let roots = if config.codex_session_roots.is_empty() {
        default_codex_roots(config.home_dir.as_deref())
    } else {
        config.codex_session_roots.clone()
    };
    let mut paths = Vec::new();
    for root in roots {
        collect_jsonl_paths(&root, config.max_codex_files, &mut paths)?;
        if paths.len() >= config.max_codex_files {
            break;
        }
    }
    paths.sort();
    paths.truncate(config.max_codex_files);
    Ok(paths)
}

fn default_codex_roots(home: Option<&Path>) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(codex_home) = std::env::var_os("CODEX_HOME").map(PathBuf::from) {
        roots.push(codex_home.join("sessions"));
        roots.push(codex_home.join("archived_sessions"));
    }
    if let Some(home) = home {
        roots.push(home.join(".codex/sessions"));
        roots.push(home.join(".codex/archived_sessions"));
    }
    roots
}

fn collect_jsonl_paths(root: &Path, limit: usize, out: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    if out.len() >= limit || !root.exists() {
        return Ok(());
    }
    let mut entries = fs::read_dir(root)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    entries.sort();
    for path in entries {
        if out.len() >= limit {
            break;
        }
        if path.is_dir() {
            collect_jsonl_paths(&path, limit, out)?;
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("jsonl") {
            out.push(path);
        }
    }
    Ok(())
}

fn json_text_fragments(value: &Value) -> Vec<String> {
    let mut fragments = Vec::new();
    collect_json_text(value, &mut fragments);
    fragments
}

fn collect_json_text(value: &Value, fragments: &mut Vec<String>) {
    match value {
        Value::String(text) => {
            let text = text.trim();
            if !text.is_empty() {
                fragments.push(text.to_string());
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_json_text(item, fragments);
            }
        }
        Value::Object(object) => {
            for key in [
                "task", "title", "summary", "prompt", "message", "content", "text", "result",
                "output",
            ] {
                if let Some(value) = object.get(key) {
                    collect_json_text(value, fragments);
                }
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

fn task_from_json(value: &Value) -> Option<String> {
    ["task", "title", "summary", "prompt"]
        .iter()
        .filter_map(|key| value.get(key).and_then(Value::as_str))
        .map(str::trim)
        .find(|text| !text.is_empty())
        .map(|text| compact(redact_secrets(text), 160))
}

fn timestamp_from_json(value: &Value) -> Option<String> {
    ["timestamp", "created_at", "time"]
        .iter()
        .filter_map(|key| value.get(key).and_then(Value::as_str))
        .map(ToString::to_string)
        .next()
}

fn task_from_text(slot: &str, text: &str) -> String {
    let first = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or(text);
    compact(format!("{slot}: {first}"), 160)
}

fn task_key(task: &str) -> String {
    let stop_words = [
        "a", "an", "and", "for", "from", "in", "of", "on", "or", "the", "to", "with",
    ];
    let stop_words = stop_words.into_iter().collect::<BTreeSet<_>>();
    task.split(|character: char| !character.is_ascii_alphanumeric())
        .map(str::trim)
        .filter(|word| !word.is_empty())
        .map(str::to_ascii_lowercase)
        .filter(|word| !stop_words.contains(word.as_str()))
        .take(12)
        .collect::<Vec<_>>()
        .join("-")
}

fn committed_state_from_signals(
    signals: &[CompletedSessionSignal],
    budget_tokens: usize,
) -> CommittedContextState {
    let mut ccs = CommittedContextState::new(
        CcsSchema::new(["skill", "task-state", "fact", "evidence"]),
        budget_tokens.max(1),
    );
    for signal in signals
        .iter()
        .filter(|signal| signal.source == SessionSignalSource::OcfSession)
    {
        if ccs.token_count() >= ccs.budget_tokens() {
            break;
        }
        let slot = if signal.task.to_ascii_lowercase().contains("skill") {
            "skill"
        } else {
            "evidence"
        };
        ccs.admit(CommittedEntry::new(
            signal.id.clone(),
            slot,
            signal.text.clone(),
            1.0,
        ));
    }
    ccs
}

fn edit_from_variance(
    candidate: &SkillCandidate,
    variance: VarianceGateReport,
) -> Option<SkillEdit> {
    let best = variance
        .outcomes
        .iter()
        .max_by(|left, right| {
            left.score
                .partial_cmp(&right.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })?
        .clone();
    let kind = if variance.high_variance {
        SkillEditKind::Rewrite
    } else {
        SkillEditKind::Reuse
    };
    let content = match kind {
        SkillEditKind::Reuse => best.improved_skill,
        SkillEditKind::Rewrite => format!(
            "rewrite skill for recurring task `{}`; ACC/qualify disagreement rate {:.2}: {}",
            candidate.task, variance.disagreement_rate, best.improved_skill
        ),
    };
    let priority =
        (candidate.occurrences() as f32 + best.score - variance.disagreement_rate).max(0.0);
    Some(SkillEdit {
        id: format!("consolidated-skill-{}", stable_hash(&content)),
        candidate_id: candidate.id.clone(),
        task: candidate.task.clone(),
        estimated_tokens: count_tokens(&content),
        estimated_cost_micros: best.estimated_cost_micros,
        content,
        kind,
        priority,
        variance,
    })
}

async fn write_skill_ocf_session(
    backend: Arc<dyn MemoryBackend>,
    edit: &SkillEdit,
    decision: QualifyDecision,
) -> anyhow::Result<SessionKey> {
    let mut metadata = BTreeMap::new();
    metadata.insert("kind".to_string(), format!("{:?}", edit.kind));
    metadata.insert(
        "variance_disagreement_rate".to_string(),
        format!("{:.3}", edit.variance.disagreement_rate),
    );
    let skill_entry = SnapshotEntry::now(
        edit.id.clone(),
        "skill",
        edit.content.clone(),
        decision.score,
    );
    let evidence_entry = SnapshotEntry::now(
        format!("{}:task", edit.id),
        "evidence",
        format!(
            "task: {}\noccurrences: {}\nkind: {:?}\nmetadata: {:?}",
            edit.task, edit.variance.runs, edit.kind, metadata
        ),
        1.0,
    );
    let entries = vec![skill_entry, evidence_entry];
    let token_count: usize = entries.iter().map(|entry| entry.tokens).sum();
    let snapshot = WorkingContextSnapshot {
        schema: vec![
            "skill".to_string(),
            "task-state".to_string(),
            "fact".to_string(),
            "evidence".to_string(),
        ],
        budget_tokens: token_count.max(1) + 1_024,
        token_count,
        entries,
    };
    let lifecycle = vec![decision
        .audit
        .map(|audit| LifecycleEntry::commit_with_audit(edit.id.clone(), audit))
        .unwrap_or_else(|| LifecycleEntry::commit(edit.id.clone()))];
    let bundle = WorkingContextBundle::new(snapshot, lifecycle);
    let key = SessionKey::new(
        Some("flume".to_string()),
        Some(CONSOLIDATED_SKILL_SESSION_ID.to_string()),
        Some(edit.id.clone()),
    );
    let session = bundle.to_ocf_session(&key, Some(CONSOLIDATED_SKILL_PRODUCER.to_string()))?;
    SessionStore::new(backend).store(session).await?;
    Ok(key)
}

fn compact(input: impl AsRef<str>, limit: usize) -> String {
    input
        .as_ref()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(limit)
        .collect()
}

fn stable_hash(input: &str) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in input.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn redact_secrets(input: &str) -> String {
    let mut output = input.to_string();
    for prefix in ["sk-", "ghp_", "github_pat_", "xoxb-", "xoxp-", "hf_"] {
        output = redact_prefixed_token(&output, prefix);
    }
    output
}

fn redact_prefixed_token(input: &str, prefix: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut cursor = 0usize;
    while let Some(relative_start) = input[cursor..].find(prefix) {
        let start = cursor + relative_start;
        output.push_str(&input[cursor..start]);
        output.push_str("[REDACTED]");
        let mut end = start + prefix.len();
        for (offset, character) in input[end..].char_indices() {
            if !(character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.')) {
                break;
            }
            end = start + prefix.len() + offset + character.len_utf8();
        }
        cursor = end;
    }
    output.push_str(&input[cursor..]);
    output
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use aquifer::{FilesBackend, SessionStore};
    use artesian_test_support::TempDir;
    use headgate::{
        QualifyAudit, QualifySignal, SnapshotEntry, WorkingContextBundle, WorkingContextSnapshot,
    };

    use super::*;

    struct SequenceReplayer {
        outcomes: Mutex<Vec<SkillReplayOutcome>>,
    }

    impl SequenceReplayer {
        fn new(outcomes: Vec<SkillReplayOutcome>) -> Self {
            Self {
                outcomes: Mutex::new(outcomes.into_iter().rev().collect()),
            }
        }
    }

    impl SkillReplayer for SequenceReplayer {
        fn replay<'a>(
            &'a mut self,
            _candidate: &'a SkillCandidate,
            _attempt: usize,
        ) -> SkillReplayFuture<'a> {
            Box::pin(async move {
                self.outcomes
                    .lock()
                    .unwrap()
                    .pop()
                    .ok_or_else(|| anyhow::anyhow!("no replay outcome"))
            })
        }
    }

    struct StaticGate {
        admitted: bool,
    }

    impl QualifyGate for StaticGate {
        fn qualify<'a>(
            &'a self,
            item: &'a RecallItem,
            _ccs: &'a CommittedContextState,
        ) -> Pin<Box<dyn Future<Output = QualifyDecision> + Send + 'a>> {
            Box::pin(async move {
                let audit = QualifyAudit::from_signals(
                    self.admitted,
                    vec![QualifySignal::new("test", item.score, 0.5, self.admitted)],
                );
                if self.admitted {
                    QualifyDecision::admit("skill", item.score).with_audit(audit)
                } else {
                    QualifyDecision::reject("test reject", item.score).with_audit(audit)
                }
            })
        }
    }

    fn replay_outcome(text: &str, acc_accepted: bool, score: f32) -> SkillReplayOutcome {
        SkillReplayOutcome {
            improved_skill: text.to_string(),
            acc_accepted,
            score,
            estimated_cost_micros: 1_000,
        }
    }

    fn edit(id: &str, priority: f32, tokens: usize, cost: u64) -> SkillEdit {
        SkillEdit {
            id: id.to_string(),
            candidate_id: format!("{id}-candidate"),
            task: "task".to_string(),
            content: format!("skill {id}"),
            kind: SkillEditKind::Reuse,
            priority,
            estimated_tokens: tokens,
            estimated_cost_micros: cost,
            variance: VarianceGateReport {
                runs: 1,
                acc_accepted: 1,
                qualify_admitted: 1,
                disagreements: 0,
                disagreement_rate: 0.0,
                high_variance: false,
                outcomes: vec![replay_outcome("skill", true, priority)],
                decisions: vec![QualifyDecision::admit("skill", priority)],
            },
        }
    }

    #[tokio::test]
    async fn consolidation_cycle_writes_only_after_gate_approval() {
        let tempdir = TempDir::new("flume-consolidation-cycle");
        let backend = Arc::new(FilesBackend::new(tempdir.path()));
        store_ocf_fixture(
            backend.clone(),
            "format rust workspace after edits",
            "format rust workspace after edits",
        )
        .await;
        let claude_history = tempdir.join("history.jsonl");
        std::fs::write(
            &claude_history,
            r#"{"timestamp":"2026-06-29T00:00:00Z","task":"format rust workspace after edits","result":"cargo fmt passed"}"#,
        )
        .unwrap();
        let codex_root = tempdir.join("codex");
        std::fs::create_dir_all(&codex_root).unwrap();
        std::fs::write(
            codex_root.join("session.jsonl"),
            r#"{"timestamp":"2026-06-29T00:01:00Z","task":"format rust workspace after edits","message":"rerun cargo fmt after patches"}"#,
        )
        .unwrap();

        let config = OfflineConsolidationConfig {
            enabled: true,
            harvest: HarvestConfig {
                home_dir: Some(tempdir.path().to_path_buf()),
                claude_history_path: Some(claude_history),
                codex_session_roots: vec![codex_root],
                ..HarvestConfig::default()
            },
            variance: VarianceGateConfig {
                runs: 1,
                disagreement_threshold: 0.5,
            },
            max_turns: 2,
            ..OfflineConsolidationConfig::default()
        };

        let mut replayer = SequenceReplayer::new(vec![replay_outcome(
            "verified skill: run cargo fmt after patching Rust files",
            true,
            0.9,
        )]);
        let approved = run_offline_consolidation_cycle(
            backend.clone(),
            &StaticGate { admitted: true },
            &mut replayer,
            config.clone(),
        )
        .await
        .expect("approved cycle should run");
        assert_eq!(approved.written, 1);
        let key = approved
            .sessions_written
            .first()
            .expect("approved cycle should write an OCF session");
        let session = SessionStore::new(backend.clone())
            .load(key)
            .await
            .unwrap()
            .expect("written OCF session should load");
        assert_eq!(
            session.handed_off_from().as_deref(),
            Some(CONSOLIDATED_SKILL_PRODUCER)
        );

        let rejecting_backend = Arc::new(FilesBackend::new(tempdir.join("rejecting")));
        store_ocf_fixture(
            rejecting_backend.clone(),
            "format rust workspace after edits",
            "format rust workspace after edits",
        )
        .await;
        let mut rejecting_replayer = SequenceReplayer::new(vec![replay_outcome(
            "verified skill: run cargo fmt after patching Rust files",
            true,
            0.9,
        )]);
        let rejected = run_offline_consolidation_cycle(
            rejecting_backend,
            &StaticGate { admitted: false },
            &mut rejecting_replayer,
            OfflineConsolidationConfig {
                harvest: HarvestConfig {
                    claude_history_path: None,
                    codex_session_roots: Vec::new(),
                    ..config.harvest
                },
                min_occurrences: 1,
                ..config
            },
        )
        .await
        .expect("rejected cycle should run");
        assert_eq!(rejected.written, 0);
        assert_eq!(rejected.gate_rejected, 1);
        assert!(rejected.sessions_written.is_empty());
    }

    #[test]
    fn rank_and_select_bounds_edits_to_budget() {
        let selected = rank_and_select(
            vec![
                edit("low", 0.1, 10, 100),
                edit("high", 3.0, 40, 200),
                edit("too-expensive", 4.0, 40, 10_000),
                edit("mid", 2.0, 30, 200),
            ],
            EditBudget {
                max_edits: 2,
                max_estimated_tokens: 75,
                max_estimated_cost_micros: 500,
            },
        );
        assert_eq!(
            selected
                .iter()
                .map(|edit| edit.id.as_str())
                .collect::<Vec<_>>(),
            vec!["high", "mid"]
        );
    }

    #[tokio::test]
    async fn variance_gate_flags_high_disagreement_for_rewrite() {
        let candidate = SkillCandidate {
            id: "candidate".to_string(),
            task: "rerun flaky verifier".to_string(),
            evidence: Vec::new(),
        };
        let mut replayer = SequenceReplayer::new(vec![
            replay_outcome("skill", true, 0.9),
            replay_outcome("skill", true, 0.9),
            replay_outcome("skill", true, 0.9),
        ]);
        let ccs = CommittedContextState::new(CcsSchema::default(), 1024);
        let report = variance_gate(
            &candidate,
            &mut replayer,
            &StaticGate { admitted: false },
            &ccs,
            VarianceGateConfig {
                runs: 3,
                disagreement_threshold: 0.5,
            },
        )
        .await
        .unwrap();
        assert!(report.high_variance);
        let edit = edit_from_variance(&candidate, report).expect("variance should produce edit");
        assert_eq!(edit.kind, SkillEditKind::Rewrite);
    }

    async fn store_ocf_fixture(backend: Arc<dyn MemoryBackend>, task: &str, content: &str) {
        let entry = SnapshotEntry::now("fixture-entry", "skill", content, 1.0);
        let snapshot = WorkingContextSnapshot {
            schema: vec!["skill".to_string(), "fact".to_string()],
            budget_tokens: entry.tokens + 256,
            token_count: entry.tokens,
            entries: vec![entry],
        };
        let bundle =
            WorkingContextBundle::new(snapshot, vec![LifecycleEntry::commit("fixture-entry")]);
        let key = SessionKey::new(
            Some("test".to_string()),
            Some("fixture".to_string()),
            Some(task.to_string()),
        );
        let session = bundle
            .to_ocf_session(&key, Some("fixture".to_string()))
            .unwrap();
        SessionStore::new(backend).store(session).await.unwrap();
    }
}
