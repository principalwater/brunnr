// SPDX-License-Identifier: Apache-2.0

//! Autonomous memory-first agentic loop — the shared core used by both the CLI `loop` command
//! and the MCP `orchestrate.loop` tool.
//!
//! The loop repeats a worker action until a goal verifier command exits 0 (or until a brake
//! fires). After each turn it:
//! 1. Recalls goal-relevant memory from the backend (MMR-diversified).
//! 2. Assembles a bounded goal packet (goal + invariants + last-failed-check + recall).
//! 3. Runs the worker action with the packet injected via `ARTESIAN_PACKET` / env vars.
//! 4. Writes a resume anchor so the run survives compaction.
//! 5. Verifies the goal; on success commits a verified skill + spec + auto-invariants.
//!
//! The actual command execution is injected through the [`LoopCommands`] trait, keeping the
//! core free of shell / process specifics so the MCP path can supply its own worker executor.

use std::{
    collections::BTreeSet,
    fs,
    io::Write,
    path::{Path, PathBuf},
    pin::Pin,
    sync::Arc,
    time::{Duration, Instant},
};

use tokio_util::sync::CancellationToken;

use aquifer::{
    AnchorAnchorStore, MemoryBackend, MemoryQuery, MemoryScope, MemoryTier, SessionAnchor,
    StoreMemory,
};
use headgate::{count_tokens, record_savings};
use serde_json::{json, Value};

use crate::quota::{
    quota_threshold_events, read_local_quota_with_options, write_quota_continuation,
    QuotaContinuationContext, QuotaLoopConfig, QuotaThresholdEvent, QUOTA_CONTINUATION_NOTE,
};

// ── Brakes / constants ─────────────────────────────────────────────────────────────────────────

/// Per-turn recall limit injected into the worker — kept small to stay token-cheap.
pub const LOOP_RECALL_LIMIT: usize = 5;
/// Tag that marks a memory as a project invariant — always injected into the goal packet.
pub const LOOP_INVARIANT_TAG: &str = "invariant";
/// Cap on invariants injected into a goal packet (ranked by goal relevance).
pub const LOOP_GOAL_INVARIANT_LIMIT: usize = 8;
/// Tag that marks a memory as a verified skill — a previously goal-verified loop approach.
pub const LOOP_SKILL_TAG: &str = "skill";
/// Tag that marks a distilled, verifier-backed goal restatement.
pub const LOOP_SPEC_TAG: &str = "spec";
/// Cap on verified skills surfaced in a goal packet.
pub const LOOP_GOAL_SKILL_LIMIT: usize = 2;
/// Cap on the captured "last failed check" detail carried into the next turn.
pub const LOOP_LAST_CHECK_CHARS: usize = 800;
/// Cap on learned invariant snippets.
pub const LOOP_AUTO_INVARIANT_CHARS: usize = 240;
/// Environment variable name that holds the STOP sentinel file path.
pub const ARTESIAN_STOP_FILE_ENV: &str = "ARTESIAN_STOP_FILE";
/// Environment variable name that holds the run-log directory path.
pub const ARTESIAN_RUNS_DIR_ENV: &str = "ARTESIAN_RUNS_DIR";
/// Default sleep between poll turns.
pub const DEFAULT_LOOP_SLEEP: Duration = Duration::from_millis(500);

// ── Remediation constants ──────────────────────────────────────────────────────────────────────

/// Default maximum number of *consecutive* verify failures before the loop escalates.
/// Each failure injects the failure output into the next worker invocation via
/// [`ARTESIAN_LAST_FAILURE_ENV`] so each retry is informed rather than blind.
/// Set `max_remediation_attempts = 0` in [`LoopRunOptions`] to disable escalation entirely.
pub const LOOP_REMEDIATION_ATTEMPTS_DEFAULT: u32 = 3;
/// Environment variable injected into the worker when the previous verify failed.
/// Its value is a structured remediation directive:
/// `"PREVIOUS ATTEMPT FAILED — fix exactly this:\n<bounded verifier output>\n"`.
/// Workers that read this can target the specific failure rather than blindly retrying.
pub const ARTESIAN_LAST_FAILURE_ENV: &str = "ARTESIAN_LAST_FAILURE";
/// Cap on individual failure-reason text captured into the remediation trail and directive.
pub const LOOP_FAILURE_TRAIL_REASON_CHARS: usize = 800;

// ── Worker / verifier abstraction ─────────────────────────────────────────────────────────────

pub type LoopCommandFuture<'a, T> =
    Pin<Box<dyn std::future::Future<Output = anyhow::Result<T>> + Send + 'a>>;

/// Worker and verifier execution — injected so the CLI and MCP paths can differ.
///
/// The CLI uses shell (`sh -c`) execution. The MCP path can supply an implementation that drives
/// a `ProcessAgent` or any other executor without touching shell process semantics.
///
/// Implementations must be `Send` so they can be used inside async MCP tool handlers.
pub trait LoopCommands: Send {
    /// Run the per-turn worker action with the provided environment overrides.
    /// Returns `Ok(true)` on exit 0, `Ok(false)` on non-zero exit.
    fn run_worker<'a>(
        &'a mut self,
        cmd: &'a str,
        env: Vec<(String, String)>,
        timeout: Option<Duration>,
    ) -> LoopCommandFuture<'a, bool>;

    /// Run the verifier command. Returns `(passed, output_text)`.
    fn verify_goal<'a>(
        &'a mut self,
        cmd: &'a str,
        timeout: Option<Duration>,
    ) -> LoopCommandFuture<'a, (bool, String)>;
}

// ── Run options and report ─────────────────────────────────────────────────────────────────────

/// Type alias for the per-turn progress callback stored in [`LoopRunOptions::on_progress`].
///
/// Called with `(progress, total, message)` where `progress` is the current turn number
/// (1-based), `total` is `max_turns as f64`, and `message` is a human-readable stage label.
/// Implement using `tokio::spawn` if you need to drive async work (e.g. MCP notifications).
pub type LoopProgressCallback = Arc<dyn Fn(f64, Option<f64>, Option<String>) + Send + Sync>;

/// Runtime parameters for `run_loop_core`.
pub struct LoopRunOptions {
    /// Verifier command — exit 0 means the goal holds.
    pub goal: String,
    /// Per-turn worker command. `None` in poll mode.
    pub worker_cmd: Option<String>,
    /// Maximum turns before the loop gives up.
    pub max_turns: u32,
    /// Maximum wall-clock time before the loop aborts.
    pub max_wall: Option<Duration>,
    /// Poll mode: skip the worker, only re-check.
    pub poll: bool,
    /// Whether to store verified skill/spec/invariant on success.
    pub learn: bool,
    /// Stable run identifier written into the run-log file name and memory records.
    pub run_id: String,
    /// Directory where the JSONL run log is written.
    pub run_log_dir: PathBuf,
    /// Sentinel file path — loop stops if it exists at turn start.
    pub stop_file: PathBuf,
    /// Memory collection label used in token-savings statistics.  Pass an empty string to
    /// omit a collection label in the stats entry.
    pub collection: String,
    /// When `true` (the default), each per-turn `loop.recall` records a token-savings entry.
    /// Mirror of `config.memory.track_savings`.
    pub track_savings: bool,
    /// Maximum number of *consecutive* verify failures before escalating with a failure trail
    /// (outcome `"escalated"`). Each failing turn injects [`ARTESIAN_LAST_FAILURE_ENV`] into
    /// the next worker invocation so retries are informed by the specific failure output.
    /// Defaults to [`LOOP_REMEDIATION_ATTEMPTS_DEFAULT`]. Set to `0` to disable escalation
    /// (the loop then runs to max-turns if the goal never holds).
    pub max_remediation_attempts: u32,
    /// Cancellation token — checked at the start of every turn and inside the `tokio::select!`
    /// that drives the worker and verifier.  When fired the loop returns promptly with outcome
    /// `"cancelled"`.  Use [`CancellationToken::new()`] (never-cancelled default) when
    /// cancellation is not needed; existing call-sites are unaffected.
    pub cancel: CancellationToken,
    /// Optional per-turn progress callback.  When `Some`, called three times per turn:
    /// at turn-start, after the worker completes (pre-verify), and after the verifier.
    /// `progress` is the current 1-based turn number; `total` is `max_turns`.
    /// Drive `peer.notify_progress` from inside a `tokio::spawn` in the closure.
    /// Pass `None` to disable (default for CLI callers).
    pub on_progress: Option<LoopProgressCallback>,
    /// Local token-free coding-agent quota checks at loop turn boundaries.
    pub quota: QuotaLoopConfig,
}

/// Summary returned after `run_loop_core` completes (successfully or via a brake).
#[derive(Debug, Clone)]
pub struct LoopRunReport {
    /// How many turns ran before the loop stopped.
    pub turns: u32,
    /// Outcome label: `"success"`, `"wall-cap"`, `"max-turns"`, `"stopped"`, `"error"`,
    /// or `"escalated"` when the remediation budget is exhausted.
    pub outcome: String,
    /// Human-readable stop reason.  When `outcome == "escalated"` this includes a compact
    /// per-turn failure summary; the full structured trail is in `failure_trail`.
    pub why_stopped: String,
    /// Absolute path of the JSONL run log.
    pub run_log_path: PathBuf,
    /// Accumulated remediation failure trail.  Non-empty only when `outcome == "escalated"`.
    pub failure_trail: Vec<RemediationAttempt>,
}

// ── Run-log ───────────────────────────────────────────────────────────────────────────────────

pub struct LoopRunLog {
    path: PathBuf,
    file: fs::File,
}

impl LoopRunLog {
    pub fn create(dir: &Path, run_id: &str) -> anyhow::Result<Self> {
        fs::create_dir_all(dir)
            .map_err(|e| anyhow::anyhow!("create run-log dir {}: {e}", dir.display()))?;
        let path = dir.join(format!("{run_id}.jsonl"));
        let file = fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&path)
            .map_err(|e| anyhow::anyhow!("open run-log {}: {e}", path.display()))?;
        Ok(Self { path, file })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn write_turn(
        &mut self,
        run_id: &str,
        turn: u32,
        action: &str,
        goal_met: bool,
        check_output: &str,
        elapsed: Duration,
    ) -> anyhow::Result<()> {
        let verify_result = if goal_met { "passed" } else { "failed" };
        self.write_value(json!({
            "type": "turn",
            "run_id": run_id,
            "turn": turn,
            "action": action,
            "verify_result": verify_result,
            "verify": {
                "passed": goal_met,
                "output": compact_inline(check_output, LOOP_LAST_CHECK_CHARS),
            },
            "elapsed_ms": duration_millis(elapsed),
        }))
    }

    pub fn write_summary(
        &mut self,
        run_id: &str,
        outcome: &str,
        turns: u32,
        elapsed: Duration,
        why_stopped: &str,
    ) -> anyhow::Result<()> {
        self.write_value(json!({
            "type": "summary",
            "run_id": run_id,
            "outcome": outcome,
            "turns": turns,
            "elapsed_ms": duration_millis(elapsed),
            "why_stopped": why_stopped,
        }))?;
        self.file
            .flush()
            .map_err(|e| anyhow::anyhow!("flush run-log {}: {e}", self.path.display()))
    }

    /// Write a per-turn remediation log entry — one per turn where the previous verify failed
    /// and the worker was re-invoked with [`ARTESIAN_LAST_FAILURE_ENV`] set.
    pub fn write_remediation_attempt(
        &mut self,
        run_id: &str,
        turn: u32,
        consecutive_failures: u32,
        reason: &str,
        fix_attempt: Option<&str>,
    ) -> anyhow::Result<()> {
        self.write_value(json!({
            "type": "remediation",
            "run_id": run_id,
            "turn": turn,
            "consecutive_failures": consecutive_failures,
            "reason": compact_inline(reason, LOOP_FAILURE_TRAIL_REASON_CHARS),
            "fix_attempt": fix_attempt,
        }))
    }

    pub fn write_quota_warning(
        &mut self,
        run_id: &str,
        turn: u32,
        event: &QuotaThresholdEvent,
    ) -> anyhow::Result<()> {
        self.write_value(json!({
            "type": "quota-warning",
            "run_id": run_id,
            "turn": turn,
            "threshold_pct": event.threshold_pct,
            "high": event.high,
            "quota": event.status,
        }))
    }

    pub fn write_quota_checkpoint(
        &mut self,
        run_id: &str,
        turn: u32,
        event: &QuotaThresholdEvent,
        key: &aquifer::SessionKey,
    ) -> anyhow::Result<()> {
        self.write_value(json!({
            "type": "quota-checkpoint",
            "run_id": run_id,
            "turn": turn,
            "note": QUOTA_CONTINUATION_NOTE,
            "session": {
                "user_id": &key.user_id,
                "session_id": &key.session_id,
                "task_id": &key.task_id,
            },
            "quota": event.status,
        }))
    }

    /// Write the escalation summary entry including the full failure trail when the remediation
    /// budget is exhausted.
    pub fn write_escalation_summary(
        &mut self,
        run_id: &str,
        turns: u32,
        elapsed: Duration,
        why_stopped: &str,
        trail: &[RemediationAttempt],
    ) -> anyhow::Result<()> {
        let trail_json: Vec<Value> = trail
            .iter()
            .map(|a| {
                json!({
                    "turn": a.turn,
                    "reason": a.reason,
                    "fix_attempt": a.fix_attempt,
                })
            })
            .collect();
        self.write_value(json!({
            "type": "summary",
            "run_id": run_id,
            "outcome": "escalated",
            "turns": turns,
            "elapsed_ms": duration_millis(elapsed),
            "why_stopped": why_stopped,
            "failure_trail": trail_json,
        }))?;
        self.file
            .flush()
            .map_err(|e| anyhow::anyhow!("flush run-log {}: {e}", self.path.display()))
    }

    fn write_value(&mut self, value: Value) -> anyhow::Result<()> {
        serde_json::to_writer(&mut self.file, &value)
            .map_err(|e| anyhow::anyhow!("write run-log {}: {e}", self.path.display()))?;
        writeln!(self.file)
            .map_err(|e| anyhow::anyhow!("write run-log {}: {e}", self.path.display()))
    }
}

// ── Utilities ─────────────────────────────────────────────────────────────────────────────────

/// Track a single verifier failure for later auto-invariant extraction.
#[derive(Debug, Clone)]
pub struct FailedCheck {
    pub turn: u32,
    pub output: String,
}

/// One entry in the remediation failure trail recorded when the escalation budget is exhausted.
/// Carries the turn number, the bounded verifier output (the rejection reason), and the worker
/// command that was attempted as the fix.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RemediationAttempt {
    pub turn: u32,
    pub reason: String,
    pub fix_attempt: Option<String>,
}

pub fn duration_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

pub fn compact_inline(text: &str, limit: usize) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(limit)
        .collect()
}

pub fn stable_content_hash(text: &str) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// Generate a stable, time-based run ID.
pub fn loop_run_id() -> String {
    format!(
        "loop-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_millis())
            .unwrap_or(0)
    )
}

/// Resolve the run-log directory from env or home.
pub fn loop_run_log_dir() -> anyhow::Result<PathBuf> {
    if let Some(path) = std::env::var_os(ARTESIAN_RUNS_DIR_ENV) {
        return Ok(PathBuf::from(path));
    }
    Ok(home_dir()?.join(".artesian").join("runs"))
}

/// Resolve the STOP sentinel file path from env or home.
pub fn loop_stop_file() -> anyhow::Result<PathBuf> {
    if let Some(path) = std::env::var_os(ARTESIAN_STOP_FILE_ENV) {
        return Ok(PathBuf::from(path));
    }
    Ok(home_dir()?.join(".artesian").join("STOP"))
}

fn home_dir() -> anyhow::Result<PathBuf> {
    #[allow(deprecated)]
    std::env::home_dir().ok_or_else(|| anyhow::anyhow!("could not determine home directory"))
}

pub fn remaining_wall_budget(started_at: Instant, max_wall: Option<Duration>) -> Option<Duration> {
    max_wall.map(|max_wall| max_wall.saturating_sub(started_at.elapsed()))
}

pub fn wall_cap_message(
    started_at: Instant,
    max_wall: Option<Duration>,
    before: &str,
) -> Option<String> {
    let max_wall = max_wall?;
    (started_at.elapsed() >= max_wall).then(|| {
        format!(
            "loop exceeded max-wall-secs ({}) before {before}",
            max_wall.as_secs()
        )
    })
}

// ── Memory helpers ────────────────────────────────────────────────────────────────────────────

/// Search the backend for memory relevant to the goal; MMR-diversify to avoid crowding from
/// near-duplicate turn commits.
pub async fn loop_recall(backend: &dyn MemoryBackend, goal: &str) -> String {
    loop_recall_inner(backend, goal).await.0
}

/// Like [`loop_recall`] but also returns `(baseline_tokens, returned_tokens)` for savings
/// accounting.  `baseline_tokens` is the sum of full record content token counts for each
/// MMR-selected hit.  `returned_tokens` is the token count of the formatted output (each
/// record truncated to 280 chars).
async fn loop_recall_inner(backend: &dyn MemoryBackend, goal: &str) -> (String, usize, usize) {
    let Ok(hits) = backend
        .find(MemoryQuery::new(goal).with_limit(LOOP_RECALL_LIMIT * 3))
        .await
    else {
        return (String::new(), 0, 0);
    };
    let hits = aquifer::mmr_diversify(hits, LOOP_RECALL_LIMIT, aquifer::MMR_DEFAULT_LAMBDA);
    let baseline_tokens: usize = hits.iter().map(|h| count_tokens(&h.record.content)).sum();
    let mut lines = Vec::new();
    for hit in hits {
        let content = hit.record.content.replace('\n', " ");
        let trimmed: String = content.chars().take(280).collect();
        lines.push(format!("- {trimmed}"));
    }
    let text = lines.join("\n");
    let returned_tokens = count_tokens(&text);
    (text, baseline_tokens, returned_tokens)
}

async fn packet_tag_section(
    backend: &dyn MemoryBackend,
    goal: &str,
    tag: &str,
    limit: usize,
    title: &str,
) -> Option<String> {
    let mut query = MemoryQuery::new(goal).with_limit(limit);
    query.tags = vec![tag.to_string()];
    match backend.find(query).await {
        Ok(hits) if !hits.is_empty() => {
            let lines: Vec<String> = hits
                .iter()
                .map(|hit| format!("- {}", hit.record.content.replace('\n', " ")))
                .collect();
            Some(format!("# {title}\n{}", lines.join("\n")))
        }
        _ => None,
    }
}

/// Assemble the bounded goal packet: goal + invariants + verified skills/specs + last failed
/// check + recall — what the worker needs, not a flat dump.
pub async fn assemble_goal_packet(
    backend: Option<&dyn MemoryBackend>,
    goal: &str,
    last_check: Option<&str>,
    recall: &str,
) -> String {
    let mut sections = vec![format!("# Goal\n{goal}")];

    if let Some(backend) = backend {
        if let Some(section) = packet_tag_section(
            backend,
            goal,
            LOOP_INVARIANT_TAG,
            LOOP_GOAL_INVARIANT_LIMIT,
            "Invariants (must hold)",
        )
        .await
        {
            sections.push(section);
        }
        if let Some(section) = packet_tag_section(
            backend,
            goal,
            LOOP_SKILL_TAG,
            LOOP_GOAL_SKILL_LIMIT,
            "Known approach (verified)",
        )
        .await
        {
            sections.push(section);
        }
        if let Some(section) = packet_tag_section(
            backend,
            goal,
            LOOP_SPEC_TAG,
            LOOP_GOAL_SKILL_LIMIT,
            "Sharper specs (verified)",
        )
        .await
        {
            sections.push(section);
        }
    }

    if let Some(last_check) = last_check.filter(|check| !check.is_empty()) {
        let detail: String = last_check.chars().take(LOOP_LAST_CHECK_CHARS).collect();
        sections.push(format!("# Last failed check\n{detail}"));
    }

    if !recall.is_empty() {
        sections.push(format!("# Relevant memory\n{recall}"));
    }

    sections.join("\n\n")
}

/// Commit a concise, run-scoped atom for this turn's outcome (survives compaction via session
/// scope + run_id tag).
pub async fn loop_commit_turn(
    backend: &dyn MemoryBackend,
    run_id: &str,
    turn: u32,
    goal: &str,
    worker_cmd: Option<&str>,
    goal_met: bool,
) {
    let status = if goal_met { "goal met" } else { "goal not met" };
    let action = worker_cmd.unwrap_or("(poll)");
    let mut memory = StoreMemory::atom(format!(
        "loop {run_id} turn {turn}: ran `{action}` to verify `{goal}` -> {status}"
    ));
    memory.tier = MemoryTier::L0Raw;
    memory.tags = vec![
        "loop".to_string(),
        run_id.to_string(),
        format!("turn-{turn}"),
        if goal_met { "goal-met" } else { "goal-unmet" }.to_string(),
    ];
    memory.scope = Some(MemoryScope::Session);
    memory.session_id = Some(run_id.to_string());
    let _ = backend.store(memory).await;
}

/// Store the verified worker approach as a durable skill for future goal packets.
pub async fn loop_store_skill(
    backend: &dyn MemoryBackend,
    goal: &str,
    worker_cmd: &str,
    turns: u32,
) {
    let mut memory = StoreMemory::atom(format!(
        "verified approach for `{goal}`: run `{worker_cmd}` (passed in {turns} turn(s))"
    ));
    memory.tier = MemoryTier::L2Scenario;
    memory.tags = vec![LOOP_SKILL_TAG.to_string(), "verified".to_string()];
    let _ = backend.store(memory).await;
}

pub async fn loop_store_spec(
    backend: &dyn MemoryBackend,
    goal: &str,
    worker_cmd: Option<&str>,
    turns: u32,
) {
    let action = worker_cmd.unwrap_or("(poll)");
    let mut memory = StoreMemory::atom(format!(
        "sharper spec for future runs: make `{goal}` pass without weakening the check; \
         preserve project invariants and use `{action}` as the previously verified action."
    ));
    memory.tier = MemoryTier::L2Scenario;
    memory.tags = vec![LOOP_SPEC_TAG.to_string(), "verified".to_string()];
    memory
        .metadata
        .insert("turns".to_string(), turns.to_string());
    let _ = backend.store(memory).await;
}

pub async fn loop_store_auto_invariants(
    backend: &dyn MemoryBackend,
    goal: &str,
    worker_cmd: Option<&str>,
    failures: &[FailedCheck],
) {
    for failure in failures {
        loop_store_auto_invariant(backend, goal, worker_cmd, failure).await;
    }
}

async fn loop_store_auto_invariant(
    backend: &dyn MemoryBackend,
    goal: &str,
    worker_cmd: Option<&str>,
    failure: &FailedCheck,
) {
    let action = compact_inline(worker_cmd.unwrap_or("(poll)"), LOOP_AUTO_INVARIANT_CHARS);
    let check = compact_inline(&failure.output, LOOP_AUTO_INVARIANT_CHARS);
    let check = if check.is_empty() {
        goal.to_string()
    } else {
        check
    };
    let canonical = format!("goal={goal}\naction={action}\ncheck={check}");
    let content_hash = stable_content_hash(&canonical);
    let node_id = format!("auto-invariant:{content_hash}");
    if backend.get_node(&node_id).await.ok().flatten().is_some() {
        return;
    }
    let mut query = MemoryQuery::new(&canonical).with_limit(LOOP_GOAL_INVARIANT_LIMIT * 3);
    query.tags = vec![LOOP_INVARIANT_TAG.to_string()];
    if let Ok(hits) = backend.find(query).await {
        let already_stored = hits.iter().any(|hit| {
            hit.record.metadata.get("content_hash") == Some(&content_hash)
                || hit.record.node_id == node_id
        });
        if already_stored {
            return;
        }
    }

    let mut memory = StoreMemory::atom(format!(
        "auto-invariant: do not treat `{action}` as complete until `{goal}` passes \
         - it broke `{goal}` at turn {}: {check}",
        failure.turn
    ));
    memory.tier = MemoryTier::L3Project;
    memory.tags = vec![LOOP_INVARIANT_TAG.to_string(), "auto-invariant".to_string()];
    memory.node_id = Some(node_id);
    memory
        .metadata
        .insert("content_hash".to_string(), content_hash);
    memory.source = Some("artesian-loop".to_string());
    let _ = backend.store(memory).await;
}

// ── Core loop ─────────────────────────────────────────────────────────────────────────────────

/// The autonomous memory-first loop: each turn recalls goal-relevant memory, runs the worker
/// action (with that recall in env vars), writes a resume anchor, verifies the goal, and commits
/// a run-scoped record. Bounded by max-turns, max-wall-secs, and a STOP sentinel file.
///
/// This is the single implementation shared by the CLI `loop` command and the MCP
/// `orchestrate.loop` tool. Both sides supply their own [`LoopCommands`] implementation.
pub async fn run_loop_core(
    options: LoopRunOptions,
    backend: Option<&dyn MemoryBackend>,
    anchor_store: &AnchorAnchorStore,
    commands: &mut dyn LoopCommands,
) -> anyhow::Result<LoopRunReport> {
    let mut log = LoopRunLog::create(&options.run_log_dir, &options.run_id)?;
    let started_at = Instant::now();

    if let Some(reason) = wall_cap_message(started_at, options.max_wall, "initial check") {
        return finish_loop_early(
            &mut log,
            &options.run_id,
            "wall-cap",
            0,
            started_at,
            &reason,
        );
    }
    let initial_result = commands
        .verify_goal(
            &options.goal,
            remaining_wall_budget(started_at, options.max_wall),
        )
        .await;
    let (initial_goal_met, _) = match initial_result {
        Ok(result) => result,
        Err(error) => {
            let reason = error.to_string();
            let outcome = if reason.contains("wall-clock budget") {
                "wall-cap"
            } else {
                "error"
            };
            return finish_loop_early(&mut log, &options.run_id, outcome, 0, started_at, &reason);
        }
    };
    if initial_goal_met {
        log.write_summary(
            &options.run_id,
            "success",
            0,
            started_at.elapsed(),
            "goal already held",
        )?;
        return Ok(LoopRunReport {
            turns: 0,
            outcome: "success".to_string(),
            why_stopped: "goal already held".to_string(),
            run_log_path: log.path().to_path_buf(),
            failure_trail: Vec::new(),
        });
    }
    let mut last_check: Option<String> = None;
    let mut corrected_failures: Vec<FailedCheck> = Vec::new();
    // Remediation state: consecutive failure count, accumulated trail, and the directive injected
    // into the next worker so each retry targets the specific failure rather than blindly retrying.
    let mut consecutive_failures: u32 = 0;
    let mut failure_trail: Vec<RemediationAttempt> = Vec::new();
    let mut last_failure_reason: Option<String> = None;
    let mut warned_quota_windows: BTreeSet<(String, String)> = BTreeSet::new();
    for turn in 1..=options.max_turns {
        if let Some(reason) =
            wall_cap_message(started_at, options.max_wall, &format!("turn {turn}"))
        {
            return finish_loop_early(
                &mut log,
                &options.run_id,
                "wall-cap",
                turn.saturating_sub(1),
                started_at,
                &reason,
            );
        }
        if options.stop_file.exists() {
            let reason = format!("loop stopped by sentinel {}", options.stop_file.display());
            return finish_loop_early(
                &mut log,
                &options.run_id,
                "stopped",
                turn.saturating_sub(1),
                started_at,
                &reason,
            );
        }
        // Check the cancellation token so a client interrupt is acted on at every turn boundary.
        if options.cancel.is_cancelled() {
            return finish_loop_early(
                &mut log,
                &options.run_id,
                "cancelled",
                turn.saturating_sub(1),
                started_at,
                "loop cancelled by client",
            );
        }
        let quota_statuses = read_local_quota_with_options(&options.quota.reader);
        let quota_events = quota_threshold_events(
            &quota_statuses,
            options.quota.warn_pct,
            options.quota.high_pct,
        );
        for event in &quota_events {
            let window_key = (event.status.agent.clone(), event.status.window.clone());
            if warned_quota_windows.insert(window_key) {
                if let Some(message) = event.status.threshold_message(event.threshold_pct) {
                    eprintln!("{message}");
                }
                log.write_quota_warning(&options.run_id, turn, event)?;
            }
        }
        let checkpoint_event = quota_events.iter().find(|event| event.high).or_else(|| {
            options
                .quota
                .checkpoint_on_quota
                .then(|| quota_events.first())
                .flatten()
        });
        if let Some(event) = checkpoint_event {
            let checkpoint = write_quota_continuation(
                anchor_store,
                backend,
                &QuotaContinuationContext {
                    run_id: &options.run_id,
                    goal: &options.goal,
                    worker_cmd: options.worker_cmd.as_deref(),
                    turn,
                    run_log_path: log.path(),
                    last_failed_check: last_check.as_deref(),
                    event,
                },
            )
            .await?;
            log.write_quota_checkpoint(&options.run_id, turn, event, &checkpoint.key)?;
            let reason = format!(
                "{}; checkpoint session={} task={}",
                QUOTA_CONTINUATION_NOTE, checkpoint.key.session_id, checkpoint.key.task_id
            );
            return finish_loop_early(
                &mut log,
                &options.run_id,
                "quota-checkpoint",
                turn.saturating_sub(1),
                started_at,
                &reason,
            );
        }
        if let Some(cb) = options.on_progress.as_ref() {
            cb(
                turn as f64,
                Some(options.max_turns as f64),
                Some(format!("turn {turn}/{}: starting", options.max_turns)),
            );
        }
        let recall = match backend {
            Some(backend) => {
                let (text, baseline_tokens, returned_tokens) =
                    loop_recall_inner(backend, &options.goal).await;
                record_savings(
                    "loop.recall",
                    &options.collection,
                    returned_tokens,
                    baseline_tokens,
                    options.track_savings,
                );
                text
            }
            None => String::new(),
        };
        let packet =
            assemble_goal_packet(backend, &options.goal, last_check.as_deref(), &recall).await;
        let action = options.worker_cmd.as_deref().unwrap_or("(poll)");
        if options.poll {
            let sleep_for = remaining_wall_budget(started_at, options.max_wall)
                .map_or(DEFAULT_LOOP_SLEEP, |d| d.min(DEFAULT_LOOP_SLEEP));
            tokio::time::sleep(sleep_for).await;
        } else if let Some(cmd) = &options.worker_cmd {
            let mut env = vec![
                ("ARTESIAN_PACKET".to_string(), packet),
                ("ARTESIAN_RECALL".to_string(), recall.clone()),
                ("ARTESIAN_GOAL".to_string(), options.goal.clone()),
                ("ARTESIAN_RUN_ID".to_string(), options.run_id.clone()),
                ("ARTESIAN_TURN".to_string(), turn.to_string()),
            ];
            // Inject the structured remediation directive when the previous verify failed.
            // The worker can read $ARTESIAN_LAST_FAILURE to target the specific failure rather
            // than blindly retrying the same action.
            if let Some(ref directive) = last_failure_reason {
                env.push((ARTESIAN_LAST_FAILURE_ENV.to_string(), directive.clone()));
            }
            // Drive the worker inside a select so a cancellation token fires promptly even
            // when the worker is a long-running subprocess.  `kill_on_drop(true)` on the
            // underlying `tokio::process::Command` ensures the child is reaped when the
            // future is dropped by the cancel branch.
            let worker_result = tokio::select! {
                result = commands.run_worker(
                    cmd,
                    env,
                    remaining_wall_budget(started_at, options.max_wall),
                ) => result,
                () = options.cancel.cancelled() => {
                    return finish_loop_early(
                        &mut log,
                        &options.run_id,
                        "cancelled",
                        turn.saturating_sub(1),
                        started_at,
                        "cancelled during worker execution",
                    );
                }
            };
            match worker_result {
                Ok(true) | Ok(false) => {}
                Err(error) => {
                    let reason = error.to_string();
                    let outcome = if reason.contains("wall-clock budget") {
                        "wall-cap"
                    } else {
                        "error"
                    };
                    return finish_loop_early(
                        &mut log,
                        &options.run_id,
                        outcome,
                        turn.saturating_sub(1),
                        started_at,
                        &reason,
                    );
                }
            }
        }
        // Emit pre-verify progress so the client gets a continuous "still working" heartbeat
        // even when individual worker turns take a long time.
        if let Some(cb) = options.on_progress.as_ref() {
            cb(
                turn as f64,
                Some(options.max_turns as f64),
                Some(format!("turn {turn}/{}: verifying goal", options.max_turns)),
            );
        }
        let _ = anchor_store
            .set(SessionAnchor::new(
                format!(
                    "loop turn {turn}: {}",
                    options.worker_cmd.as_deref().unwrap_or("(poll)")
                ),
                format!("verify goal: {}", options.goal),
            ))
            .await;
        let verify_result = tokio::select! {
            result = commands.verify_goal(
                &options.goal,
                remaining_wall_budget(started_at, options.max_wall),
            ) => result,
            () = options.cancel.cancelled() => {
                return finish_loop_early(
                    &mut log,
                    &options.run_id,
                    "cancelled",
                    turn.saturating_sub(1),
                    started_at,
                    "cancelled during goal verification",
                );
            }
        };
        let (goal_met, check_output) = match verify_result {
            Ok(result) => result,
            Err(error) => {
                let reason = error.to_string();
                let outcome = if reason.contains("wall-clock budget") {
                    "wall-cap"
                } else {
                    "error"
                };
                return finish_loop_early(
                    &mut log,
                    &options.run_id,
                    outcome,
                    turn.saturating_sub(1),
                    started_at,
                    &reason,
                );
            }
        };
        if let Some(cb) = options.on_progress.as_ref() {
            let msg = if goal_met {
                format!("turn {turn}/{}: goal met", options.max_turns)
            } else {
                format!("turn {turn}/{}: not yet, retrying", options.max_turns)
            };
            cb(turn as f64, Some(options.max_turns as f64), Some(msg));
        }
        log.write_turn(
            &options.run_id,
            turn,
            action,
            goal_met,
            &check_output,
            started_at.elapsed(),
        )?;
        if goal_met {
            // Success: reset remediation state so a later failure in the same session starts fresh.
            consecutive_failures = 0;
            last_failure_reason = None;
            last_check = None;
        } else {
            // Failure: record for auto-invariant extraction, accumulate the remediation trail,
            // and build the directive that will be injected into the next worker invocation.
            consecutive_failures += 1;
            let reason = compact_inline(&check_output, LOOP_FAILURE_TRAIL_REASON_CHARS);
            corrected_failures.push(FailedCheck {
                turn,
                output: check_output.clone(),
            });
            failure_trail.push(RemediationAttempt {
                turn,
                reason: reason.clone(),
                fix_attempt: options.worker_cmd.clone(),
            });
            // Write a "remediation" log entry only for turns where ARTESIAN_LAST_FAILURE was
            // already injected — i.e. the second consecutive failure onward.  The first failure
            // is the initial detection; subsequent turns are the actual remediation attempts.
            if consecutive_failures > 1 {
                log.write_remediation_attempt(
                    &options.run_id,
                    turn,
                    consecutive_failures,
                    &reason,
                    options.worker_cmd.as_deref(),
                )?;
            }
            last_check = Some(format!(
                "turn {turn}: `{}` failed\n{check_output}",
                options.goal
            ));
            last_failure_reason = Some(format!(
                "PREVIOUS ATTEMPT FAILED — fix exactly this:\n{reason}\n"
            ));
        }
        if let Some(backend) = backend {
            loop_commit_turn(
                backend,
                &options.run_id,
                turn,
                &options.goal,
                options.worker_cmd.as_deref(),
                goal_met,
            )
            .await;
        }
        // Escalation check: after recording the failure in memory, stop if the remediation budget
        // is exhausted.  A success always resets the counter, so this only fires on consecutive
        // failures.  The escalation outcome is distinct from "max-turns" and carries the full trail.
        if !goal_met
            && options.max_remediation_attempts > 0
            && consecutive_failures >= options.max_remediation_attempts
        {
            return finish_loop_escalation(
                &mut log,
                &options.run_id,
                turn,
                started_at,
                failure_trail,
            );
        }
        if goal_met {
            if options.learn {
                if let Some(backend) = backend {
                    if let Some(cmd) = &options.worker_cmd {
                        loop_store_skill(backend, &options.goal, cmd, turn).await;
                    }
                    loop_store_spec(backend, &options.goal, options.worker_cmd.as_deref(), turn)
                        .await;
                    // The auto-invariants capture each failure->fix pair as a structured lesson so
                    // future runs avoid repeating the same mistake.
                    loop_store_auto_invariants(
                        backend,
                        &options.goal,
                        options.worker_cmd.as_deref(),
                        &corrected_failures,
                    )
                    .await;
                }
            }
            log.write_summary(
                &options.run_id,
                "success",
                turn,
                started_at.elapsed(),
                "goal held",
            )?;
            return Ok(LoopRunReport {
                turns: turn,
                outcome: "success".to_string(),
                why_stopped: "goal held".to_string(),
                run_log_path: log.path().to_path_buf(),
                failure_trail: Vec::new(),
            });
        }
    }
    let reason = format!(
        "loop reached max-turns ({}) without the goal holding",
        options.max_turns
    );
    finish_loop_early(
        &mut log,
        &options.run_id,
        "max-turns",
        options.max_turns,
        started_at,
        &reason,
    )
}

fn finish_loop_early(
    log: &mut LoopRunLog,
    run_id: &str,
    outcome: &str,
    turns: u32,
    started_at: Instant,
    reason: &str,
) -> anyhow::Result<LoopRunReport> {
    log.write_summary(run_id, outcome, turns, started_at.elapsed(), reason)?;
    Ok(LoopRunReport {
        turns,
        outcome: outcome.to_string(),
        why_stopped: reason.to_string(),
        run_log_path: log.path().to_path_buf(),
        failure_trail: Vec::new(),
    })
}

/// Write the escalation summary and return an `"escalated"` report with the full failure trail.
/// Called when consecutive verify failures exhaust `max_remediation_attempts`.
fn finish_loop_escalation(
    log: &mut LoopRunLog,
    run_id: &str,
    turns: u32,
    started_at: Instant,
    trail: Vec<RemediationAttempt>,
) -> anyhow::Result<LoopRunReport> {
    let compact_trail: Vec<String> = trail
        .iter()
        .map(|a| format!("turn {}: {}", a.turn, a.reason))
        .collect();
    let why_stopped = format!(
        "escalated: {} remediation attempt(s) exhausted; failures: [{}]",
        trail.len(),
        compact_trail.join("; ")
    );
    log.write_escalation_summary(run_id, turns, started_at.elapsed(), &why_stopped, &trail)?;
    Ok(LoopRunReport {
        turns,
        outcome: "escalated".to_string(),
        why_stopped,
        run_log_path: log.path().to_path_buf(),
        failure_trail: trail,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use aquifer::{FilesBackend, SessionStore};
    use artesian_test_support::TempDir;

    use super::*;

    // A deterministic mock: worker always succeeds; verifier passes on the Nth call.
    struct MockLoopCommands {
        // Which call index (0-based) to the verifier returns true.
        pass_on_call: usize,
        verify_calls: Mutex<usize>,
        worker_env: Mutex<Vec<Vec<(String, String)>>>,
    }

    impl MockLoopCommands {
        fn new(pass_on_call: usize) -> Self {
            Self {
                pass_on_call,
                verify_calls: Mutex::new(0),
                worker_env: Mutex::new(Vec::new()),
            }
        }
    }

    impl LoopCommands for MockLoopCommands {
        fn run_worker<'a>(
            &'a mut self,
            _cmd: &'a str,
            env: Vec<(String, String)>,
            _timeout: Option<Duration>,
        ) -> LoopCommandFuture<'a, bool> {
            self.worker_env.lock().unwrap().push(env);
            Box::pin(async move { Ok(true) })
        }

        fn verify_goal<'a>(
            &'a mut self,
            _cmd: &'a str,
            _timeout: Option<Duration>,
        ) -> LoopCommandFuture<'a, (bool, String)> {
            let mut calls = self.verify_calls.lock().unwrap();
            let call_index = *calls;
            *calls += 1;
            let pass = call_index == self.pass_on_call;
            let output = if pass {
                "ok".to_string()
            } else {
                format!("not ready at call {call_index}")
            };
            Box::pin(async move { Ok((pass, output)) })
        }
    }

    fn test_quota_config(tempdir: &TempDir) -> QuotaLoopConfig {
        QuotaLoopConfig {
            reader: crate::quota::QuotaReadOptions {
                codex_home: Some(tempdir.join("missing-codex")),
                claude_roots: vec![tempdir.join("missing-claude")],
            },
            ..QuotaLoopConfig::default()
        }
    }

    #[tokio::test]
    async fn loop_core_succeeds_when_goal_holds_on_first_check() {
        let tempdir = TempDir::new("loop-core-immediate");
        let backend = FilesBackend::new(tempdir.path());
        let anchor = AnchorAnchorStore::new(tempdir.path());
        let mut commands = MockLoopCommands::new(0); // passes on initial check (call 0)
        let run_id = "test-run-immediate".to_string();
        let run_log_dir = tempdir.join("runs");
        let stop_file = tempdir.join("STOP");

        let report = run_loop_core(
            LoopRunOptions {
                goal: "goal-cmd".to_string(),
                worker_cmd: Some("worker-cmd".to_string()),
                max_turns: 5,
                max_wall: None,
                poll: false,
                learn: false,
                run_id: run_id.clone(),
                run_log_dir,
                stop_file,
                collection: String::new(),
                track_savings: false,
                max_remediation_attempts: LOOP_REMEDIATION_ATTEMPTS_DEFAULT,
                cancel: CancellationToken::new(),
                on_progress: None,
                quota: test_quota_config(&tempdir),
            },
            Some(&backend),
            &anchor,
            &mut commands,
        )
        .await
        .expect("loop should succeed");

        assert_eq!(report.turns, 0);
        assert_eq!(report.outcome, "success");
    }

    #[tokio::test]
    async fn loop_core_succeeds_after_one_worker_turn() {
        let tempdir = TempDir::new("loop-core-one-turn");
        let backend = FilesBackend::new(tempdir.path());
        let anchor = AnchorAnchorStore::new(tempdir.path());
        // Call 0 = initial check (fails), call 1 = after turn 1 (passes).
        let mut commands = MockLoopCommands::new(1);
        let run_log_dir = tempdir.join("runs");
        let stop_file = tempdir.join("STOP");

        let report = run_loop_core(
            LoopRunOptions {
                goal: "goal-cmd".to_string(),
                worker_cmd: Some("worker-cmd".to_string()),
                max_turns: 5,
                max_wall: None,
                poll: false,
                learn: false,
                run_id: "test-run-one-turn".to_string(),
                run_log_dir,
                stop_file,
                collection: String::new(),
                track_savings: false,
                max_remediation_attempts: LOOP_REMEDIATION_ATTEMPTS_DEFAULT,
                cancel: CancellationToken::new(),
                on_progress: None,
                quota: test_quota_config(&tempdir),
            },
            Some(&backend),
            &anchor,
            &mut commands,
        )
        .await
        .expect("loop should succeed");

        assert_eq!(report.turns, 1);
        assert_eq!(report.outcome, "success");
        // Worker should have received ARTESIAN_GOAL in env.
        let env_calls = commands.worker_env.lock().unwrap();
        assert!(!env_calls.is_empty(), "worker should have been called once");
        let had_goal = env_calls[0]
            .iter()
            .any(|(k, v)| k == "ARTESIAN_GOAL" && v == "goal-cmd");
        assert!(had_goal, "worker env must contain ARTESIAN_GOAL");
    }

    #[tokio::test]
    async fn loop_core_stops_at_max_turns() {
        let tempdir = TempDir::new("loop-core-max-turns");
        let backend = FilesBackend::new(tempdir.path());
        let anchor = AnchorAnchorStore::new(tempdir.path());
        // Never passes (pass_on_call = 999 > max_turns). Disable escalation so max-turns fires.
        let mut commands = MockLoopCommands::new(999);
        let run_log_dir = tempdir.join("runs");
        let stop_file = tempdir.join("STOP");

        let report = run_loop_core(
            LoopRunOptions {
                goal: "goal-cmd".to_string(),
                worker_cmd: Some("worker-cmd".to_string()),
                max_turns: 3,
                max_wall: None,
                poll: false,
                learn: false,
                run_id: "test-run-max".to_string(),
                run_log_dir,
                stop_file,
                collection: String::new(),
                track_savings: false,
                max_remediation_attempts: 0, // disabled so we reach max-turns
                cancel: CancellationToken::new(),
                on_progress: None,
                quota: test_quota_config(&tempdir),
            },
            Some(&backend),
            &anchor,
            &mut commands,
        )
        .await
        .expect("loop should return a report even on max-turns");

        assert_eq!(report.turns, 3);
        assert_eq!(report.outcome, "max-turns");
    }

    #[tokio::test]
    async fn loop_core_respects_stop_sentinel() {
        let tempdir = TempDir::new("loop-core-stop");
        let backend = FilesBackend::new(tempdir.path());
        let anchor = AnchorAnchorStore::new(tempdir.path());
        let mut commands = MockLoopCommands::new(999);
        let run_log_dir = tempdir.join("runs");
        let stop_file = tempdir.join("STOP");
        // Pre-create the sentinel before the first turn.
        std::fs::write(&stop_file, "").unwrap();

        let report = run_loop_core(
            LoopRunOptions {
                goal: "goal-cmd".to_string(),
                worker_cmd: Some("worker-cmd".to_string()),
                max_turns: 10,
                max_wall: None,
                poll: false,
                learn: false,
                run_id: "test-run-stop".to_string(),
                run_log_dir,
                stop_file,
                collection: String::new(),
                track_savings: false,
                max_remediation_attempts: LOOP_REMEDIATION_ATTEMPTS_DEFAULT,
                cancel: CancellationToken::new(),
                on_progress: None,
                quota: test_quota_config(&tempdir),
            },
            Some(&backend),
            &anchor,
            &mut commands,
        )
        .await
        .expect("loop should return a report on stop sentinel");

        assert_eq!(report.outcome, "stopped");
    }

    #[tokio::test]
    async fn loop_core_logs_quota_warning_at_threshold() {
        let tempdir = TempDir::new("loop-core-quota-warning");
        let codex_home = tempdir.join("codex");
        std::fs::create_dir_all(&codex_home).unwrap();
        std::fs::write(
            codex_home.join("rate_limits.json"),
            include_str!("../tests/fixtures/codex-rate-limits.json"),
        )
        .unwrap();
        let backend = FilesBackend::new(tempdir.path());
        let anchor = AnchorAnchorStore::new(tempdir.path());
        let mut commands = MockLoopCommands::new(1);
        let mut quota = test_quota_config(&tempdir);
        quota.reader.codex_home = Some(codex_home);

        let report = run_loop_core(
            LoopRunOptions {
                goal: "goal-cmd".to_string(),
                worker_cmd: Some("worker-cmd".to_string()),
                max_turns: 3,
                max_wall: None,
                poll: false,
                learn: false,
                run_id: "test-quota-warning".to_string(),
                run_log_dir: tempdir.join("runs"),
                stop_file: tempdir.join("STOP"),
                collection: String::new(),
                track_savings: false,
                max_remediation_attempts: 0,
                cancel: CancellationToken::new(),
                on_progress: None,
                quota,
            },
            Some(&backend),
            &anchor,
            &mut commands,
        )
        .await
        .expect("loop should succeed after warning");

        assert_eq!(report.outcome, "success");
        let log_content = std::fs::read_to_string(&report.run_log_path).unwrap();
        assert!(log_content.contains(r#""type":"quota-warning""#));
        assert!(!log_content.contains(r#""type":"quota-checkpoint""#));
    }

    #[tokio::test]
    async fn loop_core_writes_continuation_anchor_when_quota_is_high() {
        let tempdir = TempDir::new("loop-core-quota-checkpoint");
        let codex_home = tempdir.join("codex");
        std::fs::create_dir_all(&codex_home).unwrap();
        std::fs::write(
            codex_home.join("rate_limits.json"),
            r#"{
                "rateLimits": {
                    "primary": {
                        "usedPercent": 96.0,
                        "windowDurationMins": 300,
                        "resetsAt": 1782864000
                    },
                    "secondary": {
                        "usedPercent": 12.0,
                        "windowDurationMins": 10080,
                        "resetsAt": 1783296000
                    }
                }
            }"#,
        )
        .unwrap();
        let backend = FilesBackend::new(tempdir.path());
        let anchor_store = AnchorAnchorStore::new(tempdir.path());
        let mut commands = MockLoopCommands::new(999);
        let mut quota = test_quota_config(&tempdir);
        quota.reader.codex_home = Some(codex_home);
        let run_id = "test-quota-checkpoint";

        let report = run_loop_core(
            LoopRunOptions {
                goal: "goal-cmd".to_string(),
                worker_cmd: Some("worker-cmd".to_string()),
                max_turns: 3,
                max_wall: None,
                poll: false,
                learn: false,
                run_id: run_id.to_string(),
                run_log_dir: tempdir.join("runs"),
                stop_file: tempdir.join("STOP"),
                collection: String::new(),
                track_savings: false,
                max_remediation_attempts: 0,
                cancel: CancellationToken::new(),
                on_progress: None,
                quota,
            },
            Some(&backend),
            &anchor_store,
            &mut commands,
        )
        .await
        .expect("loop should checkpoint on high quota");

        assert_eq!(report.outcome, "quota-checkpoint");
        assert_eq!(report.turns, 0);
        assert!(
            commands.worker_env.lock().unwrap().is_empty(),
            "worker should not run after quota brake"
        );

        let key = crate::quota::quota_session_key(run_id, "goal-cmd");
        let anchor = anchor_store
            .get_for_session(&key)
            .await
            .expect("anchor read should succeed")
            .expect("quota anchor should exist");
        assert_eq!(anchor.current_task, "Loop goal: goal-cmd");
        assert!(anchor.next_step.contains(QUOTA_CONTINUATION_NOTE));
        assert!(anchor
            .last_decisions
            .iter()
            .any(|decision| decision == QUOTA_CONTINUATION_NOTE));

        let session = SessionStore::new(Arc::new(backend))
            .load(&key)
            .await
            .expect("session load should succeed")
            .expect("session checkpoint should exist");
        let packet = headgate::WorkingContextBundle::resume_packet_from_session(&session)
            .expect("resume packet should render");
        assert_eq!(packet["goal"], "Loop goal: goal-cmd");
        assert!(packet["restored_working_state"]
            .as_str()
            .unwrap()
            .contains(QUOTA_CONTINUATION_NOTE));
    }

    // ── Remediation arc tests ─────────────────────────────────────────────────────────────────

    /// A stub that fails the first K verifier calls then passes — used to test the remediation arc.
    /// Call 0 is the initial check; call N+1 is after turn N.
    struct FailThenPassCommands {
        /// The 0-based verifier call index on which to first return `true`.
        pass_on_call: usize,
        verify_calls: Mutex<usize>,
        /// Env vars passed to each worker invocation (one entry per worker call).
        worker_env: Mutex<Vec<Vec<(String, String)>>>,
    }

    impl FailThenPassCommands {
        fn new(pass_on_call: usize) -> Self {
            Self {
                pass_on_call,
                verify_calls: Mutex::new(0),
                worker_env: Mutex::new(Vec::new()),
            }
        }
    }

    impl LoopCommands for FailThenPassCommands {
        fn run_worker<'a>(
            &'a mut self,
            _cmd: &'a str,
            env: Vec<(String, String)>,
            _timeout: Option<Duration>,
        ) -> LoopCommandFuture<'a, bool> {
            self.worker_env.lock().unwrap().push(env);
            Box::pin(async move { Ok(true) })
        }

        fn verify_goal<'a>(
            &'a mut self,
            _cmd: &'a str,
            _timeout: Option<Duration>,
        ) -> LoopCommandFuture<'a, (bool, String)> {
            let mut calls = self.verify_calls.lock().unwrap();
            let call_index = *calls;
            *calls += 1;
            let pass = call_index >= self.pass_on_call;
            let output = if pass {
                "ok".to_string()
            } else {
                format!("stub failure at call {call_index}")
            };
            Box::pin(async move { Ok((pass, output)) })
        }
    }

    /// When the goal fails once and then passes, the loop remediates successfully:
    /// - the captured failure output is injected into the next worker as ARTESIAN_LAST_FAILURE,
    /// - the outcome is "success" (not "escalated"),
    /// - the first worker call has no ARTESIAN_LAST_FAILURE (no prior failure),
    /// - the second worker call has ARTESIAN_LAST_FAILURE set with the failure directive.
    #[tokio::test]
    async fn loop_core_remediates_fail_then_pass() {
        let tempdir = TempDir::new("loop-core-remediate");
        let backend = FilesBackend::new(tempdir.path());
        let anchor = AnchorAnchorStore::new(tempdir.path());
        // Call 0 = initial check (fails), call 1 = after turn 1 (fails), call 2 = turn 2 (passes).
        let mut commands = FailThenPassCommands::new(2);
        let run_log_dir = tempdir.join("runs");
        let stop_file = tempdir.join("STOP");

        let report = run_loop_core(
            LoopRunOptions {
                goal: "goal-cmd".to_string(),
                worker_cmd: Some("worker-cmd".to_string()),
                max_turns: 5,
                max_wall: None,
                poll: false,
                learn: false,
                run_id: "test-remediate".to_string(),
                run_log_dir,
                stop_file,
                collection: String::new(),
                track_savings: false,
                max_remediation_attempts: 3,
                cancel: CancellationToken::new(),
                on_progress: None,
                quota: test_quota_config(&tempdir),
            },
            Some(&backend),
            &anchor,
            &mut commands,
        )
        .await
        .expect("loop should succeed after remediation");

        assert_eq!(report.outcome, "success");
        assert_eq!(report.turns, 2);
        assert!(report.failure_trail.is_empty(), "success clears the trail");

        let env_calls = commands.worker_env.lock().unwrap();
        assert_eq!(env_calls.len(), 2, "worker should have run twice");

        // Turn 1: no prior failure — ARTESIAN_LAST_FAILURE must NOT be present.
        let turn1_has_last_failure = env_calls[0]
            .iter()
            .any(|(k, _)| k == ARTESIAN_LAST_FAILURE_ENV);
        assert!(
            !turn1_has_last_failure,
            "turn 1 worker env must not have ARTESIAN_LAST_FAILURE (first attempt)"
        );

        // Turn 2: previous verify failed — ARTESIAN_LAST_FAILURE must carry the directive.
        let turn2_directive = env_calls[1]
            .iter()
            .find(|(k, _)| k == ARTESIAN_LAST_FAILURE_ENV)
            .map(|(_, v)| v.as_str());
        assert!(
            turn2_directive.is_some(),
            "turn 2 worker env must have ARTESIAN_LAST_FAILURE"
        );
        assert!(
            turn2_directive.unwrap().contains("PREVIOUS ATTEMPT FAILED"),
            "ARTESIAN_LAST_FAILURE must contain the structured directive"
        );
        assert!(
            turn2_directive.unwrap().contains("stub failure at call 1"),
            "ARTESIAN_LAST_FAILURE must carry the captured failure output"
        );
    }

    /// When a stub never passes and the remediation budget is exhausted, the loop escalates with:
    /// - outcome "escalated" (not "max-turns"),
    /// - a failure trail with one entry per consecutive failure,
    /// - why_stopped containing "escalated: N remediation attempt(s) exhausted",
    /// - the escalation fires at the budget boundary, not at max-turns.
    #[tokio::test]
    async fn loop_core_escalates_within_budget_not_max_turns() {
        let tempdir = TempDir::new("loop-core-escalate");
        let backend = FilesBackend::new(tempdir.path());
        let anchor = AnchorAnchorStore::new(tempdir.path());
        let mut commands = FailThenPassCommands::new(999); // never passes
        let run_log_dir = tempdir.join("runs");
        let stop_file = tempdir.join("STOP");

        // max_turns=10 but max_remediation_attempts=2: escalation fires at turn 2, not turn 10.
        let report = run_loop_core(
            LoopRunOptions {
                goal: "goal-cmd".to_string(),
                worker_cmd: Some("worker-cmd".to_string()),
                max_turns: 10,
                max_wall: None,
                poll: false,
                learn: false,
                run_id: "test-escalate".to_string(),
                run_log_dir,
                stop_file,
                collection: String::new(),
                track_savings: false,
                max_remediation_attempts: 2,
                cancel: CancellationToken::new(),
                on_progress: None,
                quota: test_quota_config(&tempdir),
            },
            Some(&backend),
            &anchor,
            &mut commands,
        )
        .await
        .expect("loop should return an escalation report");

        assert_eq!(report.outcome, "escalated", "outcome must be 'escalated'");
        assert_eq!(
            report.turns, 2,
            "escalation fires at budget (turn 2), not max-turns (10)"
        );
        assert_eq!(
            report.failure_trail.len(),
            2,
            "trail must have one entry per consecutive failure"
        );
        assert!(
            report.why_stopped.contains("escalated: 2"),
            "why_stopped must name the budget: {}",
            report.why_stopped
        );
        assert!(
            report.why_stopped.contains("remediation attempt"),
            "why_stopped must mention remediation: {}",
            report.why_stopped
        );
        // Trail entries must carry the turn and non-empty reason.
        assert_eq!(report.failure_trail[0].turn, 1);
        assert!(!report.failure_trail[0].reason.is_empty());
        assert_eq!(report.failure_trail[1].turn, 2);
        assert!(!report.failure_trail[1].reason.is_empty());
    }

    /// The run-log records a "remediation" entry for each failing turn and an "escalated" summary
    /// with the failure trail when the budget is exhausted.
    #[tokio::test]
    async fn loop_core_run_log_records_remediation_entries() {
        let tempdir = TempDir::new("loop-core-remediation-log");
        let backend = FilesBackend::new(tempdir.path());
        let anchor = AnchorAnchorStore::new(tempdir.path());
        let mut commands = FailThenPassCommands::new(999); // never passes
        let run_log_dir = tempdir.join("runs");
        let stop_file = tempdir.join("STOP");

        let report = run_loop_core(
            LoopRunOptions {
                goal: "goal-cmd".to_string(),
                worker_cmd: Some("worker-cmd".to_string()),
                max_turns: 5,
                max_wall: None,
                poll: false,
                learn: false,
                run_id: "test-run-log-remediation".to_string(),
                run_log_dir: run_log_dir.clone(),
                stop_file,
                collection: String::new(),
                track_savings: false,
                max_remediation_attempts: 2,
                cancel: CancellationToken::new(),
                on_progress: None,
                quota: test_quota_config(&tempdir),
            },
            Some(&backend),
            &anchor,
            &mut commands,
        )
        .await
        .expect("loop should return an escalation report");

        assert_eq!(report.outcome, "escalated");

        let log_content = std::fs::read_to_string(&report.run_log_path)
            .expect("run log must exist after the loop");

        let has_remediation_entry = log_content
            .lines()
            .any(|line| line.contains(r#""type":"remediation""#));
        assert!(
            has_remediation_entry,
            "run log must contain at least one remediation entry"
        );

        let escalated_summary = log_content
            .lines()
            .find(|line| line.contains(r#""outcome":"escalated""#));
        assert!(
            escalated_summary.is_some(),
            "run log must contain an escalated summary entry"
        );
        let summary_line = escalated_summary.unwrap();
        assert!(
            summary_line.contains(r#""failure_trail""#),
            "escalated summary must include failure_trail field"
        );
    }

    /// Disabling escalation (max_remediation_attempts = 0) lets the loop run all the way to
    /// max-turns even when every verify fails — backward-compatible with the original behaviour.
    #[tokio::test]
    async fn loop_core_disabled_escalation_runs_to_max_turns() {
        let tempdir = TempDir::new("loop-core-no-escalate");
        let backend = FilesBackend::new(tempdir.path());
        let anchor = AnchorAnchorStore::new(tempdir.path());
        let mut commands = FailThenPassCommands::new(999); // never passes
        let run_log_dir = tempdir.join("runs");
        let stop_file = tempdir.join("STOP");

        let report = run_loop_core(
            LoopRunOptions {
                goal: "goal-cmd".to_string(),
                worker_cmd: Some("worker-cmd".to_string()),
                max_turns: 4,
                max_wall: None,
                poll: false,
                learn: false,
                run_id: "test-no-escalate".to_string(),
                run_log_dir,
                stop_file,
                collection: String::new(),
                track_savings: false,
                max_remediation_attempts: 0, // escalation disabled
                cancel: CancellationToken::new(),
                on_progress: None,
                quota: test_quota_config(&tempdir),
            },
            Some(&backend),
            &anchor,
            &mut commands,
        )
        .await
        .expect("loop should run to max-turns when escalation is disabled");

        assert_eq!(report.outcome, "max-turns");
        assert_eq!(report.turns, 4);
        assert!(report.failure_trail.is_empty());
    }

    // ── Cancellation and progress tests ───────────────────────────────────────────────────────

    /// A pre-cancelled token must exit with outcome "cancelled" before the first turn runs.
    #[tokio::test]
    async fn loop_core_cancelled_before_first_turn() {
        let tempdir = TempDir::new("loop-core-cancel-before");
        let backend = FilesBackend::new(tempdir.path());
        let anchor = AnchorAnchorStore::new(tempdir.path());
        let mut commands = MockLoopCommands::new(999); // never passes

        let cancel = CancellationToken::new();
        cancel.cancel(); // fire before the call

        let report = run_loop_core(
            LoopRunOptions {
                goal: "goal-cmd".to_string(),
                worker_cmd: Some("worker-cmd".to_string()),
                max_turns: 10,
                max_wall: None,
                poll: false,
                learn: false,
                run_id: "test-cancel-before".to_string(),
                run_log_dir: tempdir.join("runs"),
                stop_file: tempdir.join("STOP"),
                collection: String::new(),
                track_savings: false,
                max_remediation_attempts: 0,
                cancel,
                on_progress: None,
                quota: test_quota_config(&tempdir),
            },
            Some(&backend),
            &anchor,
            &mut commands,
        )
        .await
        .expect("run_loop_core should not error on a pre-cancelled token");

        assert_eq!(
            report.outcome, "cancelled",
            "pre-cancelled token must yield outcome 'cancelled'"
        );
        assert_eq!(
            report.turns, 0,
            "no turns should run when already cancelled"
        );
    }

    /// A cancellation token fired mid-run exits promptly from inside the worker select! branch.
    #[tokio::test]
    async fn loop_core_cancels_during_worker() {
        use std::sync::atomic::{AtomicU32, Ordering};

        // Worker that sleeps until it is dropped (kill_on_drop equivalent in tests).
        struct SlowWorkerCommands {
            worker_calls: Arc<AtomicU32>,
        }
        impl LoopCommands for SlowWorkerCommands {
            fn run_worker<'a>(
                &'a mut self,
                _cmd: &'a str,
                _env: Vec<(String, String)>,
                _timeout: Option<Duration>,
            ) -> LoopCommandFuture<'a, bool> {
                self.worker_calls.fetch_add(1, Ordering::Relaxed);
                Box::pin(async move {
                    tokio::time::sleep(Duration::from_secs(60)).await;
                    Ok(true)
                })
            }
            fn verify_goal<'a>(
                &'a mut self,
                _cmd: &'a str,
                _timeout: Option<Duration>,
            ) -> LoopCommandFuture<'a, (bool, String)> {
                Box::pin(async move { Ok((false, "not done".to_string())) })
            }
        }

        let tempdir = TempDir::new("loop-core-cancel-during");
        let backend = FilesBackend::new(tempdir.path());
        let anchor = AnchorAnchorStore::new(tempdir.path());
        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();
        let worker_calls = Arc::new(AtomicU32::new(0));
        let mut commands = SlowWorkerCommands {
            worker_calls: Arc::clone(&worker_calls),
        };

        // Fire cancellation after 50 ms — well before the 60-second worker would finish.
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            cancel_clone.cancel();
        });

        let started = std::time::Instant::now();
        let report = run_loop_core(
            LoopRunOptions {
                goal: "goal-cmd".to_string(),
                worker_cmd: Some("worker-cmd".to_string()),
                max_turns: 10,
                max_wall: None,
                poll: false,
                learn: false,
                run_id: "test-cancel-during".to_string(),
                run_log_dir: tempdir.join("runs"),
                stop_file: tempdir.join("STOP"),
                collection: String::new(),
                track_savings: false,
                max_remediation_attempts: 0,
                cancel,
                on_progress: None,
                quota: test_quota_config(&tempdir),
            },
            Some(&backend),
            &anchor,
            &mut commands,
        )
        .await
        .expect("run_loop_core should not error on cancellation during worker");

        assert_eq!(report.outcome, "cancelled");
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "cancellation should exit within 5 s, not wait for the 60 s worker"
        );
        // Worker must have been entered (turn 1 started) before the token fired.
        assert_eq!(
            worker_calls.load(Ordering::Relaxed),
            1,
            "worker should have been entered once"
        );
    }

    /// `on_progress` is called at least 3 times when the loop runs multiple turns.
    #[tokio::test]
    async fn loop_core_emits_progress_events() {
        use std::sync::atomic::{AtomicU32, Ordering};

        let tempdir = TempDir::new("loop-core-progress");
        let backend = FilesBackend::new(tempdir.path());
        let anchor = AnchorAnchorStore::new(tempdir.path());
        // Passes on the 4th verifier call (0 = initial, 1/2/3 = turns 1-3).
        let mut commands = MockLoopCommands::new(3);

        let event_count = Arc::new(AtomicU32::new(0));
        let ec = Arc::clone(&event_count);
        let on_progress: Option<LoopProgressCallback> = Some(Arc::new(
            move |_progress: f64, _total: Option<f64>, _msg: Option<String>| {
                ec.fetch_add(1, Ordering::Relaxed);
            },
        ));

        let report = run_loop_core(
            LoopRunOptions {
                goal: "goal-cmd".to_string(),
                worker_cmd: Some("worker-cmd".to_string()),
                max_turns: 5,
                max_wall: None,
                poll: false,
                learn: false,
                run_id: "test-progress".to_string(),
                run_log_dir: tempdir.join("runs"),
                stop_file: tempdir.join("STOP"),
                collection: String::new(),
                track_savings: false,
                max_remediation_attempts: 0,
                cancel: CancellationToken::new(),
                on_progress,
                quota: test_quota_config(&tempdir),
            },
            Some(&backend),
            &anchor,
            &mut commands,
        )
        .await
        .expect("loop should succeed");

        assert_eq!(report.outcome, "success");
        let count = event_count.load(Ordering::Relaxed);
        // 3 turns × 3 events each (turn-start, pre-verify, post-verify) = 9 events minimum.
        assert!(
            count >= 3,
            "expected at least 3 progress events (got {count}); one per turn minimum"
        );
    }
}
