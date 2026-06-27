// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::BTreeMap,
    env,
    ffi::OsString,
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::Stdio,
    str::FromStr,
    sync::Arc,
    time::Duration,
};

use anyhow::{bail, Context, Result};
use aquifer::{
    append_eviction_log, consolidation_pass, default_migration_collection, entity_timeline, evict,
    export_okf_bundle, insert_skill_procedure_metadata, recover_after_compaction,
    verify_okf_bundle, AnchorAnchorStore, CollectionCompat, ConsolidationOptions, DecayConfig,
    EvictionPolicy, MemoryBackend, MemoryQuery, MemoryRecord, MemoryScope, MemoryState, MemoryTier,
    MigrationPlan, ProcedureStep, SearchHit, SessionAnchor, SessionKey, SessionListFilter,
    SessionStore, StoreMemory, VectorMemoryConfig,
};
use artesian_core::{
    Agent, AgentBinding, ArtesianConfig, MemoryBackendKind, MemoryConfig, Mode, Role, SpawnRequest,
};
use artesian_mcp::{
    build_session_bundle_for_cli, checkpoint_anchor_for_cli, qualify_memory_candidate,
    session_scoped_hits_for_cli, QualifyResponse, SessionCheckpointRequest,
};
use artesian_process_agent::{
    fallback_agent_catalog, refresh_agent_catalog, ProcessAgent, ProcessAgentConfig,
    ProcessSupervisor,
};
use clap::{Parser, Subcommand, ValueEnum};
use flume::loop_core::{
    assemble_goal_packet, loop_recall, loop_run_id, loop_run_log_dir, loop_stop_file,
    run_loop_core, stable_content_hash, LoopCommandFuture, LoopCommands, LoopRunOptions,
    LOOP_SKILL_TAG,
};
use flume::quota::{read_local_quota, QuotaLoopConfig, QuotaStatusKind};
use flume::{
    load_role_definitions, role_summaries, TeamCreate, TeamGcOptions, TeamMessage, TeamMessageKind,
    TeamRuntime, TeamRuntimeConfig, TeamSpawn, TeamTaskAdd, TeamTaskClaim, TeamTaskComplete,
    TeamWorkerEvent,
};
use headgate::{
    count_tokens, load_savings_rollup, record_savings, Headgate, HeadgateConfig, LifecycleEntry,
    MemoryRecallStore, RecallStore, SnapshotEntry, WorkingContextBundle, WorkingContextSnapshot,
};
use headrace::{
    ClaimRequest, CommandVerifier, FilesTaskStore, NewTask, TaskKind, TaskStore, VectorTaskStore,
    Verifier, VerifierGate,
};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::process::Command as TokioCommand;
use toml_edit::{value, Array, DocumentMut, Item, Table};

const DEFAULT_CONFIG: &str = "artesian.toml";
const MCP_SERVER_NAME: &str = "artesian-memory";
const MCP_TOOL_HINT: &str =
    "ALWAYS search the project memory before non-trivial work; store durable, reusable learnings.";
const CLAUDE_SESSION_START_COMMAND: &str =
    "artesian hooks claude session-start --config artesian.toml --root .artesian";
const CLAUDE_PRE_COMPACT_COMMAND: &str =
    "artesian hooks claude pre-compact --config artesian.toml --root .artesian";
const CLAUDE_ARTESIAN_LOOP_SKILL: &str = r#"---
name: artesian-loop
description: Recall committed Artesian memory, act, verify, and checkpoint durable context with artesian-memory MCP tools.
---

# Artesian Loop

Use this loop for non-trivial implementation work in an Artesian-enabled project.

1. Recall committed context before acting. Prefer `memory.session.resume` when a user/session/task key is available; otherwise query with `memory.find` for the current goal and relevant constraints.
2. Act in small, reversible steps. Keep durable project knowledge in memory, not transient chat.
3. Verify the change with the repository's normal checks before claiming completion.
4. Commit useful learnings with `memory.store`, scoped and tagged so future agents can retrieve them.
5. Before handoff or compaction, call `memory.session.checkpoint` with the current user/session/task key, current task, next step, decisions, and last failed check when one exists.

Use the `artesian-memory` MCP server and its `memory.find`, `memory.store`, `memory.session.resume`, and `memory.session.checkpoint` tools. Do not use stale `qdrant-find` or `qdrant-store` tool names.
"#;

mod artesiand;
mod import;
mod runtime;
mod update;
use import::{import_directory, ImportOptions};
use runtime::{
    build_orchestrator, load_config, open_memory_backend, open_memory_backend_with_relations,
    process_supervisor_from_config, shutdown_signal,
};

#[derive(Debug, Parser)]
#[command(
    name = "artesian",
    about = "Multi-agent context orchestration",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

struct InitOptions {
    memory_root: PathBuf,
    backend: BackendArg,
    collection: String,
    project: Option<String>,
    qdrant_url: Option<String>,
    qdrant_rest_url: Option<String>,
    qdrant_api_key_env: String,
    qdrant_api_key_file: Option<String>,
    register_mcp: bool,
}

#[derive(Debug, Subcommand)]
enum Command {
    Init {
        #[arg(long)]
        memory_root: Option<PathBuf>,
        #[arg(long)]
        project: Option<String>,
        #[arg(long, value_enum, default_value_t = BackendArg::Files)]
        backend: BackendArg,
        #[arg(long)]
        collection: Option<String>,
        #[arg(long, env = "QDRANT_URL")]
        qdrant_url: Option<String>,
        #[arg(long, env = "QDRANT_REST_URL")]
        qdrant_rest_url: Option<String>,
        #[arg(long, default_value = "QDRANT_API_KEY")]
        qdrant_api_key_env: String,
        #[arg(long)]
        qdrant_api_key_file: Option<String>,
        #[arg(long)]
        non_interactive: bool,
        #[arg(long, default_value_t = true)]
        register_mcp: bool,
    },
    Spawn {
        role: String,
        agent: String,
        #[arg(long = "arg")]
        args: Vec<String>,
        #[arg(long)]
        model: Option<String>,
        #[arg(long, default_value_t = 30)]
        timeout_seconds: u64,
    },
    Agents {
        #[command(subcommand)]
        command: AgentsCommand,
    },
    Hooks {
        #[command(subcommand)]
        command: HooksCommand,
    },
    Run {
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long)]
        root: Option<PathBuf>,
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        once: bool,
    },
    Orchestrate {
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long)]
        root: Option<PathBuf>,
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        once: bool,
    },
    Memory {
        #[command(subcommand)]
        command: MemoryCommand,
    },
    /// Run the ACC qualify-gate on one candidate without storing it.
    Qualify {
        candidate: String,
        /// Optional current goal/task used for relevance scoring and committed-state recall.
        #[arg(long)]
        goal: Option<String>,
        /// Emit the audited gate decision as JSON.
        #[arg(long)]
        json: bool,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
    },
    Skill {
        #[command(subcommand)]
        command: SkillCommand,
    },
    /// Print the committed resume packet for a cross-agent session.
    Handoff {
        session_id: String,
        #[arg(long = "user")]
        user_id: Option<String>,
        #[arg(long = "task")]
        task_id: Option<String>,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
    },
    Session {
        #[command(subcommand)]
        command: SessionCommand,
    },
    Task {
        #[command(subcommand)]
        command: TaskCommand,
    },
    Team {
        #[command(subcommand)]
        command: TeamCommand,
    },
    Backfill {
        directory: PathBuf,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
        #[arg(long)]
        user_id: Option<String>,
        /// Disable deterministic entity-relation extraction during import.
        /// By default, import extracts `mentions` relations from each chunk's content and tags
        /// (no LLM required), so `neighbors` and `by_entity` return links immediately.
        #[arg(long)]
        no_link: bool,
        /// After import, run the LLM consolidation pass (same as `artesian consolidate`).
        /// Requires an LLM configured under `[acc.compressor]` or `[acc.judge]` in artesian.toml.
        /// If no LLM is configured, prints a note and continues without failing.
        #[arg(long)]
        consolidate: bool,
    },
    Onboard {
        project: String,
        directory: PathBuf,
        #[arg(long, value_enum, default_value_t = BackendArg::Qdrant)]
        backend: BackendArg,
        #[arg(long)]
        memory_root: Option<PathBuf>,
        #[arg(long)]
        collection: Option<String>,
        #[arg(long, env = "QDRANT_URL")]
        qdrant_url: Option<String>,
        #[arg(long, env = "QDRANT_REST_URL")]
        qdrant_rest_url: Option<String>,
        #[arg(long, default_value = "QDRANT_API_KEY")]
        qdrant_api_key_env: String,
        #[arg(long)]
        qdrant_api_key_file: Option<String>,
        #[arg(long)]
        user_id: Option<String>,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        /// Disable deterministic entity-relation extraction during import.
        /// By default, import extracts `mentions` relations from each chunk's content and tags
        /// (no LLM required), so `neighbors` and `by_entity` return links immediately.
        #[arg(long)]
        no_link: bool,
        /// After import, run the LLM consolidation pass (same as `artesian consolidate`).
        /// Requires an LLM configured under `[acc.compressor]` or `[acc.judge]` in artesian.toml.
        /// If no LLM is configured, prints a note and continues without failing.
        #[arg(long)]
        consolidate: bool,
    },
    Consolidate {
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
        #[arg(long)]
        allow_llm: bool,
    },
    /// Bundle-to-bundle OCF memory consolidation: read committed records, score by access signals,
    /// write a new OCF bundle with every admit/reject/merge decision logged in qualify.jsonl.
    /// Source collection is NEVER mutated. Schedule with cron or trigger at compaction boundaries.
    ///
    /// Example: artesian dream --out .artesian/dreams/2026-06-23 --diary
    Dream {
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
        /// Collection to read records from (defaults to the configured collection).
        #[arg(long)]
        collection: Option<String>,
        /// Output directory to write the OCF bundle into (created if absent).
        #[arg(long)]
        out: PathBuf,
        /// Also write a human-readable DREAMS.md narrative alongside the OCF files.
        #[arg(long, default_value_t = false)]
        diary: bool,
        /// Admission score threshold [0.0–1.0]. Records scoring below this are rejected.
        #[arg(long, default_value_t = 0.3)]
        admit_threshold: f32,
        /// Jaccard similarity threshold for the dedup consolidation grouping pass [0.0–1.0].
        #[arg(long, default_value_t = 0.6)]
        similarity_threshold: f32,
    },
    Migrate {
        #[command(subcommand)]
        command: MigrateCommand,
    },
    Snapshot {
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
        #[arg(long)]
        collection: Option<String>,
    },
    Okf {
        #[command(subcommand)]
        command: OkfCommand,
    },
    /// Loop memory kit: initialize and export the anchor-set bundle (vision / prompt / memory /
    /// skills), portable across Codex and Claude Code.
    Kit {
        #[command(subcommand)]
        command: KitCommand,
    },
    /// Print tokens-per-iteration saved vs full-context replay, and verify compaction survival.
    Perf {
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
        #[arg(long)]
        budget_tokens: Option<usize>,
    },
    /// Show cumulative token-savings from targeted recall: how many tokens Artesian saved
    /// vs loading the full source records.
    Tokens {
        /// Emit the raw JSON rollup (machine-readable / badge use).
        #[arg(long)]
        json: bool,
        /// Only count recalls at or after this UTC timestamp (ISO 8601, e.g.
        /// `2026-01-01T00:00:00Z`).
        #[arg(long)]
        since: Option<String>,
        /// Show per-operation breakdown alongside the total.
        #[arg(long)]
        by_op: bool,
    },
    /// Print token-free local coding-agent quota status.
    Quota {
        /// Emit machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Run an autonomous memory-first loop: repeat a worker action until a goal command succeeds
    /// (exit 0), writing a resume anchor each turn. The worker can be any shell command — a script
    /// or an agent CLI such as `codex exec '...'`.
    Loop {
        /// Verifier command; exit code 0 means the goal holds (the stop condition).
        #[arg(long)]
        goal: String,
        /// Per-iteration worker action (a shell command). Omit with `--poll` to only re-check.
        #[arg(long)]
        worker_cmd: Option<String>,
        /// Maximum iterations before giving up.
        #[arg(long, default_value_t = 10)]
        max_turns: u32,
        /// Maximum wall-clock seconds before aborting the loop.
        #[arg(long)]
        max_wall_secs: Option<u64>,
        /// Poll mode: do not run a worker, only re-check the goal each turn.
        #[arg(long)]
        poll: bool,
        /// Disable durable skill/spec/invariant learning for this loop run.
        #[arg(long)]
        no_learn: bool,
        /// Maximum consecutive verify failures before escalating with a failure trail.
        /// Each failing turn injects $ARTESIAN_LAST_FAILURE into the next worker so retries
        /// target the specific failure rather than blindly retrying. Set to 0 to disable
        /// escalation (the loop will run to --max-turns instead).
        #[arg(long, default_value_t = flume::loop_core::LOOP_REMEDIATION_ATTEMPTS_DEFAULT)]
        max_remediation_attempts: u32,
        /// Warn when a local coding-agent quota window reaches this used percentage.
        #[arg(long, default_value_t = flume::quota::DEFAULT_QUOTA_WARN_PCT)]
        quota_warn_pct: f64,
        /// Write a continuation checkpoint as soon as any quota warning threshold is crossed.
        #[arg(long)]
        checkpoint_on_quota: bool,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
        /// Project config; its memory backend is used for per-turn recall/commit. Falls back to a
        /// local files backend under `root` when the file is absent.
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
    },
    /// Replicate a Qdrant collection between two endpoints (e.g. a LAN instance and a local Docker
    /// instance) — scroll + upsert, merging by point id. Pass your own URLs/keys; the endpoints
    /// live in your local config, never in the repo.
    ///
    /// Default mode is incremental: only points missing from the target are sent. Use `--full` to
    /// force a complete scroll-and-upsert of all source points. Use `--prune` (incremental only)
    /// to also delete from the target any points whose IDs are no longer in the source.
    Replicate {
        /// Source Qdrant URL.
        #[arg(long)]
        from_url: String,
        /// Target Qdrant URL.
        #[arg(long)]
        to_url: String,
        #[arg(long)]
        from_key: Option<String>,
        #[arg(long)]
        to_key: Option<String>,
        /// Collection name to replicate (read from the source).
        #[arg(long)]
        collection: String,
        /// Target collection name (defaults to --collection).
        #[arg(long)]
        to_collection: Option<String>,
        /// Only check that both endpoints are reachable; do not copy.
        #[arg(long)]
        status: bool,
        /// Force a full scroll-and-upsert of all source points (the old behaviour).
        #[arg(long)]
        full: bool,
        /// Incremental mode (default): only send points missing from the target.
        /// Passing `--incremental` is a no-op if `--full` is not also set (incremental is the default).
        #[arg(long)]
        incremental: bool,
        /// Delete from the target any points whose IDs are no longer in the source (incremental only).
        #[arg(long)]
        prune: bool,
        #[arg(long, default_value_t = 256)]
        batch: u32,
    },
    /// Health check: verify the binary, config, backend reachability, collection compatibility,
    /// and MCP registrations — and print the exact fix for anything that drifted (e.g. after an
    /// upgrade). Exits non-zero if a critical problem is found.
    Doctor {
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
    },
    /// Update Artesian via its package manager, then report installed surfaces and stale MCP
    /// servers. A convenience wrapper — config, MCP registrations, and stored memory are untouched.
    Update {
        /// Best-effort restart of stale artesian-mcp servers after the update.
        #[arg(long)]
        restart_stale: bool,
    },
}

#[derive(Debug, Subcommand)]
enum MigrateCommand {
    /// Rebuild a Qdrant collection from an OKF markdown bundle (atomic alias swap).
    OkfBundle {
        okf_root: PathBuf,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long)]
        new_collection: Option<String>,
        #[arg(long, default_value_t = 30)]
        retention_days: u32,
    },
    /// Re-chunk oversized whole-file records in a SQLite-vec collection.
    ///
    /// Scans all records in the configured collection. Records whose content exceeds
    /// the chunk size limit and that were stored before chunking was introduced (no
    /// `parent_node` metadata) are re-stored as bounded chunks and the originals deleted.
    Rechunk {
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
    },
}

#[derive(Debug, Subcommand)]
enum OkfCommand {
    Verify { root: PathBuf },
    Export { source: PathBuf, target: PathBuf },
}

#[derive(Debug, Subcommand)]
enum AgentsCommand {
    Refresh {
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long)]
        cache: Option<PathBuf>,
    },
}

#[derive(Debug, Subcommand)]
enum HooksCommand {
    /// Install native agent hooks and skills.
    Install {
        /// Install Claude Code hooks. This is the default target today.
        #[arg(long)]
        claude: bool,
        #[arg(long, value_enum, default_value_t = HookScope::Project)]
        scope: HookScope,
    },
    #[command(hide = true)]
    Claude {
        #[command(subcommand)]
        command: ClaudeHookCommand,
    },
}

#[derive(Debug, Subcommand)]
enum ClaudeHookCommand {
    SessionStart {
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
    },
    PreCompact {
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum HookScope {
    User,
    Project,
}

#[derive(Debug, Subcommand)]
enum TeamCommand {
    Create {
        name: String,
        #[arg(long)]
        id: Option<String>,
        #[arg(long)]
        max_teammates: Option<usize>,
        #[arg(long)]
        plan_approval_required: bool,
        #[arg(long = "plan-approval-role")]
        plan_approval_roles: Vec<String>,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
    },
    Spawn {
        team_id: String,
        definition: String,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
    },
    Task {
        #[command(subcommand)]
        command: TeamTaskCommand,
    },
    Message {
        team_id: String,
        #[arg(long)]
        from: String,
        #[arg(long)]
        to: Option<String>,
        #[arg(long, value_enum, default_value_t = TeamMessageKindArg::Ask)]
        kind: TeamMessageKindArg,
        content: String,
        #[arg(long)]
        task_id: Option<String>,
        #[arg(long)]
        approved: Option<bool>,
        #[arg(long)]
        execute: bool,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
    },
    Status {
        team_id: String,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
    },
    Cleanup {
        team_id: String,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
    },
    /// Garbage-collect orphaned, expired, or hung teammate process groups across the registry
    /// (not scoped to one team). Reclaims spawns whose owner exited, whose age exceeds --ttl-secs,
    /// or whose last heartbeat is older than --heartbeat-timeout-secs.
    Gc {
        #[arg(long)]
        ttl_secs: Option<u64>,
        #[arg(long)]
        heartbeat_timeout_secs: Option<u64>,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
    },
    /// Show live presence: which lane/agent is active on what task, current load vs. global cap.
    Presence {
        team_id: String,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
    },
}

#[derive(Debug, Subcommand)]
enum TeamTaskCommand {
    Add {
        team_id: String,
        title: String,
        #[arg(long, default_value = "")]
        description: String,
        #[arg(long)]
        definition: Option<String>,
        #[arg(long = "blocker")]
        blockers: Vec<String>,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
    },
    Claim {
        team_id: String,
        #[arg(long)]
        task_id: Option<String>,
        #[arg(long)]
        teammate: String,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
    },
    Complete {
        team_id: String,
        task_id: String,
        #[arg(long)]
        reviewer: String,
        #[arg(long)]
        approved: bool,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum TeamMessageKindArg {
    Ask,
    Result,
    Review,
    Done,
}

impl From<TeamMessageKindArg> for TeamMessageKind {
    fn from(value: TeamMessageKindArg) -> Self {
        match value {
            TeamMessageKindArg::Ask => Self::Ask,
            TeamMessageKindArg::Result => Self::Result,
            TeamMessageKindArg::Review => Self::Review,
            TeamMessageKindArg::Done => Self::Done,
        }
    }
}

#[derive(Debug, Subcommand)]
enum SessionCommand {
    /// List resumable cross-agent sessions.
    List {
        #[arg(long = "user")]
        user_id: Option<String>,
        #[arg(long = "session")]
        session_id: Option<String>,
        #[arg(long = "task")]
        task_id: Option<String>,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
    },
    /// Write a keyed (user, session, task) OCF checkpoint — symmetric with the MCP
    /// memory.session.checkpoint tool so any agent can checkpoint from the CLI.
    Checkpoint {
        #[arg(long = "user")]
        user_id: Option<String>,
        #[arg(long = "session")]
        session_id: Option<String>,
        #[arg(long = "task")]
        task_id: Option<String>,
        #[arg(long)]
        agent_id: String,
        #[arg(long)]
        current_task: Option<String>,
        #[arg(long)]
        next_step: Option<String>,
        #[arg(long = "decision")]
        last_decisions: Vec<String>,
        #[arg(long)]
        plan_pointer: Option<String>,
        #[arg(long)]
        last_failed_check: Option<String>,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
    },
    /// Resume a session by fuzzy task query (case-insensitive substring, most-recent tiebreak).
    /// Prints the full resume packet so the operator does not need to know the session_id.
    Resume {
        /// Substring to match against session task ids (e.g. "DPT-4477").
        task_query: String,
        #[arg(long = "user")]
        user_id: Option<String>,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
    },
}

#[derive(Debug, Subcommand)]
enum TaskCommand {
    Add {
        title: String,
        #[arg(long)]
        id: Option<String>,
        #[arg(long, default_value = "")]
        description: String,
        #[arg(long = "blocker")]
        blockers: Vec<String>,
        #[arg(long, value_enum, default_value_t = TaskKindArg::Primitive)]
        kind: TaskKindArg,
        #[arg(long, default_value = "worker")]
        role: String,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
    },
    List {
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
    },
    Claim {
        id: Option<String>,
        #[arg(long, default_value = "worker")]
        claimant: String,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
    },
    Done {
        id: String,
        #[arg(long = "verify-command")]
        verify_commands: Vec<String>,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
    },
    Find {
        query: String,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum TaskKindArg {
    Compound,
    Primitive,
}

impl From<TaskKindArg> for TaskKind {
    fn from(value: TaskKindArg) -> Self {
        match value {
            TaskKindArg::Compound => Self::Compound,
            TaskKindArg::Primitive => Self::Primitive,
        }
    }
}

#[derive(Debug, Subcommand)]
enum MemoryCommand {
    Store {
        content: String,
        #[arg(long = "tag")]
        tags: Vec<String>,
        #[arg(long)]
        node_id: Option<String>,
        #[arg(long)]
        source: Option<String>,
        #[arg(long, value_parser = parse_confidence)]
        confidence: Option<f32>,
        #[arg(long, value_enum)]
        scope: Option<ScopeArg>,
        #[arg(long)]
        agent_id: Option<String>,
        #[arg(long)]
        session_id: Option<String>,
        #[arg(long)]
        task_id: Option<String>,
        #[arg(long)]
        user_id: Option<String>,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
    },
    Answer {
        question: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
    },
    Find {
        query: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        #[arg(long)]
        node_id: Option<String>,
        #[arg(long, value_enum)]
        scope: Option<ScopeArg>,
        #[arg(long)]
        agent_id: Option<String>,
        #[arg(long)]
        session_id: Option<String>,
        #[arg(long)]
        task_id: Option<String>,
        #[arg(long)]
        user_id: Option<String>,
        /// Expand results one hop through explicit entity-relation links.
        #[arg(long)]
        expand: bool,
        /// Diversify results with Maximal Marginal Relevance to drop near-duplicates (fetches a
        /// larger pool, then re-ranks to --limit).
        #[arg(long)]
        mmr: bool,
        /// MMR relevance/novelty trade-off in [0,1] (1 = pure relevance). Implies --mmr.
        #[arg(long)]
        mmr_lambda: Option<f32>,
        /// Include Archived records in results (excluded by default).
        #[arg(long, default_value_t = false)]
        include_archived: bool,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
    },
    Context {
        query: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        #[arg(long, default_value_t = 4000)]
        index_chars: usize,
        /// Assemble a goal-scoped packet (goal + invariants + relevant memory) for this goal
        /// instead of a flat index + hit list. Invariants are memories tagged `invariant`.
        #[arg(long)]
        goal: Option<String>,
        /// Expand memory hits one hop through explicit entity-relation links.
        #[arg(long)]
        expand: bool,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
    },
    /// Distill raw text into self-contained atomic facts via an LLM (resolve coreferences, anchor
    /// relative dates), optionally storing each as a memory atom. Needs an `[acc.compressor]` or
    /// `[acc.judge]` LLM endpoint (point it at a local Ollama / LM Studio for zero token cost) and
    /// a build with `--features llm`.
    Distill {
        /// Raw text to distill. Omit and pass --file to read from a file instead.
        text: Option<String>,
        #[arg(long)]
        file: Option<PathBuf>,
        /// Reference date for anchoring relative time references (defaults to today).
        #[arg(long)]
        reference_date: Option<String>,
        /// Store each extracted fact as a memory atom (tagged `fact`).
        #[arg(long)]
        store: bool,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
    },
    /// Rebuild the searchable projection from the OKF source of truth: re-index every OKF markdown
    /// file under `--from` into the backend (transactional, idempotent). The durable md files are
    /// the append-only source; the vector store is a derived projection you can rebuild any time.
    Rebuild {
        /// OKF source directory to re-index (defaults to the configured memory root).
        #[arg(long)]
        from: Option<PathBuf>,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
    },
    /// Run one ACC commit-loop cycle: recall, qualify-gate, and admit into the bounded
    /// committed context state; print the committed context and the cycle metrics. Defaults
    /// come from the `[acc]` block of artesian.toml; flags override per invocation.
    Commit {
        query: String,
        #[arg(long)]
        budget_tokens: Option<usize>,
        #[arg(long)]
        recall_limit: Option<usize>,
        #[arg(long)]
        min_score: Option<f32>,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
    },
    Anchor {
        #[command(subcommand)]
        command: AnchorCommand,
    },
    /// Commit a curated SKILL memory record with provenance and governance.
    ///
    /// Unlike flat-file skill stores (e.g. Hermes /learn, deepagents Skills), Artesian skill
    /// records carry provenance (`source`), usage signals (`access_count`), and participate in
    /// the normal decay/eviction lifecycle — making them governed, portable, and auditable.
    ///
    /// DISCIPLINE: commit a CURATED skill — clear title, polished body, explicit source paths.
    /// Do not dump raw conversation output; distill it first (`artesian memory distill`).
    ///
    /// Re-learning the same title + body is idempotent (content-hash dedup via `node_id`).
    /// Supplying `--step` stores an ordered guarded procedure for `artesian skill replay`.
    Learn {
        /// Short human-readable title for this skill (used as a stable lookup key in listings).
        title: String,
        /// Inline body text for the skill. May be combined with --from.
        #[arg(long)]
        content: Option<String>,
        /// File path (or URL recorded as provenance) contributing to the skill body.
        /// For local paths the file is read and appended to the body; URLs are recorded as
        /// provenance metadata only (no network fetch in the CLI).
        /// Repeat to combine multiple sources.
        #[arg(long = "from")]
        from: Vec<String>,
        /// Extra tags to attach (the `skill` tag is always present).
        #[arg(long = "tag")]
        tags: Vec<String>,
        /// Add a replayable shell command/action step. Repeat for ordered procedures.
        #[arg(long = "step")]
        steps: Vec<String>,
        /// Attach a precondition check to the preceding --step. Exit 0 means the guard holds.
        #[arg(long = "guard")]
        guards: Vec<String>,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
    },
    /// List learned skills (memories tagged `skill`) with title, usage, provenance, last access.
    ///
    /// Shows both manually learned skills (`artesian learn`) and skills auto-committed by the
    /// autonomous loop on verified goal success.
    Skills {
        /// Sort by usage (access_count descending) rather than the default relevance order.
        #[arg(long)]
        by_usage: bool,
        /// Maximum number of skills to return (default 20).
        #[arg(long, default_value_t = 20)]
        limit: usize,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
    },
    /// Print the temporal profile (facts ordered by time) for a named entity.
    ///
    /// Fetches all records mentioning `entity` (matched case-insensitively against tags and
    /// extracted named entities in content) and prints them oldest-first. Useful for tracing
    /// how knowledge about a specific component, acronym, or concept has evolved over time.
    ///
    /// Examples:
    ///   artesian memory timeline RateLimit
    ///   artesian memory timeline BackgroundJobRetryPolicy --limit 50
    Timeline {
        /// Entity name to look up (a tag, CamelCase identifier, or ALL-CAPS acronym).
        entity: String,
        /// Maximum number of records to display (oldest N after entity filtering).
        #[arg(long, default_value_t = 20)]
        limit: usize,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
    },
    /// Evict memories: soft-archive (default) or hard-delete (--hard) by TTL, LRU, or score.
    ///
    /// Soft-archive sets `state=archived` — records remain stored and retrievable via
    /// `get_node` or `--include-archived`, but are excluded from default `find` results.
    /// Use `--hard` to permanently delete records that are already archived (two-pass safety).
    /// Every archive/delete decision is appended to `~/.artesian/eviction.jsonl`.
    Evict {
        /// Archive records whose last-access (or created_at) is older than N days.
        #[arg(long)]
        ttl_days: Option<f32>,
        /// Archive records with the lowest retrieval strength (bottom 50% by default).
        #[arg(long, default_value_t = false)]
        lru: bool,
        /// Archive records whose decay-adjusted retrieval strength is below this threshold.
        #[arg(long)]
        min_strength: Option<f32>,
        /// After other policies, archive lowest-strength records until at most N Active remain.
        #[arg(long)]
        max_keep: Option<usize>,
        /// Permanently delete already-Archived records (does NOT archive additional records).
        /// Combine with a prior `evict` run that used --ttl-days / --lru.
        #[arg(long, default_value_t = false)]
        hard: bool,
        /// Dry-run: print what would be archived/deleted without changing any data.
        #[arg(long, default_value_t = false)]
        dry_run: bool,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
    },
}

#[derive(Debug, Subcommand)]
enum SkillCommand {
    /// Replay a learned skill's guarded procedure.
    Replay {
        /// Skill title to replay.
        title: String,
        /// Print the plan without running guards or step commands. This is the default.
        #[arg(long, conflicts_with = "execute")]
        dry_run: bool,
        /// Execute the guarded replay. Each guard must pass before its step command runs.
        #[arg(long, conflicts_with = "dry_run")]
        execute: bool,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
    },
}

#[derive(Debug, Subcommand)]
enum AnchorCommand {
    Get {
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
    },
    Set {
        #[arg(long)]
        current_task: String,
        #[arg(long)]
        next_step: String,
        #[arg(long)]
        plan_pointer: Option<String>,
        #[arg(long = "decision")]
        last_decisions: Vec<String>,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
    },
    Recover {
        #[arg(long, default_value_t = 5)]
        limit: usize,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
    },
}

#[derive(Debug, Subcommand)]
enum KitCommand {
    /// Initialize the loop memory kit in the memory root: writes vision.md, agents.md, and
    /// kit/index.md so a new session or a different model can load context in one step.
    Init {
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
        /// One-line description of the project vision (written to vision.md).
        #[arg(long)]
        vision: Option<String>,
    },
    /// Print the current kit: vision summary + most-recent session anchor.
    Status {
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
    },
    /// Export the kit: a single markdown file (default), or a portable working-context bundle
    /// directory (`--format bundle`) another runtime can import to resume.
    Export {
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = KitFormat::Markdown)]
        format: KitFormat,
    },
    /// Import a portable working-context bundle directory: validate it and print the committed
    /// working context an agent would resume from.
    Import {
        /// Bundle directory written by `kit export --format bundle`.
        input: PathBuf,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
    },
    /// Validate a bundle (OCF or native); with `--against <other>`, check schema compatibility for
    /// a resume — the pre-import check the OCF spec describes.
    Validate {
        /// Bundle directory to validate.
        input: PathBuf,
        /// Another bundle whose schema/budget this one must be compatible with to resume into.
        #[arg(long)]
        against: Option<PathBuf>,
    },
}

/// Output shape for `kit export`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum KitFormat {
    /// A single human-readable markdown file (anchors concatenated).
    Markdown,
    /// A portable working-context bundle directory (manifest + snapshot + lifecycle log).
    Bundle,
    /// An Open Cognitive Format (OCF) bundle directory (manifest + schema + snapshot + qualify).
    Ocf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum BackendArg {
    Files,
    SqliteVec,
    Qdrant,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ScopeArg {
    Shared,
    Agent,
    Session,
    Task,
}

impl From<ScopeArg> for MemoryScope {
    fn from(value: ScopeArg) -> Self {
        match value {
            ScopeArg::Shared => Self::Shared,
            ScopeArg::Agent => Self::Agent,
            ScopeArg::Session => Self::Session,
            ScopeArg::Task => Self::Task,
        }
    }
}

impl From<BackendArg> for MemoryBackendKind {
    fn from(value: BackendArg) -> Self {
        match value {
            BackendArg::Files => Self::Files,
            BackendArg::SqliteVec => Self::SqliteVec,
            BackendArg::Qdrant => Self::Qdrant,
        }
    }
}

/// The basename this binary was invoked as, for multi-call dispatch.
fn invoked_as() -> Option<String> {
    std::env::args_os().next().and_then(|arg0| {
        std::path::Path::new(&arg0)
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    // Multi-call binary: a single executable installed once and symlinked, so the MCP server and
    // the daemon don't each ship a second copy of the runtime (and ONNX Runtime).
    match invoked_as().as_deref() {
        Some("artesian-mcp") => return artesian_mcp::cli::run().await,
        Some("artesiand") => return artesiand::run().await,
        _ => {}
    }
    let raw_args: Vec<OsString> = std::env::args_os().collect();
    let cli = Cli::parse_from(raw_args.clone());
    match cli.command {
        Command::Init {
            memory_root,
            project,
            backend,
            collection,
            qdrant_url,
            qdrant_rest_url,
            qdrant_api_key_env,
            qdrant_api_key_file,
            non_interactive,
            register_mcp,
        } => {
            let memory_root = project_memory_root(memory_root, project.as_deref());
            let collection = project_collection(collection, project.as_deref());
            init(
                InitOptions {
                    memory_root,
                    backend,
                    collection,
                    project,
                    qdrant_url,
                    qdrant_rest_url,
                    qdrant_api_key_env,
                    qdrant_api_key_file,
                    register_mcp,
                },
                non_interactive,
            )
            .await
        }
        Command::Spawn {
            role,
            agent,
            args,
            model,
            timeout_seconds,
        } => spawn(&role, &agent, args, model, timeout_seconds).await,
        Command::Agents { command } => agents(command).await,
        Command::Hooks { command } => hooks(command).await,
        Command::Run {
            config,
            root,
            dry_run,
            once,
        }
        | Command::Orchestrate {
            config,
            root,
            dry_run,
            once,
        } => run_orchestrator(config, root, dry_run, once).await,
        Command::Memory { command } => memory(command, &raw_args).await,
        Command::Qualify {
            candidate,
            goal,
            json,
            config,
            root,
            backend,
        } => qualify(candidate, goal, json, config, root, backend).await,
        Command::Skill { command } => skill(command).await,
        Command::Handoff {
            session_id,
            user_id,
            task_id,
            config,
            root,
            backend,
        } => handoff(session_id, user_id, task_id, config, root, backend).await,
        Command::Session { command } => session(command).await,
        Command::Task { command } => task(command).await,
        Command::Team { command } => team(command).await,
        Command::Backfill {
            directory,
            config,
            root,
            backend,
            user_id,
            no_link,
            consolidate,
        } => {
            backfill(
                directory,
                config,
                root,
                backend,
                user_id,
                no_link,
                consolidate,
            )
            .await
        }
        Command::Onboard {
            project,
            directory,
            backend,
            memory_root,
            collection,
            qdrant_url,
            qdrant_rest_url,
            qdrant_api_key_env,
            qdrant_api_key_file,
            user_id,
            config,
            no_link,
            consolidate,
        } => {
            let memory_root = project_memory_root(memory_root, Some(&project));
            let collection = project_collection(collection, Some(&project));
            onboard(
                project.clone(),
                directory,
                InitOptions {
                    memory_root,
                    backend,
                    collection,
                    qdrant_url,
                    qdrant_rest_url,
                    qdrant_api_key_env,
                    qdrant_api_key_file,
                    register_mcp: true,
                    project: Some(project.clone()),
                },
                config,
                user_id,
                no_link,
                consolidate,
            )
            .await
        }
        Command::Consolidate {
            config,
            root,
            allow_llm,
        } => consolidate(config, root, allow_llm).await,
        Command::Dream {
            config,
            root,
            collection,
            out,
            diary,
            admit_threshold,
            similarity_threshold,
        } => {
            dream_command(
                config,
                root,
                collection,
                out,
                diary,
                admit_threshold,
                similarity_threshold,
            )
            .await
        }
        Command::Migrate { command } => match command {
            MigrateCommand::OkfBundle {
                okf_root,
                config,
                new_collection,
                retention_days,
            } => migrate_okf(okf_root, config, new_collection, retention_days).await,
            MigrateCommand::Rechunk { config } => migrate_rechunk(config).await,
        },
        Command::Snapshot {
            config,
            output_dir,
            collection,
        } => snapshot(config, output_dir, collection).await,
        Command::Okf { command } => okf(command),
        Command::Kit { command } => kit(command).await,
        Command::Perf {
            config,
            root,
            budget_tokens,
        } => perf(config, root, budget_tokens).await,
        Command::Tokens { json, since, by_op } => tokens_command(json, since, by_op),
        Command::Quota { json } => quota_command(json),
        Command::Loop {
            goal,
            worker_cmd,
            max_turns,
            max_wall_secs,
            poll,
            no_learn,
            max_remediation_attempts,
            quota_warn_pct,
            checkpoint_on_quota,
            root,
            config,
        } => {
            run_loop(LoopCliOptions {
                goal,
                worker_cmd,
                max_turns,
                max_wall_secs,
                poll,
                learn: !no_learn,
                max_remediation_attempts,
                quota_warn_pct,
                checkpoint_on_quota,
                root,
                config,
            })
            .await
        }
        Command::Replicate {
            from_url,
            to_url,
            from_key,
            to_key,
            collection,
            to_collection,
            status,
            full,
            incremental: _,
            prune,
            batch,
        } => {
            run_replicate(
                from_url,
                to_url,
                from_key,
                to_key,
                collection,
                to_collection,
                status,
                full,
                prune,
                batch,
            )
            .await
        }
        Command::Doctor {
            config,
            root,
            backend,
        } => doctor(config, root, backend).await,
        Command::Update { restart_stale } => update::update(restart_stale),
    }
}

// LoopCommands, LoopCommandFuture, LoopRunOptions are imported from flume::loop_core above.

struct ShellLoopCommands;

impl LoopCommands for ShellLoopCommands {
    fn run_worker<'a>(
        &'a mut self,
        cmd: &'a str,
        env: Vec<(String, String)>,
        timeout: Option<Duration>,
    ) -> LoopCommandFuture<'a, bool> {
        Box::pin(async move {
            let (success, _) = run_shell_capture_with_env(cmd, env, timeout).await?;
            Ok(success)
        })
    }

    fn verify_goal<'a>(
        &'a mut self,
        cmd: &'a str,
        timeout: Option<Duration>,
    ) -> LoopCommandFuture<'a, (bool, String)> {
        Box::pin(async move { run_shell_capture_with_env(cmd, Vec::new(), timeout).await })
    }
}

/// Run a shell command with extra environment variables, capturing combined stdout+stderr so a
/// failing verifier's detail can be surfaced as the next turn's "last failed check". `kill_on_drop`
/// keeps wall-clock caps from leaving a hung child behind when timeout aborts the future.
async fn run_shell_capture_with_env(
    cmd: &str,
    env: Vec<(String, String)>,
    timeout: Option<Duration>,
) -> Result<(bool, String)> {
    let mut command = TokioCommand::new("sh");
    command
        .arg("-c")
        .arg(cmd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    for (key, value) in env {
        command.env(key, value);
    }
    let output = match timeout {
        Some(timeout) => tokio::time::timeout(timeout, command.output())
            .await
            .with_context(|| {
                format!("loop exceeded wall-clock budget while running command: {cmd}")
            })?
            .with_context(|| format!("run command: {cmd}"))?,
        None => command
            .output()
            .await
            .with_context(|| format!("run command: {cmd}"))?,
    };
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    Ok((output.status.success(), text.trim().to_string()))
}

struct LoopCliOptions {
    goal: String,
    worker_cmd: Option<String>,
    max_turns: u32,
    max_wall_secs: Option<u64>,
    poll: bool,
    learn: bool,
    max_remediation_attempts: u32,
    quota_warn_pct: f64,
    checkpoint_on_quota: bool,
    root: PathBuf,
    config: PathBuf,
}

async fn run_loop(options: LoopCliOptions) -> Result<()> {
    let anchor_store = AnchorAnchorStore::new(&options.root);
    // Use the project's configured memory backend when present; otherwise a local files backend
    // under the loop root. Recall/commit degrade to a no-op if the backend cannot be opened.
    // No-config fallback: a files backend rooted at `--root`, the same memory root the other
    // `memory` subcommands use for that root, so invariants stored via `memory store` are visible.
    let memory_config = load_config(&options.config)
        .map(|cfg| cfg.memory)
        .unwrap_or_else(|_| {
            ArtesianConfig::memory_files(options.root.display().to_string(), Vec::new()).memory
        });
    let backend = match open_memory_backend(&memory_config) {
        Ok(backend) => Some(backend),
        Err(error) => {
            eprintln!("  note: memory recall/commit disabled ({error})");
            None
        }
    };
    let run_options = LoopRunOptions {
        goal: options.goal,
        worker_cmd: options.worker_cmd,
        max_turns: options.max_turns,
        max_wall: options.max_wall_secs.map(Duration::from_secs),
        poll: options.poll,
        learn: options.learn,
        run_id: loop_run_id(),
        run_log_dir: loop_run_log_dir()?,
        stop_file: loop_stop_file()?,
        collection: memory_config.collection.clone(),
        track_savings: memory_config.track_savings,
        max_remediation_attempts: options.max_remediation_attempts,
        cancel: Default::default(),
        on_progress: None,
        quota: QuotaLoopConfig {
            warn_pct: options.quota_warn_pct,
            checkpoint_on_quota: options.checkpoint_on_quota,
            ..QuotaLoopConfig::default()
        },
    };
    let mut commands = ShellLoopCommands;
    let report = run_loop_core(
        run_options,
        backend.as_deref(),
        &anchor_store,
        &mut commands,
    )
    .await?;
    if report.outcome != "success" {
        bail!("{}", report.why_stopped);
    }
    Ok(())
}

/// Distill raw text into self-contained atomic facts via an LLM, optionally storing each as an atom.
#[cfg(feature = "llm")]
#[allow(clippy::too_many_arguments)]
async fn distill(
    text: Option<String>,
    file: Option<PathBuf>,
    reference_date: Option<String>,
    store: bool,
    config: PathBuf,
    root: PathBuf,
    backend: Option<BackendArg>,
) -> Result<()> {
    let raw = match (text, file) {
        (Some(text), _) => text,
        (None, Some(file)) => {
            fs::read_to_string(&file).with_context(|| format!("read {}", file.display()))?
        }
        (None, None) => bail!("provide text to distill, or --file <path>"),
    };
    let acc = load_config(&config)
        .map(|loaded| loaded.acc)
        .unwrap_or_default();
    let llm = acc
        .compressor
        .as_ref()
        .or(acc.judge.as_ref())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "distill needs an LLM endpoint; configure [acc.compressor] or [acc.judge] in {}",
                config.display()
            )
        })?;
    let client = headgate::llm_client_from_config(llm)?;
    let reference_date =
        reference_date.unwrap_or_else(|| chrono::Utc::now().format("%Y-%m-%d").to_string());
    let facts = headgate::extract_atomic_facts(client.as_ref(), &raw, &reference_date).await?;
    if facts.is_empty() {
        println!("(no durable facts extracted)");
        return Ok(());
    }
    for (index, fact) in facts.iter().enumerate() {
        println!("{}. {fact}", index + 1);
    }
    if store {
        let memory_config = memory_config_for_command(&config, root, backend)?;
        let backend = open_memory_backend(&memory_config)?;
        let mut stored = 0usize;
        for fact in &facts {
            let mut memory = StoreMemory::atom(fact.clone());
            memory.tags = vec!["fact".to_string()];
            if backend.store(memory).await.is_ok() {
                stored += 1;
            }
        }
        println!("\nstored {stored} fact atom(s)");
    }
    Ok(())
}

/// `memory distill` is unavailable without the LLM layer compiled in.
#[cfg(not(feature = "llm"))]
#[allow(clippy::too_many_arguments)]
async fn distill(
    _text: Option<String>,
    _file: Option<PathBuf>,
    _reference_date: Option<String>,
    _store: bool,
    _config: PathBuf,
    _root: PathBuf,
    _backend: Option<BackendArg>,
) -> Result<()> {
    bail!("`memory distill` requires the `llm` feature; rebuild with --features llm")
}

/// Replicate a Qdrant collection from one endpoint to another (scroll + upsert; merges by id).
#[cfg(feature = "qdrant")]
#[allow(clippy::too_many_arguments)]
async fn run_replicate(
    from_url: String,
    to_url: String,
    from_key: Option<String>,
    to_key: Option<String>,
    collection: String,
    to_collection: Option<String>,
    status: bool,
    full: bool,
    prune: bool,
    batch: u32,
) -> Result<()> {
    use aquifer::{
        replicate_collection, replicate_collection_incremental, QdrantVectorStore,
        QdrantVectorStoreConfig,
    };
    let target_collection = to_collection.unwrap_or_else(|| collection.clone());
    let mut from_cfg = QdrantVectorStoreConfig::new(from_url);
    from_cfg.api_key = from_key;
    let mut to_cfg = QdrantVectorStoreConfig::new(to_url);
    to_cfg.api_key = to_key;
    let source =
        QdrantVectorStore::connect(from_cfg).map_err(|error| anyhow::anyhow!("source: {error}"))?;
    let target =
        QdrantVectorStore::connect(to_cfg).map_err(|error| anyhow::anyhow!("target: {error}"))?;
    source
        .client()
        .health_check()
        .await
        .map_err(|error| anyhow::anyhow!("source unreachable: {error}"))?;
    target
        .client()
        .health_check()
        .await
        .map_err(|error| anyhow::anyhow!("target unreachable: {error}"))?;
    if status {
        println!("both Qdrant endpoints reachable; collection = {collection}");
        return Ok(());
    }
    if full {
        let copied = replicate_collection(&source, &target, &collection, &target_collection, batch)
            .await
            .map_err(|error| anyhow::anyhow!("replicate (full): {error}"))?;
        println!("replicated {copied} points (full): {collection} -> {target_collection}");
    } else {
        let report = replicate_collection_incremental(
            &source,
            &target,
            &collection,
            &target_collection,
            prune,
            batch,
        )
        .await
        .map_err(|error| anyhow::anyhow!("replicate (incremental): {error}"))?;
        println!(
            "replicated incremental: upserted={} deleted={} unchanged={} ({collection} -> {target_collection})",
            report.upserted, report.deleted, report.unchanged,
        );
    }
    Ok(())
}

/// Replicate is unavailable without the Qdrant backend compiled in.
#[cfg(not(feature = "qdrant"))]
#[allow(clippy::too_many_arguments)]
async fn run_replicate(
    _from_url: String,
    _to_url: String,
    _from_key: Option<String>,
    _to_key: Option<String>,
    _collection: String,
    _to_collection: Option<String>,
    _status: bool,
    _full: bool,
    _prune: bool,
    _batch: u32,
) -> Result<()> {
    bail!("`replicate` requires the `qdrant` feature; rebuild with --features qdrant")
}

/// Whether `path` contains `needle` (a cheap "is this registered?" check for config files).
fn file_mentions(path: &Path, needle: &str) -> bool {
    fs::read_to_string(path)
        .map(|text| text.contains(needle))
        .unwrap_or(false)
}

/// Which agent surfaces have the `artesian-memory` MCP server registered.
pub(crate) fn mcp_registration_status() -> Vec<(&'static str, bool)> {
    let claude_user = home_dir().ok().map(|home| home.join(".claude.json"));
    let codex = home_dir()
        .ok()
        .map(|home| home.join(".codex").join("config.toml"));
    let zed = zed_settings_path().ok();
    vec![
        (
            "Claude Code (project .mcp.json)",
            file_mentions(Path::new(".mcp.json"), MCP_SERVER_NAME),
        ),
        (
            "Claude Code (user ~/.claude.json)",
            claude_user.is_some_and(|path| file_mentions(&path, MCP_SERVER_NAME)),
        ),
        (
            "Codex",
            codex.is_some_and(|path| file_mentions(&path, MCP_SERVER_NAME)),
        ),
        (
            "Zed",
            zed.is_some_and(|path| file_mentions(&path, MCP_SERVER_NAME)),
        ),
    ]
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DoctorDiagnostic {
    summary: String,
    fix: String,
}

fn qdrant_backend_diagnostic(config: &MemoryConfig, error_text: &str) -> Option<DoctorDiagnostic> {
    qdrant_backend_diagnostic_with_key_state(
        config,
        error_text,
        config.resolve_qdrant_api_key().is_some(),
    )
}

fn qdrant_backend_diagnostic_with_key_state(
    config: &MemoryConfig,
    error_text: &str,
    api_key_set: bool,
) -> Option<DoctorDiagnostic> {
    if config.backend != MemoryBackendKind::Qdrant {
        return None;
    }
    if qdrant_error_looks_auth(error_text) {
        let env_name = qdrant_api_key_env_name(config);
        let key_state = if api_key_set {
            "Qdrant rejected the configured API key"
        } else {
            "the API key is not set"
        };
        return Some(DoctorDiagnostic {
            summary: format!(
                "Qdrant authentication failed — {key_state}. The key env for this config is `{env_name}` (from qdrant_api_key_env)."
            ),
            fix: qdrant_api_key_fix(config),
        });
    }
    if qdrant_error_looks_connection(error_text) || error_text.contains("backend error:") {
        return Some(DoctorDiagnostic {
            summary: format!("Qdrant backend failed — {error_text}"),
            fix: "check qdrant_url/qdrant_rest_url, the configured API key, and that the Qdrant server is reachable".to_string(),
        });
    }
    None
}

fn qdrant_error_looks_auth(error_text: &str) -> bool {
    let lower = error_text.to_ascii_lowercase();
    lower.contains("unauthenticated")
        || lower.contains("unauthorized")
        || lower.contains("forbidden")
        || lower.contains("401")
        || lower.contains("403")
        || lower.contains("api key")
}

fn qdrant_error_looks_connection(error_text: &str) -> bool {
    let lower = error_text.to_ascii_lowercase();
    lower.contains("connection refused")
        || lower.contains("failed to connect")
        || lower.contains("tcp connect")
        || lower.contains("timeout")
        || lower.contains("deadline")
        || lower.contains("unavailable")
        || lower.contains("transport error")
        || lower.contains("qdrant grpc preflight failed")
        || lower.contains("qdrant rest preflight failed")
}

fn qdrant_api_key_env_name(config: &MemoryConfig) -> &str {
    config
        .qdrant_api_key_env
        .as_deref()
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .unwrap_or("QDRANT_API_KEY")
}

fn qdrant_api_key_fix(config: &MemoryConfig) -> String {
    let env_name = qdrant_api_key_env_name(config);
    if let Some(file) = config
        .qdrant_api_key_file
        .as_deref()
        .map(str::trim)
        .filter(|file| !file.is_empty())
    {
        format!(
            "export `{env_name}`, or make `qdrant_api_key_file = \"{file}\"` readable and contain `{env_name}=...`"
        )
    } else {
        format!("export `{env_name}`, or set `qdrant_api_key_file` in artesian.toml")
    }
}

/// Health check: binary, config, backend reachability, collection compatibility, and MCP
/// registrations — printing the exact fix for anything that drifted (e.g. after an upgrade).
async fn doctor(config_path: PathBuf, root: PathBuf, backend: Option<BackendArg>) -> Result<()> {
    let mut problems = 0usize;
    println!("artesian doctor (v{})", env!("CARGO_PKG_VERSION"));

    let memory_config = memory_config_for_command(&config_path, root, backend)?;
    let source = if config_path.exists() {
        config_path.display().to_string()
    } else {
        "defaults (no artesian.toml)".to_string()
    };
    println!(
        "  config:  {source} — backend={:?}, root={}, collection={}",
        memory_config.backend, memory_config.root, memory_config.collection
    );

    // Qdrant reachability gives a clearer message than the generic open error.
    #[cfg(feature = "qdrant")]
    if memory_config.backend == MemoryBackendKind::Qdrant {
        match runtime::qdrant_config_from(&memory_config) {
            Ok(qdrant_config) => match aquifer::preflight_qdrant(qdrant_config).await {
                Ok(report) => println!(
                    "  qdrant:  reachable (gRPC {}, REST {})",
                    report.grpc_url, report.rest_status
                ),
                Err(error) => {
                    problems += 1;
                    if let Some(diagnostic) =
                        qdrant_backend_diagnostic(&memory_config, &error.to_string())
                    {
                        println!("  qdrant:  ERROR — {}", diagnostic.summary);
                        println!("           fix: {}", diagnostic.fix);
                    } else {
                        println!("  qdrant:  UNREACHABLE — {error}");
                        println!(
                                "           fix: check the URL + API key env, and that the server is up"
                            );
                    }
                }
            },
            Err(error) => {
                problems += 1;
                println!("  qdrant:  misconfigured — {error}");
            }
        }
    }

    // Opening + probing the backend exercises the collection-compatibility check.
    match open_memory_backend(&memory_config) {
        Ok(backend) => match backend
            .find(MemoryQuery::new("artesian doctor probe").with_limit(1))
            .await
        {
            Ok(hits) => println!(
                "  memory:  responsive ({})",
                if hits.is_empty() {
                    "reachable"
                } else {
                    "has entries"
                }
            ),
            Err(error) => {
                problems += 1;
                if let Some(diagnostic) =
                    qdrant_backend_diagnostic(&memory_config, &error.to_string())
                {
                    println!("  memory:  ERROR — {}", diagnostic.summary);
                    println!("           fix: {}", diagnostic.fix);
                } else {
                    println!("  memory:  ERROR — {error}");
                    println!(
                        "           fix: `artesian memory rebuild` (or `artesian migrate` if the embedding model changed)"
                    );
                }
            }
        },
        Err(error) => {
            problems += 1;
            if let Some(diagnostic) = qdrant_backend_diagnostic(&memory_config, &error.to_string())
            {
                println!("  backend: ERROR opening — {}", diagnostic.summary);
                println!("           fix: {}", diagnostic.fix);
            } else {
                println!("  backend: ERROR opening — {error}");
            }
        }
    }

    let registrations = mcp_registration_status();
    let registered: Vec<&str> = registrations
        .iter()
        .filter(|(_, ok)| *ok)
        .map(|(name, _)| *name)
        .collect();
    if registered.is_empty() {
        println!("  mcp:     not registered for any agent");
        println!("           fix: `artesian init --register-mcp` (or `artesian onboard …`)");
    } else {
        println!("  mcp:     registered for {}", registered.join(", "));
    }

    println!();
    if problems == 0 {
        println!("✓ all checks passed");
        Ok(())
    } else {
        bail!("{problems} problem(s) found — see the fixes above")
    }
}

async fn task(command: TaskCommand) -> Result<()> {
    match command {
        TaskCommand::Add {
            title,
            id,
            description,
            blockers,
            kind,
            role,
            config,
            root,
            backend,
        } => {
            let source = FilesTaskStore::new(&root);
            let memory = open_backend_for_command(&config, root, backend)?;
            let store = VectorTaskStore::new(source, memory);
            let mut task = NewTask::primitive(title);
            task.id = id;
            task.description = description;
            task.blockers = blockers;
            task.kind = kind.into();
            task.role = Role::from_str(&role)?;
            let task = store.create(task).await?;
            println!("created task id={} status=todo", task.id);
        }
        TaskCommand::List { root } => {
            let store = FilesTaskStore::new(root);
            for task in store.list().await? {
                println!("{}\t{:?}\t{}", task.id, task.status, task.title);
            }
        }
        TaskCommand::Claim { id, claimant, root } => {
            let store = FilesTaskStore::new(root);
            match store
                .claim(ClaimRequest {
                    task_id: id,
                    claimant,
                })
                .await?
            {
                Some(task) => println!(
                    "claimed task id={} claimant={}",
                    task.id,
                    task.claimed_by.unwrap_or_default()
                ),
                None => println!("no dispatch-eligible task"),
            }
        }
        TaskCommand::Done {
            id,
            verify_commands,
            root,
        } => {
            let store = FilesTaskStore::new(root);
            let verifiers: Vec<Arc<dyn Verifier>> = verify_commands
                .into_iter()
                .map(|command| Arc::new(CommandVerifier::new(command.clone(), command)) as _)
                .collect();
            let gate = VerifierGate::new(verifiers);
            let task = gate.mark_done(&store, &id).await?;
            println!("completed task id={} status=done", task.id);
        }
        TaskCommand::Find {
            query,
            config,
            root,
            backend,
        } => {
            let source = FilesTaskStore::new(&root);
            let memory = open_backend_for_command(&config, root, backend)?;
            let store = VectorTaskStore::new(source, memory);
            for task in store.find(&query).await? {
                println!("{}\t{:?}\t{}", task.id, task.status, task.title);
            }
        }
    }
    Ok(())
}

async fn team(command: TeamCommand) -> Result<()> {
    match command {
        TeamCommand::Create {
            name,
            id,
            max_teammates,
            plan_approval_required,
            plan_approval_roles,
            config,
        } => {
            let mut runtime = team_runtime(&config).await?;
            let team = runtime.create_team(TeamCreate {
                id,
                name,
                max_teammates,
                plan_approval_required,
                plan_approval_roles,
            });
            println!("{}", serde_json::to_string_pretty(&team)?);
        }
        TeamCommand::Spawn {
            team_id,
            definition,
            config,
        } => {
            let mut runtime = team_runtime(&config).await?;
            ensure_ephemeral_team(&mut runtime, &team_id);
            let teammate = runtime
                .spawn_teammate(TeamSpawn {
                    team_id,
                    definition,
                })
                .await?;
            println!("{}", serde_json::to_string_pretty(&teammate)?);
        }
        TeamCommand::Task { command } => team_task(command).await?,
        TeamCommand::Message {
            team_id,
            from,
            to,
            kind,
            content,
            task_id,
            approved,
            execute,
            config,
        } => {
            let mut runtime = team_runtime(&config).await?;
            ensure_ephemeral_team(&mut runtime, &team_id);
            let (event_sender, event_printer) = if execute {
                let (sender, mut receiver) =
                    tokio::sync::mpsc::unbounded_channel::<TeamWorkerEvent>();
                let printer = tokio::spawn(async move {
                    while let Some(event) = receiver.recv().await {
                        for line in event.text.lines().filter(|line| !line.trim().is_empty()) {
                            eprintln!("[worker:{}] {}", event.teammate, line);
                        }
                    }
                });
                (Some(sender), Some(printer))
            } else {
                (None, None)
            };
            let outcome = runtime
                .message_with_worker_events(
                    TeamMessage {
                        team_id,
                        from,
                        to,
                        kind: kind.into(),
                        content,
                        task_id,
                        approved,
                        execute,
                        resume_packet: None,
                    },
                    event_sender,
                )
                .await;
            if let Some(printer) = event_printer {
                let _ = printer.await;
            }
            let outcome = outcome?;
            println!("{}", serde_json::to_string_pretty(&outcome)?);
        }
        TeamCommand::Status { team_id, config } => {
            let mut runtime = team_runtime(&config).await?;
            ensure_ephemeral_team(&mut runtime, &team_id);
            println!(
                "{}",
                serde_json::to_string_pretty(&runtime.status(&team_id)?)?
            );
        }
        TeamCommand::Cleanup { team_id, config } => {
            let mut runtime = team_runtime(&config).await?;
            ensure_ephemeral_team(&mut runtime, &team_id);
            println!(
                "{}",
                serde_json::to_string_pretty(&runtime.cleanup(&team_id)?)?
            );
        }
        TeamCommand::Gc {
            ttl_secs,
            heartbeat_timeout_secs,
            config,
        } => {
            let runtime = team_runtime(&config).await?;
            let mut options = TeamGcOptions::default();
            if let Some(secs) = ttl_secs {
                options = options.with_ttl(Duration::from_secs(secs));
            }
            if let Some(secs) = heartbeat_timeout_secs {
                options = options.with_heartbeat_timeout(Duration::from_secs(secs));
            }
            let report = runtime.gc(options)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "scanned": report.scanned,
                    "terminated": report.terminated,
                    "removed": report.removed,
                    "expired": report.expired,
                    "skipped_unverified": report.skipped_unverified,
                }))?
            );
        }
        TeamCommand::Presence { team_id, config } => {
            let mut runtime = team_runtime(&config).await?;
            ensure_ephemeral_team(&mut runtime, &team_id);
            let snapshot = runtime.presence(&team_id)?;
            println!("{}", serde_json::to_string_pretty(&snapshot)?);
        }
    }
    Ok(())
}

async fn team_task(command: TeamTaskCommand) -> Result<()> {
    match command {
        TeamTaskCommand::Add {
            team_id,
            title,
            description,
            definition,
            blockers,
            config,
        } => {
            let mut runtime = team_runtime(&config).await?;
            ensure_ephemeral_team(&mut runtime, &team_id);
            let task = runtime
                .add_task(TeamTaskAdd {
                    team_id,
                    title,
                    description,
                    definition,
                    blockers,
                })
                .await?;
            println!("{}", serde_json::to_string_pretty(&task)?);
        }
        TeamTaskCommand::Claim {
            team_id,
            task_id,
            teammate,
            config,
        } => {
            let mut runtime = team_runtime(&config).await?;
            ensure_ephemeral_team(&mut runtime, &team_id);
            let task = runtime
                .claim_task(TeamTaskClaim {
                    team_id,
                    task_id,
                    teammate,
                })
                .await?;
            println!("{}", serde_json::to_string_pretty(&task)?);
        }
        TeamTaskCommand::Complete {
            team_id,
            task_id,
            reviewer,
            approved,
            config,
        } => {
            let mut runtime = team_runtime(&config).await?;
            ensure_ephemeral_team(&mut runtime, &team_id);
            let task = runtime
                .complete_task(TeamTaskComplete {
                    team_id,
                    task_id,
                    reviewer,
                    approved,
                })
                .await?;
            println!("{}", serde_json::to_string_pretty(&task)?);
        }
    }
    Ok(())
}

async fn team_runtime(config_path: &Path) -> Result<TeamRuntime> {
    let config = load_config(config_path)?;
    if !matches!(config.mode, Mode::Orchestrate | Mode::Full) {
        bail!(
            "team tools require mode orchestrate or full, got {:?}",
            config.mode
        );
    }
    let repo_root = env::current_dir()?;
    let definitions = load_role_definitions(&repo_root)?;
    let mut catalog = fallback_agent_catalog(&config.agents);
    catalog.roles = role_summaries(&definitions);
    let process_defaults = process_supervisor_from_config(&config, &repo_root);
    Ok(TeamRuntime::new(TeamRuntimeConfig {
        repo_root,
        task_root: PathBuf::from(&config.memory.root).join("tasks"),
        registry_dir: process_defaults.registry_dir().to_path_buf(),
        bindings: config.agents,
        catalog,
        definitions,
        max_teammates: config
            .coordination
            .max_concurrent_spawns
            .unwrap_or(4)
            .max(1),
        max_concurrent_spawns: config
            .coordination
            .max_concurrent_spawns
            .unwrap_or(4)
            .max(1),
        max_lifetime: Duration::from_secs(
            config
                .coordination
                .spawn_max_lifetime_seconds
                .unwrap_or(30 * 60),
        ),
        termination_grace: Duration::from_millis(
            config
                .coordination
                .spawn_shutdown_grace_millis
                .unwrap_or(2_000),
        ),
    }))
}

fn ensure_ephemeral_team(runtime: &mut TeamRuntime, team_id: &str) {
    if runtime.status(team_id).is_ok() {
        return;
    }
    let _ = runtime.create_team(TeamCreate {
        id: Some(team_id.to_string()),
        name: team_id.to_string(),
        max_teammates: None,
        plan_approval_required: false,
        plan_approval_roles: Vec::new(),
    });
}

async fn init(options: InitOptions, _non_interactive: bool) -> Result<()> {
    fs::create_dir_all(options.memory_root.join("memory"))
        .with_context(|| format!("create memory root {}", options.memory_root.display()))?;
    if options.backend == BackendArg::Qdrant {
        preflight_qdrant_options(&options).await?;
    }
    let mut agents = detect_agents();
    prefill_models_from_catalog(&mut agents);
    let config = ArtesianConfig {
        mode: artesian_core::Mode::Memory,
        memory: MemoryConfig {
            backend: options.backend.into(),
            root: options.memory_root.display().to_string(),
            collection: options.collection,
            qdrant_url: options.qdrant_url,
            qdrant_rest_url: options.qdrant_rest_url,
            qdrant_api_key_env: Some(options.qdrant_api_key_env),
            qdrant_api_key_file: options.qdrant_api_key_file,
            local_rerank_enabled: true,
            hyde_enabled: false,
            multi_query_enabled: false,
            debate_enabled: false,
            llm_consolidation_enabled: false,
            semantic_cache: Default::default(),
            track_access: true,
            track_savings: true,
        },
        agents,
        coordination: Default::default(),
        acc: Default::default(),
        dream_on_compact: false,
    };
    let config_path = Path::new(DEFAULT_CONFIG);
    if !config_path.exists() || options.project.is_some() {
        fs::write(config_path, config.to_toml()?)?;
    }
    if options.register_mcp {
        write_mcp_registrations(
            &env::current_dir()?.join(config_path),
            config.memory.backend,
        )?;
    }
    if claude_code_detected(&config.agents) {
        let report = install_claude_hooks(HookScope::Project)?;
        print_claude_hooks_report(&report)?;
    }
    write_master_role_skill(&options.memory_root)?;
    println!(
        "initialized Artesian memory mode at {} collection={} project={}",
        options.memory_root.display(),
        config.memory.collection,
        options.project.as_deref().unwrap_or("default")
    );
    Ok(())
}

fn claude_code_detected(agents: &[AgentBinding]) -> bool {
    agents
        .iter()
        .any(|binding| matches!(binding.agent.as_str(), "claude" | "claude-code"))
}

fn prefill_models_from_catalog(agents: &mut [AgentBinding]) {
    let catalog = fallback_agent_catalog(agents);
    for binding in agents {
        if binding.model.is_some() {
            continue;
        }
        binding.model = catalog
            .agents
            .iter()
            .find(|entry| entry.agent == binding.agent)
            .and_then(|entry| entry.models.iter().find(|model| model.reachable))
            .map(|model| model.id.clone());
    }
}

fn write_master_role_skill(memory_root: &Path) -> Result<()> {
    let path = memory_root.join("master-role.md");
    if path.exists() {
        return Ok(());
    }
    fs::write(
        path,
        "<!-- SPDX-License-Identifier: Apache-2.0 -->\n\n# Artesian Lead Role Skill\n\nWhen Artesian is running in `orchestrate` or `full` mode, inspect `agents.list` for reachable agents, models, and role definitions. Use `memory.context` for compact project recall. For multi-teammate work, create a Flume with `team.create`, admit definitions with `team.spawn`, coordinate through `team.task.*` and `team.message`, and gate accepted outcomes through the judge/master path before marking work done. For a single bounded subtask, `orchestrate.delegate(worker)` is still sufficient.\n",
    )?;
    Ok(())
}

fn project_memory_root(memory_root: Option<PathBuf>, project: Option<&str>) -> PathBuf {
    memory_root.unwrap_or_else(|| {
        project
            .map(|project| PathBuf::from(".artesian").join(project))
            .unwrap_or_else(|| PathBuf::from(".artesian"))
    })
}

fn project_collection(collection: Option<String>, project: Option<&str>) -> String {
    collection
        .or_else(|| project.map(sanitize_project_name))
        .unwrap_or_else(|| "artesian-memory".to_string())
}

fn sanitize_project_name(project: &str) -> String {
    project
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}

async fn agents(command: AgentsCommand) -> Result<()> {
    match command {
        AgentsCommand::Refresh { config, cache } => {
            let config = load_config(&config)?;
            let cache =
                cache.unwrap_or_else(|| PathBuf::from(&config.memory.root).join("agents.json"));
            let mut catalog = refresh_agent_catalog(&config.agents, &cache).await?;
            catalog.roles = role_summaries(&load_role_definitions(env::current_dir()?)?);
            println!("{}", serde_json::to_string_pretty(&catalog)?);
        }
    }
    Ok(())
}

async fn hooks(command: HooksCommand) -> Result<()> {
    match command {
        HooksCommand::Install {
            claude: _claude,
            scope,
        } => {
            let report = install_claude_hooks(scope)?;
            print_claude_hooks_report(&report)?;
        }
        HooksCommand::Claude { command } => match command {
            ClaudeHookCommand::SessionStart {
                config,
                root,
                backend,
            } => claude_session_start(config, root, backend).await?,
            ClaudeHookCommand::PreCompact {
                config,
                root,
                backend,
            } => claude_pre_compact(config, root, backend).await?,
        },
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ClaudeHooksInstallReport {
    settings_path: PathBuf,
    skill_path: PathBuf,
    settings_changed: bool,
    skill_created: bool,
}

fn install_claude_hooks(scope: HookScope) -> Result<ClaudeHooksInstallReport> {
    let settings_path = claude_settings_path(scope)?;
    let skill_path = claude_skill_path(scope)?;
    install_claude_hooks_at(settings_path, skill_path)
}

fn install_claude_hooks_at(
    settings_path: PathBuf,
    skill_path: PathBuf,
) -> Result<ClaudeHooksInstallReport> {
    if let Some(parent) = settings_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut root = read_json_object(&settings_path)?;
    let original = root.clone();
    ensure_hook_command(&mut root, "SessionStart", CLAUDE_SESSION_START_COMMAND)?;
    ensure_hook_command(&mut root, "PreCompact", CLAUDE_PRE_COMPACT_COMMAND)?;
    let settings_changed = root != original;
    if settings_changed {
        write_json(&settings_path, &root)?;
    }

    let skill_created = write_claude_artesian_loop_skill(&skill_path)?;
    Ok(ClaudeHooksInstallReport {
        settings_path,
        skill_path,
        settings_changed,
        skill_created,
    })
}

fn print_claude_hooks_report(report: &ClaudeHooksInstallReport) -> Result<()> {
    println!(
        "Claude Code hooks {}: {}",
        if report.settings_changed {
            "written"
        } else {
            "already present"
        },
        report.settings_path.display()
    );
    println!("SessionStart command: {CLAUDE_SESSION_START_COMMAND}");
    println!("PreCompact command: {CLAUDE_PRE_COMPACT_COMMAND}");
    println!(
        "Claude Code skill {}: {}",
        if report.skill_created {
            "written"
        } else {
            "already present"
        },
        report.skill_path.display()
    );
    println!(
        "hooks JSON:\n{}",
        serde_json::to_string_pretty(&claude_hooks_json())?
    );
    println!("skill content:\n{CLAUDE_ARTESIAN_LOOP_SKILL}");
    Ok(())
}

fn claude_settings_path(scope: HookScope) -> Result<PathBuf> {
    match scope {
        HookScope::User => Ok(home_dir()?.join(".claude").join("settings.json")),
        HookScope::Project => Ok(PathBuf::from(".claude").join("settings.json")),
    }
}

fn claude_skill_path(scope: HookScope) -> Result<PathBuf> {
    match scope {
        HookScope::User => Ok(home_dir()?
            .join(".claude")
            .join("skills")
            .join("artesian-loop")
            .join("SKILL.md")),
        HookScope::Project => Ok(PathBuf::from(".claude")
            .join("skills")
            .join("artesian-loop")
            .join("SKILL.md")),
    }
}

fn claude_hooks_json() -> Value {
    json!({
        "hooks": {
            "SessionStart": [
                {
                    "hooks": [
                        command_hook(CLAUDE_SESSION_START_COMMAND)
                    ]
                }
            ],
            "PreCompact": [
                {
                    "hooks": [
                        command_hook(CLAUDE_PRE_COMPACT_COMMAND)
                    ]
                }
            ]
        }
    })
}

fn command_hook(command: &str) -> Value {
    json!({
        "type": "command",
        "command": command
    })
}

fn ensure_hook_command(root: &mut Value, event: &str, command: &str) -> Result<bool> {
    if !root.is_object() {
        *root = json!({});
    }
    let hooks = ensure_object(root, "hooks")?;
    let event_value = hooks
        .entry(event.to_string())
        .or_insert_with(|| Value::Array(Vec::new()));
    let event_groups = event_value
        .as_array_mut()
        .with_context(|| format!("hooks.{event} must be a JSON array"))?;
    if event_groups
        .iter()
        .any(|group| hook_group_has_command(group, command))
    {
        return Ok(false);
    }

    let handler = command_hook(command);
    if let Some(group) = event_groups.iter_mut().find(|group| {
        group
            .as_object()
            .and_then(|object| object.get("matcher"))
            .is_none()
    }) {
        let object = group
            .as_object_mut()
            .with_context(|| format!("hooks.{event} entries must be JSON objects"))?;
        let handlers = object
            .entry("hooks".to_string())
            .or_insert_with(|| Value::Array(Vec::new()))
            .as_array_mut()
            .with_context(|| format!("hooks.{event}[].hooks must be a JSON array"))?;
        handlers.push(handler);
    } else {
        event_groups.push(json!({ "hooks": [handler] }));
    }
    Ok(true)
}

fn hook_group_has_command(group: &Value, command: &str) -> bool {
    group
        .get("hooks")
        .and_then(Value::as_array)
        .is_some_and(|handlers| {
            handlers.iter().any(|handler| {
                handler
                    .get("type")
                    .and_then(Value::as_str)
                    .is_some_and(|hook_type| hook_type == "command")
                    && handler
                        .get("command")
                        .and_then(Value::as_str)
                        .is_some_and(|existing| existing == command)
            })
        })
}

fn write_claude_artesian_loop_skill(path: &Path) -> Result<bool> {
    if path.exists() {
        return Ok(false);
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, CLAUDE_ARTESIAN_LOOP_SKILL)?;
    Ok(true)
}

async fn spawn(
    role: &str,
    agent: &str,
    args: Vec<String>,
    model: Option<String>,
    timeout_seconds: u64,
) -> Result<()> {
    let role = Role::from_str(role)?;
    let cwd = env::current_dir()?;
    let supervisor = ProcessSupervisor::default_for_current_dir();
    let reaped = supervisor.reap_stale()?;
    if reaped.terminated > 0 {
        eprintln!(
            "reaped stale process groups before spawn: terminated={}",
            reaped.terminated
        );
    }
    let request = SpawnRequest {
        role,
        agent: agent.to_string(),
        model,
        working_dir: Some(cwd.display().to_string()),
        resume_packet: None,
    };
    let process = ProcessAgent::new(
        ProcessAgentConfig::new(agent)
            .with_agent_id(agent)
            .with_args(args)
            .with_working_dir(cwd)
            .with_registry_dir(supervisor.registry_dir().to_path_buf())
            .with_timeout(Duration::from_secs(timeout_seconds)),
    );
    let session = process.spawn(request.clone()).await?;
    let response = process
        .send(
            &session,
            artesian_core::AgentMessage {
                content: String::new(),
            },
        )
        .await?;
    println!(
        "spawn completed: role={} agent={} cwd={}",
        request.role.canonical_alias(),
        request.agent,
        request.working_dir.as_deref().unwrap_or(".")
    );
    if !response.content.is_empty() {
        print!("{}", response.content);
    }
    Ok(())
}

async fn run_orchestrator(
    config_path: PathBuf,
    root: Option<PathBuf>,
    dry_run: bool,
    once: bool,
) -> Result<()> {
    let config = load_config(&config_path)?;
    if !matches!(config.mode, Mode::Orchestrate | Mode::Full) {
        bail!(
            "orchestration is disabled for mode {:?}; use orchestrate or full",
            config.mode
        );
    }
    let root = root.unwrap_or_else(|| PathBuf::from(&config.memory.root));
    let repo_root = env::current_dir()?;
    let supervisor = process_supervisor_from_config(&config, &repo_root);
    let reaped = supervisor.reap_stale()?;
    if reaped.terminated > 0 {
        eprintln!(
            "reaped stale process groups before orchestration: terminated={}",
            reaped.terminated
        );
    }
    let mut orchestrator = build_orchestrator(config, root, repo_root, dry_run)?;
    if once {
        let report = tokio::select! {
            report = orchestrator.run_once() => report?,
            signal = shutdown_signal() => {
                let signal = signal?;
                let report = supervisor.terminate_current_owner()?;
                eprintln!(
                    "orchestrator received {signal}; terminated tracked process groups={}",
                    report.terminated
                );
                return Ok(());
            }
        };
        println!(
            "orchestrator tick: dispatched={} completed={} blocked={} idle={}",
            report.dispatched, report.completed, report.blocked, report.idle
        );
    } else {
        let report = tokio::select! {
            report = orchestrator.run_until_idle(100) => report?,
            signal = shutdown_signal() => {
                let signal = signal?;
                let report = supervisor.terminate_current_owner()?;
                eprintln!(
                    "orchestrator received {signal}; terminated tracked process groups={}",
                    report.terminated
                );
                return Ok(());
            }
        };
        println!(
            "orchestrator stopped: ticks={} completed={} blocked={} events={}",
            report.ticks,
            report.completed,
            report.blocked,
            orchestrator.run_log().events.len()
        );
    }
    Ok(())
}

async fn qualify(
    candidate: String,
    goal: Option<String>,
    json_output: bool,
    config: PathBuf,
    root: PathBuf,
    backend: Option<BackendArg>,
) -> Result<()> {
    let acc = load_config(&config)
        .map(|loaded| loaded.acc)
        .unwrap_or_default();
    let backend = open_backend_for_command(&config, root, backend)?;
    let response =
        qualify_memory_candidate(backend.as_ref(), &acc, &candidate, goal.as_deref()).await?;
    if json_output {
        println!("{}", serde_json::to_string_pretty(&response)?);
    } else {
        print_qualify_response(&response);
    }
    Ok(())
}

fn print_qualify_response(response: &QualifyResponse) {
    println!("admitted: {}", response.admitted);
    println!("reason: {}", response.reason);
    if let Some(slot) = &response.slot {
        println!("slot: {slot}");
    }
    println!("score: {:.3}", response.score);
    println!("agreement: {:.3}", response.agreement);
    match response.chance_corrected_agreement {
        Some(value) => println!("chance_corrected_agreement: {value:.3}"),
        None => println!("chance_corrected_agreement: n/a"),
    }
    println!("confidence: {:.3}", response.confidence);
    println!("signals:");
    for signal in &response.signals {
        println!(
            "  {} value={:.3} threshold={:.3} passed={} margin={:.3}",
            signal.name, signal.value, signal.threshold, signal.passed, signal.margin
        );
    }
}

async fn memory(command: MemoryCommand, raw_args: &[OsString]) -> Result<()> {
    match command {
        MemoryCommand::Store {
            content,
            tags,
            node_id,
            source,
            confidence,
            scope,
            agent_id,
            session_id,
            task_id,
            user_id,
            config,
            root,
            backend,
        } => {
            let backend = open_backend_for_command(&config, root, backend)?;
            let record = backend
                .store(StoreMemory {
                    content,
                    tags,
                    metadata: Default::default(),
                    tier: MemoryTier::L1Atom,
                    node_id,
                    created_at: None,
                    scope: scope.map(Into::into),
                    agent_id,
                    session_id,
                    task_id,
                    user_id,
                    source,
                    confidence,
                    relations: Vec::new(),
                })
                .await?;
            println!("stored memory id={} node_id={}", record.id, record.node_id);
        }
        MemoryCommand::Answer {
            question,
            limit,
            config,
            root,
            backend,
        } => {
            let acc = load_config(&config)
                .map(|loaded| loaded.acc)
                .unwrap_or_default();
            let backend = open_backend_for_command(&config, root, backend)?;
            let response =
                artesian_mcp::answer_memory(backend.as_ref(), &acc, &question, limit).await?;
            println!("{}", serde_json::to_string_pretty(&response)?);
        }
        MemoryCommand::Find {
            query,
            limit,
            node_id,
            scope,
            agent_id,
            session_id,
            task_id,
            user_id,
            expand,
            mmr,
            mmr_lambda,
            include_archived,
            config,
            root,
            backend,
        } => {
            // Use memory_config_for_command (instead of the one-shot open_backend_for_command)
            // so we have access to collection + track_savings for savings recording.
            let memory_config = memory_config_for_command(&config, root, backend)?;
            let backend = open_memory_backend(&memory_config)?;
            let diversify = mmr || mmr_lambda.is_some();
            // For MMR, fetch a larger pool so the re-rank has duplicates to shed.
            let fetch_limit = if diversify {
                limit.saturating_mul(3).max(limit)
            } else {
                limit
            };
            let mut memory_query = MemoryQuery::new(query).with_limit(fetch_limit);
            memory_query.node_id = node_id;
            memory_query.scope = scope.map(Into::into);
            memory_query.agent_id = agent_id;
            memory_query.session_id = session_id;
            memory_query.task_id = task_id;
            memory_query.user_id = user_id;
            memory_query.include_archived = include_archived;
            let mut hits = backend.find(memory_query).await?;
            if diversify {
                hits = aquifer::mmr_diversify(
                    hits,
                    limit,
                    mmr_lambda.unwrap_or(aquifer::MMR_DEFAULT_LAMBDA),
                );
            }
            if expand {
                hits = aquifer::expand_hits_with_neighbors(
                    backend.as_ref(),
                    hits,
                    aquifer::DEFAULT_GRAPH_HOPS,
                )
                .await?;
            }
            // ── Token-savings accounting (best-effort, respects track_savings) ──────────
            // CLI `memory find` returns full record content (no truncation), so
            // baseline == returned and saved ≈ 0 — matching the MCP memory.find behaviour.
            // Recording the call still lets `artesian tokens --by-op` show the recall count.
            let baseline_tokens: usize = hits.iter().map(|h| count_tokens(&h.record.content)).sum();
            for hit in &hits {
                println!("{}", format_memory_hit(hit));
            }
            record_savings(
                "memory.find",
                &memory_config.collection,
                baseline_tokens,
                baseline_tokens,
                memory_config.track_savings,
            );
        }
        MemoryCommand::Context {
            query,
            limit,
            index_chars,
            goal,
            expand,
            config,
            root,
            backend,
        } => {
            let memory_config = memory_config_for_command(&config, root, backend)?;
            let index = read_index_slice(&memory_config.root, index_chars)?;
            let backend = open_memory_backend(&memory_config)?;
            // `--goal` returns the bounded goal packet (goal + invariants + relevant memory);
            // otherwise the flat index + hit list.
            if let Some(goal) = goal {
                let recall = loop_recall(backend.as_ref(), &goal).await;
                let packet =
                    assemble_goal_packet(Some(backend.as_ref()), &goal, None, &recall).await;
                println!("{packet}");
                return Ok(());
            }
            let mut hits = backend
                .find(MemoryQuery::new(query).with_limit(limit))
                .await?;
            if expand {
                hits = aquifer::expand_hits_with_neighbors(
                    backend.as_ref(),
                    hits,
                    aquifer::DEFAULT_GRAPH_HOPS,
                )
                .await?;
            }
            // ── Token-savings accounting (best-effort, respects track_savings) ──────────
            // Baseline = full index.md content + full hit record content.
            // Returned = truncated index slice (`index_chars`) + full hit content.
            // The savings come from index truncation when index.md > index_chars characters.
            let index_baseline_tokens = {
                let full_path = PathBuf::from(&memory_config.root)
                    .join("memory")
                    .join("index.md");
                if full_path.exists() {
                    count_tokens(&fs::read_to_string(&full_path).unwrap_or_default())
                } else {
                    0
                }
            };
            let index_returned_tokens = index.as_deref().map(count_tokens).unwrap_or(0);
            // CLI context returns full record content (no per-hit truncation).
            let hits_baseline: usize = hits.iter().map(|h| count_tokens(&h.record.content)).sum();
            if let Some(index) = &index {
                println!("# index.md\n{index}");
            }
            println!("# memory.find");
            for hit in &hits {
                println!("{}", format_memory_hit(hit));
            }
            record_savings(
                "memory.context",
                &memory_config.collection,
                index_returned_tokens + hits_baseline,
                index_baseline_tokens + hits_baseline,
                memory_config.track_savings,
            );
        }
        MemoryCommand::Distill {
            text,
            file,
            reference_date,
            store,
            config,
            root,
            backend,
        } => distill(text, file, reference_date, store, config, root, backend).await?,
        MemoryCommand::Rebuild {
            from,
            config,
            root,
            backend,
        } => {
            let memory_config = memory_config_for_command(&config, root, backend)?;
            let source = from.unwrap_or_else(|| PathBuf::from(&memory_config.root));
            let backend = open_memory_backend(&memory_config)?;
            let report = aquifer::sync_okf_directory(&source, backend.as_ref()).await?;
            println!(
                "rebuilt projection from {}: files_scanned={} records_indexed={} parse_failures={}",
                source.display(),
                report.files_scanned,
                report.records_indexed,
                report.parse_failures
            );
        }
        MemoryCommand::Commit {
            query,
            budget_tokens,
            recall_limit,
            min_score,
            config,
            root,
            backend,
        } => {
            let acc = load_config(&config)
                .map(|loaded| loaded.acc)
                .unwrap_or_default();
            let mut headgate_config = HeadgateConfig::from(&acc);
            if let Some(budget) = budget_tokens {
                headgate_config.budget_tokens = budget;
            }
            if let Some(limit) = recall_limit {
                headgate_config.recall_limit = limit;
            }
            if let Some(score) = min_score {
                headgate_config.min_score = score;
            }
            let budget_tokens = headgate_config.budget_tokens;
            let backend = open_backend_for_command(&config, root, backend)?;
            let recall: Arc<dyn RecallStore> = Arc::new(MemoryRecallStore::new(backend));
            let mut headgate = Headgate::new(recall, headgate_config);
            #[cfg(feature = "llm")]
            {
                if let Some(judge) = &acc.judge {
                    let client = headgate::llm_client_from_config(judge)?;
                    headgate =
                        headgate.with_gate(Arc::new(headgate::JudgeQualifyGate::new(client)));
                }
                if let Some(compressor) = &acc.compressor {
                    let client = headgate::llm_client_from_config(compressor)?;
                    headgate =
                        headgate.with_compressor(Arc::new(headgate::LlmCompressor::new(client)));
                }
            }
            let metrics = headgate.cycle(&query).await?;
            println!("# committed context (budget {budget_tokens} tokens)");
            let rendered = headgate.render();
            if rendered.is_empty() {
                println!("(nothing qualified)");
            } else {
                println!("{rendered}");
            }
            println!("\n# metrics");
            println!("{}", serde_json::to_string_pretty(&metrics)?);
        }
        MemoryCommand::Anchor { command } => anchor(command).await?,
        MemoryCommand::Learn {
            title,
            content,
            from,
            tags,
            steps,
            guards,
            config,
            root,
            backend,
        } => {
            let backend = open_backend_for_command(&config, root, backend)?;
            let procedure = learn_procedure_from_args(steps, guards, raw_args)?;

            // Assemble the skill body from --content text and --from file reads.
            let mut body_parts: Vec<String> = Vec::new();
            if let Some(text) = content {
                body_parts.push(text);
            }
            let mut provenance_sources: Vec<String> = Vec::new();
            for src in &from {
                provenance_sources.push(src.clone());
                // Read local files; URLs are recorded as provenance metadata only (no fetch).
                if !src.starts_with("http://") && !src.starts_with("https://") {
                    let text = fs::read_to_string(src)
                        .with_context(|| format!("read --from file {src}"))?;
                    body_parts.push(text.trim().to_string());
                }
            }

            if body_parts.is_empty() {
                bail!(
                    "`artesian learn` requires --content and/or --from <PATH> \
                     to provide the skill body"
                );
            }

            // Canonical body: title as a heading followed by the assembled content.
            let raw_body = body_parts.join("\n\n");
            let body = format!("# {title}\n\n{raw_body}");

            // Stable node_id for idempotency. A procedure participates in identity so adding
            // guarded replay steps to an existing prose skill creates a procedural variant.
            let identity = artesian_mcp::skill_identity_material(&title, &raw_body, &procedure)?;
            let hash = stable_content_hash(&identity);
            let node_id = format!("skill:{hash}");

            // Source provenance: join multiple --from values; fall back to "artesian-learn".
            let source = if provenance_sources.is_empty() {
                Some("artesian-learn".to_string())
            } else {
                Some(provenance_sources.join(", "))
            };

            // Metadata: explicit title key (for structured listing) + sources list when multiple.
            let mut metadata = BTreeMap::<String, String>::new();
            metadata.insert("title".to_string(), title.clone());
            if provenance_sources.len() > 1 {
                metadata.insert("sources".to_string(), provenance_sources.join(", "));
            }
            insert_skill_procedure_metadata(&mut metadata, &procedure)?;

            // Tags: always include "skill"; append caller-supplied extras.
            let mut all_tags = vec![LOOP_SKILL_TAG.to_string()];
            for t in tags {
                if t != LOOP_SKILL_TAG {
                    all_tags.push(t);
                }
            }

            let record = backend
                .store(StoreMemory {
                    content: body,
                    tags: all_tags,
                    metadata,
                    tier: MemoryTier::L2Scenario,
                    node_id: Some(node_id),
                    created_at: None,
                    scope: None,
                    agent_id: None,
                    session_id: None,
                    task_id: None,
                    user_id: None,
                    source,
                    confidence: None,
                    relations: Vec::new(),
                })
                .await?;
            println!("learned skill id={} node_id={}", record.id, record.node_id);
        }
        MemoryCommand::Skills {
            by_usage,
            limit,
            config,
            root,
            backend,
        } => {
            let backend = open_backend_for_command(&config, root, backend)?;

            // Tag-only query: fetch all active records tagged "skill".
            // Empty query text with a non-empty tag filter returns all tag-matched records
            // with score 1.0 (see files backend score_record behaviour).
            let mut query = MemoryQuery::new("").with_limit(limit);
            query.tags = vec![LOOP_SKILL_TAG.to_string()];
            let mut hits = backend.find(query).await?;

            if by_usage {
                hits.sort_by_key(|h| std::cmp::Reverse(h.record.access_count));
            }

            if hits.is_empty() {
                println!("no skills found");
            } else {
                for hit in &hits {
                    let title = hit
                        .record
                        .metadata
                        .get("title")
                        .cloned()
                        .unwrap_or_else(|| hit.record.node_id.clone());
                    let last = hit
                        .record
                        .last_access
                        .map(|dt| dt.format("%Y-%m-%d").to_string())
                        .unwrap_or_else(|| "-".to_string());
                    let source = hit.record.source.as_deref().unwrap_or("-");
                    println!(
                        "{}\tusage={}\tsource={}\tlast_access={}",
                        title, hit.record.access_count, source, last
                    );
                }
            }
        }
        MemoryCommand::Evict {
            ttl_days,
            lru,
            min_strength,
            max_keep,
            hard,
            dry_run,
            config,
            root,
            backend,
        } => {
            memory_evict(EvictArgs {
                ttl_days,
                lru,
                min_strength,
                max_keep,
                hard,
                dry_run,
                config,
                root,
                backend,
            })
            .await?
        }
        MemoryCommand::Timeline {
            entity,
            limit,
            config,
            root,
            backend,
        } => {
            let backend = open_backend_for_command(&config, root, backend)?;

            // Gather candidates from both the relation graph (by_entity) and text/vector find.
            // Most records are stored without explicit relations, so find() is the primary path.
            let by_entity_records: Vec<MemoryRecord> = backend.by_entity(&entity).await?;
            let fetch_limit = limit.saturating_mul(5).max(50);
            let find_hits = backend
                .find(MemoryQuery::new(entity.clone()).with_limit(fetch_limit))
                .await?;

            // Merge the two result sets; deduplicate by node_id
            let mut seen: std::collections::HashSet<String> = by_entity_records
                .iter()
                .map(|r| r.node_id.clone())
                .collect();
            let mut all_records: Vec<MemoryRecord> = by_entity_records;
            for hit in find_hits {
                if seen.insert(hit.record.node_id.clone()) {
                    all_records.push(hit.record);
                }
            }

            // Filter to records actually mentioning the entity, sort oldest-first
            let timeline: Vec<MemoryRecord> = entity_timeline(&all_records, &entity)
                .into_iter()
                .take(limit)
                .collect();

            if timeline.is_empty() {
                println!("(no records found for entity: {entity})");
            } else {
                println!("timeline for: {entity}  ({} records)", timeline.len());
                println!("{:-<60}", "");
                for record in &timeline {
                    let date = record.created_at.format("%Y-%m-%d %H:%M:%S UTC");
                    let snippet = if record.content.len() > 120 {
                        format!("{}…", &record.content[..120])
                    } else {
                        record.content.clone()
                    };
                    println!("  {date}  [{}]\n    {snippet}", record.node_id);
                }
            }
        }
    }
    Ok(())
}

async fn skill(command: SkillCommand) -> Result<()> {
    match command {
        SkillCommand::Replay {
            title,
            dry_run,
            execute,
            config,
            root,
            backend,
        } => {
            let memory_config = memory_config_for_command(&config, root, backend)?;
            let backend = open_memory_backend(&memory_config)?;
            let response = artesian_mcp::replay_skill_procedure(
                backend.as_ref(),
                &title,
                execute && !dry_run,
                &memory_config.collection,
                memory_config.track_savings,
            )
            .await?;
            print_skill_replay_response(&response);
        }
    }
    Ok(())
}

fn print_skill_replay_response(response: &artesian_mcp::SkillReplayResponse) {
    println!("# skill replay: {}", response.title);
    if let Some(node_id) = &response.node_id {
        println!("node_id={node_id}");
    }
    println!(
        "status={} execute={} fallback={}",
        response.status, response.execute, response.fallback
    );
    println!("{}", response.message);
    for step in &response.steps {
        println!("step {}:", step.index);
        if let Some(guard) = &step.guard {
            println!("  guard: {guard}");
            println!("  guard_status={}", step.guard_status);
            print_replay_output("guard_output", step.guard_output.as_deref());
        } else {
            println!("  guard_status=not-run");
        }
        println!("  run: {}", step.run);
        println!("  run_status={}", step.run_status);
        print_replay_output("run_output", step.run_output.as_deref());
    }
}

fn print_replay_output(label: &str, output: Option<&str>) {
    let Some(output) = output.filter(|output| !output.is_empty()) else {
        return;
    };
    let indented = output.replace('\n', "\n    ");
    println!("  {label}: {indented}");
}

fn learn_procedure_from_args(
    steps: Vec<String>,
    guards: Vec<String>,
    raw_args: &[OsString],
) -> Result<Vec<ProcedureStep>> {
    let procedure = match parse_learn_procedure_from_raw_args(raw_args)? {
        Some(procedure) => procedure,
        None => procedure_from_ordinal_args(steps, guards)?,
    };
    artesian_mcp::normalize_skill_procedure(procedure)
}

fn parse_learn_procedure_from_raw_args(
    raw_args: &[OsString],
) -> Result<Option<Vec<ProcedureStep>>> {
    let Some(start) = raw_args
        .windows(2)
        .position(|pair| pair[0].to_str() == Some("memory") && pair[1].to_str() == Some("learn"))
    else {
        return Ok(None);
    };

    let mut procedure = Vec::<ProcedureStep>::new();
    let mut current_step: Option<usize> = None;
    let mut index = start + 2;
    while index < raw_args.len() {
        let arg = os_arg_to_string(&raw_args[index], "argument")?;
        if arg == "--step" {
            let run = raw_value_after(raw_args, index, "--step")?;
            procedure.push(ProcedureStep::new(run, None));
            current_step = Some(procedure.len() - 1);
            index += 2;
            continue;
        }
        if let Some(run) = arg.strip_prefix("--step=") {
            procedure.push(ProcedureStep::new(run.to_string(), None));
            current_step = Some(procedure.len() - 1);
            index += 1;
            continue;
        }
        if arg == "--guard" {
            let guard = raw_value_after(raw_args, index, "--guard")?;
            let Some(step_index) = current_step else {
                bail!("--guard must follow a preceding --step");
            };
            attach_guard(&mut procedure, step_index, guard)?;
            index += 2;
            continue;
        }
        if let Some(guard) = arg.strip_prefix("--guard=") {
            let Some(step_index) = current_step else {
                bail!("--guard must follow a preceding --step");
            };
            attach_guard(&mut procedure, step_index, guard.to_string())?;
            index += 1;
            continue;
        }
        index += 1;
    }

    if procedure.is_empty() {
        Ok(None)
    } else {
        Ok(Some(procedure))
    }
}

fn procedure_from_ordinal_args(
    steps: Vec<String>,
    guards: Vec<String>,
) -> Result<Vec<ProcedureStep>> {
    if guards.len() > steps.len() {
        bail!("--guard requires a preceding --step");
    }
    Ok(steps
        .into_iter()
        .enumerate()
        .map(|(index, run)| ProcedureStep::new(run, guards.get(index).cloned()))
        .collect())
}

fn attach_guard(procedure: &mut [ProcedureStep], step_index: usize, guard: String) -> Result<()> {
    let step = procedure
        .get_mut(step_index)
        .with_context(|| format!("missing procedure step {}", step_index + 1))?;
    if step.guard.is_some() {
        bail!("procedure step {} already has a guard", step_index + 1);
    }
    step.guard = Some(guard);
    Ok(())
}

fn raw_value_after(raw_args: &[OsString], index: usize, option: &str) -> Result<String> {
    let value = raw_args
        .get(index + 1)
        .with_context(|| format!("{option} requires a value"))?;
    os_arg_to_string(value, option)
}

fn os_arg_to_string(value: &OsString, label: &str) -> Result<String> {
    value
        .to_str()
        .map(str::to_string)
        .with_context(|| format!("{label} is not valid UTF-8"))
}

/// Bundled arguments for the `memory evict` sub-command (avoids a 9-arg function).
struct EvictArgs {
    ttl_days: Option<f32>,
    lru: bool,
    min_strength: Option<f32>,
    max_keep: Option<usize>,
    hard: bool,
    dry_run: bool,
    config: std::path::PathBuf,
    root: PathBuf,
    backend: Option<BackendArg>,
}

async fn memory_evict(args: EvictArgs) -> Result<()> {
    let EvictArgs {
        ttl_days,
        lru,
        min_strength,
        max_keep,
        hard,
        dry_run,
        config,
        root,
        backend,
    } = args;
    // Load all records via the files backend (both active and archived so --hard can delete).
    let memory_config = memory_config_for_command(&config, root.clone(), backend)?;
    let files_root = memory_config.root.clone();
    let files_backend = aquifer::FilesBackend::new(&files_root).with_track_access(false);

    // Collect all records (including archived) for the eviction pass.
    let all_hits = files_backend
        .find({
            let mut q = MemoryQuery::new("").with_limit(usize::MAX);
            q.include_archived = true;
            q
        })
        .await?;
    let all_records: Vec<aquifer::MemoryRecord> = all_hits.into_iter().map(|h| h.record).collect();

    let policy = EvictionPolicy {
        ttl_days,
        lru,
        min_strength,
        max_keep,
        hard,
        decay_config: DecayConfig::default(),
    };

    let report = evict(&all_records, &policy);

    if dry_run {
        println!(
            "dry-run: would archive {} record(s), would delete {} record(s)",
            report.archived, report.deleted
        );
        for entry in &report.log_entries {
            println!(
                "  {:?} id={} node={} strength={:.3} reason={}",
                entry.action,
                entry.record_id,
                entry.node_id,
                entry.retrieval_strength,
                entry.reason
            );
        }
        return Ok(());
    }

    // Apply decisions: update the on-disk records.
    // Build a quick lookup from id → decision action.
    let archive_ids: std::collections::BTreeSet<String> = report
        .log_entries
        .iter()
        .filter(|e| e.action == aquifer::EvictionAction::Archive)
        .map(|e| e.record_id.clone())
        .collect();
    let delete_ids: std::collections::BTreeSet<String> = report
        .log_entries
        .iter()
        .filter(|e| e.action == aquifer::EvictionAction::Delete)
        .map(|e| e.record_id.clone())
        .collect();

    // Walk the memory directory and apply mutations.
    let memory_dir = std::path::PathBuf::from(&files_root).join("memory");
    let mut archived_count = 0usize;
    let mut deleted_count = 0usize;

    if memory_dir.exists() {
        apply_eviction_to_dir(
            &memory_dir,
            &archive_ids,
            &delete_ids,
            &mut archived_count,
            &mut deleted_count,
        )?;
    }

    // Append to the eviction audit log.
    if let Err(error) = append_eviction_log(&report.log_entries) {
        eprintln!("warning: failed to write eviction log: {error}");
    }

    println!(
        "eviction complete: archived={archived_count} deleted={deleted_count} (log: ~/.artesian/eviction.jsonl)"
    );
    Ok(())
}

/// Walk a directory tree and apply archive / delete decisions to `.md` files by their record ID.
fn apply_eviction_to_dir(
    dir: &std::path::Path,
    archive_ids: &std::collections::BTreeSet<String>,
    delete_ids: &std::collections::BTreeSet<String>,
    archived: &mut usize,
    deleted: &mut usize,
) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            apply_eviction_to_dir(&path, archive_ids, delete_ids, archived, deleted)?;
            continue;
        }
        let Some(ext) = path.extension() else {
            continue;
        };
        if ext != "md" {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        // Skip reserved OKF files (index.md, log.md).
        if matches!(
            path.file_name().and_then(|n| n.to_str()).unwrap_or(""),
            "index.md" | "log.md"
        ) {
            continue;
        }
        if delete_ids.contains(stem) {
            std::fs::remove_file(&path)?;
            *deleted += 1;
        } else if archive_ids.contains(stem) {
            // Re-read, set state = Archived, write back.
            let text = std::fs::read_to_string(&path)?;
            if let Ok(mut record) = aquifer::files_parse_record(&text) {
                record.state = MemoryState::Archived;
                if let Ok(rendered) = aquifer::files_render_record(&record) {
                    std::fs::write(&path, rendered)?;
                    *archived += 1;
                }
            }
        }
    }
    Ok(())
}

async fn handoff(
    session_id: String,
    user_id: Option<String>,
    task_id: Option<String>,
    config: PathBuf,
    root: PathBuf,
    backend: Option<BackendArg>,
) -> Result<()> {
    let backend = open_backend_for_command(&config, root, backend)?;
    let key = SessionKey::new(user_id, Some(session_id), task_id);
    let store = SessionStore::new(backend);
    let Some(session) = store.load(&key).await? else {
        bail!(
            "no resumable session for user_id={} session_id={} task_id={}",
            key.user_id,
            key.session_id,
            key.task_id
        );
    };
    let packet = WorkingContextBundle::resume_packet_from_session(&session)?;
    println!("{}", serde_json::to_string_pretty(&packet)?);
    Ok(())
}

#[derive(Debug, Default, Deserialize)]
struct ClaudeHookInput {
    session_id: Option<String>,
    cwd: Option<String>,
}

async fn claude_session_start(
    config: PathBuf,
    root: PathBuf,
    backend: Option<BackendArg>,
) -> Result<()> {
    let input = read_claude_hook_input()?;
    enter_hook_cwd(&input)?;
    let memory = memory_config_for_command(&config, root, backend)?;
    let backend = open_memory_backend(&memory)?;
    let key = SessionKey::new(None, input.session_id.clone(), None);
    let store = SessionStore::new(backend.clone());
    if let Some(session) = store.load(&key).await? {
        let packet = WorkingContextBundle::resume_packet_from_session(&session)?;
        println!("{}", serde_json::to_string_pretty(&packet)?);
        return Ok(());
    }

    let anchor_store = AnchorAnchorStore::new(&memory.root);
    if let Some(recovered) = recover_after_compaction(&anchor_store, backend.as_ref(), 5).await? {
        let bundle = hook_recovery_bundle(&recovered.anchor, &recovered.hits);
        let session = bundle.to_ocf_session(&key, Some("artesian".to_string()))?;
        let packet = WorkingContextBundle::resume_packet_from_session(&session)?;
        println!("{}", serde_json::to_string_pretty(&packet)?);
    }
    Ok(())
}

async fn claude_pre_compact(
    config: PathBuf,
    root: PathBuf,
    backend: Option<BackendArg>,
) -> Result<()> {
    let input = read_claude_hook_input()?;
    enter_hook_cwd(&input)?;
    let memory = memory_config_for_command(&config, root.clone(), backend)?;
    let backend_handle = open_memory_backend(&memory)?;
    let key = SessionKey::new(None, input.session_id.clone(), None);
    let anchor_store = AnchorAnchorStore::new(&memory.root);
    let anchor = anchor_store
        .get_for_session(&key)
        .await?
        .or(anchor_store.get().await?)
        .unwrap_or_else(|| {
            SessionAnchor::new(
                format!("Claude Code session {}", key.session_id),
                "continue after compaction",
            )
        });
    let query_text = format!("{} {}", anchor.current_task, anchor.next_step);
    let hits = backend_handle
        .find(MemoryQuery::new(query_text).with_limit(8))
        .await
        .unwrap_or_default();
    let bundle = hook_recovery_bundle(&anchor, &hits);
    let session = bundle.to_ocf_session(&key, Some("claude-code".to_string()))?;
    let packet = WorkingContextBundle::resume_packet_from_session(&session)?;
    let summary = SessionStore::new(backend_handle).store(session).await?;

    // ── PHASE 1 COMPLETE: synchronous checkpoint always runs first ──────────
    // Print the recovery packet now, before any dream logic, so the compaction
    // boundary always gets its data promptly regardless of what follows.
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "summary": summary,
            "packet": packet
        }))?
    );

    // ── PHASE 2 (OPTIONAL): detached background dream ────────────────────────
    // Gate: config flag `dream_on_compact = true` OR env var ARTESIAN_DREAM_ON_COMPACT=1.
    // The dream is heavy (reads all records + optional LLM) and MUST NOT block the hook
    // or delay the prompt return.  We spawn it fully detached (fire-and-forget) — the
    // Child handle is intentionally dropped immediately.  A dream error never propagates
    // here; the checkpoint already succeeded.
    let env_override = env::var("ARTESIAN_DREAM_ON_COMPACT").ok();
    let dream_enabled = pre_compact_dream_enabled(&config, env_override.as_deref());
    if dream_enabled {
        spawn_detached_dream(&config, &root);
    }

    Ok(())
}

/// Returns `true` when the detached dream should run after the pre-compact checkpoint.
///
/// Precedence (highest first):
/// 1. `env_override = Some("1" | "true" | "yes")` → enabled.
/// 2. `env_override = Some("0" | "false" | "no")` → disabled (explicit opt-out).
/// 3. `dream_on_compact = true` in `artesian.toml` → enabled.
/// 4. Default → disabled.
///
/// The caller is responsible for reading `ARTESIAN_DREAM_ON_COMPACT` from the environment
/// and passing it as `env_override`; this keeps the function pure and easily testable.
fn pre_compact_dream_enabled(config: &Path, env_override: Option<&str>) -> bool {
    match env_override {
        Some("1") | Some("true") | Some("yes") => return true,
        Some("0") | Some("false") | Some("no") => return false,
        _ => {}
    }
    // Fall back to the config file flag.
    if config.exists() {
        if let Ok(text) = fs::read_to_string(config) {
            if let Ok(ac) = ArtesianConfig::from_toml(&text) {
                return ac.dream_on_compact;
            }
        }
    }
    false
}

/// Spawn `artesian dream` as a fully detached background process (fire-and-forget).
///
/// # Coexistence guarantees
///
/// - The synchronous checkpoint has already completed and its output has been printed
///   before this function is called.  A spawn failure here is logged to stderr and
///   silently swallowed — it never affects the checkpoint result or the hook return.
/// - The dream writes to a timestamped subdirectory of `~/.artesian/dreams/` and
///   never mutates the live store, anchor, or any session file.
/// - The child process runs with inherited stdio redirected to `/dev/null` so it
///   cannot interfere with the hook's stdout (which Claude Code has already read).
fn spawn_detached_dream(config: &Path, root: &Path) {
    let ts = chrono::Utc::now().format("%Y%m%dT%H%M%SZ");
    let dreams_base = dirs_dream_base();
    let out_dir = dreams_base.join(ts.to_string());

    // Resolve the current executable so the child reuses the same binary.
    let exe = env::current_exe().unwrap_or_else(|_| PathBuf::from("artesian"));

    let mut cmd = std::process::Command::new(&exe);
    cmd.arg("dream")
        .arg("--config")
        .arg(config)
        .arg("--root")
        .arg(root)
        .arg("--out")
        .arg(&out_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    // Detach on Unix: new process group so the child outlives the hook process.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }

    match cmd.spawn() {
        Ok(_child) => {
            // Drop the Child handle immediately — we do not wait for it.
            eprintln!(
                "artesian: dream-on-compact: spawned background dream → {}",
                out_dir.display()
            );
        }
        Err(e) => {
            // Non-fatal: the checkpoint already succeeded.
            eprintln!("artesian: dream-on-compact: failed to spawn background dream: {e}");
        }
    }
}

/// Default directory for dreams written by the pre-compact hook.
fn dirs_dream_base() -> PathBuf {
    dirs_home()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".artesian")
        .join("dreams")
}

/// Portable home-directory resolution (no extra crate dependency).
fn dirs_home() -> Option<PathBuf> {
    env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

fn read_claude_hook_input() -> Result<ClaudeHookInput> {
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text)?;
    if text.trim().is_empty() {
        return Ok(ClaudeHookInput::default());
    }
    serde_json::from_str(&text).context("parse Claude Code hook input")
}

fn enter_hook_cwd(input: &ClaudeHookInput) -> Result<()> {
    let Some(cwd) = input.cwd.as_deref().filter(|cwd| !cwd.trim().is_empty()) else {
        return Ok(());
    };
    let path = Path::new(cwd);
    if path.is_dir() {
        env::set_current_dir(path).with_context(|| format!("enter Claude hook cwd {cwd}"))?;
    }
    Ok(())
}

fn hook_recovery_bundle(anchor: &SessionAnchor, hits: &[SearchHit]) -> WorkingContextBundle {
    let mut entries = vec![
        SnapshotEntry::now(
            "anchor-current-task",
            "task-state",
            anchor.current_task.clone(),
            1.0,
        ),
        SnapshotEntry::now(
            "anchor-next-step",
            "task-state",
            anchor.next_step.clone(),
            1.0,
        ),
    ];
    if let Some(plan_pointer) = anchor
        .plan_pointer
        .as_ref()
        .filter(|value| !value.is_empty())
    {
        entries.push(SnapshotEntry::now(
            "anchor-plan-pointer",
            "task-state",
            plan_pointer.clone(),
            1.0,
        ));
    }
    for (index, decision) in anchor.last_decisions.iter().enumerate() {
        if decision.trim().is_empty() {
            continue;
        }
        entries.push(SnapshotEntry::now(
            format!("anchor-decision-{index}"),
            "decision",
            decision.clone(),
            1.0,
        ));
    }
    for (index, hit) in hits.iter().enumerate() {
        let mut entry = SnapshotEntry::now(
            format!("memory-hit-{index}"),
            "fact",
            hit.record.content.clone(),
            hit.score,
        );
        entry.unit_ref = Some(hit.record.node_id.clone());
        entries.push(entry);
    }
    let token_count = entries.iter().map(|entry| entry.tokens).sum();
    let lifecycle = entries
        .iter()
        .map(|entry| LifecycleEntry::commit(entry.id.as_str()))
        .collect();
    WorkingContextBundle::new(
        WorkingContextSnapshot {
            schema: vec![
                "decision".to_string(),
                "constraint".to_string(),
                "fact".to_string(),
                "task-state".to_string(),
            ],
            budget_tokens: 4096,
            token_count,
            entries,
        },
        lifecycle,
    )
}

async fn session(command: SessionCommand) -> Result<()> {
    match command {
        SessionCommand::List {
            user_id,
            session_id,
            task_id,
            config,
            root,
            backend,
        } => {
            let backend = open_backend_for_command(&config, root, backend)?;
            let summaries = SessionStore::new(backend)
                .list(SessionListFilter {
                    user_id,
                    session_id,
                    task_id,
                })
                .await?;
            println!("{}", serde_json::to_string_pretty(&summaries)?);
        }
        SessionCommand::Checkpoint {
            user_id,
            session_id,
            task_id,
            agent_id,
            current_task,
            next_step,
            last_decisions,
            plan_pointer,
            last_failed_check,
            config,
            root,
            backend,
        } => {
            let memory = memory_config_for_command(&config, root, backend)?;
            let backend = open_memory_backend(&memory)?;
            let anchor_store = AnchorAnchorStore::new(&memory.root);
            let key = SessionKey::new(user_id, session_id, task_id);
            let request = SessionCheckpointRequest {
                agent_id,
                user_id: Some(key.user_id.clone()),
                session_id: Some(key.session_id.clone()),
                task_id: Some(key.task_id.clone()),
                current_task,
                next_step,
                last_decisions: if last_decisions.is_empty() {
                    None
                } else {
                    Some(last_decisions)
                },
                plan_pointer,
                goal: None,
                last_failed_check,
                limit: None,
            };
            let anchor = checkpoint_anchor_for_cli(&anchor_store, &key, &request).await?;
            let session_hits = session_scoped_hits_for_cli(backend.as_ref(), &key, 8).await?;
            let bundle = build_session_bundle_for_cli(
                Some(&anchor),
                &session_hits,
                &[],
                request.last_failed_check.as_deref(),
            );
            let session = bundle.to_ocf_session(&key, Some(request.agent_id.clone()))?;
            let summary = SessionStore::new(backend).store(session).await?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        SessionCommand::Resume {
            task_query,
            user_id,
            config,
            root,
            backend,
        } => {
            let backend = open_backend_for_command(&config, root, backend)?;
            let all_summaries = SessionStore::new(backend.clone())
                .list(SessionListFilter {
                    user_id,
                    ..SessionListFilter::default()
                })
                .await?;
            let query_lower = task_query.to_lowercase();
            let mut matches: Vec<_> = all_summaries
                .iter()
                .filter(|summary| summary.key.task_id.to_lowercase().contains(&query_lower))
                .collect();
            if matches.is_empty() {
                bail!("no session matched task query {:?}", task_query);
            }
            // Sort by recency descending — most recently updated first.
            matches.sort_by_key(|summary| std::cmp::Reverse(summary.updated_at));
            let best = matches[0];
            if matches.len() > 1 {
                let alternatives: Vec<_> = matches[1..]
                    .iter()
                    .map(|summary| {
                        format!(
                            "  {} / {} (updated {})",
                            summary.key.session_id,
                            summary.key.task_id,
                            summary.updated_at.format("%Y-%m-%dT%H:%M:%SZ")
                        )
                    })
                    .collect();
                eprintln!(
                    "note: {} session(s) matched {:?}; resuming most recent ({} / {}). Alternatives:\n{}",
                    matches.len(),
                    task_query,
                    best.key.session_id,
                    best.key.task_id,
                    alternatives.join("\n")
                );
            }
            let store = SessionStore::new(backend);
            let Some(session) = store.load(&best.key).await? else {
                bail!(
                    "session {} not found (listed but load failed)",
                    best.key.session_id
                );
            };
            let packet = WorkingContextBundle::resume_packet_from_session(&session)?;
            println!("{}", serde_json::to_string_pretty(&packet)?);
        }
    }
    Ok(())
}

fn parse_confidence(value: &str) -> std::result::Result<f32, String> {
    let confidence = value
        .parse::<f32>()
        .map_err(|error| format!("confidence must be a number in 0.0..=1.0: {error}"))?;
    if !(0.0..=1.0).contains(&confidence) {
        return Err(format!(
            "confidence must be within 0.0..=1.0, got {confidence}"
        ));
    }
    Ok(confidence)
}

fn format_memory_hit(hit: &SearchHit) -> String {
    let mut fields = vec![
        format!("{:.4}", hit.score),
        hit.record.id.to_string(),
        hit.record.node_id.clone(),
        hit.record.content.clone(),
    ];
    if let Some(source) = &hit.record.source {
        fields.push(format!("source={source}"));
    }
    if let Some(confidence) = hit.record.confidence {
        fields.push(format!("confidence={confidence}"));
    }
    fields.join("\t")
}

async fn anchor(command: AnchorCommand) -> Result<()> {
    match command {
        AnchorCommand::Get { config, root } => {
            let memory = memory_config_for_command(&config, root, None)?;
            let store = AnchorAnchorStore::new(&memory.root);
            println!("{}", serde_json::to_string_pretty(&store.get().await?)?);
        }
        AnchorCommand::Set {
            current_task,
            next_step,
            plan_pointer,
            last_decisions,
            config,
            root,
        } => {
            let memory = memory_config_for_command(&config, root, None)?;
            let store = AnchorAnchorStore::new(&memory.root);
            let mut anchor = SessionAnchor::new(current_task, next_step);
            anchor.plan_pointer = plan_pointer;
            anchor.last_decisions = last_decisions;
            let written = store.set(anchor).await?;
            println!("{}", serde_json::to_string_pretty(&written)?);
        }
        AnchorCommand::Recover {
            limit,
            config,
            root,
            backend,
        } => {
            let memory = memory_config_for_command(&config, root, backend)?;
            let anchor_store = AnchorAnchorStore::new(&memory.root);
            let backend = open_memory_backend(&memory)?;
            let recovered =
                recover_after_compaction(&anchor_store, backend.as_ref(), limit).await?;
            println!("{}", serde_json::to_string_pretty(&recovered)?);
        }
    }
    Ok(())
}

async fn kit(command: KitCommand) -> Result<()> {
    match command {
        KitCommand::Init { root, vision } => {
            let kit_dir = root.join("kit");
            fs::create_dir_all(&kit_dir).context("create kit directory")?;

            let vision_text = vision
                .as_deref()
                .unwrap_or("(describe the project vision here)");
            let vision_path = kit_dir.join("vision.md");
            if !vision_path.exists() {
                fs::write(
                    &vision_path,
                    format!(
                        "<!-- SPDX-License-Identifier: Apache-2.0 -->\n\n\
# Vision\n\n{vision_text}\n\n\
## Goals\n\n- (list goals here)\n\n\
## Current Phase\n\n- (describe current phase)\n"
                    ),
                )
                .context("write vision.md")?;
            }

            let agents_path = kit_dir.join("agents.md");
            if !agents_path.exists() {
                fs::write(
                    &agents_path,
                    "<!-- SPDX-License-Identifier: Apache-2.0 -->\n\n\
# Agent Roster\n\n\
| Role | Agent | Capabilities |\n\
|---|---|---|\n\
| master | (name) | orchestrate, plan |\n\
| worker | (name) | implement, test |\n\
| judge  | (name) | verify, review |\n",
                )
                .context("write agents.md")?;
            }

            let index_path = kit_dir.join("index.md");
            fs::write(
                &index_path,
                "<!-- SPDX-License-Identifier: Apache-2.0 -->\n\n\
# Loop Memory Kit\n\n\
This kit is the portable anchor-set for this agent loop. Load it at the start of any session.\n\n\
## Contents\n\n\
- [vision.md](vision.md) — project purpose, goals, current phase\n\
- [agents.md](agents.md) — agent roster, roles, capabilities\n\n\
## At session start\n\n\
```sh\n\
artesian memory anchor recover   # restore last anchor + targeted recall\n\
artesian kit status              # print vision + anchor summary\n\
```\n\n\
## Portable across agents\n\n\
This kit works identically in Codex and Claude Code: the MCP tools (`memory.anchor.get`,\n\
`memory.find`) are agent-agnostic. Swap the model; keep the kit.\n",
            )
            .context("write kit/index.md")?;

            println!("kit initialized at {}", kit_dir.display());
            println!("  {}", vision_path.display());
            println!("  {}", agents_path.display());
            println!("  {}", index_path.display());
        }
        KitCommand::Status { root } => {
            let kit_dir = root.join("kit");
            let index_path = kit_dir.join("index.md");
            let vision_path = kit_dir.join("vision.md");
            if vision_path.exists() {
                let text = fs::read_to_string(&vision_path)?;
                let summary: String = text.lines().take(5).collect::<Vec<_>>().join("\n");
                println!("=== vision ===\n{summary}");
            } else {
                println!("kit not initialized — run: artesian kit init");
                return Ok(());
            }
            if index_path.exists() {
                println!("\n=== kit index ===");
                println!("{}", index_path.display());
            }
            let anchor_store = AnchorAnchorStore::new(&root);
            if let Some(anchor) = anchor_store.get().await? {
                println!(
                    "\n=== last anchor ===\ntask: {}\nnext: {}",
                    anchor.current_task, anchor.next_step
                );
            } else {
                println!("\n=== last anchor ===\n(none set)");
            }
        }
        KitCommand::Export {
            root,
            output,
            format,
        } => match format {
            KitFormat::Markdown => {
                let kit_dir = root.join("kit");
                let mut bundle = String::new();
                for name in &["index.md", "vision.md", "agents.md"] {
                    let path = kit_dir.join(name);
                    if path.exists() {
                        bundle.push_str(&format!("# {name}\n\n"));
                        bundle.push_str(&fs::read_to_string(&path)?);
                        bundle.push_str("\n\n---\n\n");
                    }
                }
                let anchor_store = AnchorAnchorStore::new(&root);
                if let Some(anchor) = anchor_store.get().await? {
                    bundle.push_str("# last-anchor\n\n");
                    bundle.push_str(&serde_json::to_string_pretty(&anchor)?);
                    bundle.push('\n');
                }
                match output {
                    Some(path) => {
                        fs::write(&path, &bundle)
                            .with_context(|| format!("write kit bundle to {}", path.display()))?;
                        println!("kit exported to {}", path.display());
                    }
                    None => print!("{bundle}"),
                }
            }
            KitFormat::Bundle => {
                let kit_dir = root.join("kit");
                let anchor = AnchorAnchorStore::new(&root).get().await?;
                let wc_bundle = build_kit_bundle(&kit_dir, anchor.as_ref())?;
                wc_bundle
                    .validate()
                    .context("built an invalid working-context bundle")?;
                let out = output.context("--output <dir> is required for --format bundle")?;
                wc_bundle.write_dir(&out).with_context(|| {
                    format!("write working-context bundle to {}", out.display())
                })?;
                println!(
                    "working-context bundle written to {} ({} entries, {} tokens)",
                    out.display(),
                    wc_bundle.snapshot.entries.len(),
                    wc_bundle.snapshot.token_count
                );
            }
            KitFormat::Ocf => {
                let kit_dir = root.join("kit");
                let anchor = AnchorAnchorStore::new(&root).get().await?;
                let wc_bundle = build_kit_bundle(&kit_dir, anchor.as_ref())?;
                wc_bundle
                    .validate()
                    .context("built an invalid working-context bundle")?;
                let out = output.context("--output <dir> is required for --format ocf")?;
                wc_bundle
                    .write_ocf_dir(&out)
                    .with_context(|| format!("write OCF bundle to {}", out.display()))?;
                println!(
                    "OCF bundle written to {} ({} entries, {} tokens)",
                    out.display(),
                    wc_bundle.snapshot.entries.len(),
                    wc_bundle.snapshot.token_count
                );
            }
        },
        KitCommand::Import { input, root: _root } => {
            let wc_bundle = read_kit_bundle(&input)?;
            println!(
                "=== bundle: {} v{} (units: {}) ===",
                wc_bundle.manifest.format,
                wc_bundle.manifest.version,
                wc_bundle.manifest.unit_source
            );
            println!(
                "committed working context: {} entries, {} / {} tokens, {} lifecycle event(s)\n",
                wc_bundle.snapshot.entries.len(),
                wc_bundle.snapshot.token_count,
                wc_bundle.snapshot.budget_tokens,
                wc_bundle.lifecycle.len()
            );
            println!("{}", wc_bundle.snapshot.render_markdown());
        }
        KitCommand::Validate { input, against } => {
            let wc_bundle = read_kit_bundle(&input)?;
            let mut problems = wc_bundle.schema_issues();
            if let Some(other) = &against {
                let target = read_kit_bundle(other)?;
                problems.extend(
                    wc_bundle.compatibility_issues(
                        &target.snapshot.schema,
                        target.snapshot.budget_tokens,
                    ),
                );
            }
            if problems.is_empty() {
                println!(
                    "OK: {} v{} — {} entries, {} / {} tokens, schema {:?}{}",
                    wc_bundle.manifest.format,
                    wc_bundle.manifest.version,
                    wc_bundle.snapshot.entries.len(),
                    wc_bundle.snapshot.token_count,
                    wc_bundle.snapshot.budget_tokens,
                    wc_bundle.snapshot.schema,
                    if against.is_some() {
                        " — compatible with target"
                    } else {
                        ""
                    }
                );
            } else {
                eprintln!("INCOMPATIBLE ({} issue(s)):", problems.len());
                for problem in &problems {
                    eprintln!("  - {problem}");
                }
                bail!("bundle validation failed");
            }
        }
    }
    Ok(())
}

/// Read a kit bundle from a directory, auto-detecting the OCF four-file layout (`schema.json`
/// present) versus the native working-context bundle layout.
fn read_kit_bundle(dir: &Path) -> Result<WorkingContextBundle> {
    if dir.join("schema.json").exists() {
        WorkingContextBundle::read_ocf_dir(dir)
            .with_context(|| format!("read/validate OCF bundle at {}", dir.display()))
    } else {
        WorkingContextBundle::read_dir(dir)
            .with_context(|| format!("read/validate bundle at {}", dir.display()))
    }
}

/// Build a portable working-context bundle from the on-disk kit (anchors + last session anchor).
/// This is the reference path that turns Artesian's loop memory into the portable layer other
/// runtimes can import; a live ACC session can produce a richer snapshot via
/// `WorkingContextSnapshot::from_ccs`.
fn build_kit_bundle(
    kit_dir: &Path,
    anchor: Option<&SessionAnchor>,
) -> Result<WorkingContextBundle> {
    let mut entries: Vec<SnapshotEntry> = Vec::new();
    for (name, id, slot) in [
        ("vision.md", "vision", "decision"),
        ("agents.md", "agents", "constraint"),
        ("index.md", "index", "task-state"),
    ] {
        let path = kit_dir.join(name);
        if path.exists() {
            let content = fs::read_to_string(&path)
                .with_context(|| format!("read {}", path.display()))?
                .trim()
                .to_string();
            if !content.is_empty() {
                entries.push(SnapshotEntry::now(id, slot, content, 1.0));
            }
        }
    }
    if let Some(anchor) = anchor {
        let task = anchor.current_task.trim();
        if !task.is_empty() {
            entries.push(SnapshotEntry::now("anchor-task", "task-state", task, 1.0));
        }
        let next = anchor.next_step.trim();
        if !next.is_empty() {
            entries.push(SnapshotEntry::now("anchor-next", "task-state", next, 1.0));
        }
    }
    let token_count = entries.iter().map(|entry| entry.tokens).sum();
    let lifecycle = entries
        .iter()
        .map(|entry| LifecycleEntry::commit(entry.id.as_str()))
        .collect();
    let snapshot = WorkingContextSnapshot {
        schema: vec![
            "decision".to_string(),
            "constraint".to_string(),
            "fact".to_string(),
            "task-state".to_string(),
        ],
        budget_tokens: 4096,
        token_count,
        entries,
    };
    Ok(WorkingContextBundle::new(snapshot, lifecycle))
}

async fn backfill(
    directory: PathBuf,
    config: PathBuf,
    root: PathBuf,
    backend: Option<BackendArg>,
    user_id: Option<String>,
    no_link: bool,
    run_consolidate: bool,
) -> Result<()> {
    let artesian_root = root.clone();
    let memory = memory_config_for_command(&config, root, backend)?;
    if memory.backend == MemoryBackendKind::Qdrant {
        preflight_qdrant_memory(&memory).await?;
    }
    // Deterministic relation extraction is on by default: cheap, no LLM, and makes
    // `neighbors`/`by_entity` return links immediately after import.  Pass `--no-link` to opt out.
    let backend = if no_link {
        open_memory_backend(&memory)?
    } else {
        open_memory_backend_with_relations(&memory)?
    };
    let task_store = VectorTaskStore::new(FilesTaskStore::new(&memory.root), backend.clone());
    let report = import_directory(
        ImportOptions {
            directory,
            okf_root: PathBuf::from(&memory.root),
            user_id,
            progress: true,
        },
        backend,
        memory.backend != MemoryBackendKind::Files,
        &task_store,
    )
    .await?;
    let imported = report.memory_imported + report.task_imported;
    let skipped_duplicates = report.memory_skipped_duplicates + report.task_skipped_duplicates;
    let link_note = if no_link {
        " (relation extraction disabled via --no-link)"
    } else {
        " (entity relations extracted)"
    };
    println!(
        "backfill scanned={} imported={} skipped_duplicates={} failed={}{}",
        report.scanned,
        imported,
        skipped_duplicates,
        report.failed.len(),
        link_note,
    );
    println!("{}", serde_json::to_string_pretty(&report)?);
    if run_consolidate {
        println!("running consolidation pass (--consolidate)…");
        consolidate_after_import(&config, artesian_root).await;
    } else {
        println!(
            "next: run `artesian consolidate --allow-llm` to build higher-tier linked memory with an LLM\n\
             (entity relations were extracted automatically; consolidate adds LLM semantic grouping)"
        );
    }
    Ok(())
}

async fn onboard(
    project: String,
    directory: PathBuf,
    options: InitOptions,
    config_path: PathBuf,
    user_id: Option<String>,
    no_link: bool,
    run_consolidate: bool,
) -> Result<()> {
    init(options, true).await?;
    let memory = load_config(&config_path)
        .map(|config| config.memory)
        .unwrap_or_else(|_| {
            let text = fs::read_to_string(DEFAULT_CONFIG).expect("init wrote artesian.toml");
            ArtesianConfig::from_toml(&text)
                .expect("init config should parse")
                .memory
        });
    // Deterministic relation extraction is on by default (cheap, no LLM).  --no-link opts out.
    let backend = if no_link {
        open_memory_backend(&memory)?
    } else {
        open_memory_backend_with_relations(&memory)?
    };
    let task_store = VectorTaskStore::new(FilesTaskStore::new(&memory.root), backend.clone());
    let report = import_directory(
        ImportOptions {
            directory,
            okf_root: PathBuf::from(&memory.root),
            user_id,
            progress: true,
        },
        backend.clone(),
        memory.backend != MemoryBackendKind::Files,
        &task_store,
    )
    .await?;
    let verification_hits = backend
        .find(MemoryQuery::new("").with_limit(1))
        .await
        .map(|hits| hits.len())
        .unwrap_or_default();
    let link_note = if no_link {
        " (relation extraction disabled via --no-link)"
    } else {
        " (entity relations extracted)"
    };
    println!(
        "onboard project={} collection={} root={} verification_hits={}{}",
        project, memory.collection, memory.root, verification_hits, link_note
    );
    println!("{}", serde_json::to_string_pretty(&report)?);
    if run_consolidate {
        println!("running consolidation pass (--consolidate)…");
        // Use the memory root as the artesian root; consolidate reads it from config anyway.
        consolidate_after_import(&config_path, PathBuf::from(&memory.root)).await;
    } else {
        println!(
            "next: run `artesian consolidate --allow-llm` to build higher-tier linked memory with an LLM\n\
             (entity relations were extracted automatically; consolidate adds LLM semantic grouping)"
        );
    }
    Ok(())
}

async fn migrate_okf(
    okf_root: PathBuf,
    config_path: PathBuf,
    new_collection: Option<String>,
    retention_days: u32,
) -> Result<()> {
    let config = load_config(&config_path)?;
    if config.memory.backend != MemoryBackendKind::Qdrant {
        bail!(
            "artesian migrate okf-bundle currently requires backend = qdrant for atomic alias swap"
        );
    }
    let vector_config = VectorMemoryConfig::new(&config.memory.collection);
    let compat = CollectionCompat::from_config(&vector_config);
    let plan = MigrationPlan {
        okf_root,
        alias: config.memory.collection.clone(),
        new_collection: new_collection
            .unwrap_or_else(|| default_migration_collection(&config.memory.collection, &compat)),
        retention_days,
        config: vector_config,
    };
    let report = migrate_qdrant(&config.memory, plan).await?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

async fn migrate_rechunk(config_path: PathBuf) -> Result<()> {
    let config = load_config(&config_path)?;
    if config.memory.backend != MemoryBackendKind::SqliteVec {
        bail!("artesian migrate rechunk currently requires backend = sqlite-vec; for Qdrant use artesian migrate okf-bundle");
    }
    use aquifer::{rechunk_oversized_sqlite, SqliteVecVectorStore, SqliteVecVectorStoreConfig};
    use std::path::PathBuf as SPath;
    let db_path =
        SPath::from(&config.memory.root).join(format!("{}.sqlite", config.memory.collection));
    let store =
        SqliteVecVectorStore::open(SqliteVecVectorStoreConfig::new(&db_path)).map_err(|e| {
            anyhow::anyhow!(
                "failed to open sqlite-vec store at {}: {e}",
                db_path.display()
            )
        })?;
    let backend = store.clone().memory_backend(&config.memory.collection)?;
    let report = rechunk_oversized_sqlite(&store, &backend, &config.memory.collection).await?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    println!(
        "rechunked {}/{} oversized records in collection '{}'",
        report.rechunked, report.oversized, config.memory.collection
    );
    Ok(())
}

async fn snapshot(
    config_path: PathBuf,
    output_dir: PathBuf,
    collection: Option<String>,
) -> Result<()> {
    let config = load_config(&config_path)?;
    if config.memory.backend != MemoryBackendKind::Qdrant {
        bail!(
            "artesian snapshot currently requires backend = qdrant; use artesian okf export for files"
        );
    }
    let collection = collection.unwrap_or_else(|| config.memory.collection.clone());
    let report = snapshot_qdrant(&config.memory, &collection, &output_dir).await?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn okf(command: OkfCommand) -> Result<()> {
    match command {
        OkfCommand::Verify { root } => {
            let report = verify_okf_bundle(root)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        OkfCommand::Export { source, target } => {
            let report = export_okf_bundle(source, target)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
    }
    Ok(())
}

async fn consolidate(config_path: PathBuf, root: PathBuf, allow_llm: bool) -> Result<()> {
    let memory_config = memory_config_for_command(&config_path, root, None)?;
    let memory_root = PathBuf::from(&memory_config.root);
    let log_path = memory_root.join("memory").join("log.md");
    fs::create_dir_all(memory_root.join("memory"))?;

    // --- Consolidation pass: group near-duplicates into canonical QA claim units ---
    let backend = open_memory_backend(&memory_config)?;
    let hits = backend
        .find(MemoryQuery::new("").with_limit(1000))
        .await
        .unwrap_or_default();
    let records: Vec<_> = hits.into_iter().map(|h| h.record).collect();

    let options = ConsolidationOptions::default();
    let report = consolidation_pass(&records, &options);

    println!("# artesian consolidate");
    println!();
    println!("  Input records (sampled): {}", report.input_records);
    println!("  Output canonical claims: {}", report.output_claims);
    println!("  Near-duplicates removed: {}", report.dedup_removed);
    println!(
        "  Token footprint:         {} → {} tokens (estimated)",
        report.footprint_tokens_before, report.footprint_tokens_after
    );
    println!();
    if report.dedup_removed > 0 {
        println!(
            "  {} near-duplicate records can be consolidated. \
             To apply, use `artesian consolidate --apply` when the flag ships.",
            report.dedup_removed
        );
    } else if report.input_records == 0 {
        println!("  No memories found. Run `artesian memory store` to populate.");
    } else {
        println!("  No near-duplicates found at the default similarity threshold (0.6).");
    }

    if allow_llm {
        println!();
        println!(
            "  LLM semantic consolidation is opt-in. Wire a provider in artesian.toml \
             and the consolidation pass will call it to rephrase claims — no-op today."
        );
    }

    // Append a consolidation log entry for audit.
    let mode = if allow_llm {
        "llm-semantic-requested"
    } else {
        "structural"
    };
    let entry = format!(
        "\n- {} consolidate mode={mode} input={} claims={} dedup={}\n",
        chrono_like_timestamp(),
        report.input_records,
        report.output_claims,
        report.dedup_removed,
    );
    fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(&log_path)?
        .write_all(entry.as_bytes())?;

    Ok(())
}

/// Bundle-to-bundle OCF memory consolidation.
///
/// Reads committed records, scores them by access signals (access_count / last_access /
/// creation freshness / content richness), and writes a new OCF bundle to `--out`.
/// The source collection is NEVER mutated. Every admit/reject/merge/supersede/decay
/// decision is logged as a line in `qualify.jsonl`.
///
/// Recommended usage: schedule with cron (`0 3 * * * artesian dream --out …`) or
/// trigger at compaction boundaries.
#[allow(clippy::too_many_arguments)]
async fn dream_command(
    config_path: PathBuf,
    root: PathBuf,
    collection: Option<String>,
    out: PathBuf,
    diary: bool,
    admit_threshold: f32,
    similarity_threshold: f32,
) -> Result<()> {
    let mut memory_config = memory_config_for_command(&config_path, root, None)?;
    // If the caller specified a collection override, apply it before opening the backend.
    if let Some(col) = collection {
        memory_config.collection = col;
    }
    let collection_name = memory_config.collection.clone();

    // Read all records (up to 10 000) from the source backend without mutating them.
    let backend = open_memory_backend(&memory_config)?;
    let hits = backend
        .find(MemoryQuery::new("").with_limit(10_000))
        .await
        .unwrap_or_default();
    let records: Vec<_> = hits.into_iter().map(|h| h.record).collect();

    let record_count = records.len();
    let opts = aquifer::DreamOptions {
        admit_threshold,
        similarity_threshold,
        diary,
        source_label: collection_name.clone(),
        ..Default::default()
    };

    // No LLM merge callback wired in the CLI (opt-in, see artesian.toml [acc.compressor]).
    let result = aquifer::dream(&records, &opts, None)?;

    println!("# artesian dream");
    println!();
    println!("  Source collection: {collection_name}");
    println!("  Records read:      {record_count}");
    println!("  Admitted:          {}", result.admitted);
    println!("  Rejected:          {}", result.rejected);
    println!(
        "  LLM synthesis:     {}",
        if result.llm_ran {
            "yes"
        } else {
            "no (deterministic only)"
        }
    );
    println!();

    aquifer::write_dream_bundle(&result, &opts, &out, &collection_name)?;

    println!("  Bundle written to: {}", out.display());
    for file in [
        "manifest.json",
        "schema.json",
        "snapshot.json",
        "qualify.jsonl",
    ] {
        println!("    {}", out.join(file).display());
    }
    if diary {
        println!("    {}", out.join("DREAMS.md").display());
    }
    println!();
    println!("  Inspect qualify.jsonl for admit/reject/merge/supersede/decay decisions.");
    println!(
        "  To promote this bundle, review and run `artesian memory store` on accepted entries."
    );

    Ok(())
}

/// Run a consolidation pass after import (`--consolidate` flag on backfill/onboard).
///
/// `artesian_root` is the `--root` value (`.artesian` by default), the same parameter that the
/// standalone `artesian consolidate` command accepts.
///
/// Never fails: if the config is missing, the backend can't be opened, or no LLM is wired, we
/// print a clear note and return normally.  The caller (backfill/onboard) must not propagate
/// any error from here — import already succeeded.
async fn consolidate_after_import(config_path: &Path, artesian_root: PathBuf) {
    // Check whether an LLM is configured; if not, print a note and return.
    let acc = load_config(config_path)
        .map(|cfg| cfg.acc)
        .unwrap_or_default();
    let llm_available = acc.compressor.is_some() || acc.judge.is_some();
    if !llm_available {
        println!(
            "note: --consolidate skipped — no LLM configured under [acc.compressor] or \
             [acc.judge] in artesian.toml.\n\
             To enable LLM consolidation, add an endpoint and re-run \
             `artesian consolidate --allow-llm`."
        );
        return;
    }

    // Run the same structural pass as `artesian consolidate --allow-llm`.
    if let Err(error) = consolidate(config_path.to_path_buf(), artesian_root, true).await {
        println!("note: --consolidate pass encountered an error (import succeeded): {error}");
    }
}

fn read_index_slice(root: &str, index_chars: usize) -> Result<Option<String>> {
    let path = PathBuf::from(root).join("memory").join("index.md");
    if !path.exists() {
        return Ok(None);
    }
    let text = fs::read_to_string(path)?;
    Ok(Some(text.chars().take(index_chars).collect()))
}

fn chrono_like_timestamp() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs().to_string())
        .unwrap_or_else(|_| "0".to_string())
}

#[cfg(feature = "qdrant")]
async fn migrate_qdrant(
    memory: &MemoryConfig,
    plan: MigrationPlan,
) -> Result<aquifer::MigrationReport> {
    use aquifer::{migrate_okf_bundle, FastembedTextEmbedder, QdrantVectorStore};

    let store = QdrantVectorStore::connect(qdrant_config(memory)?)?;
    Ok(migrate_okf_bundle(&store, plan, Arc::new(FastembedTextEmbedder::new()?)).await?)
}

#[cfg(not(feature = "qdrant"))]
async fn migrate_qdrant(
    _memory: &MemoryConfig,
    _plan: MigrationPlan,
) -> Result<aquifer::MigrationReport> {
    bail!("artesian migrate requires building artesian-cli with the qdrant feature")
}

#[cfg(feature = "qdrant")]
async fn snapshot_qdrant(
    memory: &MemoryConfig,
    collection: &str,
    output_dir: &Path,
) -> Result<aquifer::SnapshotReport> {
    use aquifer::{QdrantVectorStore, VectorCollectionAdmin};

    let store = QdrantVectorStore::connect(qdrant_config(memory)?)?;
    Ok(store.snapshot_collection(collection, output_dir).await?)
}

#[cfg(not(feature = "qdrant"))]
async fn snapshot_qdrant(
    _memory: &MemoryConfig,
    _collection: &str,
    _output_dir: &Path,
) -> Result<aquifer::SnapshotReport> {
    bail!("artesian snapshot requires building artesian-cli with the qdrant feature")
}

#[cfg(feature = "qdrant")]
fn qdrant_config(memory: &MemoryConfig) -> Result<aquifer::QdrantVectorStoreConfig> {
    let url = memory
        .qdrant_url
        .clone()
        .or_else(|| env::var("QDRANT_URL").ok())
        .context("Qdrant backend requires qdrant_url in config or QDRANT_URL")?;
    let mut config = aquifer::QdrantVectorStoreConfig::new(url);
    config.rest_url = memory
        .qdrant_rest_url
        .clone()
        .or_else(|| env::var("QDRANT_REST_URL").ok());
    config.api_key = memory.resolve_qdrant_api_key();
    Ok(config)
}

#[cfg(feature = "qdrant")]
async fn preflight_qdrant_memory(memory: &MemoryConfig) -> Result<()> {
    let report = aquifer::preflight_qdrant(qdrant_config(memory)?).await?;
    eprintln!(
        "Qdrant preflight ok: grpc={} rest={} version={}",
        report.grpc_url, report.rest_url, report.grpc_version
    );
    Ok(())
}

#[cfg(not(feature = "qdrant"))]
async fn preflight_qdrant_memory(_memory: &MemoryConfig) -> Result<()> {
    bail!("Qdrant preflight requires building artesian-cli with the qdrant feature")
}

async fn preflight_qdrant_options(options: &InitOptions) -> Result<()> {
    let memory = MemoryConfig {
        backend: MemoryBackendKind::Qdrant,
        root: options.memory_root.display().to_string(),
        collection: options.collection.clone(),
        qdrant_url: options.qdrant_url.clone(),
        qdrant_rest_url: options.qdrant_rest_url.clone(),
        qdrant_api_key_env: Some(options.qdrant_api_key_env.clone()),
        qdrant_api_key_file: options.qdrant_api_key_file.clone(),
        local_rerank_enabled: true,
        hyde_enabled: false,
        multi_query_enabled: false,
        debate_enabled: false,
        llm_consolidation_enabled: false,
        semantic_cache: Default::default(),
        track_access: true,
        track_savings: true,
    };
    preflight_qdrant_memory(&memory).await
}

fn open_backend_for_command(
    config_path: &Path,
    root: PathBuf,
    backend: Option<BackendArg>,
) -> Result<Arc<dyn MemoryBackend>> {
    let config = memory_config_for_command(config_path, root, backend)?;
    open_memory_backend(&config)
}

fn memory_config_for_command(
    config_path: &Path,
    root: PathBuf,
    backend: Option<BackendArg>,
) -> Result<MemoryConfig> {
    let config = if config_path.exists() {
        let text = fs::read_to_string(config_path)
            .with_context(|| format!("read {}", config_path.display()))?;
        let config = ArtesianConfig::from_toml(&text)
            .with_context(|| format!("parse {}", config_path.display()))?;
        if config.mode != Mode::Memory {
            bail!("memory commands require mode = memory");
        }
        config.memory
    } else {
        MemoryConfig {
            backend: backend.unwrap_or(BackendArg::Files).into(),
            root: root.display().to_string(),
            collection: "artesian-memory".to_string(),
            qdrant_url: env::var("QDRANT_URL").ok(),
            qdrant_rest_url: env::var("QDRANT_REST_URL").ok(),
            qdrant_api_key_env: Some("QDRANT_API_KEY".to_string()),
            qdrant_api_key_file: None,
            local_rerank_enabled: true,
            hyde_enabled: false,
            multi_query_enabled: false,
            debate_enabled: false,
            llm_consolidation_enabled: false,
            semantic_cache: Default::default(),
            track_access: true,
            track_savings: true,
        }
    };
    let config = if let Some(backend) = backend {
        MemoryConfig {
            backend: backend.into(),
            ..config
        }
    } else {
        config
    };
    Ok(config)
}

fn write_mcp_registrations(config_path: &Path, backend: MemoryBackendKind) -> Result<()> {
    let config_path = config_path
        .canonicalize()
        .unwrap_or_else(|_| config_path.to_path_buf());
    // Qdrant needs QDRANT_API_KEY, which MCP clients do not load from the login
    // shell (they exec the server directly). Register a tiny wrapper that sources
    // the shell rc when the key is absent — the secret never enters an MCP config.
    let command = if backend == MemoryBackendKind::Qdrant {
        write_mcp_wrapper(&home_dir()?)?.display().to_string()
    } else {
        "artesian-mcp".to_string()
    };
    write_claude_mcp(config_path.as_path(), &command)?;
    write_codex_mcp(config_path.as_path(), &command)?;
    write_zed_mcp(config_path.as_path(), &command)?;
    // Project-scoped .mcp.json is not auto-loaded by Claude Code; also register at
    // user scope so the server is available without copying .mcp.json per project.
    register_claude_user_scope(&command, config_path.as_path());
    Ok(())
}

/// Generate `~/artesian/run-artesian-mcp.sh`: a shim the MCP clients launch instead
/// of the bare server, so a missing `QDRANT_API_KEY` is recovered from the user's
/// shell rc. The API key is never written into this file or any MCP config.
fn write_mcp_wrapper(home: &Path) -> Result<PathBuf> {
    let dir = home.join("artesian");
    fs::create_dir_all(&dir)?;
    let path = dir.join("run-artesian-mcp.sh");
    let script = "#!/bin/sh\n\
# Generated by `artesian init`. MCP clients launch this server directly, without\n\
# the user's login shell, so QDRANT_API_KEY may be missing. Load the shell rc when\n\
# absent, then exec the real server. The API key is never stored in this file.\n\
if [ -z \"${QDRANT_API_KEY:-}\" ]; then\n\
  [ -r \"$HOME/.zshrc\" ] && . \"$HOME/.zshrc\" >/dev/null 2>&1 || true\n\
  [ -r \"$HOME/.bashrc\" ] && . \"$HOME/.bashrc\" >/dev/null 2>&1 || true\n\
fi\n\
exec artesian-mcp \"$@\"\n";
    fs::write(&path, script)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755))?;
    }
    Ok(path)
}

fn command_on_path(cmd: &str) -> bool {
    env::var_os("PATH")
        .map(|paths| env::split_paths(&paths).any(|dir| dir.join(cmd).is_file()))
        .unwrap_or(false)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClaudeUserRegistrationStrategy {
    ExternalCli,
    DirectFile,
}

fn claude_user_registration_strategy() -> ClaudeUserRegistrationStrategy {
    let artesian_home = env::var_os("ARTESIAN_HOME").map(PathBuf::from);
    let real_home = env::var_os("HOME").map(PathBuf::from);
    claude_user_registration_strategy_for(
        artesian_home.as_deref(),
        real_home.as_deref(),
        command_on_path("claude"),
    )
}

fn claude_user_registration_strategy_for(
    artesian_home: Option<&Path>,
    real_home: Option<&Path>,
    claude_on_path: bool,
) -> ClaudeUserRegistrationStrategy {
    let sandboxed_home = artesian_home.is_some_and(|home| {
        real_home
            .map(|real_home| paths_resolve_differently(home, real_home))
            .unwrap_or(true)
    });
    if sandboxed_home || !claude_on_path {
        ClaudeUserRegistrationStrategy::DirectFile
    } else {
        ClaudeUserRegistrationStrategy::ExternalCli
    }
}

fn paths_resolve_differently(left: &Path, right: &Path) -> bool {
    resolved_home_path(left) != resolved_home_path(right)
}

fn resolved_home_path(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| {
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            env::current_dir()
                .map(|current_dir| current_dir.join(path))
                .unwrap_or_else(|_| path.to_path_buf())
        }
    })
}

/// Register the server at Claude Code **user scope** (project `.mcp.json` is not
/// auto-loaded). Prefer the `claude` CLI; fall back to editing `~/.claude.json`.
/// Best-effort: never fail `init` if Claude Code is absent.
fn register_claude_user_scope(command: &str, config_path: &Path) {
    let config_arg = config_path.display().to_string();
    // ARTESIAN_HOME is the test/sandbox home. The external `claude` CLI ignores it
    // and writes to the real HOME, so sandboxed runs must use the direct writer.
    if claude_user_registration_strategy() == ClaudeUserRegistrationStrategy::ExternalCli {
        // Idempotent: drop any prior entry, then add fresh.
        let _ = std::process::Command::new("claude")
            .args(["mcp", "remove", "--scope", "user", MCP_SERVER_NAME])
            .output();
        let added = std::process::Command::new("claude")
            .args([
                "mcp",
                "add",
                "--scope",
                "user",
                MCP_SERVER_NAME,
                "--",
                command,
                "--config",
                &config_arg,
            ])
            .status()
            .map(|status| status.success())
            .unwrap_or(false);
        if added {
            return;
        }
    }
    let _ = write_claude_user_json(command, config_path);
}

fn write_claude_user_json(command: &str, config_path: &Path) -> Result<()> {
    let path = home_dir()?.join(".claude.json");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut root = read_json_object(&path)?;
    let server = json!({
        "command": command,
        "args": mcp_args(config_path),
        "env": { "ARTESIAN_MCP_TOOL_HINT": MCP_TOOL_HINT }
    });
    ensure_object(&mut root, "mcpServers")?.insert(MCP_SERVER_NAME.to_string(), server);
    write_json(&path, &root)
}

fn mcp_args(config_path: &Path) -> Vec<String> {
    vec!["--config".to_string(), config_path.display().to_string()]
}

fn write_claude_mcp(config_path: &Path, command: &str) -> Result<()> {
    let path = Path::new(".mcp.json");
    let mut root = read_json_object(path)?;
    let server = json!({
        "command": command,
        "args": mcp_args(config_path),
        "env": {
            "ARTESIAN_MCP_TOOL_HINT": MCP_TOOL_HINT
        }
    });
    ensure_object(&mut root, "mcpServers")?.insert(MCP_SERVER_NAME.to_string(), server);
    write_json(path, &root)
}

fn write_codex_mcp(config_path: &Path, command: &str) -> Result<()> {
    let path = home_dir()?.join(".codex").join("config.toml");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let text = fs::read_to_string(&path).unwrap_or_default();
    let mut document = text.parse::<DocumentMut>().unwrap_or_default();
    ensure_toml_table(&mut document, "mcp_servers");
    document["mcp_servers"][MCP_SERVER_NAME]["command"] = value(command);
    let mut args = Array::new();
    for arg in mcp_args(config_path) {
        args.push(arg);
    }
    document["mcp_servers"][MCP_SERVER_NAME]["args"] = value(args);
    document["mcp_servers"][MCP_SERVER_NAME]["env"]["ARTESIAN_MCP_TOOL_HINT"] =
        value(MCP_TOOL_HINT);
    fs::write(path, document.to_string())?;
    Ok(())
}

fn write_zed_mcp(config_path: &Path, command: &str) -> Result<()> {
    let path = zed_settings_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut root = read_json_object(&path)?;
    let server = json!({
        "command": {
            "path": command,
            "args": mcp_args(config_path),
            "env": {
                "ARTESIAN_MCP_TOOL_HINT": MCP_TOOL_HINT
            }
        }
    });
    ensure_object(&mut root, "context_servers")?.insert(MCP_SERVER_NAME.to_string(), server);
    write_json(&path, &root)
}

fn read_json_object(path: &Path) -> Result<Value> {
    if !path.exists() {
        return Ok(json!({}));
    }
    let value = serde_json::from_str(&fs::read_to_string(path)?)
        .with_context(|| format!("parse {}", path.display()))?;
    Ok(value)
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    fs::write(path, serde_json::to_string_pretty(value)? + "\n")?;
    Ok(())
}

fn ensure_object<'a>(
    root: &'a mut Value,
    key: &str,
) -> Result<&'a mut serde_json::Map<String, Value>> {
    if !root.is_object() {
        *root = json!({});
    }
    let object = root.as_object_mut().expect("root object ensured");
    object.entry(key).or_insert_with(|| json!({}));
    object
        .get_mut(key)
        .and_then(Value::as_object_mut)
        .with_context(|| format!("{key} must be a JSON object"))
}

fn ensure_toml_table(document: &mut DocumentMut, key: &str) {
    if !document.as_table().contains_key(key) {
        document[key] = Item::Table(Table::new());
    }
}

pub(crate) fn home_dir() -> Result<PathBuf> {
    if let Some(home) = env::var_os("ARTESIAN_HOME").or_else(|| env::var_os("HOME")) {
        return Ok(PathBuf::from(home));
    }
    bail!("HOME is not set")
}

fn zed_settings_path() -> Result<PathBuf> {
    let home = home_dir()?;
    let app_support = home
        .join("Library")
        .join("Application Support")
        .join("Zed")
        .join("settings.json");
    if app_support.parent().is_some_and(|parent| parent.exists()) {
        Ok(app_support)
    } else {
        Ok(home.join(".config").join("zed").join("settings.json"))
    }
}

fn detect_agents() -> Vec<AgentBinding> {
    let detected = [
        "claude",
        "claude-code",
        "codex",
        "gemini",
        "opencode",
        "ollama",
    ]
    .into_iter()
    .filter(|name| agent_detected(name))
    .map(str::to_string)
    .collect::<Vec<_>>();

    let master = pick(&detected, &["claude-code", "claude", "codex"]);
    let worker = pick(&detected, &["codex", "opencode", "claude"]);
    let judge = pick(&detected, &["claude-code", "claude", "gemini", "codex"]);

    [
        (Role::Master, master),
        (Role::Worker, worker),
        (Role::Judge, judge),
    ]
    .into_iter()
    .filter_map(|(role, agent)| {
        agent.map(|agent| AgentBinding {
            role,
            command: Some(agent.clone()),
            agent,
            model: None,
            args: Vec::new(),
            timeout_seconds: Some(120),
        })
    })
    .collect()
}

fn agent_detected(name: &str) -> bool {
    command_exists(name)
        || agent_config_locations(name)
            .iter()
            .any(|path| path.exists())
}

fn agent_config_locations(name: &str) -> Vec<PathBuf> {
    let Ok(home) = home_dir() else {
        return Vec::new();
    };
    match name {
        "claude" | "claude-code" => vec![home.join(".claude")],
        "codex" => vec![home.join(".codex")],
        "gemini" => vec![home.join(".gemini"), home.join(".config").join("gemini")],
        "opencode" => vec![home.join(".config").join("opencode")],
        "ollama" => vec![home.join(".ollama")],
        _ => Vec::new(),
    }
}

fn pick(detected: &[String], preferred: &[&str]) -> Option<String> {
    preferred
        .iter()
        .find_map(|candidate| detected.iter().find(|agent| agent == candidate).cloned())
}

fn command_exists(command: &str) -> bool {
    let Some(path) = env::var_os("PATH") else {
        return false;
    };
    env::split_paths(&path).any(|dir| dir.join(command).is_file())
}

/// Print cumulative token-savings statistics collected by Artesian's targeted recall.
///
/// The baseline assumption is documented in `docs/token-savings.md`: `baseline_tokens` is the
/// sum of `count_tokens(record.content)` for each source record that contributed a hit.
/// `saved_tokens = max(0, baseline_tokens - returned_tokens)`.
fn tokens_command(json: bool, since: Option<String>, by_op: bool) -> Result<()> {
    let since_dt = since
        .as_deref()
        .map(|s| {
            chrono::DateTime::parse_from_rfc3339(s)
                .with_context(|| format!("invalid --since timestamp: {s}"))
                .map(|dt| dt.with_timezone(&chrono::Utc))
        })
        .transpose()?;

    let rollup = load_savings_rollup(since_dt);

    if json {
        println!("{}", serde_json::to_string_pretty(&rollup)?);
        return Ok(());
    }

    if rollup.calls == 0 {
        println!("No token-savings data recorded yet. Run some recalls first.");
        return Ok(());
    }

    let pct = if rollup.baseline_total > 0 {
        100.0 * rollup.saved_total as f64 / rollup.baseline_total as f64
    } else {
        0.0
    };
    println!(
        "Artesian saved ~{} tokens across {} recalls (~{:.0}% vs loading the source records)",
        rollup.saved_total, rollup.calls, pct
    );

    if by_op && !rollup.by_op.is_empty() {
        println!();
        println!("By operation:");
        let mut ops: Vec<_> = rollup.by_op.iter().collect();
        ops.sort_by_key(|(name, _)| name.as_str());
        for (op, stats) in ops {
            let op_pct = if stats.baseline_total > 0 {
                100.0 * stats.saved_total as f64 / stats.baseline_total as f64
            } else {
                0.0
            };
            println!(
                "  {op}: saved {saved} tokens / {calls} recalls ({pct:.0}%)",
                saved = stats.saved_total,
                calls = stats.calls,
                pct = op_pct,
            );
        }
    }

    Ok(())
}

fn quota_command(json: bool) -> Result<()> {
    let statuses = read_local_quota();
    if json {
        println!("{}", serde_json::to_string_pretty(&statuses)?);
        return Ok(());
    }

    println!(
        "{:<8} {:<8} {:<9} {:>7} {:>11} source",
        "agent", "window", "status", "used", "resets"
    );
    for status in statuses {
        let used = status
            .pct
            .map(|pct| format!("{pct:.1}%"))
            .unwrap_or_else(|| "-".to_string());
        let resets = status
            .resets_at
            .map(|value| value.format("%Y-%m-%dT%H:%MZ").to_string())
            .unwrap_or_else(|| "-".to_string());
        let source = status
            .source
            .as_ref()
            .map(|path| path.display().to_string())
            .or(status.message)
            .unwrap_or_else(|| "-".to_string());
        let status_label = match status.status {
            QuotaStatusKind::Known => "known",
            QuotaStatusKind::Unknown => "unknown",
        };
        println!(
            "{:<8} {:<8} {:<9} {:>7} {:>11} {}",
            status.agent, status.window, status_label, used, resets, source
        );
    }
    Ok(())
}

async fn perf(config_path: PathBuf, root: PathBuf, budget_tokens: Option<usize>) -> Result<()> {
    let memory_config = memory_config_for_command(&config_path, root.clone(), None)?;
    let backend = open_memory_backend(&memory_config)?;

    // --- (1) Estimate full-replay cost ---
    let hits = backend
        .find(MemoryQuery::new("").with_limit(2000))
        .await
        .unwrap_or_default();

    let total_tokens: usize = hits.iter().map(|h| count_tokens(&h.record.content)).sum();
    let stored_count = hits.len();

    // --- (2) CCS budget ---
    let ccs_budget = budget_tokens.unwrap_or(2048);
    let saving_pct = if total_tokens > 0 {
        let saved = total_tokens.saturating_sub(ccs_budget);
        100.0 * saved as f64 / total_tokens as f64
    } else {
        0.0
    };

    println!("# artesian perf");
    println!();
    println!("  Stored memories (sampled): {stored_count}");
    println!("  Estimated full-replay cost: {total_tokens} tokens");
    println!("  ACC committed-context budget: {ccs_budget} tokens");
    if total_tokens > ccs_budget {
        println!("  Saving vs full replay: {saving_pct:.1}%");
    } else if total_tokens == 0 {
        println!("  (No memories stored yet — run `artesian memory store` to populate.)");
    } else {
        println!("  Memory fits in CCS budget ({total_tokens} ≤ {ccs_budget} tokens).");
    }
    println!();

    // --- (3) Compaction-survival check ---
    println!("## Compaction-survival check");
    let temp_dir = std::env::temp_dir().join(format!("artesian-perf-{}", std::process::id()));
    std::fs::create_dir_all(&temp_dir)?;
    let anchor_store = AnchorAnchorStore::from_log_path(temp_dir.join("perf-log.md"));

    let test_anchor = SessionAnchor::new(
        "perf-check: compaction-survival probe",
        "verify: plan and next-step are intact after simulated compaction",
    );
    anchor_store.set(test_anchor.clone()).await?;

    // Simulate compaction: fresh recovery from the written anchor.
    let context = recover_after_compaction(&anchor_store, backend.as_ref(), 3).await?;
    std::fs::remove_dir_all(&temp_dir).ok();

    match context {
        Some(ctx) if ctx.anchor.current_task == test_anchor.current_task => {
            println!("  PASS  plan + next-step intact after simulated compaction");
            println!("  anchor.current_task = {:?}", ctx.anchor.current_task);
            println!("  anchor.next_step    = {:?}", ctx.anchor.next_step);
        }
        Some(ctx) => {
            anyhow::bail!(
                "compaction-survival FAIL: recovered task {:?} ≠ expected {:?}",
                ctx.anchor.current_task,
                test_anchor.current_task
            );
        }
        None => {
            anyhow::bail!("compaction-survival FAIL: no anchor found after write + recover");
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use flume::loop_core::{
        LOOP_INVARIANT_TAG as INVARIANT_TAG, LOOP_SKILL_TAG as SKILL_TAG, LOOP_SPEC_TAG as SPEC_TAG,
    };
    use std::collections::VecDeque;

    #[test]
    fn qdrant_auth_failure_diagnostic_names_key_env_without_rebuild_hint() {
        let config = MemoryConfig {
            backend: MemoryBackendKind::Qdrant,
            root: ".artesian".to_string(),
            collection: "artesian-memory".to_string(),
            qdrant_url: Some("http://127.0.0.1:6334".to_string()),
            qdrant_rest_url: None,
            qdrant_api_key_env: Some("QDRANT__SERVICE__API_KEY".to_string()),
            qdrant_api_key_file: Some("~/.macray/qdrant.env".to_string()),
            local_rerank_enabled: true,
            hyde_enabled: false,
            multi_query_enabled: false,
            debate_enabled: false,
            llm_consolidation_enabled: false,
            semantic_cache: Default::default(),
            track_access: true,
            track_savings: true,
        };

        let diagnostic = qdrant_backend_diagnostic_with_key_state(
            &config,
            "backend error: status: Unauthenticated, message: invalid API key",
            false,
        )
        .expect("auth failures should produce a Qdrant diagnostic");

        assert!(diagnostic.summary.contains("QDRANT__SERVICE__API_KEY"));
        assert!(diagnostic.fix.contains("QDRANT__SERVICE__API_KEY"));
        assert!(diagnostic.fix.contains("qdrant_api_key_file"));
        assert!(!diagnostic.summary.contains("not available in this build"));
        assert!(!diagnostic.fix.contains("memory rebuild"));
    }

    #[test]
    fn claude_registration_strategy_uses_file_for_sandboxed_artesian_home() {
        let tmp = temp_path("artesian-claude-home-strategy");
        let real_home = tmp.join("real-home");
        let artesian_home = tmp.join("artesian-home");
        fs::create_dir_all(&real_home).expect("create real home");
        fs::create_dir_all(&artesian_home).expect("create artesian home");

        assert_eq!(
            claude_user_registration_strategy_for(Some(&artesian_home), Some(&real_home), true),
            ClaudeUserRegistrationStrategy::DirectFile
        );

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn claude_registration_strategy_prefers_cli_without_artesian_home() {
        assert_eq!(
            claude_user_registration_strategy_for(None, Some(Path::new("/home/user")), true),
            ClaudeUserRegistrationStrategy::ExternalCli
        );
    }

    struct ScriptedLoopCommands {
        verify_results: VecDeque<(bool, String)>,
        worker_results: VecDeque<bool>,
        worker_calls: u32,
    }

    impl ScriptedLoopCommands {
        fn new<I, S>(verify_results: I) -> Self
        where
            I: IntoIterator<Item = (bool, S)>,
            S: Into<String>,
        {
            Self {
                verify_results: verify_results
                    .into_iter()
                    .map(|(passed, output)| (passed, output.into()))
                    .collect(),
                worker_results: VecDeque::new(),
                worker_calls: 0,
            }
        }

        fn with_workers(mut self, worker_results: impl IntoIterator<Item = bool>) -> Self {
            self.worker_results = worker_results.into_iter().collect();
            self
        }
    }

    impl LoopCommands for ScriptedLoopCommands {
        fn run_worker<'a>(
            &'a mut self,
            _cmd: &'a str,
            _env: Vec<(String, String)>,
            _timeout: Option<Duration>,
        ) -> LoopCommandFuture<'a, bool> {
            self.worker_calls += 1;
            let result = self.worker_results.pop_front().unwrap_or(true);
            Box::pin(async move { Ok(result) })
        }

        fn verify_goal<'a>(
            &'a mut self,
            _cmd: &'a str,
            _timeout: Option<Duration>,
        ) -> LoopCommandFuture<'a, (bool, String)> {
            let result = self
                .verify_results
                .pop_front()
                .unwrap_or((false, "script exhausted".to_string()));
            Box::pin(async move { Ok(result) })
        }
    }

    fn loop_options(tmp: &Path, run_id: &str) -> LoopRunOptions {
        LoopRunOptions {
            goal: "cargo test".to_string(),
            worker_cmd: Some("fix-it".to_string()),
            max_turns: 2,
            max_wall: None,
            poll: false,
            learn: true,
            run_id: run_id.to_string(),
            run_log_dir: tmp.join("runs"),
            stop_file: tmp.join("STOP"),
            collection: String::new(),
            track_savings: false,
            max_remediation_attempts: flume::loop_core::LOOP_REMEDIATION_ATTEMPTS_DEFAULT,
            cancel: Default::default(),
            on_progress: None,
            quota: QuotaLoopConfig {
                reader: flume::quota::QuotaReadOptions {
                    codex_home: Some(tmp.join("missing-codex")),
                    claude_roots: vec![tmp.join("missing-claude")],
                },
                ..QuotaLoopConfig::default()
            },
        }
    }

    async fn run_scripted_loop(
        tmp: &Path,
        run_id: &str,
        backend: Option<&dyn MemoryBackend>,
        commands: &mut ScriptedLoopCommands,
        mut options: LoopRunOptions,
    ) -> anyhow::Result<flume::loop_core::LoopRunReport> {
        options.run_id = run_id.to_string();
        options.run_log_dir = tmp.join("runs");
        options.stop_file = tmp.join("STOP");
        let anchor_store = AnchorAnchorStore::new(tmp.join("anchors"));
        run_loop_core(options, backend, &anchor_store, commands).await
    }

    #[tokio::test]
    async fn loop_run_log_writes_turn_and_summary_jsonl() {
        let tmp = temp_path("artesian-loop-log");
        fs::create_dir_all(&tmp).expect("tmp should be created");
        let mut commands =
            ScriptedLoopCommands::new([(false, "initial fail"), (false, "still failing")])
                .with_workers([true]);
        let mut options = loop_options(&tmp, "run-log");
        options.max_turns = 1;
        options.learn = false;

        let result = run_scripted_loop(&tmp, "run-log", None, &mut commands, options)
            .await
            .expect("run_scripted_loop should not error");
        assert_eq!(result.outcome, "max-turns", "loop should stop at max-turns");
        let log_path = tmp.join("runs").join("run-log.jsonl");
        let lines: Vec<String> = fs::read_to_string(&log_path)
            .expect("run-log should exist")
            .lines()
            .map(str::to_string)
            .collect();
        assert_eq!(lines.len(), 2);
        let turn: Value = serde_json::from_str(&lines[0]).expect("turn should be JSON");
        assert_eq!(turn["type"], "turn");
        assert_eq!(turn["turn"], 1);
        assert_eq!(turn["verify_result"], "failed");
        let summary: Value = serde_json::from_str(&lines[1]).expect("summary should be JSON");
        assert_eq!(summary["type"], "summary");
        assert_eq!(summary["outcome"], "max-turns");
        assert_eq!(summary["turns"], 1);
        let _ = fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn loop_success_stores_verified_skill_and_spec() {
        let tmp = temp_path("artesian-loop-learn");
        fs::create_dir_all(&tmp).expect("tmp should be created");
        let backend = aquifer::FilesBackend::new(tmp.join("memory"));
        let mut commands =
            ScriptedLoopCommands::new([(false, "initial fail"), (true, "ok")]).with_workers([true]);
        let options = loop_options(&tmp, "learn-success");

        let report = run_scripted_loop(
            &tmp,
            "learn-success",
            Some(&backend),
            &mut commands,
            options,
        )
        .await
        .expect("loop should succeed");
        assert_eq!(report.turns, 1);
        assert!(report.run_log_path.exists());

        let mut skill_query = MemoryQuery::new("cargo test").with_limit(10);
        skill_query.tags = vec![SKILL_TAG.to_string()];
        let skills = backend.find(skill_query).await.expect("find skills");
        assert!(skills
            .iter()
            .any(|hit| hit.record.content.contains("verified approach")));

        let mut spec_query = MemoryQuery::new("cargo test").with_limit(10);
        spec_query.tags = vec![SPEC_TAG.to_string()];
        let specs = backend.find(spec_query).await.expect("find specs");
        assert!(specs
            .iter()
            .any(|hit| hit.record.content.contains("sharper spec")));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn loop_corrected_failed_check_stores_deduplicated_invariant() {
        let tmp = temp_path("artesian-loop-invariant");
        fs::create_dir_all(&tmp).expect("tmp should be created");
        let backend = aquifer::FilesBackend::new(tmp.join("memory"));

        for run_id in ["invariant-one", "invariant-two"] {
            let mut commands = ScriptedLoopCommands::new([
                (false, "initial fail"),
                (false, "missing required fixture"),
                (true, "ok"),
            ])
            .with_workers([true, true]);
            let options = loop_options(&tmp, run_id);
            run_scripted_loop(&tmp, run_id, Some(&backend), &mut commands, options)
                .await
                .expect("loop should succeed after correction");
        }

        let mut invariant_query = MemoryQuery::new("missing required fixture").with_limit(10);
        invariant_query.tags = vec![INVARIANT_TAG.to_string()];
        let invariants = backend
            .find(invariant_query)
            .await
            .expect("find invariants");
        let learned: Vec<_> = invariants
            .iter()
            .filter(|hit| hit.record.tags.iter().any(|tag| tag == "auto-invariant"))
            .collect();
        assert_eq!(learned.len(), 1);
        assert!(learned[0].record.content.contains("turn 1"));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn claude_hooks_install_merges_and_is_idempotent() {
        let tmp = temp_path("artesian-claude-hooks");
        let settings = tmp.join(".claude").join("settings.json");
        let skill = tmp
            .join(".claude")
            .join("skills")
            .join("artesian-loop")
            .join("SKILL.md");
        fs::create_dir_all(settings.parent().expect("settings parent")).expect("create settings");
        fs::write(
            &settings,
            serde_json::to_string_pretty(&json!({
                "theme": "dark",
                "hooks": {
                    "PostToolUse": [
                        {
                            "matcher": "Edit|Write",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "prettier --write"
                                }
                            ]
                        }
                    ],
                    "SessionStart": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "echo existing"
                                }
                            ]
                        }
                    ]
                }
            }))
            .expect("settings should encode"),
        )
        .expect("settings should write");

        let first =
            install_claude_hooks_at(settings.clone(), skill.clone()).expect("hooks should install");
        assert!(first.settings_changed);
        assert!(first.skill_created);
        let value: Value =
            serde_json::from_str(&fs::read_to_string(&settings).expect("settings should read"))
                .expect("settings should parse");
        assert_eq!(value["theme"], "dark");
        assert_eq!(
            count_hook_command(&value, "PostToolUse", "prettier --write"),
            1
        );
        assert_eq!(
            count_hook_command(&value, "SessionStart", CLAUDE_SESSION_START_COMMAND),
            1
        );
        assert_eq!(
            count_hook_command(&value, "PreCompact", CLAUDE_PRE_COMPACT_COMMAND),
            1
        );

        let second = install_claude_hooks_at(settings.clone(), skill.clone())
            .expect("second install should be clean");
        assert!(!second.settings_changed);
        assert!(!second.skill_created);
        let rerun: Value = serde_json::from_str(
            &fs::read_to_string(&settings).expect("settings should read after rerun"),
        )
        .expect("settings should parse after rerun");
        assert_eq!(value, rerun);
        assert_eq!(
            count_hook_command(&rerun, "SessionStart", CLAUDE_SESSION_START_COMMAND),
            1
        );
        assert_eq!(
            count_hook_command(&rerun, "PreCompact", CLAUDE_PRE_COMPACT_COMMAND),
            1
        );
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn claude_artesian_loop_skill_has_valid_frontmatter() {
        let tmp = temp_path("artesian-claude-skill");
        let skill = tmp.join("SKILL.md");
        assert!(write_claude_artesian_loop_skill(&skill).expect("skill should write"));
        let text = fs::read_to_string(&skill).expect("skill should read");
        let (header, body) = text
            .strip_prefix("---\n")
            .and_then(|rest| rest.split_once("\n---\n"))
            .expect("skill should have YAML frontmatter");
        assert!(header.lines().any(|line| line == "name: artesian-loop"));
        assert!(header.lines().any(|line| line.starts_with("description: ")));
        assert!(body.contains("memory.session.resume"));
        assert!(!write_claude_artesian_loop_skill(&skill).expect("existing skill should stay"));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn mcp_wrapper_sources_shell_rc_and_execs_server() {
        let tmp = std::env::temp_dir().join(format!("artesian-wrapper-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        let path = write_mcp_wrapper(&tmp).expect("write wrapper");
        assert_eq!(path, tmp.join("artesian").join("run-artesian-mcp.sh"));
        let body = fs::read_to_string(&path).expect("read wrapper");
        assert!(
            body.contains("exec artesian-mcp \"$@\""),
            "wrapper must exec the real server: {body}"
        );
        assert!(
            body.contains("QDRANT_API_KEY"),
            "wrapper must recover the key from the shell rc"
        );
        assert!(
            body.contains(".zshrc"),
            "wrapper must source the login shell rc"
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&path).expect("stat").permissions().mode();
            assert_eq!(mode & 0o777, 0o755, "wrapper must be executable");
        }
        let _ = fs::remove_dir_all(&tmp);
    }

    fn count_hook_command(value: &Value, event: &str, command: &str) -> usize {
        value["hooks"][event]
            .as_array()
            .into_iter()
            .flatten()
            .flat_map(|group| {
                group
                    .get("hooks")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
            })
            .filter(|handler| {
                handler.get("type").and_then(Value::as_str) == Some("command")
                    && handler.get("command").and_then(Value::as_str) == Some(command)
            })
            .count()
    }

    fn temp_path(name: &str) -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("{name}-{}-{unique}", std::process::id()))
    }

    // ── dream-on-compact unit tests ──────────────────────────────────────────

    /// Helper: write a minimal artesian.toml with `dream_on_compact` set.
    fn write_compact_config(dir: &Path, dream_on_compact: bool) -> PathBuf {
        let config = dir.join("artesian.toml");
        let toml = format!(
            r#"mode = "memory"
dream_on_compact = {dream_on_compact}

[memory]
backend = "files"
root = ".artesian"
collection = "artesian-memory"

[[agents]]
role = "master"
agent = "claude-code"
"#
        );
        fs::write(&config, toml).expect("write test config");
        config
    }

    /// With no config file and no env override, the dream is disabled (default = off).
    #[test]
    fn dream_on_compact_defaults_to_off() {
        // Use a path that definitely does not exist so config-file fallback is skipped.
        let nonexistent = PathBuf::from("/tmp/artesian-no-such-config-xyz.toml");
        assert!(!pre_compact_dream_enabled(&nonexistent, None));
    }

    /// `dream_on_compact = false` in the config file keeps the dream disabled.
    #[test]
    fn dream_on_compact_false_in_config_stays_disabled() {
        let tmp = temp_path("doc-false");
        fs::create_dir_all(&tmp).expect("create tmp");
        let config = write_compact_config(&tmp, false);
        assert!(!pre_compact_dream_enabled(&config, None));
        let _ = fs::remove_dir_all(&tmp);
    }

    /// `dream_on_compact = true` in the config file enables the dream.
    #[test]
    fn dream_on_compact_true_in_config_enables_dream() {
        let tmp = temp_path("doc-true");
        fs::create_dir_all(&tmp).expect("create tmp");
        let config = write_compact_config(&tmp, true);
        assert!(pre_compact_dream_enabled(&config, None));
        let _ = fs::remove_dir_all(&tmp);
    }

    /// `env_override = "1"` enables the dream even when the config says false.
    #[test]
    fn dream_on_compact_env_var_overrides_config_enabled() {
        let tmp = temp_path("doc-env-on");
        fs::create_dir_all(&tmp).expect("create tmp");
        let config = write_compact_config(&tmp, false);
        assert!(
            pre_compact_dream_enabled(&config, Some("1")),
            "env override=1 must override config false"
        );
        let _ = fs::remove_dir_all(&tmp);
    }

    /// `env_override = "0"` disables the dream even when the config says true.
    #[test]
    fn dream_on_compact_env_var_overrides_config_disabled() {
        let tmp = temp_path("doc-env-off");
        fs::create_dir_all(&tmp).expect("create tmp");
        let config = write_compact_config(&tmp, true);
        assert!(
            !pre_compact_dream_enabled(&config, Some("0")),
            "env override=0 must override config true"
        );
        let _ = fs::remove_dir_all(&tmp);
    }

    /// `spawn_detached_dream` with a non-existent executable must not panic and must not
    /// return an error (errors are swallowed — the checkpoint already succeeded).
    #[test]
    fn spawn_detached_dream_error_is_non_fatal() {
        let tmp = temp_path("spawn-nonfatal");
        let config = tmp.join("artesian.toml");
        let root = tmp.join(".artesian");
        // spawn_detached_dream resolves via current_exe() which exists; but since the
        // dream target dir does not need to actually run, we only verify the function
        // returns without panicking. We can't intercept the child process in a unit test,
        // so we use a config path that guarantees the dream binary's `dream` subcommand
        // will immediately exit with an error (config does not exist) — which should be
        // silently swallowed.
        // Just call the function; the test passes if it does not panic.
        spawn_detached_dream(&config, &root);
    }
}
