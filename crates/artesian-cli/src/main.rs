// SPDX-License-Identifier: Apache-2.0

use std::{
    env, fs,
    io::Write,
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
    time::Duration,
};

use anyhow::{bail, Context, Result};
use aquifer::{
    consolidation_pass, default_migration_collection, export_okf_bundle, recover_after_compaction,
    verify_okf_bundle, AnchorAnchorStore, CollectionCompat, ConsolidationOptions, MemoryBackend,
    MemoryQuery, MemoryScope, MemoryTier, MigrationPlan, SearchHit, SessionAnchor, StoreMemory,
    VectorMemoryConfig,
};
use artesian_core::{
    Agent, AgentBinding, ArtesianConfig, MemoryBackendKind, MemoryConfig, Mode, Role, SpawnRequest,
};
use artesian_process_agent::{
    fallback_agent_catalog, refresh_agent_catalog, ProcessAgent, ProcessAgentConfig,
    ProcessSupervisor,
};
use clap::{Parser, Subcommand, ValueEnum};
use flume::{
    load_role_definitions, role_summaries, TeamCreate, TeamGcOptions, TeamMessage, TeamMessageKind,
    TeamRuntime, TeamRuntimeConfig, TeamSpawn, TeamTaskAdd, TeamTaskClaim, TeamTaskComplete,
};
use headgate::{
    count_tokens, Headgate, HeadgateConfig, LifecycleEntry, MemoryRecallStore, RecallStore,
    SnapshotEntry, WorkingContextBundle, WorkingContextSnapshot,
};
use headrace::{
    ClaimRequest, CommandVerifier, FilesTaskStore, NewTask, TaskKind, TaskStore, VectorTaskStore,
    Verifier, VerifierGate,
};
use serde_json::{json, Value};
use toml_edit::{value, Array, DocumentMut, Item, Table};

const DEFAULT_CONFIG: &str = "artesian.toml";
const MCP_SERVER_NAME: &str = "artesian-memory";
const MCP_TOOL_HINT: &str =
    "ALWAYS search the project memory before non-trivial work; store durable, reusable learnings.";

mod artesiand;
mod import;
mod runtime;
use import::{import_directory, ImportOptions};
use runtime::{
    build_orchestrator, load_config, open_memory_backend, process_supervisor_from_config,
    shutdown_signal,
};

#[derive(Debug, Parser)]
#[command(name = "artesian", about = "Multi-agent context orchestration")]
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
        user_id: Option<String>,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
    },
    Consolidate {
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".artesian")]
        root: PathBuf,
        #[arg(long)]
        allow_llm: bool,
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
        /// Poll mode: do not run a worker, only re-check the goal each turn.
        #[arg(long)]
        poll: bool,
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
    /// Update Artesian via its package manager (Homebrew), then suggest `artesian doctor`. A
    /// convenience wrapper — your config, MCP registrations, and stored memory are untouched.
    Update,
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
        /// Diversify results with Maximal Marginal Relevance to drop near-duplicates (fetches a
        /// larger pool, then re-ranks to --limit).
        #[arg(long)]
        mmr: bool,
        /// MMR relevance/novelty trade-off in [0,1] (1 = pure relevance). Implies --mmr.
        #[arg(long)]
        mmr_lambda: Option<f32>,
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
    let cli = Cli::parse();
    match cli.command {
        Command::Init {
            memory_root,
            project,
            backend,
            collection,
            qdrant_url,
            qdrant_rest_url,
            qdrant_api_key_env,
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
        Command::Memory { command } => memory(command).await,
        Command::Task { command } => task(command).await,
        Command::Team { command } => team(command).await,
        Command::Backfill {
            directory,
            config,
            root,
            backend,
            user_id,
        } => backfill(directory, config, root, backend, user_id).await,
        Command::Onboard {
            project,
            directory,
            backend,
            memory_root,
            collection,
            qdrant_url,
            qdrant_rest_url,
            qdrant_api_key_env,
            user_id,
            config,
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
                    register_mcp: true,
                    project: Some(project.clone()),
                },
                config,
                user_id,
            )
            .await
        }
        Command::Consolidate {
            config,
            root,
            allow_llm,
        } => consolidate(config, root, allow_llm).await,
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
        Command::Loop {
            goal,
            worker_cmd,
            max_turns,
            poll,
            root,
            config,
        } => run_loop(goal, worker_cmd, max_turns, poll, root, config).await,
        Command::Replicate {
            from_url,
            to_url,
            from_key,
            to_key,
            collection,
            to_collection,
            status,
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
                batch,
            )
            .await
        }
        Command::Doctor {
            config,
            root,
            backend,
        } => doctor(config, root, backend).await,
        Command::Update => update(),
    }
}

/// Run a shell command, returning whether it exited successfully (exit code 0).
fn run_shell(cmd: &str) -> Result<bool> {
    run_shell_with_env(cmd, &[])
}

/// Run a shell command with extra environment variables exported to it.
fn run_shell_with_env(cmd: &str, env: &[(&str, &str)]) -> Result<bool> {
    let mut command = std::process::Command::new("sh");
    command.arg("-c").arg(cmd);
    for (key, value) in env {
        command.env(key, value);
    }
    let status = command
        .status()
        .with_context(|| format!("run command: {cmd}"))?;
    Ok(status.success())
}

/// Run a shell command, capturing combined stdout+stderr so a failing verifier's detail can be
/// surfaced as the "last failed check" in the next turn's goal packet.
fn run_shell_capture(cmd: &str) -> Result<(bool, String)> {
    let output = std::process::Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .output()
        .with_context(|| format!("run command: {cmd}"))?;
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    Ok((output.status.success(), text.trim().to_string()))
}

/// Per-turn recall limit injected into the worker (kept small to stay token-cheap).
const LOOP_RECALL_LIMIT: usize = 5;
/// Tag that marks a memory as a project invariant — always injected into the goal packet.
const INVARIANT_TAG: &str = "invariant";
/// Cap on invariants injected into a goal packet (ranked by goal relevance).
const GOAL_INVARIANT_LIMIT: usize = 8;
/// Tag that marks a memory as a verified skill — a previously goal-verified loop approach.
const SKILL_TAG: &str = "skill";
/// Cap on verified skills surfaced in a goal packet (the closest prior approaches).
const GOAL_SKILL_LIMIT: usize = 2;
/// Cap on the captured "last failed check" detail carried into the next turn.
const LAST_CHECK_CHARS: usize = 800;

/// Render a goal packet section from the records carrying `tag`, ranked by goal relevance.
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

/// Assemble a bounded, goal-scoped context packet — the goal, the invariants that must hold, any
/// verified prior approach, the last failed verifier check, and the most relevant memory — rather
/// than a flat recall dump. This is the "hand the agent just the goal, invariants, and last failed
/// check" packet.
async fn assemble_goal_packet(
    backend: Option<&dyn MemoryBackend>,
    goal: &str,
    last_check: Option<&str>,
    recall: &str,
) -> String {
    let mut sections = vec![format!("# Goal\n{goal}")];

    if let Some(backend) = backend {
        // Invariants are always injected (tag-filtered); the verified skill is the prior approach
        // that already passed this goal's verifier — reuse it, the verifier still gates each turn.
        if let Some(section) = packet_tag_section(
            backend,
            goal,
            INVARIANT_TAG,
            GOAL_INVARIANT_LIMIT,
            "Invariants (must hold)",
        )
        .await
        {
            sections.push(section);
        }
        if let Some(section) = packet_tag_section(
            backend,
            goal,
            SKILL_TAG,
            GOAL_SKILL_LIMIT,
            "Known approach (verified)",
        )
        .await
        {
            sections.push(section);
        }
    }

    if let Some(last_check) = last_check.filter(|check| !check.is_empty()) {
        let detail: String = last_check.chars().take(LAST_CHECK_CHARS).collect();
        sections.push(format!("# Last failed check\n{detail}"));
    }

    if !recall.is_empty() {
        sections.push(format!("# Relevant memory\n{recall}"));
    }

    sections.join("\n\n")
}

/// Search the backend for memory relevant to the goal and render a compact recall block.
/// MMR-diversifies a larger pool down to the limit so near-duplicate turn commits do not crowd out
/// distinct context.
async fn loop_recall(backend: &dyn MemoryBackend, goal: &str) -> String {
    let Ok(hits) = backend
        .find(MemoryQuery::new(goal).with_limit(LOOP_RECALL_LIMIT * 3))
        .await
    else {
        return String::new();
    };
    let hits = aquifer::mmr_diversify(hits, LOOP_RECALL_LIMIT, aquifer::MMR_DEFAULT_LAMBDA);
    let mut lines = Vec::new();
    for hit in hits {
        let content = hit.record.content.replace('\n', " ");
        let trimmed: String = content.chars().take(280).collect();
        lines.push(format!("- {trimmed}"));
    }
    lines.join("\n")
}

/// Commit a concise, run-scoped atom recording this turn's outcome, so the loop's
/// working state survives compaction and a later sweep can reclaim it by run id.
async fn loop_commit_turn(
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

/// On a successful loop, store the worker approach as a verified skill — admitted only because the
/// goal verifier passed (PreAct's verify-before-store, provided by the loop by construction).
/// Durable (not run-scoped) and content-deduped, so future goal packets can reuse the approach.
async fn loop_store_skill(backend: &dyn MemoryBackend, goal: &str, worker_cmd: &str, turns: u32) {
    let mut memory = StoreMemory::atom(format!(
        "verified approach for `{goal}`: run `{worker_cmd}` (passed in {turns} turn(s))"
    ));
    memory.tier = MemoryTier::L2Scenario;
    memory.tags = vec![SKILL_TAG.to_string(), "verified".to_string()];
    let _ = backend.store(memory).await;
}

/// An autonomous memory-first loop: each turn recalls goal-relevant memory, runs the worker
/// action (with that recall in `ARTESIAN_RECALL`), writes a resume anchor, verifies the goal,
/// and commits a run-scoped record of the turn. Bounded by `max_turns`.
async fn run_loop(
    goal: String,
    worker_cmd: Option<String>,
    max_turns: u32,
    poll: bool,
    root: PathBuf,
    config: PathBuf,
) -> Result<()> {
    let anchor_store = AnchorAnchorStore::new(&root);
    // Use the project's configured memory backend when present; otherwise a local files backend
    // under the loop root. Recall/commit degrade to a no-op if the backend cannot be opened.
    // No-config fallback: a files backend rooted at `--root`, the same memory root the other
    // `memory` subcommands use for that root, so invariants stored via `memory store` are visible.
    let memory_config = load_config(&config)
        .map(|cfg| cfg.memory)
        .unwrap_or_else(|_| {
            ArtesianConfig::memory_files(root.display().to_string(), Vec::new()).memory
        });
    let backend = match open_memory_backend(&memory_config) {
        Ok(backend) => Some(backend),
        Err(error) => {
            eprintln!("  note: memory recall/commit disabled ({error})");
            None
        }
    };
    let run_id = format!(
        "loop-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_millis())
            .unwrap_or(0)
    );
    println!("loop: goal = {goal:?}, max-turns = {max_turns}, run = {run_id}");
    if run_shell(&goal)? {
        println!("✓ goal already holds (0 turns)");
        return Ok(());
    }
    let mut last_check: Option<String> = None;
    for turn in 1..=max_turns {
        let recall = match &backend {
            Some(backend) => loop_recall(backend.as_ref(), &goal).await,
            None => String::new(),
        };
        // Goal-scoped packet: goal + invariants + last failed check + relevant memory — what the
        // worker actually needs, not a flat recall dump.
        let packet =
            assemble_goal_packet(backend.as_deref(), &goal, last_check.as_deref(), &recall).await;
        if !recall.is_empty() {
            println!("  recall ({} relevant)", recall.lines().count());
        }
        if poll {
            tokio::time::sleep(Duration::from_millis(500)).await;
        } else if let Some(cmd) = &worker_cmd {
            println!("── turn {turn}/{max_turns}: worker ─ {cmd}");
            let env = [
                ("ARTESIAN_PACKET", packet.as_str()),
                ("ARTESIAN_RECALL", recall.as_str()),
                ("ARTESIAN_GOAL", goal.as_str()),
                ("ARTESIAN_RUN_ID", run_id.as_str()),
                ("ARTESIAN_TURN", &turn.to_string()),
            ];
            if !run_shell_with_env(cmd, &env)? {
                eprintln!("  worker exited non-zero on turn {turn}");
            }
        }
        let _ = anchor_store
            .set(SessionAnchor::new(
                format!(
                    "loop turn {turn}: {}",
                    worker_cmd.as_deref().unwrap_or("(poll)")
                ),
                format!("verify goal: {goal}"),
            ))
            .await;
        let (goal_met, check_output) = run_shell_capture(&goal)?;
        // Carry the verifier's failure detail into the next turn's packet (cleared on success).
        last_check = if goal_met {
            None
        } else {
            Some(format!("turn {turn}: `{goal}` failed\n{check_output}"))
        };
        if let Some(backend) = &backend {
            loop_commit_turn(
                backend.as_ref(),
                &run_id,
                turn,
                &goal,
                worker_cmd.as_deref(),
                goal_met,
            )
            .await;
        }
        if goal_met {
            // Verified-skill memory: the goal verifier just passed, so the worker approach is
            // verified by construction — store it (durable) for reuse in future goal packets.
            if let (Some(backend), Some(cmd)) = (&backend, &worker_cmd) {
                loop_store_skill(backend.as_ref(), &goal, cmd, turn).await;
            }
            println!("✓ goal holds after {turn} turn(s)");
            return Ok(());
        }
        println!("  goal not met after turn {turn}");
    }
    bail!("loop reached max-turns ({max_turns}) without the goal holding");
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
    batch: u32,
) -> Result<()> {
    use aquifer::{replicate_collection, QdrantVectorStore, QdrantVectorStoreConfig};
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
        println!("✓ both Qdrant endpoints reachable; collection = {collection}");
        return Ok(());
    }
    let copied = replicate_collection(&source, &target, &collection, &target_collection, batch)
        .await
        .map_err(|error| anyhow::anyhow!("replicate: {error}"))?;
    println!("✓ replicated {copied} points: {collection} -> {target_collection}");
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
fn mcp_registration_status() -> Vec<(&'static str, bool)> {
    let codex = home_dir()
        .ok()
        .map(|home| home.join(".codex").join("config.toml"));
    let zed = zed_settings_path().ok();
    vec![
        (
            "Claude Code (.mcp.json)",
            file_mentions(Path::new(".mcp.json"), MCP_SERVER_NAME),
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
            Ok(qdrant_config) => {
                match aquifer::preflight_qdrant(qdrant_config).await {
                    Ok(report) => println!(
                        "  qdrant:  reachable (gRPC {}, REST {})",
                        report.grpc_url, report.rest_status
                    ),
                    Err(error) => {
                        problems += 1;
                        println!("  qdrant:  UNREACHABLE — {error}");
                        println!("           fix: check the URL + API key env, and that the server is up");
                    }
                }
            }
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
                println!("  memory:  ERROR — {error}");
                println!(
                    "           fix: `artesian memory rebuild` (or `artesian migrate` if the embedding model changed)"
                );
            }
        },
        Err(error) => {
            problems += 1;
            println!("  backend: ERROR opening — {error}");
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

/// Update Artesian via Homebrew (if present), then point at `artesian doctor`. A convenience
/// wrapper — the package manager owns the binary; config, MCP registrations, and stored memory are
/// untouched.
fn update() -> Result<()> {
    let brew_available = std::process::Command::new("brew")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false);
    if brew_available {
        println!("Updating Artesian via Homebrew…");
        if !run_shell("brew upgrade aquifer-labs/tap/artesian")? {
            eprintln!("note: brew reported a non-zero exit (e.g. already up to date)");
        }
        println!("\nNext: run `artesian doctor` to verify the install and your setup survived the upgrade.");
    } else {
        println!(
            "Homebrew not found — update with your install method, then run `artesian doctor`:"
        );
        println!("  brew upgrade aquifer-labs/tap/artesian");
        println!(
            "  or download the latest binary: https://github.com/aquifer-labs/artesian/releases"
        );
    }
    Ok(())
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
            let outcome = runtime
                .message(TeamMessage {
                    team_id,
                    from,
                    to,
                    kind: kind.into(),
                    content,
                    task_id,
                    approved,
                    execute,
                })
                .await?;
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
            local_rerank_enabled: true,
            hyde_enabled: false,
            multi_query_enabled: false,
            debate_enabled: false,
            llm_consolidation_enabled: false,
            semantic_cache: Default::default(),
        },
        agents,
        coordination: Default::default(),
        acc: Default::default(),
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
    write_master_role_skill(&options.memory_root)?;
    println!(
        "initialized Artesian memory mode at {} collection={} project={}",
        options.memory_root.display(),
        config.memory.collection,
        options.project.as_deref().unwrap_or("default")
    );
    Ok(())
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

async fn memory(command: MemoryCommand) -> Result<()> {
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
            mmr,
            mmr_lambda,
            config,
            root,
            backend,
        } => {
            let backend = open_backend_for_command(&config, root, backend)?;
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
            let mut hits = backend.find(memory_query).await?;
            if diversify {
                hits = aquifer::mmr_diversify(
                    hits,
                    limit,
                    mmr_lambda.unwrap_or(aquifer::MMR_DEFAULT_LAMBDA),
                );
            }
            for hit in hits {
                println!("{}", format_memory_hit(&hit));
            }
        }
        MemoryCommand::Context {
            query,
            limit,
            index_chars,
            goal,
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
            let hits = backend
                .find(MemoryQuery::new(query).with_limit(limit))
                .await?;
            if let Some(index) = index {
                println!("# index.md\n{index}");
            }
            println!("# memory.find");
            for hit in hits {
                println!("{}", format_memory_hit(&hit));
            }
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
) -> Result<()> {
    let memory = memory_config_for_command(&config, root, backend)?;
    if memory.backend == MemoryBackendKind::Qdrant {
        preflight_qdrant_memory(&memory).await?;
    }
    let backend = open_memory_backend(&memory)?;
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
    println!(
        "backfill scanned={} imported={} skipped_duplicates={} failed={}",
        report.scanned,
        imported,
        skipped_duplicates,
        report.failed.len()
    );
    println!("{}", serde_json::to_string_pretty(&report)?);
    println!(
        "next best step: run `artesian consolidate` when you want the opt-in LLM semantic pass"
    );
    Ok(())
}

async fn onboard(
    project: String,
    directory: PathBuf,
    options: InitOptions,
    config_path: PathBuf,
    user_id: Option<String>,
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
    let backend = open_memory_backend(&memory)?;
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
    println!(
        "onboard project={} collection={} root={} verification_hits={}",
        project, memory.collection, memory.root, verification_hits
    );
    println!("{}", serde_json::to_string_pretty(&report)?);
    println!(
        "next best step: run `artesian consolidate` when you want the opt-in LLM semantic pass"
    );
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
    if let Some(env_name) = &memory.qdrant_api_key_env {
        config.api_key = env::var(env_name).ok();
    }
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
        local_rerank_enabled: true,
        hyde_enabled: false,
        multi_query_enabled: false,
        debate_enabled: false,
        llm_consolidation_enabled: false,
        semantic_cache: Default::default(),
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
            local_rerank_enabled: true,
            hyde_enabled: false,
            multi_query_enabled: false,
            debate_enabled: false,
            llm_consolidation_enabled: false,
            semantic_cache: Default::default(),
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

/// Register the server at Claude Code **user scope** (project `.mcp.json` is not
/// auto-loaded). Prefer the `claude` CLI; fall back to editing `~/.claude.json`.
/// Best-effort: never fail `init` if Claude Code is absent.
fn register_claude_user_scope(command: &str, config_path: &Path) {
    let config_arg = config_path.display().to_string();
    if command_on_path("claude") {
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

fn home_dir() -> Result<PathBuf> {
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
}
