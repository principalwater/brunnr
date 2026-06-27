// SPDX-License-Identifier: Apache-2.0

pub mod cli;

use std::{
    collections::HashMap,
    path::PathBuf,
    str::FromStr,
    sync::{Arc, Mutex},
    time::Duration,
};

#[cfg(feature = "qdrant")]
use std::env;

use aquifer::{
    insert_skill_procedure_metadata, skill_procedure_from_metadata, AnchorAnchorStore,
    FilesBackend, MemoryBackend, MemoryQuery, MemoryRecord, MemoryScope, MemoryTier, ProcedureStep,
    Relation, SearchHit, SessionAnchor, SessionKey, SessionStore, SessionSummary,
    SqliteVecVectorStore, SqliteVecVectorStoreConfig, StoreMemory, VectorMemoryBackend,
    VectorMemoryConfig, SESSION_RECORD_TAG, SKILL_PROCEDURE_METADATA_KEY,
};
use artesian_core::{
    AccConfig, Agent, AgentBinding, AgentCatalog, AgentMessage, ArtesianConfig, MemoryBackendKind,
    MemoryConfig, Mode, Role, SpawnRequest,
};
use artesian_process_agent::{
    fallback_agent_catalog, load_or_refresh_agent_catalog, validate_binding_model, ProcessAgent,
    ProcessAgentConfig, ProcessSupervisor,
};
use flume::{
    load_role_definitions, loop_core, role_summaries, Lane, LaneBudget, LaneContract, TeamCreate,
    TeamGcOptions, TeamLaneAdd, TeamLaneAssignTask, TeamMessage, TeamMessageKind, TeamRuntime,
    TeamRuntimeConfig, TeamSpawn, TeamTaskAdd, TeamTaskClaim, TeamTaskComplete,
};
use headgate::{
    count_tokens, load_savings_rollup, record_savings, CcsSchema, CommittedContextState,
    CommittedEntry, DefaultQualifyGate, Headgate, HeadgateConfig, LifecycleEntry,
    MemoryRecallStore, OpSavings, QualifyAudit, QualifyDecision, QualifyGate, QualifySignal,
    RecallItem, RecallStore, Resolution, SnapshotEntry, TokenSavingsRollup, WorkingContextBundle,
    WorkingContextSnapshot,
};
use headrace::{
    ClaimRequest, FilesTaskStore, NewTask, Task, TaskStatus, TaskStore, TransitionTask,
};
use rmcp::{
    handler::server::{
        router::tool::ToolRouter,
        wrapper::{Json, Parameters},
    },
    model::{Meta, ProgressNotificationParam, ProgressToken, ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router,
    transport::stdio,
    ErrorData, Peer, RoleServer, ServerHandler, ServiceExt,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex as AsyncMutex;
use tokio_util::sync::CancellationToken;

#[cfg(feature = "qdrant")]
use aquifer::{QdrantVectorStore, QdrantVectorStoreConfig};

const TOOL_INSTRUCTIONS: &str =
    "ALWAYS search the project memory before non-trivial work; store durable, reusable learnings.";
const MASTER_ROLE_SKILL: &str = "In orchestrate/full mode, first call agents.list to inspect reachable agents, models, and role definitions. Use memory.context for compact project recall, create Flume teams with team.create/team.spawn when several teammates are useful, delegate bounded subtasks through team.task.* or orchestrate.delegate(worker), and gate accepted outcomes through the judge/master path before marking work done. After adding a teammate task, BLOCK on team.task.await until it completes, or use orchestrate.delegate for a single worker; do not end your turn or poll in a loop, because blocking keeps the client request active and interruptible.";
const INVARIANT_TAG: &str = "invariant";
const GOAL_INVARIANT_LIMIT: usize = 8;
const TEAM_TASK_AWAIT_DEFAULT_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const TEAM_TASK_AWAIT_DEFAULT_POLL: Duration = Duration::from_millis(500);
const TEAM_TASK_AWAIT_MIN_POLL: Duration = Duration::from_millis(50);
const TEAM_TASK_AWAIT_MAX_POLL: Duration = Duration::from_secs(5);
const ORCHESTRATION_TOOLS: &[&str] = &[
    "agents.list",
    "orchestrate.bind",
    "orchestrate.delegate",
    "orchestrate.loop",
    "orchestrate.status",
    "orchestrate.handoff",
    "team.create",
    "team.spawn",
    "team.task.add",
    "team.task.await",
    "team.task.claim",
    "team.task.complete",
    "team.message",
    "team.status",
    "team.presence",
    "team.lane.add",
    "team.lane.assign",
    "team.cleanup",
    "team.gc",
];

#[derive(Clone)]
pub struct MemoryServer {
    backend: Arc<dyn MemoryBackend>,
    anchor_store: Option<AnchorAnchorStore>,
    okf_root: Option<PathBuf>,
    router_enabled: bool,
    mode: Mode,
    bindings: Arc<Mutex<Vec<AgentBinding>>>,
    catalog: Arc<Mutex<AgentCatalog>>,
    delegate_results: Arc<Mutex<HashMap<String, DelegateRecord>>>,
    team_runtime: Arc<AsyncMutex<TeamRuntime>>,
    task_root: PathBuf,
    repo_root: PathBuf,
    process_defaults: ProcessDefaults,
    acc: AccConfig,
    tool_router: ToolRouter<Self>,
    /// Collection name passed to token-savings entries.
    collection: String,
    /// Mirror of `config.memory.track_savings`.
    track_savings: bool,
}

// ── Progress / cancellation helpers ───────────────────────────────────────────────────────────

/// Build a [`loop_core::LoopProgressCallback`] that forwards each event to the MCP client via
/// `peer.notify_progress`.  Returns `None` when `progress_token` is absent (the client did not
/// request progress notifications), so the loop runs without any overhead in that case.
fn make_mcp_progress_callback(
    peer: Peer<RoleServer>,
    progress_token: Option<ProgressToken>,
) -> Option<loop_core::LoopProgressCallback> {
    let token = progress_token?;
    Some(Arc::new(
        move |progress: f64, total: Option<f64>, message: Option<String>| {
            let peer = peer.clone();
            let token = token.clone();
            tokio::spawn(async move {
                let _ = peer
                    .notify_progress(ProgressNotificationParam {
                        progress_token: token,
                        progress,
                        total,
                        message,
                    })
                    .await;
            });
        },
    ))
}

/// Spawn a background task that sends a `notifications/progress` heartbeat every `interval`
/// while a long-running MCP tool call is in progress.  The returned `JoinHandle` **must** be
/// `.abort()`ed when the operation completes (success, error, or cancel) so the task stops.
///
/// When `progress_token` is `None` the spawned task exits immediately (no-op path).
fn spawn_progress_heartbeat(
    peer: Peer<RoleServer>,
    progress_token: Option<ProgressToken>,
    interval: Duration,
    label: &'static str,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let Some(token) = progress_token else { return };
        let mut tick: u64 = 0;
        loop {
            tokio::time::sleep(interval).await;
            tick += 1;
            let _ = peer
                .notify_progress(ProgressNotificationParam {
                    progress_token: token.clone(),
                    progress: tick as f64,
                    total: None,
                    message: Some(format!("{label} ({}s elapsed)", tick * interval.as_secs())),
                })
                .await;
        }
    })
}

impl MemoryServer {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        Self::with_backend_and_anchor(
            Arc::new(FilesBackend::new(&root)),
            Some(AnchorAnchorStore::new(&root)),
        )
        .with_okf_root(Some(root))
    }

    pub fn with_backend(backend: Arc<dyn MemoryBackend>) -> Self {
        Self::with_backend_and_anchor(backend, None)
    }

    pub fn with_backend_and_anchor(
        backend: Arc<dyn MemoryBackend>,
        anchor_store: Option<AnchorAnchorStore>,
    ) -> Self {
        Self {
            backend,
            anchor_store,
            okf_root: None,
            router_enabled: false,
            mode: Mode::Memory,
            bindings: Arc::new(Mutex::new(Vec::new())),
            catalog: Arc::new(Mutex::new(AgentCatalog::default())),
            delegate_results: Arc::new(Mutex::new(HashMap::new())),
            team_runtime: Arc::new(AsyncMutex::new(TeamRuntime::new(TeamRuntimeConfig::new(
                std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
                PathBuf::from(".artesian").join("tasks"),
            )))),
            task_root: PathBuf::from(".artesian").join("tasks"),
            repo_root: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            process_defaults: ProcessDefaults::default(),
            acc: AccConfig::default(),
            tool_router: Self::mode_tool_router(Mode::Memory),
            collection: String::new(),
            track_savings: true,
        }
    }

    pub fn with_okf_root(mut self, root: Option<PathBuf>) -> Self {
        self.okf_root = root;
        self
    }

    pub fn with_router_enabled(mut self, enabled: bool) -> Self {
        self.router_enabled = enabled;
        self
    }

    pub fn with_runtime_config(mut self, config: &ArtesianConfig) -> Self {
        self.mode = config.mode;
        self.acc = config.acc.clone();
        self.bindings = Arc::new(Mutex::new(config.agents.clone()));
        let mut catalog = fallback_agent_catalog(&config.agents);
        self.task_root = PathBuf::from(&config.memory.root).join("tasks");
        self.repo_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        self.process_defaults = ProcessDefaults::from_config(config);
        let definitions = load_role_definitions(&self.repo_root).unwrap_or_default();
        catalog.roles = role_summaries(&definitions);
        self.catalog = Arc::new(Mutex::new(catalog.clone()));
        self.team_runtime = Arc::new(AsyncMutex::new(self.build_team_runtime(
            config.agents.clone(),
            catalog,
            definitions,
        )));
        self.tool_router = Self::mode_tool_router(config.mode);
        self.collection = config.memory.collection.clone();
        self.track_savings = config.memory.track_savings;
        self
    }

    pub fn with_catalog(mut self, catalog: AgentCatalog) -> Self {
        let bindings = self
            .bindings
            .lock()
            .map(|bindings| bindings.clone())
            .unwrap_or_default();
        let definitions = load_role_definitions(&self.repo_root).unwrap_or_default();
        let mut catalog = catalog;
        if catalog.roles.is_empty() {
            catalog.roles = role_summaries(&definitions);
        }
        self.team_runtime = Arc::new(AsyncMutex::new(self.build_team_runtime(
            bindings,
            catalog.clone(),
            definitions,
        )));
        self.catalog = Arc::new(Mutex::new(catalog));
        self
    }

    pub fn with_bindings(mut self, bindings: Vec<AgentBinding>) -> Self {
        let mut catalog = fallback_agent_catalog(&bindings);
        let definitions = load_role_definitions(&self.repo_root).unwrap_or_default();
        catalog.roles = role_summaries(&definitions);
        self.team_runtime = Arc::new(AsyncMutex::new(self.build_team_runtime(
            bindings.clone(),
            catalog.clone(),
            definitions,
        )));
        self.catalog = Arc::new(Mutex::new(catalog));
        self.bindings = Arc::new(Mutex::new(bindings));
        self
    }

    pub fn with_mode(mut self, mode: Mode) -> Self {
        self.mode = mode;
        self.tool_router = Self::mode_tool_router(mode);
        self
    }

    pub fn with_task_root(mut self, task_root: impl Into<PathBuf>) -> Self {
        self.task_root = task_root.into();
        self
    }

    pub fn with_repo_root(mut self, repo_root: impl Into<PathBuf>) -> Self {
        self.repo_root = repo_root.into();
        self
    }

    pub fn with_process_registry_dir(mut self, registry_dir: impl Into<PathBuf>) -> Self {
        self.process_defaults.registry_dir = registry_dir.into();
        self
    }

    pub fn visible_tool_names(&self) -> Vec<String> {
        self.tool_router
            .list_all()
            .into_iter()
            .map(|tool| tool.name.to_string())
            .collect()
    }

    pub fn from_config(config: &MemoryConfig) -> anyhow::Result<Self> {
        let mut server = Self::with_backend_and_anchor(
            open_memory_backend(config)?,
            Some(AnchorAnchorStore::new(&config.root)),
        )
        .with_okf_root(Some(PathBuf::from(&config.root)));
        server.collection = config.collection.clone();
        server.track_savings = config.track_savings;
        Ok(server)
    }

    pub async fn from_artesian_config(config: &ArtesianConfig) -> anyhow::Result<Self> {
        let mut server = Self::from_config(&config.memory)?.with_runtime_config(config);
        if matches!(config.mode, Mode::Orchestrate | Mode::Full) {
            let cache_path = PathBuf::from(&config.memory.root).join("agents.json");
            let catalog = load_or_refresh_agent_catalog(&config.agents, &cache_path, false)
                .await
                .unwrap_or_else(|_| fallback_agent_catalog(&config.agents));
            server = server.with_catalog(catalog);
        }
        Ok(server)
    }

    fn mode_tool_router(mode: Mode) -> ToolRouter<Self> {
        let mut router = Self::tool_router();
        if !matches!(mode, Mode::Orchestrate | Mode::Full) {
            for tool in ORCHESTRATION_TOOLS {
                router.disable_route(*tool);
            }
        }
        router
    }

    fn ensure_orchestration_enabled(&self) -> Result<(), ErrorData> {
        if matches!(self.mode, Mode::Orchestrate | Mode::Full) {
            Ok(())
        } else {
            Err(ErrorData::internal_error(
                "orchestration tools require mode orchestrate or full".to_string(),
                None,
            ))
        }
    }

    fn binding_for_role(&self, role: Role) -> Result<AgentBinding, ErrorData> {
        self.bindings
            .lock()
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?
            .iter()
            .find(|binding| binding.role == role)
            .cloned()
            .ok_or_else(|| {
                ErrorData::invalid_params(
                    format!("no binding configured for role {}", role.canonical_alias()),
                    None,
                )
            })
    }

    fn process_agent_for_binding(&self, binding: &AgentBinding) -> ProcessAgent {
        let command = binding
            .command
            .clone()
            .unwrap_or_else(|| binding.agent.clone());
        let static_models = self
            .catalog
            .lock()
            .ok()
            .and_then(|catalog| {
                catalog
                    .agents
                    .iter()
                    .find(|entry| entry.agent == binding.agent)
                    .map(|entry| {
                        entry
                            .models
                            .iter()
                            .filter(|model| model.reachable)
                            .map(|model| model.id.clone())
                            .collect::<Vec<_>>()
                    })
            })
            .unwrap_or_default();
        ProcessAgent::new(
            ProcessAgentConfig::new(command)
                .with_agent_id(binding.agent.clone())
                .with_default_model(binding.model.clone())
                .with_args(binding.args.clone())
                .with_static_models(static_models)
                .with_working_dir(&self.repo_root)
                .with_timeout(Duration::from_secs(binding.timeout_seconds.unwrap_or(120)))
                .with_registry_dir(self.process_defaults.registry_dir.clone())
                .with_max_concurrent_spawns(self.process_defaults.max_concurrent_spawns)
                .with_max_lifetime(self.process_defaults.max_lifetime)
                .with_termination_grace(self.process_defaults.termination_grace),
        )
    }

    fn record_delegate(
        &self,
        task_id: String,
        status: String,
        result: Option<String>,
    ) -> Result<(), ErrorData> {
        self.delegate_results
            .lock()
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?
            .insert(task_id, DelegateRecord { status, result });
        Ok(())
    }

    fn build_team_runtime(
        &self,
        bindings: Vec<AgentBinding>,
        catalog: AgentCatalog,
        definitions: Vec<flume::RoleDefinition>,
    ) -> TeamRuntime {
        TeamRuntime::new(TeamRuntimeConfig {
            repo_root: self.repo_root.clone(),
            task_root: self.task_root.clone(),
            registry_dir: self.process_defaults.registry_dir.clone(),
            bindings,
            catalog,
            definitions,
            max_teammates: self.process_defaults.max_concurrent_spawns,
            max_concurrent_spawns: self.process_defaults.max_concurrent_spawns,
            max_lifetime: self.process_defaults.max_lifetime,
            termination_grace: self.process_defaults.termination_grace,
        })
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct FindRequest {
    pub query: String,
    pub limit: Option<usize>,
    pub node_id: Option<String>,
    pub expand: Option<bool>,
    pub scope: Option<ScopeRequest>,
    pub agent_id: Option<String>,
    pub session_id: Option<String>,
    pub task_id: Option<String>,
    pub user_id: Option<String>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct FindResponse {
    pub hits: Vec<FindHit>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct FindHit {
    pub id: String,
    pub node_id: String,
    pub content: String,
    pub score: f32,
    pub tags: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct AnswerRequest {
    pub question: String,
    pub limit: Option<usize>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct AnswerResponse {
    pub answer: String,
    pub extractive: bool,
    pub sources: Vec<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ContextRequest {
    pub query: String,
    pub limit: Option<usize>,
    pub index_chars: Option<usize>,
    pub expand: Option<bool>,
    pub scope: Option<ScopeRequest>,
    pub agent_id: Option<String>,
    pub session_id: Option<String>,
    pub task_id: Option<String>,
    pub user_id: Option<String>,
    /// When set, also return the project invariants relevant to this goal (memories tagged
    /// `invariant`), so the caller can assemble a goal-scoped packet rather than a flat dump.
    pub goal: Option<String>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct ContextResponse {
    pub index: Option<String>,
    pub hits: Vec<FindHit>,
    /// Invariants relevant to `goal` (empty unless `goal` was provided).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub invariants: Vec<FindHit>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CommitRequest {
    pub query: String,
    pub budget_tokens: Option<usize>,
    pub recall_limit: Option<usize>,
    pub min_score: Option<f32>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct CommitResponse {
    /// The slot-grouped committed context the agent should read.
    pub committed_context: String,
    pub candidates: usize,
    pub admitted: usize,
    pub rejected_relevance: usize,
    pub rejected_redundant: usize,
    pub rejected_saturated: usize,
    pub compressed: usize,
    pub evicted: usize,
    pub footprint_tokens: usize,
    pub budget_tokens: usize,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct QualifyRequest {
    /// Candidate text to run through the ACC qualify-gate.
    pub candidate: String,
    /// Optional current goal/task. When set, it drives relevance scoring and state recall.
    pub goal: Option<String>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct QualifyResponse {
    pub admitted: bool,
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slot: Option<String>,
    pub score: f32,
    pub signals: Vec<QualifySignalResponse>,
    pub agreement: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chance_corrected_agreement: Option<f32>,
    pub confidence: f32,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct QualifySignalResponse {
    pub name: String,
    pub value: f32,
    pub threshold: f32,
    pub passed: bool,
    pub margin: f32,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct StoreRequest {
    pub content: String,
    pub tags: Option<Vec<String>>,
    pub node_id: Option<String>,
    pub relations: Option<Vec<RelationRequest>>,
    pub source: Option<String>,
    pub confidence: Option<f32>,
    pub scope: Option<ScopeRequest>,
    pub agent_id: Option<String>,
    pub session_id: Option<String>,
    pub task_id: Option<String>,
    pub user_id: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RelationRequest {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    #[serde(default)]
    pub source_node_id: Option<String>,
}

impl From<RelationRequest> for Relation {
    fn from(value: RelationRequest) -> Self {
        Relation::new(
            value.subject,
            value.predicate,
            value.object,
            value.source_node_id.unwrap_or_default(),
        )
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum ScopeRequest {
    Shared,
    Agent,
    Session,
    Task,
}

impl From<ScopeRequest> for MemoryScope {
    fn from(value: ScopeRequest) -> Self {
        match value {
            ScopeRequest::Shared => Self::Shared,
            ScopeRequest::Agent => Self::Agent,
            ScopeRequest::Session => Self::Session,
            ScopeRequest::Task => Self::Task,
        }
    }
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct StoreResponse {
    pub id: String,
    pub node_id: String,
}

/// Request for `memory.learn`: commit a curated, governed SKILL memory record.
///
/// Skills committed here carry provenance (`sources`), usage signals (`access_count`), and
/// participate in the normal decay/eviction lifecycle — making them governed and portable.
///
/// DISCIPLINE: commit a CURATED skill — clear title, polished body, explicit sources.
///
/// Re-learning the same title + content is idempotent (content-hash dedup via `node_id`).
/// When `procedure` is supplied, it is stored additively in record metadata and participates in
/// the skill identity so procedural and non-procedural versions do not overwrite each other.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct LearnRequest {
    /// Short human-readable title for the skill (used as a stable lookup key in listings).
    pub title: String,
    /// Curated body text of the skill. Should be polished prose or step-by-step instructions,
    /// not raw conversation output. Combine with `sources` for provenance attribution.
    pub content: String,
    /// Provenance source identifiers (file paths, URLs, doc references).  Stored on the record
    /// and visible in `memory.skills` listings; multiple values are joined and stored together.
    pub sources: Option<Vec<String>>,
    /// Additional tags to attach (the `skill` tag is always included automatically).
    pub tags: Option<Vec<String>>,
    /// Optional guarded procedure for replay without per-step model calls.
    pub procedure: Option<Vec<SkillProcedureStep>>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct LearnResponse {
    /// Stable record ID (content-hash based; identical on idempotent re-learn).
    pub id: String,
    /// Stable node_id (`skill:<hash>`); use this to retrieve the skill by node_id.
    pub node_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SkillProcedureStep {
    /// Shell command/action to run for this step.
    pub run: String,
    /// Optional precondition check. Exit 0 means the guard holds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guard: Option<String>,
}

impl From<ProcedureStep> for SkillProcedureStep {
    fn from(step: ProcedureStep) -> Self {
        Self {
            run: step.run,
            guard: step.guard,
        }
    }
}

impl From<SkillProcedureStep> for ProcedureStep {
    fn from(step: SkillProcedureStep) -> Self {
        Self {
            run: step.run,
            guard: step.guard,
        }
    }
}

/// A single skill entry returned by `memory.skills`.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct SkillHit {
    pub id: String,
    pub node_id: String,
    /// Full skill content (title heading + body).
    pub content: String,
    /// Human-readable title extracted from `metadata["title"]` when available (manually learned
    /// skills); `None` for auto-committed loop skills that lack an explicit title.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// How many times this skill has been surfaced by `find`/`context` (usage signal for decay).
    pub access_count: u32,
    /// Provenance: source paths, URLs, or origin label recorded at learn time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// ISO-8601 timestamp of the most recent retrieval (`None` if never retrieved).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_access: Option<String>,
    /// Optional guarded procedure stored on this skill.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub procedure: Option<Vec<SkillProcedureStep>>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SkillsRequest {
    /// Maximum number of skills to return (default 20).
    pub limit: Option<usize>,
    /// Sort by usage (access_count descending) instead of default relevance/decay order.
    pub by_usage: Option<bool>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct SkillsResponse {
    pub skills: Vec<SkillHit>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SkillReplayRequest {
    /// Human-readable skill title to replay.
    pub title: String,
    /// Execute run commands only when explicitly true. Defaults to false (dry-run).
    pub execute: Option<bool>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct SkillReplayStepResult {
    pub index: usize,
    pub run: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guard: Option<String>,
    /// `passed`, `failed`, or `not-run`.
    pub guard_status: String,
    /// `passed`, `failed`, or `not-run`.
    pub run_status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guard_output: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_output: Option<String>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct SkillReplayResponse {
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_id: Option<String>,
    pub execute: bool,
    /// `dry-run`, `success`, `guard-failed`, `run-failed`, `no-procedure`, or `not-found`.
    pub status: String,
    pub message: String,
    /// True when the caller should abandon replay and proceed with normal reasoning.
    pub fallback: bool,
    pub steps: Vec<SkillReplayStepResult>,
}

pub async fn answer_memory(
    backend: &dyn MemoryBackend,
    acc: &AccConfig,
    question: &str,
    limit: usize,
) -> anyhow::Result<AnswerResponse> {
    let hits = backend
        .find(MemoryQuery::new(question).with_limit(limit.max(1)))
        .await?;
    let sources = answer_sources(&hits);
    if let Some(answer) = maybe_llm_answer(acc, question, &hits).await? {
        return Ok(AnswerResponse {
            answer,
            extractive: false,
            sources,
        });
    }
    Ok(AnswerResponse {
        answer: extractive_answer(&hits),
        extractive: true,
        sources,
    })
}

pub async fn qualify_memory_candidate(
    backend: &dyn MemoryBackend,
    acc: &AccConfig,
    candidate: &str,
    goal: Option<&str>,
) -> anyhow::Result<QualifyResponse> {
    let config = HeadgateConfig::from(acc);
    let query = goal
        .map(str::trim)
        .filter(|goal| !goal.is_empty())
        .unwrap_or(candidate);
    let hits = backend
        .find(MemoryQuery::new(query).with_limit(config.recall_limit.max(1)))
        .await?;
    let ccs = committed_state_from_hits(hits, config.budget_tokens);
    let score = candidate_relevance_score(candidate, goal);
    let item = RecallItem::new("qualify:candidate", candidate, score);
    let gate = qualify_gate_from_acc(acc, &config)?;
    let decision = gate.qualify(&item, &ccs).await;
    Ok(qualify_response_from_decision(decision))
}

pub fn normalize_skill_procedure(
    procedure: Vec<ProcedureStep>,
) -> anyhow::Result<Vec<ProcedureStep>> {
    procedure
        .into_iter()
        .enumerate()
        .map(|(index, step)| {
            let run = step.run.trim().to_string();
            if run.is_empty() {
                anyhow::bail!("procedure step {} has an empty run command", index + 1);
            }
            let guard = step.guard.and_then(|guard| {
                let guard = guard.trim().to_string();
                (!guard.is_empty()).then_some(guard)
            });
            Ok(ProcedureStep { run, guard })
        })
        .collect()
}

pub fn skill_identity_material(
    title: &str,
    content: &str,
    procedure: &[ProcedureStep],
) -> anyhow::Result<String> {
    let mut material = format!("{title}\n\n{content}");
    if !procedure.is_empty() {
        material.push_str("\n\nprocedure:");
        material.push_str(&serde_json::to_string(procedure)?);
    }
    Ok(material)
}

pub async fn replay_skill_procedure(
    backend: &dyn MemoryBackend,
    title: &str,
    execute: bool,
    collection: &str,
    track_savings: bool,
) -> anyhow::Result<SkillReplayResponse> {
    let Some(skill) = find_skill_by_title(backend, title).await? else {
        return Ok(SkillReplayResponse {
            title: title.to_string(),
            node_id: None,
            execute,
            status: "not-found".to_string(),
            message: format!("skill `{title}` was not found"),
            fallback: true,
            steps: Vec::new(),
        });
    };
    let procedure = skill_procedure_from_metadata(&skill.metadata)?.unwrap_or_default();
    if procedure.is_empty() {
        return Ok(SkillReplayResponse {
            title: skill_title(&skill).unwrap_or_else(|| title.to_string()),
            node_id: Some(skill.node_id),
            execute,
            status: "no-procedure".to_string(),
            message: format!("skill `{title}` has no procedure; proceed with normal reasoning"),
            fallback: true,
            steps: Vec::new(),
        });
    }

    let mut steps = replay_plan_steps(&procedure);
    if !execute {
        return Ok(SkillReplayResponse {
            title: skill_title(&skill).unwrap_or_else(|| title.to_string()),
            node_id: Some(skill.node_id),
            execute,
            status: "dry-run".to_string(),
            message: "dry-run: no guard or run commands executed".to_string(),
            fallback: false,
            steps,
        });
    }

    for (index, step) in procedure.iter().enumerate() {
        if let Some(guard) = &step.guard {
            let (passed, output) = mcp_run_shell_capture(guard, Vec::new(), None).await?;
            steps[index].guard_status = if passed { "passed" } else { "failed" }.to_string();
            if !output.is_empty() {
                steps[index].guard_output = Some(output);
            }
            if !passed {
                return Ok(SkillReplayResponse {
                    title: skill_title(&skill).unwrap_or_else(|| title.to_string()),
                    node_id: Some(skill.node_id),
                    execute,
                    status: "guard-failed".to_string(),
                    message: format!(
                        "guard failed at step {}; replay aborted. Proceed with normal reasoning.",
                        index + 1
                    ),
                    fallback: true,
                    steps,
                });
            }
        }

        let (passed, output) = mcp_run_shell_capture(&step.run, Vec::new(), None).await?;
        steps[index].run_status = if passed { "passed" } else { "failed" }.to_string();
        if !output.is_empty() {
            steps[index].run_output = Some(output);
        }
        if !passed {
            return Ok(SkillReplayResponse {
                title: skill_title(&skill).unwrap_or_else(|| title.to_string()),
                node_id: Some(skill.node_id),
                execute,
                status: "run-failed".to_string(),
                message: format!(
                    "run command failed at step {}; replay aborted. Proceed with normal reasoning.",
                    index + 1
                ),
                fallback: true,
                steps,
            });
        }
    }

    record_savings(
        "skill.replay",
        collection,
        0,
        count_tokens(&skill.content),
        track_savings,
    );

    Ok(SkillReplayResponse {
        title: skill_title(&skill).unwrap_or_else(|| title.to_string()),
        node_id: Some(skill.node_id),
        execute,
        status: "success".to_string(),
        message: format!("guarded replay completed {} step(s)", steps.len()),
        fallback: false,
        steps,
    })
}

async fn find_skill_by_title(
    backend: &dyn MemoryBackend,
    title: &str,
) -> anyhow::Result<Option<MemoryRecord>> {
    let mut records = query_skill_records(backend, title, 50).await?;
    if records
        .iter()
        .all(|record| !skill_title_matches(record, title))
    {
        records.extend(query_skill_records(backend, "", 1_000).await?);
    }

    let mut deduped = HashMap::<String, MemoryRecord>::new();
    for record in records {
        deduped.entry(record.node_id.clone()).or_insert(record);
    }
    let mut matches = deduped
        .into_values()
        .filter(|record| skill_title_matches(record, title))
        .collect::<Vec<_>>();
    matches.sort_by(|left, right| {
        skill_replay_rank(right)
            .cmp(&skill_replay_rank(left))
            .then_with(|| right.created_at.cmp(&left.created_at))
    });
    Ok(matches.into_iter().next())
}

async fn query_skill_records(
    backend: &dyn MemoryBackend,
    text: &str,
    limit: usize,
) -> anyhow::Result<Vec<MemoryRecord>> {
    let mut query = MemoryQuery::new(text.to_string()).with_limit(limit);
    query.tags = vec![loop_core::LOOP_SKILL_TAG.to_string()];
    Ok(backend
        .find(query)
        .await?
        .into_iter()
        .map(|hit| hit.record)
        .collect())
}

fn skill_replay_rank(record: &MemoryRecord) -> u8 {
    u8::from(record.metadata.contains_key(SKILL_PROCEDURE_METADATA_KEY))
}

fn skill_title_matches(record: &MemoryRecord, title: &str) -> bool {
    record.node_id == title
        || record
            .metadata
            .get("title")
            .is_some_and(|value| value == title)
        || record
            .content
            .lines()
            .next()
            .is_some_and(|line| line == format!("# {title}"))
}

fn skill_title(record: &MemoryRecord) -> Option<String> {
    record.metadata.get("title").cloned().or_else(|| {
        record
            .content
            .lines()
            .next()
            .and_then(|line| line.strip_prefix("# "))
            .map(str::to_string)
    })
}

fn replay_plan_steps(procedure: &[ProcedureStep]) -> Vec<SkillReplayStepResult> {
    procedure
        .iter()
        .enumerate()
        .map(|(index, step)| SkillReplayStepResult {
            index: index + 1,
            run: step.run.clone(),
            guard: step.guard.clone(),
            guard_status: "not-run".to_string(),
            run_status: "not-run".to_string(),
            guard_output: None,
            run_output: None,
        })
        .collect()
}

fn answer_sources(hits: &[SearchHit]) -> Vec<String> {
    let mut sources = Vec::new();
    for hit in hits {
        if !sources.contains(&hit.record.node_id) {
            sources.push(hit.record.node_id.clone());
        }
    }
    sources
}

fn extractive_answer(hits: &[SearchHit]) -> String {
    if hits.is_empty() {
        return "Extractive answer: no committed memory matched the question.".to_string();
    }
    let mut answer = String::from("Extractive answer from retrieved memory:");
    for hit in hits {
        answer.push_str(&format!(
            "\n\n[{}]\n{}",
            hit.record.node_id,
            hit.record.content.trim()
        ));
    }
    answer
}

#[cfg(feature = "llm")]
async fn maybe_llm_answer(
    acc: &AccConfig,
    question: &str,
    hits: &[SearchHit],
) -> anyhow::Result<Option<String>> {
    let Some(llm) = acc.compressor.as_ref().or(acc.judge.as_ref()) else {
        return Ok(None);
    };
    if hits.is_empty() {
        return Ok(None);
    }
    let client = headgate::llm_client_from_config(llm)?;
    let answer = client
        .complete(
            headgate::LlmRequest::new(answer_prompt(question, hits))
                .with_system(
                    "You answer questions using only the supplied committed memory. Cite source node_ids.",
                )
                .with_temperature(0.0)
                .with_max_tokens(400),
        )
        .await?
        .trim()
        .to_string();
    Ok((!answer.is_empty()).then_some(answer))
}

#[cfg(not(feature = "llm"))]
async fn maybe_llm_answer(
    acc: &AccConfig,
    question: &str,
    hits: &[SearchHit],
) -> anyhow::Result<Option<String>> {
    let _ = (acc, question, hits);
    Ok(None)
}

#[cfg(feature = "llm")]
fn answer_prompt(question: &str, hits: &[SearchHit]) -> String {
    let mut prompt = String::from(
        "Use the grounding chunks below to answer the question concisely. \
         Use bracket citations with node_ids, for example [node:abc]. \
         If the grounding is insufficient, say so and cite the closest source.\n\nGrounding:\n",
    );
    for hit in hits {
        prompt.push_str(&format!(
            "\n[{}]\n{}\n",
            hit.record.node_id,
            hit.record.content.trim()
        ));
    }
    prompt.push_str(&format!("\nQuestion: {question}\nAnswer:"));
    prompt
}

fn committed_state_from_hits(hits: Vec<SearchHit>, budget_tokens: usize) -> CommittedContextState {
    let mut ccs = CommittedContextState::new(CcsSchema::default(), budget_tokens);
    for hit in hits {
        ccs.admit(CommittedEntry::new(
            hit.record.node_id,
            "fact",
            hit.record.content,
            hit.score,
        ));
    }
    ccs
}

fn qualify_gate_from_acc(
    acc: &AccConfig,
    config: &HeadgateConfig,
) -> anyhow::Result<Arc<dyn QualifyGate>> {
    #[cfg(not(feature = "llm"))]
    let _ = acc;
    #[cfg(feature = "llm")]
    {
        if let Some(judge) = &acc.judge {
            let client = headgate::llm_client_from_config(judge)?;
            return Ok(Arc::new(headgate::JudgeQualifyGate::new(client)));
        }
    }
    Ok(Arc::new(DefaultQualifyGate::new(
        config.min_score,
        config.redundancy_threshold,
    )))
}

fn candidate_relevance_score(candidate: &str, goal: Option<&str>) -> f32 {
    if candidate.trim().is_empty() {
        return 0.0;
    }
    let Some(goal) = goal.map(str::trim).filter(|goal| !goal.is_empty()) else {
        return 1.0;
    };
    let haystack = candidate.to_ascii_lowercase();
    goal.split_whitespace()
        .map(str::to_ascii_lowercase)
        .filter(|term| !term.is_empty())
        .map(|term| haystack.matches(&term).count() as f32)
        .sum()
}

fn qualify_response_from_decision(decision: QualifyDecision) -> QualifyResponse {
    let audit = decision
        .audit
        .unwrap_or_else(|| QualifyAudit::from_signals(decision.admitted, Vec::new()));
    QualifyResponse {
        admitted: decision.admitted,
        reason: decision.reason,
        slot: decision.slot,
        score: decision.score,
        signals: audit.signals.into_iter().map(Into::into).collect(),
        agreement: audit.agreement,
        chance_corrected_agreement: audit.chance_corrected_agreement,
        confidence: audit.confidence,
    }
}

impl From<QualifySignal> for QualifySignalResponse {
    fn from(signal: QualifySignal) -> Self {
        Self {
            name: signal.name,
            value: signal.value,
            threshold: signal.threshold,
            passed: signal.passed,
            margin: signal.margin,
        }
    }
}

fn validate_confidence(confidence: Option<f32>) -> Result<Option<f32>, ErrorData> {
    if let Some(value) = confidence {
        if !(0.0..=1.0).contains(&value) {
            return Err(ErrorData::invalid_params(
                format!("confidence must be within 0.0..=1.0, got {value}"),
                None,
            ));
        }
    }
    Ok(confidence)
}

fn find_hit(hit: SearchHit) -> FindHit {
    FindHit {
        id: hit.record.id.to_string(),
        node_id: hit.record.node_id,
        content: hit.record.content,
        score: hit.score,
        tags: hit.record.tags,
        source: hit.record.source,
        confidence: hit.record.confidence,
    }
}

async fn checkpoint_anchor(
    store: Option<&AnchorAnchorStore>,
    key: &SessionKey,
    request: &SessionCheckpointRequest,
) -> Result<SessionAnchor, ErrorData> {
    let existing = if let Some(store) = store {
        store
            .get_for_session(key)
            .await
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?
    } else {
        None
    };
    let current_task = request
        .current_task
        .clone()
        .or_else(|| existing.as_ref().map(|anchor| anchor.current_task.clone()))
        .or_else(|| request.goal.clone())
        .unwrap_or_else(|| format!("session {}", key.session_id));
    let next_step = request
        .next_step
        .clone()
        .or_else(|| existing.as_ref().map(|anchor| anchor.next_step.clone()))
        .unwrap_or_else(|| "continue from the checkpoint".to_string());
    let mut anchor = existing.unwrap_or_else(|| SessionAnchor::new(&current_task, &next_step));
    anchor.current_task = current_task;
    anchor.next_step = next_step;
    if let Some(plan_pointer) = &request.plan_pointer {
        anchor.plan_pointer = Some(plan_pointer.clone());
    }
    if let Some(last_decisions) = &request.last_decisions {
        anchor.last_decisions = last_decisions.clone();
    }
    if let Some(store) = store {
        store
            .set_for_session(key, anchor)
            .await
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))
    } else {
        Ok(anchor)
    }
}

async fn session_scoped_hits(
    backend: &dyn MemoryBackend,
    key: &SessionKey,
    query_text: &str,
    limit: usize,
) -> Result<Vec<SearchHit>, ErrorData> {
    let mut query = MemoryQuery::new(query_text).with_limit(limit);
    query.scope = Some(MemoryScope::Session);
    query.user_id = Some(key.user_id.clone());
    query.session_id = Some(key.session_id.clone());
    query.task_id = Some(key.task_id.clone());
    let hits = backend
        .find(query)
        .await
        .map_err(|error| ErrorData::internal_error(error.to_string(), None))?
        .into_iter()
        .filter(|hit| !hit.record.tags.iter().any(|tag| tag == SESSION_RECORD_TAG))
        .collect();
    Ok(hits)
}

async fn invariant_hits(
    backend: &dyn MemoryBackend,
    key: &SessionKey,
    query_text: &str,
) -> Result<Vec<SearchHit>, ErrorData> {
    let mut query = MemoryQuery::new(query_text).with_limit(GOAL_INVARIANT_LIMIT);
    query.tags = vec![INVARIANT_TAG.to_string()];
    if !key.is_default() {
        query.user_id = Some(key.user_id.clone());
    }
    backend
        .find(query)
        .await
        .map_err(|error| ErrorData::internal_error(error.to_string(), None))
}

fn build_session_bundle(
    anchor: Option<&SessionAnchor>,
    session_hits: &[SearchHit],
    invariant_hits: &[SearchHit],
    last_failed_check: Option<&str>,
) -> WorkingContextBundle {
    let mut entries = Vec::new();
    if let Some(anchor) = anchor {
        push_entry(
            &mut entries,
            "anchor-task",
            "task-state",
            anchor.current_task.trim(),
            1.0,
            None,
        );
        push_entry(
            &mut entries,
            "anchor-next",
            "task-state",
            anchor.next_step.trim(),
            1.0,
            None,
        );
        if let Some(plan_pointer) = &anchor.plan_pointer {
            push_entry(
                &mut entries,
                "anchor-plan",
                "task-state",
                plan_pointer.trim(),
                1.0,
                None,
            );
        }
        for (index, decision) in anchor.last_decisions.iter().enumerate() {
            push_entry(
                &mut entries,
                &format!("anchor-decision-{index}"),
                "decision",
                decision.trim(),
                1.0,
                None,
            );
        }
    }
    if let Some(last_failed_check) = last_failed_check {
        push_entry(
            &mut entries,
            "last-failed-check",
            "task-state",
            last_failed_check.trim(),
            1.0,
            None,
        );
    }
    for hit in invariant_hits {
        push_memory_hit(&mut entries, hit, "constraint");
    }
    for hit in session_hits {
        push_memory_hit(&mut entries, hit, "fact");
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

fn push_entry(
    entries: &mut Vec<SnapshotEntry>,
    id: &str,
    slot: &str,
    content: &str,
    score: f32,
    unit_ref: Option<String>,
) {
    if content.is_empty() {
        return;
    }
    let mut entry = SnapshotEntry::now(id, slot, content, score);
    entry.unit_ref = unit_ref;
    entries.push(entry);
}

fn push_memory_hit(entries: &mut Vec<SnapshotEntry>, hit: &SearchHit, slot: &str) {
    if hit.record.content.trim().is_empty() {
        return;
    }
    entries.push(SnapshotEntry {
        id: hit.record.id.to_string(),
        slot: slot.to_string(),
        content: hit.record.content.clone(),
        tokens: count_tokens(&hit.record.content),
        score: hit.score,
        resolution: Resolution::Full,
        unit_ref: Some(hit.record.node_id.clone()),
        committed_at: hit.record.created_at,
        last_access: hit.record.last_access,
        access_count: hit.record.access_count,
    });
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct AnchorSetRequest {
    pub current_task: String,
    pub plan_pointer: Option<String>,
    pub last_decisions: Option<Vec<String>>,
    pub next_step: String,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct AnchorGetResponse {
    pub anchor: Option<AnchorPayload>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct AnchorPayload {
    pub current_task: String,
    pub plan_pointer: Option<String>,
    pub last_decisions: Vec<String>,
    pub next_step: String,
    pub updated_at: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SessionCheckpointRequest {
    pub agent_id: String,
    pub session_id: Option<String>,
    pub user_id: Option<String>,
    pub task_id: Option<String>,
    pub current_task: Option<String>,
    pub next_step: Option<String>,
    pub plan_pointer: Option<String>,
    pub last_decisions: Option<Vec<String>>,
    pub goal: Option<String>,
    pub last_failed_check: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SessionResumeRequest {
    pub session_id: Option<String>,
    pub user_id: Option<String>,
    pub task_id: Option<String>,
}

/// Resume by fuzzy task query instead of exact session_id.
///
/// Matches the query (case-insensitive substring) against stored sessions' `task_id` fields,
/// selects the most-recently-updated match, and returns the full resume packet. When multiple
/// sessions match, the most recent is returned and the alternatives are listed in `alternatives`.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SessionResumeByTaskRequest {
    /// Substring to match against session task ids/titles (e.g. "DPT-4477").
    pub task_query: String,
    /// Optional user_id to narrow the search scope.
    pub user_id: Option<String>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct SessionCheckpointResponse {
    pub summary: SessionSummaryPayload,
    pub packet: serde_json::Value,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct SessionResumeResponse {
    pub packet: serde_json::Value,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct SessionResumeByTaskResponse {
    /// The full resume packet (same shape as memory.session.resume).
    pub packet: serde_json::Value,
    /// The matched session's task_id.
    pub matched_task_id: String,
    /// The matched session's session_id.
    pub matched_session_id: String,
    /// Alternative session task_ids that also matched the query (excluding the selected one).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub alternatives: Vec<String>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct SessionSummaryPayload {
    pub user_id: String,
    pub session_id: String,
    pub task_id: String,
    pub updated_at: String,
    pub handed_off_from: Option<String>,
    pub entry_count: usize,
    pub token_count: Option<usize>,
}

// ── Token-savings MCP tool structs ────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SavingsRequest {
    /// Only count recalls at or after this UTC timestamp (ISO 8601, e.g.
    /// `"2026-01-01T00:00:00Z"`). Omit for all-time totals.
    pub since: Option<String>,
}

/// Per-operation token-savings breakdown returned by `memory.savings`.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct OpSavingsInfo {
    pub calls: u64,
    pub returned_total: u64,
    pub baseline_total: u64,
    pub saved_total: u64,
}

impl From<&OpSavings> for OpSavingsInfo {
    fn from(o: &OpSavings) -> Self {
        Self {
            calls: o.calls,
            returned_total: o.returned_total,
            baseline_total: o.baseline_total,
            saved_total: o.saved_total,
        }
    }
}

/// Response for `memory.savings`.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct SavingsResponse {
    /// Total number of recall operations measured.
    pub calls: u64,
    /// Sum of `returned_tokens` across all recalls.
    pub returned_total: u64,
    /// Sum of `baseline_tokens` across all recalls.
    pub baseline_total: u64,
    /// Sum of `saved_tokens` (`max(0, baseline - returned)`) across all recalls.
    pub saved_total: u64,
    /// Per-operation breakdown.
    pub by_op: std::collections::HashMap<String, OpSavingsInfo>,
    /// RFC 3339 timestamp of the first recorded recall, if any.
    pub first_ts: Option<String>,
    /// RFC 3339 timestamp of the most recent recorded recall, if any.
    pub last_ts: Option<String>,
}

impl From<TokenSavingsRollup> for SavingsResponse {
    fn from(r: TokenSavingsRollup) -> Self {
        Self {
            calls: r.calls,
            returned_total: r.returned_total,
            baseline_total: r.baseline_total,
            saved_total: r.saved_total,
            by_op: r.by_op.iter().map(|(k, v)| (k.clone(), v.into())).collect(),
            first_ts: r.first_ts.map(|dt| dt.to_rfc3339()),
            last_ts: r.last_ts.map(|dt| dt.to_rfc3339()),
        }
    }
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct KitGetResponse {
    pub vision: Option<String>,
    pub agents: Option<String>,
    pub anchor: Option<AnchorPayload>,
    pub kit_initialized: bool,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct KitSetRequest {
    /// Updated vision content to write to kit/vision.md.
    pub vision: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct DreamRequest {
    /// Collection to read records from. Defaults to the server-configured collection.
    pub collection: Option<String>,
    /// Output directory path for the OCF bundle (created if absent).
    pub out: String,
    /// Write a human-readable DREAMS.md narrative alongside the OCF files.
    #[serde(default)]
    pub diary: bool,
    /// Admission score threshold [0.0–1.0]. Records below this are logged as reject.
    pub admit_threshold: Option<f32>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct DreamResponse {
    /// Records read from the source collection.
    pub records_read: usize,
    /// Records admitted into the output OCF bundle.
    pub admitted: usize,
    /// Records rejected (score below threshold or superseded).
    pub rejected: usize,
    /// Whether the LLM synthesis pass ran.
    pub llm_ran: bool,
    /// Absolute path to the output bundle directory.
    pub out: String,
    /// Files written inside the bundle directory.
    pub files: Vec<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ToolsFindRequest {
    pub task: String,
    pub limit: Option<usize>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct ToolsFindResponse {
    pub tools: Vec<ToolMatch>,
    pub prompt_tokens_before: usize,
    pub prompt_tokens_after: usize,
    pub prompt_tokens_delta: isize,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct ToolMatch {
    pub name: String,
    pub description: String,
    pub score: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ProcessDefaults {
    registry_dir: PathBuf,
    max_concurrent_spawns: usize,
    max_lifetime: Duration,
    termination_grace: Duration,
}

impl Default for ProcessDefaults {
    fn default() -> Self {
        Self {
            registry_dir: PathBuf::from(".artesian").join("spawns"),
            max_concurrent_spawns: 32,
            max_lifetime: Duration::from_secs(30 * 60),
            termination_grace: Duration::from_secs(2),
        }
    }
}

impl ProcessDefaults {
    fn from_config(config: &ArtesianConfig) -> Self {
        let repo_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let registry_dir = config
            .coordination
            .spawn_registry_path
            .as_deref()
            .map(PathBuf::from)
            .map(|path| {
                if path.is_absolute() {
                    path
                } else {
                    repo_root.join(path)
                }
            })
            .unwrap_or_else(|| repo_root.join(".artesian").join("spawns"));
        Self {
            registry_dir,
            max_concurrent_spawns: config
                .coordination
                .max_concurrent_spawns
                .unwrap_or(32)
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
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, schemars::JsonSchema)]
pub struct AgentsListResponse {
    pub catalog: AgentCatalog,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct BindRequest {
    pub role: String,
    pub agent: String,
    pub model: String,
    pub command: Option<String>,
    pub args: Option<Vec<String>>,
    pub timeout_seconds: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, schemars::JsonSchema)]
pub struct BindResponse {
    pub binding: AgentBinding,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct DelegateRequest {
    pub role: String,
    pub task: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, schemars::JsonSchema)]
pub struct DelegateResponse {
    pub task_id: String,
    pub status: String,
    pub role: String,
    pub agent: String,
    pub model: Option<String>,
    pub result: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct StatusRequest {
    pub task_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, schemars::JsonSchema)]
pub struct StatusResponse {
    pub task_id: String,
    pub status: String,
    pub result: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct HandoffRequest {
    pub to: String,
    pub task_id: Option<String>,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, schemars::JsonSchema)]
pub struct HandoffResponse {
    pub accepted: bool,
    pub to: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct TeamCreateRequest {
    pub id: Option<String>,
    pub name: String,
    pub max_teammates: Option<usize>,
    pub plan_approval_required: Option<bool>,
    pub plan_approval_roles: Option<Vec<String>>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct TeamResponse {
    pub team: serde_json::Value,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct TeamSpawnRequest {
    pub team_id: String,
    pub definition: String,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct TeamSpawnResponse {
    pub teammate: serde_json::Value,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct TeamTaskAddRequest {
    pub team_id: String,
    pub title: String,
    pub description: Option<String>,
    pub definition: Option<String>,
    pub blockers: Option<Vec<String>>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct TeamTaskResponse {
    pub task_id: String,
    pub status: String,
    pub task: serde_json::Value,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct TeamTaskClaimRequest {
    pub team_id: String,
    pub task_id: Option<String>,
    pub teammate: String,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct TeamTaskClaimResponse {
    pub task: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct TeamTaskCompleteRequest {
    pub team_id: String,
    pub task_id: String,
    pub reviewer: String,
    pub approved: bool,
}

#[derive(Debug, Clone, Deserialize, schemars::JsonSchema)]
pub struct TeamTaskAwaitRequest {
    pub team_id: String,
    /// Singular task id for the common one-task wait path.
    pub task_id: Option<String>,
    /// Optional batch of task ids to wait for in one blocking MCP call.
    #[serde(default)]
    pub task_ids: Vec<String>,
    /// Maximum wait time in seconds. Defaults to 30 minutes; 0 performs one check then times out.
    pub timeout_secs: Option<u64>,
    /// Poll cadence in milliseconds. Defaults to 500 ms and is clamped to 50 ms..=5 s.
    pub poll_interval_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, schemars::JsonSchema)]
pub struct TeamTaskAwaitResponse {
    pub team_id: String,
    pub task_ids: Vec<String>,
    /// `completed`, `failed`, `timeout`, or `spawn-exited`.
    pub outcome: String,
    pub elapsed_ms: u64,
    pub tasks: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Copy, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum TeamMessageKindRequest {
    Ask,
    Result,
    Review,
    Done,
}

impl From<TeamMessageKindRequest> for TeamMessageKind {
    fn from(value: TeamMessageKindRequest) -> Self {
        match value {
            TeamMessageKindRequest::Ask => Self::Ask,
            TeamMessageKindRequest::Result => Self::Result,
            TeamMessageKindRequest::Review => Self::Review,
            TeamMessageKindRequest::Done => Self::Done,
        }
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct TeamMessageRequest {
    pub team_id: String,
    pub from: String,
    pub to: Option<String>,
    pub kind: TeamMessageKindRequest,
    pub content: String,
    pub task_id: Option<String>,
    pub approved: Option<bool>,
    pub execute: Option<bool>,
    pub resume_packet: Option<String>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct TeamMessageResponse {
    pub event: serde_json::Value,
    pub response: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub worker_events: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct TeamStatusRequest {
    pub team_id: String,
}

#[derive(Debug, Default, Deserialize, schemars::JsonSchema)]
pub struct TeamGcRequest {
    /// Reclaim spawns older than this many seconds, regardless of liveness
    /// (runaway-worker guard). Omit to disable the age bound.
    pub ttl_secs: Option<u64>,
    /// Reclaim spawns whose last heartbeat is older than this many seconds
    /// (hung-worker guard). Omit to disable the heartbeat bound.
    pub heartbeat_timeout_secs: Option<u64>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct TeamGcResponse {
    pub scanned: usize,
    pub terminated: usize,
    pub removed: usize,
    pub expired: usize,
    pub skipped_unverified: usize,
}

// ── team.presence / team.lane.* ───────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct TeamPresenceRequest {
    pub team_id: String,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct TeamPresenceResponse {
    pub presence: serde_json::Value,
}

/// Request to register a lane with a team.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct TeamLaneAddRequest {
    pub team_id: String,
    /// Stable unique lane name (e.g. `"security"`, `"test-runner"`).
    pub name: String,
    /// Role definition name this lane uses (must match a `.agent/agents` or `.claude/agents` file).
    pub definition: String,
    /// Human-readable description of what this lane owns.
    pub owned_scope: String,
    /// Things this lane must NOT do (non-goals prevent scope overlap).
    #[serde(default)]
    pub non_goals: Vec<String>,
    /// Maximum tasks this lane may run concurrently (within the global cap).
    pub max_concurrent_tasks: Option<usize>,
    /// Maximum total worker-turn budget across all tasks in this lane.
    pub max_turns: Option<u32>,
    /// Name of the lane (or teammate) to receive handoff summaries on task completion.
    pub handoff_to: Option<String>,
    /// Agent/tool names allowed in this lane (empty = no additional restriction).
    #[serde(default)]
    pub allowed_tools: Vec<String>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct TeamLaneResponse {
    pub lane: serde_json::Value,
}

/// Request to assign a task to a lane (enforcing dedup and budget).
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct TeamLaneAssignRequest {
    pub team_id: String,
    pub lane_name: String,
    pub task_id: String,
    pub task_title: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DelegateRecord {
    status: String,
    result: Option<String>,
}

// ── orchestrate.loop ──────────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct LoopRequest {
    /// Verifier command; exit 0 means the goal holds.
    pub goal: String,
    /// Per-iteration worker command (shell). Omit to poll-only.
    pub worker: Option<String>,
    /// Maximum turns before the loop gives up (default 10).
    pub max_turns: Option<u32>,
    /// Maximum wall-clock seconds before aborting (default unlimited).
    pub max_wall_secs: Option<u64>,
    /// Disable durable skill/spec/invariant learning for this run.
    pub no_learn: Option<bool>,
    /// Maximum consecutive verify failures before escalating with a failure trail
    /// (outcome `"escalated"`). Each failing turn injects `$ARTESIAN_LAST_FAILURE` into
    /// the next worker so retries target the specific failure. Defaults to 3. Set to 0
    /// to disable escalation (the loop will run to max_turns instead).
    pub max_remediation_attempts: Option<u32>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct LoopResponse {
    /// Outcome label: `"success"`, `"max-turns"`, `"wall-cap"`, `"stopped"`, `"error"`,
    /// or `"escalated"` when the remediation budget is exhausted.
    pub outcome: String,
    /// Human-readable explanation of why the loop stopped.  When `outcome == "escalated"`
    /// this includes a compact per-turn failure summary.
    pub why_stopped: String,
    /// Number of turns executed.
    pub turns: u32,
    /// Absolute path to the JSONL run log.
    pub run_log_path: String,
    /// Accumulated failure trail when `outcome == "escalated"`; empty otherwise.
    /// Each entry has `turn`, `reason` (bounded verifier output), and `fix_attempt`.
    pub failure_trail: Vec<serde_json::Value>,
}

/// Shell-backed [`LoopCommands`] for the MCP path — captures both stdout and stderr for the
/// verifier so failures are surfaced as the next turn's "last failed check" context.
struct McpShellLoopCommands;

impl loop_core::LoopCommands for McpShellLoopCommands {
    fn run_worker<'a>(
        &'a mut self,
        cmd: &'a str,
        env: Vec<(String, String)>,
        timeout: Option<Duration>,
    ) -> loop_core::LoopCommandFuture<'a, bool> {
        Box::pin(async move {
            let (success, _) = mcp_run_shell_capture(cmd, env, timeout).await?;
            Ok(success)
        })
    }

    fn verify_goal<'a>(
        &'a mut self,
        cmd: &'a str,
        timeout: Option<Duration>,
    ) -> loop_core::LoopCommandFuture<'a, (bool, String)> {
        Box::pin(async move { mcp_run_shell_capture(cmd, Vec::new(), timeout).await })
    }
}

async fn mcp_run_shell_capture(
    cmd: &str,
    env: Vec<(String, String)>,
    timeout: Option<Duration>,
) -> anyhow::Result<(bool, String)> {
    use std::process::Stdio;
    use tokio::process::Command;
    let mut command = Command::new("sh");
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
        Some(t) => tokio::time::timeout(t, command.output())
            .await
            .map_err(|_| {
                anyhow::anyhow!("loop exceeded wall-clock budget while running command: {cmd}")
            })?
            .map_err(|e| anyhow::anyhow!("run command: {cmd}: {e}"))?,
        None => command
            .output()
            .await
            .map_err(|e| anyhow::anyhow!("run command: {cmd}: {e}"))?,
    };
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    Ok((output.status.success(), text.trim().to_string()))
}

#[tool_router]
impl MemoryServer {
    #[tool(
        name = "memory.find",
        description = "Find durable project memories by query. ALWAYS search the project memory before non-trivial work."
    )]
    pub async fn memory_find(
        &self,
        Parameters(request): Parameters<FindRequest>,
    ) -> Result<Json<FindResponse>, ErrorData> {
        let mut query = MemoryQuery::new(request.query);
        query.limit = request.limit.unwrap_or(10);
        query.node_id = request.node_id;
        let expand = request.expand.unwrap_or(false);
        query.scope = request.scope.map(Into::into);
        query.agent_id = request.agent_id;
        query.session_id = request.session_id;
        query.task_id = request.task_id;
        query.user_id = request.user_id;
        let mut hits = self
            .backend
            .find(query)
            .await
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        if expand {
            hits = aquifer::expand_hits_with_neighbors(
                self.backend.as_ref(),
                hits,
                aquifer::DEFAULT_GRAPH_HOPS,
            )
            .await
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        }
        let baseline_tokens: usize = hits.iter().map(|h| count_tokens(&h.record.content)).sum();
        let hits: Vec<FindHit> = hits.into_iter().map(find_hit).collect();
        let returned_tokens: usize = hits.iter().map(|h| count_tokens(&h.content)).sum();
        record_savings(
            "memory.find",
            &self.collection,
            returned_tokens,
            baseline_tokens,
            self.track_savings,
        );
        Ok(Json(FindResponse { hits }))
    }

    #[tool(
        name = "memory.answer",
        description = "Answer one question from committed project memory. Uses retrieved chunks as grounding and cites source node_ids; without a configured LLM it returns an extractive answer."
    )]
    pub async fn memory_answer(
        &self,
        Parameters(request): Parameters<AnswerRequest>,
    ) -> Result<Json<AnswerResponse>, ErrorData> {
        let response = answer_memory(
            self.backend.as_ref(),
            &self.acc,
            &request.question,
            request.limit.unwrap_or(10),
        )
        .await
        .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        Ok(Json(response))
    }

    #[tool(
        name = "memory.qualify",
        description = "Run the ACC qualify-gate on one candidate without storing it. Returns admitted/rejected plus audited deterministic signals, agreement, chance-corrected agreement, confidence, and reason for use as a PreToolUse-style gate."
    )]
    pub async fn memory_qualify(
        &self,
        Parameters(request): Parameters<QualifyRequest>,
    ) -> Result<Json<QualifyResponse>, ErrorData> {
        let response = qualify_memory_candidate(
            self.backend.as_ref(),
            &self.acc,
            &request.candidate,
            request.goal.as_deref(),
        )
        .await
        .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        Ok(Json(response))
    }

    #[tool(
        name = "memory.context",
        description = "Return a compact index.md slice plus targeted memory.find hits; no LLM call is made."
    )]
    pub async fn memory_context(
        &self,
        Parameters(request): Parameters<ContextRequest>,
    ) -> Result<Json<ContextResponse>, ErrorData> {
        // Read the index, computing full-vs-truncated token counts for savings accounting.
        let (index, index_baseline_tokens, index_returned_tokens) =
            if let Some(root) = self.okf_root.as_ref() {
                match std::fs::read_to_string(root.join("memory").join("index.md")) {
                    Ok(full_index) => {
                        let index_limit = request.index_chars.unwrap_or(4_000);
                        let baseline = count_tokens(&full_index);
                        let truncated: String = full_index.chars().take(index_limit).collect();
                        let returned = count_tokens(&truncated);
                        (Some(truncated), baseline, returned)
                    }
                    Err(_) => (None, 0usize, 0usize),
                }
            } else {
                (None, 0usize, 0usize)
            };
        let mut query = MemoryQuery::new(request.query);
        query.limit = request.limit.unwrap_or(10);
        let expand = request.expand.unwrap_or(false);
        query.scope = request.scope.map(Into::into);
        query.agent_id = request.agent_id;
        query.session_id = request.session_id;
        query.task_id = request.task_id;
        query.user_id = request.user_id;
        let mut hits = self
            .backend
            .find(query)
            .await
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        if expand {
            hits = aquifer::expand_hits_with_neighbors(
                self.backend.as_ref(),
                hits,
                aquifer::DEFAULT_GRAPH_HOPS,
            )
            .await
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        }
        let hits_baseline: usize = hits.iter().map(|h| count_tokens(&h.record.content)).sum();
        let hits: Vec<FindHit> = hits.into_iter().map(find_hit).collect();
        let hits_returned: usize = hits.iter().map(|h| count_tokens(&h.content)).sum();
        record_savings(
            "memory.context",
            &self.collection,
            index_returned_tokens + hits_returned,
            index_baseline_tokens + hits_baseline,
            self.track_savings,
        );
        // When a goal is given, also surface the invariants relevant to it (tag-filtered), so the
        // caller can assemble a goal-scoped packet — goal + invariants + relevant memory.
        let mut invariants = Vec::new();
        if let Some(goal) = request.goal {
            let mut invariant_query = MemoryQuery::new(goal).with_limit(8);
            invariant_query.tags = vec!["invariant".to_string()];
            invariants = self
                .backend
                .find(invariant_query)
                .await
                .map_err(|error| ErrorData::internal_error(error.to_string(), None))?
                .into_iter()
                .map(find_hit)
                .collect();
        }
        Ok(Json(ContextResponse {
            index,
            hits,
            invariants,
        }))
    }

    #[tool(
        name = "memory.store",
        description = "Store durable, reusable learnings in project memory."
    )]
    pub async fn memory_store(
        &self,
        Parameters(request): Parameters<StoreRequest>,
    ) -> Result<Json<StoreResponse>, ErrorData> {
        let confidence = validate_confidence(request.confidence)?;
        let record = self
            .backend
            .store(StoreMemory {
                content: request.content,
                tags: request.tags.unwrap_or_default(),
                metadata: Default::default(),
                tier: MemoryTier::L1Atom,
                node_id: request.node_id,
                created_at: None,
                scope: request.scope.map(Into::into),
                agent_id: request.agent_id,
                session_id: request.session_id,
                task_id: request.task_id,
                user_id: request.user_id,
                source: request.source,
                confidence,
                relations: request
                    .relations
                    .unwrap_or_default()
                    .into_iter()
                    .map(Into::into)
                    .collect(),
            })
            .await
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        Ok(Json(StoreResponse {
            id: record.id.to_string(),
            node_id: record.node_id,
        }))
    }

    #[tool(
        name = "memory.commit",
        description = "Run one ACC commit-loop cycle: recall, qualify-gate, and admit relevant \
non-redundant knowledge into a bounded, schema-governed committed context. Returns the committed \
context to read plus per-cycle control metrics (admitted, rejected, footprint)."
    )]
    pub async fn memory_commit(
        &self,
        Parameters(request): Parameters<CommitRequest>,
    ) -> Result<Json<CommitResponse>, ErrorData> {
        let recall: Arc<dyn RecallStore> = Arc::new(MemoryRecallStore::new(self.backend.clone()));
        let mut config = HeadgateConfig::from(&self.acc);
        if let Some(budget) = request.budget_tokens {
            config.budget_tokens = budget;
        }
        if let Some(limit) = request.recall_limit {
            config.recall_limit = limit;
        }
        if let Some(score) = request.min_score {
            config.min_score = score;
        }
        let mut headgate = Headgate::new(recall, config);
        #[cfg(feature = "llm")]
        {
            if let Some(judge) = &self.acc.judge {
                let client = headgate::llm_client_from_config(judge)
                    .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
                headgate = headgate.with_gate(Arc::new(headgate::JudgeQualifyGate::new(client)));
            }
            if let Some(compressor) = &self.acc.compressor {
                let client = headgate::llm_client_from_config(compressor)
                    .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
                headgate = headgate.with_compressor(Arc::new(headgate::LlmCompressor::new(client)));
            }
        }
        let metrics = headgate
            .cycle(&request.query)
            .await
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        Ok(Json(CommitResponse {
            committed_context: headgate.render(),
            candidates: metrics.candidates,
            admitted: metrics.admitted,
            rejected_relevance: metrics.rejected_relevance,
            rejected_redundant: metrics.rejected_redundant,
            rejected_saturated: metrics.rejected_saturated,
            compressed: metrics.compressed,
            evicted: metrics.evicted,
            footprint_tokens: metrics.footprint_tokens,
            budget_tokens: metrics.budget_tokens,
        }))
    }

    #[tool(
        name = "memory.anchor.get",
        description = "Read the current Anchor session anchor from OKF log.md before resuming work."
    )]
    pub async fn memory_anchor_get(&self) -> Result<Json<AnchorGetResponse>, ErrorData> {
        let store = self.anchor_store.as_ref().ok_or_else(|| {
            ErrorData::internal_error("Anchor anchor store is not configured".to_string(), None)
        })?;
        let anchor = store
            .get()
            .await
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?
            .map(AnchorPayload::from);
        Ok(Json(AnchorGetResponse { anchor }))
    }

    #[tool(
        name = "memory.anchor.set",
        description = "Write the current task, plan pointer, decisions, and next step to OKF log.md."
    )]
    pub async fn memory_anchor_set(
        &self,
        Parameters(request): Parameters<AnchorSetRequest>,
    ) -> Result<Json<AnchorGetResponse>, ErrorData> {
        let store = self.anchor_store.as_ref().ok_or_else(|| {
            ErrorData::internal_error("Anchor anchor store is not configured".to_string(), None)
        })?;
        let mut anchor = SessionAnchor::new(request.current_task, request.next_step);
        anchor.plan_pointer = request.plan_pointer;
        anchor.last_decisions = request.last_decisions.unwrap_or_default();
        let anchor = store
            .set(anchor)
            .await
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        Ok(Json(AnchorGetResponse {
            anchor: Some(AnchorPayload::from(anchor)),
        }))
    }

    #[tool(
        name = "memory.session.checkpoint",
        description = "Checkpoint the current resumable session before yielding to another agent. \
Addresses the bundle by user_id/session_id/task_id, records this producer as handed_off_from, and \
stores the committed OCF snapshot for memory.session.resume."
    )]
    pub async fn memory_session_checkpoint(
        &self,
        Parameters(request): Parameters<SessionCheckpointRequest>,
    ) -> Result<Json<SessionCheckpointResponse>, ErrorData> {
        let key = SessionKey::new(
            request.user_id.clone(),
            request.session_id.clone(),
            request.task_id.clone(),
        );
        let anchor = checkpoint_anchor(self.anchor_store.as_ref(), &key, &request).await?;
        let query_text = request
            .goal
            .clone()
            .unwrap_or_else(|| format!("{} {}", anchor.current_task, anchor.next_step));
        let limit = request.limit.unwrap_or(8);
        let session_hits =
            session_scoped_hits(self.backend.as_ref(), &key, &query_text, limit).await?;
        let invariant_hits = invariant_hits(self.backend.as_ref(), &key, &query_text).await?;
        let bundle = build_session_bundle(
            Some(&anchor),
            &session_hits,
            &invariant_hits,
            request.last_failed_check.as_deref(),
        );
        let session = bundle
            .to_ocf_session(&key, Some(request.agent_id))
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        let packet = WorkingContextBundle::resume_packet_from_session(&session)
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        let summary = SessionStore::new(self.backend.clone())
            .store(session)
            .await
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        Ok(Json(SessionCheckpointResponse {
            summary: SessionSummaryPayload::from(summary),
            packet,
        }))
    }

    #[tool(
        name = "memory.session.resume",
        description = "Resume a committed cross-agent session by session_id plus optional user_id \
and task_id. Restores the OCF snapshot without re-qualifying and never matches on agent_id."
    )]
    pub async fn memory_session_resume(
        &self,
        Parameters(request): Parameters<SessionResumeRequest>,
    ) -> Result<Json<SessionResumeResponse>, ErrorData> {
        let key = SessionKey::new(request.user_id, request.session_id, request.task_id);
        let store = SessionStore::new(self.backend.clone());
        let Some(session) = store
            .load(&key)
            .await
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?
        else {
            return Err(ErrorData::invalid_params(
                format!(
                    "no resumable session for user_id={} session_id={} task_id={}",
                    key.user_id, key.session_id, key.task_id
                ),
                None,
            ));
        };
        let packet = WorkingContextBundle::resume_packet_from_session(&session)
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        // Baseline = returned for session resume (the packet IS the full bounded output).
        let packet_tokens = count_tokens(&packet.to_string());
        record_savings(
            "memory.session.resume",
            &self.collection,
            packet_tokens,
            packet_tokens,
            self.track_savings,
        );
        Ok(Json(SessionResumeResponse { packet }))
    }

    #[tool(
        name = "memory.session.resume_by_task",
        description = "Resume a cross-agent session by fuzzy task query (case-insensitive substring \
match on task_id/title, recency tiebreak). The operator says 'continue the DPT-4477 work' and the \
agent resumes the right session WITHOUT knowing the session_id. Returns the full resume packet plus \
the matched task/session ids and any alternative matches."
    )]
    pub async fn memory_session_resume_by_task(
        &self,
        Parameters(request): Parameters<SessionResumeByTaskRequest>,
    ) -> Result<Json<SessionResumeByTaskResponse>, ErrorData> {
        use aquifer::SessionListFilter;
        let store = SessionStore::new(self.backend.clone());
        let all_summaries = store
            .list(SessionListFilter {
                user_id: request.user_id.clone(),
                ..SessionListFilter::default()
            })
            .await
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        let query_lower = request.task_query.to_lowercase();
        let mut matches: Vec<_> = all_summaries
            .iter()
            .filter(|summary| summary.key.task_id.to_lowercase().contains(&query_lower))
            .collect();
        if matches.is_empty() {
            return Err(ErrorData::invalid_params(
                format!("no session matched task query {:?}", request.task_query),
                None,
            ));
        }
        // Most-recently-updated first.
        matches.sort_by_key(|summary| std::cmp::Reverse(summary.updated_at));
        let best = matches[0];
        let alternatives: Vec<String> = matches[1..]
            .iter()
            .map(|summary| {
                format!(
                    "{} / {} (updated {})",
                    summary.key.session_id,
                    summary.key.task_id,
                    summary.updated_at.to_rfc3339()
                )
            })
            .collect();
        let Some(session) = store
            .load(&best.key)
            .await
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?
        else {
            return Err(ErrorData::internal_error(
                format!("session {} listed but load failed", best.key.session_id),
                None,
            ));
        };
        let packet = WorkingContextBundle::resume_packet_from_session(&session)
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        let packet_tokens = count_tokens(&packet.to_string());
        record_savings(
            "memory.session.resume_by_task",
            &self.collection,
            packet_tokens,
            packet_tokens,
            self.track_savings,
        );
        Ok(Json(SessionResumeByTaskResponse {
            packet,
            matched_task_id: best.key.task_id.clone(),
            matched_session_id: best.key.session_id.clone(),
            alternatives,
        }))
    }

    #[tool(
        name = "memory.savings",
        description = "Return cumulative token-savings statistics: how many tokens Artesian's \
targeted recall saved vs loading the full source records. Baseline assumption: each hit's full \
record content token count is the baseline; the actual response payload is the returned count; \
saved = max(0, baseline - returned). Pass `since` (ISO 8601) for a time-windowed view."
    )]
    pub async fn memory_savings(
        &self,
        Parameters(request): Parameters<SavingsRequest>,
    ) -> Result<Json<SavingsResponse>, ErrorData> {
        let since = request
            .since
            .as_deref()
            .map(|s| {
                chrono::DateTime::parse_from_rfc3339(s)
                    .map(|dt| dt.with_timezone(&chrono::Utc))
                    .map_err(|e| {
                        ErrorData::invalid_params(format!("invalid since timestamp: {e}"), None)
                    })
            })
            .transpose()?;
        let rollup = load_savings_rollup(since);
        Ok(Json(SavingsResponse::from(rollup)))
    }

    #[tool(
        name = "memory.kit.get",
        description = "Return the loop memory kit: vision.md, agents.md, and last session anchor. \
Load at session start so the agent has immediate context without re-reading all memory."
    )]
    pub async fn memory_kit_get(&self) -> Result<Json<KitGetResponse>, ErrorData> {
        let kit_root = self
            .okf_root
            .as_deref()
            .map(|r| r.join("kit"))
            .or_else(|| Some(std::path::PathBuf::from(".artesian").join("kit")));
        let kit_root = kit_root.unwrap();

        let vision = kit_root
            .join("vision.md")
            .exists()
            .then(|| std::fs::read_to_string(kit_root.join("vision.md")).ok())
            .flatten();
        let agents = kit_root
            .join("agents.md")
            .exists()
            .then(|| std::fs::read_to_string(kit_root.join("agents.md")).ok())
            .flatten();
        let kit_initialized = vision.is_some() || agents.is_some();

        let anchor = if let Some(store) = &self.anchor_store {
            store.get().await.ok().flatten().map(AnchorPayload::from)
        } else {
            None
        };

        Ok(Json(KitGetResponse {
            vision,
            agents,
            anchor,
            kit_initialized,
        }))
    }

    #[tool(
        name = "memory.kit.set",
        description = "Update the loop memory kit vision (writes kit/vision.md). \
Call when the project vision or current phase changes."
    )]
    pub async fn memory_kit_set(
        &self,
        Parameters(request): Parameters<KitSetRequest>,
    ) -> Result<Json<KitGetResponse>, ErrorData> {
        let kit_root = self
            .okf_root
            .as_deref()
            .map(|r| r.join("kit"))
            .or_else(|| Some(std::path::PathBuf::from(".artesian").join("kit")));
        let kit_root = kit_root.unwrap();

        std::fs::create_dir_all(&kit_root)
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;

        if let Some(vision) = &request.vision {
            std::fs::write(kit_root.join("vision.md"), vision)
                .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        }

        // Re-read and return the current kit state.
        let vision = kit_root
            .join("vision.md")
            .exists()
            .then(|| std::fs::read_to_string(kit_root.join("vision.md")).ok())
            .flatten();
        let agents = kit_root
            .join("agents.md")
            .exists()
            .then(|| std::fs::read_to_string(kit_root.join("agents.md")).ok())
            .flatten();
        Ok(Json(KitGetResponse {
            kit_initialized: vision.is_some(),
            vision,
            agents,
            anchor: None,
        }))
    }

    #[tool(
        name = "tools.find",
        description = "Opt-in router: return only MCP tools relevant to a task and estimate prompt-token savings."
    )]
    pub async fn tools_find(
        &self,
        Parameters(request): Parameters<ToolsFindRequest>,
    ) -> Result<Json<ToolsFindResponse>, ErrorData> {
        if !self.router_enabled {
            return Err(ErrorData::internal_error(
                "tools.find router is disabled by config".to_string(),
                None,
            ));
        }
        let limit = request.limit.unwrap_or(3).max(1);
        let mut tools = tool_registry()
            .iter()
            .map(|tool| ToolMatch {
                name: tool.name.to_string(),
                description: tool.description.to_string(),
                score: lexical_score(&request.task, tool.description),
            })
            .filter(|tool| tool.score > 0.0)
            .collect::<Vec<_>>();
        tools.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        tools.truncate(limit);
        let prompt_tokens_before = estimate_tokens(&request.task)
            + tool_registry()
                .iter()
                .map(|tool| estimate_tokens(tool.description))
                .sum::<usize>();
        let prompt_tokens_after = estimate_tokens(&request.task)
            + tools
                .iter()
                .map(|tool| estimate_tokens(&tool.description))
                .sum::<usize>();
        Ok(Json(ToolsFindResponse {
            tools,
            prompt_tokens_before,
            prompt_tokens_after,
            prompt_tokens_delta: prompt_tokens_before as isize - prompt_tokens_after as isize,
        }))
    }

    #[tool(
        name = "agents.list",
        description = "List reachable configured agent CLIs and their available models."
    )]
    pub async fn agents_list(&self) -> Result<Json<AgentsListResponse>, ErrorData> {
        self.ensure_orchestration_enabled()?;
        let catalog = self
            .catalog
            .lock()
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?
            .clone();
        Ok(Json(AgentsListResponse { catalog }))
    }

    #[tool(
        name = "orchestrate.bind",
        description = "Bind a role to a reachable agent/model for this MCP session."
    )]
    pub async fn orchestrate_bind(
        &self,
        Parameters(request): Parameters<BindRequest>,
    ) -> Result<Json<BindResponse>, ErrorData> {
        self.ensure_orchestration_enabled()?;
        let role = Role::from_str(&request.role)
            .map_err(|error| ErrorData::invalid_params(error.to_string(), None))?;
        let binding = AgentBinding {
            role,
            agent: request.agent,
            model: Some(request.model),
            command: request.command,
            args: request.args.unwrap_or_default(),
            timeout_seconds: request.timeout_seconds,
        };
        let mut bindings = self
            .bindings
            .lock()
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        let mut next_bindings = bindings.clone();
        next_bindings.retain(|existing| existing.role != role);
        next_bindings.push(binding.clone());
        let mut catalog = self
            .catalog
            .lock()
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?
            .clone();
        if !catalog
            .agents
            .iter()
            .any(|entry| entry.agent == binding.agent)
        {
            catalog
                .agents
                .extend(fallback_agent_catalog(std::slice::from_ref(&binding)).agents);
        }
        validate_binding_model(&binding, &catalog)
            .map_err(|error| ErrorData::invalid_params(error.to_string(), None))?;
        *bindings = next_bindings;
        *self
            .catalog
            .lock()
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))? = catalog;
        Ok(Json(BindResponse { binding }))
    }

    #[tool(
        name = "orchestrate.delegate",
        description = "Enqueue and dispatch one bounded task through the supervised role agent."
    )]
    pub async fn orchestrate_delegate(
        &self,
        peer: Peer<RoleServer>,
        meta: Meta,
        ct: CancellationToken,
        Parameters(request): Parameters<DelegateRequest>,
    ) -> Result<Json<DelegateResponse>, ErrorData> {
        let progress_token = meta.get_progress_token();
        let heartbeat = spawn_progress_heartbeat(
            peer,
            progress_token,
            Duration::from_secs(5),
            "orchestrate.delegate running",
        );
        let result = self.orchestrate_delegate_inner(request, ct).await;
        heartbeat.abort();
        result
    }

    /// Core implementation of `orchestrate.delegate`, callable without MCP context for tests.
    pub async fn orchestrate_delegate_inner(
        &self,
        request: DelegateRequest,
        ct: CancellationToken,
    ) -> Result<Json<DelegateResponse>, ErrorData> {
        self.ensure_orchestration_enabled()?;
        let role = Role::from_str(&request.role)
            .map_err(|error| ErrorData::invalid_params(error.to_string(), None))?;
        let binding = self.binding_for_role(role)?;
        let catalog = self
            .catalog
            .lock()
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?
            .clone();
        validate_binding_model(&binding, &catalog)
            .map_err(|error| ErrorData::invalid_params(error.to_string(), None))?;
        let task_id = format!("mcp-{}-{}", role.canonical_alias(), now_id());
        let task_store = FilesTaskStore::new(&self.task_root);
        let mut task = NewTask::primitive(first_line_or_default(&request.task));
        task.id = Some(task_id.clone());
        task.role = role;
        task.description = request.task.clone();
        let _task = task_store
            .create(task)
            .await
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        let _claimed = task_store
            .claim(ClaimRequest {
                task_id: Some(task_id.clone()),
                claimant: format!("mcp-{}", role.canonical_alias()),
            })
            .await
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?
            .ok_or_else(|| ErrorData::internal_error("task was not claimable".to_string(), None))?;
        let process = self.process_agent_for_binding(&binding);
        let session = process
            .spawn(SpawnRequest {
                role,
                agent: binding.agent.clone(),
                model: binding.model.clone(),
                working_dir: Some(self.repo_root.display().to_string()),
                resume_packet: None,
            })
            .await
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        // Drive the agent call inside a select so a client cancel returns promptly.
        // Dropping `session` when the cancel branch fires triggers process-group cleanup.
        let response = tokio::select! {
            result = process.send(
                &session,
                AgentMessage {
                    content: format!(
                        "Task ID: {task_id}\nRole: {}\n\n{}",
                        role.canonical_alias(),
                        request.task
                    ),
                },
            ) => result,
            () = ct.cancelled() => {
                let _ = task_store
                    .transition(TransitionTask {
                        id: task_id.clone(),
                        status: TaskStatus::Blocked,
                    })
                    .await;
                self.record_delegate(task_id, "cancelled".to_string(), None)?;
                return Err(ErrorData::invalid_request(
                    "request cancelled by client".to_string(),
                    None,
                ));
            }
        };
        match response {
            Ok(response) => {
                task_store
                    .transition(TransitionTask {
                        id: task_id.clone(),
                        status: TaskStatus::Done,
                    })
                    .await
                    .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
                self.record_delegate(
                    task_id.clone(),
                    "done".to_string(),
                    Some(response.content.clone()),
                )?;
                Ok(Json(DelegateResponse {
                    task_id,
                    status: "done".to_string(),
                    role: role.canonical_alias().to_string(),
                    agent: binding.agent,
                    model: binding.model,
                    result: Some(response.content),
                }))
            }
            Err(error) => {
                task_store
                    .transition(TransitionTask {
                        id: task_id.clone(),
                        status: TaskStatus::Blocked,
                    })
                    .await
                    .map_err(|task_error| {
                        ErrorData::internal_error(task_error.to_string(), None)
                    })?;
                self.record_delegate(task_id, "blocked".to_string(), None)?;
                Err(ErrorData::internal_error(error.to_string(), None))
            }
        }
    }

    #[tool(
        name = "orchestrate.status",
        description = "Return status for a delegated MCP orchestration task."
    )]
    pub async fn orchestrate_status(
        &self,
        Parameters(request): Parameters<StatusRequest>,
    ) -> Result<Json<StatusResponse>, ErrorData> {
        self.ensure_orchestration_enabled()?;
        let records = self
            .delegate_results
            .lock()
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        let record = records
            .get(&request.task_id)
            .cloned()
            .unwrap_or(DelegateRecord {
                status: "unknown".to_string(),
                result: None,
            });
        Ok(Json(StatusResponse {
            task_id: request.task_id,
            status: record.status,
            result: record.result,
        }))
    }

    #[tool(
        name = "orchestrate.handoff",
        description = "Record a handoff to judge or master for follow-up orchestration."
    )]
    pub async fn orchestrate_handoff(
        &self,
        Parameters(request): Parameters<HandoffRequest>,
    ) -> Result<Json<HandoffResponse>, ErrorData> {
        self.ensure_orchestration_enabled()?;
        let to = Role::from_str(&request.to)
            .map_err(|error| ErrorData::invalid_params(error.to_string(), None))?;
        if !matches!(to, Role::Judge | Role::Master) {
            return Err(ErrorData::invalid_params(
                "handoff target must be judge or master".to_string(),
                None,
            ));
        }
        let _ = self
            .backend
            .store(StoreMemory {
                content: request.content,
                tags: vec!["orchestration".to_string(), "handoff".to_string()],
                metadata: Default::default(),
                tier: MemoryTier::L1Atom,
                node_id: request
                    .task_id
                    .map(|task_id| format!("task:{task_id}:handoff:{}", to.canonical_alias())),
                created_at: None,
                scope: Some(MemoryScope::Task),
                agent_id: Some(to.canonical_alias().to_string()),
                session_id: None,
                task_id: None,
                user_id: None,
                source: None,
                confidence: None,
                relations: Vec::new(),
            })
            .await;
        Ok(Json(HandoffResponse {
            accepted: true,
            to: to.canonical_alias().to_string(),
        }))
    }

    #[tool(
        name = "orchestrate.loop",
        description = "Run the Artesian autonomous memory-first agentic loop through the MCP server. \
Repeats `worker` until `goal` exits 0 (or a brake fires). Each turn: recalls goal-relevant memory, \
assembles a bounded goal packet, runs `worker` with ARTESIAN_PACKET/ARTESIAN_GOAL/ARTESIAN_RECALL/\
ARTESIAN_TURN env vars, writes a resume anchor, verifies the goal, and (on success) commits a \
verified skill + spec + auto-invariants to memory. Brakes: max_turns (default 10), max_wall_secs \
(default unlimited), ~/.artesian/STOP sentinel. Same implementation as the CLI `artesian loop` command."
    )]
    pub async fn orchestrate_loop(
        &self,
        peer: Peer<RoleServer>,
        meta: Meta,
        ct: CancellationToken,
        Parameters(request): Parameters<LoopRequest>,
    ) -> Result<Json<LoopResponse>, ErrorData> {
        let progress_token = meta.get_progress_token();
        let on_progress = make_mcp_progress_callback(peer.clone(), progress_token.clone());
        // Heartbeat so the client sees a signal even when a single worker turn is long.
        let heartbeat = spawn_progress_heartbeat(
            peer,
            progress_token,
            Duration::from_secs(5),
            "orchestrate.loop running",
        );
        let result = self.orchestrate_loop_inner(request, ct, on_progress).await;
        heartbeat.abort();
        result
    }

    /// Core implementation of `orchestrate.loop`, exposed without MCP context so tests can
    /// call it directly without needing a `Peer<RoleServer>`.
    pub async fn orchestrate_loop_inner(
        &self,
        request: LoopRequest,
        cancel: CancellationToken,
        on_progress: Option<loop_core::LoopProgressCallback>,
    ) -> Result<Json<LoopResponse>, ErrorData> {
        self.ensure_orchestration_enabled()?;
        let run_id = loop_core::loop_run_id();
        let run_log_dir = loop_core::loop_run_log_dir()
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        let stop_file = loop_core::loop_stop_file()
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        let anchor_store = AnchorAnchorStore::new(&self.repo_root);
        // Re-use the server's already-open memory backend for recall/commit.
        let backend_ref: &dyn aquifer::MemoryBackend = self.backend.as_ref();
        let options = loop_core::LoopRunOptions {
            goal: request.goal,
            worker_cmd: request.worker,
            max_turns: request.max_turns.unwrap_or(10),
            max_wall: request.max_wall_secs.map(Duration::from_secs),
            poll: false,
            learn: !request.no_learn.unwrap_or(false),
            run_id,
            run_log_dir,
            stop_file,
            collection: self.collection.clone(),
            track_savings: self.track_savings,
            max_remediation_attempts: request
                .max_remediation_attempts
                .unwrap_or(loop_core::LOOP_REMEDIATION_ATTEMPTS_DEFAULT),
            cancel,
            on_progress,
        };
        let mut commands = McpShellLoopCommands;
        let report =
            loop_core::run_loop_core(options, Some(backend_ref), &anchor_store, &mut commands)
                .await
                .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        let failure_trail: Vec<serde_json::Value> = report
            .failure_trail
            .iter()
            .map(|a| {
                serde_json::json!({
                    "turn": a.turn,
                    "reason": a.reason,
                    "fix_attempt": a.fix_attempt,
                })
            })
            .collect();
        Ok(Json(LoopResponse {
            outcome: report.outcome,
            why_stopped: report.why_stopped,
            turns: report.turns,
            run_log_path: report.run_log_path.display().to_string(),
            failure_trail,
        }))
    }

    #[tool(
        name = "team.create",
        description = "Create an opt-in Flume team topology for orchestrate/full mode."
    )]
    pub async fn team_create(
        &self,
        Parameters(request): Parameters<TeamCreateRequest>,
    ) -> Result<Json<TeamResponse>, ErrorData> {
        self.ensure_orchestration_enabled()?;
        let mut runtime = self.team_runtime.lock().await;
        let team = runtime.create_team(TeamCreate {
            id: request.id,
            name: request.name,
            max_teammates: request.max_teammates,
            plan_approval_required: request.plan_approval_required.unwrap_or(false),
            plan_approval_roles: request.plan_approval_roles.unwrap_or_default(),
        });
        Ok(Json(TeamResponse {
            team: serde_json::to_value(team)
                .map_err(|error| ErrorData::internal_error(error.to_string(), None))?,
        }))
    }

    #[tool(
        name = "team.spawn",
        description = "Admit and spawn a teammate from a .agent/agents or .claude/agents definition through the supervised ProcessAgent path. After adding work for a spawned teammate, block on team.task.await until it completes so the client request remains active and interruptible; for one-off work, prefer orchestrate.delegate."
    )]
    pub async fn team_spawn(
        &self,
        Parameters(request): Parameters<TeamSpawnRequest>,
    ) -> Result<Json<TeamSpawnResponse>, ErrorData> {
        self.ensure_orchestration_enabled()?;
        let mut runtime = self.team_runtime.lock().await;
        let teammate = runtime
            .spawn_teammate(TeamSpawn {
                team_id: request.team_id,
                definition: request.definition,
            })
            .await
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        Ok(Json(TeamSpawnResponse {
            teammate: serde_json::to_value(teammate)
                .map_err(|error| ErrorData::internal_error(error.to_string(), None))?,
        }))
    }

    #[tool(
        name = "team.task.add",
        description = "Add a task to the shared headrace task board for a Flume team. After adding a teammate task, immediately call team.task.await for that task instead of ending the turn or polling manually; the blocking wait keeps the request active and interruptible."
    )]
    pub async fn team_task_add(
        &self,
        Parameters(request): Parameters<TeamTaskAddRequest>,
    ) -> Result<Json<TeamTaskResponse>, ErrorData> {
        self.ensure_orchestration_enabled()?;
        let mut runtime = self.team_runtime.lock().await;
        let task = runtime
            .add_task(TeamTaskAdd {
                team_id: request.team_id,
                title: request.title,
                description: request.description.unwrap_or_default(),
                definition: request.definition,
                blockers: request.blockers.unwrap_or_default(),
            })
            .await
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        Ok(Json(TeamTaskResponse {
            task_id: task.id.clone(),
            status: format!("{:?}", task.status).to_ascii_lowercase(),
            task: serde_json::to_value(task)
                .map_err(|error| ErrorData::internal_error(error.to_string(), None))?,
        }))
    }

    #[tool(
        name = "team.task.claim",
        description = "Atomically claim an eligible team task, respecting opt-in plan approval gates."
    )]
    pub async fn team_task_claim(
        &self,
        Parameters(request): Parameters<TeamTaskClaimRequest>,
    ) -> Result<Json<TeamTaskClaimResponse>, ErrorData> {
        self.ensure_orchestration_enabled()?;
        let mut runtime = self.team_runtime.lock().await;
        let task = runtime
            .claim_task(TeamTaskClaim {
                team_id: request.team_id,
                task_id: request.task_id,
                teammate: request.teammate,
            })
            .await
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        Ok(Json(TeamTaskClaimResponse {
            task: task
                .map(serde_json::to_value)
                .transpose()
                .map_err(|error| ErrorData::internal_error(error.to_string(), None))?,
        }))
    }

    #[tool(
        name = "team.task.complete",
        description = "Complete or block a team task through the judge/master gate."
    )]
    pub async fn team_task_complete(
        &self,
        Parameters(request): Parameters<TeamTaskCompleteRequest>,
    ) -> Result<Json<TeamTaskResponse>, ErrorData> {
        self.ensure_orchestration_enabled()?;
        let mut runtime = self.team_runtime.lock().await;
        let task = runtime
            .complete_task(TeamTaskComplete {
                team_id: request.team_id,
                task_id: request.task_id,
                reviewer: request.reviewer,
                approved: request.approved,
            })
            .await
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        Ok(Json(TeamTaskResponse {
            task_id: task.id.clone(),
            status: format!("{:?}", task.status).to_ascii_lowercase(),
            task: serde_json::to_value(task)
                .map_err(|error| ErrorData::internal_error(error.to_string(), None))?,
        }))
    }

    #[tool(
        name = "team.task.await",
        description = "Block on one or more team tasks until every task reaches a terminal state (`done` or `blocked`), a matching task spawn exits, or timeout/cancel fires. Use this immediately after team.task.add so the MCP request stays active and interruptible; do not end the master turn or poll team.status in a loop."
    )]
    pub async fn team_task_await(
        &self,
        peer: Peer<RoleServer>,
        meta: Meta,
        ct: CancellationToken,
        Parameters(request): Parameters<TeamTaskAwaitRequest>,
    ) -> Result<Json<TeamTaskAwaitResponse>, ErrorData> {
        let progress_token = meta.get_progress_token();
        let heartbeat = spawn_progress_heartbeat(
            peer,
            progress_token,
            Duration::from_secs(5),
            "team.task.await waiting",
        );
        let result = self.team_task_await_inner(request, ct, None).await;
        heartbeat.abort();
        result
    }

    /// Core implementation of `team.task.await`, callable without MCP context for tests.
    pub async fn team_task_await_inner(
        &self,
        request: TeamTaskAwaitRequest,
        ct: CancellationToken,
        on_progress: Option<loop_core::LoopProgressCallback>,
    ) -> Result<Json<TeamTaskAwaitResponse>, ErrorData> {
        self.ensure_orchestration_enabled()?;
        let task_ids = normalize_await_task_ids(&request)?;
        {
            let runtime = self.team_runtime.lock().await;
            runtime
                .status(&request.team_id)
                .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        }

        let timeout = await_timeout(request.timeout_secs);
        let poll_interval = await_poll_interval(request.poll_interval_ms);
        let deadline = tokio::time::Instant::now() + timeout;
        let started = std::time::Instant::now();
        let task_store = FilesTaskStore::new(&self.task_root);
        let mut saw_matching_spawn = false;

        loop {
            let tasks = load_await_tasks(&task_store, &task_ids).await?;
            if tasks.iter().all(|task| task.status.is_terminal()) {
                let outcome = if tasks.iter().any(|task| task.status == TaskStatus::Blocked) {
                    "failed"
                } else {
                    "completed"
                };
                return await_response(&request.team_id, &task_ids, outcome, started, tasks);
            }

            let active_spawns = self.active_matching_spawn_count(&task_ids)?;
            if active_spawns > 0 {
                saw_matching_spawn = true;
            } else if saw_matching_spawn {
                return await_response(&request.team_id, &task_ids, "spawn-exited", started, tasks);
            }

            if let Some(on_progress) = &on_progress {
                on_progress(
                    started.elapsed().as_secs_f64(),
                    (timeout > Duration::ZERO).then_some(timeout.as_secs_f64()),
                    Some(format!(
                        "waiting for {} team task(s) to finish",
                        task_ids.len()
                    )),
                );
            }

            tokio::select! {
                () = ct.cancelled() => {
                    return Err(ErrorData::invalid_request(
                        "request cancelled by client".to_string(),
                        None,
                    ));
                }
                () = tokio::time::sleep_until(deadline) => {
                    let tasks = load_await_tasks(&task_store, &task_ids).await?;
                    return await_response(&request.team_id, &task_ids, "timeout", started, tasks);
                }
                () = tokio::time::sleep(poll_interval) => {}
            }
        }
    }

    fn active_matching_spawn_count(&self, task_ids: &[String]) -> Result<usize, ErrorData> {
        let supervisor = ProcessSupervisor::new(self.process_defaults.registry_dir.clone())
            .with_termination_grace(self.process_defaults.termination_grace)
            .with_max_concurrent_spawns(self.process_defaults.max_concurrent_spawns);
        let entries = supervisor
            .entries()
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        Ok(entries
            .into_iter()
            .filter(|entry| {
                entry
                    .task_id
                    .as_ref()
                    .is_some_and(|task_id| task_ids.iter().any(|id| id == task_id))
                    && supervisor.group_alive(entry.pgid)
            })
            .count())
    }

    #[tool(
        name = "team.message",
        description = "Post a typed Flume message (ASK/RESULT/REVIEW/DONE); direct addressing rides the shared EventEnvelope pool."
    )]
    pub async fn team_message(
        &self,
        peer: Peer<RoleServer>,
        meta: Meta,
        ct: CancellationToken,
        Parameters(request): Parameters<TeamMessageRequest>,
    ) -> Result<Json<TeamMessageResponse>, ErrorData> {
        // Only wire heartbeat and cancel for the blocking execute=true variant.
        let execute = request.execute.unwrap_or(false);
        let progress_token = if execute {
            meta.get_progress_token()
        } else {
            None
        };
        let heartbeat = spawn_progress_heartbeat(
            peer,
            progress_token,
            Duration::from_secs(5),
            "team.message executing",
        );
        let cancel = if execute {
            ct
        } else {
            CancellationToken::new()
        };
        let result = self.team_message_inner(request, cancel).await;
        heartbeat.abort();
        result
    }

    /// Core implementation of `team.message`, callable without MCP context for tests.
    pub async fn team_message_inner(
        &self,
        request: TeamMessageRequest,
        ct: CancellationToken,
    ) -> Result<Json<TeamMessageResponse>, ErrorData> {
        self.ensure_orchestration_enabled()?;
        // Wrap in select! so a client cancel releases the runtime mutex promptly.
        let outcome = tokio::select! {
            result = async {
                let mut runtime = self.team_runtime.lock().await;
                runtime
                    .message(TeamMessage {
                        team_id: request.team_id,
                        from: request.from,
                        to: request.to,
                        kind: request.kind.into(),
                        content: request.content,
                        task_id: request.task_id,
                        approved: request.approved,
                        execute: request.execute.unwrap_or(false),
                        resume_packet: request.resume_packet,
                    })
                    .await
                    .map_err(|error| ErrorData::internal_error(error.to_string(), None))
            } => result?,
            () = ct.cancelled() => {
                return Err(ErrorData::invalid_request(
                    "request cancelled by client".to_string(),
                    None,
                ));
            }
        };
        Ok(Json(TeamMessageResponse {
            event: serde_json::to_value(outcome.event)
                .map_err(|error| ErrorData::internal_error(error.to_string(), None))?,
            response: outcome.response,
            worker_events: outcome
                .worker_events
                .into_iter()
                .map(serde_json::to_value)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| ErrorData::internal_error(error.to_string(), None))?,
        }))
    }

    #[tool(
        name = "team.status",
        description = "Return Flume team lifecycle, teammates, and redacted EventEnvelope pool. This remains a non-blocking snapshot; do not poll it in a loop after team.task.add. Use team.task.await to block on teammate completion, or orchestrate.delegate for a single worker."
    )]
    pub async fn team_status(
        &self,
        Parameters(request): Parameters<TeamStatusRequest>,
    ) -> Result<Json<TeamResponse>, ErrorData> {
        self.ensure_orchestration_enabled()?;
        let runtime = self.team_runtime.lock().await;
        let team = runtime
            .status(&request.team_id)
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        Ok(Json(TeamResponse {
            team: serde_json::to_value(team)
                .map_err(|error| ErrorData::internal_error(error.to_string(), None))?,
        }))
    }

    #[tool(
        name = "team.cleanup",
        description = "Terminate tracked teammate process groups for the current owner and mark the Flume team cleaned up."
    )]
    pub async fn team_cleanup(
        &self,
        Parameters(request): Parameters<TeamStatusRequest>,
    ) -> Result<Json<TeamResponse>, ErrorData> {
        self.ensure_orchestration_enabled()?;
        let mut runtime = self.team_runtime.lock().await;
        let team = runtime
            .cleanup(&request.team_id)
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        Ok(Json(TeamResponse {
            team: serde_json::to_value(team)
                .map_err(|error| ErrorData::internal_error(error.to_string(), None))?,
        }))
    }

    #[tool(
        name = "team.gc",
        description = "Garbage-collect orphaned, expired, or hung teammate process groups across the whole registry (not scoped to one team). Reclaims spawns whose owner has exited, whose age exceeds ttl_secs, or whose last heartbeat is older than heartbeat_timeout_secs. Call it on a timer or after a run so abandoned workers never linger and clog the host."
    )]
    pub async fn team_gc(
        &self,
        Parameters(request): Parameters<TeamGcRequest>,
    ) -> Result<Json<TeamGcResponse>, ErrorData> {
        self.ensure_orchestration_enabled()?;
        let runtime = self.team_runtime.lock().await;
        let mut options = TeamGcOptions::default();
        if let Some(secs) = request.ttl_secs {
            options = options.with_ttl(Duration::from_secs(secs));
        }
        if let Some(secs) = request.heartbeat_timeout_secs {
            options = options.with_heartbeat_timeout(Duration::from_secs(secs));
        }
        let report = runtime
            .gc(options)
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        Ok(Json(TeamGcResponse {
            scanned: report.scanned,
            terminated: report.terminated,
            removed: report.removed,
            expired: report.expired,
            skipped_unverified: report.skipped_unverified,
        }))
    }

    #[tool(
        name = "team.presence",
        description = "Return a live presence snapshot for a Flume team: which lane/agent is active on what task right now, current load vs. global spawn cap, and per-lane budget usage. Use this before spawning or assigning work to avoid stepping on an already-active lane."
    )]
    pub async fn team_presence(
        &self,
        Parameters(request): Parameters<TeamPresenceRequest>,
    ) -> Result<Json<TeamPresenceResponse>, ErrorData> {
        self.ensure_orchestration_enabled()?;
        let runtime = self.team_runtime.lock().await;
        let snapshot = runtime
            .presence(&request.team_id)
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        Ok(Json(TeamPresenceResponse {
            presence: serde_json::to_value(snapshot)
                .map_err(|error| ErrorData::internal_error(error.to_string(), None))?,
        }))
    }

    #[tool(
        name = "team.lane.add",
        description = "Register a named specialist lane with a team's coordinator. A lane has a written contract (owned scope, non-goals, budget, handoff target) that makes parallelism legible and prevents scope overlap. Lanes run under the existing global concurrency cap — this does not increase it."
    )]
    pub async fn team_lane_add(
        &self,
        Parameters(request): Parameters<TeamLaneAddRequest>,
    ) -> Result<Json<TeamLaneResponse>, ErrorData> {
        self.ensure_orchestration_enabled()?;
        let mut runtime = self.team_runtime.lock().await;
        let lane = Lane::new(
            request.name,
            request.definition,
            LaneContract {
                owned_scope: request.owned_scope,
                non_goals: request.non_goals,
                budget: LaneBudget {
                    max_concurrent_tasks: request.max_concurrent_tasks,
                    max_turns: request.max_turns,
                    token_cap: None,
                },
                handoff_to: request.handoff_to,
                allowed_tools: request.allowed_tools,
                agent_constraint: None,
            },
        );
        let summary = runtime
            .add_lane(TeamLaneAdd {
                team_id: request.team_id,
                lane,
            })
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        Ok(Json(TeamLaneResponse {
            lane: serde_json::to_value(summary)
                .map_err(|error| ErrorData::internal_error(error.to_string(), None))?,
        }))
    }

    #[tool(
        name = "team.lane.assign",
        description = "Assign a headrace task to a named lane, enforcing deduplication (same task cannot be active in two lanes simultaneously) and the lane's concurrent-task budget. Returns the updated lane summary."
    )]
    pub async fn team_lane_assign(
        &self,
        Parameters(request): Parameters<TeamLaneAssignRequest>,
    ) -> Result<Json<TeamLaneResponse>, ErrorData> {
        self.ensure_orchestration_enabled()?;
        let mut runtime = self.team_runtime.lock().await;
        let summary = runtime
            .assign_task_to_lane(TeamLaneAssignTask {
                team_id: request.team_id,
                lane_name: request.lane_name,
                task_id: request.task_id,
                task_title: request.task_title,
            })
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        Ok(Json(TeamLaneResponse {
            lane: serde_json::to_value(summary)
                .map_err(|error| ErrorData::internal_error(error.to_string(), None))?,
        }))
    }

    #[tool(
        name = "memory.dream",
        description = "Bundle-to-bundle OCF memory consolidation: read all committed records, \
score by access signals (access_count / last_access / recency / richness), and write a new \
OCF bundle to `out` without mutating the source collection. Every admit/reject/merge/supersede/\
decay decision is logged in qualify.jsonl. Opt-in — run on a schedule or at compaction boundaries."
    )]
    pub async fn memory_dream(
        &self,
        Parameters(request): Parameters<DreamRequest>,
    ) -> Result<Json<DreamResponse>, ErrorData> {
        let out_path = std::path::PathBuf::from(&request.out);
        std::fs::create_dir_all(&out_path)
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;

        // Read all records (up to 10 000) from the source backend without mutating them.
        let query = MemoryQuery::new("").with_limit(10_000);
        let hits = self
            .backend
            .find(query)
            .await
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        let records_read = hits.len();
        let records: Vec<_> = hits.into_iter().map(|h| h.record).collect();

        let opts = aquifer::DreamOptions {
            admit_threshold: request.admit_threshold.unwrap_or(0.3),
            diary: request.diary,
            ..Default::default()
        };
        let source_label = request
            .collection
            .clone()
            .unwrap_or_else(|| "artesian-memory".to_string());

        let result = aquifer::dream(&records, &opts, None)
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;

        aquifer::write_dream_bundle(&result, &opts, &out_path, &source_label)
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;

        let mut files = vec![
            out_path.join("manifest.json").display().to_string(),
            out_path.join("schema.json").display().to_string(),
            out_path.join("snapshot.json").display().to_string(),
            out_path.join("qualify.jsonl").display().to_string(),
        ];
        if request.diary {
            files.push(out_path.join("DREAMS.md").display().to_string());
        }

        Ok(Json(DreamResponse {
            records_read,
            admitted: result.admitted,
            rejected: result.rejected,
            llm_ran: result.llm_ran,
            out: out_path.display().to_string(),
            files,
        }))
    }

    #[tool(
        name = "memory.learn",
        description = "Commit a curated GOVERNED SKILL memory record with provenance tracking. \
Unlike flat-file skill stores (Hermes /learn, deepagents Skills), Artesian skill records carry \
provenance (sources), usage signals (access_count), and participate in the normal decay/eviction \
lifecycle. DISCIPLINE: commit a curated skill — clear title, polished body, explicit sources. \
Re-learning the same title+content is idempotent (content-hash node_id dedup). \
Optional procedure steps enable guarded replay without per-step model calls."
    )]
    pub async fn memory_learn(
        &self,
        Parameters(request): Parameters<LearnRequest>,
    ) -> Result<Json<LearnResponse>, ErrorData> {
        let procedure = normalize_skill_procedure(
            request
                .procedure
                .unwrap_or_default()
                .into_iter()
                .map(Into::into)
                .collect(),
        )
        .map_err(|error| ErrorData::invalid_params(error.to_string(), None))?;

        // Canonical body: title as a heading followed by the skill content.
        let body = format!("# {}\n\n{}", request.title, request.content);

        // Stable node_id: same title + content → same hash → idempotent re-store.
        let identity = skill_identity_material(&request.title, &request.content, &procedure)
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        let hash = loop_core::stable_content_hash(&identity);
        let node_id = format!("skill:{hash}");

        // Provenance: join supplied sources; fall back to "artesian-learn".
        let sources = request.sources.unwrap_or_default();
        let source = if sources.is_empty() {
            Some("artesian-learn".to_string())
        } else {
            Some(sources.join(", "))
        };

        // Metadata: explicit title key for structured listing; sources list when multiple.
        let mut metadata = std::collections::BTreeMap::<String, String>::new();
        metadata.insert("title".to_string(), request.title.clone());
        if sources.len() > 1 {
            metadata.insert("sources".to_string(), sources.join(", "));
        }
        insert_skill_procedure_metadata(&mut metadata, &procedure)
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;

        // Tags: always include "skill"; append caller-supplied extras without duplicates.
        let mut tags = vec![loop_core::LOOP_SKILL_TAG.to_string()];
        for t in request.tags.unwrap_or_default() {
            if t != loop_core::LOOP_SKILL_TAG {
                tags.push(t);
            }
        }

        let record = self
            .backend
            .store(StoreMemory {
                content: body,
                tags,
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
            .await
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;

        Ok(Json(LearnResponse {
            id: record.id.to_string(),
            node_id: record.node_id,
        }))
    }

    #[tool(
        name = "memory.skills",
        description = "List all learned skill records (memories tagged `skill`) with title, \
usage count (access_count), provenance (source), and last retrieval timestamp. Includes both \
manually committed skills (`memory.learn`) and skills auto-committed by the loop on verified \
goal success. Pass by_usage=true to sort by most-used first."
    )]
    pub async fn memory_skills(
        &self,
        Parameters(request): Parameters<SkillsRequest>,
    ) -> Result<Json<SkillsResponse>, ErrorData> {
        let limit = request.limit.unwrap_or(20);
        let by_usage = request.by_usage.unwrap_or(false);

        // Tag-only query: empty text + non-empty tag filter returns all tag-matched records.
        let mut query = MemoryQuery::new("").with_limit(limit);
        query.tags = vec![loop_core::LOOP_SKILL_TAG.to_string()];
        let mut hits = self
            .backend
            .find(query)
            .await
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;

        if by_usage {
            hits.sort_by_key(|h| std::cmp::Reverse(h.record.access_count));
        }

        let skills = hits
            .into_iter()
            .map(|hit| {
                let rec = hit.record;
                let title = rec.metadata.get("title").cloned();
                let last_access = rec.last_access.map(|dt| dt.to_rfc3339());
                let procedure = skill_procedure_from_metadata(&rec.metadata)
                    .map_err(|error| ErrorData::internal_error(error.to_string(), None))?
                    .map(|procedure| procedure.into_iter().map(Into::into).collect());
                Ok(SkillHit {
                    id: rec.id.to_string(),
                    node_id: rec.node_id,
                    content: rec.content,
                    title,
                    access_count: rec.access_count,
                    source: rec.source,
                    last_access,
                    procedure,
                })
            })
            .collect::<Result<Vec<_>, ErrorData>>()?;

        Ok(Json(SkillsResponse { skills }))
    }

    #[tool(
        name = "memory.skill.replay",
        description = "Dry-run or explicitly execute a learned skill's guarded procedure. \
Default execute=false runs no commands. With execute=true, each guard must pass before its run \
command executes; guard mismatch aborts replay and tells the caller to proceed with normal reasoning."
    )]
    pub async fn memory_skill_replay(
        &self,
        Parameters(request): Parameters<SkillReplayRequest>,
    ) -> Result<Json<SkillReplayResponse>, ErrorData> {
        let response = replay_skill_procedure(
            self.backend.as_ref(),
            &request.title,
            request.execute.unwrap_or(false),
            &self.collection,
            self.track_savings,
        )
        .await
        .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        Ok(Json(response))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for MemoryServer {
    fn get_info(&self) -> ServerInfo {
        let instructions = if matches!(self.mode, Mode::Orchestrate | Mode::Full) {
            format!(
                "Artesian memory and orchestration server. {TOOL_INSTRUCTIONS} {MASTER_ROLE_SKILL}"
            )
        } else {
            format!(
                "Artesian memory server exposing memory.find and memory.store. {TOOL_INSTRUCTIONS}"
            )
        };
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(instructions)
    }
}

impl From<SessionAnchor> for AnchorPayload {
    fn from(anchor: SessionAnchor) -> Self {
        Self {
            current_task: anchor.current_task,
            plan_pointer: anchor.plan_pointer,
            last_decisions: anchor.last_decisions,
            next_step: anchor.next_step,
            updated_at: anchor.updated_at.to_rfc3339(),
        }
    }
}

impl From<SessionSummary> for SessionSummaryPayload {
    fn from(summary: SessionSummary) -> Self {
        Self {
            user_id: summary.key.user_id,
            session_id: summary.key.session_id,
            task_id: summary.key.task_id,
            updated_at: summary.updated_at.to_rfc3339(),
            handed_off_from: summary.handed_off_from,
            entry_count: summary.entry_count,
            token_count: summary.token_count,
        }
    }
}

// ── Public helpers for the CLI ────────────────────────────────────────────────────────────────
//
// These thin wrappers expose the core checkpoint/bundle logic with `anyhow::Result` returns so the
// CLI does not need to depend on `rmcp::ErrorData`. The MCP tool methods call the private
// functions directly (same code path, zero duplication).

/// Write (or update) a session anchor for `key` and return the resulting [`SessionAnchor`].
pub async fn checkpoint_anchor_for_cli(
    store: &AnchorAnchorStore,
    key: &SessionKey,
    request: &SessionCheckpointRequest,
) -> anyhow::Result<SessionAnchor> {
    let existing = store.get_for_session(key).await?;
    let current_task = request
        .current_task
        .clone()
        .or_else(|| existing.as_ref().map(|anchor| anchor.current_task.clone()))
        .or_else(|| request.goal.clone())
        .unwrap_or_else(|| format!("session {}", key.session_id));
    let next_step = request
        .next_step
        .clone()
        .or_else(|| existing.as_ref().map(|anchor| anchor.next_step.clone()))
        .unwrap_or_else(|| "continue from the checkpoint".to_string());
    let mut anchor = existing.unwrap_or_else(|| SessionAnchor::new(&current_task, &next_step));
    anchor.current_task = current_task;
    anchor.next_step = next_step;
    if let Some(plan_pointer) = &request.plan_pointer {
        anchor.plan_pointer = Some(plan_pointer.clone());
    }
    if let Some(last_decisions) = &request.last_decisions {
        anchor.last_decisions = last_decisions.clone();
    }
    Ok(store.set_for_session(key, anchor).await?)
}

/// Return session-scoped memory hits for `key` (excludes session-record tags).
pub async fn session_scoped_hits_for_cli(
    backend: &dyn MemoryBackend,
    key: &SessionKey,
    limit: usize,
) -> anyhow::Result<Vec<SearchHit>> {
    let query_text = format!("{} {}", key.task_id, key.session_id);
    let mut query = MemoryQuery::new(query_text).with_limit(limit);
    query.scope = Some(MemoryScope::Session);
    query.user_id = Some(key.user_id.clone());
    query.session_id = Some(key.session_id.clone());
    query.task_id = Some(key.task_id.clone());
    let hits = backend
        .find(query)
        .await?
        .into_iter()
        .filter(|hit| !hit.record.tags.iter().any(|tag| tag == SESSION_RECORD_TAG))
        .collect();
    Ok(hits)
}

/// Build an OCF session bundle from anchor, hits, and optional `last_failed_check`.
pub fn build_session_bundle_for_cli(
    anchor: Option<&SessionAnchor>,
    session_hits: &[SearchHit],
    invariant_hits: &[SearchHit],
    last_failed_check: Option<&str>,
) -> WorkingContextBundle {
    build_session_bundle(anchor, session_hits, invariant_hits, last_failed_check)
}

pub async fn run_stdio(root: impl Into<PathBuf>) -> anyhow::Result<()> {
    let server = MemoryServer::new(root);
    server.serve(stdio()).await?.waiting().await?;
    Ok(())
}

pub async fn run_stdio_with_config(config: &MemoryConfig) -> anyhow::Result<()> {
    let server = MemoryServer::from_config(config)?;
    server.serve(stdio()).await?.waiting().await?;
    Ok(())
}

pub async fn run_stdio_with_config_and_router(
    config: &MemoryConfig,
    router_enabled: bool,
) -> anyhow::Result<()> {
    let server = MemoryServer::from_config(config)?.with_router_enabled(router_enabled);
    server.serve(stdio()).await?.waiting().await?;
    Ok(())
}

pub async fn run_stdio_with_artesian_config(config: ArtesianConfig) -> anyhow::Result<()> {
    let router_enabled = config.coordination.router_enabled;
    let server = MemoryServer::from_artesian_config(&config)
        .await?
        .with_router_enabled(router_enabled);
    server.serve(stdio()).await?.waiting().await?;
    Ok(())
}

/// Serve the MCP memory tools over streamable HTTP at `bind` (path `/mcp`), so memory can be shared
/// across machines on a LAN instead of by a single stdio client. Each session gets a fresh memory
/// server built from `config`. Bind to a trusted interface only (no auth is enforced here).
#[cfg(feature = "http")]
pub async fn run_http(config: ArtesianConfig, bind: std::net::SocketAddr) -> anyhow::Result<()> {
    use rmcp::transport::streamable_http_server::{
        session::local::LocalSessionManager, StreamableHttpService,
    };
    let memory_config = config.memory.clone();
    let router_enabled = config.coordination.router_enabled;
    let service = StreamableHttpService::new(
        move || {
            MemoryServer::from_config(&memory_config)
                .map(|server| server.with_router_enabled(router_enabled))
                .map_err(std::io::Error::other)
        },
        std::sync::Arc::new(LocalSessionManager::default()),
        Default::default(),
    );
    let app = axum::Router::new().nest_service("/mcp", service);
    let listener = tokio::net::TcpListener::bind(bind).await?;
    eprintln!("artesian-mcp serving MCP over HTTP at http://{bind}/mcp");
    axum::serve(listener, app).await?;
    Ok(())
}

pub fn open_memory_backend(config: &MemoryConfig) -> anyhow::Result<Arc<dyn MemoryBackend>> {
    match config.backend {
        MemoryBackendKind::Files => Ok(Arc::new(
            FilesBackend::new(&config.root).with_track_access(config.track_access),
        )),
        MemoryBackendKind::SqliteVec => {
            let store = SqliteVecVectorStore::open(SqliteVecVectorStoreConfig::new(sqlite_path(
                &config.root,
            )))?;
            let backend = VectorMemoryBackend::new(store, vector_memory_config_from(config))?;
            let backend = attach_configured_reranker(backend, config);
            Ok(finish_vector_backend(backend, config))
        }
        MemoryBackendKind::Qdrant => open_qdrant_backend(config),
        MemoryBackendKind::TencentDb => anyhow::bail!("TencentDB backend is not available yet"),
    }
}

fn semantic_cache_from_config(config: &MemoryConfig) -> Option<aquifer::SemanticCache> {
    if !config.semantic_cache.enabled {
        return None;
    }
    let mut cache = aquifer::SemanticCache::new(
        config.semantic_cache.capacity,
        config.semantic_cache.min_similarity,
    );
    if let Some(ttl) = config.semantic_cache.ttl_seconds {
        cache = cache.with_ttl(Duration::from_secs(ttl));
    }
    Some(cache)
}

fn vector_memory_config_from(config: &MemoryConfig) -> VectorMemoryConfig {
    let mut vector_config =
        VectorMemoryConfig::new(&config.collection).with_track_access(config.track_access);
    if config.rerank {
        vector_config = vector_config
            .with_rerank(true)
            .with_rerank_candidates(config.effective_rerank_candidates());
    }
    vector_config
}

fn attach_configured_reranker<V>(
    backend: VectorMemoryBackend<V>,
    config: &MemoryConfig,
) -> VectorMemoryBackend<V>
where
    V: aquifer::VectorStore,
{
    attach_configured_reranker_with(backend, config, || {
        aquifer::FastembedReranker::new()
            .map(|reranker| Arc::new(reranker) as Arc<dyn aquifer::Reranker>)
    })
}

fn attach_configured_reranker_with<V, F, E>(
    backend: VectorMemoryBackend<V>,
    config: &MemoryConfig,
    load_reranker: F,
) -> VectorMemoryBackend<V>
where
    V: aquifer::VectorStore,
    F: FnOnce() -> std::result::Result<Arc<dyn aquifer::Reranker>, E>,
    E: std::fmt::Display,
{
    if !config.rerank {
        return backend;
    }

    match load_reranker() {
        Ok(reranker) => backend.with_reranker(reranker),
        Err(error) => {
            eprintln!(
                "warning: neural rerank requested but Fastembed reranker failed to load; \
                 falling back to hybrid RRF: {error}"
            );
            backend
        }
    }
}

fn finish_vector_backend<V: aquifer::VectorStore + Send + Sync + 'static>(
    backend: VectorMemoryBackend<V>,
    config: &MemoryConfig,
) -> Arc<dyn MemoryBackend> {
    match semantic_cache_from_config(config) {
        Some(cache) => Arc::new(backend.into_cached(cache)),
        None => Arc::new(backend),
    }
}

#[cfg(feature = "qdrant")]
fn open_qdrant_backend(config: &MemoryConfig) -> anyhow::Result<Arc<dyn MemoryBackend>> {
    let url = config
        .qdrant_url
        .clone()
        .or_else(|| env::var("QDRANT_URL").ok())
        .ok_or_else(|| anyhow::anyhow!("Qdrant backend requires qdrant_url or QDRANT_URL"))?;
    let mut vector_config = QdrantVectorStoreConfig::new(url);
    vector_config.rest_url = config
        .qdrant_rest_url
        .clone()
        .or_else(|| env::var("QDRANT_REST_URL").ok());
    vector_config.api_key = config.resolve_qdrant_api_key();
    let store = QdrantVectorStore::connect(vector_config)?;
    let backend = VectorMemoryBackend::new(store, vector_memory_config_from(config))?;
    let backend = attach_configured_reranker(backend, config);
    Ok(finish_vector_backend(backend, config))
}

#[cfg(not(feature = "qdrant"))]
fn open_qdrant_backend(_config: &MemoryConfig) -> anyhow::Result<Arc<dyn MemoryBackend>> {
    anyhow::bail!("Qdrant backend requires building artesian-mcp with the qdrant feature")
}

fn sqlite_path(root: &str) -> PathBuf {
    let path = PathBuf::from(root);
    if path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| matches!(extension, "db" | "sqlite" | "sqlite3"))
    {
        path
    } else {
        path.join("memory.sqlite3")
    }
}

#[derive(Debug, Clone, Copy)]
struct RegisteredTool {
    name: &'static str,
    description: &'static str,
}

fn tool_registry() -> &'static [RegisteredTool] {
    &[
        RegisteredTool {
            name: "memory.find",
            description: "Find durable project memories by query before non-trivial work.",
        },
        RegisteredTool {
            name: "memory.store",
            description: "Store durable reusable learnings in project memory.",
        },
        RegisteredTool {
            name: "memory.answer",
            description: "Answer one question from committed memory with node_id citations.",
        },
        RegisteredTool {
            name: "memory.context",
            description: "Read index.md first, then return a targeted memory.find slice.",
        },
        RegisteredTool {
            name: "memory.anchor.get",
            description: "Read Anchor session anchor from OKF log.md before resuming work.",
        },
        RegisteredTool {
            name: "memory.anchor.set",
            description:
                "Write current task, plan pointer, decisions, and next step to OKF log.md.",
        },
        RegisteredTool {
            name: "memory.session.checkpoint",
            description:
                "Checkpoint a resumable OCF session before yielding to another agent.",
        },
        RegisteredTool {
            name: "memory.session.resume",
            description:
                "Resume a committed OCF session by user_id/session_id/task_id without matching agent_id.",
        },
        RegisteredTool {
            name: "agents.list",
            description: "List reachable configured agent CLIs and available models.",
        },
        RegisteredTool {
            name: "orchestrate.delegate",
            description: "Delegate a bounded subtask to a configured supervised role agent.",
        },
        RegisteredTool {
            name: "orchestrate.bind",
            description: "Bind an orchestration role to a reachable agent model for this session.",
        },
        RegisteredTool {
            name: "orchestrate.status",
            description: "Check the status and result of a delegated orchestration task.",
        },
        RegisteredTool {
            name: "orchestrate.handoff",
            description: "Hand results to judge or master for orchestration follow-up.",
        },
        RegisteredTool {
            name: "orchestrate.loop",
            description: "Run the Artesian agentic loop: recall, act (worker), verify (goal), commit skill/spec/invariant.",
        },
        RegisteredTool {
            name: "team.create",
            description: "Create a Flume agent team topology in orchestrate or full mode.",
        },
        RegisteredTool {
            name: "team.spawn",
            description: "Spawn a defined teammate; after adding work, block with team.task.await so the request stays active.",
        },
        RegisteredTool {
            name: "team.task.add",
            description: "Add a team task, then immediately block on team.task.await instead of ending the turn or polling.",
        },
        RegisteredTool {
            name: "team.task.await",
            description: "Blocking, heartbeated wait for one or more team tasks to finish; keeps the client request active and interruptible.",
        },
        RegisteredTool {
            name: "team.task.claim",
            description: "Atomically claim an eligible team task with plan gate checks.",
        },
        RegisteredTool {
            name: "team.task.complete",
            description: "Complete or block a team task through judge or master review.",
        },
        RegisteredTool {
            name: "team.message",
            description: "Post ASK, RESULT, REVIEW, or DONE messages to the EventEnvelope pool.",
        },
        RegisteredTool {
            name: "team.status",
            description: "Inspect a non-blocking team snapshot; use team.task.await to block on teammate completion.",
        },
        RegisteredTool {
            name: "team.presence",
            description: "Live presence snapshot: which lane/agent is active on what task, load vs. cap.",
        },
        RegisteredTool {
            name: "team.lane.add",
            description: "Register a specialist lane with a contract (scope, non-goals, budget, handoff).",
        },
        RegisteredTool {
            name: "team.lane.assign",
            description: "Assign a task to a lane, enforcing dedup and budget across all lanes.",
        },
        RegisteredTool {
            name: "team.cleanup",
            description: "Clean up tracked teammate process groups and mark a team cleaned up.",
        },
        RegisteredTool {
            name: "team.gc",
            description: "Garbage-collect orphaned, expired, or hung teammate process groups across the registry.",
        },
    ]
}

fn normalize_await_task_ids(request: &TeamTaskAwaitRequest) -> Result<Vec<String>, ErrorData> {
    let mut task_ids = Vec::new();
    if let Some(task_id) = request
        .task_id
        .as_deref()
        .map(str::trim)
        .filter(|task_id| !task_id.is_empty())
    {
        task_ids.push(task_id.to_string());
    }
    for task_id in request
        .task_ids
        .iter()
        .map(|task_id| task_id.trim())
        .filter(|task_id| !task_id.is_empty())
    {
        if task_ids.iter().all(|existing| existing != task_id) {
            task_ids.push(task_id.to_string());
        }
    }
    if task_ids.is_empty() {
        return Err(ErrorData::invalid_params(
            "team.task.await requires task_id or task_ids".to_string(),
            None,
        ));
    }
    Ok(task_ids)
}

fn await_timeout(timeout_secs: Option<u64>) -> Duration {
    timeout_secs
        .map(Duration::from_secs)
        .unwrap_or(TEAM_TASK_AWAIT_DEFAULT_TIMEOUT)
}

fn await_poll_interval(poll_interval_ms: Option<u64>) -> Duration {
    poll_interval_ms
        .map(Duration::from_millis)
        .unwrap_or(TEAM_TASK_AWAIT_DEFAULT_POLL)
        .clamp(TEAM_TASK_AWAIT_MIN_POLL, TEAM_TASK_AWAIT_MAX_POLL)
}

async fn load_await_tasks(
    task_store: &FilesTaskStore,
    task_ids: &[String],
) -> Result<Vec<Task>, ErrorData> {
    let mut tasks = Vec::with_capacity(task_ids.len());
    for task_id in task_ids {
        let task = task_store
            .get(task_id)
            .await
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?
            .ok_or_else(|| {
                ErrorData::invalid_params(format!("team task not found: {task_id}"), None)
            })?;
        tasks.push(task);
    }
    Ok(tasks)
}

fn await_response(
    team_id: &str,
    task_ids: &[String],
    outcome: &str,
    started: std::time::Instant,
    tasks: Vec<Task>,
) -> Result<Json<TeamTaskAwaitResponse>, ErrorData> {
    let elapsed_ms = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
    let tasks = tasks
        .into_iter()
        .map(serde_json::to_value)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
    Ok(Json(TeamTaskAwaitResponse {
        team_id: team_id.to_string(),
        task_ids: task_ids.to_vec(),
        outcome: outcome.to_string(),
        elapsed_ms,
        tasks,
    }))
}

fn lexical_score(task: &str, description: &str) -> f32 {
    let task_terms = terms(task);
    if task_terms.is_empty() {
        return 0.0;
    }
    let description = description.to_ascii_lowercase();
    let matches = task_terms
        .iter()
        .filter(|term| description.contains(term.as_str()))
        .count();
    matches as f32 / task_terms.len() as f32
}

fn terms(input: &str) -> Vec<String> {
    input
        .split_whitespace()
        .map(|term| {
            term.trim_matches(|character: char| !character.is_ascii_alphanumeric())
                .to_ascii_lowercase()
        })
        .filter(|term| term.len() > 2)
        .collect()
}

fn estimate_tokens(text: &str) -> usize {
    text.split_whitespace().count().max(1)
}

fn now_id() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn first_line_or_default(input: &str) -> String {
    input
        .lines()
        .next()
        .filter(|line| !line.trim().is_empty())
        .unwrap_or("Delegated task")
        .to_string()
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };

    use aquifer::{MemoryResult, SearchHit};
    use artesian_core::DEFAULT_RERANK_CANDIDATES;

    use super::*;

    struct TestEmbedder;

    impl aquifer::TextEmbedder for TestEmbedder {
        fn embed_query(&self, _text: &str) -> MemoryResult<Vec<f32>> {
            Ok(vec![0.0; aquifer::PINNED_FASTEMBED_DIMENSIONS])
        }

        fn embed_passage(&self, _text: &str) -> MemoryResult<Vec<f32>> {
            Ok(vec![0.0; aquifer::PINNED_FASTEMBED_DIMENSIONS])
        }
    }

    struct TestReranker;

    impl aquifer::Reranker for TestReranker {
        fn rerank(
            &self,
            _query: &str,
            mut hits: Vec<SearchHit>,
            limit: usize,
        ) -> MemoryResult<Vec<SearchHit>> {
            hits.truncate(limit);
            Ok(hits)
        }
    }

    fn memory_config(rerank: bool, rerank_candidates: usize) -> MemoryConfig {
        MemoryConfig {
            backend: MemoryBackendKind::SqliteVec,
            root: ".artesian".to_string(),
            collection: "artesian-memory".to_string(),
            qdrant_url: None,
            qdrant_rest_url: None,
            qdrant_api_key_env: None,
            qdrant_api_key_file: None,
            local_rerank_enabled: true,
            hyde_enabled: false,
            multi_query_enabled: false,
            debate_enabled: false,
            llm_consolidation_enabled: false,
            rerank,
            rerank_candidates,
            semantic_cache: Default::default(),
            track_access: true,
            track_savings: true,
        }
    }

    fn vector_backend(config: &MemoryConfig) -> VectorMemoryBackend<aquifer::SqliteVecVectorStore> {
        VectorMemoryBackend::with_embedder(
            aquifer::SqliteVecVectorStore::in_memory().expect("sqlite store"),
            vector_memory_config_from(config),
            Arc::new(TestEmbedder),
        )
        .expect("vector backend")
    }

    #[test]
    fn rerank_false_does_not_load_or_attach_reranker() {
        let config = memory_config(false, 0);
        let loader_called = AtomicBool::new(false);

        let backend = attach_configured_reranker_with(vector_backend(&config), &config, || {
            loader_called.store(true, Ordering::SeqCst);
            Ok::<Arc<dyn aquifer::Reranker>, &str>(
                Arc::new(TestReranker) as Arc<dyn aquifer::Reranker>
            )
        });

        assert!(!loader_called.load(Ordering::SeqCst));
        assert!(!backend.has_reranker());
        assert!(!backend.config().rerank);
        assert_eq!(backend.config().rerank_candidates, 0);
        assert!(!backend.rerank_active_for_limit(10));
    }

    #[test]
    fn rerank_true_uses_default_pool_and_attaches_mock_reranker() {
        let config = memory_config(true, 0);

        let backend = attach_configured_reranker_with(vector_backend(&config), &config, || {
            Ok::<Arc<dyn aquifer::Reranker>, &str>(
                Arc::new(TestReranker) as Arc<dyn aquifer::Reranker>
            )
        });

        assert!(backend.has_reranker());
        assert!(backend.config().rerank);
        assert_eq!(
            backend.config().rerank_candidates,
            DEFAULT_RERANK_CANDIDATES
        );
        assert!(backend.config().rerank_candidates > 10);
        assert!(backend.rerank_active_for_limit(10));
    }

    #[test]
    fn rerank_load_failure_falls_back_without_attachment() {
        let config = memory_config(true, 0);

        let backend = attach_configured_reranker_with(vector_backend(&config), &config, || {
            Err::<Arc<dyn aquifer::Reranker>, &str>("offline")
        });

        assert!(!backend.has_reranker());
        assert!(backend.config().rerank);
        assert_eq!(
            backend.config().rerank_candidates,
            DEFAULT_RERANK_CANDIDATES
        );
        assert!(!backend.rerank_active_for_limit(10));
    }
}
