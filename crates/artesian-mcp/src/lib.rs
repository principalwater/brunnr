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
    AnchorAnchorStore, FilesBackend, MemoryBackend, MemoryQuery, MemoryScope, MemoryTier,
    SearchHit, SessionAnchor, SqliteVecVectorStore, SqliteVecVectorStoreConfig, StoreMemory,
    VectorMemoryBackend, VectorMemoryConfig,
};
use artesian_core::{
    AccConfig, Agent, AgentBinding, AgentCatalog, AgentMessage, ArtesianConfig, MemoryBackendKind,
    MemoryConfig, Mode, Role, SpawnRequest,
};
use artesian_process_agent::{
    fallback_agent_catalog, load_or_refresh_agent_catalog, validate_binding_model, ProcessAgent,
    ProcessAgentConfig,
};
use flume::{
    load_role_definitions, role_summaries, TeamCreate, TeamGcOptions, TeamMessage, TeamMessageKind,
    TeamRuntime, TeamRuntimeConfig, TeamSpawn, TeamTaskAdd, TeamTaskClaim, TeamTaskComplete,
};
use headgate::{Headgate, HeadgateConfig, MemoryRecallStore, RecallStore};
use headrace::{ClaimRequest, FilesTaskStore, NewTask, TaskStatus, TaskStore, TransitionTask};
use rmcp::{
    handler::server::{
        router::tool::ToolRouter,
        wrapper::{Json, Parameters},
    },
    model::{ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router,
    transport::stdio,
    ErrorData, ServerHandler, ServiceExt,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex as AsyncMutex;

#[cfg(feature = "qdrant")]
use aquifer::{QdrantVectorStore, QdrantVectorStoreConfig};

const TOOL_INSTRUCTIONS: &str =
    "ALWAYS search the project memory before non-trivial work; store durable, reusable learnings.";
const MASTER_ROLE_SKILL: &str = "In orchestrate/full mode, first call agents.list to inspect reachable agents, models, and role definitions. Use memory.context for compact project recall, create Flume teams with team.create/team.spawn when several teammates are useful, delegate bounded subtasks through team.task.* or orchestrate.delegate(worker), and gate accepted outcomes through the judge/master path before marking work done.";
const ORCHESTRATION_TOOLS: &[&str] = &[
    "agents.list",
    "orchestrate.bind",
    "orchestrate.delegate",
    "orchestrate.status",
    "orchestrate.handoff",
    "team.create",
    "team.spawn",
    "team.task.add",
    "team.task.claim",
    "team.task.complete",
    "team.message",
    "team.status",
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
        Ok(Self::with_backend_and_anchor(
            open_memory_backend(config)?,
            Some(AnchorAnchorStore::new(&config.root)),
        )
        .with_okf_root(Some(PathBuf::from(&config.root))))
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
pub struct StoreRequest {
    pub content: String,
    pub tags: Option<Vec<String>>,
    pub node_id: Option<String>,
    pub source: Option<String>,
    pub confidence: Option<f32>,
    pub scope: Option<ScopeRequest>,
    pub agent_id: Option<String>,
    pub session_id: Option<String>,
    pub task_id: Option<String>,
    pub user_id: Option<String>,
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
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct TeamMessageResponse {
    pub event: serde_json::Value,
    pub response: Option<String>,
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct DelegateRecord {
    status: String,
    result: Option<String>,
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
        query.scope = request.scope.map(Into::into);
        query.agent_id = request.agent_id;
        query.session_id = request.session_id;
        query.task_id = request.task_id;
        query.user_id = request.user_id;
        let hits = self
            .backend
            .find(query)
            .await
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?
            .into_iter()
            .map(find_hit)
            .collect();
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
        name = "memory.context",
        description = "Return a compact index.md slice plus targeted memory.find hits; no LLM call is made."
    )]
    pub async fn memory_context(
        &self,
        Parameters(request): Parameters<ContextRequest>,
    ) -> Result<Json<ContextResponse>, ErrorData> {
        let index = self
            .okf_root
            .as_ref()
            .and_then(|root| std::fs::read_to_string(root.join("memory").join("index.md")).ok())
            .map(|index| {
                let limit = request.index_chars.unwrap_or(4_000);
                index.chars().take(limit).collect::<String>()
            });
        let mut query = MemoryQuery::new(request.query);
        query.limit = request.limit.unwrap_or(10);
        query.scope = request.scope.map(Into::into);
        query.agent_id = request.agent_id;
        query.session_id = request.session_id;
        query.task_id = request.task_id;
        query.user_id = request.user_id;
        let hits = self
            .backend
            .find(query)
            .await
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?
            .into_iter()
            .map(find_hit)
            .collect();
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
        Parameters(request): Parameters<DelegateRequest>,
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
            })
            .await
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        let response = process
            .send(
                &session,
                AgentMessage {
                    content: format!(
                        "Task ID: {task_id}\nRole: {}\n\n{}",
                        role.canonical_alias(),
                        request.task
                    ),
                },
            )
            .await;
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
            })
            .await;
        Ok(Json(HandoffResponse {
            accepted: true,
            to: to.canonical_alias().to_string(),
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
        description = "Admit and spawn a teammate from a .agent/agents or .claude/agents definition through the supervised ProcessAgent path."
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
        description = "Add a task to the shared headrace task board for a Flume team."
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
        name = "team.message",
        description = "Post a typed Flume message (ASK/RESULT/REVIEW/DONE); direct addressing rides the shared EventEnvelope pool."
    )]
    pub async fn team_message(
        &self,
        Parameters(request): Parameters<TeamMessageRequest>,
    ) -> Result<Json<TeamMessageResponse>, ErrorData> {
        self.ensure_orchestration_enabled()?;
        let mut runtime = self.team_runtime.lock().await;
        let outcome = runtime
            .message(TeamMessage {
                team_id: request.team_id,
                from: request.from,
                to: request.to,
                kind: request.kind.into(),
                content: request.content,
                task_id: request.task_id,
                approved: request.approved,
                execute: request.execute.unwrap_or(false),
            })
            .await
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        Ok(Json(TeamMessageResponse {
            event: serde_json::to_value(outcome.event)
                .map_err(|error| ErrorData::internal_error(error.to_string(), None))?,
            response: outcome.response,
        }))
    }

    #[tool(
        name = "team.status",
        description = "Return Flume team lifecycle, teammates, and redacted EventEnvelope pool."
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
        MemoryBackendKind::Files => Ok(Arc::new(FilesBackend::new(&config.root))),
        MemoryBackendKind::SqliteVec => {
            let store = SqliteVecVectorStore::open(SqliteVecVectorStoreConfig::new(sqlite_path(
                &config.root,
            )))?;
            let backend =
                VectorMemoryBackend::new(store, VectorMemoryConfig::new(&config.collection))?;
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
    if let Some(env_name) = &config.qdrant_api_key_env {
        vector_config.api_key = env::var(env_name).ok();
    }
    let store = QdrantVectorStore::connect(vector_config)?;
    let backend = VectorMemoryBackend::new(store, VectorMemoryConfig::new(&config.collection))?;
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
            name: "team.create",
            description: "Create a Flume agent team topology in orchestrate or full mode.",
        },
        RegisteredTool {
            name: "team.spawn",
            description: "Spawn a defined teammate through the supervised ProcessAgent path.",
        },
        RegisteredTool {
            name: "team.task.add",
            description: "Add a task to the shared headrace task board for a team.",
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
            description: "Inspect team lifecycle state, teammates, and redacted message pool.",
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
