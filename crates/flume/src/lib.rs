// SPDX-License-Identifier: Apache-2.0

//! Artesian-native agent teams (Flume).

pub mod lane;
pub mod loop_core;
pub mod quota;

pub use lane::{
    Lane, LaneBudget, LaneContract, LaneCoordinator, LaneError, LaneSummary, PresenceSnapshot,
    TeammatePresence,
};

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    str::FromStr,
    time::Duration,
};

use artesian_core::{
    Agent, AgentBinding, AgentCatalog, AgentError, AgentMessage, AgentRoleDefinition, AgentSession,
    EventEnvelope, EventSender, EventType, Role, SpawnRequest,
};
use artesian_process_agent::{
    validate_binding_model, GcOptions, ProcessAgent, ProcessAgentConfig, ProcessSupervisor,
    ReapReport, WorkerEvent,
};
pub use artesian_process_agent::{GcOptions as TeamGcOptions, ReapReport as TeamReapReport};

/// The three canonical Conductor-style knobs for delegating work to a worker.
///
/// Every spawn or delegate operation in Flume is ultimately described by a `Delegation`.
/// Role-based spawns produce a default delegation from the definition + binding; callers that need
/// fine-grained control (e.g. the MCP `orchestrate.loop` path or an outer orchestrator) can supply
/// the knobs explicitly.
///
/// # Knobs
///
/// 1. **Agent selection** — which agent CLI and model to use, resolved from the pool or supplied
///    directly as an [`AgentBinding`].
/// 2. **Targeted instruction** — the per-worker system text (from the role definition's
///    `prompt_addendum`) combined with the caller's task content, plus an optional *resume packet*
///    that carries prior-session state into a fresh process.
/// 3. **Context visibility** — which tools the worker is permitted to call, and an optional memory
///    session slice (user / session / task identifiers) that gates the worker's shared memory view.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Delegation {
    /// Knob 1 — agent selection: the resolved agent/model binding.
    pub binding: AgentBinding,
    /// Knob 2 — targeted instruction: the combined system + task prompt the worker receives.
    pub instruction: String,
    /// Knob 2 (cont.) — optional resume packet injected as process startup state.
    pub resume_packet: Option<String>,
    /// Knob 3 — context visibility: tools the worker may call (empty = unrestricted by this layer).
    pub allowed_tools: Vec<String>,
    /// Knob 3 (cont.) — optional memory session slice for shared-memory scoping.
    pub session_context: Option<DelegationSessionContext>,
}

/// Memory session scope for a delegation (user / session / task tenancy keys).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DelegationSessionContext {
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub task_id: Option<String>,
}

impl Delegation {
    /// Build a delegation from a resolved binding and a full composed prompt, with no session
    /// scoping and tools inherited from the definition.
    pub fn new(binding: AgentBinding, instruction: impl Into<String>) -> Self {
        Self {
            binding,
            instruction: instruction.into(),
            resume_packet: None,
            allowed_tools: Vec::new(),
            session_context: None,
        }
    }

    /// Override the resume packet (Knob 2).
    pub fn with_resume_packet(mut self, packet: impl Into<String>) -> Self {
        self.resume_packet = Some(packet.into());
        self
    }

    /// Override the allowed tools (Knob 3 — tool scoping).
    pub fn with_allowed_tools(mut self, tools: Vec<String>) -> Self {
        self.allowed_tools = tools;
        self
    }

    /// Override the session context (Knob 3 — memory scoping).
    pub fn with_session_context(mut self, context: DelegationSessionContext) -> Self {
        self.session_context = Some(context);
        self
    }

    /// Build a default delegation from a role definition (binding + prompt addendum) and caller
    /// task content. Tool visibility comes from the definition; no session scoping is applied.
    /// This is the path taken by `TeamRuntime::spawn_teammate` / role-based spawns.
    pub fn from_definition(
        definition: &RoleDefinition,
        binding: AgentBinding,
        task_content: &str,
    ) -> Self {
        let instruction = if definition.prompt_addendum.is_empty() {
            task_content.to_string()
        } else {
            format!("{}\n\n{}", definition.prompt_addendum, task_content)
        };
        Self {
            binding,
            instruction,
            resume_packet: None,
            allowed_tools: definition.allow_tools.clone(),
            session_context: None,
        }
    }
}
use chrono::Utc;
use headrace::{
    ClaimRequest, FilesTaskStore, NewTask, Task, TaskStatus, TaskStore, TransitionTask,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::mpsc;

#[derive(Debug, thiserror::Error)]
pub enum FlumeError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to decode agent definition: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("agent definition is invalid: {0}")]
    InvalidDefinition(String),
    #[error("team not found: {0}")]
    TeamNotFound(String),
    #[error("teammate not found: {0}")]
    TeammateNotFound(String),
    #[error("task not found: {0}")]
    TaskNotFound(String),
    #[error("plan approval required before execution for task {0}")]
    PlanApprovalRequired(String),
    #[error("team admission paused teammate {name}: {reason}")]
    AdmissionPaused { name: String, reason: String },
    #[error("agent failed: {0}")]
    Agent(String),
    #[error("task store failed: {0}")]
    Task(#[from] headrace::TaskError),
    #[error("lane duplicate task: {0}")]
    LaneDuplicate(String),
    #[error("lane budget exceeded: {0}")]
    LaneBudget(String),
}

impl From<AgentError> for FlumeError {
    fn from(value: AgentError) -> Self {
        Self::Agent(value.to_string())
    }
}

pub type FlumeResult<T> = Result<T, FlumeError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RoleDefinitionSource {
    Artesian,
    ClaudeInterop,
}

impl RoleDefinitionSource {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Artesian => "artesian",
            Self::ClaudeInterop => "claude-interop",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoleDefinition {
    pub name: String,
    pub kind: Role,
    pub description: String,
    pub agent: Option<String>,
    pub model: Option<String>,
    pub allow_tools: Vec<String>,
    pub prompt_addendum: String,
    pub source: RoleDefinitionSource,
    pub path: PathBuf,
}

impl RoleDefinition {
    pub fn summary(&self) -> AgentRoleDefinition {
        AgentRoleDefinition {
            name: self.name.clone(),
            kind: self.kind,
            description: self.description.clone(),
            agent: self.agent.clone(),
            model: self.model.clone(),
            allow_tools: self.allow_tools.clone(),
            source: self.source.as_str().to_string(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct DefinitionFrontmatter {
    name: Option<String>,
    kind: Option<String>,
    description: Option<String>,
    agent: Option<String>,
    model: Option<String>,
    allow_tools: Option<ToolList>,
    tools: Option<ToolList>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ToolList {
    List(Vec<String>),
    Csv(String),
}

impl ToolList {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::List(items) => items
                .into_iter()
                .map(|item| item.trim().to_string())
                .filter(|item| !item.is_empty())
                .collect(),
            Self::Csv(items) => items
                .split(',')
                .map(str::trim)
                .filter(|item| !item.is_empty())
                .map(str::to_string)
                .collect(),
        }
    }
}

pub fn load_role_definitions(repo_root: impl AsRef<Path>) -> FlumeResult<Vec<RoleDefinition>> {
    let repo_root = repo_root.as_ref();
    let mut definitions = Vec::new();
    definitions.extend(load_definition_dir(
        repo_root.join(".agent").join("agents"),
        RoleDefinitionSource::Artesian,
    )?);
    definitions.extend(load_definition_dir(
        repo_root.join(".claude").join("agents"),
        RoleDefinitionSource::ClaudeInterop,
    )?);
    definitions.sort_by(|left, right| left.name.cmp(&right.name).then(left.path.cmp(&right.path)));
    Ok(definitions)
}

pub fn role_summaries(definitions: &[RoleDefinition]) -> Vec<AgentRoleDefinition> {
    definitions
        .iter()
        .map(RoleDefinition::summary)
        .collect::<Vec<_>>()
}

fn load_definition_dir(
    directory: PathBuf,
    source: RoleDefinitionSource,
) -> FlumeResult<Vec<RoleDefinition>> {
    let mut definitions = Vec::new();
    let read_dir = match fs::read_dir(&directory) {
        Ok(read_dir) => read_dir,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(definitions),
        Err(error) => return Err(error.into()),
    };
    let mut paths = Vec::new();
    for entry in read_dir {
        let path = entry?.path();
        if path.extension().and_then(|extension| extension.to_str()) == Some("md") {
            paths.push(path);
        }
    }
    paths.sort();
    for path in paths {
        let text = fs::read_to_string(&path)?;
        definitions.push(parse_role_definition(&path, &text, source)?);
    }
    Ok(definitions)
}

pub fn parse_role_definition(
    path: impl AsRef<Path>,
    text: &str,
    source: RoleDefinitionSource,
) -> FlumeResult<RoleDefinition> {
    let path = path.as_ref();
    let (header, body) = split_frontmatter(text)?;
    let header: DefinitionFrontmatter = serde_yaml::from_str(header)?;
    let name = required_header(header.name, "name", path)?;
    let description = required_header(header.description, "description", path)?;
    let kind = match source {
        RoleDefinitionSource::Artesian => {
            let kind = required_header(header.kind, "kind", path)?;
            parse_kind(&kind)?
        }
        RoleDefinitionSource::ClaudeInterop => header
            .kind
            .as_deref()
            .map(parse_kind)
            .transpose()?
            .unwrap_or_else(|| infer_kind(&name)),
    };
    let allow_tools = header
        .allow_tools
        .or(header.tools)
        .map(ToolList::into_vec)
        .unwrap_or_default();
    Ok(RoleDefinition {
        name,
        kind,
        description,
        agent: header.agent.and_then(empty_to_none),
        model: header.model.and_then(empty_to_none),
        allow_tools,
        prompt_addendum: body.trim().to_string(),
        source,
        path: path.to_path_buf(),
    })
}

fn split_frontmatter(text: &str) -> FlumeResult<(&str, &str)> {
    let rest = text
        .strip_prefix("---\n")
        .ok_or_else(|| FlumeError::InvalidDefinition("missing YAML frontmatter".to_string()))?;
    rest.split_once("\n---\n")
        .ok_or_else(|| FlumeError::InvalidDefinition("unterminated YAML frontmatter".to_string()))
}

fn required_header(value: Option<String>, name: &str, path: &Path) -> FlumeResult<String> {
    let Some(value) = value.and_then(empty_to_none) else {
        return Err(FlumeError::InvalidDefinition(format!(
            "{} missing required `{name}`",
            path.display()
        )));
    };
    Ok(value)
}

fn empty_to_none(value: String) -> Option<String> {
    let value = value.trim().to_string();
    (!value.is_empty()).then_some(value)
}

fn parse_kind(input: &str) -> FlumeResult<Role> {
    Role::from_str(input).map_err(|error| FlumeError::InvalidDefinition(error.to_string()))
}

fn infer_kind(name: &str) -> Role {
    let name = name.to_ascii_lowercase();
    if name.contains("judge") || name.contains("review") {
        Role::Judge
    } else if name.contains("master") || name.contains("lead") || name.contains("coordinator") {
        Role::Master
    } else {
        Role::Worker
    }
}

#[derive(Debug, Clone)]
pub struct TeamRuntimeConfig {
    pub repo_root: PathBuf,
    pub task_root: PathBuf,
    pub registry_dir: PathBuf,
    pub bindings: Vec<AgentBinding>,
    pub catalog: AgentCatalog,
    pub definitions: Vec<RoleDefinition>,
    pub max_teammates: usize,
    pub max_concurrent_spawns: usize,
    pub max_lifetime: Duration,
    pub termination_grace: Duration,
}

impl TeamRuntimeConfig {
    pub fn new(repo_root: impl Into<PathBuf>, task_root: impl Into<PathBuf>) -> Self {
        let repo_root = repo_root.into();
        Self {
            registry_dir: repo_root.join(".artesian").join("spawns"),
            repo_root,
            task_root: task_root.into(),
            bindings: Vec::new(),
            catalog: AgentCatalog::default(),
            definitions: Vec::new(),
            max_teammates: 4,
            max_concurrent_spawns: 4,
            max_lifetime: Duration::from_secs(30 * 60),
            termination_grace: Duration::from_secs(2),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TeamRecord {
    pub id: String,
    pub name: String,
    pub status: TeamStatus,
    pub max_teammates: usize,
    pub plan_approval_required: bool,
    pub plan_approval_roles: Vec<String>,
    pub teammates: Vec<TeammateRecord>,
    pub events: Vec<EventEnvelope>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TeamStatus {
    Idle,
    Active,
    Complete,
    CleanedUp,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TeammateRecord {
    pub name: String,
    pub kind: Role,
    pub agent: String,
    pub model: Option<String>,
    pub status: TeammateStatus,
    pub paused_reason: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TeammateStatus {
    Idle,
    Active,
    Paused,
    Complete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TeamCreate {
    pub id: Option<String>,
    pub name: String,
    pub max_teammates: Option<usize>,
    pub plan_approval_required: bool,
    pub plan_approval_roles: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TeamSpawn {
    pub team_id: String,
    pub definition: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TeamTaskAdd {
    pub team_id: String,
    pub title: String,
    pub description: String,
    pub definition: Option<String>,
    pub blockers: Vec<String>,
}

/// Request to register a [`Lane`] with a team.  The coordinator starts tracking the lane
/// immediately; tasks are assigned to the lane via [`TeamRuntime::assign_task_to_lane`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TeamLaneAdd {
    pub team_id: String,
    pub lane: Lane,
}

/// Request to assign a task to a lane, enforcing dedup and budget.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TeamLaneAssignTask {
    pub team_id: String,
    pub lane_name: String,
    pub task_id: String,
    pub task_title: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TeamTaskClaim {
    pub team_id: String,
    pub task_id: Option<String>,
    pub teammate: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TeamTaskComplete {
    pub team_id: String,
    pub task_id: String,
    pub reviewer: String,
    pub approved: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TeamMessage {
    pub team_id: String,
    pub from: String,
    pub to: Option<String>,
    pub kind: TeamMessageKind,
    pub content: String,
    pub task_id: Option<String>,
    pub approved: Option<bool>,
    pub execute: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resume_packet: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TeamMessageKind {
    Ask,
    Result,
    Review,
    Done,
}

impl TeamMessageKind {
    const fn event_type(self) -> EventType {
        match self {
            Self::Ask => EventType::Ask,
            Self::Result => EventType::Result,
            Self::Review => EventType::Review,
            Self::Done => EventType::Done,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TeamMessageOutcome {
    pub event: EventEnvelope,
    pub response: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub worker_events: Vec<TeamWorkerEvent>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TeamWorkerEvent {
    pub team_id: String,
    pub teammate: String,
    pub kind: String,
    pub text: String,
    pub raw: String,
}

#[derive(Debug, Clone)]
pub struct TeamRuntime {
    config: TeamRuntimeConfig,
    teams: BTreeMap<String, TeamState>,
    event_counter: u64,
}

impl TeamRuntime {
    pub fn new(config: TeamRuntimeConfig) -> Self {
        Self {
            config,
            teams: BTreeMap::new(),
            event_counter: 0,
        }
    }

    pub fn definitions(&self) -> &[RoleDefinition] {
        &self.config.definitions
    }

    pub fn create_team(&mut self, request: TeamCreate) -> TeamRecord {
        let id = request.id.unwrap_or_else(|| stable_team_id(&request.name));
        let max_teammates = request
            .max_teammates
            .unwrap_or(self.config.max_teammates)
            .max(1);
        let team = TeamState {
            id: id.clone(),
            name: request.name,
            status: TeamStatus::Idle,
            max_teammates,
            plan_approval_required: request.plan_approval_required,
            plan_approval_roles: request.plan_approval_roles,
            teammates: BTreeMap::new(),
            events: Vec::new(),
            task_definitions: BTreeMap::new(),
            approved_plans: BTreeSet::new(),
            lane_coordinator: LaneCoordinator::new(),
        };
        let record = team.record();
        self.teams.insert(id, team);
        record
    }

    pub async fn spawn_teammate(&mut self, request: TeamSpawn) -> FlumeResult<TeammateRecord> {
        // Opportunistically reclaim orphaned spawns from a prior crashed owner so
        // the registry never accumulates abandoned workers. Best-effort: the
        // current owner's live spawns are never touched, so a failure here must
        // not block a legitimate spawn.
        let _ = self.gc(GcOptions::default());
        let definition = self.definition(&request.definition)?.clone();
        let binding = self.binding_for_definition(&definition)?;
        let team = self.team(&request.team_id)?;
        let cap_reached = team.active_teammates() >= team.max_teammates;
        if cap_reached {
            let teammate = TeammateState {
                definition: definition.clone(),
                binding,
                session: None,
                status: TeammateStatus::Paused,
                paused_reason: Some("team teammate cap reached".to_string()),
            };
            let record = teammate.record();
            self.team_mut(&request.team_id)?
                .teammates
                .insert(definition.name.clone(), teammate);
            return Ok(record);
        }
        let process = self.process_agent(&binding);
        let working_dir = self.config.repo_root.display().to_string();
        let session = process
            .spawn(SpawnRequest {
                role: definition.kind,
                agent: binding.agent.clone(),
                model: binding.model.clone(),
                working_dir: Some(working_dir),
                resume_packet: None,
            })
            .await?;
        let teammate = TeammateState {
            definition: definition.clone(),
            binding,
            session: Some(session),
            status: TeammateStatus::Idle,
            paused_reason: None,
        };
        let record = teammate.record();
        let team = self.team_mut(&request.team_id)?;
        team.teammates.insert(definition.name.clone(), teammate);
        team.status = TeamStatus::Active;
        Ok(record)
    }

    pub async fn add_task(&mut self, request: TeamTaskAdd) -> FlumeResult<Task> {
        let definition_name = match request.definition {
            Some(name) => Some(name),
            None => self.default_worker_definition_name(),
        };
        let role = definition_name
            .as_deref()
            .and_then(|name| self.definition(name).ok())
            .map_or(Role::Worker, |definition| definition.kind);
        let task_store = FilesTaskStore::new(&self.config.task_root);
        let mut task = NewTask::primitive(request.title);
        task.description = request.description;
        task.role = role;
        task.blockers = request.blockers;
        let task = task_store.create(task).await?;
        let team = self.team_mut(&request.team_id)?;
        if let Some(definition_name) = definition_name {
            team.task_definitions
                .insert(task.id.clone(), definition_name.to_string());
        }
        Ok(task)
    }

    pub async fn claim_task(&mut self, request: TeamTaskClaim) -> FlumeResult<Option<Task>> {
        let requires_approval = self.requires_plan_approval(&request.team_id, &request)?;
        if let (true, Some(task_id)) = (requires_approval, request.task_id.as_ref()) {
            return Err(FlumeError::PlanApprovalRequired(task_id.clone()));
        }
        let task_store = FilesTaskStore::new(&self.config.task_root);
        let claimed = task_store
            .claim(ClaimRequest {
                task_id: request.task_id.clone(),
                claimant: request.teammate.clone(),
            })
            .await?;
        if let Some(task) = &claimed {
            self.push_team_event(
                &request.team_id,
                task.id.clone(),
                self.teammate_role(&request.team_id, &request.teammate)
                    .unwrap_or(Role::Worker),
                &request.teammate,
                EventType::TaskClaimed,
                json!({
                    "task_id": task.id,
                    "teammate": request.teammate
                }),
            )?;
        }
        Ok(claimed)
    }

    pub async fn complete_task(&mut self, request: TeamTaskComplete) -> FlumeResult<Task> {
        let reviewer_role = self
            .teammate_role(&request.team_id, &request.reviewer)
            .unwrap_or(Role::Master);
        if !matches!(reviewer_role, Role::Judge | Role::Master) {
            return Err(FlumeError::InvalidDefinition(
                "only judge or master teammates may complete a task".to_string(),
            ));
        }
        let status = if request.approved {
            TaskStatus::Done
        } else {
            TaskStatus::Blocked
        };
        let task_store = FilesTaskStore::new(&self.config.task_root);
        let task = task_store
            .transition(TransitionTask {
                id: request.task_id.clone(),
                status,
            })
            .await?;
        self.push_team_event(
            &request.team_id,
            request.task_id.clone(),
            reviewer_role,
            &request.reviewer,
            EventType::Review,
            json!({
                "task_id": request.task_id,
                "approved": request.approved
            }),
        )?;
        if request.approved {
            self.push_team_event(
                &request.team_id,
                task.id.clone(),
                reviewer_role,
                &request.reviewer,
                EventType::Done,
                json!({ "task_id": task.id }),
            )?;
        }
        Ok(task)
    }

    pub async fn message(&mut self, request: TeamMessage) -> FlumeResult<TeamMessageOutcome> {
        self.message_with_worker_events(request, None).await
    }

    pub async fn message_with_worker_events(
        &mut self,
        request: TeamMessage,
        event_sender: Option<mpsc::UnboundedSender<TeamWorkerEvent>>,
    ) -> FlumeResult<TeamMessageOutcome> {
        let correlation_id = request
            .task_id
            .clone()
            .unwrap_or_else(|| request.team_id.clone());
        let from_role = self
            .teammate_role(&request.team_id, &request.from)
            .unwrap_or(Role::Worker);
        let event = self.push_team_event(
            &request.team_id,
            correlation_id.clone(),
            from_role,
            &request.from,
            request.kind.event_type(),
            json!({
                "from": request.from,
                "to": request.to,
                "task_id": request.task_id,
                "content": redact_secrets(&request.content),
                "approved": request.approved
            }),
        )?;
        if request.kind == TeamMessageKind::Review && request.approved.unwrap_or(false) {
            if let Some(task_id) = request.task_id.as_ref() {
                self.team_mut(&request.team_id)?
                    .approved_plans
                    .insert(task_id.clone());
            }
        }
        let (response, worker_events) = if request.execute {
            let Some(to) = request.to.as_ref() else {
                return Err(FlumeError::TeammateNotFound(
                    "execute requires a target teammate".to_string(),
                ));
            };
            let execution = self
                .execute_teammate(
                    &request.team_id,
                    to,
                    &request.content,
                    request.resume_packet.as_deref(),
                    event_sender,
                )
                .await?;
            (Some(execution.response), execution.events)
        } else {
            (None, Vec::new())
        };
        Ok(TeamMessageOutcome {
            event,
            response,
            worker_events,
        })
    }

    pub fn status(&self, team_id: &str) -> FlumeResult<TeamRecord> {
        self.team(team_id).map(TeamState::record)
    }

    pub fn cleanup(&mut self, team_id: &str) -> FlumeResult<TeamRecord> {
        let supervisor = ProcessSupervisor::new(&self.config.registry_dir)
            .with_termination_grace(self.config.termination_grace)
            .with_max_concurrent_spawns(self.config.max_concurrent_spawns);
        supervisor
            .terminate_current_owner()
            .map_err(|error| FlumeError::Agent(error.to_string()))?;
        let team = self.team_mut(team_id)?;
        for teammate in team.teammates.values_mut() {
            teammate.status = TeammateStatus::Complete;
            teammate.session = None;
        }
        team.status = TeamStatus::CleanedUp;
        Ok(team.record())
    }

    /// Registry-wide garbage collection across every tracked teammate, not just
    /// one team: reclaim orphaned process groups (dead owner), spawns past the
    /// TTL, and heartbeat-stale (hung) spawns. Safe to call on a timer so a
    /// crashed or abandoned worker never lingers and clogs the host.
    pub fn gc(&self, options: GcOptions) -> FlumeResult<ReapReport> {
        let supervisor = ProcessSupervisor::new(&self.config.registry_dir)
            .with_termination_grace(self.config.termination_grace)
            .with_max_concurrent_spawns(self.config.max_concurrent_spawns);
        supervisor
            .gc(options)
            .map_err(|error| FlumeError::Agent(error.to_string()))
    }

    // ── Lane / Coordinator ─────────────────────────────────────────────────────────────────────

    /// Register a [`Lane`] with a team's coordinator.  The lane's contract is stored immediately;
    /// tasks are assigned to lanes separately via [`assign_task_to_lane`].
    pub fn add_lane(&mut self, request: TeamLaneAdd) -> FlumeResult<LaneSummary> {
        let team = self.team_mut(&request.team_id)?;
        let summary = LaneSummary {
            name: request.lane.name.clone(),
            definition: request.lane.definition.clone(),
            contract: request.lane.contract.clone(),
            active_task_ids: Vec::new(),
            turns_used: 0,
            tokens_used: 0,
        };
        team.lane_coordinator.register_lane(request.lane);
        Ok(summary)
    }

    /// Attempt to assign a task to a named lane, enforcing dedup across all lanes in the team.
    ///
    /// Returns [`FlumeError::LaneDuplicate`] if another lane is already working on the same
    /// task id or canonical title.  Returns [`FlumeError::LaneBudget`] if the lane's
    /// concurrent-task cap is exhausted.
    pub fn assign_task_to_lane(&mut self, request: TeamLaneAssignTask) -> FlumeResult<LaneSummary> {
        let team = self.team_mut(&request.team_id)?;
        team.lane_coordinator
            .assign_task(&request.lane_name, &request.task_id, &request.task_title)
            .map_err(|error| match &error {
                LaneError::DuplicateTask { .. } => FlumeError::LaneDuplicate(error.to_string()),
                LaneError::BudgetExceeded(_) => FlumeError::LaneBudget(error.to_string()),
                LaneError::LaneNotFound(_) => FlumeError::InvalidDefinition(error.to_string()),
            })?;
        team.lane_coordinator
            .lane_summaries()
            .into_iter()
            .find(|s| s.name == request.lane_name)
            .ok_or_else(|| {
                FlumeError::InvalidDefinition(format!(
                    "lane '{}' disappeared after assignment",
                    request.lane_name
                ))
            })
    }

    /// Return the coordinator's lane summaries for a team (without the full teammate list).
    pub fn lane_summaries(&self, team_id: &str) -> FlumeResult<Vec<LaneSummary>> {
        Ok(self.team(team_id)?.lane_coordinator.lane_summaries())
    }

    // ── Presence ───────────────────────────────────────────────────────────────────────────────

    /// Return a live [`PresenceSnapshot`] for a team: which lane/agent is active on what task,
    /// current load vs. global cap.  Reuses the existing heartbeat / registry infrastructure —
    /// the presence view is derived from in-memory state with no additional I/O.
    pub fn presence(&self, team_id: &str) -> FlumeResult<PresenceSnapshot> {
        let team = self.team(team_id)?;
        let supervisor = ProcessSupervisor::new(&self.config.registry_dir)
            .with_termination_grace(self.config.termination_grace)
            .with_max_concurrent_spawns(self.config.max_concurrent_spawns);
        let spawns_active = supervisor
            .entries()
            .map(|entries| {
                entries
                    .into_iter()
                    .filter(|e| supervisor.group_alive(e.pgid))
                    .count()
            })
            .unwrap_or(0);

        let lane_summaries = team.lane_coordinator.lane_summaries();

        // Build a per-lane index of active task ids for teammate matching.
        let lane_tasks: std::collections::HashMap<String, Vec<String>> = lane_summaries
            .iter()
            .map(|s| (s.name.clone(), s.active_task_ids.clone()))
            .collect();

        let teammates = team
            .teammates
            .iter()
            .map(|(name, state)| {
                // Find which lane this teammate is associated with (by definition name).
                let lane = lane_summaries
                    .iter()
                    .find(|s| s.definition == state.definition.name)
                    .map(|s| s.name.clone());
                let active_task_ids = lane
                    .as_ref()
                    .and_then(|l| lane_tasks.get(l))
                    .cloned()
                    .unwrap_or_default();
                TeammatePresence {
                    name: name.clone(),
                    lane,
                    status: format!("{:?}", state.status).to_ascii_lowercase(),
                    active_task_ids,
                }
            })
            .collect::<Vec<_>>();

        let total_active_tasks = lane_summaries.iter().map(|s| s.active_task_ids.len()).sum();

        Ok(PresenceSnapshot {
            team_id: team_id.to_string(),
            lanes: lane_summaries,
            teammates,
            total_active_tasks,
            spawns_active,
            spawns_cap: self.config.max_concurrent_spawns,
        })
    }

    async fn execute_teammate(
        &self,
        team_id: &str,
        teammate_name: &str,
        content: &str,
        resume_packet: Option<&str>,
        event_sender: Option<mpsc::UnboundedSender<TeamWorkerEvent>>,
    ) -> FlumeResult<TeamExecution> {
        let team = self.team(team_id)?;
        let teammate = team
            .teammates
            .get(teammate_name)
            .ok_or_else(|| FlumeError::TeammateNotFound(teammate_name.to_string()))?;
        if teammate.status == TeammateStatus::Paused {
            return Err(FlumeError::AdmissionPaused {
                name: teammate_name.to_string(),
                reason: teammate
                    .paused_reason
                    .clone()
                    .unwrap_or_else(|| "paused".to_string()),
            });
        }
        let definition = teammate.definition.clone();
        let binding = teammate.binding.clone();
        // Build a Delegation from the definition (all three knobs), allowing callers that supply a
        // resume_packet override to set Knob 2 without duplicating prompt assembly logic.
        let mut delegation = Delegation::from_definition(&definition, binding, content);
        if let Some(packet) = resume_packet {
            delegation = delegation.with_resume_packet(packet);
        }
        self.execute_delegation(delegation, team_id, teammate_name, event_sender)
            .await
    }

    /// Execute a fully-specified [`Delegation`] (the canonical spawn + send path for all three
    /// knobs). Callers such as `execute_teammate` build delegations from role definitions; external
    /// callers (e.g. `orchestrate.loop`) can supply delegations directly.
    ///
    /// `team_id` and `teammate_name` are used purely for bookkeeping in [`TeamWorkerEvent`] payloads;
    /// pass empty strings when the execution is not part of a team (e.g. the loop path).
    pub async fn execute_delegation(
        &self,
        delegation: Delegation,
        team_id: &str,
        teammate_name: &str,
        event_sender: Option<mpsc::UnboundedSender<TeamWorkerEvent>>,
    ) -> FlumeResult<TeamExecution> {
        let binding = &delegation.binding;
        let process = self.process_agent(binding);
        let session = process
            .spawn(SpawnRequest {
                role: binding.role,
                agent: binding.agent.clone(),
                model: binding.model.clone(),
                working_dir: Some(self.config.repo_root.display().to_string()),
                resume_packet: delegation.resume_packet.clone(),
            })
            .await?;
        let (worker_sender, mut worker_receiver) = mpsc::unbounded_channel();
        let response = process.send_with_event_sender(
            &session,
            AgentMessage {
                content: delegation.instruction.clone(),
            },
            Some(worker_sender),
        );
        tokio::pin!(response);
        let mut events = Vec::new();
        let response = loop {
            tokio::select! {
                maybe_event = worker_receiver.recv() => {
                    match maybe_event {
                        Some(event) => push_worker_event(
                            &mut events,
                            event_sender.as_ref(),
                            team_id,
                            teammate_name,
                            event,
                        ),
                        None => break response.await,
                    }
                }
                response = &mut response => break response,
            }
        };
        while let Some(event) = worker_receiver.recv().await {
            push_worker_event(
                &mut events,
                event_sender.as_ref(),
                team_id,
                teammate_name,
                event,
            );
        }
        let response = response?;
        Ok(TeamExecution {
            response: redact_secrets(&response.content),
            events,
        })
    }

    fn requires_plan_approval(&self, team_id: &str, request: &TeamTaskClaim) -> FlumeResult<bool> {
        let team = self.team(team_id)?;
        let Some(task_id) = request.task_id.as_ref() else {
            return Ok(false);
        };
        if team.approved_plans.contains(task_id) {
            return Ok(false);
        }
        let role_requires = team
            .task_definitions
            .get(task_id)
            .is_some_and(|definition| team.plan_approval_roles.contains(definition));
        Ok(team.plan_approval_required || role_requires)
    }

    fn binding_for_definition(&self, definition: &RoleDefinition) -> FlumeResult<AgentBinding> {
        let base = definition
            .agent
            .as_ref()
            .and_then(|agent| {
                self.config
                    .bindings
                    .iter()
                    .find(|binding| binding.agent == *agent)
            })
            .or_else(|| {
                self.config
                    .bindings
                    .iter()
                    .find(|binding| binding.role == definition.kind)
            });
        let Some(agent) = definition
            .agent
            .clone()
            .or_else(|| base.map(|binding| binding.agent.clone()))
        else {
            return Err(FlumeError::InvalidDefinition(format!(
                "definition '{}' has no agent and no {} binding is configured",
                definition.name,
                definition.kind.canonical_alias()
            )));
        };
        if !self
            .config
            .catalog
            .agents
            .iter()
            .any(|entry| entry.agent == agent && entry.reachable)
        {
            return Err(FlumeError::Agent(format!(
                "agent '{agent}' is not reachable in the catalog; run `artesian agents refresh`"
            )));
        }
        let binding = AgentBinding {
            role: definition.kind,
            agent,
            model: definition
                .model
                .clone()
                .or_else(|| base.and_then(|binding| binding.model.clone())),
            command: base.and_then(|binding| binding.command.clone()),
            args: base.map(|binding| binding.args.clone()).unwrap_or_default(),
            timeout_seconds: base.and_then(|binding| binding.timeout_seconds),
        };
        validate_binding_model(&binding, &self.config.catalog)?;
        Ok(binding)
    }

    fn process_agent(&self, binding: &AgentBinding) -> ProcessAgent {
        let command = binding
            .command
            .clone()
            .unwrap_or_else(|| binding.agent.clone());
        let static_models = self
            .config
            .catalog
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
            .unwrap_or_default();
        ProcessAgent::new(
            ProcessAgentConfig::new(command)
                .with_agent_id(binding.agent.clone())
                .with_default_model(binding.model.clone())
                .with_args(binding.args.clone())
                .with_static_models(static_models)
                .with_working_dir(&self.config.repo_root)
                .with_timeout(Duration::from_secs(binding.timeout_seconds.unwrap_or(120)))
                .with_registry_dir(self.config.registry_dir.clone())
                .with_max_concurrent_spawns(self.config.max_concurrent_spawns)
                .with_max_lifetime(self.config.max_lifetime)
                .with_termination_grace(self.config.termination_grace),
        )
    }

    fn definition(&self, name: &str) -> FlumeResult<&RoleDefinition> {
        self.config
            .definitions
            .iter()
            .find(|definition| definition.name == name)
            .ok_or_else(|| {
                FlumeError::InvalidDefinition(format!("unknown role definition: {name}"))
            })
    }

    fn team(&self, team_id: &str) -> FlumeResult<&TeamState> {
        self.teams
            .get(team_id)
            .ok_or_else(|| FlumeError::TeamNotFound(team_id.to_string()))
    }

    fn team_mut(&mut self, team_id: &str) -> FlumeResult<&mut TeamState> {
        self.teams
            .get_mut(team_id)
            .ok_or_else(|| FlumeError::TeamNotFound(team_id.to_string()))
    }

    fn teammate_role(&self, team_id: &str, teammate_name: &str) -> Option<Role> {
        self.teams
            .get(team_id)
            .and_then(|team| team.teammates.get(teammate_name))
            .map(|teammate| teammate.definition.kind)
    }

    fn default_worker_definition_name(&self) -> Option<String> {
        self.config
            .definitions
            .iter()
            .find(|definition| definition.kind == Role::Worker)
            .map(|definition| definition.name.clone())
    }

    fn push_team_event(
        &mut self,
        team_id: &str,
        correlation_id: String,
        role: Role,
        agent_id: &str,
        event_type: EventType,
        payload: serde_json::Value,
    ) -> FlumeResult<EventEnvelope> {
        self.event_counter += 1;
        let event = EventEnvelope::new(
            format!("team-evt-{}", self.event_counter),
            correlation_id,
            EventSender {
                role,
                agent_id: agent_id.to_string(),
            },
            event_type,
            payload,
        );
        self.team_mut(team_id)?.events.push(event.clone());
        Ok(event)
    }
}

#[derive(Debug, Clone)]
struct TeamState {
    id: String,
    name: String,
    status: TeamStatus,
    max_teammates: usize,
    plan_approval_required: bool,
    plan_approval_roles: Vec<String>,
    teammates: BTreeMap<String, TeammateState>,
    events: Vec<EventEnvelope>,
    task_definitions: BTreeMap<String, String>,
    approved_plans: BTreeSet<String>,
    /// Lane coordinator: tracks parallel specialist lanes and enforces dedup.
    lane_coordinator: LaneCoordinator,
}

impl TeamState {
    fn record(&self) -> TeamRecord {
        TeamRecord {
            id: self.id.clone(),
            name: self.name.clone(),
            status: self.status,
            max_teammates: self.max_teammates,
            plan_approval_required: self.plan_approval_required,
            plan_approval_roles: self.plan_approval_roles.clone(),
            teammates: self.teammates.values().map(TeammateState::record).collect(),
            events: self.events.clone(),
        }
    }

    fn active_teammates(&self) -> usize {
        self.teammates
            .values()
            .filter(|teammate| teammate.status != TeammateStatus::Paused)
            .count()
    }
}

#[derive(Debug, Clone)]
struct TeammateState {
    definition: RoleDefinition,
    binding: AgentBinding,
    session: Option<AgentSession>,
    status: TeammateStatus,
    paused_reason: Option<String>,
}

impl TeammateState {
    fn record(&self) -> TeammateRecord {
        TeammateRecord {
            name: self.definition.name.clone(),
            kind: self.definition.kind,
            agent: self.binding.agent.clone(),
            model: self.binding.model.clone(),
            status: self.status,
            paused_reason: self.paused_reason.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TeamExecution {
    pub response: String,
    pub events: Vec<TeamWorkerEvent>,
}

fn push_worker_event(
    events: &mut Vec<TeamWorkerEvent>,
    event_sender: Option<&mpsc::UnboundedSender<TeamWorkerEvent>>,
    team_id: &str,
    teammate: &str,
    event: WorkerEvent,
) {
    let event = TeamWorkerEvent {
        team_id: team_id.to_string(),
        teammate: teammate.to_string(),
        kind: event.kind,
        text: redact_secrets(&event.text),
        raw: redact_secrets(&event.raw),
    };
    if let Some(sender) = event_sender {
        let _ = sender.send(event.clone());
    }
    events.push(event);
}

fn stable_team_id(name: &str) -> String {
    let mut id = name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>();
    id = id.trim_matches('-').to_string();
    if id.is_empty() {
        format!("team-{}", Utc::now().timestamp_millis())
    } else {
        id
    }
}

pub fn redact_secrets(input: &str) -> String {
    let mut output = input.to_string();
    for prefix in [
        "sk-",
        "ghp_",
        "gho_",
        "github_pat_",
        "xoxb-",
        "xoxp-",
        "hf_",
    ] {
        output = redact_prefixed_token(&output, prefix);
    }
    for key in [
        "api_key",
        "api-key",
        "apikey",
        "access_token",
        "access-token",
        "authorization",
        "bearer",
        "password",
        "secret",
        "token",
    ] {
        output = redact_key_value_token(&output, key);
    }
    output
}

fn redact_prefixed_token(input: &str, prefix: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut cursor = 0;
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

fn redact_key_value_token(input: &str, key: &str) -> String {
    let lower = input.to_ascii_lowercase();
    let mut output = String::with_capacity(input.len());
    let mut cursor = 0;
    while let Some(relative_start) = lower[cursor..].find(key) {
        let start = cursor + relative_start;
        output.push_str(&input[cursor..start]);
        let mut value_start = start + key.len();
        while input[value_start..]
            .chars()
            .next()
            .is_some_and(|character| matches!(character, ' ' | '\t' | ':' | '='))
        {
            value_start += input[value_start..]
                .chars()
                .next()
                .map_or(0, char::len_utf8);
        }
        if value_start == start + key.len() {
            output.push_str(&input[start..value_start]);
            cursor = value_start;
            continue;
        }
        output.push_str(&input[start..value_start]);
        output.push_str("[REDACTED]");
        let mut end = value_start;
        for (offset, character) in input[value_start..].char_indices() {
            if character.is_whitespace() || matches!(character, ',' | ';' | '"' | '\'') {
                break;
            }
            end = value_start + offset + character.len_utf8();
        }
        cursor = end;
    }
    output.push_str(&input[cursor..]);
    output
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use artesian_core::{AgentCatalogEntry, AgentModel};
    use artesian_test_support::TempDir;

    use super::*;

    #[test]
    fn parses_artesian_and_claude_role_definitions() {
        let tempdir = TempDir::new("delta-definitions");
        let artesian_dir = tempdir.join(".agent").join("agents");
        let claude_dir = tempdir.join(".claude").join("agents");
        fs::create_dir_all(&artesian_dir).expect("artesian dir should exist");
        fs::create_dir_all(&claude_dir).expect("claude dir should exist");
        fs::write(
            artesian_dir.join("security.md"),
            "---\nname: security-reviewer\nkind: worker\ndescription: Reviews security-sensitive code.\nagent: codex\nmodel: gpt-5\nallow_tools: [read, grep, memory.find]\n---\nSecurity prompt.\n",
        )
        .expect("definition should write");
        fs::write(
            claude_dir.join("lead.md"),
            "---\nname: lead-planner\ndescription: Plans decomposition.\ntools: Read, Grep\nmodel: claude-sonnet\n---\nLead prompt.\n",
        )
        .expect("definition should write");

        let definitions = load_role_definitions(tempdir.path()).expect("definitions should load");

        assert_eq!(definitions.len(), 2);
        let security = definitions
            .iter()
            .find(|definition| definition.name == "security-reviewer")
            .expect("security definition should load");
        assert_eq!(security.kind, Role::Worker);
        assert_eq!(security.allow_tools, vec!["read", "grep", "memory.find"]);
        let lead = definitions
            .iter()
            .find(|definition| definition.name == "lead-planner")
            .expect("claude interop definition should load");
        assert_eq!(lead.kind, Role::Master);
        assert_eq!(lead.source, RoleDefinitionSource::ClaudeInterop);
    }

    #[test]
    fn user_named_role_maps_to_kind() {
        let definition = parse_role_definition(
            "architect.md",
            "---\nname: architect\nkind: worker\ndescription: Designs module boundaries.\n---\nPrompt.\n",
            RoleDefinitionSource::Artesian,
        )
        .expect("definition should parse");

        assert_eq!(definition.name, "architect");
        assert_eq!(definition.kind, Role::Worker);
    }

    #[tokio::test]
    async fn unavailable_agent_or_model_fails_before_spawn() {
        let tempdir = TempDir::new("delta-unavailable");
        let mut runtime = TeamRuntime::new(runtime_config(
            tempdir.path(),
            vec![definition(
                "security-reviewer",
                Role::Worker,
                Some("missing"),
                Some("ghost"),
            )],
            vec![binding(
                Role::Worker,
                "missing",
                Some("ghost"),
                "echo",
                vec![],
            )],
            catalog(vec![entry("missing", true, vec![model("other", true)])]),
        ));
        runtime.create_team(TeamCreate {
            id: Some("team".to_string()),
            name: "Team".to_string(),
            max_teammates: None,
            plan_approval_required: false,
            plan_approval_roles: Vec::new(),
        });

        let error = runtime
            .spawn_teammate(TeamSpawn {
                team_id: "team".to_string(),
                definition: "security-reviewer".to_string(),
            })
            .await
            .expect_err("unavailable model should fail");

        assert!(error.to_string().contains("ghost"));
        assert!(
            ProcessSupervisor::new(tempdir.join("spawns"))
                .entries()
                .expect("registry should read")
                .is_empty(),
            "early validation must not spawn a process"
        );
    }

    #[tokio::test]
    async fn plan_approval_gate_blocks_until_review_approves() {
        let tempdir = TempDir::new("delta-plan-gate");
        let mut runtime = TeamRuntime::new(runtime_config(
            tempdir.path(),
            vec![
                definition("planner", Role::Master, Some("echo"), Some("ok")),
                definition("worker-a", Role::Worker, Some("echo"), Some("ok")),
                definition("judge-a", Role::Judge, Some("echo"), Some("ok")),
            ],
            vec![binding(
                Role::Worker,
                "echo",
                Some("ok"),
                "echo",
                vec!["ok".into()],
            )],
            catalog(vec![entry("echo", true, vec![model("ok", true)])]),
        ));
        runtime.create_team(TeamCreate {
            id: Some("team".to_string()),
            name: "Team".to_string(),
            max_teammates: None,
            plan_approval_required: true,
            plan_approval_roles: Vec::new(),
        });
        runtime
            .spawn_teammate(TeamSpawn {
                team_id: "team".to_string(),
                definition: "worker-a".to_string(),
            })
            .await
            .expect("worker should spawn");
        runtime
            .spawn_teammate(TeamSpawn {
                team_id: "team".to_string(),
                definition: "judge-a".to_string(),
            })
            .await
            .expect("judge should spawn");
        let task = runtime
            .add_task(TeamTaskAdd {
                team_id: "team".to_string(),
                title: "Implement feature".to_string(),
                description: "Do bounded work".to_string(),
                definition: Some("worker-a".to_string()),
                blockers: Vec::new(),
            })
            .await
            .expect("task should be added");

        let blocked = runtime
            .claim_task(TeamTaskClaim {
                team_id: "team".to_string(),
                task_id: Some(task.id.clone()),
                teammate: "worker-a".to_string(),
            })
            .await
            .expect_err("plan approval should block claim");
        assert!(matches!(blocked, FlumeError::PlanApprovalRequired(_)));

        runtime
            .message(TeamMessage {
                team_id: "team".to_string(),
                from: "judge-a".to_string(),
                to: Some("worker-a".to_string()),
                kind: TeamMessageKind::Review,
                content: "Plan approved".to_string(),
                task_id: Some(task.id.clone()),
                approved: Some(true),
                execute: false,
                resume_packet: None,
            })
            .await
            .expect("review should approve plan");

        let claimed = runtime
            .claim_task(TeamTaskClaim {
                team_id: "team".to_string(),
                task_id: Some(task.id),
                teammate: "worker-a".to_string(),
            })
            .await
            .expect("approved task should claim")
            .expect("task should be claimed");
        assert_eq!(claimed.status, TaskStatus::Doing);
    }

    #[tokio::test]
    async fn team_message_redacts_successful_secret_output() {
        let tempdir = TempDir::new("delta-secret-output");
        let secret = "sk-team-secret-123456";
        let mut runtime = TeamRuntime::new(runtime_config(
            tempdir.path(),
            vec![definition("worker-a", Role::Worker, Some("sh"), None)],
            vec![binding(
                Role::Worker,
                "sh",
                None,
                "sh",
                vec!["-c".to_string(), format!("printf 'api_key={secret}\\n'")],
            )],
            catalog(vec![entry("sh", true, Vec::new())]),
        ));
        runtime.create_team(TeamCreate {
            id: Some("team".to_string()),
            name: "Team".to_string(),
            max_teammates: None,
            plan_approval_required: false,
            plan_approval_roles: Vec::new(),
        });
        runtime
            .spawn_teammate(TeamSpawn {
                team_id: "team".to_string(),
                definition: "worker-a".to_string(),
            })
            .await
            .expect("worker should spawn");

        let outcome = runtime
            .message(TeamMessage {
                team_id: "team".to_string(),
                from: "worker-a".to_string(),
                to: Some("worker-a".to_string()),
                kind: TeamMessageKind::Ask,
                content: format!("use token={secret}"),
                task_id: None,
                approved: None,
                execute: true,
                resume_packet: None,
            })
            .await
            .expect("message should execute");
        let status = runtime.status("team").expect("status should read");
        let json = serde_json::to_string(&status).expect("status should encode");

        assert!(!json.contains(secret));
        assert!(!outcome.response.unwrap_or_default().contains(secret));
        assert!(json.contains("[REDACTED]"));
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn team_run_cleans_up_process_group_children() {
        let tempdir = TempDir::new("delta-cleanup");
        let parent_pid = tempdir.join("parent.pid");
        let child_pid = tempdir.join("child.pid");
        let script = "echo $$ > \"$1\"; sleep 30 & echo $! > \"$2\"; wait";
        let mut runtime = TeamRuntime::new(runtime_config(
            tempdir.path(),
            vec![
                definition("worker-a", Role::Worker, Some("sh"), None),
                definition("judge-a", Role::Judge, Some("echo"), None),
            ],
            vec![
                binding(
                    Role::Worker,
                    "sh",
                    None,
                    "sh",
                    vec![
                        "-c".to_string(),
                        script.to_string(),
                        "artesian-test-sh".to_string(),
                        parent_pid.display().to_string(),
                        child_pid.display().to_string(),
                    ],
                ),
                binding(Role::Judge, "echo", None, "echo", vec!["review".into()]),
            ],
            catalog(vec![
                entry("sh", true, Vec::new()),
                entry("echo", true, Vec::new()),
            ]),
        ));
        runtime.config.max_lifetime = Duration::from_millis(200);
        runtime.config.termination_grace = Duration::from_millis(50);
        runtime.create_team(TeamCreate {
            id: Some("team".to_string()),
            name: "Team".to_string(),
            max_teammates: None,
            plan_approval_required: false,
            plan_approval_roles: Vec::new(),
        });
        runtime
            .spawn_teammate(TeamSpawn {
                team_id: "team".to_string(),
                definition: "worker-a".to_string(),
            })
            .await
            .expect("worker should spawn");
        let task = runtime
            .add_task(TeamTaskAdd {
                team_id: "team".to_string(),
                title: "Slow task".to_string(),
                description: "Exercise process cleanup".to_string(),
                definition: Some("worker-a".to_string()),
                blockers: Vec::new(),
            })
            .await
            .expect("task should add");
        runtime
            .claim_task(TeamTaskClaim {
                team_id: "team".to_string(),
                task_id: Some(task.id.clone()),
                teammate: "worker-a".to_string(),
            })
            .await
            .expect("task should claim");
        let error = runtime
            .message(TeamMessage {
                team_id: "team".to_string(),
                from: "worker-a".to_string(),
                to: Some("worker-a".to_string()),
                kind: TeamMessageKind::Ask,
                content: "run slow subprocess".to_string(),
                task_id: Some(task.id.clone()),
                approved: None,
                execute: true,
                resume_packet: None,
            })
            .await
            .expect_err("message should time out");
        assert!(error.to_string().contains("timed out"));
        runtime
            .complete_task(TeamTaskComplete {
                team_id: "team".to_string(),
                task_id: task.id,
                reviewer: "judge-a".to_string(),
                approved: false,
            })
            .await
            .expect("judge can reject");
        runtime.cleanup("team").expect("cleanup should run");

        assert_pid_gone(read_pid(&parent_pid));
        assert_pid_gone(read_pid(&child_pid));
        assert!(
            ProcessSupervisor::new(tempdir.join("spawns"))
                .entries()
                .expect("registry should read")
                .is_empty(),
            "team cleanup should leave no registry entries"
        );
    }

    fn runtime_config(
        root: &Path,
        definitions: Vec<RoleDefinition>,
        bindings: Vec<AgentBinding>,
        mut catalog: AgentCatalog,
    ) -> TeamRuntimeConfig {
        catalog.roles = role_summaries(&definitions);
        TeamRuntimeConfig {
            repo_root: root.to_path_buf(),
            task_root: root.join("tasks"),
            registry_dir: root.join("spawns"),
            bindings,
            catalog,
            definitions,
            max_teammates: 4,
            max_concurrent_spawns: 4,
            max_lifetime: Duration::from_secs(30),
            termination_grace: Duration::from_millis(50),
        }
    }

    fn definition(
        name: &str,
        kind: Role,
        agent: Option<&str>,
        model: Option<&str>,
    ) -> RoleDefinition {
        RoleDefinition {
            name: name.to_string(),
            kind,
            description: format!("{name} definition"),
            agent: agent.map(str::to_string),
            model: model.map(str::to_string),
            allow_tools: Vec::new(),
            prompt_addendum: "Prompt addendum.".to_string(),
            source: RoleDefinitionSource::Artesian,
            path: PathBuf::from(format!("{name}.md")),
        }
    }

    fn binding(
        role: Role,
        agent: &str,
        model: Option<&str>,
        command: &str,
        args: Vec<String>,
    ) -> AgentBinding {
        AgentBinding {
            role,
            agent: agent.to_string(),
            model: model.map(str::to_string),
            command: Some(command.to_string()),
            args,
            timeout_seconds: Some(1),
        }
    }

    fn catalog(entries: Vec<AgentCatalogEntry>) -> AgentCatalog {
        AgentCatalog {
            generated_at: Some("test".to_string()),
            agents: entries,
            roles: Vec::new(),
        }
    }

    fn entry(agent: &str, reachable: bool, models: Vec<AgentModel>) -> AgentCatalogEntry {
        AgentCatalogEntry {
            agent: agent.to_string(),
            command: Some(agent.to_string()),
            reachable,
            unreachable_reason: None,
            last_checked: Some("test".to_string()),
            models,
        }
    }

    fn model(id: &str, reachable: bool) -> AgentModel {
        AgentModel {
            id: id.to_string(),
            reachable,
            source: "test".to_string(),
        }
    }

    #[cfg(unix)]
    fn read_pid(path: &Path) -> u32 {
        fs::read_to_string(path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
            .trim()
            .parse()
            .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
    }

    #[cfg(unix)]
    fn assert_pid_gone(pid: u32) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if !pid_alive(pid) {
                return;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        panic!("pid {pid} survived cleanup");
    }

    #[cfg(unix)]
    fn pid_alive(pid: u32) -> bool {
        std::process::Command::new("kill")
            .args(["-0", &pid.to_string()])
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    }

    // ── Delegation unit tests ──────────────────────────────────────────────────────────────────

    /// A `Delegation` built from a role definition maps its three knobs to the values that the
    /// spawn + send path will actually use: the binding is the agent/model selection (knob 1),
    /// the instruction is the assembled prompt addendum + task content (knob 2), and allowed_tools
    /// comes from the definition's `allow_tools` list (knob 3).
    #[test]
    fn delegation_from_definition_maps_three_knobs() {
        let def = definition(
            "security-reviewer",
            Role::Worker,
            Some("codex"),
            Some("gpt-5"),
        );
        let binding = AgentBinding {
            role: Role::Worker,
            agent: "codex".to_string(),
            model: Some("gpt-5".to_string()),
            command: Some("codex".to_string()),
            args: Vec::new(),
            timeout_seconds: Some(60),
        };
        let task_content = "review auth.rs for injection risks";

        let delegation = Delegation::from_definition(&def, binding.clone(), task_content);

        // Knob 1 — agent selection: binding is passed through unchanged.
        assert_eq!(delegation.binding.agent, "codex");
        assert_eq!(delegation.binding.model.as_deref(), Some("gpt-5"));

        // Knob 2 — targeted instruction: prompt_addendum prefixed before task content.
        assert!(
            delegation.instruction.contains(&def.prompt_addendum),
            "instruction should include definition prompt addendum"
        );
        assert!(
            delegation.instruction.contains(task_content),
            "instruction should include task content"
        );
        // No resume packet by default.
        assert!(delegation.resume_packet.is_none());

        // Knob 3 — context visibility: allow_tools from the definition.
        assert_eq!(delegation.allowed_tools, def.allow_tools);
        // No session scoping by default.
        assert!(delegation.session_context.is_none());
    }

    /// Builder overrides let callers replace individual knobs without touching the others.
    #[test]
    fn delegation_builder_overrides_individual_knobs() {
        let def = definition("worker-a", Role::Worker, Some("echo"), None);
        let binding = AgentBinding {
            role: Role::Worker,
            agent: "echo".to_string(),
            model: None,
            command: Some("echo".to_string()),
            args: Vec::new(),
            timeout_seconds: None,
        };

        let delegation = Delegation::from_definition(&def, binding, "task")
            .with_resume_packet("prior-session-state")
            .with_allowed_tools(vec!["memory.find".to_string()])
            .with_session_context(DelegationSessionContext {
                user_id: Some("user-1".to_string()),
                session_id: Some("ses-1".to_string()),
                task_id: Some("task-1".to_string()),
            });

        // Knob 2 — resume packet set.
        assert_eq!(
            delegation.resume_packet.as_deref(),
            Some("prior-session-state")
        );

        // Knob 3 — tool override applied.
        assert_eq!(delegation.allowed_tools, vec!["memory.find"]);

        // Knob 3 — session context populated.
        let ctx = delegation
            .session_context
            .as_ref()
            .expect("session context should be set");
        assert_eq!(ctx.user_id.as_deref(), Some("user-1"));
        assert_eq!(ctx.session_id.as_deref(), Some("ses-1"));
        assert_eq!(ctx.task_id.as_deref(), Some("task-1"));
    }

    /// When a definition has an empty `prompt_addendum`, the instruction equals the task content.
    #[test]
    fn delegation_empty_addendum_yields_plain_task_content() {
        let def = RoleDefinition {
            name: "simple".to_string(),
            kind: Role::Worker,
            description: "Simple worker.".to_string(),
            agent: Some("echo".to_string()),
            model: None,
            allow_tools: Vec::new(),
            prompt_addendum: String::new(),
            source: RoleDefinitionSource::Artesian,
            path: PathBuf::from("simple.md"),
        };
        let binding = AgentBinding {
            role: Role::Worker,
            agent: "echo".to_string(),
            model: None,
            command: Some("echo".to_string()),
            args: Vec::new(),
            timeout_seconds: None,
        };
        let delegation = Delegation::from_definition(&def, binding, "just the task");
        assert_eq!(delegation.instruction, "just the task");
    }
}
