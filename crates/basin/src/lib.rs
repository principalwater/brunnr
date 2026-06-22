// SPDX-License-Identifier: Apache-2.0

//! Basin orchestration loop.

use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Duration};

use aquifer::{
    MemoryBackend, MemoryError, MemoryQuery, MemoryScope, MemoryTier, SearchHit, StoreMemory,
};
use artesian_core::{
    Agent, AgentCapabilities, AgentError, AgentEvent, AgentEventStream, AgentMessage,
    AgentResponse, AgentResult, AgentSession, ArtesianConfig, CompletedJob, EventEnvelope,
    EventSender, EventType, Job, JobStatus, Mode, ResourceQuota, Role, SpawnRequest,
    TokenAccounting,
};
use chrono::Utc;
use futures_util::{future::BoxFuture, stream, FutureExt};
use headrace::{
    ClaimRequest, Task, TaskError, TaskStatus, TaskStore, TransitionTask, VerifierGate,
};
use sandbox::{WorkspaceError, WorkspaceProvider};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::time;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Topology {
    #[default]
    Hierarchical,
    Debate,
    Router,
    Pipeline,
}

#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    pub mode: Mode,
    pub repo_root: PathBuf,
    pub concurrency_limit: usize,
    pub max_retries: usize,
    pub retry_backoff: Duration,
    pub memory_limit: usize,
    pub quotas: Vec<ResourceQuota>,
    pub topology: Topology,
}

impl OrchestratorConfig {
    pub fn from_artesian(config: &ArtesianConfig, repo_root: impl Into<PathBuf>) -> Self {
        Self {
            mode: config.mode,
            repo_root: repo_root.into(),
            concurrency_limit: config.coordination.concurrency_limit.unwrap_or(2).max(1),
            max_retries: config.coordination.max_retries.unwrap_or(1),
            retry_backoff: Duration::from_millis(
                config.coordination.retry_backoff_millis.unwrap_or_default(),
            ),
            memory_limit: 5,
            quotas: config
                .coordination
                .quotas
                .iter()
                .map(|quota| ResourceQuota {
                    agent_id: quota.agent_id.clone(),
                    user_id: quota.user_id.clone(),
                    max_prompt_tokens: quota.max_prompt_tokens,
                    max_requests_per_minute: quota.max_requests_per_minute,
                })
                .collect(),
            topology: parse_topology(config.coordination.topology.as_deref()),
        }
    }

    pub fn memory_disabled(repo_root: impl Into<PathBuf>) -> Self {
        Self {
            mode: Mode::Memory,
            repo_root: repo_root.into(),
            concurrency_limit: 1,
            max_retries: 0,
            retry_backoff: Duration::ZERO,
            memory_limit: 0,
            quotas: Vec::new(),
            topology: Topology::Hierarchical,
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ReconcileReport {
    pub dispatched: usize,
    pub completed: usize,
    pub blocked: usize,
    pub idle: bool,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RunUntilIdleReport {
    pub ticks: usize,
    pub completed: usize,
    pub blocked: usize,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RunLog {
    pub events: Vec<EventEnvelope>,
    pub token_accounting: Vec<TokenAccounting>,
    pub completed: Vec<CompletedJob>,
}

#[derive(Debug, thiserror::Error)]
pub enum OrchestratorError {
    #[error("orchestration is disabled in mode {0:?}")]
    Disabled(Mode),
    #[error("task store failed: {0}")]
    Task(#[from] TaskError),
    #[error("memory backend failed: {0}")]
    Memory(#[from] MemoryError),
    #[error("workspace provider failed: {0}")]
    Workspace(#[from] WorkspaceError),
    #[error("agent failed: {0}")]
    Agent(String),
}

impl From<AgentError> for OrchestratorError {
    fn from(value: AgentError) -> Self {
        Self::Agent(value.to_string())
    }
}

pub type OrchestratorResult<T> = Result<T, OrchestratorError>;

pub struct Orchestrator {
    config: OrchestratorConfig,
    task_store: Arc<dyn TaskStore>,
    memory: Arc<dyn MemoryBackend>,
    workspace_provider: Arc<dyn WorkspaceProvider>,
    worker: Arc<dyn Agent>,
    judge: Option<Arc<dyn Agent>>,
    verifier_gate: VerifierGate,
    run_log: RunLog,
    state: LoopState,
}

impl Orchestrator {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: OrchestratorConfig,
        task_store: Arc<dyn TaskStore>,
        memory: Arc<dyn MemoryBackend>,
        workspace_provider: Arc<dyn WorkspaceProvider>,
        worker: Arc<dyn Agent>,
        judge: Option<Arc<dyn Agent>>,
        verifier_gate: VerifierGate,
    ) -> Self {
        Self {
            config,
            task_store,
            memory,
            workspace_provider,
            worker,
            judge,
            verifier_gate,
            run_log: RunLog::default(),
            state: LoopState::default(),
        }
    }

    pub fn run_log(&self) -> &RunLog {
        &self.run_log
    }

    pub fn config_mut(&mut self) -> &mut OrchestratorConfig {
        &mut self.config
    }

    pub async fn run_until_idle(
        &mut self,
        max_ticks: usize,
    ) -> OrchestratorResult<RunUntilIdleReport> {
        let mut report = RunUntilIdleReport::default();
        for _ in 0..max_ticks {
            let tick = self.run_once().await?;
            report.ticks += 1;
            report.completed += tick.completed;
            report.blocked += tick.blocked;
            if tick.idle {
                break;
            }
        }
        Ok(report)
    }

    pub async fn run_once(&mut self) -> OrchestratorResult<ReconcileReport> {
        if !matches!(self.config.mode, Mode::Orchestrate | Mode::Full) {
            return Err(OrchestratorError::Disabled(self.config.mode));
        }

        let tasks = self.task_store.list().await?;
        let mut eligible = tasks
            .iter()
            .filter(|task| task.is_dispatch_eligible(&tasks))
            .cloned()
            .collect::<Vec<_>>();
        eligible.sort_by_key(|task| task.created_at);

        let mut claimed = Vec::new();
        for task in eligible {
            if claimed.len() >= self.config.concurrency_limit {
                break;
            }
            let estimated_prompt_tokens = estimate_task_prompt_tokens(&task);
            if !self.quota_allows("worker", estimated_prompt_tokens) {
                self.push_event(
                    task.id.clone(),
                    Role::Master,
                    "basin",
                    EventType::Status,
                    json!({
                        "task_id": task.id,
                        "status": "quota-exhausted"
                    }),
                );
                continue;
            }
            self.push_event(
                task.id.clone(),
                Role::Master,
                "basin",
                EventType::TaskAnnounced,
                json!({
                    "task_id": task.id,
                    "title": task.title
                }),
            );
            let Some(task) = self
                .task_store
                .claim(ClaimRequest {
                    task_id: Some(task.id.clone()),
                    claimant: "basin-worker".to_string(),
                })
                .await?
            else {
                continue;
            };
            self.record_request("worker", estimated_prompt_tokens);
            self.push_event(
                task.id.clone(),
                Role::Worker,
                "basin-worker",
                EventType::TaskClaimed,
                json!({
                    "task_id": task.id,
                    "claimed_by": task.claimed_by
                }),
            );
            claimed.push(task);
        }

        if claimed.is_empty() {
            let doing = tasks
                .iter()
                .filter(|task| task.status == TaskStatus::Doing)
                .map(|task| task.id.clone())
                .collect::<Vec<_>>();
            if !doing.is_empty() {
                self.push_event(
                    "orchestrator".to_string(),
                    Role::Master,
                    "basin",
                    EventType::Status,
                    json!({
                        "status": "stalled",
                        "doing": doing
                    }),
                );
            }
            return Ok(ReconcileReport {
                dispatched: 0,
                completed: 0,
                blocked: 0,
                idle: doing.is_empty(),
            });
        }

        let dispatches = claimed.into_iter().map(|task| {
            dispatch_task(
                self.config.clone(),
                self.memory.clone(),
                self.workspace_provider.clone(),
                self.worker.clone(),
                self.judge.clone(),
                self.verifier_gate.clone(),
                task,
            )
        });
        let outcomes = futures_util::future::join_all(dispatches).await;
        let mut report = ReconcileReport {
            dispatched: outcomes.len(),
            ..ReconcileReport::default()
        };
        for outcome in outcomes {
            match outcome {
                Ok(DispatchOutcome::Passed {
                    task,
                    response,
                    accounting,
                }) => {
                    self.run_log.token_accounting.push(accounting);
                    self.push_event(
                        task.id.clone(),
                        Role::Worker,
                        "basin-worker",
                        EventType::Result,
                        json!({
                            "task_id": task.id,
                            "content": response.content
                        }),
                    );
                    self.push_event(
                        task.id.clone(),
                        Role::Judge,
                        "basin-judge",
                        EventType::Verdict,
                        json!({
                            "task_id": task.id,
                            "passed": true
                        }),
                    );
                    let completed = CompletedJob {
                        job: job_from_task(&task, JobStatus::Done),
                        commit: None,
                        completed_at: Utc::now(),
                    };
                    let done = self
                        .task_store
                        .transition(TransitionTask {
                            id: task.id.clone(),
                            status: TaskStatus::Done,
                        })
                        .await?;
                    self.state.retries.remove(&done.id);
                    self.run_log.completed.push(completed);
                    report.completed += 1;
                }
                Ok(DispatchOutcome::Failed { task, reason }) => {
                    self.push_event(
                        task.id.clone(),
                        Role::Judge,
                        "basin-judge",
                        EventType::Verdict,
                        json!({
                            "task_id": task.id,
                            "passed": false,
                            "reason": reason
                        }),
                    );
                    if self.should_retry(&task.id).await {
                        self.task_store
                            .transition(TransitionTask {
                                id: task.id,
                                status: TaskStatus::Todo,
                            })
                            .await?;
                    } else {
                        self.push_event(
                            task.id.clone(),
                            Role::Master,
                            "basin",
                            EventType::Blocked,
                            json!({
                                "task_id": task.id,
                                "reason": reason
                            }),
                        );
                        self.task_store
                            .transition(TransitionTask {
                                id: task.id,
                                status: TaskStatus::Blocked,
                            })
                            .await?;
                        report.blocked += 1;
                    }
                }
                Err(error) => {
                    self.push_event(
                        "orchestrator".to_string(),
                        Role::Master,
                        "basin",
                        EventType::Error,
                        json!({
                            "error": error.to_string()
                        }),
                    );
                    return Err(error);
                }
            }
        }
        Ok(report)
    }

    fn quota_allows(&self, agent_id: &str, prompt_tokens: u64) -> bool {
        for quota in &self.config.quotas {
            if quota
                .agent_id
                .as_deref()
                .is_some_and(|quota_agent| quota_agent != agent_id)
            {
                continue;
            }
            let usage = self
                .state
                .prompt_usage
                .get(agent_id)
                .copied()
                .unwrap_or_default();
            if quota
                .max_prompt_tokens
                .is_some_and(|max| usage + prompt_tokens > max)
            {
                return false;
            }
            let requests = self
                .state
                .request_counts
                .get(agent_id)
                .copied()
                .unwrap_or_default();
            if quota
                .max_requests_per_minute
                .is_some_and(|max| requests >= max)
            {
                return false;
            }
        }
        true
    }

    fn record_request(&mut self, agent_id: &str, prompt_tokens: u64) {
        *self
            .state
            .prompt_usage
            .entry(agent_id.to_string())
            .or_default() += prompt_tokens;
        *self
            .state
            .request_counts
            .entry(agent_id.to_string())
            .or_default() += 1;
    }

    async fn should_retry(&mut self, task_id: &str) -> bool {
        let retries = self.state.retries.entry(task_id.to_string()).or_default();
        if *retries < self.config.max_retries {
            let delay = self
                .config
                .retry_backoff
                .saturating_mul(2_u32.saturating_pow(*retries as u32));
            *retries += 1;
            if delay > Duration::ZERO {
                time::sleep(delay).await;
            }
            true
        } else {
            false
        }
    }

    fn push_event(
        &mut self,
        correlation_id: String,
        role: Role,
        agent_id: &str,
        event_type: EventType,
        payload: serde_json::Value,
    ) {
        self.state.event_counter += 1;
        self.run_log.events.push(EventEnvelope::new(
            format!("evt-{}", self.state.event_counter),
            correlation_id,
            EventSender {
                role,
                agent_id: agent_id.to_string(),
            },
            event_type,
            payload,
        ));
    }
}

#[derive(Default)]
struct LoopState {
    retries: HashMap<String, usize>,
    prompt_usage: HashMap<String, u64>,
    request_counts: HashMap<String, u32>,
    event_counter: u64,
}

enum DispatchOutcome {
    Passed {
        task: Task,
        response: AgentResponse,
        accounting: TokenAccounting,
    },
    Failed {
        task: Task,
        reason: String,
    },
}

async fn dispatch_task(
    config: OrchestratorConfig,
    memory: Arc<dyn MemoryBackend>,
    workspace_provider: Arc<dyn WorkspaceProvider>,
    worker: Arc<dyn Agent>,
    judge: Option<Arc<dyn Agent>>,
    verifier_gate: VerifierGate,
    task: Task,
) -> OrchestratorResult<DispatchOutcome> {
    let lease = workspace_provider
        .lease(&config.repo_root, &format!("worker-{}", task.id))
        .await?;
    let memory_hits = memory
        .find(
            MemoryQuery::new(format!("{} {}", task.title, task.description))
                .with_limit(config.memory_limit),
        )
        .await?;
    let prompt = assemble_worker_prompt(&task, &memory_hits);
    let session = worker
        .spawn(SpawnRequest {
            role: Role::Worker,
            agent: "worker".to_string(),
            model: None,
            working_dir: Some(lease.path.display().to_string()),
        })
        .await?;
    let response = worker
        .send(
            &session,
            AgentMessage {
                content: prompt.clone(),
            },
        )
        .await;
    lease.cleanup()?;
    let response = match response {
        Ok(response) => response,
        Err(error) => {
            return Ok(DispatchOutcome::Failed {
                task,
                reason: error.to_string(),
            });
        }
    };
    memory
        .store(StoreMemory {
            content: response.content.clone(),
            tags: vec!["orchestration".to_string(), "worker-result".to_string()],
            metadata: std::collections::BTreeMap::new(),
            tier: MemoryTier::L1Atom,
            node_id: Some(format!("task:{}:worker-result", task.id)),
            created_at: None,
            scope: Some(MemoryScope::Task),
            agent_id: Some("worker".to_string()),
            session_id: Some(session.id.clone()),
            task_id: Some(task.id.clone()),
            user_id: None,
            source: None,
            confidence: None,
        })
        .await?;

    if let Err(error) = verifier_gate.verify(&task).await {
        return Ok(DispatchOutcome::Failed {
            task,
            reason: error.to_string(),
        });
    }
    if let Some(judge) = judge {
        let session = judge
            .spawn(SpawnRequest {
                role: Role::Judge,
                agent: "judge".to_string(),
                model: None,
                working_dir: None,
            })
            .await?;
        let review = judge
            .send(
                &session,
                AgentMessage {
                    content: format!(
                        "Review task {} after verifier pass.\n\nWorker result:\n{}",
                        task.id, response.content
                    ),
                },
            )
            .await?;
        if rejects(&review.content) {
            return Ok(DispatchOutcome::Failed {
                task,
                reason: review.content,
            });
        }
    }

    let accounting = TokenAccounting {
        agent_id: "worker".to_string(),
        session_id: Some(session.id),
        prompt_tokens: estimate_tokens(&prompt),
        completion_tokens: estimate_tokens(&response.content),
    };
    Ok(DispatchOutcome::Passed {
        task,
        response,
        accounting,
    })
}

fn assemble_worker_prompt(task: &Task, memory_hits: &[SearchHit]) -> String {
    let mut prompt = format!(
        "Task ID: {}\nTitle: {}\nRole: {}\n\nDescription:\n{}\n",
        task.id,
        task.title,
        task.role.canonical_alias(),
        task.description
    );
    if !memory_hits.is_empty() {
        prompt.push_str("\nRelevant memory:\n");
        for hit in memory_hits {
            prompt.push_str(&format!(
                "- {}: {}\n",
                hit.record.node_id,
                first_line(&hit.record.content)
            ));
        }
    }
    prompt
}

fn rejects(review: &str) -> bool {
    let review = review.to_ascii_lowercase();
    review.contains("reject") || review.contains("fail")
}

fn first_line(content: &str) -> &str {
    content.lines().next().unwrap_or_default()
}

fn estimate_task_prompt_tokens(task: &Task) -> u64 {
    estimate_tokens(&format!("{} {}", task.title, task.description))
}

fn estimate_tokens(text: &str) -> u64 {
    text.split_whitespace().count() as u64
}

fn job_from_task(task: &Task, status: JobStatus) -> Job {
    Job {
        id: task.id.clone(),
        title: task.title.clone(),
        role: task.role,
        status,
    }
}

fn parse_topology(input: Option<&str>) -> Topology {
    match input
        .unwrap_or("hierarchical")
        .to_ascii_lowercase()
        .as_str()
    {
        "debate" => Topology::Debate,
        "router" => Topology::Router,
        "pipeline" => Topology::Pipeline,
        _ => Topology::Hierarchical,
    }
}

#[derive(Debug, Clone)]
pub struct DryRunAgent {
    agent_id: String,
}

impl DryRunAgent {
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
        }
    }
}

impl Agent for DryRunAgent {
    fn spawn(&self, request: SpawnRequest) -> BoxFuture<'_, AgentResult<AgentSession>> {
        async move {
            Ok(AgentSession {
                id: format!(
                    "dry-run-{}-{}",
                    request.role.canonical_alias(),
                    request.agent
                ),
                role: request.role,
                agent: self.agent_id.clone(),
            })
        }
        .boxed()
    }

    fn send(
        &self,
        _session: &AgentSession,
        message: AgentMessage,
    ) -> BoxFuture<'_, AgentResult<AgentResponse>> {
        async move {
            Ok(AgentResponse {
                content: format!("dry-run ok: {}", first_line(&message.content)),
            })
        }
        .boxed()
    }

    fn stream(
        &self,
        _session: &AgentSession,
        message: AgentMessage,
    ) -> BoxFuture<'_, AgentResult<AgentEventStream>> {
        async move {
            Ok(Box::pin(stream::iter([
                Ok(AgentEvent::Text(format!(
                    "dry-run ok: {}",
                    first_line(&message.content)
                ))),
                Ok(AgentEvent::Done),
            ])) as AgentEventStream)
        }
        .boxed()
    }

    fn capabilities(&self) -> AgentCapabilities {
        AgentCapabilities {
            streaming: false,
            tools: false,
            mcp: false,
        }
    }
}
