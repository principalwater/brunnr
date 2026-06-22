// SPDX-License-Identifier: Apache-2.0

//! Process-backed [`artesian_core::Agent`] adapter.

use std::{
    collections::HashMap,
    fs, io,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use artesian_core::{
    Agent, AgentBinding, AgentCapabilities, AgentCatalog, AgentCatalogEntry, AgentError,
    AgentEvent, AgentEventStream, AgentMessage, AgentModel, AgentResponse, AgentResult,
    AgentSession, AgentUnreachableReason, Role, SpawnRequest,
};
use futures_util::{future::BoxFuture, stream, FutureExt};
use serde::{Deserialize, Serialize};
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    process::Command,
    sync::mpsc,
    task::JoinHandle,
    time,
};

#[cfg(unix)]
use nix::{
    errno::Errno,
    sys::signal::{kill, killpg, Signal},
    unistd::{getpgid, Pid},
};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_MAX_LIFETIME: Duration = Duration::from_secs(30 * 60);
const DEFAULT_TERMINATION_GRACE: Duration = Duration::from_secs(2);
const DEFAULT_MAX_CONCURRENT_SPAWNS: usize = 32;
const REGISTRY_VERSION: u32 = 1;
const MAX_PROCESS_ERROR_OUTPUT_CHARS: usize = 2_048;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessAgentConfig {
    pub agent_id: String,
    pub command: String,
    pub args: Vec<String>,
    pub working_dir: Option<PathBuf>,
    pub timeout: Duration,
    pub max_lifetime: Duration,
    pub termination_grace: Duration,
    pub registry_dir: PathBuf,
    pub max_concurrent_spawns: usize,
    pub static_models: Vec<String>,
    pub default_model: Option<String>,
}

impl ProcessAgentConfig {
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            agent_id: String::new(),
            command: command.into(),
            args: Vec::new(),
            working_dir: None,
            timeout: DEFAULT_TIMEOUT,
            max_lifetime: DEFAULT_MAX_LIFETIME,
            termination_grace: DEFAULT_TERMINATION_GRACE,
            registry_dir: default_registry_dir(),
            max_concurrent_spawns: DEFAULT_MAX_CONCURRENT_SPAWNS,
            static_models: Vec::new(),
            default_model: None,
        }
    }

    pub fn with_agent_id(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = agent_id.into();
        self
    }

    pub fn with_args(mut self, args: Vec<String>) -> Self {
        self.args = args;
        self
    }

    pub fn with_working_dir(mut self, working_dir: impl Into<PathBuf>) -> Self {
        self.working_dir = Some(working_dir.into());
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_max_lifetime(mut self, max_lifetime: Duration) -> Self {
        self.max_lifetime = max_lifetime;
        self
    }

    pub fn with_termination_grace(mut self, termination_grace: Duration) -> Self {
        self.termination_grace = termination_grace;
        self
    }

    pub fn with_registry_dir(mut self, registry_dir: impl Into<PathBuf>) -> Self {
        self.registry_dir = registry_dir.into();
        self
    }

    pub fn with_max_concurrent_spawns(mut self, max_concurrent_spawns: usize) -> Self {
        self.max_concurrent_spawns = max_concurrent_spawns.max(1);
        self
    }

    pub fn with_static_models(mut self, models: Vec<String>) -> Self {
        self.static_models = models;
        self
    }

    pub fn with_default_model(mut self, model: Option<String>) -> Self {
        self.default_model = model;
        self
    }

    pub fn supervisor(&self) -> ProcessSupervisor {
        ProcessSupervisor::new(&self.registry_dir)
            .with_termination_grace(self.termination_grace)
            .with_max_concurrent_spawns(self.max_concurrent_spawns)
    }
}

#[derive(Debug, Clone)]
pub struct ProcessAgent {
    config: ProcessAgentConfig,
    sessions: Arc<Mutex<HashMap<String, SessionContext>>>,
    next_session: Arc<AtomicU64>,
}

impl ProcessAgent {
    pub fn new(config: ProcessAgentConfig) -> Self {
        let _ = config.supervisor().cleanup_dead_entries();
        Self {
            config,
            sessions: Arc::new(Mutex::new(HashMap::new())),
            next_session: Arc::new(AtomicU64::new(1)),
        }
    }

    pub fn config(&self) -> &ProcessAgentConfig {
        &self.config
    }

    pub async fn send_with_event_sender(
        &self,
        session: &AgentSession,
        message: AgentMessage,
        event_sender: Option<mpsc::UnboundedSender<WorkerEvent>>,
    ) -> AgentResult<AgentResponse> {
        let context = self
            .sessions
            .lock()
            .map_err(|error| AgentError::Session(error.to_string()))?
            .get(&session.id)
            .cloned()
            .ok_or_else(|| AgentError::Session(format!("unknown session: {}", session.id)))?;
        let output = run_process(&self.config, &context, &message.content, event_sender).await?;
        Ok(AgentResponse { content: output })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerEvent {
    pub kind: String,
    pub text: String,
    pub raw: String,
}

impl Agent for ProcessAgent {
    fn spawn(&self, request: SpawnRequest) -> BoxFuture<'_, AgentResult<AgentSession>> {
        async move {
            if self.config.command.trim().is_empty() {
                return Err(AgentError::Unavailable(
                    "process agent command is empty".to_string(),
                ));
            }
            let model = request
                .model
                .clone()
                .or_else(|| self.config.default_model.clone());
            validate_model(&self.config, model.as_deref()).await?;
            let id = format!(
                "{}-{}-{}",
                request.role.canonical_alias(),
                sanitize_agent_id(&request.agent),
                self.next_session.fetch_add(1, Ordering::Relaxed)
            );
            let context = SessionContext {
                role: request.role,
                agent: request.agent.clone(),
                model,
                working_dir: request
                    .working_dir
                    .map(PathBuf::from)
                    .or_else(|| self.config.working_dir.clone()),
            };
            self.sessions
                .lock()
                .map_err(|error| AgentError::Session(error.to_string()))?
                .insert(id.clone(), context);
            Ok(AgentSession {
                id,
                role: request.role,
                agent: request.agent,
            })
        }
        .boxed()
    }

    fn send(
        &self,
        session: &AgentSession,
        message: AgentMessage,
    ) -> BoxFuture<'_, AgentResult<AgentResponse>> {
        let session = session.clone();
        async move { self.send_with_event_sender(&session, message, None).await }.boxed()
    }

    fn stream(
        &self,
        session: &AgentSession,
        message: AgentMessage,
    ) -> BoxFuture<'_, AgentResult<AgentEventStream>> {
        let session = session.clone();
        let agent = self.clone();
        async move {
            let (event_tx, event_rx) = mpsc::unbounded_channel();
            let (worker_tx, mut worker_rx) = mpsc::unbounded_channel();
            let forward_tx = event_tx.clone();
            let forward = tokio::spawn(async move {
                while let Some(event) = worker_rx.recv().await {
                    if forward_tx
                        .send(Ok(AgentEvent::Text(worker_event_log_line(&event))))
                        .is_err()
                    {
                        return;
                    }
                }
            });
            tokio::spawn(async move {
                let result = agent
                    .send_with_event_sender(&session, message, Some(worker_tx))
                    .await;
                let _ = forward.await;
                match result {
                    Ok(_) => {
                        let _ = event_tx.send(Ok(AgentEvent::Done));
                    }
                    Err(error) => {
                        let _ = event_tx.send(Err(error));
                    }
                }
            });
            Ok(Box::pin(stream::unfold(event_rx, |mut receiver| async {
                receiver.recv().await.map(|event| (event, receiver))
            })) as AgentEventStream)
        }
        .boxed()
    }

    fn capabilities(&self) -> AgentCapabilities {
        AgentCapabilities {
            streaming: true,
            tools: false,
            mcp: false,
        }
    }

    fn list_models(&self) -> BoxFuture<'_, AgentResult<Vec<AgentModel>>> {
        async move { Ok(discover_models(&self.config).await) }.boxed()
    }
}

pub async fn refresh_agent_catalog(
    bindings: &[AgentBinding],
    cache_path: impl AsRef<Path>,
) -> AgentResult<AgentCatalog> {
    let mut entries = Vec::new();
    for binding in bindings {
        let command = binding
            .command
            .clone()
            .unwrap_or_else(|| binding.agent.clone());
        let config = ProcessAgentConfig::new(command.clone()).with_agent_id(binding.agent.clone());
        let reachability = command_reachability(&config.command);
        entries.push(AgentCatalogEntry {
            agent: binding.agent.clone(),
            command: Some(command),
            reachable: reachability.reachable,
            unreachable_reason: reachability.unreachable_reason,
            last_checked: Some(reachability.last_checked),
            models: discover_models(&config).await,
        });
    }
    entries.sort_by(|left, right| {
        left.agent
            .cmp(&right.agent)
            .then_with(|| left.command.cmp(&right.command))
    });
    entries.dedup_by(|left, right| left.agent == right.agent && left.command == right.command);
    let catalog = AgentCatalog {
        generated_at: Some(now_unix_ms().to_string()),
        agents: entries,
        roles: Vec::new(),
    };
    write_catalog(cache_path, &catalog)?;
    Ok(catalog)
}

pub fn fallback_agent_catalog(bindings: &[AgentBinding]) -> AgentCatalog {
    let mut entries = bindings
        .iter()
        .map(|binding| {
            let command = binding
                .command
                .clone()
                .unwrap_or_else(|| binding.agent.clone());
            let reachability = command_reachability(&command);
            let models = curated_static_models(&binding.agent)
                .iter()
                .map(|model| AgentModel {
                    id: model.to_string(),
                    reachable: reachability.reachable,
                    source: "static-fallback".to_string(),
                })
                .collect();
            AgentCatalogEntry {
                agent: binding.agent.clone(),
                command: Some(command),
                reachable: reachability.reachable,
                unreachable_reason: reachability.unreachable_reason,
                last_checked: Some(reachability.last_checked),
                models,
            }
        })
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| {
        left.agent
            .cmp(&right.agent)
            .then_with(|| left.command.cmp(&right.command))
    });
    entries.dedup_by(|left, right| left.agent == right.agent && left.command == right.command);
    AgentCatalog {
        generated_at: Some(now_unix_ms().to_string()),
        agents: entries,
        roles: Vec::new(),
    }
}

pub fn load_agent_catalog(cache_path: impl AsRef<Path>) -> AgentResult<AgentCatalog> {
    let text = fs::read_to_string(cache_path.as_ref())
        .map_err(|error| AgentError::Unavailable(error.to_string()))?;
    serde_json::from_str(&text).map_err(|error| AgentError::Unavailable(error.to_string()))
}

pub async fn load_or_refresh_agent_catalog(
    bindings: &[AgentBinding],
    cache_path: impl AsRef<Path>,
    refresh: bool,
) -> AgentResult<AgentCatalog> {
    let cache_path = cache_path.as_ref();
    if !refresh {
        if let Ok(catalog) = load_agent_catalog(cache_path) {
            return Ok(catalog);
        }
    }
    refresh_agent_catalog(bindings, cache_path).await
}

pub fn validate_binding_model(binding: &AgentBinding, catalog: &AgentCatalog) -> AgentResult<()> {
    let Some(model) = binding.model.as_deref() else {
        return Ok(());
    };
    let Some(entry) = catalog
        .agents
        .iter()
        .find(|entry| entry.agent == binding.agent)
    else {
        return Err(AgentError::Unavailable(format!(
            "agent '{}' is not in the model catalog; run `artesian agents refresh`",
            binding.agent
        )));
    };
    if entry
        .models
        .iter()
        .any(|candidate| candidate.id == model && candidate.reachable)
    {
        return Ok(());
    }
    let available = entry
        .models
        .iter()
        .filter(|model| model.reachable)
        .map(|model| model.id.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    Err(AgentError::Unavailable(format!(
        "model '{model}' is unavailable for agent '{}'; available models: {}",
        binding.agent,
        if available.is_empty() {
            "<none>; run `artesian agents refresh` or configure a supported model"
        } else {
            available.as_str()
        }
    )))
}

fn write_catalog(cache_path: impl AsRef<Path>, catalog: &AgentCatalog) -> AgentResult<()> {
    let cache_path = cache_path.as_ref();
    if let Some(parent) = cache_path.parent() {
        fs::create_dir_all(parent).map_err(|error| AgentError::Unavailable(error.to_string()))?;
    }
    fs::write(
        cache_path,
        serde_json::to_vec_pretty(catalog)
            .map_err(|error| AgentError::Unavailable(error.to_string()))?,
    )
    .map_err(|error| AgentError::Unavailable(error.to_string()))?;
    restrict_file_permissions(cache_path)
        .map_err(|error| AgentError::Unavailable(error.to_string()))
}

#[cfg(unix)]
fn restrict_file_permissions(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let mut permissions = fs::metadata(path)?.permissions();
    permissions.set_mode(0o600);
    fs::set_permissions(path, permissions)
}

#[cfg(not(unix))]
fn restrict_file_permissions(_path: &Path) -> io::Result<()> {
    Ok(())
}

async fn validate_model(config: &ProcessAgentConfig, model: Option<&str>) -> AgentResult<()> {
    let Some(model) = model else {
        return Ok(());
    };
    let models = discover_models(config).await;
    if models
        .iter()
        .any(|candidate| candidate.id == model && candidate.reachable)
    {
        return Ok(());
    }
    let available = models
        .iter()
        .filter(|model| model.reachable)
        .map(|model| model.id.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    Err(AgentError::Unavailable(format!(
        "model '{model}' is unavailable for agent '{}'; available models: {}",
        logical_agent_id(config),
        if available.is_empty() {
            "<none>; run `artesian agents refresh` or choose a configured model"
        } else {
            available.as_str()
        }
    )))
}

async fn discover_models(config: &ProcessAgentConfig) -> Vec<AgentModel> {
    let reachable = command_reachable(&config.command);
    let mut models = Vec::new();
    models.extend(config.static_models.iter().map(|model| AgentModel {
        id: model.clone(),
        reachable,
        source: "config".to_string(),
    }));
    if models.is_empty() {
        if let Some(cli_models) = discover_models_from_env_command(config).await {
            models.extend(cli_models.into_iter().map(|model| AgentModel {
                id: model,
                reachable,
                source: "cli-list-models".to_string(),
            }));
        }
    }
    if models.is_empty() {
        if let Some(provider_models) = discover_provider_models(config).await {
            models.extend(provider_models.into_iter().map(|model| AgentModel {
                id: model,
                reachable,
                source: "provider-api".to_string(),
            }));
        }
    }
    for model in curated_static_models(logical_agent_id(config)) {
        if models.iter().any(|existing| existing.id == *model) {
            continue;
        }
        models.push(AgentModel {
            id: model.to_string(),
            reachable,
            source: "static-fallback".to_string(),
        });
    }
    models.sort_by(|left, right| left.id.cmp(&right.id));
    models.dedup_by(|left, right| left.id == right.id);
    models
}

async fn discover_models_from_env_command(config: &ProcessAgentConfig) -> Option<Vec<String>> {
    let env_name = format!(
        "ARTESIAN_{}_MODELS_CMD",
        sanitize_env_token(logical_agent_id(config))
    );
    let command = std::env::var(env_name).ok()?;
    let output = time::timeout(Duration::from_secs(2), async {
        Command::new("sh").arg("-c").arg(command).output().await
    })
    .await
    .ok()?
    .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    Some(parse_model_list(&redact_secrets(&text)))
}

async fn discover_provider_models(config: &ProcessAgentConfig) -> Option<Vec<String>> {
    let agent = normalize_agent_id(logical_agent_id(config));
    if !matches!(agent.as_str(), "codex" | "openai") {
        return None;
    }
    let api_key = std::env::var("OPENAI_API_KEY").ok()?;
    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .ok()?
        .get("https://api.openai.com/v1/models")
        .bearer_auth(api_key)
        .send()
        .await
        .ok()?;
    if !response.status().is_success() {
        return None;
    }
    let payload = response.json::<OpenAiModels>().await.ok()?;
    let mut models = payload
        .data
        .into_iter()
        .map(|model| model.id)
        .collect::<Vec<_>>();
    models.sort();
    models.dedup();
    Some(models)
}

#[derive(Debug, Deserialize)]
struct OpenAiModels {
    data: Vec<OpenAiModel>,
}

#[derive(Debug, Deserialize)]
struct OpenAiModel {
    id: String,
}

fn parse_model_list(text: &str) -> Vec<String> {
    if let Ok(models) = serde_json::from_str::<Vec<String>>(text) {
        return models;
    }
    text.lines()
        .flat_map(|line| line.split(','))
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .map(str::to_string)
        .collect()
}

fn curated_static_models(agent_id: &str) -> &'static [&'static str] {
    match normalize_agent_id(agent_id).as_str() {
        "claude" | "claude-code" => &["claude-haiku", "claude-opus", "claude-sonnet"],
        "codex" => &["gpt-5", "gpt-5.5", "gpt-5-mini"],
        "gemini" => &["gemini-flash", "gemini-pro"],
        "ollama" => &["llama3.2", "mistral", "qwen2.5-coder"],
        "opencode" => &["opencode-default"],
        _ => &[],
    }
}

fn logical_agent_id(config: &ProcessAgentConfig) -> &str {
    if config.agent_id.trim().is_empty() {
        &config.command
    } else {
        &config.agent_id
    }
}

fn normalize_agent_id(agent_id: &str) -> String {
    agent_id
        .trim()
        .to_ascii_lowercase()
        .replace(['_', ' '], "-")
}

fn sanitize_env_token(agent_id: &str) -> String {
    agent_id
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect()
}

fn command_reachable(command: &str) -> bool {
    command_reachability(command).reachable
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CommandReachability {
    reachable: bool,
    unreachable_reason: Option<AgentUnreachableReason>,
    last_checked: String,
}

fn command_reachability(command: &str) -> CommandReachability {
    let last_checked = now_unix_ms().to_string();
    if command.trim().is_empty() {
        return CommandReachability {
            reachable: false,
            unreachable_reason: Some(AgentUnreachableReason::NoCommand),
            last_checked,
        };
    }
    let path = Path::new(command);
    if path.components().count() > 1 {
        let reachable = path.exists();
        return CommandReachability {
            reachable,
            unreachable_reason: (!reachable).then_some(AgentUnreachableReason::NoCommand),
            last_checked,
        };
    }
    let reachable = std::env::var_os("PATH").is_some_and(|paths| {
        std::env::split_paths(&paths).any(|directory| directory.join(command).exists())
    });
    CommandReachability {
        reachable,
        unreachable_reason: (!reachable).then_some(AgentUnreachableReason::NoCommand),
        last_checked,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SessionContext {
    role: Role,
    agent: String,
    model: Option<String>,
    working_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpawnRegistryEntry {
    pub version: u32,
    pub id: String,
    pub owner_pid: u32,
    #[serde(default)]
    pub owner_exe: Option<String>,
    pub pid: u32,
    pub pgid: i32,
    pub task_id: Option<String>,
    pub started_at_unix_ms: u128,
    #[serde(default)]
    pub last_heartbeat_unix_ms: Option<u128>,
    pub command: Vec<String>,
    pub working_dir: Option<String>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ReapReport {
    pub scanned: usize,
    pub terminated: usize,
    pub removed: usize,
    pub skipped_unverified: usize,
    /// Of the reaped entries, how many were reaped because they exceeded the
    /// TTL or went heartbeat-stale (as opposed to having a dead owner).
    pub expired: usize,
}

/// Tunables for [`ProcessSupervisor::gc`]. Each bound is opt-in: `None` disables
/// that reaping rule, so a bare `GcOptions::default()` reaps only orphans
/// (entries whose owner process is gone) plus dead registry entries.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct GcOptions {
    /// Maximum wall-clock age of a spawn before it is reclaimed, regardless of
    /// liveness — a guard against runaway workers that never exit.
    pub ttl: Option<Duration>,
    /// Maximum time without a heartbeat before a spawn is considered hung and
    /// reclaimed. Only applies to entries that have recorded a heartbeat.
    pub heartbeat_timeout: Option<Duration>,
}

impl GcOptions {
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    pub fn with_heartbeat_timeout(mut self, heartbeat_timeout: Duration) -> Self {
        self.heartbeat_timeout = Some(heartbeat_timeout);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessSupervisor {
    registry_dir: PathBuf,
    termination_grace: Duration,
    max_concurrent_spawns: usize,
}

impl ProcessSupervisor {
    pub fn new(registry_dir: impl Into<PathBuf>) -> Self {
        Self {
            registry_dir: registry_dir.into(),
            termination_grace: DEFAULT_TERMINATION_GRACE,
            max_concurrent_spawns: DEFAULT_MAX_CONCURRENT_SPAWNS,
        }
    }

    pub fn default_for_current_dir() -> Self {
        Self::new(default_registry_dir())
    }

    pub fn with_termination_grace(mut self, termination_grace: Duration) -> Self {
        self.termination_grace = termination_grace;
        self
    }

    pub fn with_max_concurrent_spawns(mut self, max_concurrent_spawns: usize) -> Self {
        self.max_concurrent_spawns = max_concurrent_spawns.max(1);
        self
    }

    pub fn registry_dir(&self) -> &Path {
        &self.registry_dir
    }

    pub fn cleanup_dead_entries(&self) -> io::Result<ReapReport> {
        let mut report = ReapReport::default();
        for entry in self.entries()? {
            report.scanned += 1;
            if self.verified_alive(&entry) {
                continue;
            }
            self.remove_entry(&entry)?;
            report.removed += 1;
        }
        Ok(report)
    }

    pub fn reap_stale(&self) -> io::Result<ReapReport> {
        let mut report = ReapReport::default();
        for entry in self.entries()? {
            report.scanned += 1;
            if owner_alive(&entry) {
                continue;
            }
            if self.verified_alive(&entry) {
                self.terminate_group(entry.pgid)?;
                report.terminated += 1;
            } else {
                report.skipped_unverified += 1;
            }
            self.remove_entry(&entry)?;
            report.removed += 1;
        }
        Ok(report)
    }

    /// Registry-wide garbage collection. Reclaims, in one pass:
    /// - dead registry entries (the process group is already gone);
    /// - orphans (the owner process that spawned them has exited);
    /// - spawns older than `options.ttl` (runaway guard);
    /// - spawns whose last heartbeat is older than `options.heartbeat_timeout` (hung guard).
    ///
    /// Live, owned, fresh spawns are left untouched, so this is safe to call on
    /// a timer or before every spawn without disturbing healthy workers.
    pub fn gc(&self, options: GcOptions) -> io::Result<ReapReport> {
        let mut report = ReapReport::default();
        let now = now_unix_ms();
        for entry in self.entries()? {
            report.scanned += 1;
            let alive = self.verified_alive(&entry);
            if !alive {
                // The process group is gone; just drop the stale record.
                self.remove_entry(&entry)?;
                report.removed += 1;
                continue;
            }
            let ttl_expired = options
                .ttl
                .is_some_and(|ttl| age_since(now, entry.started_at_unix_ms) > ttl);
            let heartbeat_stale = match (options.heartbeat_timeout, entry.last_heartbeat_unix_ms) {
                (Some(timeout), Some(beat)) => age_since(now, beat) > timeout,
                _ => false,
            };
            let orphaned = !owner_alive(&entry);
            if orphaned || ttl_expired || heartbeat_stale {
                self.terminate_group(entry.pgid)?;
                report.terminated += 1;
                self.remove_entry(&entry)?;
                report.removed += 1;
                if ttl_expired || heartbeat_stale {
                    report.expired += 1;
                }
            }
        }
        Ok(report)
    }

    /// Refresh the heartbeat timestamp on a tracked spawn so [`gc`](Self::gc)
    /// with a `heartbeat_timeout` does not reclaim it. Returns `false` when no
    /// entry with that id exists (already reaped or never registered).
    pub fn record_heartbeat(&self, id: &str) -> io::Result<bool> {
        let Some(mut entry) = self.entry_by_id(id)? else {
            return Ok(false);
        };
        entry.last_heartbeat_unix_ms = Some(now_unix_ms());
        let path = self.entry_path(&entry);
        let temporary = path.with_extension("json.tmp");
        fs::write(&temporary, serde_json::to_vec_pretty(&entry)?)?;
        fs::rename(temporary, path)?;
        Ok(true)
    }

    fn entry_by_id(&self, id: &str) -> io::Result<Option<SpawnRegistryEntry>> {
        Ok(self.entries()?.into_iter().find(|entry| entry.id == id))
    }

    pub fn terminate_all_tracked(&self) -> io::Result<ReapReport> {
        self.terminate_where(|_| true)
    }

    pub fn terminate_current_owner(&self) -> io::Result<ReapReport> {
        let owner_pid = std::process::id();
        self.terminate_where(|entry| entry.owner_pid == owner_pid)
    }

    fn terminate_where(
        &self,
        mut should_terminate: impl FnMut(&SpawnRegistryEntry) -> bool,
    ) -> io::Result<ReapReport> {
        let mut report = ReapReport::default();
        for entry in self.entries()? {
            report.scanned += 1;
            if !should_terminate(&entry) {
                continue;
            }
            if self.verified_alive(&entry) {
                self.terminate_group(entry.pgid)?;
                report.terminated += 1;
            } else {
                report.skipped_unverified += 1;
            }
            self.remove_entry(&entry)?;
            report.removed += 1;
        }
        Ok(report)
    }

    pub fn terminate_group(&self, pgid: i32) -> io::Result<()> {
        terminate_group(pgid, self.termination_grace)
    }

    pub fn group_alive(&self, pgid: i32) -> bool {
        group_alive(pgid)
    }

    pub fn entries(&self) -> io::Result<Vec<SpawnRegistryEntry>> {
        let mut entries = Vec::new();
        match fs::read_dir(&self.registry_dir) {
            Ok(read_dir) => {
                for item in read_dir {
                    let path = item?.path();
                    if path.extension().and_then(|extension| extension.to_str()) != Some("json") {
                        continue;
                    }
                    let text = fs::read_to_string(&path)?;
                    match serde_json::from_str::<SpawnRegistryEntry>(&text) {
                        Ok(entry) => entries.push(entry),
                        Err(_) => {
                            let _ = fs::remove_file(path);
                        }
                    }
                }
                entries.sort_by(|left, right| left.id.cmp(&right.id));
                Ok(entries)
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(entries),
            Err(error) => Err(error),
        }
    }

    fn ensure_spawn_capacity(&self) -> io::Result<()> {
        self.cleanup_dead_entries()?;
        let active = self
            .entries()?
            .into_iter()
            .filter(|entry| self.verified_alive(entry))
            .count();
        if active >= self.max_concurrent_spawns {
            return Err(io::Error::other(format!(
                "spawn cap reached: active={} cap={}",
                active, self.max_concurrent_spawns
            )));
        }
        Ok(())
    }

    fn register_spawn(
        &self,
        pid: u32,
        pgid: i32,
        task_id: Option<String>,
        command: Vec<String>,
        working_dir: Option<String>,
    ) -> io::Result<ManagedProcessGuard> {
        fs::create_dir_all(&self.registry_dir)?;
        let now = now_unix_ms();
        let entry = SpawnRegistryEntry {
            version: REGISTRY_VERSION,
            id: format!("spawn-{pid}-{now}"),
            owner_pid: std::process::id(),
            owner_exe: current_exe_name(),
            pid,
            pgid,
            task_id,
            started_at_unix_ms: now,
            last_heartbeat_unix_ms: Some(now),
            command,
            working_dir,
        };
        let path = self.entry_path(&entry);
        let temporary = path.with_extension("json.tmp");
        fs::write(&temporary, serde_json::to_vec_pretty(&entry)?)?;
        fs::rename(temporary, path)?;
        Ok(ManagedProcessGuard {
            supervisor: self.clone(),
            entry,
            active: true,
        })
    }

    fn verified_alive(&self, entry: &SpawnRegistryEntry) -> bool {
        entry.version == REGISTRY_VERSION && group_alive(entry.pgid) && leader_still_matches(entry)
    }

    fn remove_entry(&self, entry: &SpawnRegistryEntry) -> io::Result<()> {
        match fs::remove_file(self.entry_path(entry)) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
        }
    }

    fn entry_path(&self, entry: &SpawnRegistryEntry) -> PathBuf {
        self.registry_dir.join(format!("{}.json", entry.id))
    }
}

#[derive(Debug)]
struct ManagedProcessGuard {
    supervisor: ProcessSupervisor,
    entry: SpawnRegistryEntry,
    active: bool,
}

impl ManagedProcessGuard {
    fn cleanup(&mut self) -> io::Result<()> {
        if !self.active {
            return Ok(());
        }
        self.supervisor.terminate_group(self.entry.pgid)?;
        self.supervisor.remove_entry(&self.entry)?;
        self.active = false;
        Ok(())
    }
}

impl Drop for ManagedProcessGuard {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

async fn run_process(
    config: &ProcessAgentConfig,
    context: &SessionContext,
    prompt: &str,
    event_sender: Option<mpsc::UnboundedSender<WorkerEvent>>,
) -> AgentResult<String> {
    let supervisor = config.supervisor();
    supervisor
        .ensure_spawn_capacity()
        .map_err(|error| AgentError::Unavailable(error.to_string()))?;

    let invocation = build_invocation(config, context, prompt);
    let mut command = Command::new(&invocation.command);
    configure_process_group(&mut command);
    for arg in &invocation.args {
        command.arg(arg);
    }
    if let Some(working_dir) = &context.working_dir {
        command.current_dir(working_dir);
    }
    command.kill_on_drop(true);
    command.stdin(std::process::Stdio::piped());
    command.stdout(std::process::Stdio::piped());
    command.stderr(std::process::Stdio::piped());
    let mut child = command
        .spawn()
        .map_err(|error| AgentError::Unavailable(error.to_string()))?;
    let pid = child
        .id()
        .ok_or_else(|| AgentError::Session("spawned process had no pid".to_string()))?;
    let pgid = process_group_id(pid);
    let command_line = invocation.command_line();
    let working_dir = context
        .working_dir
        .as_ref()
        .map(|path| path.display().to_string());
    let mut guard = supervisor
        .register_spawn(
            pid,
            pgid,
            task_id_from_prompt(prompt),
            command_line,
            working_dir,
        )
        .map_err(|error| AgentError::Session(error.to_string()))?;

    if invocation.prompt_on_stdin && !prompt.is_empty() {
        if let Some(mut stdin) = child.stdin.take() {
            // A failing one-shot worker can exit before reading its prompt, closing the
            // stdin read end; the resulting broken pipe is benign and OS-timing dependent
            // (it races worker startup vs. this write). Tolerate it so we still capture and
            // redact the worker's stderr/exit status instead of aborting with a bare pipe error.
            match stdin.write_all(prompt.as_bytes()).await {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::BrokenPipe => {}
                Err(error) => return Err(AgentError::Session(error.to_string())),
            }
        }
    }
    drop(child.stdin.take());

    let stdout = child
        .stdout
        .take()
        .map(|pipe| read_stdout(pipe, invocation.event_format, event_sender));
    let stderr = child.stderr.take().map(read_pipe);
    let deadline = config.timeout.min(config.max_lifetime);
    let status = match time::timeout(deadline, child.wait()).await {
        Ok(Ok(status)) => status,
        Ok(Err(error)) => {
            guard
                .cleanup()
                .map_err(|cleanup| AgentError::Session(cleanup.to_string()))?;
            return Err(AgentError::Session(error.to_string()));
        }
        Err(_) => {
            guard
                .cleanup()
                .map_err(|error| AgentError::Session(error.to_string()))?;
            let _ = time::timeout(config.termination_grace, child.wait()).await;
            return Err(AgentError::Session(format!(
                "process timed out after {}s",
                deadline.as_secs()
            )));
        }
    };

    guard
        .cleanup()
        .map_err(|error| AgentError::Session(error.to_string()))?;
    let stdout = collect_pipe(stdout, config.termination_grace).await?;
    let stderr = collect_pipe(stderr, config.termination_grace).await?;
    let mut text = String::from_utf8_lossy(&stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&stderr));
    if !status.success() {
        let text = redact_and_truncate(&text, MAX_PROCESS_ERROR_OUTPUT_CHARS);
        return Err(AgentError::Session(format!(
            "process exited with status {status}: {text}"
        )));
    }
    Ok(text)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AgentKind {
    Claude,
    Codex,
    Gemini,
    Opencode,
    Generic,
}

impl AgentKind {
    fn from_config(config: &ProcessAgentConfig) -> Self {
        let agent = normalize_agent_id(logical_agent_id(config));
        let command = normalize_agent_id(command_basename(&config.command));
        match (agent.as_str(), command.as_str()) {
            ("claude" | "claude-code", "claude" | "claude-code") => Self::Claude,
            ("codex", "codex") => Self::Codex,
            ("gemini", "gemini") => Self::Gemini,
            ("opencode", "opencode") => Self::Opencode,
            _ => Self::Generic,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EventFormat {
    Jsonl,
    Text,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ProcessInvocation {
    command: String,
    args: Vec<String>,
    prompt_on_stdin: bool,
    event_format: EventFormat,
}

impl ProcessInvocation {
    fn command_line(&self) -> Vec<String> {
        std::iter::once(self.command.clone())
            .chain(self.args.iter().cloned())
            .map(|part| redact_secrets(&part))
            .collect()
    }
}

fn build_invocation(
    config: &ProcessAgentConfig,
    context: &SessionContext,
    prompt: &str,
) -> ProcessInvocation {
    match AgentKind::from_config(config) {
        AgentKind::Claude => native_invocation(
            config,
            vec![
                "-p".to_string(),
                prompt.to_string(),
                "--output-format".to_string(),
                "stream-json".to_string(),
                "--verbose".to_string(),
                "--permission-mode".to_string(),
                "acceptEdits".to_string(),
            ],
            context
                .model
                .as_ref()
                .map(|model| vec!["--model".to_string(), model.clone()])
                .unwrap_or_default(),
            EventFormat::Jsonl,
        ),
        AgentKind::Codex => {
            let workdir = context
                .working_dir
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| ".".to_string());
            let mut args = vec![
                "exec".to_string(),
                "--json".to_string(),
                "--dangerously-bypass-approvals-and-sandbox".to_string(),
                "-C".to_string(),
                workdir,
            ];
            if let Some(effort) = codex_reasoning_effort(&config.args) {
                args.extend(["-c".to_string(), format!("model_reasoning_effort={effort}")]);
            }
            if let Some(model) = &context.model {
                args.extend(["-m".to_string(), model.clone()]);
            }
            args.push(prompt.to_string());
            native_invocation(config, args, Vec::new(), EventFormat::Jsonl)
        }
        AgentKind::Gemini => native_invocation(
            config,
            vec!["-p".to_string(), prompt.to_string(), "--yolo".to_string()],
            context
                .model
                .as_ref()
                .map(|model| vec!["-m".to_string(), model.clone()])
                .unwrap_or_default(),
            EventFormat::Text,
        ),
        AgentKind::Opencode => native_invocation(
            config,
            vec!["run".to_string(), prompt.to_string()],
            context
                .model
                .as_ref()
                .map(|model| vec!["--model".to_string(), model.clone()])
                .unwrap_or_default(),
            EventFormat::Text,
        ),
        AgentKind::Generic => generic_invocation(config, context, prompt),
    }
}

fn native_invocation(
    config: &ProcessAgentConfig,
    mut args: Vec<String>,
    extra_args: Vec<String>,
    event_format: EventFormat,
) -> ProcessInvocation {
    args.extend(extra_args);
    ProcessInvocation {
        command: config.command.clone(),
        args,
        prompt_on_stdin: false,
        event_format,
    }
}

fn generic_invocation(
    config: &ProcessAgentConfig,
    context: &SessionContext,
    prompt: &str,
) -> ProcessInvocation {
    let mut prompt_was_arg = false;
    let args = config
        .args
        .iter()
        .map(|arg| {
            prompt_was_arg |= arg.contains("{prompt}");
            render_arg(arg, context, prompt)
        })
        .collect();
    ProcessInvocation {
        command: config.command.clone(),
        args,
        prompt_on_stdin: !prompt_was_arg,
        event_format: EventFormat::Text,
    }
}

fn command_basename(command: &str) -> &str {
    Path::new(command)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(command)
}

fn codex_reasoning_effort(args: &[String]) -> Option<String> {
    args.iter()
        .filter_map(|arg| {
            arg.strip_prefix("model_reasoning_effort=")
                .or_else(|| arg.strip_prefix("reasoning_effort="))
                .or_else(|| arg.strip_prefix("reasoning="))
        })
        .chain(
            args.windows(2)
                .filter(|pair| {
                    matches!(
                        pair.first().map(String::as_str),
                        Some("-c" | "--config" | "--model-reasoning-effort" | "--reasoning-effort")
                    )
                })
                .map(|pair| {
                    pair[1]
                        .strip_prefix("model_reasoning_effort=")
                        .unwrap_or(pair[1].as_str())
                }),
        )
        .map(str::trim)
        .find(|value| !value.is_empty())
        .map(str::to_string)
}

fn render_arg(template: &str, context: &SessionContext, prompt: &str) -> String {
    template
        .replace("{prompt}", prompt)
        .replace("{role}", context.role.canonical_alias())
        .replace("{alias}", context.role.canonical_alias())
        .replace("{agent}", &context.agent)
        .replace("{model}", context.model.as_deref().unwrap_or_default())
}

fn sanitize_agent_id(agent: &str) -> String {
    let mut output = String::new();
    for character in agent.chars() {
        if character.is_ascii_alphanumeric() {
            output.push(character.to_ascii_lowercase());
        } else {
            output.push('-');
        }
    }
    let output = output.trim_matches('-');
    if output.is_empty() {
        "agent".to_string()
    } else {
        output.to_string()
    }
}

fn default_registry_dir() -> PathBuf {
    std::env::var_os("ARTESIAN_SPAWN_REGISTRY")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".artesian").join("spawns"))
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

/// Saturating wall-clock age between two unix-millisecond stamps, clamped to the
/// `Duration` millisecond range so a clock skew can never panic the reaper.
fn age_since(now_ms: u128, then_ms: u128) -> Duration {
    Duration::from_millis(now_ms.saturating_sub(then_ms).min(u64::MAX as u128) as u64)
}

#[cfg(test)]
fn command_line(config: &ProcessAgentConfig) -> Vec<String> {
    std::iter::once(config.command.clone())
        .chain(config.args.iter().cloned())
        .map(|part| redact_secrets(&part))
        .collect()
}

fn current_exe_name() -> Option<String> {
    std::env::current_exe()
        .ok()
        .and_then(|path| path.file_name().map(|name| name.to_owned()))
        .and_then(|name| name.to_str().map(str::to_string))
}

fn task_id_from_prompt(prompt: &str) -> Option<String> {
    prompt
        .lines()
        .find_map(|line| line.strip_prefix("Task ID: "))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn redact_and_truncate(input: &str, limit: usize) -> String {
    let redacted = redact_secrets(input);
    if redacted.chars().count() <= limit {
        return redacted;
    }
    let mut output = redacted.chars().take(limit).collect::<String>();
    output.push_str("…[truncated]");
    output
}

fn redact_secrets(input: &str) -> String {
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

fn worker_event_log_line(event: &WorkerEvent) -> String {
    if event.text.trim().is_empty() {
        event.raw.clone()
    } else {
        event.text.clone()
    }
}

fn parse_stdout_event(format: EventFormat, line: &str) -> WorkerEvent {
    let raw = trim_line_end(line);
    match format {
        EventFormat::Jsonl => parse_jsonl_event(raw).unwrap_or_else(|| raw_stdout_event(raw)),
        EventFormat::Text => raw_stdout_event(raw),
    }
}

fn parse_jsonl_event(line: &str) -> Option<WorkerEvent> {
    let value = serde_json::from_str::<serde_json::Value>(line).ok()?;
    let kind = value
        .get("type")
        .or_else(|| value.get("event"))
        .or_else(|| value.get("kind"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("json")
        .to_string();
    let text = extract_json_text(&value).unwrap_or_else(|| kind.clone());
    Some(WorkerEvent {
        kind: redact_secrets(&kind),
        text: redact_secrets(&text),
        raw: redact_secrets(line),
    })
}

fn raw_stdout_event(line: &str) -> WorkerEvent {
    let text = redact_secrets(line);
    WorkerEvent {
        kind: "stdout".to_string(),
        text: text.clone(),
        raw: text,
    }
}

fn extract_json_text(value: &serde_json::Value) -> Option<String> {
    let mut parts = Vec::new();
    collect_json_text(value, &mut parts);
    (!parts.is_empty()).then(|| parts.join("\n"))
}

fn collect_json_text(value: &serde_json::Value, parts: &mut Vec<String>) {
    match value {
        serde_json::Value::String(text) => {
            let text = text.trim();
            if !text.is_empty() {
                parts.push(text.to_string());
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                collect_json_text(item, parts);
            }
        }
        serde_json::Value::Object(object) => {
            for key in [
                "text", "content", "message", "delta", "summary", "result", "output",
            ] {
                if let Some(value) = object.get(key) {
                    collect_json_text(value, parts);
                }
            }
        }
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {}
    }
}

fn trim_line_end(line: &str) -> &str {
    line.trim_end_matches(['\r', '\n'])
}

fn read_stdout<T>(
    pipe: T,
    event_format: EventFormat,
    event_sender: Option<mpsc::UnboundedSender<WorkerEvent>>,
) -> JoinHandle<io::Result<Vec<u8>>>
where
    T: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let mut reader = BufReader::new(pipe);
        let mut bytes = Vec::new();
        loop {
            let mut line = Vec::new();
            let count = reader.read_until(b'\n', &mut line).await?;
            if count == 0 {
                break;
            }
            bytes.extend_from_slice(&line);
            if let Some(sender) = &event_sender {
                let line = String::from_utf8_lossy(&line);
                let event = parse_stdout_event(event_format, &line);
                if !event.text.is_empty() {
                    let _ = sender.send(event);
                }
            }
        }
        Ok(bytes)
    })
}

fn read_pipe<T>(mut pipe: T) -> JoinHandle<io::Result<Vec<u8>>>
where
    T: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let mut bytes = Vec::new();
        pipe.read_to_end(&mut bytes).await?;
        Ok(bytes)
    })
}

async fn collect_pipe(
    handle: Option<JoinHandle<io::Result<Vec<u8>>>>,
    grace: Duration,
) -> AgentResult<Vec<u8>> {
    let Some(handle) = handle else {
        return Ok(Vec::new());
    };
    match time::timeout(grace.max(Duration::from_millis(1)), handle).await {
        Ok(Ok(Ok(bytes))) => Ok(bytes),
        Ok(Ok(Err(error))) => Err(AgentError::Session(error.to_string())),
        Ok(Err(error)) => Err(AgentError::Session(error.to_string())),
        Err(_) => Err(AgentError::Session(
            "process output pipe did not close after termination".to_string(),
        )),
    }
}

#[cfg(unix)]
fn configure_process_group(command: &mut Command) {
    command.process_group(0);
}

#[cfg(not(unix))]
fn configure_process_group(_command: &mut Command) {}

#[cfg(unix)]
fn process_group_id(pid: u32) -> i32 {
    pid as i32
}

#[cfg(not(unix))]
fn process_group_id(pid: u32) -> i32 {
    pid as i32
}

#[cfg(unix)]
fn terminate_group(pgid: i32, grace: Duration) -> io::Result<()> {
    if !send_group_signal(pgid, Signal::SIGTERM)? {
        return Ok(());
    }
    std::thread::sleep(grace);
    if group_alive(pgid) {
        let _ = send_group_signal(pgid, Signal::SIGKILL)?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn terminate_group(_pgid: i32, _grace: Duration) -> io::Result<()> {
    Ok(())
}

#[cfg(unix)]
fn send_group_signal(pgid: i32, signal: Signal) -> io::Result<bool> {
    match killpg(Pid::from_raw(pgid), signal) {
        Ok(()) => Ok(true),
        Err(Errno::ESRCH) => Ok(false),
        Err(Errno::EPERM) if signal == Signal::SIGKILL => Ok(false),
        Err(error) => Err(io::Error::other(format!(
            "signal {signal:?} to pgid {pgid}: {error}"
        ))),
    }
}

#[cfg(unix)]
fn group_alive(pgid: i32) -> bool {
    match killpg(Pid::from_raw(pgid), None) {
        Ok(()) => true,
        Err(Errno::EPERM) => true,
        Err(_) => false,
    }
}

#[cfg(not(unix))]
fn group_alive(_pgid: i32) -> bool {
    false
}

#[cfg(unix)]
fn owner_alive(entry: &SpawnRegistryEntry) -> bool {
    match kill(Pid::from_raw(entry.owner_pid as i32), None) {
        Ok(()) => owner_command_matches(entry),
        Err(Errno::EPERM) => true,
        Err(_) => false,
    }
}

#[cfg(not(unix))]
fn owner_alive(_entry: &SpawnRegistryEntry) -> bool {
    false
}

#[cfg(unix)]
fn leader_still_matches(entry: &SpawnRegistryEntry) -> bool {
    let Ok(actual_pgid) = getpgid(Some(Pid::from_raw(entry.pid as i32))) else {
        return false;
    };
    if actual_pgid.as_raw() != entry.pgid {
        return false;
    }
    let Some(program) = entry.command.first().and_then(|command| {
        Path::new(command)
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| !name.is_empty())
    }) else {
        return false;
    };
    process_command_contains(entry.pid, program)
}

#[cfg(unix)]
fn owner_command_matches(entry: &SpawnRegistryEntry) -> bool {
    match entry.owner_exe.as_deref() {
        Some(owner_exe) => process_command_contains(entry.owner_pid, owner_exe),
        None => true,
    }
}

#[cfg(unix)]
fn process_command_contains(pid: u32, needle: &str) -> bool {
    let output = std::process::Command::new("ps")
        .args(["-o", "command=", "-p", &pid.to_string()])
        .output();
    let Ok(output) = output else {
        return false;
    };
    if !output.status.success() {
        return false;
    }
    String::from_utf8_lossy(&output.stdout).contains(needle)
}

#[cfg(not(unix))]
fn leader_still_matches(_entry: &SpawnRegistryEntry) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use artesian_core::{AgentMessage, Role, SpawnRequest};
    use artesian_test_support::TempDir;
    #[cfg(unix)]
    use std::os::unix::process::CommandExt;

    use super::*;

    #[tokio::test]
    async fn launches_real_echo_subprocess() {
        let tempdir = TempDir::new("process-agent-echo");
        let agent = ProcessAgent::new(
            ProcessAgentConfig::new("echo")
                .with_args(vec!["artesian".into()])
                .with_registry_dir(tempdir.join("spawns"))
                .with_termination_grace(Duration::from_millis(10)),
        );
        let session = agent
            .spawn(SpawnRequest {
                role: Role::Worker,
                agent: "echo".to_string(),
                model: None,
                working_dir: None,
            })
            .await
            .expect("spawn should register session");
        let response = agent
            .send(
                &session,
                AgentMessage {
                    content: String::new(),
                },
            )
            .await
            .expect("echo should launch");

        assert_eq!(response.content.trim(), "artesian");
    }

    #[tokio::test]
    async fn role_bindings_render_distinct_models_for_same_cli() {
        let tempdir = TempDir::new("process-agent-model-render");
        let agent = ProcessAgent::new(
            ProcessAgentConfig::new("echo")
                .with_agent_id("claude")
                .with_args(vec!["{role}:{model}".into()])
                .with_static_models(vec!["claude-opus".into(), "claude-sonnet".into()])
                .with_registry_dir(tempdir.join("spawns"))
                .with_termination_grace(Duration::from_millis(10)),
        );

        let master = agent
            .spawn(SpawnRequest {
                role: Role::Master,
                agent: "claude".to_string(),
                model: Some("claude-opus".to_string()),
                working_dir: None,
            })
            .await
            .expect("master model should be available");
        let worker = agent
            .spawn(SpawnRequest {
                role: Role::Worker,
                agent: "claude".to_string(),
                model: Some("claude-sonnet".to_string()),
                working_dir: None,
            })
            .await
            .expect("worker model should be available");

        let master_response = agent
            .send(
                &master,
                AgentMessage {
                    content: String::new(),
                },
            )
            .await
            .expect("master command should run");
        let worker_response = agent
            .send(
                &worker,
                AgentMessage {
                    content: String::new(),
                },
            )
            .await
            .expect("worker command should run");

        assert_eq!(master_response.content.trim(), "master:claude-opus");
        assert_eq!(worker_response.content.trim(), "worker:claude-sonnet");
    }

    #[tokio::test]
    async fn role_binding_default_model_renders_when_spawn_omits_model() {
        let tempdir = TempDir::new("process-agent-default-model");
        let master = ProcessAgent::new(
            ProcessAgentConfig::new("echo")
                .with_agent_id("claude")
                .with_args(vec!["{role}:{model}".into()])
                .with_static_models(vec!["claude-opus".into(), "claude-sonnet".into()])
                .with_default_model(Some("claude-opus".to_string()))
                .with_registry_dir(tempdir.join("master-spawns"))
                .with_termination_grace(Duration::from_millis(10)),
        );
        let worker = ProcessAgent::new(
            ProcessAgentConfig::new("echo")
                .with_agent_id("claude")
                .with_args(vec!["{role}:{model}".into()])
                .with_static_models(vec!["claude-opus".into(), "claude-sonnet".into()])
                .with_default_model(Some("claude-sonnet".to_string()))
                .with_registry_dir(tempdir.join("worker-spawns"))
                .with_termination_grace(Duration::from_millis(10)),
        );

        let master_session = master
            .spawn(SpawnRequest {
                role: Role::Master,
                agent: "claude".to_string(),
                model: None,
                working_dir: None,
            })
            .await
            .expect("master default model should be available");
        let worker_session = worker
            .spawn(SpawnRequest {
                role: Role::Worker,
                agent: "claude".to_string(),
                model: None,
                working_dir: None,
            })
            .await
            .expect("worker default model should be available");

        let master_response = master
            .send(
                &master_session,
                AgentMessage {
                    content: String::new(),
                },
            )
            .await
            .expect("master command should run");
        let worker_response = worker
            .send(
                &worker_session,
                AgentMessage {
                    content: String::new(),
                },
            )
            .await
            .expect("worker command should run");

        assert_eq!(master_response.content.trim(), "master:claude-opus");
        assert_eq!(worker_response.content.trim(), "worker:claude-sonnet");
    }

    #[test]
    fn native_agent_invocations_use_streaming_cli_args() {
        let prompt = "Do bounded work.";
        let claude_context = test_context("claude", Some("claude-sonnet"), None);
        let claude = build_invocation(
            &ProcessAgentConfig::new("claude").with_agent_id("claude"),
            &claude_context,
            prompt,
        );
        assert_eq!(
            claude.args,
            vec![
                "-p",
                prompt,
                "--output-format",
                "stream-json",
                "--verbose",
                "--permission-mode",
                "acceptEdits",
                "--model",
                "claude-sonnet",
            ]
        );
        assert_eq!(claude.event_format, EventFormat::Jsonl);
        assert!(!claude.prompt_on_stdin);

        let codex_context = test_context("codex", Some("gpt-5"), Some("/repo"));
        let codex = build_invocation(
            &ProcessAgentConfig::new("codex")
                .with_agent_id("codex")
                .with_args(vec!["model_reasoning_effort=xhigh".to_string()]),
            &codex_context,
            prompt,
        );
        assert_eq!(
            codex.args,
            vec![
                "exec",
                "--json",
                "--dangerously-bypass-approvals-and-sandbox",
                "-C",
                "/repo",
                "-c",
                "model_reasoning_effort=xhigh",
                "-m",
                "gpt-5",
                prompt,
            ]
        );
        assert_eq!(codex.event_format, EventFormat::Jsonl);
        assert!(!codex.prompt_on_stdin);

        let gemini_context = test_context("gemini", Some("gemini-pro"), None);
        let gemini = build_invocation(
            &ProcessAgentConfig::new("gemini").with_agent_id("gemini"),
            &gemini_context,
            prompt,
        );
        assert_eq!(
            gemini.args,
            vec!["-p", prompt, "--yolo", "-m", "gemini-pro"]
        );
        assert_eq!(gemini.event_format, EventFormat::Text);
        assert!(!gemini.prompt_on_stdin);

        let opencode_context = test_context("opencode", Some("opencode-default"), None);
        let opencode = build_invocation(
            &ProcessAgentConfig::new("opencode").with_agent_id("opencode"),
            &opencode_context,
            prompt,
        );
        assert_eq!(
            opencode.args,
            vec!["run", prompt, "--model", "opencode-default"]
        );
        assert_eq!(opencode.event_format, EventFormat::Text);
        assert!(!opencode.prompt_on_stdin);
    }

    #[test]
    fn jsonl_stdout_lines_normalize_worker_events() {
        let event = parse_stdout_event(
            EventFormat::Jsonl,
            r#"{"type":"assistant","message":{"content":[{"type":"text","text":"hello sk-secret-value"}]}}"#,
        );
        assert_eq!(event.kind, "assistant");
        assert_eq!(event.text, "hello [REDACTED]");
        assert!(event.raw.contains("[REDACTED]"));
        assert!(!event.raw.contains("sk-secret-value"));

        let codex_event = parse_stdout_event(
            EventFormat::Jsonl,
            r#"{"type":"agent_message","message":"implementation complete"}"#,
        );
        assert_eq!(codex_event.kind, "agent_message");
        assert_eq!(codex_event.text, "implementation complete");

        let raw_event = parse_stdout_event(EventFormat::Jsonl, "plain progress\n");
        assert_eq!(
            raw_event,
            WorkerEvent {
                kind: "stdout".to_string(),
                text: "plain progress".to_string(),
                raw: "plain progress".to_string(),
            }
        );
    }

    #[tokio::test]
    async fn unavailable_model_fails_before_process_spawn() {
        let tempdir = TempDir::new("process-agent-model-missing");
        let agent = ProcessAgent::new(
            ProcessAgentConfig::new("echo")
                .with_agent_id("codex")
                .with_static_models(vec!["gpt-5".into()])
                .with_registry_dir(tempdir.join("spawns")),
        );

        let error = agent
            .spawn(SpawnRequest {
                role: Role::Worker,
                agent: "codex".to_string(),
                model: Some("missing-model".to_string()),
                working_dir: None,
            })
            .await
            .expect_err("unavailable model should fail");

        assert!(error.to_string().contains("missing-model"));
        assert!(
            ProcessSupervisor::new(tempdir.join("spawns"))
                .entries()
                .expect("registry should read")
                .is_empty(),
            "failed validation must not spawn a process"
        );
    }

    #[tokio::test]
    async fn failing_process_error_redacts_and_limits_secret_output() {
        let tempdir = TempDir::new("process-agent-secret-error");
        let secret = "sk-artesian-secret-1234567890";
        let script = format!(
            "printf 'token=artesian-token-value\\n{secret}\\n'; printf '%4096s\\n' x; exit 7"
        );
        let agent = ProcessAgent::new(
            ProcessAgentConfig::new("sh")
                .with_args(vec!["-c".to_string(), script])
                .with_registry_dir(tempdir.join("spawns"))
                .with_termination_grace(Duration::from_millis(10)),
        );
        let session = agent
            .spawn(SpawnRequest {
                role: Role::Worker,
                agent: "sh".to_string(),
                model: None,
                working_dir: None,
            })
            .await
            .expect("spawn should register session");

        let error = agent
            .send(
                &session,
                AgentMessage {
                    content: String::new(),
                },
            )
            .await
            .expect_err("process should fail");
        let text = error.to_string();

        assert!(!text.contains(secret));
        assert!(!text.contains("artesian-token-value"));
        assert!(text.contains("[REDACTED]"));
        assert!(text.len() < MAX_PROCESS_ERROR_OUTPUT_CHARS + 256);
    }

    #[tokio::test]
    async fn discovery_output_and_registry_command_are_redacted() {
        let tempdir = TempDir::new("process-agent-secret-discovery");
        let env_name = "ARTESIAN_SECRET_PROBE_MODELS_CMD";
        std::env::set_var(
            env_name,
            "printf 'sk-discovery-secret-123456\\nsafe-model\\n'",
        );
        let binding = AgentBinding {
            role: Role::Worker,
            agent: "secret-probe".to_string(),
            model: None,
            command: Some("sh".to_string()),
            args: vec!["--api-key=registry-secret-value".to_string()],
            timeout_seconds: None,
        };
        let cache_path = tempdir.join("agents.json");
        let catalog = refresh_agent_catalog(&[binding], &cache_path)
            .await
            .expect("catalog refresh should succeed");
        std::env::remove_var(env_name);
        let json = serde_json::to_string(&catalog).expect("catalog should encode");

        assert!(!json.contains("sk-discovery-secret-123456"));
        assert!(json.contains("[REDACTED]"));
        assert!(json.contains("safe-model"));

        let config = ProcessAgentConfig::new("sh")
            .with_args(vec!["--api-key=registry-secret-value".to_string()]);
        let command = command_line(&config).join(" ");
        assert!(!command.contains("registry-secret-value"));
        assert!(command.contains("[REDACTED]"));

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            let mode = std::fs::metadata(cache_path)
                .expect("cache metadata should read")
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(mode, 0o600);
        }
    }

    #[test]
    #[cfg(unix)]
    fn nonexistent_group_termination_is_noop() {
        let tempdir = TempDir::new("process-agent-noop");
        let supervisor = ProcessSupervisor::new(tempdir.join("spawns"))
            .with_termination_grace(Duration::from_millis(10));

        supervisor
            .terminate_group(999_999)
            .expect("missing group should be a no-op");
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn timeout_kills_process_group_children() {
        let tempdir = TempDir::new("process-agent-timeout");
        let child_pid_file = tempdir.join("child.pid");
        let parent_pid_file = tempdir.join("parent.pid");
        let agent = shell_agent(
            tempdir.path(),
            "echo $$ > \"$1\"; sleep 30 & echo $! > \"$2\"; wait",
            &parent_pid_file,
            &child_pid_file,
            Duration::from_millis(200),
        );
        let session = spawn_session(&agent).await;

        let error = agent
            .send(
                &session,
                AgentMessage {
                    content: String::new(),
                },
            )
            .await
            .expect_err("long process should time out");

        assert!(
            error.to_string().contains("timed out"),
            "unexpected error: {error}"
        );
        let parent_pid = read_pid(&parent_pid_file);
        let child_pid = read_pid(&child_pid_file);
        assert_pid_gone(parent_pid);
        assert_pid_gone(child_pid);
        assert!(
            ProcessSupervisor::new(tempdir.join("spawns"))
                .entries()
                .expect("registry should read")
                .is_empty(),
            "timeout cleanup should remove registry entry"
        );
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn cancellation_drops_guard_and_kills_process_group_children() {
        let tempdir = TempDir::new("process-agent-cancel");
        let child_pid_file = tempdir.join("child.pid");
        let parent_pid_file = tempdir.join("parent.pid");
        let agent = Arc::new(shell_agent(
            tempdir.path(),
            "echo $$ > \"$1\"; sleep 30 & echo $! > \"$2\"; wait",
            &parent_pid_file,
            &child_pid_file,
            Duration::from_secs(30),
        ));
        let session = spawn_session(&agent).await;
        let task = tokio::spawn({
            let agent = agent.clone();
            async move {
                agent
                    .send(
                        &session,
                        AgentMessage {
                            content: String::new(),
                        },
                    )
                    .await
            }
        });
        wait_for_file(&child_pid_file).await;

        task.abort();
        let _ = task.await;

        let parent_pid = read_pid(&parent_pid_file);
        let child_pid = read_pid(&child_pid_file);
        assert_pid_gone(parent_pid);
        assert_pid_gone(child_pid);
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn spawn_cap_refuses_extra_live_processes() {
        let tempdir = TempDir::new("process-agent-cap");
        let first_child_pid_file = tempdir.join("first-child.pid");
        let first_parent_pid_file = tempdir.join("first-parent.pid");
        let agent = Arc::new(shell_agent_with_cap(
            tempdir.path(),
            "echo $$ > \"$1\"; sleep 30 & echo $! > \"$2\"; wait",
            &first_parent_pid_file,
            &first_child_pid_file,
            Duration::from_secs(30),
            1,
        ));
        let first_session = spawn_session(&agent).await;
        let first = tokio::spawn({
            let agent = agent.clone();
            async move {
                agent
                    .send(
                        &first_session,
                        AgentMessage {
                            content: String::new(),
                        },
                    )
                    .await
            }
        });
        wait_for_file(&first_child_pid_file).await;
        wait_for_registry_count(&tempdir.join("spawns"), 1).await;

        let second_session = spawn_session(&agent).await;
        let error = agent
            .send(
                &second_session,
                AgentMessage {
                    content: String::new(),
                },
            )
            .await
            .expect_err("second live process should exceed cap");

        assert!(
            error.to_string().contains("spawn cap reached"),
            "unexpected error: {error}"
        );
        first.abort();
        let _ = first.await;
        assert_pid_gone(read_pid(&first_parent_pid_file));
        assert_pid_gone(read_pid(&first_child_pid_file));
    }

    #[test]
    #[cfg(unix)]
    fn startup_reaper_kills_orphaned_registry_entry() {
        let tempdir = TempDir::new("process-agent-reaper");
        let registry = tempdir.join("spawns");
        fs::create_dir_all(&registry).expect("registry should be created");
        let child_pid_file = tempdir.join("child.pid");
        let mut command = std::process::Command::new("sh");
        command.process_group(0);
        command.arg("-c").arg(format!(
            "sleep 30 & echo $! > {}; wait",
            child_pid_file.display()
        ));
        let mut child = command.spawn().expect("test process should spawn");
        let pid = child.id();
        wait_for_file_sync(&child_pid_file);
        let entry = SpawnRegistryEntry {
            version: REGISTRY_VERSION,
            id: format!("spawn-{pid}-test"),
            owner_pid: 999_999,
            owner_exe: Some("artesian-dead-owner".to_string()),
            pid,
            pgid: pid as i32,
            task_id: Some("task-restart".to_string()),
            started_at_unix_ms: now_unix_ms(),
            last_heartbeat_unix_ms: Some(now_unix_ms()),
            command: vec!["sh".to_string()],
            working_dir: None,
        };
        fs::write(
            registry.join(format!("{}.json", entry.id)),
            serde_json::to_vec_pretty(&entry).expect("entry should serialize"),
        )
        .expect("registry entry should write");

        let report = ProcessSupervisor::new(&registry)
            .with_termination_grace(Duration::from_millis(50))
            .reap_stale()
            .expect("startup reaper should run");
        let grandchild_pid = read_pid(&child_pid_file);
        let _ = child.wait();

        assert_eq!(report.terminated, 1);
        assert_pid_gone(pid);
        assert_pid_gone(grandchild_pid);
        assert!(
            ProcessSupervisor::new(&registry)
                .entries()
                .expect("registry should read")
                .is_empty(),
            "reaper should remove stale registry entry"
        );
    }

    #[cfg(unix)]
    fn spawn_tracked_group(pid_file: &Path) -> std::process::Child {
        let mut command = std::process::Command::new("sh");
        command.process_group(0);
        command
            .arg("-c")
            .arg(format!("sleep 30 & echo $! > {}; wait", pid_file.display()));
        let child = command.spawn().expect("test process should spawn");
        wait_for_file_sync(pid_file);
        child
    }

    #[test]
    #[cfg(unix)]
    fn gc_reaps_ttl_expired_spawn() {
        let tempdir = TempDir::new("process-agent-gc-ttl");
        let registry = tempdir.join("spawns");
        fs::create_dir_all(&registry).expect("registry should be created");
        let child_pid_file = tempdir.join("child.pid");
        let mut child = spawn_tracked_group(&child_pid_file);
        let pid = child.id();
        // Owner is the live current process (NOT an orphan), but the spawn is old.
        let entry = SpawnRegistryEntry {
            version: REGISTRY_VERSION,
            id: format!("spawn-{pid}-ttl"),
            owner_pid: std::process::id(),
            owner_exe: current_exe_name(),
            pid,
            pgid: pid as i32,
            task_id: None,
            started_at_unix_ms: now_unix_ms().saturating_sub(10_000),
            last_heartbeat_unix_ms: Some(now_unix_ms()),
            command: vec!["sh".to_string()],
            working_dir: None,
        };
        fs::write(
            registry.join(format!("{}.json", entry.id)),
            serde_json::to_vec_pretty(&entry).expect("entry should serialize"),
        )
        .expect("registry entry should write");

        let report = ProcessSupervisor::new(&registry)
            .with_termination_grace(Duration::from_millis(50))
            .gc(GcOptions::default().with_ttl(Duration::from_millis(1)))
            .expect("gc should run");
        let grandchild_pid = read_pid(&child_pid_file);
        let _ = child.wait();

        assert_eq!(
            report.terminated, 1,
            "ttl-expired spawn should be terminated"
        );
        assert_eq!(report.expired, 1, "ttl reap should count as expired");
        assert_pid_gone(pid);
        assert_pid_gone(grandchild_pid);
        assert!(
            ProcessSupervisor::new(&registry)
                .entries()
                .expect("registry should read")
                .is_empty(),
            "gc should remove the reaped entry"
        );
    }

    #[test]
    #[cfg(unix)]
    fn gc_reaps_heartbeat_stale_spawn() {
        let tempdir = TempDir::new("process-agent-gc-heartbeat");
        let registry = tempdir.join("spawns");
        fs::create_dir_all(&registry).expect("registry should be created");
        let child_pid_file = tempdir.join("child.pid");
        let mut child = spawn_tracked_group(&child_pid_file);
        let pid = child.id();
        // Fresh age, live owner, but the heartbeat went stale (hung worker).
        let entry = SpawnRegistryEntry {
            version: REGISTRY_VERSION,
            id: format!("spawn-{pid}-heartbeat"),
            owner_pid: std::process::id(),
            owner_exe: current_exe_name(),
            pid,
            pgid: pid as i32,
            task_id: None,
            started_at_unix_ms: now_unix_ms(),
            last_heartbeat_unix_ms: Some(now_unix_ms().saturating_sub(10_000)),
            command: vec!["sh".to_string()],
            working_dir: None,
        };
        fs::write(
            registry.join(format!("{}.json", entry.id)),
            serde_json::to_vec_pretty(&entry).expect("entry should serialize"),
        )
        .expect("registry entry should write");

        let report = ProcessSupervisor::new(&registry)
            .with_termination_grace(Duration::from_millis(50))
            .gc(GcOptions::default().with_heartbeat_timeout(Duration::from_millis(1)))
            .expect("gc should run");
        let _ = child.wait();

        assert_eq!(report.terminated, 1, "hung spawn should be terminated");
        assert_eq!(report.expired, 1, "heartbeat reap should count as expired");
        assert_pid_gone(pid);
    }

    #[test]
    #[cfg(unix)]
    fn gc_keeps_fresh_spawn_and_record_heartbeat_refreshes() {
        let tempdir = TempDir::new("process-agent-gc-keep");
        let registry = tempdir.join("spawns");
        fs::create_dir_all(&registry).expect("registry should be created");
        let child_pid_file = tempdir.join("child.pid");
        let mut child = spawn_tracked_group(&child_pid_file);
        let pid = child.id();
        let id = format!("spawn-{pid}-keep");
        let stale_beat = now_unix_ms().saturating_sub(5_000);
        let entry = SpawnRegistryEntry {
            version: REGISTRY_VERSION,
            id: id.clone(),
            owner_pid: std::process::id(),
            owner_exe: current_exe_name(),
            pid,
            pgid: pid as i32,
            task_id: None,
            started_at_unix_ms: now_unix_ms(),
            last_heartbeat_unix_ms: Some(stale_beat),
            command: vec!["sh".to_string()],
            working_dir: None,
        };
        fs::write(
            registry.join(format!("{id}.json")),
            serde_json::to_vec_pretty(&entry).expect("entry should serialize"),
        )
        .expect("registry entry should write");

        let supervisor =
            ProcessSupervisor::new(&registry).with_termination_grace(Duration::from_millis(50));
        // Default GC (no ttl, no heartbeat bound) must not touch a live, owned spawn.
        let report = supervisor.gc(GcOptions::default()).expect("gc should run");
        assert_eq!(report.scanned, 1);
        assert_eq!(report.terminated, 0, "fresh owned spawn must survive");
        assert_eq!(report.removed, 0);

        // record_heartbeat advances the timestamp so a heartbeat bound would spare it.
        assert!(supervisor
            .record_heartbeat(&id)
            .expect("heartbeat should write"));
        let refreshed = supervisor
            .entries()
            .expect("registry should read")
            .into_iter()
            .next()
            .expect("entry should remain");
        assert!(
            refreshed.last_heartbeat_unix_ms.unwrap() > stale_beat,
            "heartbeat should advance"
        );
        assert!(
            !supervisor
                .record_heartbeat("spawn-missing")
                .expect("missing heartbeat is ok"),
            "record_heartbeat returns false for an unknown id"
        );

        supervisor.terminate_group(pid as i32).ok();
        let _ = child.wait();
        assert_pid_gone(pid);
    }

    fn shell_agent(
        root: &Path,
        script: &str,
        parent_pid_file: &Path,
        child_pid_file: &Path,
        timeout: Duration,
    ) -> ProcessAgent {
        shell_agent_with_cap(
            root,
            script,
            parent_pid_file,
            child_pid_file,
            timeout,
            DEFAULT_MAX_CONCURRENT_SPAWNS,
        )
    }

    fn shell_agent_with_cap(
        root: &Path,
        script: &str,
        parent_pid_file: &Path,
        child_pid_file: &Path,
        timeout: Duration,
        max_concurrent_spawns: usize,
    ) -> ProcessAgent {
        ProcessAgent::new(
            ProcessAgentConfig::new("sh")
                .with_args(vec![
                    "-c".to_string(),
                    script.to_string(),
                    "artesian-test-sh".to_string(),
                    parent_pid_file.display().to_string(),
                    child_pid_file.display().to_string(),
                ])
                .with_registry_dir(root.join("spawns"))
                .with_timeout(timeout)
                .with_max_lifetime(timeout)
                .with_max_concurrent_spawns(max_concurrent_spawns)
                .with_termination_grace(Duration::from_millis(50)),
        )
    }

    async fn spawn_session(agent: &ProcessAgent) -> AgentSession {
        agent
            .spawn(SpawnRequest {
                role: Role::Worker,
                agent: "sh".to_string(),
                model: None,
                working_dir: None,
            })
            .await
            .expect("session should spawn")
    }

    fn test_context(agent: &str, model: Option<&str>, working_dir: Option<&str>) -> SessionContext {
        SessionContext {
            role: Role::Worker,
            agent: agent.to_string(),
            model: model.map(str::to_string),
            working_dir: working_dir.map(PathBuf::from),
        }
    }

    async fn wait_for_file(path: &Path) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if path.exists() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("{} was not written", path.display());
    }

    async fn wait_for_registry_count(registry: &Path, count: usize) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            let entries = ProcessSupervisor::new(registry)
                .entries()
                .expect("registry should read");
            if entries.len() == count {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!(
            "registry {} did not reach count {count}",
            registry.display()
        );
    }

    fn wait_for_file_sync(path: &Path) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if path.exists() {
                return;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        panic!("{} was not written", path.display());
    }

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
        panic!("pid {pid} survived process-group cleanup");
    }

    #[cfg(unix)]
    fn pid_alive(pid: u32) -> bool {
        match kill(Pid::from_raw(pid as i32), None) {
            Ok(()) => true,
            Err(Errno::EPERM) => true,
            Err(_) => false,
        }
    }
}
