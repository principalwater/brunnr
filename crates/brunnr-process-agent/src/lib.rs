// SPDX-License-Identifier: Apache-2.0

//! Process-backed [`brunnr_core::Agent`] adapter.

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

use brunnr_core::{
    Agent, AgentBinding, AgentCapabilities, AgentCatalog, AgentCatalogEntry, AgentError,
    AgentEvent, AgentEventStream, AgentMessage, AgentModel, AgentResponse, AgentResult,
    AgentSession, AgentUnreachableReason, Role, SpawnRequest,
};
use futures_util::{future::BoxFuture, stream, FutureExt};
use serde::{Deserialize, Serialize};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    process::Command,
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
        let session_id = session.id.clone();
        async move {
            let context = self
                .sessions
                .lock()
                .map_err(|error| AgentError::Session(error.to_string()))?
                .get(&session_id)
                .cloned()
                .ok_or_else(|| AgentError::Session(format!("unknown session: {session_id}")))?;
            let output = run_process(&self.config, &context, &message.content).await?;
            Ok(AgentResponse { content: output })
        }
        .boxed()
    }

    fn stream(
        &self,
        session: &AgentSession,
        message: AgentMessage,
    ) -> BoxFuture<'_, AgentResult<AgentEventStream>> {
        let session = session.clone();
        async move {
            let response = self.send(&session, message).await?;
            Ok(Box::pin(stream::iter([
                Ok(AgentEvent::Text(response.content)),
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
            "agent '{}' is not in the model catalog; run `brunnr agents refresh`",
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
            "<none>; run `brunnr agents refresh` or configure a supported model"
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
            "<none>; run `brunnr agents refresh` or choose a configured model"
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
        "BRUNNR_{}_MODELS_CMD",
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
    pub command: Vec<String>,
    pub working_dir: Option<String>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ReapReport {
    pub scanned: usize,
    pub terminated: usize,
    pub removed: usize,
    pub skipped_unverified: usize,
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
        let entry = SpawnRegistryEntry {
            version: REGISTRY_VERSION,
            id: format!("spawn-{pid}-{}", now_unix_ms()),
            owner_pid: std::process::id(),
            owner_exe: current_exe_name(),
            pid,
            pgid,
            task_id,
            started_at_unix_ms: now_unix_ms(),
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
) -> AgentResult<String> {
    let supervisor = config.supervisor();
    supervisor
        .ensure_spawn_capacity()
        .map_err(|error| AgentError::Unavailable(error.to_string()))?;

    let mut command = Command::new(&config.command);
    configure_process_group(&mut command);
    let mut prompt_was_arg = false;
    for arg in &config.args {
        let rendered = render_arg(arg, context, prompt);
        prompt_was_arg |= arg.contains("{prompt}");
        command.arg(rendered);
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
    let command_line = command_line(config);
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

    if !prompt_was_arg && !prompt.is_empty() {
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(prompt.as_bytes())
                .await
                .map_err(|error| AgentError::Session(error.to_string()))?;
        }
    }
    drop(child.stdin.take());

    let stdout = child.stdout.take().map(read_pipe);
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

fn render_arg(template: &str, context: &SessionContext, prompt: &str) -> String {
    template
        .replace("{prompt}", prompt)
        .replace("{role}", context.role.canonical_alias())
        .replace("{alias}", context.role.norse_alias())
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
    std::env::var_os("BRUNNR_SPAWN_REGISTRY")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".brunnr").join("spawns"))
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

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

    use brunnr_core::{AgentMessage, Role, SpawnRequest};
    use brunnr_test_support::TempDir;
    #[cfg(unix)]
    use std::os::unix::process::CommandExt;

    use super::*;

    #[tokio::test]
    async fn launches_real_echo_subprocess() {
        let tempdir = TempDir::new("process-agent-echo");
        let agent = ProcessAgent::new(
            ProcessAgentConfig::new("echo")
                .with_args(vec!["brunnr".into()])
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

        assert_eq!(response.content.trim(), "brunnr");
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
        let secret = "sk-brunnr-secret-1234567890";
        let script = format!(
            "printf 'token=brunnr-token-value\\n{secret}\\n'; printf '%4096s\\n' x; exit 7"
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
        assert!(!text.contains("brunnr-token-value"));
        assert!(text.contains("[REDACTED]"));
        assert!(text.len() < MAX_PROCESS_ERROR_OUTPUT_CHARS + 256);
    }

    #[tokio::test]
    async fn discovery_output_and_registry_command_are_redacted() {
        let tempdir = TempDir::new("process-agent-secret-discovery");
        let env_name = "BRUNNR_SECRET_PROBE_MODELS_CMD";
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
            owner_exe: Some("brunnr-dead-owner".to_string()),
            pid,
            pgid: pid as i32,
            task_id: Some("task-restart".to_string()),
            started_at_unix_ms: now_unix_ms(),
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
                    "brunnr-test-sh".to_string(),
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
