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
    Agent, AgentCapabilities, AgentError, AgentEvent, AgentEventStream, AgentMessage,
    AgentResponse, AgentResult, AgentSession, Role, SpawnRequest,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessAgentConfig {
    pub command: String,
    pub args: Vec<String>,
    pub working_dir: Option<PathBuf>,
    pub timeout: Duration,
    pub max_lifetime: Duration,
    pub termination_grace: Duration,
    pub registry_dir: PathBuf,
    pub max_concurrent_spawns: usize,
}

impl ProcessAgentConfig {
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            args: Vec::new(),
            working_dir: None,
            timeout: DEFAULT_TIMEOUT,
            max_lifetime: DEFAULT_MAX_LIFETIME,
            termination_grace: DEFAULT_TERMINATION_GRACE,
            registry_dir: default_registry_dir(),
            max_concurrent_spawns: DEFAULT_MAX_CONCURRENT_SPAWNS,
        }
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
}

impl Agent for ProcessAgent {
    fn spawn(&self, request: SpawnRequest) -> BoxFuture<'_, AgentResult<AgentSession>> {
        async move {
            if self.config.command.trim().is_empty() {
                return Err(AgentError::Unavailable(
                    "process agent command is empty".to_string(),
                ));
            }
            let id = format!(
                "{}-{}-{}",
                request.role.canonical_alias(),
                sanitize_agent_id(&request.agent),
                self.next_session.fetch_add(1, Ordering::Relaxed)
            );
            let context = SessionContext {
                role: request.role,
                agent: request.agent.clone(),
                model: request.model.clone(),
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
