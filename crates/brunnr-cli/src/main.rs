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
use brunnr_core::{
    Agent, AgentBinding, BrunnrConfig, MemoryBackendKind, MemoryConfig, Mode, Role, SpawnRequest,
};
use brunnr_process_agent::{
    fallback_agent_catalog, refresh_agent_catalog, ProcessAgent, ProcessAgentConfig,
    ProcessSupervisor,
};
use clap::{Parser, Subcommand, ValueEnum};
use hird::{
    load_role_definitions, role_summaries, TeamCreate, TeamMessage, TeamMessageKind, TeamRuntime,
    TeamRuntimeConfig, TeamSpawn, TeamTaskAdd, TeamTaskClaim, TeamTaskComplete,
};
use mimisbrunnr::{
    default_migration_collection, export_okf_bundle, recover_after_compaction, verify_okf_bundle,
    CollectionCompat, MemoryBackend, MemoryQuery, MemoryScope, MemoryTier, MigrationPlan,
    MuninnAnchorStore, SessionAnchor, StoreMemory, VectorMemoryConfig,
};
use serde_json::{json, Value};
use thingr::{
    ClaimRequest, CommandVerifier, FilesTaskStore, NewTask, TaskKind, TaskStore, VectorTaskStore,
    Verifier, VerifierGate,
};
use toml_edit::{value, Array, DocumentMut, Item, Table};

const DEFAULT_CONFIG: &str = "brunnr.toml";
const MCP_SERVER_NAME: &str = "brunnr-memory";
const MCP_TOOL_HINT: &str =
    "ALWAYS search the project memory before non-trivial work; store durable, reusable learnings.";

mod import;
mod runtime;
use import::{import_directory, ImportOptions};
use runtime::{
    build_orchestrator, load_config, open_memory_backend, process_supervisor_from_config,
    shutdown_signal,
};

#[derive(Debug, Parser)]
#[command(name = "brunnr", about = "Multi-agent context orchestration")]
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
        #[arg(long, default_value = ".brunnr")]
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
        #[arg(long, default_value = ".brunnr")]
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
        #[arg(long, default_value = ".brunnr")]
        root: PathBuf,
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
    },
    List {
        #[arg(long, default_value = ".brunnr")]
        root: PathBuf,
    },
    Claim {
        id: Option<String>,
        #[arg(long, default_value = "worker")]
        claimant: String,
        #[arg(long, default_value = ".brunnr")]
        root: PathBuf,
    },
    Done {
        id: String,
        #[arg(long = "verify-command")]
        verify_commands: Vec<String>,
        #[arg(long, default_value = ".brunnr")]
        root: PathBuf,
    },
    Find {
        query: String,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".brunnr")]
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
        #[arg(long, default_value = ".brunnr")]
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
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".brunnr")]
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
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".brunnr")]
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
        #[arg(long, default_value = ".brunnr")]
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
        #[arg(long, default_value = ".brunnr")]
        root: PathBuf,
    },
    Recover {
        #[arg(long, default_value_t = 5)]
        limit: usize,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".brunnr")]
        root: PathBuf,
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
    },
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

#[tokio::main]
async fn main() -> Result<()> {
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
        } => consolidate(config, root, allow_llm),
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
    let config = BrunnrConfig {
        mode: brunnr_core::Mode::Memory,
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
        },
        agents,
        coordination: Default::default(),
    };
    let config_path = Path::new(DEFAULT_CONFIG);
    if !config_path.exists() || options.project.is_some() {
        fs::write(config_path, config.to_toml()?)?;
    }
    if options.register_mcp {
        write_mcp_registrations(&env::current_dir()?.join(config_path))?;
    }
    write_master_role_skill(&options.memory_root)?;
    println!(
        "initialized Brunnr memory mode at {} collection={} project={}",
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
        "<!-- SPDX-License-Identifier: Apache-2.0 -->\n\n# Brunnr Lead Role Skill\n\nWhen Brunnr is running in `orchestrate` or `full` mode, inspect `agents.list` for reachable agents, models, and role definitions. Use `memory.context` for compact project recall. For multi-teammate work, create a Hirð with `team.create`, admit definitions with `team.spawn`, coordinate through `team.task.*` and `team.message`, and gate accepted outcomes through the judge/master path before marking work done. For a single bounded subtask, `orchestrate.delegate(worker)` is still sufficient.\n",
    )?;
    Ok(())
}

fn project_memory_root(memory_root: Option<PathBuf>, project: Option<&str>) -> PathBuf {
    memory_root.unwrap_or_else(|| {
        project
            .map(|project| PathBuf::from(".brunnr").join(project))
            .unwrap_or_else(|| PathBuf::from(".brunnr"))
    })
}

fn project_collection(collection: Option<String>, project: Option<&str>) -> String {
    collection
        .or_else(|| project.map(sanitize_project_name))
        .unwrap_or_else(|| "brunnr-memory".to_string())
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
            brunnr_core::AgentMessage {
                content: String::new(),
            },
        )
        .await?;
    println!(
        "spawn completed: role={} alias={} agent={} cwd={}",
        request.role.canonical_alias(),
        request.role.norse_alias(),
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
                })
                .await?;
            println!("stored memory id={} node_id={}", record.id, record.node_id);
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
            config,
            root,
            backend,
        } => {
            let backend = open_backend_for_command(&config, root, backend)?;
            let mut memory_query = MemoryQuery::new(query).with_limit(limit);
            memory_query.node_id = node_id;
            memory_query.scope = scope.map(Into::into);
            memory_query.agent_id = agent_id;
            memory_query.session_id = session_id;
            memory_query.task_id = task_id;
            memory_query.user_id = user_id;
            for hit in backend.find(memory_query).await? {
                println!(
                    "{:.4}\t{}\t{}\t{}",
                    hit.score, hit.record.id, hit.record.node_id, hit.record.content
                );
            }
        }
        MemoryCommand::Context {
            query,
            limit,
            index_chars,
            config,
            root,
            backend,
        } => {
            let memory_config = memory_config_for_command(&config, root, backend)?;
            let index = read_index_slice(&memory_config.root, index_chars)?;
            let backend = open_memory_backend(&memory_config)?;
            let hits = backend
                .find(MemoryQuery::new(query).with_limit(limit))
                .await?;
            if let Some(index) = index {
                println!("# index.md\n{index}");
            }
            println!("# memory.find");
            for hit in hits {
                println!(
                    "{:.4}\t{}\t{}\t{}",
                    hit.score, hit.record.id, hit.record.node_id, hit.record.content
                );
            }
        }
        MemoryCommand::Anchor { command } => anchor(command).await?,
    }
    Ok(())
}

async fn anchor(command: AnchorCommand) -> Result<()> {
    match command {
        AnchorCommand::Get { config, root } => {
            let memory = memory_config_for_command(&config, root, None)?;
            let store = MuninnAnchorStore::new(&memory.root);
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
            let store = MuninnAnchorStore::new(&memory.root);
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
            let anchor_store = MuninnAnchorStore::new(&memory.root);
            let backend = open_memory_backend(&memory)?;
            let recovered =
                recover_after_compaction(&anchor_store, backend.as_ref(), limit).await?;
            println!("{}", serde_json::to_string_pretty(&recovered)?);
        }
    }
    Ok(())
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
    println!("next best step: run `brunnr consolidate` when you want the opt-in LLM semantic pass");
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
            let text = fs::read_to_string(DEFAULT_CONFIG).expect("init wrote brunnr.toml");
            BrunnrConfig::from_toml(&text)
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
    println!("next best step: run `brunnr consolidate` when you want the opt-in LLM semantic pass");
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
        bail!("brunnr migrate okf-bundle currently requires backend = qdrant for atomic alias swap");
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
        bail!("brunnr migrate rechunk currently requires backend = sqlite-vec; for Qdrant use brunnr migrate okf-bundle");
    }
    use mimisbrunnr::{rechunk_oversized_sqlite, SqliteVecVectorStore, SqliteVecVectorStoreConfig};
    use std::path::PathBuf as SPath;
    let db_path = SPath::from(&config.memory.root)
        .join(format!("{}.sqlite", config.memory.collection));
    let store =
        SqliteVecVectorStore::open(SqliteVecVectorStoreConfig::new(&db_path)).map_err(|e| {
            anyhow::anyhow!("failed to open sqlite-vec store at {}: {e}", db_path.display())
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
            "brunnr snapshot currently requires backend = qdrant; use brunnr okf export for files"
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

fn consolidate(config_path: PathBuf, root: PathBuf, allow_llm: bool) -> Result<()> {
    let memory = memory_config_for_command(&config_path, root, None)?;
    let memory_root = PathBuf::from(&memory.root);
    let index_path = memory_root.join("memory").join("index.md");
    let log_path = memory_root.join("memory").join("log.md");
    fs::create_dir_all(memory_root.join("memory"))?;
    if !index_path.exists() {
        fs::write(
            &index_path,
            "---\ntype: index\ntitle: Brunnr Memory Index\n---\n\n# Brunnr Memory Index\n\nNo structural import catalog exists yet.\n",
        )?;
    }
    let mode = if allow_llm {
        "llm-semantic-requested"
    } else {
        "structural-no-llm"
    };
    let entry = format!(
        "\n- {} consolidate mode={mode}; LLM semantic consolidation is opt-in and requires a configured provider adapter.\n",
        chrono_like_timestamp()
    );
    fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(&log_path)?
        .write_all(entry.as_bytes())?;
    if allow_llm {
        println!(
            "consolidate recorded opt-in LLM semantic request; provider adapter execution is not enabled by default"
        );
    } else {
        println!(
            "consolidate verified structural index/log without LLM calls; pass --allow-llm only when semantic consolidation is explicitly approved"
        );
    }
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
) -> Result<mimisbrunnr::MigrationReport> {
    use mimisbrunnr::{migrate_okf_bundle, FastembedTextEmbedder, QdrantVectorStore};

    let store = QdrantVectorStore::connect(qdrant_config(memory)?)?;
    Ok(migrate_okf_bundle(&store, plan, Arc::new(FastembedTextEmbedder::new()?)).await?)
}

#[cfg(not(feature = "qdrant"))]
async fn migrate_qdrant(
    _memory: &MemoryConfig,
    _plan: MigrationPlan,
) -> Result<mimisbrunnr::MigrationReport> {
    bail!("brunnr migrate requires building brunnr-cli with the qdrant feature")
}

#[cfg(feature = "qdrant")]
async fn snapshot_qdrant(
    memory: &MemoryConfig,
    collection: &str,
    output_dir: &Path,
) -> Result<mimisbrunnr::SnapshotReport> {
    use mimisbrunnr::{QdrantVectorStore, VectorCollectionAdmin};

    let store = QdrantVectorStore::connect(qdrant_config(memory)?)?;
    Ok(store.snapshot_collection(collection, output_dir).await?)
}

#[cfg(not(feature = "qdrant"))]
async fn snapshot_qdrant(
    _memory: &MemoryConfig,
    _collection: &str,
    _output_dir: &Path,
) -> Result<mimisbrunnr::SnapshotReport> {
    bail!("brunnr snapshot requires building brunnr-cli with the qdrant feature")
}

#[cfg(feature = "qdrant")]
fn qdrant_config(memory: &MemoryConfig) -> Result<mimisbrunnr::QdrantVectorStoreConfig> {
    let url = memory
        .qdrant_url
        .clone()
        .or_else(|| env::var("QDRANT_URL").ok())
        .context("Qdrant backend requires qdrant_url in config or QDRANT_URL")?;
    let mut config = mimisbrunnr::QdrantVectorStoreConfig::new(url);
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
    let report = mimisbrunnr::preflight_qdrant(qdrant_config(memory)?).await?;
    eprintln!(
        "Qdrant preflight ok: grpc={} rest={} version={}",
        report.grpc_url, report.rest_url, report.grpc_version
    );
    Ok(())
}

#[cfg(not(feature = "qdrant"))]
async fn preflight_qdrant_memory(_memory: &MemoryConfig) -> Result<()> {
    bail!("Qdrant preflight requires building brunnr-cli with the qdrant feature")
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
        let config = BrunnrConfig::from_toml(&text)
            .with_context(|| format!("parse {}", config_path.display()))?;
        if config.mode != Mode::Memory {
            bail!("memory commands require mode = memory");
        }
        config.memory
    } else {
        MemoryConfig {
            backend: backend.unwrap_or(BackendArg::Files).into(),
            root: root.display().to_string(),
            collection: "brunnr-memory".to_string(),
            qdrant_url: env::var("QDRANT_URL").ok(),
            qdrant_rest_url: env::var("QDRANT_REST_URL").ok(),
            qdrant_api_key_env: Some("QDRANT_API_KEY".to_string()),
            local_rerank_enabled: true,
            hyde_enabled: false,
            multi_query_enabled: false,
            debate_enabled: false,
            llm_consolidation_enabled: false,
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

fn write_mcp_registrations(config_path: &Path) -> Result<()> {
    let config_path = config_path
        .canonicalize()
        .unwrap_or_else(|_| config_path.to_path_buf());
    write_claude_mcp(config_path.as_path())?;
    write_codex_mcp(config_path.as_path())?;
    write_zed_mcp(config_path.as_path())?;
    Ok(())
}

fn mcp_args(config_path: &Path) -> Vec<String> {
    vec!["--config".to_string(), config_path.display().to_string()]
}

fn write_claude_mcp(config_path: &Path) -> Result<()> {
    let path = Path::new(".mcp.json");
    let mut root = read_json_object(path)?;
    let server = json!({
        "command": "brunnr-mcp",
        "args": mcp_args(config_path),
        "env": {
            "BRUNNR_MCP_TOOL_HINT": MCP_TOOL_HINT
        }
    });
    ensure_object(&mut root, "mcpServers")?.insert(MCP_SERVER_NAME.to_string(), server);
    write_json(path, &root)
}

fn write_codex_mcp(config_path: &Path) -> Result<()> {
    let path = home_dir()?.join(".codex").join("config.toml");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let text = fs::read_to_string(&path).unwrap_or_default();
    let mut document = text.parse::<DocumentMut>().unwrap_or_default();
    ensure_toml_table(&mut document, "mcp_servers");
    document["mcp_servers"][MCP_SERVER_NAME]["command"] = value("brunnr-mcp");
    let mut args = Array::new();
    for arg in mcp_args(config_path) {
        args.push(arg);
    }
    document["mcp_servers"][MCP_SERVER_NAME]["args"] = value(args);
    document["mcp_servers"][MCP_SERVER_NAME]["env"]["BRUNNR_MCP_TOOL_HINT"] = value(MCP_TOOL_HINT);
    fs::write(path, document.to_string())?;
    Ok(())
}

fn write_zed_mcp(config_path: &Path) -> Result<()> {
    let path = zed_settings_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut root = read_json_object(&path)?;
    let server = json!({
        "command": {
            "path": "brunnr-mcp",
            "args": mcp_args(config_path),
            "env": {
                "BRUNNR_MCP_TOOL_HINT": MCP_TOOL_HINT
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
    if let Some(home) = env::var_os("BRUNNR_HOME").or_else(|| env::var_os("HOME")) {
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
