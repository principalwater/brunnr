// SPDX-License-Identifier: Apache-2.0

use std::{
    env, fs,
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
};

use anyhow::{bail, Context, Result};
use brunnr_core::{
    AgentBinding, BrunnrConfig, MemoryBackendKind, MemoryConfig, Mode, Role, SpawnRequest,
};
use clap::{Parser, Subcommand, ValueEnum};
use mimisbrunnr::{
    backfill_directory, FilesBackend, MemoryBackend, MemoryQuery, MemoryTier, SqliteVecVectorStore,
    SqliteVecVectorStoreConfig, StoreMemory, VectorMemoryBackend, VectorMemoryConfig,
};
use serde_json::{json, Value};
use toml_edit::{value, Array, DocumentMut, Item, Table};

#[cfg(feature = "qdrant")]
use mimisbrunnr::{QdrantVectorStore, QdrantVectorStoreConfig};

const DEFAULT_CONFIG: &str = "brunnr.toml";
const MCP_SERVER_NAME: &str = "brunnr-memory";
const MCP_TOOL_HINT: &str =
    "ALWAYS search the project memory before non-trivial work; store durable, reusable learnings.";

#[derive(Debug, Parser)]
#[command(name = "brunnr", about = "Multi-agent context orchestration")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Init {
        #[arg(long, default_value = ".brunnr")]
        memory_root: PathBuf,
        #[arg(long, value_enum, default_value_t = BackendArg::Files)]
        backend: BackendArg,
        #[arg(long, default_value = "brunnr-memory")]
        collection: String,
        #[arg(long, env = "QDRANT_URL")]
        qdrant_url: Option<String>,
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
    },
    Memory {
        #[command(subcommand)]
        command: MemoryCommand,
    },
    Backfill {
        directory: PathBuf,
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".brunnr")]
        root: PathBuf,
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
    },
}

#[derive(Debug, Subcommand)]
enum MemoryCommand {
    Store {
        content: String,
        #[arg(long = "tag")]
        tags: Vec<String>,
        #[arg(long)]
        node_id: Option<String>,
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
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
        #[arg(long, default_value = ".brunnr")]
        root: PathBuf,
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum BackendArg {
    Files,
    SqliteVec,
    Qdrant,
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
            backend,
            collection,
            qdrant_url,
            qdrant_api_key_env,
            non_interactive,
            register_mcp,
        } => init(
            memory_root,
            backend,
            collection,
            qdrant_url,
            qdrant_api_key_env,
            non_interactive,
            register_mcp,
        ),
        Command::Spawn { role, agent } => spawn(&role, &agent),
        Command::Memory { command } => memory(command).await,
        Command::Backfill {
            directory,
            config,
            root,
            backend,
        } => backfill(directory, config, root, backend).await,
    }
}

fn init(
    memory_root: PathBuf,
    backend: BackendArg,
    collection: String,
    qdrant_url: Option<String>,
    qdrant_api_key_env: String,
    _non_interactive: bool,
    register_mcp: bool,
) -> Result<()> {
    fs::create_dir_all(memory_root.join("memory"))
        .with_context(|| format!("create memory root {}", memory_root.display()))?;
    let agents = detect_agents();
    let config = BrunnrConfig {
        mode: brunnr_core::Mode::Memory,
        memory: MemoryConfig {
            backend: backend.into(),
            root: memory_root.display().to_string(),
            collection,
            qdrant_url,
            qdrant_api_key_env: Some(qdrant_api_key_env),
        },
        agents,
    };
    let config_path = Path::new(DEFAULT_CONFIG);
    if !config_path.exists() {
        fs::write(config_path, config.to_toml()?)?;
    }
    if register_mcp {
        write_mcp_registrations(&env::current_dir()?.join(config_path))?;
    }
    println!(
        "initialized Brunnr memory mode at {}",
        memory_root.display()
    );
    Ok(())
}

fn spawn(role: &str, agent: &str) -> Result<()> {
    let role = Role::from_str(role)?;
    let request = SpawnRequest {
        role,
        agent: agent.to_string(),
        model: None,
        working_dir: env::current_dir()
            .ok()
            .map(|path| path.display().to_string()),
    };
    println!(
        "spawn request accepted: role={} alias={} agent={} cwd={}",
        request.role.canonical_alias(),
        request.role.norse_alias(),
        request.agent,
        request.working_dir.as_deref().unwrap_or(".")
    );
    Ok(())
}

async fn memory(command: MemoryCommand) -> Result<()> {
    match command {
        MemoryCommand::Store {
            content,
            tags,
            node_id,
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
                })
                .await?;
            println!("stored memory id={} node_id={}", record.id, record.node_id);
        }
        MemoryCommand::Find {
            query,
            limit,
            node_id,
            config,
            root,
            backend,
        } => {
            let backend = open_backend_for_command(&config, root, backend)?;
            let mut memory_query = MemoryQuery::new(query).with_limit(limit);
            memory_query.node_id = node_id;
            for hit in backend.find(memory_query).await? {
                println!(
                    "{:.4}\t{}\t{}\t{}",
                    hit.score, hit.record.id, hit.record.node_id, hit.record.content
                );
            }
        }
    }
    Ok(())
}

async fn backfill(
    directory: PathBuf,
    config: PathBuf,
    root: PathBuf,
    backend: Option<BackendArg>,
) -> Result<()> {
    let backend = open_backend_for_command(&config, root, backend)?;
    let stats = backfill_directory(backend.as_ref(), directory).await?;
    println!(
        "backfill scanned={} imported={} skipped_duplicates={}",
        stats.scanned, stats.imported, stats.skipped_duplicates
    );
    Ok(())
}

fn open_backend_for_command(
    config_path: &Path,
    root: PathBuf,
    backend: Option<BackendArg>,
) -> Result<Arc<dyn MemoryBackend>> {
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
            qdrant_api_key_env: Some("QDRANT_API_KEY".to_string()),
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
    open_memory_backend(&config)
}

fn open_memory_backend(config: &MemoryConfig) -> Result<Arc<dyn MemoryBackend>> {
    match config.backend {
        MemoryBackendKind::Files => Ok(Arc::new(FilesBackend::new(&config.root))),
        MemoryBackendKind::SqliteVec => {
            let store = SqliteVecVectorStore::open(SqliteVecVectorStoreConfig::new(sqlite_path(
                &config.root,
            )))?;
            Ok(Arc::new(VectorMemoryBackend::new(
                store,
                VectorMemoryConfig::new(&config.collection),
            )?))
        }
        MemoryBackendKind::Qdrant => open_qdrant_backend(config),
        MemoryBackendKind::TencentDb => bail!("TencentDB backend is not available yet"),
    }
}

#[cfg(feature = "qdrant")]
fn open_qdrant_backend(config: &MemoryConfig) -> Result<Arc<dyn MemoryBackend>> {
    let url = config
        .qdrant_url
        .clone()
        .or_else(|| env::var("QDRANT_URL").ok())
        .context("Qdrant backend requires qdrant_url in config or QDRANT_URL")?;
    let mut vector_config = QdrantVectorStoreConfig::new(url);
    if let Some(env_name) = &config.qdrant_api_key_env {
        vector_config.api_key = env::var(env_name).ok();
    }
    let store = QdrantVectorStore::connect(vector_config)?;
    Ok(Arc::new(VectorMemoryBackend::new(
        store,
        VectorMemoryConfig::new(&config.collection),
    )?))
}

#[cfg(not(feature = "qdrant"))]
fn open_qdrant_backend(_config: &MemoryConfig) -> Result<Arc<dyn MemoryBackend>> {
    bail!("Qdrant backend requires building brunnr-cli with the qdrant feature")
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
    .filter(|name| command_exists(name))
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
            agent,
            model: None,
        })
    })
    .collect()
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
