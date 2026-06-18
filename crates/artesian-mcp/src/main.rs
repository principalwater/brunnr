// SPDX-License-Identifier: Apache-2.0

use std::{env, fs, path::PathBuf};

use artesian_core::{ArtesianConfig, MemoryBackendKind, MemoryConfig, Mode};
use clap::{Parser, ValueEnum};

#[derive(Debug, Parser)]
#[command(name = "artesian-mcp", about = "Artesian MCP memory server")]
struct Args {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, env = "ARTESIAN_MEMORY_ROOT", default_value = ".artesian")]
    root: PathBuf,
    #[arg(long, value_enum, default_value_t = BackendArg::Files)]
    backend: BackendArg,
    #[arg(long, default_value = "artesian-memory")]
    collection: String,
    #[arg(long, env = "QDRANT_URL")]
    qdrant_url: Option<String>,
    #[arg(long, env = "QDRANT_REST_URL")]
    qdrant_rest_url: Option<String>,
    #[arg(long, default_value = "QDRANT_API_KEY")]
    qdrant_api_key_env: String,
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
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    let config = load_runtime_config(&args)?;
    artesian_mcp::run_stdio_with_artesian_config(config.config).await
}

struct RuntimeConfig {
    config: ArtesianConfig,
}

fn load_runtime_config(args: &Args) -> anyhow::Result<RuntimeConfig> {
    if let Some(path) = &args.config {
        let text = fs::read_to_string(path)?;
        let config = ArtesianConfig::from_toml(&text)?;
        return Ok(RuntimeConfig { config });
    }

    Ok(RuntimeConfig {
        config: ArtesianConfig {
            mode: Mode::Memory,
            memory: MemoryConfig {
                backend: args.backend.into(),
                root: args.root.display().to_string(),
                collection: args.collection.clone(),
                qdrant_url: args
                    .qdrant_url
                    .clone()
                    .or_else(|| env::var("QDRANT_URL").ok()),
                qdrant_rest_url: args
                    .qdrant_rest_url
                    .clone()
                    .or_else(|| env::var("QDRANT_REST_URL").ok()),
                qdrant_api_key_env: Some(args.qdrant_api_key_env.clone()),
                local_rerank_enabled: true,
                hyde_enabled: false,
                multi_query_enabled: false,
                debate_enabled: false,
                llm_consolidation_enabled: false,
                semantic_cache: Default::default(),
            },
            agents: Vec::new(),
            coordination: Default::default(),
            acc: Default::default(),
        },
    })
}
