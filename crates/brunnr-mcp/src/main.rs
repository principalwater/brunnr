// SPDX-License-Identifier: Apache-2.0

use std::{env, fs, path::PathBuf};

use brunnr_core::{BrunnrConfig, MemoryBackendKind, MemoryConfig, Mode};
use clap::{Parser, ValueEnum};

#[derive(Debug, Parser)]
#[command(name = "brunnr-mcp", about = "Brunnr MCP memory server")]
struct Args {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, env = "BRUNNR_MEMORY_ROOT", default_value = ".brunnr")]
    root: PathBuf,
    #[arg(long, value_enum, default_value_t = BackendArg::Files)]
    backend: BackendArg,
    #[arg(long, default_value = "brunnr-memory")]
    collection: String,
    #[arg(long, env = "QDRANT_URL")]
    qdrant_url: Option<String>,
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

    let config = load_memory_config(&args)?;
    brunnr_mcp::run_stdio_with_config(&config).await
}

fn load_memory_config(args: &Args) -> anyhow::Result<MemoryConfig> {
    if let Some(path) = &args.config {
        let text = fs::read_to_string(path)?;
        let config = BrunnrConfig::from_toml(&text)?;
        if config.mode != Mode::Memory {
            anyhow::bail!("brunnr-mcp requires mode = memory");
        }
        return Ok(config.memory);
    }

    Ok(MemoryConfig {
        backend: args.backend.into(),
        root: args.root.display().to_string(),
        collection: args.collection.clone(),
        qdrant_url: args
            .qdrant_url
            .clone()
            .or_else(|| env::var("QDRANT_URL").ok()),
        qdrant_api_key_env: Some(args.qdrant_api_key_env.clone()),
    })
}
