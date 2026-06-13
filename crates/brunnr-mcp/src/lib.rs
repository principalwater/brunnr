// SPDX-License-Identifier: Apache-2.0

use std::{path::PathBuf, sync::Arc};

#[cfg(feature = "qdrant")]
use std::env;

use brunnr_core::{MemoryBackendKind, MemoryConfig};
use mimisbrunnr::{
    FilesBackend, MemoryBackend, MemoryQuery, MemoryTier, SqliteVecVectorStore,
    SqliteVecVectorStoreConfig, StoreMemory, VectorMemoryBackend, VectorMemoryConfig,
};
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

#[cfg(feature = "qdrant")]
use mimisbrunnr::{QdrantVectorStore, QdrantVectorStoreConfig};

const TOOL_INSTRUCTIONS: &str =
    "ALWAYS search the project memory before non-trivial work; store durable, reusable learnings.";

#[derive(Clone)]
pub struct MemoryServer {
    backend: Arc<dyn MemoryBackend>,
    tool_router: ToolRouter<Self>,
}

impl MemoryServer {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self::with_backend(Arc::new(FilesBackend::new(root)))
    }

    pub fn with_backend(backend: Arc<dyn MemoryBackend>) -> Self {
        Self {
            backend,
            tool_router: Self::tool_router(),
        }
    }

    pub fn from_config(config: &MemoryConfig) -> anyhow::Result<Self> {
        Ok(Self::with_backend(open_memory_backend(config)?))
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct FindRequest {
    pub query: String,
    pub limit: Option<usize>,
    pub node_id: Option<String>,
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
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct StoreRequest {
    pub content: String,
    pub tags: Option<Vec<String>>,
    pub node_id: Option<String>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct StoreResponse {
    pub id: String,
    pub node_id: String,
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
        let hits = self
            .backend
            .find(query)
            .await
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?
            .into_iter()
            .map(|hit| FindHit {
                id: hit.record.id.to_string(),
                node_id: hit.record.node_id,
                content: hit.record.content,
                score: hit.score,
                tags: hit.record.tags,
            })
            .collect();
        Ok(Json(FindResponse { hits }))
    }

    #[tool(
        name = "memory.store",
        description = "Store durable, reusable learnings in project memory."
    )]
    pub async fn memory_store(
        &self,
        Parameters(request): Parameters<StoreRequest>,
    ) -> Result<Json<StoreResponse>, ErrorData> {
        let record = self
            .backend
            .store(StoreMemory {
                content: request.content,
                tags: request.tags.unwrap_or_default(),
                metadata: Default::default(),
                tier: MemoryTier::L1Atom,
                node_id: request.node_id,
                created_at: None,
            })
            .await
            .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
        Ok(Json(StoreResponse {
            id: record.id.to_string(),
            node_id: record.node_id,
        }))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for MemoryServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_instructions(
            format!(
                "Brunnr memory server exposing memory.find and memory.store. {TOOL_INSTRUCTIONS}"
            ),
        )
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

pub fn open_memory_backend(config: &MemoryConfig) -> anyhow::Result<Arc<dyn MemoryBackend>> {
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
        MemoryBackendKind::TencentDb => anyhow::bail!("TencentDB backend is not available yet"),
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
fn open_qdrant_backend(_config: &MemoryConfig) -> anyhow::Result<Arc<dyn MemoryBackend>> {
    anyhow::bail!("Qdrant backend requires building brunnr-mcp with the qdrant feature")
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
