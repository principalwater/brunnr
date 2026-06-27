// SPDX-License-Identifier: Apache-2.0

use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

#[cfg(feature = "qdrant")]
use std::env;

use anyhow::{bail, Context, Result};
use aquifer::{
    FilesBackend, MemoryBackend, SqliteVecVectorStore, SqliteVecVectorStoreConfig,
    VectorMemoryBackend, VectorMemoryConfig,
};
use artesian_core::{Agent, AgentBinding, ArtesianConfig, MemoryBackendKind, MemoryConfig, Role};
use artesian_process_agent::{ProcessAgent, ProcessAgentConfig, ProcessSupervisor};
use basin::{DryRunAgent, Orchestrator, OrchestratorConfig};
use headrace::{CommandVerifier, FilesTaskStore, Verifier, VerifierGate};
use sandbox::ScratchWorkspaceProvider;

#[cfg(feature = "qdrant")]
use aquifer::{QdrantVectorStore, QdrantVectorStoreConfig};

pub fn build_orchestrator(
    config: ArtesianConfig,
    root: PathBuf,
    repo_root: PathBuf,
    dry_run: bool,
) -> Result<Orchestrator> {
    let memory = open_memory_backend(&config.memory)?;
    let task_store = Arc::new(FilesTaskStore::new(&root));
    let workspace_provider = Arc::new(ScratchWorkspaceProvider::new(root.join("workspaces")));
    let verifier_gate = verifier_gate_from_config(&config);
    let worker: Arc<dyn Agent> = if dry_run {
        Arc::new(DryRunAgent::new("dry-run-worker"))
    } else {
        Arc::new(process_agent_from_binding(
            &config,
            Role::Worker,
            &repo_root,
        )?)
    };
    let judge = if dry_run {
        Some(Arc::new(DryRunAgent::new("dry-run-judge")) as Arc<dyn Agent>)
    } else {
        config
            .agents
            .iter()
            .find(|binding| binding.role == Role::Judge)
            .map(|binding| process_agent_from_binding_value(&config, binding, &repo_root))
            .transpose()?
            .map(|agent| Arc::new(agent) as Arc<dyn Agent>)
    };
    let orchestrator_config = OrchestratorConfig::from_artesian(&config, repo_root);
    Ok(Orchestrator::new(
        orchestrator_config,
        task_store,
        memory,
        workspace_provider,
        worker,
        judge,
        verifier_gate,
    ))
}

pub fn load_config(config_path: &Path) -> Result<ArtesianConfig> {
    let text = fs::read_to_string(config_path)
        .with_context(|| format!("read {}", config_path.display()))?;
    ArtesianConfig::from_toml(&text).with_context(|| format!("parse {}", config_path.display()))
}

pub fn process_supervisor_from_config(
    config: &ArtesianConfig,
    repo_root: &Path,
) -> ProcessSupervisor {
    ProcessSupervisor::new(spawn_registry_dir(config, repo_root))
        .with_max_concurrent_spawns(
            config
                .coordination
                .max_concurrent_spawns
                .unwrap_or(32)
                .max(1),
        )
        .with_termination_grace(Duration::from_millis(
            config
                .coordination
                .spawn_shutdown_grace_millis
                .unwrap_or(2_000),
        ))
}

pub async fn shutdown_signal() -> Result<&'static str> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};

        let mut terminate = signal(SignalKind::terminate()).context("listen for sigterm")?;
        tokio::select! {
            signal = tokio::signal::ctrl_c() => {
                signal.context("listen for ctrl-c")?;
                Ok("SIGINT")
            }
            _ = terminate.recv() => Ok("SIGTERM"),
        }
    }
    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c().await.context("listen for ctrl-c")?;
        Ok("SIGINT")
    }
}

pub fn open_memory_backend(config: &MemoryConfig) -> Result<Arc<dyn MemoryBackend>> {
    open_memory_backend_inner(config, false)
}

/// Open a memory backend with deterministic relation extraction enabled.
///
/// Relation extraction is cheap (no LLM) and builds `mentions` links between entities found in
/// each record's content and tags.  The standard `open_memory_backend` leaves it off (the
/// default); import paths call this variant so every ingested chunk arrives pre-linked.
pub fn open_memory_backend_with_relations(config: &MemoryConfig) -> Result<Arc<dyn MemoryBackend>> {
    open_memory_backend_inner(config, true)
}

fn open_memory_backend_inner(
    config: &MemoryConfig,
    relation_extraction: bool,
) -> Result<Arc<dyn MemoryBackend>> {
    match config.backend {
        MemoryBackendKind::Files => Ok(Arc::new(
            FilesBackend::new(&config.root).with_track_access(config.track_access),
        )),
        MemoryBackendKind::SqliteVec => {
            let store = SqliteVecVectorStore::open(SqliteVecVectorStoreConfig::new(sqlite_path(
                &config.root,
            )))?;
            let vector_config = vector_memory_config_from(config, relation_extraction);
            let backend = VectorMemoryBackend::new(store, vector_config)?;
            let backend = attach_configured_reranker(backend, config);
            Ok(finish_vector_backend(backend, config))
        }
        MemoryBackendKind::Qdrant => open_qdrant_backend_inner(config, relation_extraction),
        MemoryBackendKind::TencentDb => bail!("TencentDB backend is not available yet"),
    }
}

/// Build a semantic cache from config, or `None` when disabled.
fn semantic_cache_from_config(config: &MemoryConfig) -> Option<aquifer::SemanticCache> {
    if !config.semantic_cache.enabled {
        return None;
    }
    let mut cache = aquifer::SemanticCache::new(
        config.semantic_cache.capacity,
        config.semantic_cache.min_similarity,
    );
    if let Some(ttl) = config.semantic_cache.ttl_seconds {
        cache = cache.with_ttl(Duration::from_secs(ttl));
    }
    Some(cache)
}

fn vector_memory_config_from(
    config: &MemoryConfig,
    relation_extraction: bool,
) -> VectorMemoryConfig {
    let mut vector_config = VectorMemoryConfig::new(&config.collection)
        .with_relation_extraction(relation_extraction)
        .with_track_access(config.track_access);
    if config.rerank {
        vector_config = vector_config
            .with_rerank(true)
            .with_rerank_candidates(config.effective_rerank_candidates());
    }
    vector_config
}

fn attach_configured_reranker<V>(
    backend: VectorMemoryBackend<V>,
    config: &MemoryConfig,
) -> VectorMemoryBackend<V>
where
    V: aquifer::VectorStore,
{
    attach_configured_reranker_with(backend, config, || {
        aquifer::FastembedReranker::new()
            .map(|reranker| Arc::new(reranker) as Arc<dyn aquifer::Reranker>)
    })
}

fn attach_configured_reranker_with<V, F, E>(
    backend: VectorMemoryBackend<V>,
    config: &MemoryConfig,
    load_reranker: F,
) -> VectorMemoryBackend<V>
where
    V: aquifer::VectorStore,
    F: FnOnce() -> std::result::Result<Arc<dyn aquifer::Reranker>, E>,
    E: std::fmt::Display,
{
    if !config.rerank {
        return backend;
    }

    match load_reranker() {
        Ok(reranker) => backend.with_reranker(reranker),
        Err(error) => {
            eprintln!(
                "warning: neural rerank requested but Fastembed reranker failed to load; \
                 falling back to hybrid RRF: {error}"
            );
            backend
        }
    }
}

/// Box a vector backend, wrapping it in a semantic cache when one is configured.
fn finish_vector_backend<V: aquifer::VectorStore + Send + Sync + 'static>(
    backend: VectorMemoryBackend<V>,
    config: &MemoryConfig,
) -> Arc<dyn MemoryBackend> {
    match semantic_cache_from_config(config) {
        Some(cache) => Arc::new(backend.into_cached(cache)),
        None => Arc::new(backend),
    }
}

fn verifier_gate_from_config(config: &ArtesianConfig) -> VerifierGate {
    let verifiers = config
        .coordination
        .verifiers
        .iter()
        .map(|verifier| {
            Arc::new(
                CommandVerifier::new(verifier.name.clone(), verifier.command.clone())
                    .with_args(verifier.args.clone()),
            ) as Arc<dyn Verifier>
        })
        .collect();
    VerifierGate::new(verifiers)
}

fn process_agent_from_binding(
    config: &ArtesianConfig,
    role: Role,
    repo_root: &Path,
) -> Result<ProcessAgent> {
    let binding = config
        .agents
        .iter()
        .find(|binding| binding.role == role)
        .with_context(|| format!("missing agent binding for role {}", role.canonical_alias()))?;
    process_agent_from_binding_value(config, binding, repo_root)
}

fn process_agent_from_binding_value(
    config: &ArtesianConfig,
    binding: &AgentBinding,
    repo_root: &Path,
) -> Result<ProcessAgent> {
    let command = binding
        .command
        .clone()
        .unwrap_or_else(|| binding.agent.clone());
    let process_config = ProcessAgentConfig::new(command)
        .with_agent_id(binding.agent.clone())
        .with_default_model(binding.model.clone())
        .with_args(binding.args.clone())
        .with_working_dir(repo_root)
        .with_timeout(Duration::from_secs(binding.timeout_seconds.unwrap_or(120)))
        .with_registry_dir(spawn_registry_dir(config, repo_root))
        .with_max_concurrent_spawns(
            config
                .coordination
                .max_concurrent_spawns
                .unwrap_or(32)
                .max(1),
        )
        .with_max_lifetime(Duration::from_secs(
            config
                .coordination
                .spawn_max_lifetime_seconds
                .unwrap_or(30 * 60),
        ))
        .with_termination_grace(Duration::from_millis(
            config
                .coordination
                .spawn_shutdown_grace_millis
                .unwrap_or(2_000),
        ));
    Ok(ProcessAgent::new(process_config))
}

fn spawn_registry_dir(config: &ArtesianConfig, repo_root: &Path) -> PathBuf {
    config
        .coordination
        .spawn_registry_path
        .as_deref()
        .map(PathBuf::from)
        .map(|path| {
            if path.is_absolute() {
                path
            } else {
                repo_root.join(path)
            }
        })
        .unwrap_or_else(|| repo_root.join(".artesian").join("spawns"))
}

#[cfg(feature = "qdrant")]
fn open_qdrant_backend_inner(
    config: &MemoryConfig,
    relation_extraction: bool,
) -> Result<Arc<dyn MemoryBackend>> {
    let url = config
        .qdrant_url
        .clone()
        .or_else(|| env::var("QDRANT_URL").ok())
        .context("Qdrant backend requires qdrant_url in config or QDRANT_URL")?;
    let mut vector_config = QdrantVectorStoreConfig::new(url);
    vector_config.rest_url = config
        .qdrant_rest_url
        .clone()
        .or_else(|| env::var("QDRANT_REST_URL").ok());
    vector_config.api_key = config.resolve_qdrant_api_key();
    let store = QdrantVectorStore::connect(vector_config)?;
    let mem_config = vector_memory_config_from(config, relation_extraction);
    let backend = VectorMemoryBackend::new(store, mem_config)?;
    let backend = attach_configured_reranker(backend, config);
    Ok(finish_vector_backend(backend, config))
}

#[cfg(not(feature = "qdrant"))]
fn open_qdrant_backend_inner(
    _config: &MemoryConfig,
    _relation_extraction: bool,
) -> Result<Arc<dyn MemoryBackend>> {
    bail!("Qdrant backend requires building artesian-cli with the qdrant feature")
}

/// Build a Qdrant store config from the memory config (URLs + configured API key), for preflight.
#[cfg(feature = "qdrant")]
pub fn qdrant_config_from(config: &MemoryConfig) -> Result<QdrantVectorStoreConfig> {
    let url = config
        .qdrant_url
        .clone()
        .or_else(|| env::var("QDRANT_URL").ok())
        .context("Qdrant backend requires qdrant_url in config or QDRANT_URL")?;
    let mut vector_config = QdrantVectorStoreConfig::new(url);
    vector_config.rest_url = config
        .qdrant_rest_url
        .clone()
        .or_else(|| env::var("QDRANT_REST_URL").ok());
    vector_config.api_key = config.resolve_qdrant_api_key();
    Ok(vector_config)
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

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };

    use aquifer::{MemoryResult, SearchHit};
    use artesian_core::DEFAULT_RERANK_CANDIDATES;

    use super::*;

    struct TestEmbedder;

    impl aquifer::TextEmbedder for TestEmbedder {
        fn embed_query(&self, _text: &str) -> MemoryResult<Vec<f32>> {
            Ok(vec![0.0; aquifer::PINNED_FASTEMBED_DIMENSIONS])
        }

        fn embed_passage(&self, _text: &str) -> MemoryResult<Vec<f32>> {
            Ok(vec![0.0; aquifer::PINNED_FASTEMBED_DIMENSIONS])
        }
    }

    struct TestReranker;

    impl aquifer::Reranker for TestReranker {
        fn rerank(
            &self,
            _query: &str,
            mut hits: Vec<SearchHit>,
            limit: usize,
        ) -> MemoryResult<Vec<SearchHit>> {
            hits.truncate(limit);
            Ok(hits)
        }
    }

    fn memory_config(rerank: bool, rerank_candidates: usize) -> MemoryConfig {
        MemoryConfig {
            backend: MemoryBackendKind::SqliteVec,
            root: ".artesian".to_string(),
            collection: "artesian-memory".to_string(),
            qdrant_url: None,
            qdrant_rest_url: None,
            qdrant_api_key_env: None,
            qdrant_api_key_file: None,
            local_rerank_enabled: true,
            hyde_enabled: false,
            multi_query_enabled: false,
            debate_enabled: false,
            llm_consolidation_enabled: false,
            rerank,
            rerank_candidates,
            semantic_cache: Default::default(),
            track_access: true,
            track_savings: true,
        }
    }

    fn vector_backend(config: &MemoryConfig) -> VectorMemoryBackend<aquifer::SqliteVecVectorStore> {
        VectorMemoryBackend::with_embedder(
            aquifer::SqliteVecVectorStore::in_memory().expect("sqlite store"),
            vector_memory_config_from(config, false),
            Arc::new(TestEmbedder),
        )
        .expect("vector backend")
    }

    #[test]
    fn rerank_false_does_not_load_or_attach_reranker() {
        let config = memory_config(false, 0);
        let loader_called = AtomicBool::new(false);

        let backend = attach_configured_reranker_with(vector_backend(&config), &config, || {
            loader_called.store(true, Ordering::SeqCst);
            Ok::<Arc<dyn aquifer::Reranker>, &str>(
                Arc::new(TestReranker) as Arc<dyn aquifer::Reranker>
            )
        });

        assert!(!loader_called.load(Ordering::SeqCst));
        assert!(!backend.has_reranker());
        assert!(!backend.config().rerank);
        assert_eq!(backend.config().rerank_candidates, 0);
        assert!(!backend.rerank_active_for_limit(10));
    }

    #[test]
    fn rerank_true_uses_default_pool_and_attaches_mock_reranker() {
        let config = memory_config(true, 0);

        let backend = attach_configured_reranker_with(vector_backend(&config), &config, || {
            Ok::<Arc<dyn aquifer::Reranker>, &str>(
                Arc::new(TestReranker) as Arc<dyn aquifer::Reranker>
            )
        });

        assert!(backend.has_reranker());
        assert!(backend.config().rerank);
        assert_eq!(
            backend.config().rerank_candidates,
            DEFAULT_RERANK_CANDIDATES
        );
        assert!(backend.config().rerank_candidates > 10);
        assert!(backend.rerank_active_for_limit(10));
    }

    #[test]
    fn rerank_load_failure_falls_back_without_attachment() {
        let config = memory_config(true, 0);

        let backend = attach_configured_reranker_with(vector_backend(&config), &config, || {
            Err::<Arc<dyn aquifer::Reranker>, &str>("offline")
        });

        assert!(!backend.has_reranker());
        assert!(backend.config().rerank);
        assert_eq!(
            backend.config().rerank_candidates,
            DEFAULT_RERANK_CANDIDATES
        );
        assert!(!backend.rerank_active_for_limit(10));
    }
}
