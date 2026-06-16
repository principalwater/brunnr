// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};

use chrono::Utc;
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use futures_util::{future::BoxFuture, FutureExt};
use serde::{Deserialize, Serialize};

use crate::{
    chunking::{chunk_text, ChunkConfig},
    entity::EntityIndex,
    episode::EpisodeIndex,
    identity::stable_memory_id,
    reciprocal_rank_fusion,
    temporal::{apply_knowledge_supersession, apply_recency_decay},
    CollectionCompat, Distance, Filter, FilterCondition, FilterValue, MemoryBackend, MemoryError,
    MemoryId, MemoryQuery, MemoryRecord, MemoryResult, MemoryScope, MemoryTier, PayloadIndex,
    RrfOptions, SearchHit, SearchSource, SessionLaneLock, StoreMemory, VectorCollection,
    VectorPoint, VectorSearch, VectorSearchHit, VectorSearchSource, VectorStore, COMPAT_POINT_ID,
};

pub const PINNED_FASTEMBED_MODEL: &str = "intfloat/multilingual-e5-small";
pub const PINNED_FASTEMBED_DIMENSIONS: usize = 384;

pub trait TextEmbedder: Send + Sync {
    fn embed_query(&self, text: &str) -> MemoryResult<Vec<f32>>;

    fn embed_passage(&self, text: &str) -> MemoryResult<Vec<f32>>;
}

pub struct FastembedTextEmbedder {
    inner: Mutex<TextEmbedding>,
}

impl FastembedTextEmbedder {
    pub fn new() -> MemoryResult<Self> {
        let inner = TextEmbedding::try_new(
            TextInitOptions::new(EmbeddingModel::MultilingualE5Small)
                .with_show_download_progress(false),
        )
        .map_err(|error| MemoryError::BackendUnavailable(error.to_string()))?;
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    fn embed_prefixed(&self, prefix: &str, text: &str) -> MemoryResult<Vec<f32>> {
        let input = format!("{prefix}: {text}");
        let mut embedder = self
            .inner
            .lock()
            .map_err(|error| MemoryError::BackendUnavailable(error.to_string()))?;
        let mut embeddings = embedder
            .embed([input], None)
            .map_err(|error| MemoryError::BackendUnavailable(error.to_string()))?;
        embeddings.pop().ok_or_else(|| {
            MemoryError::BackendUnavailable("fastembed returned no embeddings".to_string())
        })
    }
}

impl TextEmbedder for FastembedTextEmbedder {
    fn embed_query(&self, text: &str) -> MemoryResult<Vec<f32>> {
        self.embed_prefixed("query", text)
    }

    fn embed_passage(&self, text: &str) -> MemoryResult<Vec<f32>> {
        self.embed_prefixed("passage", text)
    }
}

/// Configuration for a `VectorMemoryBackend`.
///
/// New retrieval signals default to **off** (`false` / `0.0`) pending measurement on your corpus.
/// Turn each signal on only after verifying a measurable recall improvement (see `brunnr-bench`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorMemoryConfig {
    pub collection: String,
    pub embedding_model: String,
    pub dimensions: usize,
    pub distance: Distance,

    /// D1: Add deterministic entity-overlap as a third RRF channel alongside BM25 and vector.
    /// Built from record tags, camelCase/PascalCase identifiers, backtick-quoted terms, and
    /// ALL-CAPS acronyms. No LLM required. Default off — enable after measuring recall gain.
    #[serde(default)]
    pub entity_linking: bool,

    /// D2: Multiply retrieval scores by `exp(−lambda × age_in_days)`.
    /// `0.0` disables decay. Suggested starting values: 0.005 (slow) – 0.02 (fast).
    #[serde(default)]
    pub temporal_decay_lambda: f32,

    /// D2: Downrank older records that share entities with a newer record in the same result set.
    /// Implements "knowledge-update supersession": the newer record wins; the older remains
    /// drillable via `node_id`. Default off — enable after measuring precision gain.
    #[serde(default)]
    pub knowledge_update_supersession: bool,

    /// D3: After retrieving top-k hits, expand each hit with up to `episode_context_window`
    /// additional records from the same embedding-based episode cluster.
    /// `0` disables episode expansion. Default off — enable after measuring recall gain.
    #[serde(default)]
    pub episode_context_window: usize,

    /// D3: Minimum cosine similarity to join an existing episode cluster (0.0–1.0).
    #[serde(default = "default_episode_threshold")]
    pub episode_similarity_threshold: f32,
}

fn default_episode_threshold() -> f32 {
    0.75
}

impl VectorMemoryConfig {
    pub fn new(collection: impl Into<String>) -> Self {
        Self {
            collection: collection.into(),
            embedding_model: PINNED_FASTEMBED_MODEL.to_string(),
            dimensions: PINNED_FASTEMBED_DIMENSIONS,
            distance: Distance::Cosine,
            entity_linking: false,
            temporal_decay_lambda: 0.0,
            knowledge_update_supersession: false,
            episode_context_window: 0,
            episode_similarity_threshold: 0.75,
        }
    }

    pub fn with_entity_linking(mut self, enabled: bool) -> Self {
        self.entity_linking = enabled;
        self
    }

    pub fn with_temporal_decay(mut self, lambda: f32) -> Self {
        self.temporal_decay_lambda = lambda;
        self
    }

    pub fn with_knowledge_update_supersession(mut self, enabled: bool) -> Self {
        self.knowledge_update_supersession = enabled;
        self
    }

    pub fn with_episode_context_window(mut self, window: usize) -> Self {
        self.episode_context_window = window;
        self
    }
}

pub struct VectorMemoryBackend<V: VectorStore> {
    store: V,
    config: VectorMemoryConfig,
    embedder: Arc<dyn TextEmbedder>,
    entity_index: Arc<Mutex<EntityIndex>>,
    episode_index: Arc<Mutex<EpisodeIndex>>,
}

impl<V: VectorStore> VectorMemoryBackend<V> {
    pub fn new(store: V, config: VectorMemoryConfig) -> MemoryResult<Self> {
        Self::with_embedder(store, config, Arc::new(FastembedTextEmbedder::new()?))
    }

    pub fn with_embedder(
        store: V,
        config: VectorMemoryConfig,
        embedder: Arc<dyn TextEmbedder>,
    ) -> MemoryResult<Self> {
        Ok(Self {
            store,
            config,
            embedder,
            entity_index: Arc::new(Mutex::new(EntityIndex::new())),
            episode_index: Arc::new(Mutex::new(EpisodeIndex::new())),
        })
    }

    pub fn vector_store(&self) -> &V {
        &self.store
    }

    pub fn config(&self) -> &VectorMemoryConfig {
        &self.config
    }

    async fn ensure_ready(&self) -> MemoryResult<()> {
        self.store
            .ensure_collection(VectorCollection {
                name: self.config.collection.clone(),
                dimensions: self.config.dimensions,
                distance: self.config.distance,
            })
            .await?;
        for field in [
            "node_id",
            "scope",
            "agent_id",
            "session_id",
            "task_id",
            "user_id",
        ] {
            self.store
                .ensure_payload_index(
                    &self.config.collection,
                    PayloadIndex {
                        field: field.to_string(),
                    },
                )
                .await?;
        }
        self.ensure_compat_metadata().await?;
        Ok(())
    }

    async fn ensure_compat_metadata(&self) -> MemoryResult<()> {
        let expected = CollectionCompat::from_config(&self.config);
        if let Some(point) = self
            .store
            .get(&self.config.collection, COMPAT_POINT_ID)
            .await?
        {
            let payload: CompatPayload = serde_json::from_value(point.payload)?;
            payload.compat.validate_compatible(&expected)?;
            return Ok(());
        }

        self.store
            .upsert(
                &self.config.collection,
                vec![VectorPoint {
                    id: COMPAT_POINT_ID.to_string(),
                    vector: vec![0.0; self.config.dimensions],
                    payload: serde_json::to_value(CompatPayload {
                        kind: compat_payload_kind(),
                        id: compat_point_id(),
                        node_id: compat_point_id(),
                        compat: expected,
                    })?,
                }],
            )
            .await
    }

    async fn vector_hits(&self, query: MemoryQuery) -> MemoryResult<Vec<SearchHit>> {
        self.ensure_ready().await?;
        let vector = self.embedder.embed_query(&query.text)?;
        let hits = self
            .store
            .search(
                &self.config.collection,
                VectorSearch {
                    vector: Some(vector),
                    text: None,
                    filter: filter_from_query(&query),
                    limit: query.limit,
                    source: VectorSearchSource::Vector,
                },
            )
            .await?;
        vector_hits_to_memory_hits(hits, SearchSource::Vector)
    }

    async fn keyword_hits(&self, query: MemoryQuery) -> MemoryResult<Vec<SearchHit>> {
        self.ensure_ready().await?;
        let hits = self
            .store
            .search(
                &self.config.collection,
                VectorSearch {
                    vector: None,
                    text: Some(query.text.clone()),
                    filter: filter_from_query(&query),
                    limit: query.limit,
                    source: VectorSearchSource::Keyword,
                },
            )
            .await?;
        vector_hits_to_memory_hits(hits, SearchSource::Keyword)
    }

    /// Expand hits by fetching episode-mate records up to `window` mates per hit.
    async fn expand_episode_context(
        &self,
        hits: Vec<SearchHit>,
        window: usize,
    ) -> MemoryResult<Vec<SearchHit>> {
        let mut result = hits;
        let mut seen: std::collections::BTreeSet<String> =
            result.iter().map(|h| h.record.node_id.clone()).collect();

        let mates: Vec<String> = {
            let guard = self
                .episode_index
                .lock()
                .map_err(|e| MemoryError::Database(e.to_string()))?;
            result
                .iter()
                .flat_map(|hit| guard.episode_mates(&hit.record.node_id))
                .filter(|mate_id| !seen.contains(mate_id.as_str()))
                .take(window * result.len().max(1))
                .collect()
        };

        for mate_id in mates {
            if seen.contains(&mate_id) {
                continue;
            }
            if let Some(record) = self.get_node(&mate_id).await? {
                seen.insert(mate_id);
                result.push(SearchHit {
                    score: 0.01,
                    record,
                    source: SearchSource::Keyword,
                });
            }
        }
        Ok(result)
    }
}

impl<V: VectorStore> MemoryBackend for VectorMemoryBackend<V> {
    fn find(&self, query: MemoryQuery) -> BoxFuture<'_, MemoryResult<Vec<SearchHit>>> {
        async move {
            let options = RrfOptions {
                limit: query.limit,
                ..RrfOptions::default()
            };

            let signals_active = self.config.entity_linking
                || self.config.temporal_decay_lambda > 0.0
                || self.config.knowledge_update_supersession
                || self.config.episode_context_window > 0;

            if !signals_active {
                // Fast path: existing 2-channel hybrid RRF (uses server-side if available).
                return self.hybrid_rrf(query.clone(), query, options).await;
            }

            // Signal path: client-side multi-channel RRF + post-processing.
            self.ensure_ready().await?;
            let mut channels: Vec<Vec<SearchHit>> = vec![
                self.keyword_hits(query.clone()).await?,
                self.vector_hits(query.clone()).await?,
            ];

            if self.config.entity_linking {
                let guard = self
                    .entity_index
                    .lock()
                    .map_err(|e| MemoryError::Database(e.to_string()))?;
                let entity_hits = guard.entity_hits(&query.text, query.limit);
                if !entity_hits.is_empty() {
                    channels.push(entity_hits);
                }
            }

            let mut hits = reciprocal_rank_fusion(&channels, options);

            if self.config.temporal_decay_lambda > 0.0 {
                hits = apply_recency_decay(hits, self.config.temporal_decay_lambda);
            }

            if self.config.knowledge_update_supersession {
                let guard = self
                    .entity_index
                    .lock()
                    .map_err(|e| MemoryError::Database(e.to_string()))?;
                hits = apply_knowledge_supersession(hits, &guard, 0.3);
            }

            if self.config.episode_context_window > 0 {
                hits = self
                    .expand_episode_context(hits, self.config.episode_context_window)
                    .await?;
                hits.truncate(query.limit);
            }

            Ok(hits)
        }
        .boxed()
    }

    fn store(&self, memory: StoreMemory) -> BoxFuture<'_, MemoryResult<MemoryRecord>> {
        async move {
            let _lane_guard = SessionLaneLock::default_rooted()
                .acquire(&self.config.collection, memory.session_id.as_deref())
                .await?;
            self.ensure_ready().await?;

            // Chunk by default so retrieval returns bounded, coherent slices (top-k
            // chunks) instead of whole records. Small content yields a single chunk
            // (unchanged behavior); large content is split with parent linkage so the
            // full record stays reachable via the parent node_id.
            let base_id = stable_memory_id(&memory);
            let base_node = memory
                .node_id
                .clone()
                .unwrap_or_else(|| format!("node:{base_id}"));
            let created_at = memory.created_at.unwrap_or_else(Utc::now);
            let chunks = chunk_text(&memory.content, &ChunkConfig::default());
            let single = chunks.len() == 1;
            let chunk_count = chunks.len();

            let mut representative: Option<MemoryRecord> = None;
            for chunk in chunks {
                let mut metadata = memory.metadata.clone();
                let node_id = if single {
                    base_node.clone()
                } else {
                    metadata.insert("parent_node".to_string(), base_node.clone());
                    metadata.insert("chunk_index".to_string(), chunk.index.to_string());
                    metadata.insert("chunk_count".to_string(), chunk_count.to_string());
                    if let Some(heading) = &chunk.heading {
                        metadata.insert("heading".to_string(), heading.clone());
                    }
                    format!("{base_node}#chunk-{}", chunk.index)
                };
                let chunk_memory = StoreMemory {
                    content: chunk.content.clone(),
                    tags: memory.tags.clone(),
                    metadata: metadata.clone(),
                    tier: memory.tier,
                    node_id: Some(node_id.clone()),
                    created_at: Some(created_at),
                    scope: memory.scope,
                    agent_id: memory.agent_id.clone(),
                    session_id: memory.session_id.clone(),
                    task_id: memory.task_id.clone(),
                    user_id: memory.user_id.clone(),
                };
                // Single-chunk content keeps the original id (= stable id of the whole
                // memory) so existing idempotency/dedup is unchanged; multi-chunk records
                // each get their own content-addressed id.
                let id = if single {
                    base_id.clone()
                } else {
                    stable_memory_id(&chunk_memory)
                };
                if let Some(existing) = self.store.get(&self.config.collection, id.as_str()).await? {
                    if representative.is_none() {
                        representative = Some(point_to_record(existing)?);
                    }
                    continue;
                }
                let record = MemoryRecord {
                    id,
                    node_id,
                    content: chunk.content,
                    tags: memory.tags.clone(),
                    metadata,
                    tier: memory.tier,
                    created_at,
                    scope: memory.scope,
                    agent_id: memory.agent_id.clone(),
                    session_id: memory.session_id.clone(),
                    task_id: memory.task_id.clone(),
                    user_id: memory.user_id.clone(),
                };
                let vector = self.embedder.embed_passage(&record.content)?;
                self.store
                    .upsert(
                        &self.config.collection,
                        vec![VectorPoint {
                            id: record.id.to_string(),
                            vector: vector.clone(),
                            payload: serde_json::to_value(MemoryPayload::from(&record))?,
                        }],
                    )
                    .await?;

                // Update session-local indexes per chunk.
                if self.config.entity_linking {
                    self.entity_index
                        .lock()
                        .map_err(|e| MemoryError::Database(e.to_string()))?
                        .index_record(record.clone());
                }
                if self.config.episode_context_window > 0 {
                    self.episode_index
                        .lock()
                        .map_err(|e| MemoryError::Database(e.to_string()))?
                        .add_record(
                            &record.node_id,
                            &vector,
                            self.config.episode_similarity_threshold,
                        );
                }
                if representative.is_none() {
                    representative = Some(record);
                }
            }
            representative
                .ok_or_else(|| MemoryError::Database("chunking produced no records".to_string()))
        }
        .boxed()
    }

    fn hybrid_rrf(
        &self,
        keyword_query: MemoryQuery,
        vector_query: MemoryQuery,
        options: RrfOptions,
    ) -> BoxFuture<'_, MemoryResult<Vec<SearchHit>>> {
        async move {
            self.ensure_ready().await?;
            if self.store.capabilities().supports_server_side_hybrid {
                let vector = self.embedder.embed_query(&vector_query.text)?;
                let hits = self
                    .store
                    .search(
                        &self.config.collection,
                        VectorSearch {
                            vector: Some(vector),
                            text: Some(keyword_query.text),
                            filter: filter_from_query(&vector_query),
                            limit: options.limit,
                            source: VectorSearchSource::Hybrid,
                        },
                    )
                    .await?;
                return vector_hits_to_memory_hits(hits, SearchSource::Hybrid);
            }

            let keyword_hits = self.keyword_hits(keyword_query).await?;
            let vector_hits = self.vector_hits(vector_query).await?;
            Ok(reciprocal_rank_fusion(
                &[keyword_hits, vector_hits],
                options,
            ))
        }
        .boxed()
    }

    fn get_node(&self, node_id: &str) -> BoxFuture<'_, MemoryResult<Option<MemoryRecord>>> {
        let node_id = node_id.to_string();
        async move {
            self.ensure_ready().await?;
            if let Some(point) = self.store.get(&self.config.collection, &node_id).await? {
                return point_to_record(point).map(Some);
            }
            let mut hits = self
                .store
                .search(
                    &self.config.collection,
                    VectorSearch {
                        vector: None,
                        text: None,
                        filter: Filter::node_id(node_id),
                        limit: 1,
                        source: VectorSearchSource::Keyword,
                    },
                )
                .await?;
            hits.pop().map(|hit| point_to_record(hit.point)).transpose()
        }
        .boxed()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MemoryPayload {
    id: MemoryId,
    node_id: String,
    content: String,
    tags: Vec<String>,
    metadata: std::collections::BTreeMap<String, String>,
    tier: MemoryTier,
    created_at: chrono::DateTime<Utc>,
    #[serde(default)]
    scope: Option<MemoryScope>,
    #[serde(default)]
    agent_id: Option<String>,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    task_id: Option<String>,
    #[serde(default)]
    user_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompatPayload {
    #[serde(default = "compat_payload_kind")]
    kind: String,
    #[serde(default = "compat_point_id")]
    id: String,
    #[serde(default = "compat_point_id")]
    node_id: String,
    #[serde(flatten)]
    compat: CollectionCompat,
}

fn compat_payload_kind() -> String {
    "brunnr.compat".to_string()
}

fn compat_point_id() -> String {
    COMPAT_POINT_ID.to_string()
}

impl From<&MemoryRecord> for MemoryPayload {
    fn from(record: &MemoryRecord) -> Self {
        Self {
            id: record.id.clone(),
            node_id: record.node_id.clone(),
            content: record.content.clone(),
            tags: record.tags.clone(),
            metadata: record.metadata.clone(),
            tier: record.tier,
            created_at: record.created_at,
            scope: record.scope,
            agent_id: record.agent_id.clone(),
            session_id: record.session_id.clone(),
            task_id: record.task_id.clone(),
            user_id: record.user_id.clone(),
        }
    }
}

impl From<MemoryPayload> for MemoryRecord {
    fn from(payload: MemoryPayload) -> Self {
        Self {
            id: payload.id,
            node_id: payload.node_id,
            content: payload.content,
            tags: payload.tags,
            metadata: payload.metadata,
            tier: payload.tier,
            created_at: payload.created_at,
            scope: payload.scope,
            agent_id: payload.agent_id,
            session_id: payload.session_id,
            task_id: payload.task_id,
            user_id: payload.user_id,
        }
    }
}

fn filter_from_query(query: &MemoryQuery) -> Filter {
    let mut filter = query
        .node_id
        .as_ref()
        .map_or_else(Filter::default, Filter::node_id);
    filter.must_not.push(FilterCondition::Eq {
        field: "node_id".to_string(),
        value: FilterValue::String(COMPAT_POINT_ID.to_string()),
    });
    if let Some(scope) = query.scope {
        filter.must_eq("scope", scope.as_str());
    }
    if let Some(agent_id) = &query.agent_id {
        filter.must_eq("agent_id", agent_id);
    }
    if let Some(session_id) = &query.session_id {
        filter.must_eq("session_id", session_id);
    }
    if let Some(task_id) = &query.task_id {
        filter.must_eq("task_id", task_id);
    }
    if let Some(user_id) = &query.user_id {
        filter.must_eq("user_id", user_id);
    }
    filter
}

fn vector_hits_to_memory_hits(
    hits: Vec<VectorSearchHit>,
    source: SearchSource,
) -> MemoryResult<Vec<SearchHit>> {
    hits.into_iter()
        .map(|hit| {
            Ok(SearchHit {
                record: point_to_record(hit.point)?,
                score: hit.score,
                source,
            })
        })
        .collect()
}

fn point_to_record(point: VectorPoint) -> MemoryResult<MemoryRecord> {
    let payload: MemoryPayload = serde_json::from_value(point.payload)?;
    Ok(MemoryRecord::from(payload))
}
