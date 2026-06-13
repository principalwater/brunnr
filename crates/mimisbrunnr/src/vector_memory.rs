// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};

use chrono::Utc;
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use futures_util::{future::BoxFuture, FutureExt};
use serde::{Deserialize, Serialize};

use crate::{
    identity::stable_memory_id, reciprocal_rank_fusion, Distance, Filter, MemoryBackend,
    MemoryError, MemoryId, MemoryQuery, MemoryRecord, MemoryResult, MemoryTier, PayloadIndex,
    RrfOptions, SearchHit, SearchSource, StoreMemory, VectorCollection, VectorPoint, VectorSearch,
    VectorSearchHit, VectorSearchSource, VectorStore,
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

#[derive(Debug, Clone)]
pub struct VectorMemoryConfig {
    pub collection: String,
    pub dimensions: usize,
    pub distance: Distance,
}

impl VectorMemoryConfig {
    pub fn new(collection: impl Into<String>) -> Self {
        Self {
            collection: collection.into(),
            dimensions: PINNED_FASTEMBED_DIMENSIONS,
            distance: Distance::Cosine,
        }
    }
}

pub struct VectorMemoryBackend<V: VectorStore> {
    store: V,
    config: VectorMemoryConfig,
    embedder: Arc<dyn TextEmbedder>,
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
        self.store
            .ensure_payload_index(
                &self.config.collection,
                PayloadIndex {
                    field: "node_id".to_string(),
                },
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
}

impl<V: VectorStore> MemoryBackend for VectorMemoryBackend<V> {
    fn find(&self, query: MemoryQuery) -> BoxFuture<'_, MemoryResult<Vec<SearchHit>>> {
        async move {
            let options = RrfOptions {
                limit: query.limit,
                ..RrfOptions::default()
            };
            self.hybrid_rrf(query.clone(), query, options).await
        }
        .boxed()
    }

    fn store(&self, memory: StoreMemory) -> BoxFuture<'_, MemoryResult<MemoryRecord>> {
        async move {
            self.ensure_ready().await?;
            let id = stable_memory_id(&memory);
            if let Some(existing) = self.store.get(&self.config.collection, id.as_str()).await? {
                return point_to_record(existing);
            }

            let node_id = memory.node_id.unwrap_or_else(|| format!("node:{id}"));
            let record = MemoryRecord {
                id,
                node_id,
                content: memory.content,
                tags: memory.tags,
                metadata: memory.metadata,
                tier: memory.tier,
                created_at: memory.created_at.unwrap_or_else(Utc::now),
            };
            let vector = self.embedder.embed_passage(&record.content)?;
            self.store
                .upsert(
                    &self.config.collection,
                    vec![VectorPoint {
                        id: record.id.to_string(),
                        vector,
                        payload: serde_json::to_value(MemoryPayload::from(&record))?,
                    }],
                )
                .await?;
            Ok(record)
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

            let keyword_hits = self.keyword_hits(keyword_query).await.unwrap_or_default();
            let vector_hits = self.vector_hits(vector_query).await.unwrap_or_default();
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
        }
    }
}

fn filter_from_query(query: &MemoryQuery) -> Filter {
    query
        .node_id
        .as_ref()
        .map_or_else(Filter::default, Filter::node_id)
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
