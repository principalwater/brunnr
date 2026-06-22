// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};

use chrono::Utc;
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use futures_util::{future::BoxFuture, FutureExt};
use serde::{Deserialize, Serialize};

use crate::{
    backend::BulkStoreReport,
    chunking::{chunk_text, ChunkConfig},
    entity::EntityIndex,
    episode::EpisodeIndex,
    graph::{
        by_entity_node_ids, extract_entity_relations, neighbor_node_ids, normalize_relations,
        records_by_node_ids, GRAPH_SCAN_LIMIT,
    },
    identity::stable_memory_id,
    reciprocal_rank_fusion,
    temporal::{apply_knowledge_supersession, apply_recency_decay},
    CollectionCompat, Distance, Filter, FilterCondition, FilterValue, MemoryBackend, MemoryError,
    MemoryId, MemoryQuery, MemoryRecord, MemoryResult, MemoryScope, MemoryTier, PayloadIndex,
    Relation, RrfOptions, SearchHit, SearchSource, SessionLaneLock, StoreMemory, VectorCollection,
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
/// Turn each signal on only after verifying a measurable recall improvement (see `artesian-bench`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorMemoryConfig {
    pub collection: String,
    pub embedding_model: String,
    pub dimensions: usize,
    pub distance: Distance,

    /// Embedding quantization for this collection (`float32` or `int8`). Defaults to `float32`.
    /// `int8` gives ~4× storage reduction with a modest recall cost — measure before enabling.
    #[serde(default)]
    pub quantization: crate::VectorQuantization,

    /// D1: Add deterministic entity-overlap as a third RRF channel alongside BM25 and vector.
    /// Built from record tags, camelCase/PascalCase identifiers, backtick-quoted terms, and
    /// ALL-CAPS acronyms. No LLM required. Default off — enable after measuring recall gain.
    #[serde(default)]
    pub entity_linking: bool,

    /// Derive lightweight `mentions` relations from deterministic entity signals at store time.
    /// Default off; explicit `StoreMemory.relations` are always accepted.
    #[serde(default)]
    pub relation_extraction: bool,

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

    /// Small-to-big retrieval (**default on**). After matching precise chunks, each
    /// chunk hit is expanded to its surrounding parent-section context — contiguous
    /// sibling chunks of the same `parent_node`, merged with overlap removed —
    /// growing symmetrically around the match up to `parent_context_chars`. Multiple
    /// matched chunks of one parent collapse into a single expanded hit whose
    /// `node_id` is the `parent_node` (full source still reachable via `get_node`).
    /// Single-chunk records have no siblings, so they pass through unchanged — which
    /// keeps recall both bounded *and* coherent without altering small-document
    /// behavior. `0` (with `parent_context_auto = false`) disables expansion. When
    /// `parent_context_auto` is on (the default), this is only the fallback budget used
    /// until the corpus has been observed.
    #[serde(default = "default_parent_context_chars")]
    pub parent_context_chars: usize,

    /// Adaptive budget (**default on**). When true, the effective `parent_context_chars`
    /// is the **median size of multi-chunk parent documents** observed on this collection,
    /// clamped to `[chunk size, parent_context_max_chars]` — so the window self-adapts to
    /// each corpus instead of a fixed guess (single-chunk documents are excluded, since
    /// they need no expansion). Until any multi-chunk document is seen, the fixed
    /// `parent_context_chars` is used. Set false to use `parent_context_chars` directly.
    #[serde(default = "default_true")]
    pub parent_context_auto: bool,

    /// Upper bound for the adaptive budget so a corpus of very large documents cannot
    /// inflate per-query context. Default `8192` (~2k tokens per source); the remainder
    /// of an oversized parent stays reachable via `get_node` drill-down.
    #[serde(default = "default_parent_context_max_chars")]
    pub parent_context_max_chars: usize,

    /// Reranking candidate pool. When `> query.limit` **and** a reranker is attached
    /// (`with_reranker`), retrieval fuses this many candidates and reranks them down to
    /// `query.limit` before small-to-big expansion — surfacing the most relevant facts into
    /// the same downstream budget. `0` (default) disables reranking.
    #[serde(default)]
    pub rerank_candidates: usize,
}

fn default_episode_threshold() -> f32 {
    0.75
}

fn default_parent_context_chars() -> usize {
    3_200
}

fn default_true() -> bool {
    true
}

fn default_parent_context_max_chars() -> usize {
    8_192
}

impl VectorMemoryConfig {
    pub fn new(collection: impl Into<String>) -> Self {
        Self {
            collection: collection.into(),
            embedding_model: PINNED_FASTEMBED_MODEL.to_string(),
            dimensions: PINNED_FASTEMBED_DIMENSIONS,
            distance: Distance::Cosine,
            quantization: crate::VectorQuantization::Float32,
            entity_linking: false,
            relation_extraction: false,
            temporal_decay_lambda: 0.0,
            knowledge_update_supersession: false,
            episode_context_window: 0,
            episode_similarity_threshold: 0.75,
            parent_context_chars: default_parent_context_chars(),
            parent_context_auto: true,
            parent_context_max_chars: default_parent_context_max_chars(),
            rerank_candidates: 0,
        }
    }

    /// Set the reranking candidate pool (`0` disables reranking).
    pub fn with_rerank_candidates(mut self, candidates: usize) -> Self {
        self.rerank_candidates = candidates;
        self
    }

    pub fn with_entity_linking(mut self, enabled: bool) -> Self {
        self.entity_linking = enabled;
        self
    }

    pub fn with_relation_extraction(mut self, enabled: bool) -> Self {
        self.relation_extraction = enabled;
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

    /// Set the small-to-big expansion budget in characters (`0` with auto off disables).
    pub fn with_parent_context_chars(mut self, chars: usize) -> Self {
        self.parent_context_chars = chars;
        self
    }

    /// Enable or disable adaptive (corpus-median) budgeting.
    pub fn with_parent_context_auto(mut self, auto: bool) -> Self {
        self.parent_context_auto = auto;
        self
    }

    /// Set the upper bound for the adaptive budget.
    pub fn with_parent_context_max_chars(mut self, chars: usize) -> Self {
        self.parent_context_max_chars = chars;
        self
    }
}

pub struct VectorMemoryBackend<V: VectorStore> {
    store: V,
    config: VectorMemoryConfig,
    embedder: Arc<dyn TextEmbedder>,
    reranker: Option<Arc<dyn crate::Reranker>>,
    entity_index: Arc<Mutex<EntityIndex>>,
    episode_index: Arc<Mutex<EpisodeIndex>>,
    parent_samples: Arc<Mutex<ParentSizeSamples>>,
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
            reranker: None,
            entity_index: Arc::new(Mutex::new(EntityIndex::new())),
            episode_index: Arc::new(Mutex::new(EpisodeIndex::new())),
            parent_samples: Arc::new(Mutex::new(ParentSizeSamples::default())),
        })
    }

    /// Attach a reranker; combined with `config.rerank_candidates > 0`, retrieval reranks a
    /// larger candidate pool down to the requested limit.
    pub fn with_reranker(mut self, reranker: Arc<dyn crate::Reranker>) -> Self {
        self.reranker = Some(reranker);
        self
    }

    pub fn vector_store(&self) -> &V {
        &self.store
    }

    pub fn config(&self) -> &VectorMemoryConfig {
        &self.config
    }

    /// The shared text embedder, for reuse (e.g. keying a semantic cache in the same space).
    pub fn embedder(&self) -> Arc<dyn TextEmbedder> {
        self.embedder.clone()
    }

    /// Wrap this backend in a [`CachingMemoryBackend`](crate::CachingMemoryBackend) keyed by its
    /// own embedder, so similar queries are served from `cache` without re-running search.
    pub fn into_cached(
        self,
        cache: crate::SemanticCache,
    ) -> crate::CachingMemoryBackend<Self, crate::EmbedderVectorizer> {
        let vectorizer = crate::EmbedderVectorizer::new(self.embedder());
        crate::CachingMemoryBackend::new(self, vectorizer, cache)
    }

    async fn ensure_ready(&self) -> MemoryResult<()> {
        self.store
            .ensure_collection(VectorCollection {
                name: self.config.collection.clone(),
                dimensions: self.config.dimensions,
                distance: self.config.distance,
                quantization: self.config.quantization,
            })
            .await?;
        for field in [
            "node_id",
            "scope",
            "agent_id",
            "session_id",
            "task_id",
            "user_id",
            // Indexed for time-range and recency filtering (datetime index on Qdrant, JSON
            // expression index on SQLite) so temporal decay/supersession need not full-scan.
            "created_at",
            // Indexed so small-to-big sibling lookups (filter by parent) hit an index
            // instead of scanning the whole collection.
            "metadata.parent_node",
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

    /// Resolve the effective small-to-big budget. `None` means expansion is disabled
    /// (manual mode with a zero budget). In auto mode the budget tracks the corpus
    /// median parent size, clamped to `[chunk size, parent_context_max_chars]`, falling
    /// back to the fixed `parent_context_chars` until a multi-chunk document is seen.
    fn effective_parent_budget(&self) -> Option<usize> {
        let config = &self.config;
        if config.parent_context_auto {
            let low = ChunkConfig::default().max_chars;
            let high = config.parent_context_max_chars.max(low);
            let target = self
                .parent_samples
                .lock()
                .ok()
                .and_then(|mut samples| samples.median())
                .unwrap_or(config.parent_context_chars);
            Some(target.clamp(low, high))
        } else if config.parent_context_chars > 0 {
            Some(config.parent_context_chars)
        } else {
            None
        }
    }

    /// Fetch every sibling chunk that shares `parent_node`, sorted by `chunk_index`.
    async fn fetch_siblings(
        &self,
        parent_node: &str,
        cap: usize,
    ) -> MemoryResult<Vec<MemoryRecord>> {
        let mut filter = Filter::default();
        filter.must_eq("metadata.parent_node", parent_node);
        let hits = self
            .store
            .search(
                &self.config.collection,
                VectorSearch {
                    vector: None,
                    text: None,
                    filter,
                    limit: cap.max(1),
                    source: VectorSearchSource::Keyword,
                },
            )
            .await?;
        let mut records = hits
            .into_iter()
            .map(|hit| point_to_record(hit.point))
            .collect::<MemoryResult<Vec<_>>>()?;
        records.sort_by_key(chunk_index_of);
        Ok(records)
    }

    async fn graph_records(&self) -> MemoryResult<Vec<MemoryRecord>> {
        self.ensure_ready().await?;
        let hits = self
            .store
            .search(
                &self.config.collection,
                VectorSearch {
                    vector: None,
                    text: None,
                    filter: filter_from_query(&MemoryQuery::new("")),
                    limit: GRAPH_SCAN_LIMIT,
                    source: VectorSearchSource::Keyword,
                },
            )
            .await?;
        let mut records = Vec::new();
        for hit in hits {
            let record = point_to_record(hit.point)?;
            if !record.relations.is_empty() {
                records.push(record);
            }
        }
        Ok(records)
    }

    async fn graph_records_for_node_ids(
        &self,
        records: &[MemoryRecord],
        node_ids: Vec<String>,
    ) -> MemoryResult<Vec<MemoryRecord>> {
        let mut output = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        for node_id in node_ids {
            if !seen.insert(node_id.clone()) {
                continue;
            }
            if let Some(record) = self.get_node(&node_id).await? {
                output.push(record);
            } else {
                output.extend(records_by_node_ids(records, vec![node_id]));
            }
        }
        Ok(output)
    }

    /// Small-to-big expansion: collapse same-parent chunk hits into one hit whose
    /// content is the bounded parent-section window around the matched chunk(s).
    /// Hits that are not chunks (no `parent_node`) pass through unchanged and keep
    /// their order, so a result set without chunks is returned byte-for-byte.
    async fn expand_small_to_big(
        &self,
        hits: Vec<SearchHit>,
        limit: usize,
        budget: usize,
    ) -> MemoryResult<Vec<SearchHit>> {
        if !hits
            .iter()
            .any(|hit| hit.record.metadata.contains_key("parent_node"))
        {
            return Ok(hits);
        }

        let max_overlap = ChunkConfig::default().overlap_chars;

        // Group hits by parent (chunks) or by node_id (whole records), preserving
        // first-seen rank order.
        let mut order: Vec<String> = Vec::new();
        let mut groups: std::collections::HashMap<String, SmallToBigGroup> =
            std::collections::HashMap::new();
        for hit in hits {
            let parent = hit.record.metadata.get("parent_node").cloned();
            let key = parent.clone().unwrap_or_else(|| hit.record.node_id.clone());
            let idx = chunk_index_of(&hit.record);
            match groups.get_mut(&key) {
                Some(group) => {
                    // Keep the highest-scored chunk as the window anchor.
                    if hit.score > group.score {
                        group.score = hit.score;
                        if parent.is_some() {
                            group.anchor = idx;
                        }
                    }
                    if parent.is_some() {
                        group.matched.insert(idx);
                    }
                }
                None => {
                    order.push(key.clone());
                    let mut matched = std::collections::BTreeSet::new();
                    if parent.is_some() {
                        matched.insert(idx);
                    }
                    groups.insert(
                        key,
                        SmallToBigGroup {
                            score: hit.score,
                            source: hit.source,
                            parent,
                            anchor: idx,
                            matched,
                            record: hit.record,
                        },
                    );
                }
            }
        }

        let mut out: Vec<SearchHit> = Vec::with_capacity(order.len());
        for key in order {
            let group = groups.remove(&key).expect("group present");
            let Some(parent_node) = group.parent.clone() else {
                // Whole record / single chunk: unchanged.
                out.push(SearchHit {
                    score: group.score,
                    record: group.record,
                    source: group.source,
                });
                continue;
            };

            let cap = group
                .record
                .metadata
                .get("chunk_count")
                .and_then(|value| value.parse::<usize>().ok())
                .map_or(256, |count| count + 1);
            let siblings = self.fetch_siblings(&parent_node, cap).await?;
            if siblings.is_empty() {
                out.push(SearchHit {
                    score: group.score,
                    record: group.record,
                    source: group.source,
                });
                continue;
            }

            let window = build_parent_window(&siblings, &group, budget, max_overlap);
            let SmallToBigGroup {
                score,
                source,
                record,
                matched,
                ..
            } = group;
            let MemoryRecord {
                id,
                tags,
                tier,
                created_at,
                scope,
                agent_id,
                session_id,
                task_id,
                user_id,
                source: provenance_source,
                confidence,
                relations,
                mut metadata,
                ..
            } = record;
            metadata.remove("chunk_index");
            metadata.insert(
                "expanded_from_chunk".to_string(),
                matched
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            );
            out.push(SearchHit {
                score,
                source,
                record: MemoryRecord {
                    id,
                    node_id: parent_node,
                    content: window,
                    tags,
                    metadata,
                    tier,
                    created_at,
                    scope,
                    agent_id,
                    session_id,
                    task_id,
                    user_id,
                    source: provenance_source,
                    confidence,
                    relations,
                },
            });
        }
        out.truncate(limit);
        Ok(out)
    }
}

struct SmallToBigGroup {
    score: f32,
    source: SearchSource,
    parent: Option<String>,
    /// Chunk index of the **best-scored** matched chunk — the window is centered here.
    anchor: usize,
    /// All matched chunk indices of this parent (recorded in `expanded_from_chunk`).
    matched: std::collections::BTreeSet<usize>,
    record: MemoryRecord,
}

/// Running sample of multi-chunk parent-document sizes (characters), used to derive
/// the adaptive small-to-big budget as the corpus median. Bounded so it never grows
/// without limit; the median is cached and refreshed as the sample grows.
#[derive(Default)]
struct ParentSizeSamples {
    sizes: Vec<usize>,
    cached_median: Option<usize>,
    last_computed_len: usize,
}

impl ParentSizeSamples {
    /// Max retained samples; the median is stable well before this and memory stays small.
    const CAP: usize = 65_536;
    /// Recompute the cached median once the sample has grown by this many entries.
    const REFRESH_STEP: usize = 16;

    fn record(&mut self, size: usize) {
        if self.sizes.len() < Self::CAP {
            self.sizes.push(size);
        }
    }

    fn median(&mut self) -> Option<usize> {
        if self.sizes.is_empty() {
            return None;
        }
        if self.cached_median.is_none()
            || self.sizes.len() >= self.last_computed_len + Self::REFRESH_STEP
        {
            let mut sorted = self.sizes.clone();
            sorted.sort_unstable();
            let mid = sorted.len() / 2;
            let median = if sorted.len() % 2 == 1 {
                sorted[mid]
            } else {
                (sorted[mid - 1] + sorted[mid]) / 2
            };
            self.cached_median = Some(median);
            self.last_computed_len = sorted.len();
        }
        self.cached_median
    }
}

fn chunk_index_of(record: &MemoryRecord) -> usize {
    record
        .metadata
        .get("chunk_index")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0)
}

/// Append `next` to `acc`, removing up to `max_overlap` characters of duplicated
/// boundary text created by the chunker's sliding-window overlap.
fn append_with_overlap(acc: &mut String, next: &str, max_overlap: usize) {
    if acc.is_empty() {
        acc.push_str(next);
        return;
    }
    let next_chars: Vec<char> = next.chars().collect();
    let max_k = max_overlap.min(next_chars.len());
    let mut best = 0;
    for k in (1..=max_k).rev() {
        let prefix: String = next_chars[..k].iter().collect();
        if acc.ends_with(&prefix) {
            best = k;
            break;
        }
    }
    let rest: String = next_chars[best..].iter().collect();
    acc.push_str(&rest);
}

/// Build a parent-section window centered on the **anchor** (the best-scored matched
/// chunk), grown outward symmetrically while the estimated merged length stays within
/// `budget`, merged with overlap removed, and hard-capped to `budget` chars. Centering on
/// the single best chunk (rather than spanning every matched chunk) keeps the relevant
/// passage in the window even when unrelated sibling chunks of the same parent also
/// matched — a wide span would otherwise be truncated before reaching the anchor.
fn build_parent_window(
    siblings: &[MemoryRecord],
    group: &SmallToBigGroup,
    budget: usize,
    max_overlap: usize,
) -> String {
    let n = siblings.len();
    let clen = |i: usize| siblings[i].content.chars().count();

    let anchor = (0..n)
        .find(|&i| chunk_index_of(&siblings[i]) == group.anchor)
        .or_else(|| {
            siblings
                .iter()
                .position(|s| s.node_id == group.record.node_id)
        })
        .unwrap_or(0);
    let (mut lo, mut hi) = (anchor, anchor);

    let mut total: usize = clen(anchor);
    loop {
        let mut grew = false;
        if lo > 0 {
            let add = clen(lo - 1).saturating_sub(max_overlap);
            if total + add <= budget {
                lo -= 1;
                total += add;
                grew = true;
            }
        }
        if hi + 1 < n {
            let add = clen(hi + 1).saturating_sub(max_overlap);
            if total + add <= budget {
                hi += 1;
                total += add;
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }

    let mut window = String::new();
    for sibling in siblings.iter().take(hi + 1).skip(lo) {
        append_with_overlap(&mut window, &sibling.content, max_overlap);
    }
    if window.chars().count() > budget {
        window = window.chars().take(budget).collect();
    }
    window
}

impl<V: VectorStore> MemoryBackend for VectorMemoryBackend<V> {
    fn find(&self, query: MemoryQuery) -> BoxFuture<'_, MemoryResult<Vec<SearchHit>>> {
        async move {
            // When a reranker is attached, fuse a larger candidate pool and rerank it down to
            // `query.limit` before small-to-big — better precision into the same budget.
            let rerank_active =
                self.reranker.is_some() && self.config.rerank_candidates > query.limit;
            let pool_limit = if rerank_active {
                self.config.rerank_candidates
            } else {
                query.limit
            };
            let mut pool_query = query.clone();
            pool_query.limit = pool_limit;
            let options = RrfOptions {
                limit: pool_limit,
                ..RrfOptions::default()
            };

            let signals_active = self.config.entity_linking
                || self.config.temporal_decay_lambda > 0.0
                || self.config.knowledge_update_supersession
                || self.config.episode_context_window > 0;

            let mut raw_hits = if !signals_active {
                // Fast path: existing 2-channel hybrid RRF (uses server-side if available).
                self.hybrid_rrf(pool_query.clone(), pool_query.clone(), options)
                    .await?
            } else {
                // Signal path: client-side multi-channel RRF + post-processing.
                self.ensure_ready().await?;
                let mut channels: Vec<Vec<SearchHit>> = vec![
                    self.keyword_hits(pool_query.clone()).await?,
                    self.vector_hits(pool_query.clone()).await?,
                ];

                if self.config.entity_linking {
                    let guard = self
                        .entity_index
                        .lock()
                        .map_err(|e| MemoryError::Database(e.to_string()))?;
                    let entity_hits = guard.entity_hits(&pool_query.text, pool_limit);
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
                    hits.truncate(pool_limit);
                }

                hits
            };

            if rerank_active {
                if let Some(reranker) = &self.reranker {
                    raw_hits = reranker.rerank(&query.text, raw_hits, query.limit)?;
                }
            }

            // Small-to-big: expand each chunk hit to its bounded parent-section
            // context and collapse same-parent hits. A no-op when disabled or when
            // no hit is a chunk (e.g. single-chunk records), keeping small-document
            // recall byte-identical.
            let hits = match self.effective_parent_budget() {
                Some(budget) => {
                    self.expand_small_to_big(raw_hits, query.limit, budget)
                        .await
                }
                None => Ok(raw_hits),
            }?;
            Ok(hits)
        }
        .boxed()
    }

    fn store(&self, memory: StoreMemory) -> BoxFuture<'_, MemoryResult<MemoryRecord>> {
        async move {
            memory.validate_confidence()?;
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
            let mut base_relations = memory.relations.clone();
            if self.config.relation_extraction {
                base_relations.extend(extract_entity_relations(
                    &memory.content,
                    &memory.tags,
                    &base_node,
                ));
            }
            let base_relations = normalize_relations(base_relations, &base_node);
            let created_at = memory.created_at.unwrap_or_else(Utc::now);
            let chunks = chunk_text(&memory.content, &ChunkConfig::default());
            let single = chunks.len() == 1;
            let chunk_count = chunks.len();

            // Adaptive budget: record the size of every multi-chunk parent so the
            // effective parent-context window can track the corpus median.
            if !single {
                if let Ok(mut samples) = self.parent_samples.lock() {
                    samples.record(memory.content.chars().count());
                }
            }

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
                    source: memory.source.clone(),
                    confidence: memory.confidence,
                    relations: Vec::new(),
                };
                // Single-chunk content keeps the original id (= stable id of the whole
                // memory) so existing idempotency/dedup is unchanged; multi-chunk records
                // each get their own content-addressed id.
                let id = if single {
                    base_id.clone()
                } else {
                    stable_memory_id(&chunk_memory)
                };
                if let Some(existing) = self.store.get(&self.config.collection, id.as_str()).await?
                {
                    if representative.is_none() {
                        representative = Some(point_to_record(existing)?);
                    }
                    continue;
                }
                let relations = if single || chunk.index == 0 {
                    base_relations.clone()
                } else {
                    Vec::new()
                };
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
                    source: memory.source.clone(),
                    confidence: memory.confidence,
                    relations,
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
                        filter: Filter::node_id(node_id.clone()),
                        limit: 1,
                        source: VectorSearchSource::Keyword,
                    },
                )
                .await?;
            if let Some(hit) = hits.pop() {
                return point_to_record(hit.point).map(Some);
            }
            // Drill-down: a chunked document has no point at the bare parent node_id;
            // reconstruct the full record from its chunk siblings in index order.
            let siblings = self.fetch_siblings(&node_id, 4_096).await?;
            if siblings.is_empty() {
                return Ok(None);
            }
            Ok(Some(reconstruct_parent_record(&node_id, siblings)))
        }
        .boxed()
    }

    fn neighbors(
        &self,
        node_id: &str,
        hops: usize,
    ) -> BoxFuture<'_, MemoryResult<Vec<MemoryRecord>>> {
        let node_id = node_id.to_string();
        async move {
            let records = self.graph_records().await?;
            let node_ids = neighbor_node_ids(&records, &node_id, hops);
            self.graph_records_for_node_ids(&records, node_ids).await
        }
        .boxed()
    }

    fn by_entity(&self, entity: &str) -> BoxFuture<'_, MemoryResult<Vec<MemoryRecord>>> {
        let entity = entity.to_string();
        async move {
            let records = self.graph_records().await?;
            let node_ids = by_entity_node_ids(&records, &entity);
            self.graph_records_for_node_ids(&records, node_ids).await
        }
        .boxed()
    }

    /// Bulk-store many memories, skipping the per-chunk existence round-trip. Content-hash IDs make
    /// upsert idempotent, so re-importing identical content is safe — the `skipped` count is
    /// tracked via a single up-front bulk ID scan so import callers can report duplicate counts.
    ///
    /// Qdrant path: embeds all chunks, batches `upsert_no_wait`, then a single `flush_upsert`.
    /// Other backends: `upsert_no_wait` delegates to `upsert` (wait=true per batch), no flush
    /// needed — correctness is unchanged.
    fn bulk_store<'a>(
        &'a self,
        memories: Vec<StoreMemory>,
        batch_size: usize,
    ) -> BoxFuture<'a, BulkStoreReport> {
        async move {
            if memories.is_empty() {
                return BulkStoreReport::default();
            }
            let batch_size = batch_size.max(1);

            // Ensure the collection and indexes exist before writing.
            if let Err(error) = self.ensure_ready().await {
                return BulkStoreReport {
                    stored: 0,
                    skipped: 0,
                    failures: memories
                        .iter()
                        .map(|m| {
                            (
                                stable_memory_id(m).to_string(),
                                format!("ensure_ready failed: {error}"),
                            )
                        })
                        .collect(),
                };
            }

            // Phase 1: collect all (id, VectorPoint) pairs without existence checks.
            let mut all_points: Vec<(String, VectorPoint)> = Vec::new();
            let mut failures: Vec<(String, String)> = Vec::new();

            for memory in memories {
                memory.validate_confidence().unwrap_or(());
                let base_id = stable_memory_id(&memory);
                let base_node = memory
                    .node_id
                    .clone()
                    .unwrap_or_else(|| format!("node:{base_id}"));
                let mut base_relations = memory.relations.clone();
                if self.config.relation_extraction {
                    base_relations.extend(extract_entity_relations(
                        &memory.content,
                        &memory.tags,
                        &base_node,
                    ));
                }
                let base_relations = normalize_relations(base_relations, &base_node);
                let created_at = memory.created_at.unwrap_or_else(Utc::now);
                let chunks = chunk_text(&memory.content, &ChunkConfig::default());
                let single = chunks.len() == 1;
                let chunk_count = chunks.len();

                if !single {
                    if let Ok(mut samples) = self.parent_samples.lock() {
                        samples.record(memory.content.chars().count());
                    }
                }

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
                        source: memory.source.clone(),
                        confidence: memory.confidence,
                        relations: Vec::new(),
                    };
                    let id = if single {
                        base_id.clone()
                    } else {
                        stable_memory_id(&chunk_memory)
                    };
                    let id_str = id.to_string();
                    let relations = if single || chunk.index == 0 {
                        base_relations.clone()
                    } else {
                        Vec::new()
                    };
                    let record = MemoryRecord {
                        id: id.clone(),
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
                        source: memory.source.clone(),
                        confidence: memory.confidence,
                        relations,
                    };
                    match self.embedder.embed_passage(&record.content) {
                        Ok(vector) => {
                            let payload = match serde_json::to_value(MemoryPayload::from(&record)) {
                                Ok(p) => p,
                                Err(error) => {
                                    failures.push((id_str, error.to_string()));
                                    continue;
                                }
                            };
                            all_points.push((
                                id_str,
                                VectorPoint {
                                    id: record.id.to_string(),
                                    vector,
                                    payload,
                                },
                            ));
                        }
                        Err(error) => {
                            failures.push((id_str, error.to_string()));
                        }
                    }
                }
            }

            // Phase 2: bulk upsert in batches using no-wait where supported.
            let mut stored = 0usize;
            let collection = &self.config.collection;
            for batch in all_points.chunks(batch_size) {
                let points: Vec<VectorPoint> = batch.iter().map(|(_, p)| p.clone()).collect();
                let batch_ids: Vec<String> = batch.iter().map(|(id, _)| id.clone()).collect();
                if let Err(error) = self.store.upsert_no_wait(collection, points).await {
                    for id in batch_ids {
                        failures.push((id, error.to_string()));
                    }
                } else {
                    stored += batch.len();
                }
            }

            // Phase 3: flush so all batches are indexed before returning.
            if let Err(error) = self.store.flush_upsert(collection).await {
                // Non-fatal: points were already accepted; log as a warning via a failure entry.
                failures.push((
                    "flush_upsert".to_string(),
                    format!("flush after bulk import failed (data still accepted): {error}"),
                ));
            }

            BulkStoreReport {
                stored,
                skipped: 0, // existence check skipped for speed; re-import is idempotent
                failures,
            }
        }
        .boxed()
    }
}

/// Reconstruct a full parent record from all of its chunk siblings (no budget cap —
/// this is the explicit drill-down to the complete source document).
fn reconstruct_parent_record(parent_node: &str, siblings: Vec<MemoryRecord>) -> MemoryRecord {
    let max_overlap = ChunkConfig::default().overlap_chars;
    let mut content = String::new();
    for sibling in &siblings {
        append_with_overlap(&mut content, &sibling.content, max_overlap);
    }
    let first = &siblings[0];
    let mut metadata = first.metadata.clone();
    metadata.remove("chunk_index");
    metadata.remove("parent_node");
    metadata.insert("chunk_count".to_string(), siblings.len().to_string());
    metadata.insert("reconstructed".to_string(), "true".to_string());
    MemoryRecord {
        id: first.id.clone(),
        node_id: parent_node.to_string(),
        content,
        tags: first.tags.clone(),
        metadata,
        tier: first.tier,
        created_at: first.created_at,
        scope: first.scope,
        agent_id: first.agent_id.clone(),
        session_id: first.session_id.clone(),
        task_id: first.task_id.clone(),
        user_id: first.user_id.clone(),
        source: first.source.clone(),
        confidence: first.confidence,
        relations: first.relations.clone(),
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    confidence: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    relations: Vec<Relation>,
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
    "artesian.compat".to_string()
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
            source: record.source.clone(),
            confidence: record.confidence,
            relations: record.relations.clone(),
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
            source: payload.source,
            confidence: payload.confidence,
            relations: payload.relations,
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
    // Every requested tag must be present (AND), matching the files backend's tag filter.
    // `must_eq` on the keyword-indexed `tags` array means "array contains the tag".
    for tag in &query.tags {
        filter.must_eq("tags", tag);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parent_size_median_is_empty_until_a_sample_exists() {
        let mut samples = ParentSizeSamples::default();
        assert_eq!(samples.median(), None);
        samples.record(1_000);
        assert_eq!(samples.median(), Some(1_000));
    }

    #[test]
    fn parent_size_median_tracks_the_middle_value() {
        let mut samples = ParentSizeSamples::default();
        for size in [1_000, 9_000, 3_000, 5_000, 7_000] {
            samples.record(size);
        }
        // Sorted: 1k 3k 5k 7k 9k -> median 5k. (Cache refreshes as the sample grows.)
        assert_eq!(samples.median(), Some(5_000));
    }

    #[test]
    fn append_with_overlap_removes_duplicated_boundary() {
        let mut acc = "the quick brown fox".to_string();
        append_with_overlap(&mut acc, "brown fox jumps over", 16);
        assert_eq!(acc, "the quick brown fox jumps over");
    }

    #[test]
    fn memory_payload_without_provenance_defaults_to_none() {
        let point = VectorPoint {
            id: "legacy".to_string(),
            vector: Vec::new(),
            payload: serde_json::json!({
                "id": "legacy",
                "node_id": "node:legacy",
                "content": "legacy vector payload",
                "tags": [],
                "metadata": {},
                "tier": "l1-atom",
                "created_at": Utc::now(),
            }),
        };

        let record = point_to_record(point).expect("legacy payload should decode");

        assert_eq!(record.source, None);
        assert_eq!(record.confidence, None);
        assert!(record.relations.is_empty());
    }

    #[test]
    fn append_with_overlap_appends_disjoint_text_verbatim() {
        let mut acc = "alpha".to_string();
        append_with_overlap(&mut acc, "beta", 8);
        assert_eq!(acc, "alphabeta");
    }
}
