// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeSet, HashSet};
use std::path::Path;
use std::{collections::HashMap, time::Duration};

use futures_util::{future::BoxFuture, FutureExt};
use qdrant_client::{
    qdrant::{
        Condition, CreateCollectionBuilder, CreateFieldIndexCollectionBuilder, DeletePointsBuilder,
        FieldType, Filter, GetPointsBuilder, HnswConfigDiffBuilder, PointId, PointStruct,
        PointsIdsList, QuantizationType, QueryPointsBuilder, RetrievedPoint,
        ScalarQuantizationBuilder, ScoredPoint, ScrollPointsBuilder, UpsertPointsBuilder, Value,
        VectorParamsBuilder, VectorsOutput,
    },
    Payload, Qdrant,
};
use serde_json::Value as JsonValue;
use sha2::{Digest, Sha256};

use crate::{
    vector::payload_matches_filter, Distance, Filter as MemoryFilter, FilterCondition, FilterValue,
    MemoryError, MemoryResult, PayloadIndex, RangeFilter, SnapshotReport, VectorCollection,
    VectorCollectionAdmin, VectorMemoryBackend, VectorMemoryConfig, VectorPoint,
    VectorQuantization, VectorSearch, VectorSearchHit, VectorSearchSource, VectorStore,
    VectorStoreCapabilities,
};

pub type QdrantBackend = VectorMemoryBackend<QdrantVectorStore>;

/// HNSW graph connectivity (edges per node). 16 is Qdrant's balanced default — enough recall
/// without bloating the index for the 10^3–10^5-point collections Artesian targets.
const QDRANT_HNSW_M: u64 = 16;
/// HNSW build-time neighbour search width. Higher trades slower build for better recall; 100 is
/// the recommended balance at this scale.
const QDRANT_HNSW_EF_CONSTRUCT: u64 = 100;
/// Scalar-quantization clipping quantile: drop the extreme 1% of the value distribution before
/// the int8 mapping so outliers do not compress the useful range.
const QDRANT_QUANT_QUANTILE: f32 = 0.99;

#[derive(Debug, Clone)]
pub struct QdrantVectorStoreConfig {
    pub url: String,
    pub rest_url: Option<String>,
    pub api_key: Option<String>,
}

impl QdrantVectorStoreConfig {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            rest_url: None,
            api_key: None,
        }
    }

    pub fn normalized(&self) -> MemoryResult<Self> {
        let endpoints = QdrantEndpoints::from_urls(&self.url, self.rest_url.as_deref())?;
        Ok(Self {
            url: endpoints.grpc_url,
            rest_url: Some(endpoints.rest_url),
            api_key: self.api_key.clone(),
        })
    }

    pub fn endpoints(&self) -> MemoryResult<QdrantEndpoints> {
        QdrantEndpoints::from_urls(&self.url, self.rest_url.as_deref())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QdrantEndpoints {
    pub grpc_url: String,
    pub rest_url: String,
}

impl QdrantEndpoints {
    pub fn from_urls(qdrant_url: &str, rest_url: Option<&str>) -> MemoryResult<Self> {
        let qdrant_url = normalize_url(qdrant_url)?;
        let rest_url = rest_url.map(normalize_url).transpose()?;
        let qdrant_port = qdrant_url.port();

        let (grpc_url, rest_url) = match (qdrant_port, rest_url) {
            (Some(6334), Some(rest_url)) => (qdrant_url, rest_url),
            (Some(6333), Some(rest_url)) => (derive_port(&qdrant_url, 6334)?, rest_url),
            (Some(6334), None) => (qdrant_url.clone(), derive_port(&qdrant_url, 6333)?),
            (Some(6333), None) => (derive_port(&qdrant_url, 6334)?, qdrant_url.clone()),
            (None, Some(rest_url)) => (derive_port(&qdrant_url, 6334)?, rest_url),
            (None, None) => (
                derive_port(&qdrant_url, 6334)?,
                derive_port(&qdrant_url, 6333)?,
            ),
            (Some(_), Some(rest_url)) => (qdrant_url, rest_url),
            (Some(port), None) => {
                return Err(MemoryError::InvalidFile(format!(
                    "cannot derive Qdrant REST endpoint from custom --qdrant-url port {port}; pass --qdrant-rest-url explicitly"
                )));
            }
        };

        Ok(Self {
            grpc_url: url_to_endpoint(&grpc_url),
            rest_url: url_to_endpoint(&rest_url),
        })
    }
}

pub struct QdrantVectorStore {
    config: QdrantVectorStoreConfig,
    client: Qdrant,
}

impl QdrantVectorStore {
    pub fn connect(config: QdrantVectorStoreConfig) -> MemoryResult<Self> {
        let config = config.normalized()?;
        let mut builder = Qdrant::from_url(&config.url);
        if let Some(api_key) = &config.api_key {
            builder = builder.api_key(api_key.clone());
        }
        let client = builder
            .build()
            .map_err(|error| MemoryError::Backend(error.to_string()))?;
        Ok(Self { config, client })
    }

    pub fn config(&self) -> &QdrantVectorStoreConfig {
        &self.config
    }

    pub fn client(&self) -> &Qdrant {
        &self.client
    }

    pub fn memory_backend(
        self,
        collection: impl Into<String>,
    ) -> MemoryResult<VectorMemoryBackend<Self>> {
        VectorMemoryBackend::new(self, VectorMemoryConfig::new(collection))
    }

    pub async fn preflight(config: QdrantVectorStoreConfig) -> MemoryResult<QdrantPreflightReport> {
        preflight_qdrant(config).await
    }

    /// Convert `VectorPoint`s and upsert them. `wait=true` blocks until the collection index is
    /// updated; `wait=false` returns as soon as the server has accepted the batch (used during bulk
    /// import to pipeline batches and issue one final wait at the end).
    pub(crate) async fn upsert_points_internal(
        &self,
        collection: &str,
        points: Vec<VectorPoint>,
        wait: bool,
    ) -> MemoryResult<()> {
        if points.is_empty() {
            return Ok(());
        }
        let qdrant_points = points
            .into_iter()
            .map(|point| {
                let payload = Payload::try_from(point.payload)
                    .map_err(|error| MemoryError::Backend(error.to_string()))?;
                Ok(PointStruct::new(
                    qdrant_point_id(&point.id),
                    point.vector,
                    payload,
                ))
            })
            .collect::<MemoryResult<Vec<_>>>()?;
        self.client
            .upsert_points(UpsertPointsBuilder::new(collection, qdrant_points).wait(wait))
            .await
            .map_err(qdrant_error)?;
        Ok(())
    }

    /// Scroll every point ID (as UUID strings) in `collection` into a `HashSet`. Used by
    /// incremental replication to compute the ID diff without fetching payloads/vectors.
    pub(crate) async fn scroll_all_ids(
        &self,
        collection: &str,
        batch: u32,
    ) -> MemoryResult<HashSet<String>> {
        let mut ids = HashSet::new();
        let mut offset: Option<PointId> = None;
        loop {
            let mut builder = ScrollPointsBuilder::new(collection)
                .limit(batch.max(1))
                .with_payload(false)
                .with_vectors(false);
            if let Some(off) = offset.clone() {
                builder = builder.offset(off);
            }
            let response = self.client.scroll(builder).await.map_err(qdrant_error)?;
            let next = response.next_page_offset.clone();
            for point in response.result {
                if let Some(id) = point.id {
                    ids.insert(point_id_to_string(&id));
                }
            }
            match next {
                Some(next_offset) => offset = Some(next_offset),
                None => break,
            }
        }
        Ok(ids)
    }

    /// Delete `ids` (UUID strings) from `collection` on the target, batched to avoid
    /// oversized single requests. Used by incremental replication with `--prune`.
    async fn delete_ids(
        &self,
        collection: &str,
        ids: Vec<String>,
        batch: u32,
    ) -> MemoryResult<usize> {
        let total = ids.len();
        for chunk in ids.chunks(batch.max(1) as usize) {
            let point_ids: Vec<PointId> = chunk.iter().map(|id| id.clone().into()).collect();
            self.client
                .delete_points(
                    DeletePointsBuilder::new(collection)
                        .points(PointsIdsList { ids: point_ids })
                        .wait(true),
                )
                .await
                .map_err(qdrant_error)?;
        }
        Ok(total)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QdrantPreflightReport {
    pub grpc_url: String,
    pub rest_url: String,
    pub grpc_version: String,
    pub rest_status: u16,
}

pub async fn preflight_qdrant(
    config: QdrantVectorStoreConfig,
) -> MemoryResult<QdrantPreflightReport> {
    let config = config.normalized()?;
    let mut builder = Qdrant::from_url(&config.url)
        .timeout(Duration::from_secs(3))
        .connect_timeout(Duration::from_secs(3));
    if let Some(api_key) = &config.api_key {
        builder = builder.api_key(api_key.clone());
    }
    let client = builder
        .build()
        .map_err(|error| MemoryError::Backend(error.to_string()))?;
    let health = client.health_check().await.map_err(|error| {
        MemoryError::Backend(format!(
            "Qdrant gRPC preflight failed for {}; expected the gRPC endpoint (default :6334). \
             Check that the gRPC port is exposed and that --qdrant-url is not pointing at an unrelated service. details: {error}",
            config.url
        ))
    })?;

    let rest_url = rest_url(&config);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .map_err(|error| MemoryError::Backend(error.to_string()))?;
    let mut request = client.get(format!("{rest_url}/healthz"));
    if let Some(api_key) = &config.api_key {
        request = request.header("api-key", api_key);
    }
    let response = request.send().await.map_err(|error| {
        MemoryError::Backend(format!(
            "Qdrant REST preflight failed for {rest_url}/healthz; expected the REST endpoint \
             (default :6333). Check the REST port or pass --qdrant-rest-url explicitly. details: {error}"
        ))
    })?;
    let status = response.status();
    if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
        return Err(MemoryError::Backend(format!(
            "Qdrant REST preflight failed for {rest_url}/healthz with {status}; set the configured API key env var or remove Qdrant auth for local testing"
        )));
    }
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        return Err(MemoryError::Backend(format!(
            "Qdrant REST preflight failed for {rest_url}/healthz with {status}: {text}"
        )));
    }

    Ok(QdrantPreflightReport {
        grpc_url: config.url,
        rest_url,
        grpc_version: health.version,
        rest_status: status.as_u16(),
    })
}

impl VectorStore for QdrantVectorStore {
    fn ensure_collection(&self, collection: VectorCollection) -> BoxFuture<'_, MemoryResult<()>> {
        async move {
            let exists = self
                .client
                .collection_exists(&collection.name)
                .await
                .map_err(qdrant_error)?;
            if exists {
                return Ok(());
            }

            // Tune at the collection level so the config is explicit and inspectable:
            // an HNSW graph sized for this scale, and (opt-in) Int8 scalar quantization that
            // keeps 4x-smaller vectors in RAM for fast scoring while full-precision originals
            // stay on disk for rescoring. The default stays Float32.
            let mut builder = CreateCollectionBuilder::new(&collection.name)
                .vectors_config(VectorParamsBuilder::new(
                    collection.dimensions as u64,
                    qdrant_distance(collection.distance),
                ))
                .hnsw_config(
                    HnswConfigDiffBuilder::default()
                        .m(QDRANT_HNSW_M)
                        .ef_construct(QDRANT_HNSW_EF_CONSTRUCT),
                );
            if collection.quantization == VectorQuantization::Int8 {
                builder = builder.quantization_config(
                    ScalarQuantizationBuilder::default()
                        .r#type(QuantizationType::Int8 as i32)
                        .quantile(QDRANT_QUANT_QUANTILE)
                        .always_ram(true),
                );
            }

            self.client
                .create_collection(builder)
                .await
                .map_err(qdrant_error)?;
            Ok(())
        }
        .boxed()
    }

    fn ensure_payload_index(
        &self,
        collection: &str,
        index: PayloadIndex,
    ) -> BoxFuture<'_, MemoryResult<()>> {
        let collection = collection.to_string();
        async move {
            let field_type = payload_index_field_type(&index.field);
            let result = self
                .client
                .create_field_index(
                    CreateFieldIndexCollectionBuilder::new(collection, index.field, field_type)
                        .wait(true),
                )
                .await;
            match result {
                Ok(_) => Ok(()),
                Err(error) if error.to_string().contains("already exists") => Ok(()),
                Err(error) => Err(qdrant_error(error)),
            }
        }
        .boxed()
    }

    fn upsert(
        &self,
        collection: &str,
        points: Vec<VectorPoint>,
    ) -> BoxFuture<'_, MemoryResult<()>> {
        let collection = collection.to_string();
        async move { self.upsert_points_internal(&collection, points, true).await }.boxed()
    }

    fn upsert_no_wait(
        &self,
        collection: &str,
        points: Vec<VectorPoint>,
    ) -> BoxFuture<'_, MemoryResult<()>> {
        let collection = collection.to_string();
        async move {
            self.upsert_points_internal(&collection, points, false)
                .await
        }
        .boxed()
    }

    fn flush_upsert(&self, collection: &str) -> BoxFuture<'_, MemoryResult<()>> {
        let collection = collection.to_string();
        async move {
            // Send an empty batch with wait=true so Qdrant indexes all preceding no-wait batches.
            self.client
                .upsert_points(
                    UpsertPointsBuilder::new(&collection, Vec::<PointStruct>::new()).wait(true),
                )
                .await
                .map_err(qdrant_error)?;
            Ok(())
        }
        .boxed()
    }

    fn search(
        &self,
        collection: &str,
        search: VectorSearch,
    ) -> BoxFuture<'_, MemoryResult<Vec<VectorSearchHit>>> {
        let collection = collection.to_string();
        async move {
            match search.source {
                VectorSearchSource::Vector | VectorSearchSource::Hybrid
                    if search.vector.is_some() =>
                {
                    let mut builder = QueryPointsBuilder::new(collection)
                        .query(search.vector.expect("vector checked above"))
                        .limit(search.limit as u64)
                        .with_payload(true);
                    if let Some(filter) = qdrant_filter(&search.filter) {
                        builder = builder.filter(filter);
                    }
                    let response = self.client.query(builder).await.map_err(qdrant_error)?;
                    response
                        .result
                        .into_iter()
                        .filter_map(|point| scored_point_to_hit(point, &search.filter).transpose())
                        .collect()
                }
                VectorSearchSource::Vector => Ok(Vec::new()),
                VectorSearchSource::Keyword | VectorSearchSource::Hybrid => {
                    let text = search.text.unwrap_or_default();
                    let response = self
                        .client
                        .scroll(
                            scroll_builder(&collection, &search.filter)
                                .limit((search.limit.max(1) * 10) as u32)
                                .with_payload(true),
                        )
                        .await
                        .map_err(qdrant_error)?;
                    let mut hits = response
                        .result
                        .into_iter()
                        .filter_map(|point| {
                            retrieved_point_to_hit(point, &search.filter, &text).transpose()
                        })
                        .collect::<MemoryResult<Vec<_>>>()?;
                    hits.sort_by(|left, right| {
                        right
                            .score
                            .partial_cmp(&left.score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    hits.truncate(search.limit);
                    Ok(hits)
                }
            }
        }
        .boxed()
    }

    fn get(
        &self,
        collection: &str,
        point_id: &str,
    ) -> BoxFuture<'_, MemoryResult<Option<VectorPoint>>> {
        let collection = collection.to_string();
        let point_id = point_id.to_string();
        async move {
            let response = self
                .client
                .get_points(
                    GetPointsBuilder::new(collection, vec![qdrant_point_id(&point_id).into()])
                        .with_payload(true)
                        .with_vectors(true),
                )
                .await
                .map_err(qdrant_error)?;
            response
                .result
                .into_iter()
                .next()
                .map(retrieved_point_to_point)
                .transpose()
        }
        .boxed()
    }

    fn distinct_payload_values(
        &self,
        collection: &str,
        field: &str,
    ) -> BoxFuture<'_, MemoryResult<Vec<String>>> {
        let collection = collection.to_string();
        let field = field.to_string();
        async move {
            let mut values = BTreeSet::new();
            let mut offset: Option<PointId> = None;
            loop {
                let mut builder = ScrollPointsBuilder::new(&collection)
                    .limit(256)
                    .with_payload(true)
                    .with_vectors(false);
                if let Some(off) = offset.clone() {
                    builder = builder.offset(off);
                }
                let response = self.client.scroll(builder).await.map_err(qdrant_error)?;
                let next = response.next_page_offset.clone();
                for point in response.result {
                    let point = retrieved_point_to_point(point)?;
                    if let Some(value) = json_field_string(&point.payload, &field) {
                        values.insert(value.to_string());
                    }
                }
                match next {
                    Some(next_offset) => offset = Some(next_offset),
                    None => break,
                }
            }
            Ok(values.into_iter().collect())
        }
        .boxed()
    }

    fn capabilities(&self) -> VectorStoreCapabilities {
        VectorStoreCapabilities {
            supports_server_side_hybrid: false,
            supports_sparse: false,
        }
    }
}

/// Copy every point of `collection` from `source` into `target` (scroll + upsert, batched). The
/// target collection is created if missing (dimensions inferred from the first point, cosine
/// distance). Upsert is keyed by point id, so this MERGES rather than clobbers. Returns the number
/// of points copied — the primitive behind `artesian replicate` for local <-> LAN Qdrant sync.
pub async fn replicate_collection(
    source: &QdrantVectorStore,
    target: &QdrantVectorStore,
    source_collection: &str,
    target_collection: &str,
    batch: u32,
) -> MemoryResult<usize> {
    let mut offset: Option<PointId> = None;
    let mut ensured = false;
    let mut total = 0usize;
    loop {
        let mut builder = ScrollPointsBuilder::new(source_collection)
            .limit(batch.max(1))
            .with_payload(true)
            .with_vectors(true);
        if let Some(off) = offset.clone() {
            builder = builder.offset(off);
        }
        let response = source.client.scroll(builder).await.map_err(qdrant_error)?;
        let next = response.next_page_offset.clone();
        let retrieved = response.result;
        if retrieved.is_empty() {
            break;
        }
        if !ensured {
            let dimensions = extract_vector(&retrieved[0].vectors)
                .ok_or_else(|| {
                    MemoryError::Backend(
                        "source points carry no single unnamed vector to replicate".to_string(),
                    )
                })?
                .len();
            target
                .ensure_collection(VectorCollection {
                    name: target_collection.to_string(),
                    dimensions,
                    distance: Distance::Cosine,
                    quantization: VectorQuantization::default(),
                })
                .await?;
            ensured = true;
        }
        total += retrieved.len();
        // Preserve the original id, vector, and payload (a raw point copy, not a re-embed).
        let points: Vec<PointStruct> = retrieved
            .into_iter()
            .filter_map(|point| {
                let id = point.id?;
                let vector = extract_vector(&point.vectors)?;
                Some(PointStruct::new(id, vector, Payload::from(point.payload)))
            })
            .collect();
        target
            .client
            .upsert_points(UpsertPointsBuilder::new(target_collection, points).wait(true))
            .await
            .map_err(qdrant_error)?;
        match next {
            Some(next_offset) => offset = Some(next_offset),
            None => break,
        }
    }
    Ok(total)
}

/// Report returned by `replicate_collection_incremental`.
#[derive(Debug, Clone, Default)]
pub struct ReplicateReport {
    /// Points upserted to the target (new or changed).
    pub upserted: usize,
    /// Points removed from the target because they are no longer in the source (`--prune` only).
    pub deleted: usize,
    /// Points present in both source and target with matching IDs — not re-sent.
    pub unchanged: usize,
}

/// Incremental replication: scroll IDs from both source and target, upsert only the points whose
/// IDs are absent from the target (or whose payload/vector changed), and optionally delete from the
/// target any IDs that are no longer in the source. A full-copy fallback is available via
/// `replicate_collection`.
///
/// - `prune`: if `true`, delete target points whose IDs are not in the source.
/// - `batch`: number of points per scroll/upsert call.
pub async fn replicate_collection_incremental(
    source: &QdrantVectorStore,
    target: &QdrantVectorStore,
    source_collection: &str,
    target_collection: &str,
    prune: bool,
    batch: u32,
) -> MemoryResult<ReplicateReport> {
    // Phase 1: collect the full ID sets from both endpoints.
    let source_ids = source
        .scroll_all_ids(source_collection, batch)
        .await
        .map_err(|error| {
            MemoryError::Backend(format!(
                "failed to scroll source IDs from {source_collection}: {error}"
            ))
        })?;
    let target_ids = target
        .scroll_all_ids(target_collection, batch)
        .await
        .unwrap_or_default(); // target collection may not exist yet

    // Phase 2: compute diff.
    let to_upsert: Vec<&String> = source_ids.difference(&target_ids).collect();
    let to_delete: Vec<String> = if prune {
        target_ids.difference(&source_ids).cloned().collect()
    } else {
        Vec::new()
    };
    let unchanged = source_ids.intersection(&target_ids).count();

    let mut report = ReplicateReport {
        upserted: 0,
        deleted: 0,
        unchanged,
    };

    if to_upsert.is_empty() && to_delete.is_empty() {
        return Ok(report);
    }

    // Ensure target collection exists (infer dimensions from first source point to upsert).
    if !to_upsert.is_empty() {
        // Fetch the first point to determine dimensions.
        let first_id = to_upsert[0];
        let first_point = {
            let response = source
                .client
                .get_points(
                    GetPointsBuilder::new(
                        source_collection,
                        vec![qdrant_point_id(first_id).into()],
                    )
                    .with_payload(true)
                    .with_vectors(true),
                )
                .await
                .map_err(qdrant_error)?;
            response.result.into_iter().next()
        };
        if let Some(fp) = first_point {
            let dimensions = extract_vector(&fp.vectors)
                .ok_or_else(|| {
                    MemoryError::Backend(
                        "source points carry no single unnamed vector to replicate".to_string(),
                    )
                })?
                .len();
            target
                .ensure_collection(VectorCollection {
                    name: target_collection.to_string(),
                    dimensions,
                    distance: Distance::Cosine,
                    quantization: VectorQuantization::default(),
                })
                .await?;
        }

        // Fetch and upsert the missing points in batches.
        for id_chunk in to_upsert.chunks(batch.max(1) as usize) {
            let point_ids: Vec<PointId> = id_chunk
                .iter()
                .map(|id| qdrant_point_id(id).into())
                .collect();
            let response = source
                .client
                .get_points(
                    GetPointsBuilder::new(source_collection, point_ids)
                        .with_payload(true)
                        .with_vectors(true),
                )
                .await
                .map_err(qdrant_error)?;
            let points: Vec<PointStruct> = response
                .result
                .into_iter()
                .filter_map(|point| {
                    let id = point.id?;
                    let vector = extract_vector(&point.vectors)?;
                    Some(PointStruct::new(id, vector, Payload::from(point.payload)))
                })
                .collect();
            let n = points.len();
            target
                .client
                .upsert_points(UpsertPointsBuilder::new(target_collection, points).wait(false))
                .await
                .map_err(qdrant_error)?;
            report.upserted += n;
        }
        // Final wait: ensure all batches are indexed.
        target
            .client
            .upsert_points(
                UpsertPointsBuilder::new(target_collection, Vec::<PointStruct>::new()).wait(true),
            )
            .await
            .map_err(qdrant_error)?;
    }

    // Phase 3: delete target-only points (prune mode).
    if !to_delete.is_empty() {
        report.deleted = target
            .delete_ids(target_collection, to_delete, batch)
            .await?;
    }

    Ok(report)
}

/// Extract a point's single unnamed dense vector, if present (handles both the legacy `data`
/// field and the newer nested dense oneof).
/// Choose the Qdrant payload index type for a field so filters use the right index:
/// full-text for `content`, datetime for RFC 3339 timestamp fields (range/recency filtering),
/// integer for token counts, keyword for everything else (the equality-filter default).
fn payload_index_field_type(field: &str) -> FieldType {
    match field {
        "content" => FieldType::Text,
        "project" => FieldType::Keyword,
        "created_at" | "updated_at" | "committed_at" => FieldType::Datetime,
        "tokens" | "token_count" => FieldType::Integer,
        _ => FieldType::Keyword,
    }
}

// Modern Qdrant servers (1.10+) return the unnamed dense vector through the
// `vector_output::Vector::Dense` oneof; older servers populate the now-deprecated
// flat `VectorOutput::data` field. Support both so replication works across versions.
#[allow(deprecated)]
fn extract_vector(vectors: &Option<VectorsOutput>) -> Option<Vec<f32>> {
    use qdrant_client::qdrant::vector_output::Vector as VectorKind;
    use qdrant_client::qdrant::vectors_output::VectorsOptions;
    match vectors.as_ref()?.vectors_options.as_ref()? {
        VectorsOptions::Vector(vector) => match vector.vector.as_ref() {
            Some(VectorKind::Dense(dense)) => Some(dense.data.clone()),
            Some(_) => None,
            None if !vector.data.is_empty() => Some(vector.data.clone()),
            None => None,
        },
        VectorsOptions::Vectors(_) => None,
    }
}

impl VectorCollectionAdmin for QdrantVectorStore {
    fn active_collection(&self, alias: &str) -> BoxFuture<'_, MemoryResult<Option<String>>> {
        let alias = alias.to_string();
        async move {
            let aliases = self.client.list_aliases().await.map_err(qdrant_error)?;
            Ok(aliases
                .aliases
                .into_iter()
                .find(|candidate| candidate.alias_name == alias)
                .map(|candidate| candidate.collection_name))
        }
        .boxed()
    }

    fn swap_alias(
        &self,
        alias: &str,
        old_collection: Option<&str>,
        new_collection: &str,
    ) -> BoxFuture<'_, MemoryResult<()>> {
        let alias = alias.to_string();
        let old_collection = old_collection.map(str::to_string);
        let new_collection = new_collection.to_string();
        async move {
            if old_collection.as_deref() == Some(new_collection.as_str()) {
                return Ok(());
            }
            let mut actions = Vec::new();
            if old_collection.is_some() {
                actions.push(serde_json::json!({
                    "delete_alias": {
                        "alias_name": alias
                    }
                }));
            }
            actions.push(serde_json::json!({
                "create_alias": {
                    "collection_name": new_collection,
                    "alias_name": alias
                }
            }));
            let client = reqwest::Client::new();
            let mut request = client
                .post(format!("{}/collections/aliases", rest_url(&self.config)))
                .json(&serde_json::json!({ "actions": actions }));
            if let Some(api_key) = &self.config.api_key {
                request = request.header("api-key", api_key);
            }
            let response = request
                .send()
                .await
                .map_err(|error| MemoryError::Backend(error.to_string()))?;
            if !response.status().is_success() {
                let status = response.status();
                let text = response.text().await.unwrap_or_default();
                return Err(MemoryError::Backend(format!(
                    "Qdrant alias swap failed with {status}: {text}"
                )));
            }
            Ok(())
        }
        .boxed()
    }

    fn snapshot_collection(
        &self,
        collection: &str,
        target_dir: &Path,
    ) -> BoxFuture<'_, MemoryResult<SnapshotReport>> {
        let collection = collection.to_string();
        let target_dir = target_dir.to_path_buf();
        async move {
            std::fs::create_dir_all(&target_dir)?;
            let snapshot = self
                .client
                .create_snapshot(collection.clone())
                .await
                .map_err(qdrant_error)?
                .snapshot_description
                .ok_or_else(|| {
                    MemoryError::Backend("Qdrant returned no snapshot description".to_string())
                })?;
            let path = target_dir.join(&snapshot.name);
            download_snapshot_file(&self.config, &collection, &snapshot.name, &path).await?;
            Ok(SnapshotReport {
                collection,
                snapshot_name: snapshot.name,
                path,
                size_bytes: u64::try_from(snapshot.size).ok(),
                checksum: snapshot.checksum,
            })
        }
        .boxed()
    }

    fn delete_collection(&self, collection: &str) -> BoxFuture<'_, MemoryResult<()>> {
        let collection = collection.to_string();
        async move {
            self.client
                .delete_collection(collection)
                .await
                .map_err(qdrant_error)?;
            Ok(())
        }
        .boxed()
    }
}

fn qdrant_distance(distance: Distance) -> qdrant_client::qdrant::Distance {
    match distance {
        Distance::Cosine => qdrant_client::qdrant::Distance::Cosine,
        Distance::Dot => qdrant_client::qdrant::Distance::Dot,
        Distance::Euclidean => qdrant_client::qdrant::Distance::Euclid,
    }
}

fn scroll_builder(collection: &str, filter: &MemoryFilter) -> ScrollPointsBuilder {
    let mut builder = ScrollPointsBuilder::new(collection);
    if let Some(filter) = qdrant_filter(filter) {
        builder = builder.filter(filter);
    }
    builder
}

fn qdrant_filter(filter: &MemoryFilter) -> Option<Filter> {
    if filter.is_empty() {
        return None;
    }

    Some(Filter {
        must: filter.must.iter().filter_map(qdrant_condition).collect(),
        should: filter.should.iter().filter_map(qdrant_condition).collect(),
        must_not: filter
            .must_not
            .iter()
            .filter_map(qdrant_condition)
            .collect(),
        min_should: None,
    })
}

fn qdrant_condition(condition: &FilterCondition) -> Option<Condition> {
    match condition {
        FilterCondition::Eq { field, value } => {
            value_to_match(value).map(|value| Condition::matches(field.to_string(), value))
        }
        FilterCondition::In { field, values } => {
            let values = values
                .iter()
                .filter_map(|value| match value {
                    FilterValue::String(value) => Some(value.clone()),
                    _ => None,
                })
                .collect::<Vec<String>>();
            if values.is_empty() {
                None
            } else {
                Some(Condition::matches(field.to_string(), values))
            }
        }
        FilterCondition::Range(range) => qdrant_range_condition(range),
        FilterCondition::Exists { field } => Some(Condition::is_empty(field.to_string())),
    }
}

fn value_to_match(value: &FilterValue) -> Option<qdrant_client::qdrant::r#match::MatchValue> {
    match value {
        FilterValue::String(value) => Some(qdrant_client::qdrant::r#match::MatchValue::Keyword(
            value.clone(),
        )),
        FilterValue::Bool(value) => {
            Some(qdrant_client::qdrant::r#match::MatchValue::Boolean(*value))
        }
        FilterValue::Number(_) => None,
    }
}

fn qdrant_range_condition(range: &RangeFilter) -> Option<Condition> {
    use qdrant_client::qdrant::{condition::ConditionOneOf, FieldCondition, Range};

    Some(Condition {
        condition_one_of: Some(ConditionOneOf::Field(FieldCondition {
            key: range.field.clone(),
            range: Some(Range {
                lt: range.lt,
                gt: range.gt,
                gte: range.gte,
                lte: range.lte,
            }),
            ..Default::default()
        })),
    })
}

fn scored_point_to_hit(
    point: ScoredPoint,
    filter: &MemoryFilter,
) -> MemoryResult<Option<VectorSearchHit>> {
    let vector_point = scored_point_to_point(point)?;
    if !payload_matches_filter(&vector_point.payload, filter) {
        return Ok(None);
    }
    let score = vector_point
        .payload
        .get("_score")
        .and_then(JsonValue::as_f64)
        .unwrap_or(0.0) as f32;
    Ok(Some(VectorSearchHit {
        point: strip_score(vector_point),
        score,
    }))
}

fn retrieved_point_to_hit(
    point: RetrievedPoint,
    filter: &MemoryFilter,
    text: &str,
) -> MemoryResult<Option<VectorSearchHit>> {
    let vector_point = retrieved_point_to_point(point)?;
    if !payload_matches_filter(&vector_point.payload, filter) {
        return Ok(None);
    }
    let content = vector_point
        .payload
        .get("content")
        .and_then(JsonValue::as_str)
        .unwrap_or_default();
    // Filter-only retrieval (empty query text — e.g. small-to-big sibling lookups or
    // node_id drill-down) keeps every point the filter already selected; we only drop on
    // a genuine keyword miss. Without this, an empty query scored 0 and dropped all
    // filter-matched rows, silently breaking sibling/parent retrieval on Qdrant.
    let score = if text.trim().is_empty() {
        1.0
    } else {
        keyword_score(content, text)
    };
    if score == 0.0 {
        return Ok(None);
    }
    Ok(Some(VectorSearchHit {
        point: vector_point,
        score,
    }))
}

fn scored_point_to_point(point: ScoredPoint) -> MemoryResult<VectorPoint> {
    let mut payload = json_from_payload(point.payload)?;
    if let JsonValue::Object(map) = &mut payload {
        map.insert("_score".to_string(), JsonValue::from(point.score));
    }
    Ok(VectorPoint {
        id: payload_id(&payload),
        vector: Vec::new(),
        payload,
    })
}

fn retrieved_point_to_point(point: RetrievedPoint) -> MemoryResult<VectorPoint> {
    let payload = json_from_payload(point.payload)?;
    Ok(VectorPoint {
        id: payload_id(&payload),
        vector: Vec::new(),
        payload,
    })
}

fn json_from_payload(payload: HashMap<String, Value>) -> MemoryResult<JsonValue> {
    serde_json::to_value(Payload::from(payload))
        .map_err(|error| MemoryError::Backend(error.to_string()))
}

fn payload_id(payload: &JsonValue) -> String {
    payload
        .get("id")
        .and_then(JsonValue::as_str)
        .unwrap_or_default()
        .to_string()
}

fn json_field_string<'a>(payload: &'a JsonValue, field: &str) -> Option<&'a str> {
    let mut value = payload;
    for part in field.split('.') {
        value = value.get(part)?;
    }
    value.as_str()
}

fn strip_score(mut point: VectorPoint) -> VectorPoint {
    if let JsonValue::Object(map) = &mut point.payload {
        map.remove("_score");
    }
    point
}

fn keyword_score(content: &str, query: &str) -> f32 {
    if query.trim().is_empty() {
        return 1.0;
    }
    let content = content.to_ascii_lowercase();
    query
        .split_whitespace()
        .filter(|term| content.contains(&term.to_ascii_lowercase()))
        .count() as f32
}

fn qdrant_point_id(memory_id: &str) -> String {
    let hex = if memory_id.len() >= 32
        && memory_id
            .chars()
            .all(|character| character.is_ascii_hexdigit())
    {
        memory_id.to_string()
    } else {
        format!("{:x}", Sha256::digest(memory_id.as_bytes()))
    };
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

/// Convert a Qdrant `PointId` back to a UUID string. The incremental replication ID-diff stores
/// IDs as UUID strings so they can be compared across source and target without round-tripping
/// through the Qdrant hash function.
fn point_id_to_string(id: &PointId) -> String {
    use qdrant_client::qdrant::point_id::PointIdOptions;
    match &id.point_id_options {
        Some(PointIdOptions::Uuid(uuid)) => uuid.clone(),
        Some(PointIdOptions::Num(n)) => n.to_string(),
        None => String::new(),
    }
}

fn qdrant_error(error: qdrant_client::QdrantError) -> MemoryError {
    MemoryError::Backend(error.to_string())
}

async fn download_snapshot_file(
    config: &QdrantVectorStoreConfig,
    collection: &str,
    snapshot_name: &str,
    path: &Path,
) -> MemoryResult<()> {
    let url = format!(
        "{}/collections/{}/snapshots/{}",
        rest_url(config),
        collection,
        snapshot_name
    );
    let client = reqwest::Client::new();
    let mut request = client.get(url);
    if let Some(api_key) = &config.api_key {
        request = request.header("api-key", api_key);
    }
    let response = request
        .send()
        .await
        .map_err(|error| MemoryError::Backend(error.to_string()))?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        return Err(MemoryError::Backend(format!(
            "Qdrant snapshot download failed with {status}: {text}"
        )));
    }
    let bytes = response
        .bytes()
        .await
        .map_err(|error| MemoryError::Backend(error.to_string()))?;
    std::fs::write(path, bytes)?;
    Ok(())
}

fn rest_url(config: &QdrantVectorStoreConfig) -> String {
    if let Some(rest_url) = &config.rest_url {
        return rest_url.trim_end_matches('/').to_string();
    }
    if let Some(prefix) = config.url.strip_suffix(":6334") {
        return format!("{prefix}:6333");
    }
    config.url.trim_end_matches('/').to_string()
}

fn normalize_url(input: &str) -> MemoryResult<reqwest::Url> {
    let input = input.trim().trim_end_matches('/');
    reqwest::Url::parse(input).map_err(|error| {
        MemoryError::InvalidFile(format!(
            "invalid Qdrant URL `{input}`: {error}; expected http://HOST:6333 for REST or http://HOST:6334 for gRPC"
        ))
    })
}

fn derive_port(url: &reqwest::Url, port: u16) -> MemoryResult<reqwest::Url> {
    let mut derived = url.clone();
    derived.set_port(Some(port)).map_err(|()| {
        MemoryError::InvalidFile(format!(
            "cannot derive Qdrant endpoint from {}; pass both --qdrant-url and --qdrant-rest-url explicitly",
            url_to_endpoint(url)
        ))
    })?;
    Ok(derived)
}

fn url_to_endpoint(url: &reqwest::Url) -> String {
    url.to_string().trim_end_matches('/').to_string()
}

impl std::fmt::Display for QdrantEndpoints {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "grpc={} rest={}", self.grpc_url, self.rest_url)
    }
}

#[cfg(test)]
mod tests {
    use super::{QdrantEndpoints, QdrantVectorStoreConfig};

    #[test]
    fn derives_grpc_from_rest_qdrant_url() {
        let endpoints = QdrantVectorStoreConfig::new("http://qdrant.local:6333")
            .endpoints()
            .expect("endpoints should derive");
        assert_eq!(
            endpoints,
            QdrantEndpoints {
                grpc_url: "http://qdrant.local:6334".to_string(),
                rest_url: "http://qdrant.local:6333".to_string(),
            }
        );
    }

    #[test]
    fn derives_rest_from_grpc_qdrant_url() {
        let endpoints = QdrantVectorStoreConfig::new("http://qdrant.local:6334")
            .endpoints()
            .expect("endpoints should derive");
        assert_eq!(endpoints.grpc_url, "http://qdrant.local:6334");
        assert_eq!(endpoints.rest_url, "http://qdrant.local:6333");
    }

    #[test]
    fn custom_qdrant_port_requires_explicit_rest_url() {
        let error = QdrantVectorStoreConfig::new("http://qdrant.local:7000")
            .endpoints()
            .expect_err("custom port without REST URL should fail");
        assert!(error
            .to_string()
            .contains("pass --qdrant-rest-url explicitly"));
    }
}
