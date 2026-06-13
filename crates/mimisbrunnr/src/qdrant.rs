// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use futures_util::{future::BoxFuture, FutureExt};
use qdrant_client::{
    qdrant::{
        Condition, CreateCollectionBuilder, CreateFieldIndexCollectionBuilder, FieldType, Filter,
        GetPointsBuilder, PointStruct, QueryPointsBuilder, RetrievedPoint, ScoredPoint,
        ScrollPointsBuilder, UpsertPointsBuilder, Value, VectorParamsBuilder,
    },
    Payload, Qdrant,
};
use serde_json::Value as JsonValue;
use sha2::{Digest, Sha256};

use crate::{
    vector::payload_matches_filter, Distance, Filter as MemoryFilter, FilterCondition, FilterValue,
    MemoryError, MemoryResult, PayloadIndex, RangeFilter, VectorCollection, VectorMemoryBackend,
    VectorMemoryConfig, VectorPoint, VectorSearch, VectorSearchHit, VectorSearchSource,
    VectorStore, VectorStoreCapabilities,
};

pub type QdrantBackend = VectorMemoryBackend<QdrantVectorStore>;

#[derive(Debug, Clone)]
pub struct QdrantVectorStoreConfig {
    pub url: String,
    pub api_key: Option<String>,
}

impl QdrantVectorStoreConfig {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            api_key: None,
        }
    }
}

pub struct QdrantVectorStore {
    config: QdrantVectorStoreConfig,
    client: Qdrant,
}

impl QdrantVectorStore {
    pub fn connect(config: QdrantVectorStoreConfig) -> MemoryResult<Self> {
        let mut builder = Qdrant::from_url(&config.url);
        if let Some(api_key) = &config.api_key {
            builder = builder.api_key(api_key.clone());
        }
        let client = builder
            .build()
            .map_err(|error| MemoryError::BackendUnavailable(error.to_string()))?;
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

            self.client
                .create_collection(
                    CreateCollectionBuilder::new(&collection.name).vectors_config(
                        VectorParamsBuilder::new(
                            collection.dimensions as u64,
                            qdrant_distance(collection.distance),
                        ),
                    ),
                )
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
            let field_type = if index.field == "content" {
                FieldType::Text
            } else {
                FieldType::Keyword
            };
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
        async move {
            let points = points
                .into_iter()
                .map(|point| {
                    let payload = Payload::try_from(point.payload)
                        .map_err(|error| MemoryError::BackendUnavailable(error.to_string()))?;
                    Ok(PointStruct::new(
                        qdrant_point_id(&point.id),
                        point.vector,
                        payload,
                    ))
                })
                .collect::<MemoryResult<Vec<_>>>()?;

            self.client
                .upsert_points(UpsertPointsBuilder::new(collection, points).wait(true))
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

    fn capabilities(&self) -> VectorStoreCapabilities {
        VectorStoreCapabilities {
            supports_server_side_hybrid: false,
            supports_sparse: false,
        }
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
    let score = keyword_score(content, text);
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
        .map_err(|error| MemoryError::BackendUnavailable(error.to_string()))
}

fn payload_id(payload: &JsonValue) -> String {
    payload
        .get("id")
        .and_then(JsonValue::as_str)
        .unwrap_or_default()
        .to_string()
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

fn qdrant_error(error: qdrant_client::QdrantError) -> MemoryError {
    MemoryError::BackendUnavailable(error.to_string())
}
