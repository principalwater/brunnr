// SPDX-License-Identifier: Apache-2.0

use futures_util::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::MemoryResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Distance {
    Cosine,
    Dot,
    Euclidean,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FilterValue {
    String(String),
    Number(f64),
    Bool(bool),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RangeFilter {
    pub field: String,
    pub gte: Option<f64>,
    pub gt: Option<f64>,
    pub lte: Option<f64>,
    pub lt: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "kebab-case")]
pub enum FilterCondition {
    Eq {
        field: String,
        value: FilterValue,
    },
    In {
        field: String,
        values: Vec<FilterValue>,
    },
    Range(RangeFilter),
    Exists {
        field: String,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Filter {
    pub must: Vec<FilterCondition>,
    pub should: Vec<FilterCondition>,
    pub must_not: Vec<FilterCondition>,
}

impl Filter {
    pub fn node_id(node_id: impl Into<String>) -> Self {
        Self {
            must: vec![FilterCondition::Eq {
                field: "node_id".to_string(),
                value: FilterValue::String(node_id.into()),
            }],
            should: Vec::new(),
            must_not: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.must.is_empty() && self.should.is_empty() && self.must_not.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorCollection {
    pub name: String,
    pub dimensions: usize,
    pub distance: Distance,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PayloadIndex {
    pub field: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorPoint {
    pub id: String,
    pub vector: Vec<f32>,
    pub payload: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorSearchSource {
    Keyword,
    Vector,
    Hybrid,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VectorSearch {
    pub vector: Option<Vec<f32>>,
    pub text: Option<String>,
    pub filter: Filter,
    pub limit: usize,
    pub source: VectorSearchSource,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorSearchHit {
    pub point: VectorPoint,
    pub score: f32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VectorStoreCapabilities {
    pub supports_server_side_hybrid: bool,
    pub supports_sparse: bool,
}

/// Engine-agnostic vector persistence contract.
///
/// `VectorStore` owns only collection/index management and normalized point search. Embedding,
/// memory tiering, payload schema, and RRF stay in `VectorMemoryBackend`, so engine adapters do
/// not leak backend-specific behavior into Brunnr's memory model.
///
/// ```
/// # use futures_util::{future::BoxFuture, FutureExt};
/// # use mimisbrunnr::{
/// #     Distance, Filter, MemoryResult, PayloadIndex, VectorCollection, VectorPoint,
/// #     VectorSearch, VectorSearchHit, VectorStore, VectorStoreCapabilities,
/// # };
/// # struct NoopStore;
/// # impl VectorStore for NoopStore {
/// #     fn ensure_collection(&self, _: VectorCollection) -> BoxFuture<'_, MemoryResult<()>> {
/// #         async { Ok(()) }.boxed()
/// #     }
/// #     fn ensure_payload_index(&self, _: &str, _: PayloadIndex) -> BoxFuture<'_, MemoryResult<()>> {
/// #         async { Ok(()) }.boxed()
/// #     }
/// #     fn upsert(&self, _: &str, _: Vec<VectorPoint>) -> BoxFuture<'_, MemoryResult<()>> {
/// #         async { Ok(()) }.boxed()
/// #     }
/// #     fn search(&self, _: &str, _: VectorSearch) -> BoxFuture<'_, MemoryResult<Vec<VectorSearchHit>>> {
/// #         async { Ok(Vec::new()) }.boxed()
/// #     }
/// #     fn get(&self, _: &str, _: &str) -> BoxFuture<'_, MemoryResult<Option<VectorPoint>>> {
/// #         async { Ok(None) }.boxed()
/// #     }
/// #     fn capabilities(&self) -> VectorStoreCapabilities {
/// #         VectorStoreCapabilities::default()
/// #     }
/// # }
/// let store = NoopStore;
/// assert!(!store.capabilities().supports_server_side_hybrid);
/// ```
pub trait VectorStore: Send + Sync {
    fn ensure_collection(&self, collection: VectorCollection) -> BoxFuture<'_, MemoryResult<()>>;

    fn ensure_payload_index(
        &self,
        collection: &str,
        index: PayloadIndex,
    ) -> BoxFuture<'_, MemoryResult<()>>;

    fn upsert(&self, collection: &str, points: Vec<VectorPoint>)
        -> BoxFuture<'_, MemoryResult<()>>;

    fn search(
        &self,
        collection: &str,
        search: VectorSearch,
    ) -> BoxFuture<'_, MemoryResult<Vec<VectorSearchHit>>>;

    fn get(
        &self,
        collection: &str,
        point_id: &str,
    ) -> BoxFuture<'_, MemoryResult<Option<VectorPoint>>>;

    fn capabilities(&self) -> VectorStoreCapabilities;
}

pub(crate) fn payload_matches_filter(payload: &Value, filter: &Filter) -> bool {
    let must = filter
        .must
        .iter()
        .all(|condition| payload_matches_condition(payload, condition));
    let should = filter.should.is_empty()
        || filter
            .should
            .iter()
            .any(|condition| payload_matches_condition(payload, condition));
    let must_not = filter
        .must_not
        .iter()
        .all(|condition| !payload_matches_condition(payload, condition));
    must && should && must_not
}

fn payload_matches_condition(payload: &Value, condition: &FilterCondition) -> bool {
    match condition {
        FilterCondition::Eq { field, value } => field_value(payload, field)
            .is_some_and(|candidate| filter_values_equal(candidate, value)),
        FilterCondition::In { field, values } => {
            field_value(payload, field).is_some_and(|candidate| {
                values
                    .iter()
                    .any(|expected| filter_values_equal(candidate, expected))
            })
        }
        FilterCondition::Range(range) => field_value(payload, &range.field)
            .and_then(Value::as_f64)
            .is_some_and(|candidate| range_matches(candidate, range)),
        FilterCondition::Exists { field } => field_value(payload, field).is_some(),
    }
}

fn field_value<'a>(payload: &'a Value, field: &str) -> Option<&'a Value> {
    let mut value = payload;
    for part in field.split('.') {
        value = value.get(part)?;
    }
    Some(value)
}

fn filter_values_equal(candidate: &Value, expected: &FilterValue) -> bool {
    match expected {
        FilterValue::String(expected) => candidate.as_str() == Some(expected.as_str()),
        FilterValue::Number(expected) => candidate
            .as_f64()
            .is_some_and(|candidate| (candidate - expected).abs() <= f64::EPSILON),
        FilterValue::Bool(expected) => candidate.as_bool() == Some(*expected),
    }
}

fn range_matches(candidate: f64, range: &RangeFilter) -> bool {
    range.gte.is_none_or(|value| candidate >= value)
        && range.gt.is_none_or(|value| candidate > value)
        && range.lte.is_none_or(|value| candidate <= value)
        && range.lt.is_none_or(|value| candidate < value)
}
