// SPDX-License-Identifier: Apache-2.0

use futures_util::{future::BoxFuture, FutureExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{MemoryResult, SHARED_PROJECT};

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
        let mut filter = Self::default();
        filter.must_eq("node_id", node_id);
        filter
    }

    pub fn must_eq(&mut self, field: impl Into<String>, value: impl Into<String>) {
        self.must.push(FilterCondition::Eq {
            field: field.into(),
            value: FilterValue::String(value.into()),
        });
    }

    pub fn project_union(project: &str) -> Self {
        let mut filter = Self::default();
        filter.add_project_union(project);
        filter
    }

    pub fn add_project_union(&mut self, project: &str) {
        let project = project.trim();
        let project = if project.is_empty() {
            SHARED_PROJECT
        } else {
            project
        };
        self.should.push(FilterCondition::Eq {
            field: "project".to_string(),
            value: FilterValue::String(project.to_string()),
        });
        if project != SHARED_PROJECT {
            self.should.push(FilterCondition::Eq {
                field: "project".to_string(),
                value: FilterValue::String(SHARED_PROJECT.to_string()),
            });
        }
        self.should.push(FilterCondition::Exists {
            field: "project".to_string(),
        });
    }

    pub fn is_empty(&self) -> bool {
        self.must.is_empty() && self.should.is_empty() && self.must_not.is_empty()
    }
}

/// The `must` string-equality conditions of a filter, as `(field, value)` pairs.
///
/// Adapters push these into their native filter / SQL `WHERE` so equality lookups use an
/// index and return **all** matching rows (e.g. every sibling chunk of a parent), rather
/// than scanning the first N rows. `payload_matches_filter` still runs as the
/// full-correctness backstop for any conditions not pushed down. Shared so every vector
/// backend treats filter-only retrieval identically.
pub(crate) fn must_string_eq(filter: &Filter) -> Vec<(&str, &str)> {
    filter
        .must
        .iter()
        .filter_map(|condition| match condition {
            FilterCondition::Eq {
                field,
                value: FilterValue::String(value),
            } => Some((field.as_str(), value.as_str())),
            _ => None,
        })
        .collect()
}

/// Whether a payload field is a safe dotted identifier (ASCII alphanumeric, `_`, `.`),
/// so it can be interpolated into an engine-specific path/index without injection risk.
pub(crate) fn is_safe_field_path(field: &str) -> bool {
    !field.is_empty()
        && field
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
}

/// Storage format for embedding vectors.
///
/// `Int8` stores each dimension as a signed 8-bit integer (symmetric scalar quantization to
/// the range `[−127, 127]`). This cuts the per-vector storage footprint by 4× vs `Float32`
/// with a modest, measurable recall cost — inspired by the LEANN computable-embedding approach
/// (arXiv: LEANN, SIGMOD 2024). The actual savings are honest: 4× per vector stored, not the
/// 97% LEANN claims (which comes from graph-based recomputation we do not implement here).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum VectorQuantization {
    /// Standard 32-bit float (4 bytes/dim). The default.
    #[default]
    Float32,
    /// Signed 8-bit integer (1 byte/dim). 4× storage reduction; slight recall cost.
    Int8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorCollection {
    pub name: String,
    pub dimensions: usize,
    pub distance: Distance,
    /// Embedding quantization for this collection. Defaults to `Float32`.
    pub quantization: VectorQuantization,
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
/// not leak backend-specific behavior into Artesian's memory model.
///
/// ```
/// # use futures_util::{future::BoxFuture, FutureExt};
/// # use aquifer::{
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

    /// Upsert points without waiting for the index to finish building. Returns immediately after
    /// the server accepts the batch. Call `flush_upsert` once after all batches are sent.
    ///
    /// The default implementation delegates to `upsert` (with wait), which is correct for all
    /// in-process backends (Files, SQLite-vec). Qdrant overrides this to use `wait=false`.
    fn upsert_no_wait(
        &self,
        collection: &str,
        points: Vec<VectorPoint>,
    ) -> BoxFuture<'_, MemoryResult<()>> {
        self.upsert(collection, points)
    }

    /// Issue a final synchronisation barrier after a series of `upsert_no_wait` calls. No-op
    /// for backends that do not need it (Files, SQLite-vec). Qdrant overrides this to send an
    /// empty `upsert_points(...).wait(true)` so all preceding async batches are indexed.
    fn flush_upsert(&self, _collection: &str) -> BoxFuture<'_, MemoryResult<()>> {
        async { Ok(()) }.boxed()
    }

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

    fn distinct_payload_values(
        &self,
        _collection: &str,
        _field: &str,
    ) -> BoxFuture<'_, MemoryResult<Vec<String>>> {
        async { Ok(Vec::new()) }.boxed()
    }

    fn capabilities(&self) -> VectorStoreCapabilities;
}

impl<T: VectorStore + ?Sized> VectorStore for &T {
    fn ensure_collection(&self, collection: VectorCollection) -> BoxFuture<'_, MemoryResult<()>> {
        (**self).ensure_collection(collection)
    }

    fn ensure_payload_index(
        &self,
        collection: &str,
        index: PayloadIndex,
    ) -> BoxFuture<'_, MemoryResult<()>> {
        (**self).ensure_payload_index(collection, index)
    }

    fn upsert(
        &self,
        collection: &str,
        points: Vec<VectorPoint>,
    ) -> BoxFuture<'_, MemoryResult<()>> {
        (**self).upsert(collection, points)
    }

    fn upsert_no_wait(
        &self,
        collection: &str,
        points: Vec<VectorPoint>,
    ) -> BoxFuture<'_, MemoryResult<()>> {
        (**self).upsert_no_wait(collection, points)
    }

    fn flush_upsert(&self, collection: &str) -> BoxFuture<'_, MemoryResult<()>> {
        (**self).flush_upsert(collection)
    }

    fn search(
        &self,
        collection: &str,
        search: VectorSearch,
    ) -> BoxFuture<'_, MemoryResult<Vec<VectorSearchHit>>> {
        (**self).search(collection, search)
    }

    fn get(
        &self,
        collection: &str,
        point_id: &str,
    ) -> BoxFuture<'_, MemoryResult<Option<VectorPoint>>> {
        (**self).get(collection, point_id)
    }

    fn distinct_payload_values(
        &self,
        collection: &str,
        field: &str,
    ) -> BoxFuture<'_, MemoryResult<Vec<String>>> {
        (**self).distinct_payload_values(collection, field)
    }

    fn capabilities(&self) -> VectorStoreCapabilities {
        (**self).capabilities()
    }
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
        FilterCondition::Eq { field, value } => {
            field_value(payload, field).is_some_and(|candidate| value_matches(candidate, value))
        }
        FilterCondition::In { field, values } => {
            field_value(payload, field).is_some_and(|candidate| {
                values
                    .iter()
                    .any(|expected| value_matches(candidate, expected))
            })
        }
        FilterCondition::Range(range) => field_value(payload, &range.field)
            .and_then(Value::as_f64)
            .is_some_and(|candidate| range_matches(candidate, range)),
        // `Exists` is the historical adapter name for Qdrant's `is_empty` condition in this
        // filter model. Treat it as the absent/null arm so the Rust backstop matches Qdrant.
        FilterCondition::Exists { field } => field_value(payload, field).is_none_or(Value::is_null),
    }
}

fn field_value<'a>(payload: &'a Value, field: &str) -> Option<&'a Value> {
    let mut value = payload;
    for part in field.split('.') {
        value = value.get(part)?;
    }
    Some(value)
}

/// Match a payload value against an expected scalar. When the payload value is an array
/// (e.g. the `tags` field), this is "array contains expected" — mirroring Qdrant keyword-array
/// match semantics, so an `Eq`/`In` filter on `tags` behaves identically across backends.
fn value_matches(candidate: &Value, expected: &FilterValue) -> bool {
    if let Some(array) = candidate.as_array() {
        return array.iter().any(|item| filter_values_equal(item, expected));
    }
    filter_values_equal(candidate, expected)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn must_string_eq_extracts_only_string_equality_conditions() {
        let mut filter = Filter::default();
        filter.must_eq("node_id", "n1");
        filter.must_eq("metadata.parent_node", "node:p");
        filter.must.push(FilterCondition::Range(RangeFilter {
            field: "age".to_string(),
            gte: Some(1.0),
            gt: None,
            lte: None,
            lt: None,
        }));
        assert_eq!(
            must_string_eq(&filter),
            vec![("node_id", "n1"), ("metadata.parent_node", "node:p")]
        );
    }

    #[test]
    fn is_safe_field_path_accepts_only_dotted_identifiers() {
        assert!(is_safe_field_path("metadata.parent_node"));
        assert!(is_safe_field_path("node_id"));
        assert!(!is_safe_field_path(""));
        assert!(!is_safe_field_path("a b"));
        assert!(!is_safe_field_path("x'; DROP TABLE y;--"));
    }
}
