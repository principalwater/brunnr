// SPDX-License-Identifier: Apache-2.0

//! pgvector `VectorStore` adapter.
//!
//! Requires a PostgreSQL database with the `pgvector` extension installed. Connect via
//! [`PgVectorStore::connect`] and gate integration tests with `#[ignore]` unless
//! `PGVECTOR_URL` is set:
//!
//! ```ignore
//! #[tokio::test]
//! #[ignore = "requires PGVECTOR_URL"]
//! async fn pgvector_roundtrip() {
//!     let url = std::env::var("PGVECTOR_URL").expect("PGVECTOR_URL must be set");
//!     let store = PgVectorStore::connect(&url).await.expect("connect");
//!     // ...
//! }
//! ```

use std::sync::Arc;

use futures_util::{future::BoxFuture, FutureExt};
use pgvector::Vector;
use serde_json::Value;
use tokio_postgres::{Client, NoTls};

use crate::{
    vector::{is_safe_field_path, must_string_eq, payload_matches_filter},
    Distance, Filter, MemoryError, MemoryResult, PayloadIndex, VectorCollection,
    VectorMemoryBackend, VectorMemoryConfig, VectorPoint, VectorSearch, VectorSearchHit,
    VectorSearchSource, VectorStore, VectorStoreCapabilities,
};

pub type PgVectorBackend = VectorMemoryBackend<PgVectorStore>;

/// pgvector-backed `VectorStore`.
///
/// Uses PostgreSQL with the `pgvector` extension. Two tables per collection:
/// - `artesian_{name}_records`: stores the payload JSON
/// - `artesian_{name}_vectors`: stores the embedding vector
///
/// Keyword search uses PostgreSQL full-text search (`plainto_tsquery`). Vector search uses the
/// `<=>` cosine-distance operator from pgvector.
#[derive(Clone)]
pub struct PgVectorStore {
    client: Arc<Client>,
}

impl PgVectorStore {
    /// Connect to a PostgreSQL database and enable the `vector` extension.
    pub async fn connect(url: &str) -> MemoryResult<Self> {
        let (client, connection) = tokio_postgres::connect(url, NoTls)
            .await
            .map_err(|e| MemoryError::Database(e.to_string()))?;
        tokio::spawn(async move {
            if let Err(e) = connection.await {
                eprintln!("pgvector connection error: {e}");
            }
        });
        client
            .execute("CREATE EXTENSION IF NOT EXISTS vector", &[])
            .await
            .map_err(|e| MemoryError::Database(e.to_string()))?;
        Ok(Self {
            client: Arc::new(client),
        })
    }

    /// Convenience constructor — creates a [`PgVectorBackend`] using the default embedding config.
    pub fn memory_backend(self, collection: impl Into<String>) -> MemoryResult<PgVectorBackend> {
        VectorMemoryBackend::new(self, VectorMemoryConfig::new(collection))
    }
}

fn collection_names(collection: &str) -> (String, String) {
    let safe: String = collection
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    let safe = safe.trim_matches('_').to_string();
    (
        format!("artesian_{safe}_records"),
        format!("artesian_{safe}_vectors"),
    )
}

impl VectorStore for PgVectorStore {
    fn ensure_collection(&self, collection: VectorCollection) -> BoxFuture<'_, MemoryResult<()>> {
        async move {
            let (records, vectors) = collection_names(&collection.name);
            let dist_ops = match collection.distance {
                Distance::Cosine => "vector_cosine_ops",
                Distance::Dot => "vector_ip_ops",
                Distance::Euclidean => "vector_l2_ops",
            };
            let sql = format!(
                "CREATE TABLE IF NOT EXISTS {records} (
                     id TEXT PRIMARY KEY,
                     node_id TEXT NOT NULL,
                     payload JSONB NOT NULL
                 );
                 CREATE INDEX IF NOT EXISTS {records}_node_idx ON {records}(node_id);
                 CREATE TABLE IF NOT EXISTS {vectors} (
                     id TEXT PRIMARY KEY REFERENCES {records}(id) ON DELETE CASCADE,
                     embedding VECTOR({dim})
                 );
                 CREATE INDEX IF NOT EXISTS {vectors}_embed_idx
                     ON {vectors} USING ivfflat (embedding {dist_ops})
                     WITH (lists = 10);",
                dim = collection.dimensions,
            );
            // Errors creating the ivfflat index on empty tables are non-fatal on older pgvector
            // (requires at least one row). We ignore them and fall back to sequential scan.
            if let Err(e) = self.client.batch_execute(&sql).await {
                let msg = e.to_string();
                if !msg.contains("ivfflat") && !msg.contains("does not exist") {
                    return Err(MemoryError::Database(msg));
                }
                // Retry without the index
                let sql_no_idx = format!(
                    "CREATE TABLE IF NOT EXISTS {records} (
                         id TEXT PRIMARY KEY,
                         node_id TEXT NOT NULL,
                         payload JSONB NOT NULL
                     );
                     CREATE INDEX IF NOT EXISTS {records}_node_idx ON {records}(node_id);
                     CREATE TABLE IF NOT EXISTS {vectors} (
                         id TEXT PRIMARY KEY REFERENCES {records}(id) ON DELETE CASCADE,
                         embedding VECTOR({dim})
                     );",
                    dim = collection.dimensions,
                );
                self.client
                    .batch_execute(&sql_no_idx)
                    .await
                    .map_err(|e| MemoryError::Database(e.to_string()))?;
            }
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
            // node_id already has a btree column index from ensure_collection. For other
            // payload fields (tenancy fields, metadata.parent_node) build an expression
            // index over the JSONB path so equality filters use an index instead of a
            // sequential scan.
            if index.field == "node_id" || !is_safe_field_path(&index.field) {
                return Ok(());
            }
            let (records, _) = collection_names(&collection);
            let path = index.field.split('.').collect::<Vec<_>>().join(",");
            let index_name = format!("{records}_{}_idx", index.field.replace('.', "_"));
            let sql = format!(
                "CREATE INDEX IF NOT EXISTS {index_name} ON {records} ((payload #>> '{{{path}}}'))"
            );
            self.client
                .batch_execute(&sql)
                .await
                .map_err(|e| MemoryError::Database(e.to_string()))?;
            Ok(())
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
            let (records, vectors) = collection_names(&collection);
            for point in points {
                let payload =
                    serde_json::to_string(&point.payload).map_err(MemoryError::Payload)?;
                let node_id = point
                    .payload
                    .get("node_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                self.client
                    .execute(
                        &format!(
                            "INSERT INTO {records}(id, node_id, payload)
                             VALUES ($1, $2, $3::jsonb)
                             ON CONFLICT(id) DO UPDATE SET
                                 node_id = EXCLUDED.node_id,
                                 payload = EXCLUDED.payload"
                        ),
                        &[&point.id, &node_id, &payload],
                    )
                    .await
                    .map_err(|e| MemoryError::Database(e.to_string()))?;
                let vector = Vector::from(point.vector);
                self.client
                    .execute(
                        &format!(
                            "INSERT INTO {vectors}(id, embedding)
                             VALUES ($1, $2)
                             ON CONFLICT(id) DO UPDATE SET embedding = EXCLUDED.embedding"
                        ),
                        &[&point.id, &vector],
                    )
                    .await
                    .map_err(|e| MemoryError::Database(e.to_string()))?;
            }
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
                    self.vector_search(
                        &collection,
                        search.vector.as_deref().expect("checked above"),
                        &search.filter,
                        search.limit,
                    )
                    .await
                }
                VectorSearchSource::Vector => Ok(Vec::new()),
                VectorSearchSource::Keyword | VectorSearchSource::Hybrid => {
                    self.keyword_search(
                        &collection,
                        search.text.as_deref().unwrap_or(""),
                        &search.filter,
                        search.limit,
                    )
                    .await
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
            let (records, _) = collection_names(&collection);
            let row = self
                .client
                .query_opt(
                    &format!("SELECT id, payload FROM {records} WHERE id = $1"),
                    &[&point_id],
                )
                .await
                .map_err(|e| MemoryError::Database(e.to_string()))?;
            row.map(|r| row_to_point(&r)).transpose()
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
            if !is_safe_field_path(&field) {
                return Ok(Vec::new());
            }
            let (records, _) = collection_names(&collection);
            let path = field.split('.').collect::<Vec<_>>().join(",");
            let rows = self
                .client
                .query(
                    &format!(
                        "SELECT DISTINCT payload #>> '{{{path}}}' AS value
                         FROM {records}
                         WHERE payload #>> '{{{path}}}' IS NOT NULL
                         ORDER BY value"
                    ),
                    &[],
                )
                .await
                .map_err(|e| MemoryError::Database(e.to_string()))?;
            Ok(rows.iter().map(|row| row.get::<_, String>(0)).collect())
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

impl PgVectorStore {
    async fn vector_search(
        &self,
        collection: &str,
        vector: &[f32],
        filter: &Filter,
        limit: usize,
    ) -> MemoryResult<Vec<VectorSearchHit>> {
        let (records, vectors) = collection_names(collection);
        let embed = Vector::from(vector.to_vec());
        let rows = self
            .client
            .query(
                &format!(
                    "SELECT r.id, r.payload, (v.embedding <=> $1) AS distance
                     FROM {vectors} v
                     JOIN {records} r ON r.id = v.id
                     {where_clause}
                     ORDER BY v.embedding <=> $1
                     LIMIT $2",
                    where_clause = filter_where_clause(filter),
                ),
                &[&embed, &(limit.max(1) as i64)],
            )
            .await
            .map_err(|e| MemoryError::Database(e.to_string()))?;

        rows.iter()
            .filter_map(|row| {
                let point = row_to_point(row).ok()?;
                if !payload_matches_filter(&point.payload, filter) {
                    return None;
                }
                let distance: f64 = row.get(2);
                Some(Ok(VectorSearchHit {
                    point,
                    score: (1.0 / (1.0 + distance.max(0.0))) as f32,
                }))
            })
            .collect()
    }

    async fn keyword_search(
        &self,
        collection: &str,
        text: &str,
        filter: &Filter,
        limit: usize,
    ) -> MemoryResult<Vec<VectorSearchHit>> {
        let (records, _) = collection_names(collection);
        if text.trim().is_empty() {
            // Filter-only scan
            let rows = self
                .client
                .query(
                    &format!(
                        "SELECT id, payload FROM {records} {where_clause} LIMIT $1",
                        where_clause = filter_where_clause(filter),
                    ),
                    &[&(limit.max(1) as i64)],
                )
                .await
                .map_err(|e| MemoryError::Database(e.to_string()))?;
            return rows
                .iter()
                .filter_map(|row| {
                    let point = row_to_point(row).ok()?;
                    if payload_matches_filter(&point.payload, filter) {
                        Some(Ok(VectorSearchHit { point, score: 1.0 }))
                    } else {
                        None
                    }
                })
                .collect();
        }
        let rows = self
            .client
            .query(
                &format!(
                    "SELECT id, payload,
                         ts_rank(to_tsvector('english', payload->>'content'),
                                 plainto_tsquery('english', $1)) AS rank
                     FROM {records}
                     WHERE to_tsvector('english', payload->>'content')
                           @@ plainto_tsquery('english', $1)
                     {and_clause}
                     ORDER BY rank DESC
                     LIMIT $2",
                    and_clause = filter_and_clause(filter),
                ),
                &[&text, &(limit.max(1) as i64)],
            )
            .await
            .map_err(|e| MemoryError::Database(e.to_string()))?;

        rows.iter()
            .filter_map(|row| {
                let point = row_to_point(row).ok()?;
                if !payload_matches_filter(&point.payload, filter) {
                    return None;
                }
                let rank: f32 = row.get::<_, f64>(2) as f32;
                Some(Ok(VectorSearchHit {
                    point,
                    score: 1.0 / (1.0 + rank.abs()),
                }))
            })
            .collect()
    }
}

fn row_to_point(row: &tokio_postgres::Row) -> MemoryResult<VectorPoint> {
    let id: String = row.get(0);
    let payload_str: serde_json::Value = row
        .try_get::<_, serde_json::Value>(1)
        .map_err(|e| MemoryError::Database(e.to_string()))?;
    Ok(VectorPoint {
        id,
        vector: Vec::new(),
        payload: payload_str,
    })
}

/// SQL fragments pushing the filter's string-equality conditions: `node_id` via its
/// column, other fields via the JSONB path operator (`payload #>> '{a,b}'`). This lets
/// filter-only and keyword queries hit an index and return **all** matching rows (e.g.
/// every sibling chunk of a parent), not just the first page. Values are single-quote
/// escaped and field paths are validated, so they are safe to inline. Conditions not
/// pushed here are still enforced by `payload_matches_filter` after the DB returns rows.
fn sql_eq_parts(filter: &Filter) -> Vec<String> {
    must_string_eq(filter)
        .into_iter()
        .filter_map(|(field, value)| {
            let escaped = value.replace('\'', "''");
            if field == "node_id" {
                Some(format!("node_id = '{escaped}'"))
            } else if is_safe_field_path(field) {
                let path = field.split('.').collect::<Vec<_>>().join(",");
                Some(format!("payload #>> '{{{path}}}' = '{escaped}'"))
            } else {
                None
            }
        })
        .collect()
}

fn filter_where_clause(filter: &Filter) -> String {
    let parts = sql_eq_parts(filter);
    if parts.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", parts.join(" AND "))
    }
}

fn filter_and_clause(filter: &Filter) -> String {
    let parts = sql_eq_parts(filter);
    if parts.is_empty() {
        String::new()
    } else {
        format!("AND {}", parts.join(" AND "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Full integration test — requires a PostgreSQL instance with pgvector installed.
    ///
    /// Run with: PGVECTOR_URL=postgres://user:pass@localhost/testdb cargo test -- --ignored
    #[tokio::test]
    #[ignore = "requires PGVECTOR_URL env var pointing to a pgvector-enabled database"]
    async fn pgvector_store_roundtrip() {
        let url = std::env::var("PGVECTOR_URL").expect("PGVECTOR_URL must be set");
        let store = PgVectorStore::connect(&url).await.expect("connect");

        let collection = "integration_test_roundtrip";
        store
            .ensure_collection(VectorCollection {
                name: collection.to_string(),
                dimensions: 3,
                distance: Distance::Cosine,
                quantization: Default::default(),
            })
            .await
            .expect("ensure_collection");

        let point = VectorPoint {
            id: "test-id-1".to_string(),
            vector: vec![1.0, 0.0, 0.0],
            payload: serde_json::json!({
                "id": "test-id-1",
                "node_id": "node:test-1",
                "content": "pgvector integration test record",
                "tags": [],
                "tier": "l1-atom",
                "created_at": "2024-01-01T00:00:00Z"
            }),
        };
        store
            .upsert(collection, vec![point.clone()])
            .await
            .expect("upsert");

        let fetched = store
            .get(collection, "test-id-1")
            .await
            .expect("get")
            .expect("point should exist");
        assert_eq!(fetched.id, "test-id-1");

        let hits = store
            .search(
                collection,
                VectorSearch {
                    vector: Some(vec![1.0, 0.0, 0.0]),
                    text: None,
                    filter: Filter::default(),
                    limit: 5,
                    source: VectorSearchSource::Vector,
                },
            )
            .await
            .expect("vector search");
        assert!(
            !hits.is_empty(),
            "vector search should return at least one hit"
        );
        assert_eq!(hits[0].point.id, "test-id-1");
    }
}
