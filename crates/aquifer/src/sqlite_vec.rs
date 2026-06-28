// SPDX-License-Identifier: Apache-2.0

use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex, Once},
    time::Duration,
};

use futures_util::{future::BoxFuture, FutureExt};
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::Value;

use crate::{
    vector::{is_safe_field_path, must_string_eq, payload_matches_filter},
    Distance, Filter, MemoryError, MemoryResult, PayloadIndex, VectorCollection,
    VectorMemoryBackend, VectorMemoryConfig, VectorPoint, VectorSearch, VectorSearchHit,
    VectorSearchSource, VectorStore, VectorStoreCapabilities,
};

pub type SqliteVecBackend = VectorMemoryBackend<SqliteVecVectorStore>;

#[derive(Debug, Clone)]
pub struct SqliteVecVectorStoreConfig {
    pub path: PathBuf,
}

impl SqliteVecVectorStoreConfig {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

#[derive(Debug, Clone)]
pub struct SqliteVecVectorStore {
    config: SqliteVecVectorStoreConfig,
    connection: Arc<Mutex<Connection>>,
}

impl SqliteVecVectorStore {
    pub fn open(config: SqliteVecVectorStoreConfig) -> MemoryResult<Self> {
        register_sqlite_vec();
        if let Some(parent) = config
            .path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }
        let connection = Connection::open(&config.path).map_err(sqlite_error)?;
        connection
            .busy_timeout(Duration::from_secs(5))
            .map_err(sqlite_error)?;
        connection
            .execute_batch(
                "PRAGMA journal_mode = WAL;
                 PRAGMA foreign_keys = ON;",
            )
            .map_err(sqlite_error)?;
        Ok(Self {
            config,
            connection: Arc::new(Mutex::new(connection)),
        })
    }

    pub fn in_memory() -> MemoryResult<Self> {
        register_sqlite_vec();
        let connection = Connection::open_in_memory().map_err(sqlite_error)?;
        connection
            .busy_timeout(Duration::from_secs(5))
            .map_err(sqlite_error)?;
        Ok(Self {
            config: SqliteVecVectorStoreConfig::new(Path::new(":memory:")),
            connection: Arc::new(Mutex::new(connection)),
        })
    }

    pub fn config(&self) -> &SqliteVecVectorStoreConfig {
        &self.config
    }

    pub fn memory_backend(
        self,
        collection: impl Into<String>,
    ) -> MemoryResult<VectorMemoryBackend<Self>> {
        VectorMemoryBackend::new(self, VectorMemoryConfig::new(collection))
    }
}

impl VectorStore for SqliteVecVectorStore {
    fn ensure_collection(&self, collection: VectorCollection) -> BoxFuture<'_, MemoryResult<()>> {
        async move {
            let tables = Tables::new(&collection.name)?;
            let connection = self.lock()?;
            let embedding_type = match collection.quantization {
                crate::VectorQuantization::Float32 => {
                    format!("float[{}]", collection.dimensions)
                }
                crate::VectorQuantization::Int8 => {
                    format!("int8[{}]", collection.dimensions)
                }
            };
            connection
                .execute_batch(&format!(
                    "CREATE TABLE IF NOT EXISTS _artesian_collection_meta (
                         name TEXT PRIMARY KEY,
                         quantization TEXT NOT NULL DEFAULT 'float32'
                     );
                     CREATE TABLE IF NOT EXISTS {records} (
                         id TEXT PRIMARY KEY,
                         node_id TEXT NOT NULL,
                         payload TEXT NOT NULL
                     );
                     CREATE VIRTUAL TABLE IF NOT EXISTS {fts}
                         USING fts5(id UNINDEXED, content);
                     CREATE VIRTUAL TABLE IF NOT EXISTS {vectors}
                         USING vec0(id TEXT PRIMARY KEY, embedding {embedding_type} distance_metric={distance});",
                    records = tables.records,
                    fts = tables.fts,
                    vectors = tables.vectors,
                    embedding_type = embedding_type,
                    distance = sqlite_distance(collection.distance),
                ))
                .map_err(sqlite_error)?;
            let quant_str = match collection.quantization {
                crate::VectorQuantization::Float32 => "float32",
                crate::VectorQuantization::Int8 => "int8",
            };
            connection
                .execute(
                    "INSERT OR REPLACE INTO _artesian_collection_meta(name, quantization) VALUES (?1, ?2)",
                    params![collection.name, quant_str],
                )
                .map_err(sqlite_error)?;
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
            let tables = Tables::new(&collection)?;
            let connection = self.lock()?;
            if index.field == "node_id" {
                connection
                    .execute(
                        &format!(
                            "CREATE INDEX IF NOT EXISTS {index} ON {records}(node_id)",
                            index = tables.node_index,
                            records = tables.records,
                        ),
                        [],
                    )
                    .map_err(sqlite_error)?;
                return Ok(());
            }
            // Expression index over a JSON payload field (supports nested dotted paths
            // such as `metadata.parent_node`) so equality filters use an index instead
            // of a full table scan. Unrecognized field shapes are simply not indexed.
            let Some(path) = json_field_path(&index.field) else {
                return Ok(());
            };
            let index_name = quote_identifier(&format!(
                "{}_{}_idx",
                sanitize_collection(&collection)?,
                sanitize_field(&index.field),
            ));
            connection
                .execute(
                    &format!(
                        "CREATE INDEX IF NOT EXISTS {index_name} ON {records}(json_extract(payload, '{path}'))",
                        records = tables.records,
                    ),
                    [],
                )
                .map_err(sqlite_error)?;
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
            let tables = Tables::new(&collection)?;
            let mut connection = self.lock()?;
            // Read quantization mode before starting the write transaction so the
            // explicit transaction doesn't need to mix reads of its own writes with
            // metadata reads (avoids WAL snapshot edge cases on some SQLite builds).
            let quant = collection_quantization(&connection, &collection);
            let transaction = connection.transaction().map_err(sqlite_error)?;
            for point in points {
                let point_id = point.id;
                let node_id = point
                    .payload
                    .get("node_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let content = point
                    .payload
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let payload = serde_json::to_string(&point.payload)?;
                transaction
                    .execute(
                        &format!(
                            "INSERT INTO {records}(id, node_id, payload)
                             VALUES (?1, ?2, ?3)
                             ON CONFLICT(id) DO UPDATE SET
                                 node_id = excluded.node_id,
                                 payload = excluded.payload",
                            records = tables.records,
                        ),
                        params![&point_id, node_id, payload],
                    )
                    .map_err(sqlite_error)?;
                transaction
                    .execute(
                        &format!("DELETE FROM {fts} WHERE id = ?1", fts = tables.fts),
                        params![&point_id],
                    )
                    .map_err(sqlite_error)?;
                transaction
                    .execute(
                        &format!(
                            "INSERT INTO {fts}(id, content)
                             VALUES (?1, ?2)",
                            fts = tables.fts,
                        ),
                        params![&point_id, content],
                    )
                    .map_err(sqlite_error)?;
                let (vec_sql, vec_val) =
                    vector_insert_sql_and_value(quant, &tables.vectors, &point.vector);
                transaction
                    .execute(&vec_sql, params![&point_id, vec_val])
                    .map_err(sqlite_error)?;
            }
            transaction.commit().map_err(sqlite_error)?;
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
            let tables = Tables::new(&collection)?;
            let connection = self.lock()?;
            match search.source {
                VectorSearchSource::Vector | VectorSearchSource::Hybrid
                    if search.vector.is_some() =>
                {
                    vector_search(
                        &connection,
                        &collection,
                        &tables,
                        search.vector.as_deref().expect("vector checked above"),
                        &search.filter,
                        search.limit,
                    )
                }
                VectorSearchSource::Vector => Ok(Vec::new()),
                VectorSearchSource::Keyword | VectorSearchSource::Hybrid => keyword_search(
                    &connection,
                    &tables,
                    search.text.as_deref().unwrap_or_default(),
                    &search.filter,
                    search.limit,
                ),
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
            let tables = Tables::new(&collection)?;
            let connection = self.lock()?;
            connection
                .query_row(
                    &format!(
                        "SELECT id, payload FROM {records} WHERE id = ?1",
                        records = tables.records,
                    ),
                    [point_id],
                    row_to_point,
                )
                .optional()
                .map_err(sqlite_error)
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
            let tables = Tables::new(&collection)?;
            let Some(path) = json_field_path(&field) else {
                return Ok(Vec::new());
            };
            let connection = self.lock()?;
            let mut statement = connection
                .prepare(&format!(
                    "SELECT DISTINCT json_extract(payload, '{path}')
                     FROM {records}
                     WHERE json_extract(payload, '{path}') IS NOT NULL
                     ORDER BY 1",
                    records = tables.records,
                ))
                .map_err(sqlite_error)?;
            let values = statement
                .query_map([], |row| row.get::<_, String>(0))
                .map_err(sqlite_error)?
                .collect::<Result<Vec<_>, _>>()
                .map_err(sqlite_error)?;
            Ok(values)
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

impl SqliteVecVectorStore {
    fn lock(&self) -> MemoryResult<std::sync::MutexGuard<'_, Connection>> {
        self.connection
            .lock()
            .map_err(|error| MemoryError::Database(error.to_string()))
    }

    /// Return the payload JSON for every record in `collection`.
    /// Intended for migration tooling that needs to inspect existing records.
    pub fn scan_all_records(&self, collection: &str) -> MemoryResult<Vec<serde_json::Value>> {
        let tables = Tables::new(collection)?;
        let connection = self.lock()?;
        let mut stmt = connection
            .prepare(&format!(
                "SELECT payload FROM {records}",
                records = tables.records,
            ))
            .map_err(sqlite_error)?;
        let rows: Vec<serde_json::Value> = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(sqlite_error)?
            .map(|r| {
                r.map_err(sqlite_error)
                    .and_then(|s| serde_json::from_str(&s).map_err(Into::into))
            })
            .collect::<MemoryResult<_>>()?;
        Ok(rows)
    }

    /// Delete the records identified by `ids` from `collection` (all three tables).
    pub fn delete_records(&self, collection: &str, ids: &[&str]) -> MemoryResult<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let tables = Tables::new(collection)?;
        let mut connection = self.lock()?;
        let tx = connection.transaction().map_err(sqlite_error)?;
        for id in ids {
            tx.execute(
                &format!(
                    "DELETE FROM {records} WHERE id = ?1",
                    records = tables.records
                ),
                params![id],
            )
            .map_err(sqlite_error)?;
            tx.execute(
                &format!("DELETE FROM {fts} WHERE id = ?1", fts = tables.fts),
                params![id],
            )
            .map_err(sqlite_error)?;
            tx.execute(
                &format!(
                    "DELETE FROM {vectors} WHERE id = ?1",
                    vectors = tables.vectors
                ),
                params![id],
            )
            .map_err(sqlite_error)?;
        }
        tx.commit().map_err(sqlite_error)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct Tables {
    records: String,
    fts: String,
    vectors: String,
    node_index: String,
}

impl Tables {
    fn new(collection: &str) -> MemoryResult<Self> {
        let name = sanitize_collection(collection)?;
        Ok(Self {
            records: quote_identifier(&format!("{name}_records")),
            fts: quote_identifier(&format!("{name}_fts")),
            vectors: quote_identifier(&format!("{name}_vec")),
            node_index: quote_identifier(&format!("{name}_node_idx")),
        })
    }
}

fn vector_search(
    connection: &Connection,
    collection: &str,
    tables: &Tables,
    vector: &[f32],
    filter: &Filter,
    limit: usize,
) -> MemoryResult<Vec<VectorSearchHit>> {
    let limit = limit.max(1);
    let quant = collection_quantization(connection, collection);
    let (query_sql, query_val) =
        vector_match_sql_and_value(quant, &tables.vectors, &tables.records, vector);
    let mut statement = connection.prepare(&query_sql).map_err(sqlite_error)?;
    let rows = statement
        .query_map(params![query_val, limit as i64], |row| {
            let point = row_to_point(row)?;
            let distance = row.get::<_, Option<f64>>(2)?.unwrap_or(f64::INFINITY);
            Ok((point, distance))
        })
        .map_err(sqlite_error)?;

    let mut hits = Vec::new();
    for row in rows {
        let (point, distance) = row.map_err(sqlite_error)?;
        if payload_matches_filter(&point.payload, filter) {
            hits.push(VectorSearchHit {
                point,
                score: distance_to_score(distance),
            });
        }
    }
    Ok(hits)
}

fn keyword_search(
    connection: &Connection,
    tables: &Tables,
    text: &str,
    filter: &Filter,
    limit: usize,
) -> MemoryResult<Vec<VectorSearchHit>> {
    if text.trim().is_empty() {
        return scan(connection, tables, filter, limit);
    }

    let query = fts_query(text);
    let mut statement = connection
        .prepare(&format!(
            "SELECT records.id, records.payload, bm25({fts}) AS rank
             FROM {fts} AS fts
             JOIN {records} AS records ON records.id = fts.id
             WHERE fts.content MATCH ?1
             ORDER BY rank
             LIMIT ?2",
            fts = tables.fts,
            records = tables.records,
        ))
        .map_err(sqlite_error)?;
    let rows = statement
        .query_map(params![query, limit.max(1) as i64], |row| {
            let point = row_to_point(row)?;
            let rank = row.get::<_, f64>(2)?;
            Ok((point, rank))
        })
        .map_err(sqlite_error)?;

    let mut hits = Vec::new();
    for row in rows {
        let (point, rank) = row.map_err(sqlite_error)?;
        if payload_matches_filter(&point.payload, filter) {
            hits.push(VectorSearchHit {
                point,
                score: bm25_to_score(rank),
            });
        }
    }
    Ok(hits)
}

fn scan(
    connection: &Connection,
    tables: &Tables,
    filter: &Filter,
    limit: usize,
) -> MemoryResult<Vec<VectorSearchHit>> {
    use rusqlite::types::Value as SqlValue;

    // Push simple `must` equality conditions into SQL so the query uses an index and
    // returns *all* matching rows (e.g. every sibling chunk of a parent), instead of
    // scanning the first N rows and filtering in Rust — which both was slow and could
    // miss matches that sat beyond the row cap. `payload_matches_filter` still runs as
    // the full-correctness backstop for conditions not pushed down.
    let mut clauses: Vec<String> = Vec::new();
    let mut binds: Vec<SqlValue> = Vec::new();
    for (field, text) in must_string_eq(filter) {
        if field == "node_id" {
            clauses.push("node_id = ?".to_string());
            binds.push(SqlValue::Text(text.to_string()));
        } else if let Some(path) = json_field_path(field) {
            clauses.push(format!("json_extract(payload, '{path}') = ?"));
            binds.push(SqlValue::Text(text.to_string()));
        }
    }
    let pushed = !clauses.is_empty();
    let where_clause = if pushed {
        format!("WHERE {}", clauses.join(" AND "))
    } else {
        String::new()
    };
    // When narrowed in SQL the limit is exact; otherwise keep headroom so the
    // Rust-side filter still has rows to match.
    let sql_limit = if pushed {
        limit.max(1)
    } else {
        limit.max(1) * 10
    };
    binds.push(SqlValue::Integer(sql_limit as i64));

    let mut statement = connection
        .prepare(&format!(
            "SELECT id, payload FROM {records} {where_clause} LIMIT ?",
            records = tables.records,
        ))
        .map_err(sqlite_error)?;
    let rows = statement
        .query_map(rusqlite::params_from_iter(binds), row_to_point)
        .map_err(sqlite_error)?;

    let mut hits = Vec::new();
    for row in rows {
        let point = row.map_err(sqlite_error)?;
        if payload_matches_filter(&point.payload, filter) {
            hits.push(VectorSearchHit { point, score: 1.0 });
        }
    }
    Ok(hits)
}

fn row_to_point(row: &rusqlite::Row<'_>) -> rusqlite::Result<VectorPoint> {
    let id = row.get::<_, String>(0)?;
    let payload = row.get::<_, String>(1)?;
    let payload = serde_json::from_str(&payload).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(
            payload.len(),
            rusqlite::types::Type::Text,
            Box::new(error),
        )
    })?;
    Ok(VectorPoint {
        id,
        vector: Vec::new(),
        payload,
    })
}

/// Map a filter field to a JSON path for `json_extract`, e.g. `metadata.parent_node`
/// -> `$.metadata.parent_node`. Returns `None` for unsafe field shapes (so the path is
/// never an injection vector and the equality filter falls back to the in-Rust check).
fn json_field_path(field: &str) -> Option<String> {
    is_safe_field_path(field).then(|| format!("$.{field}"))
}

/// Sanitize a field into an identifier fragment for an index name (`.` -> `_`).
fn sanitize_field(field: &str) -> String {
    field
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

fn sanitize_collection(collection: &str) -> MemoryResult<String> {
    let mut output = String::new();
    for character in collection.chars() {
        if character.is_ascii_alphanumeric() {
            output.push(character.to_ascii_lowercase());
        } else {
            output.push('_');
        }
    }
    let output = output.trim_matches('_').to_string();
    if output.is_empty() {
        return Err(MemoryError::Database(
            "collection name must contain at least one alphanumeric character".to_string(),
        ));
    }
    Ok(format!("artesian_{output}"))
}

fn quote_identifier(identifier: &str) -> String {
    format!("\"{}\"", identifier.replace('"', "\"\""))
}

fn sqlite_distance(distance: Distance) -> &'static str {
    match distance {
        Distance::Cosine => "cosine",
        Distance::Dot | Distance::Euclidean => "l2",
    }
}

fn vector_to_blob(vector: &[f32]) -> Vec<u8> {
    let mut output = Vec::with_capacity(std::mem::size_of_val(vector));
    for value in vector {
        output.extend_from_slice(&value.to_ne_bytes());
    }
    output
}

/// Symmetrically quantize float32 to a JSON int8 array (`"[127, -64, ...]"`).
///
/// Scale is `max_abs / 127.0`; the zero vector maps to all-zero values.
/// The JSON format is what sqlite-vec's `vec_int8()` SQL function expects — it sets the
/// internal `SQLITE_VEC_ELEMENT_TYPE_INT8` subtype so the vtab knows the element type.
fn quantize_f32_to_i8_json(vector: &[f32]) -> String {
    let max_abs = vector.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
    let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
    let ints: Vec<i64> = vector
        .iter()
        .map(|v| (v / scale).round().clamp(-127.0, 127.0) as i64)
        .collect();
    serde_json::to_string(&ints).unwrap_or_else(|_| "[0]".to_string())
}

/// Return the collection's quantization mode from `_artesian_collection_meta`, defaulting
/// to `float32` when the table or row does not exist (e.g. collections created before
/// this feature was added).
fn collection_quantization(
    connection: &rusqlite::Connection,
    collection: &str,
) -> crate::VectorQuantization {
    let Ok(quant_str) = connection.query_row(
        "SELECT quantization FROM _artesian_collection_meta WHERE name = ?1",
        params![collection],
        |row| row.get::<_, String>(0),
    ) else {
        return crate::VectorQuantization::Float32;
    };
    match quant_str.as_str() {
        "int8" => crate::VectorQuantization::Int8,
        _ => crate::VectorQuantization::Float32,
    }
}

/// The SQL fragment for the vector column in an INSERT and the bound value.
///
/// For Float32 we bind a raw blob; sqlite-vec reads the bytes as float32.
/// For Int8 we wrap with `vec_int8(?2)` which sets the INT8 subtype sqlite-vec requires.
fn vector_insert_sql_and_value(
    quant: crate::VectorQuantization,
    vectors_table: &str,
    vector: &[f32],
) -> (String, rusqlite::types::Value) {
    match quant {
        crate::VectorQuantization::Float32 => (
            format!("INSERT OR REPLACE INTO {vectors_table}(id, embedding) VALUES (?1, ?2)"),
            rusqlite::types::Value::Blob(vector_to_blob(vector)),
        ),
        crate::VectorQuantization::Int8 => (
            format!(
                "INSERT OR REPLACE INTO {vectors_table}(id, embedding) VALUES (?1, vec_int8(?2))"
            ),
            rusqlite::types::Value::Text(quantize_f32_to_i8_json(vector)),
        ),
    }
}

/// The SQL fragment for vector MATCH and the bound query value.
fn vector_match_sql_and_value(
    quant: crate::VectorQuantization,
    vectors_table: &str,
    records_table: &str,
    vector: &[f32],
) -> (String, rusqlite::types::Value) {
    match quant {
        crate::VectorQuantization::Float32 => (
            format!(
                "SELECT records.id, records.payload, vec.distance
                 FROM {vectors_table} AS vec
                 JOIN {records_table} AS records ON records.id = vec.id
                 WHERE vec.embedding MATCH ?1 AND k = ?2
                 ORDER BY vec.distance"
            ),
            rusqlite::types::Value::Blob(vector_to_blob(vector)),
        ),
        crate::VectorQuantization::Int8 => (
            format!(
                "SELECT records.id, records.payload, vec.distance
                 FROM {vectors_table} AS vec
                 JOIN {records_table} AS records ON records.id = vec.id
                 WHERE vec.embedding MATCH vec_int8(?1) AND k = ?2
                 ORDER BY vec.distance"
            ),
            rusqlite::types::Value::Text(quantize_f32_to_i8_json(vector)),
        ),
    }
}

fn distance_to_score(distance: f64) -> f32 {
    (1.0 / (1.0 + distance.max(0.0))) as f32
}

fn bm25_to_score(rank: f64) -> f32 {
    (1.0 / (1.0 + rank.abs())) as f32
}

fn fts_query(text: &str) -> String {
    text.split_whitespace()
        .map(|term| term.replace('"', "\"\""))
        .map(|term| format!("\"{term}\""))
        .collect::<Vec<_>>()
        .join(" OR ")
}

fn register_sqlite_vec() {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| {
        use rusqlite::ffi::sqlite3_auto_extension;
        use sqlite_vec::sqlite3_vec_init;

        type SqliteExtensionInit = unsafe extern "C" fn(
            *mut rusqlite::ffi::sqlite3,
            *mut *mut std::os::raw::c_char,
            *const rusqlite::ffi::sqlite3_api_routines,
        ) -> i32;

        // SAFETY: sqlite-vec exposes a SQLite extension initializer with the signature expected by
        // sqlite3_auto_extension. Registering it once makes vec0 available to future connections.
        unsafe {
            sqlite3_auto_extension(Some(std::mem::transmute::<*const (), SqliteExtensionInit>(
                sqlite3_vec_init as *const (),
            )));
        }
    });
}

fn sqlite_error(error: rusqlite::Error) -> MemoryError {
    MemoryError::Database(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Distance, VectorQuantization, VectorSearchSource};

    #[tokio::test]
    async fn int8_meta_is_written_and_read_back() {
        let store = SqliteVecVectorStore::in_memory().expect("in-memory store");
        store
            .ensure_collection(crate::VectorCollection {
                name: "test_int8".to_string(),
                dimensions: 4,
                distance: Distance::Cosine,
                quantization: VectorQuantization::Int8,
            })
            .await
            .expect("ensure_collection");

        // Verify meta table has the int8 row.
        let quant = {
            let conn = store.lock().expect("lock");
            collection_quantization(&conn, "test_int8")
        };
        assert_eq!(quant, VectorQuantization::Int8, "meta should show int8");

        // Verify upsert with int8 works.
        store
            .upsert(
                "test_int8",
                vec![VectorPoint {
                    id: "p1".to_string(),
                    vector: vec![1.0, 0.0, 0.0, 0.0],
                    payload: serde_json::json!({"content": "test", "node_id": "p1"}),
                }],
            )
            .await
            .expect("upsert int8 vector");

        // Verify search works.
        let hits = store
            .search(
                "test_int8",
                VectorSearch {
                    vector: Some(vec![1.0, 0.0, 0.0, 0.0]),
                    text: None,
                    filter: Default::default(),
                    limit: 1,
                    source: VectorSearchSource::Vector,
                },
            )
            .await
            .expect("search int8");
        assert_eq!(hits.len(), 1, "int8 search should return 1 hit");
    }

    #[test]
    fn quantize_f32_to_i8_json_is_valid_json_array() {
        let v = vec![0.5f32, -0.5, 1.0, -1.0, 0.0, 0.25, -0.25, 0.75];
        let json = quantize_f32_to_i8_json(&v);
        let parsed: Vec<i64> = serde_json::from_str(&json).expect("valid JSON array");
        assert_eq!(parsed.len(), v.len(), "int8 JSON: one int per dimension");
        for val in &parsed {
            assert!(*val >= -127 && *val <= 127, "int8 values in range: {val}");
        }
    }
}
