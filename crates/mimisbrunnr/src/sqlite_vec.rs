// SPDX-License-Identifier: Apache-2.0

use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex, Once},
};

use futures_util::{future::BoxFuture, FutureExt};
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::Value;

use crate::{
    vector::payload_matches_filter, Distance, Filter, MemoryError, MemoryResult, PayloadIndex,
    VectorCollection, VectorMemoryBackend, VectorMemoryConfig, VectorPoint, VectorSearch,
    VectorSearchHit, VectorSearchSource, VectorStore, VectorStoreCapabilities,
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
            connection
                .execute_batch(&format!(
                    "CREATE TABLE IF NOT EXISTS {records} (
                         id TEXT PRIMARY KEY,
                         node_id TEXT NOT NULL,
                         payload TEXT NOT NULL
                     );
                     CREATE VIRTUAL TABLE IF NOT EXISTS {fts}
                         USING fts5(id UNINDEXED, content);
                     CREATE VIRTUAL TABLE IF NOT EXISTS {vectors}
                         USING vec0(id TEXT PRIMARY KEY, embedding float[{dimensions}] distance_metric={distance});",
                    records = tables.records,
                    fts = tables.fts,
                    vectors = tables.vectors,
                    dimensions = collection.dimensions,
                    distance = sqlite_distance(collection.distance),
                ))
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
            if index.field != "node_id" {
                return Ok(());
            }
            let tables = Tables::new(&collection)?;
            let connection = self.lock()?;
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
                transaction
                    .execute(
                        &format!(
                            "INSERT OR REPLACE INTO {vectors}(id, embedding)
                             VALUES (?1, ?2)",
                            vectors = tables.vectors,
                        ),
                        params![&point_id, vector_to_blob(&point.vector)],
                    )
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
    tables: &Tables,
    vector: &[f32],
    filter: &Filter,
    limit: usize,
) -> MemoryResult<Vec<VectorSearchHit>> {
    let limit = limit.max(1);
    let mut statement = connection
        .prepare(&format!(
            "SELECT records.id, records.payload, vec.distance
             FROM {vectors} AS vec
             JOIN {records} AS records ON records.id = vec.id
             WHERE vec.embedding MATCH ?1 AND k = ?2
             ORDER BY vec.distance",
            vectors = tables.vectors,
            records = tables.records,
        ))
        .map_err(sqlite_error)?;
    let rows = statement
        .query_map(params![vector_to_blob(vector), limit as i64], |row| {
            let point = row_to_point(row)?;
            let distance = row.get::<_, f64>(2)?;
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
            "SELECT records.id, records.payload, bm25(fts) AS rank
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
    let mut statement = connection
        .prepare(&format!(
            "SELECT id, payload FROM {records} LIMIT ?1",
            records = tables.records,
        ))
        .map_err(sqlite_error)?;
    let rows = statement
        .query_map([limit.max(1) as i64], row_to_point)
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
    Ok(format!("brunnr_{output}"))
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
