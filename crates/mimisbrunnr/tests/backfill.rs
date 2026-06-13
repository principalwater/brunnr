// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use brunnr_test_support::TempDir;
use mimisbrunnr::{
    backfill_directory, Distance, FilesBackend, MemoryBackend, MemoryQuery, MemoryResult,
    SqliteVecVectorStore, TextEmbedder, VectorMemoryBackend, VectorMemoryConfig,
};

#[tokio::test]
async fn backfill_is_idempotent_for_files_backend() {
    let tempdir = TempDir::new("backfill-files");
    let source = tempdir.join("source");
    std::fs::create_dir_all(&source).expect("source dir should be created");
    std::fs::write(
        source.join("memory.md"),
        "[2026-01-02] Durable imported memory",
    )
    .expect("source memory should be written");

    let backend = FilesBackend::new(tempdir.join("files"));
    assert_backfill_idempotency(&backend, &source).await;
}

#[tokio::test]
async fn backfill_is_idempotent_for_sqlite_vec_backend() {
    let tempdir = TempDir::new("backfill-sqlite");
    let source = tempdir.join("source");
    std::fs::create_dir_all(&source).expect("source dir should be created");
    std::fs::write(
        source.join("memory.json"),
        r#"{"content":"Durable imported memory","tier":"l1-atom","tags":["import"]}"#,
    )
    .expect("source memory should be written");

    let store = SqliteVecVectorStore::in_memory().expect("sqlite-vec should open");
    let backend = VectorMemoryBackend::with_embedder(
        store,
        VectorMemoryConfig {
            collection: "backfill".to_string(),
            dimensions: TEST_DIMENSIONS,
            distance: Distance::Cosine,
        },
        Arc::new(TestEmbedder),
    )
    .expect("backend should construct");
    assert_backfill_idempotency(&backend, &source).await;
}

async fn assert_backfill_idempotency(backend: &dyn MemoryBackend, source: &std::path::Path) {
    let first = backfill_directory(backend, source)
        .await
        .expect("first backfill should succeed");
    let second = backfill_directory(backend, source)
        .await
        .expect("second backfill should succeed");
    let hits = backend
        .find(MemoryQuery::new("imported").with_limit(10))
        .await
        .expect("find should succeed");

    assert_eq!(first.scanned, 1);
    assert_eq!(first.imported, 1);
    assert_eq!(first.skipped_duplicates, 0);
    assert_eq!(second.scanned, 1);
    assert_eq!(second.imported, 0);
    assert_eq!(second.skipped_duplicates, 1);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record.content, "Durable imported memory");
}

const TEST_DIMENSIONS: usize = 8;

struct TestEmbedder;

impl TextEmbedder for TestEmbedder {
    fn embed_query(&self, text: &str) -> MemoryResult<Vec<f32>> {
        Ok(test_embedding(text))
    }

    fn embed_passage(&self, text: &str) -> MemoryResult<Vec<f32>> {
        Ok(test_embedding(text))
    }
}

fn test_embedding(text: &str) -> Vec<f32> {
    let mut vector = vec![0.0; TEST_DIMENSIONS];
    for token in text.split_whitespace() {
        let index = token.bytes().fold(0usize, |hash, byte| {
            hash.wrapping_mul(31).wrapping_add(byte as usize)
        }) % TEST_DIMENSIONS;
        vector[index] += 1.0;
    }
    let magnitude = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for value in &mut vector {
            *value /= magnitude;
        }
    }
    vector
}
