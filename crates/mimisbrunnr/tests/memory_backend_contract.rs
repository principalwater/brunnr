// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use brunnr_test_support::TempDir;
use futures_util::{future::BoxFuture, FutureExt};
use mimisbrunnr::{
    Distance, FilesBackend, MemoryBackend, MemoryId, MemoryQuery, MemoryRecord, MemoryResult,
    MemoryTier, RrfOptions, SearchHit, SqliteVecVectorStore, StoreMemory, TextEmbedder,
    VectorMemoryBackend, VectorMemoryConfig,
};

#[derive(Debug, Default)]
struct MockMemoryBackend {
    records: Arc<Mutex<Vec<MemoryRecord>>>,
}

impl MemoryBackend for MockMemoryBackend {
    fn find(&self, query: MemoryQuery) -> BoxFuture<'_, MemoryResult<Vec<SearchHit>>> {
        let records = Arc::clone(&self.records);
        async move {
            let needle = query.text.to_ascii_lowercase();
            let hits = records
                .lock()
                .expect("records lock should not be poisoned")
                .iter()
                .filter(|record| {
                    record.content.to_ascii_lowercase().contains(&needle)
                        && query
                            .node_id
                            .as_ref()
                            .is_none_or(|node_id| record.node_id == *node_id)
                })
                .cloned()
                .map(|record| SearchHit::keyword(record, 1.0))
                .collect();
            Ok(hits)
        }
        .boxed()
    }

    fn store(&self, memory: StoreMemory) -> BoxFuture<'_, MemoryResult<MemoryRecord>> {
        let records = Arc::clone(&self.records);
        async move {
            let id = MemoryId::new(format!("memory-{}", records.lock().unwrap().len() + 1));
            let record = MemoryRecord::new(
                id,
                memory
                    .node_id
                    .unwrap_or_else(|| "node:contract".to_string()),
                memory.content,
                memory.tags,
                memory.metadata,
                memory.tier,
            );
            records.lock().unwrap().push(record.clone());
            Ok(record)
        }
        .boxed()
    }

    fn get_node(&self, node_id: &str) -> BoxFuture<'_, MemoryResult<Option<MemoryRecord>>> {
        let records = Arc::clone(&self.records);
        let node_id = node_id.to_string();
        async move {
            Ok(records
                .lock()
                .expect("records lock should not be poisoned")
                .iter()
                .find(|record| record.node_id == node_id)
                .cloned())
        }
        .boxed()
    }
}

#[tokio::test]
async fn mock_backend_satisfies_memory_contract() {
    assert_backend_contract(&MockMemoryBackend::default()).await;
}

#[tokio::test]
async fn files_backend_satisfies_memory_contract() {
    let tempdir = TempDir::new("files-contract");
    assert_backend_contract(&FilesBackend::new(tempdir.path())).await;
}

#[tokio::test]
async fn sqlite_vec_backend_satisfies_memory_contract() {
    let store = SqliteVecVectorStore::in_memory().expect("sqlite-vec store should open");
    let backend = VectorMemoryBackend::with_embedder(
        store,
        VectorMemoryConfig {
            collection: "contract".to_string(),
            dimensions: TEST_DIMENSIONS,
            distance: Distance::Cosine,
        },
        Arc::new(TestEmbedder),
    )
    .expect("backend should construct");

    assert_backend_contract(&backend).await;
}

async fn assert_backend_contract<B: MemoryBackend>(backend: &B) {
    let stored = backend
        .store(StoreMemory {
            content: "Brunnr stores durable context".to_string(),
            tags: vec!["contract".to_string()],
            metadata: BTreeMap::new(),
            tier: MemoryTier::L1Atom,
            node_id: Some("node:test".to_string()),
            created_at: None,
        })
        .await
        .expect("store should succeed");

    backend
        .store(StoreMemory {
            content: "hybrid retrieval".to_string(),
            tags: Vec::new(),
            metadata: BTreeMap::new(),
            tier: MemoryTier::L1Atom,
            node_id: Some("node:rrf".to_string()),
            created_at: None,
        })
        .await
        .expect("store should succeed");

    let found = backend
        .find(MemoryQuery::new("durable").with_limit(5))
        .await
        .expect("find should succeed");
    let drill_down = backend
        .get_node("node:test")
        .await
        .expect("get_node should succeed");
    let hits = backend
        .hybrid_rrf(
            MemoryQuery::new("hybrid").with_limit(5),
            MemoryQuery::new("retrieval").with_limit(5),
            RrfOptions {
                limit: 5,
                ..RrfOptions::default()
            },
        )
        .await
        .expect("hybrid search should succeed");

    assert!(
        found.iter().any(|hit| hit.record.node_id == "node:test"),
        "find should return the durable memory, got {found:?}"
    );
    assert_eq!(drill_down, Some(stored));
    assert!(
        hits.iter().any(|hit| hit.record.node_id == "node:rrf"),
        "hybrid RRF should return the retrieval memory, got {hits:?}"
    );
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
