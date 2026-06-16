// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use brunnr_test_support::TempDir;
use futures_util::{future::BoxFuture, FutureExt};
use mimisbrunnr::{
    FilesBackend, MemoryBackend, MemoryId, MemoryQuery, MemoryRecord, MemoryResult, MemoryScope,
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
                        && query.scope.is_none_or(|scope| record.scope == Some(scope))
                        && query
                            .agent_id
                            .as_ref()
                            .is_none_or(|agent_id| record.agent_id.as_ref() == Some(agent_id))
                        && query
                            .session_id
                            .as_ref()
                            .is_none_or(|session_id| record.session_id.as_ref() == Some(session_id))
                        && query
                            .task_id
                            .as_ref()
                            .is_none_or(|task_id| record.task_id.as_ref() == Some(task_id))
                        && query
                            .user_id
                            .as_ref()
                            .is_none_or(|user_id| record.user_id.as_ref() == Some(user_id))
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
            let mut records = records.lock().expect("records lock should not be poisoned");
            if let Some(existing) = records.iter().find(|record| {
                record.content == memory.content
                    && record.node_id == memory.node_id.as_deref().unwrap_or("node:contract")
                    && record.scope == memory.scope
                    && record.agent_id == memory.agent_id
                    && record.session_id == memory.session_id
                    && record.task_id == memory.task_id
                    && record.user_id == memory.user_id
            }) {
                return Ok(existing.clone());
            }
            let id = MemoryId::new(format!("memory-{}", records.len() + 1));
            let mut record = MemoryRecord::new(
                id,
                memory
                    .node_id
                    .unwrap_or_else(|| "node:contract".to_string()),
                memory.content,
                memory.tags,
                memory.metadata,
                memory.tier,
            );
            record.scope = memory.scope;
            record.agent_id = memory.agent_id;
            record.session_id = memory.session_id;
            record.task_id = memory.task_id;
            record.user_id = memory.user_id;
            records.push(record.clone());
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
            ..VectorMemoryConfig::new("contract")
        },
        Arc::new(TestEmbedder),
    )
    .expect("backend should construct");

    assert_backend_contract(&backend).await;
}

#[tokio::test]
async fn vector_collections_isolate_two_projects_on_one_store() {
    let store = SqliteVecVectorStore::in_memory().expect("sqlite-vec store should open");
    let project_a = VectorMemoryBackend::with_embedder(
        store.clone(),
        VectorMemoryConfig {
            collection: "project-a".to_string(),
            dimensions: TEST_DIMENSIONS,
            ..VectorMemoryConfig::new("project-a")
        },
        Arc::new(TestEmbedder),
    )
    .expect("project A backend should construct");
    let project_b = VectorMemoryBackend::with_embedder(
        store,
        VectorMemoryConfig {
            collection: "project-b".to_string(),
            dimensions: TEST_DIMENSIONS,
            ..VectorMemoryConfig::new("project-b")
        },
        Arc::new(TestEmbedder),
    )
    .expect("project B backend should construct");

    project_a
        .store(StoreMemory {
            content: "shared query term belongs to project alpha".to_string(),
            tags: Vec::new(),
            metadata: BTreeMap::new(),
            tier: MemoryTier::L1Atom,
            node_id: Some("node:project-a".to_string()),
            created_at: None,
            scope: Some(MemoryScope::Shared),
            agent_id: None,
            session_id: None,
            task_id: None,
            user_id: Some("user-a".to_string()),
        })
        .await
        .expect("project A store should succeed");
    project_b
        .store(StoreMemory {
            content: "shared query term belongs to project beta".to_string(),
            tags: Vec::new(),
            metadata: BTreeMap::new(),
            tier: MemoryTier::L1Atom,
            node_id: Some("node:project-b".to_string()),
            created_at: None,
            scope: Some(MemoryScope::Shared),
            agent_id: None,
            session_id: None,
            task_id: None,
            user_id: Some("user-b".to_string()),
        })
        .await
        .expect("project B store should succeed");

    let hits_a = project_a
        .find(MemoryQuery::new("shared query term").with_limit(10))
        .await
        .expect("project A find should succeed");
    let hits_b = project_b
        .find(MemoryQuery::new("shared query term").with_limit(10))
        .await
        .expect("project B find should succeed");

    assert_eq!(hits_a.len(), 1);
    assert_eq!(hits_a[0].record.node_id, "node:project-a");
    assert_eq!(hits_b.len(), 1);
    assert_eq!(hits_b[0].record.node_id, "node:project-b");
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
            scope: None,
            agent_id: None,
            session_id: None,
            task_id: None,
            user_id: None,
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
            scope: None,
            agent_id: None,
            session_id: None,
            task_id: None,
            user_id: None,
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

    backend
        .store(StoreMemory {
            content: "tenant isolated context".to_string(),
            tags: Vec::new(),
            metadata: BTreeMap::new(),
            tier: MemoryTier::L1Atom,
            node_id: Some("node:tenant-a".to_string()),
            created_at: None,
            scope: Some(MemoryScope::Task),
            agent_id: None,
            session_id: None,
            task_id: Some("task-a".to_string()),
            user_id: None,
        })
        .await
        .expect("tenant store should succeed");
    backend
        .store(StoreMemory {
            content: "tenant isolated context".to_string(),
            tags: Vec::new(),
            metadata: BTreeMap::new(),
            tier: MemoryTier::L1Atom,
            node_id: Some("node:tenant-b".to_string()),
            created_at: None,
            scope: Some(MemoryScope::Task),
            agent_id: None,
            session_id: None,
            task_id: Some("task-b".to_string()),
            user_id: None,
        })
        .await
        .expect("tenant store should succeed");
    let mut tenant_query = MemoryQuery::new("isolated").with_limit(10);
    tenant_query.scope = Some(MemoryScope::Task);
    tenant_query.task_id = Some("task-a".to_string());
    let tenant_hits = backend
        .find(tenant_query)
        .await
        .expect("tenant find should succeed");
    assert_eq!(tenant_hits.len(), 1);
    assert_eq!(tenant_hits[0].record.node_id, "node:tenant-a");
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

#[tokio::test]
async fn large_content_is_chunked_so_recall_stays_bounded() {
    let store = SqliteVecVectorStore::in_memory().expect("sqlite-vec store should open");
    let backend = VectorMemoryBackend::with_embedder(
        store,
        VectorMemoryConfig {
            collection: "chunking".to_string(),
            dimensions: TEST_DIMENSIONS,
            ..VectorMemoryConfig::new("chunking")
        },
        Arc::new(TestEmbedder),
    )
    .expect("backend should construct");

    // A large document with a unique marker buried in the middle.
    let marker = "plum-pudding-seven";
    let big = format!(
        "{}\n\nthe decisive answer is {marker}\n\n{}",
        "alpha beta gamma. ".repeat(2_000),
        "delta epsilon zeta. ".repeat(2_000),
    );
    assert!(big.chars().count() > 50_000, "test needs a genuinely large record");
    backend
        .store(StoreMemory {
            content: big,
            tags: vec!["big".to_string()],
            metadata: Default::default(),
            tier: MemoryTier::L1Atom,
            node_id: Some("node:big".to_string()),
            created_at: None,
            scope: None,
            agent_id: None,
            session_id: None,
            task_id: None,
            user_id: None,
        })
        .await
        .expect("store should succeed");

    let hits = backend
        .find(MemoryQuery::new("decisive plum-pudding-seven").with_limit(5))
        .await
        .expect("find should succeed");

    assert!(!hits.is_empty(), "a chunk should be retrieved");
    // Bounded: no returned chunk is anywhere near the whole-document size.
    for hit in &hits {
        assert!(
            hit.record.content.chars().count() < 2_500,
            "recall must be bounded by chunk size, got {} chars",
            hit.record.content.chars().count()
        );
    }
    // Relevant: the buried marker survives in a retrieved chunk (not lost to truncation).
    assert!(
        hits.iter().any(|hit| hit.record.content.contains(marker)),
        "the relevant passage must be retrieved"
    );
    // Parent linkage: chunks point back to the source node for drill-down.
    assert!(hits.iter().all(|hit| hit.record.node_id.starts_with("node:big")));
}
