// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use aquifer::{
    FilesBackend, MemoryBackend, MemoryId, MemoryQuery, MemoryRecord, MemoryResult, MemoryScope,
    MemoryTier, RrfOptions, SearchHit, SqliteVecVectorStore, StoreMemory, TextEmbedder,
    VectorMemoryBackend, VectorMemoryConfig,
};
use artesian_test_support::TempDir;
use futures_util::{future::BoxFuture, FutureExt};

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

/// A tag filter is an explicit selection (e.g. always-inject project invariants), so a
/// tag-filtered find must return the tagged record regardless of query relevance and exclude
/// untagged records — uniformly across the lexical (files) and vector (sqlite-vec) backends.
async fn assert_tag_filter_always_injects<B: MemoryBackend>(backend: &B) {
    let invariant = StoreMemory {
        content: "never delete the production database".to_string(),
        tags: vec!["invariant".to_string()],
        node_id: Some("node:inv".to_string()),
        ..StoreMemory::atom("")
    };
    let plain = StoreMemory {
        content: "deployments run nightly".to_string(),
        node_id: Some("node:plain".to_string()),
        ..StoreMemory::atom("")
    };
    backend.store(invariant).await.expect("store invariant");
    backend.store(plain).await.expect("store plain");

    // Query text deliberately shares no token with the invariant.
    let mut query = MemoryQuery::new("zzz unrelated query token").with_limit(5);
    query.tags = vec!["invariant".to_string()];
    let hits = backend.find(query).await.expect("tag find should succeed");
    assert!(
        hits.iter()
            .any(|hit| hit.record.content.contains("production database")),
        "tag-filtered find must always return the tagged record regardless of relevance, got {hits:?}"
    );
    assert!(
        !hits
            .iter()
            .any(|hit| hit.record.content.contains("nightly")),
        "tag filter must exclude untagged records, got {hits:?}"
    );
}

#[tokio::test]
async fn files_backend_tag_filter_always_injects() {
    let tempdir = TempDir::new("files-tag-inject");
    assert_tag_filter_always_injects(&FilesBackend::new(tempdir.path())).await;
}

#[tokio::test]
async fn sqlite_vec_backend_tag_filter_always_injects() {
    let store = SqliteVecVectorStore::in_memory().expect("sqlite-vec store should open");
    let backend = VectorMemoryBackend::with_embedder(
        store,
        VectorMemoryConfig {
            collection: "tag-inject".to_string(),
            dimensions: TEST_DIMENSIONS,
            ..VectorMemoryConfig::new("tag-inject")
        },
        Arc::new(TestEmbedder),
    )
    .expect("backend should construct");
    assert_tag_filter_always_injects(&backend).await;
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
            content: "Artesian stores durable context".to_string(),
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
    assert!(
        big.chars().count() > 50_000,
        "test needs a genuinely large record"
    );
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
    // Bounded: small-to-big expansion stays within the (adaptive) budget ceiling,
    // never anywhere near the whole-document size.
    let ceiling = VectorMemoryConfig::new("chunking").parent_context_max_chars;
    for hit in &hits {
        assert!(
            hit.record.content.chars().count() <= ceiling,
            "recall must be bounded by parent_context_max_chars ({ceiling}), got {} chars",
            hit.record.content.chars().count()
        );
    }
    // Relevant: the buried marker survives in the retrieved window (not lost to truncation).
    let marker_hit = hits
        .iter()
        .find(|hit| hit.record.content.contains(marker))
        .expect("the relevant passage must be retrieved");
    // Coherent (small-to-big): the matched chunk was expanded with its surrounding
    // parent-section context, so the window is larger than a single ~1600-char chunk.
    assert!(
        marker_hit.record.content.chars().count() > 1_600,
        "small-to-big should expand the matched chunk with neighbouring context, got {} chars",
        marker_hit.record.content.chars().count()
    );
    // Single source: same-parent chunk hits collapse into one expanded hit whose
    // node_id is the parent (drill-down target for the full document).
    assert_eq!(
        hits.iter()
            .filter(|hit| hit.record.node_id == "node:big")
            .count(),
        1,
        "same-parent hits must dedup to a single expanded hit"
    );
    assert!(hits.iter().all(|hit| hit.record.node_id == "node:big"));

    // Drill-down: get_node on the parent reconstructs the full source document.
    let full = backend
        .get_node("node:big")
        .await
        .expect("get_node should succeed")
        .expect("parent node should reconstruct from chunks");
    assert!(
        full.content.chars().count() > 50_000,
        "drill-down must return the complete source, got {} chars",
        full.content.chars().count()
    );
    assert!(
        full.content.contains(marker),
        "reconstructed document must contain the buried marker"
    );
}

/// Small-to-big must be a no-op for single-chunk records: every stored document is
/// small enough to be one chunk, so retrieval returns the records verbatim (same
/// node_id, same content, no expansion). This guards the invariant that the
/// scaling benchmark — built from small single-chunk documents — is unchanged by
/// small-to-big / adaptive budgeting.
#[tokio::test]
async fn single_chunk_records_are_unaffected_by_small_to_big() {
    let store = SqliteVecVectorStore::in_memory().expect("sqlite-vec store should open");
    let backend = VectorMemoryBackend::with_embedder(
        store,
        VectorMemoryConfig {
            collection: "small".to_string(),
            dimensions: TEST_DIMENSIONS,
            ..VectorMemoryConfig::new("small")
        },
        Arc::new(TestEmbedder),
    )
    .expect("backend should construct");

    let docs = [
        ("node:a", "alpha caching ttl is ninety seconds"),
        ("node:b", "beta auth tokens expire in fifteen minutes"),
        ("node:c", "gamma payment retries are idempotent by key"),
    ];
    for (node, body) in docs {
        backend
            .store(StoreMemory {
                content: body.to_string(),
                tags: vec![],
                metadata: Default::default(),
                tier: MemoryTier::L1Atom,
                node_id: Some(node.to_string()),
                created_at: None,
                scope: None,
                agent_id: None,
                session_id: None,
                task_id: None,
                user_id: None,
            })
            .await
            .expect("store should succeed");
    }

    let hits = backend
        .find(MemoryQuery::new("auth tokens expire").with_limit(5))
        .await
        .expect("find should succeed");

    assert!(!hits.is_empty(), "a record should be retrieved");
    for hit in &hits {
        // No chunking happened, so no expansion markers and the content is verbatim.
        assert!(
            !hit.record.metadata.contains_key("parent_node"),
            "single-chunk records must not carry chunk metadata"
        );
        assert!(
            !hit.record.metadata.contains_key("expanded_from_chunk"),
            "single-chunk records must not be expanded"
        );
        let original = docs
            .iter()
            .find(|(node, _)| *node == hit.record.node_id)
            .map(|(_, body)| *body)
            .expect("hit must map to a stored doc");
        assert_eq!(
            hit.record.content, original,
            "single-chunk content must be returned verbatim"
        );
    }
}

/// Small-to-big must find a parent's sibling chunks regardless of where they sit in the
/// collection. Storing many unrelated docs first pushes the multi-chunk document's chunks
/// to high row positions; a scan that only looked at the first rows would miss them and
/// silently skip expansion. The indexed sibling lookup must still return the full window.
#[tokio::test]
async fn small_to_big_finds_siblings_beyond_the_scan_window() {
    let store = SqliteVecVectorStore::in_memory().expect("sqlite-vec store should open");
    let backend = VectorMemoryBackend::with_embedder(
        store,
        VectorMemoryConfig {
            collection: "scale".to_string(),
            dimensions: TEST_DIMENSIONS,
            ..VectorMemoryConfig::new("scale")
        },
        Arc::new(TestEmbedder),
    )
    .expect("backend should construct");

    for i in 0..60 {
        backend
            .store(StoreMemory {
                node_id: Some(format!("node:filler-{i}")),
                ..StoreMemory::atom(format!("unrelated filler note number {i} about widgets"))
            })
            .await
            .expect("store filler");
    }

    let marker = "kumquat-marker-nine";
    let big = format!(
        "{}\n\nthe key fact is {marker}\n\n{}",
        "alpha beta gamma. ".repeat(2_000),
        "delta epsilon zeta. ".repeat(2_000),
    );
    backend
        .store(StoreMemory {
            node_id: Some("node:late".to_string()),
            ..StoreMemory::atom(big)
        })
        .await
        .expect("store the late multi-chunk doc");

    let hits = backend
        .find(MemoryQuery::new("the key fact kumquat-marker-nine").with_limit(5))
        .await
        .expect("find should succeed");

    let marker_hit = hits
        .iter()
        .find(|hit| hit.record.content.contains(marker))
        .expect("the late document's marker must be retrieved");
    assert_eq!(marker_hit.record.node_id, "node:late");
    // Siblings were located despite the parent being stored after 60 other documents:
    // the window expanded past a single ~1600-char chunk.
    assert!(
        marker_hit.record.content.chars().count() > 1_600,
        "siblings must be found regardless of table position; got {} chars",
        marker_hit.record.content.chars().count()
    );
}
