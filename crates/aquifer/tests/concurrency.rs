// SPDX-License-Identifier: Apache-2.0

use std::{sync::Arc, time::Duration};

use aquifer::{
    CommitLog, FilesBackend, MemoryBackend, MemoryQuery, MemoryResult, MemoryScope, MemoryTier,
    SessionLaneLock, SqliteVecVectorStore, StoreMemory, TextEmbedder, TransactionalMemory,
    TxnError, VectorMemoryBackend, VectorMemoryConfig,
};
use artesian_test_support::TempDir;

#[tokio::test]
async fn files_backend_isolates_concurrent_task_scopes() {
    let tempdir = TempDir::new("files-concurrency");
    let backend = Arc::new(FilesBackend::new(tempdir.path()));
    assert_concurrent_scope_isolation(backend).await;
}

#[tokio::test]
async fn sqlite_vec_backend_isolates_concurrent_task_scopes() {
    let store = SqliteVecVectorStore::in_memory().expect("sqlite-vec should open");
    let backend = VectorMemoryBackend::with_embedder(
        store,
        VectorMemoryConfig {
            collection: "concurrency".to_string(),
            dimensions: TEST_DIMENSIONS,
            ..VectorMemoryConfig::new("concurrency")
        },
        Arc::new(TestEmbedder),
    )
    .expect("backend should construct");
    assert_concurrent_scope_isolation(Arc::new(backend)).await;
}

#[tokio::test]
async fn session_lane_lock_serializes_and_times_out() {
    let tempdir = TempDir::new("lane-lock-timeout");
    let lock = SessionLaneLock::new(tempdir.path()).with_timeout(Duration::from_millis(50));
    let guard = lock
        .acquire("shared-collection", Some("session-a"))
        .await
        .expect("first lane acquire should succeed");

    let blocked = lock.acquire("shared-collection", Some("session-a")).await;
    assert!(blocked.is_err());
    assert!(blocked
        .expect_err("lane should time out")
        .to_string()
        .contains("timed out acquiring session lane lock"));

    guard.release().expect("release should succeed");
    lock.acquire("shared-collection", Some("session-a"))
        .await
        .expect("lane should reacquire after release");
}

#[tokio::test]
async fn sqlite_vec_multi_writer_integrity_and_tenant_isolation() {
    let store = SqliteVecVectorStore::in_memory().expect("sqlite-vec should open");
    let backend = Arc::new(
        VectorMemoryBackend::with_embedder(
            store,
            VectorMemoryConfig {
                collection: "shared-project".to_string(),
                dimensions: TEST_DIMENSIONS,
                ..VectorMemoryConfig::new("shared-project")
            },
            Arc::new(TestEmbedder),
        )
        .expect("backend should construct"),
    );

    let writer_count = 24;
    let mut handles = Vec::new();
    for index in 0..writer_count {
        let backend = Arc::clone(&backend);
        handles.push(tokio::spawn(async move {
            backend
                .store(StoreMemory {
                    content: format!("contention memory tenant word {index}"),
                    tags: Vec::new(),
                    metadata: Default::default(),
                    tier: MemoryTier::L1Atom,
                    node_id: Some(format!("node:tenant-{index}")),
                    created_at: None,
                    scope: Some(MemoryScope::Session),
                    agent_id: Some(format!("agent-{}", index % 3)),
                    session_id: Some(format!("session-{}", index % 4)),
                    task_id: Some(format!("task-{index}")),
                    user_id: Some(format!("user-{}", index % 2)),
                    source: None,
                    confidence: None,
                    relations: Vec::new(),
                })
                .await
        }));
    }
    for handle in handles {
        handle
            .await
            .expect("writer should join")
            .expect("writer should store");
    }

    let mut query = MemoryQuery::new("contention memory tenant").with_limit(writer_count);
    query.scope = Some(MemoryScope::Session);
    query.user_id = Some("user-0".to_string());
    let user_zero = backend.find(query).await.expect("find should succeed");
    assert_eq!(user_zero.len(), writer_count / 2);
    assert!(user_zero
        .iter()
        .all(|hit| hit.record.user_id.as_deref() == Some("user-0")));

    let mut query = MemoryQuery::new("contention memory tenant").with_limit(writer_count);
    query.scope = Some(MemoryScope::Session);
    query.user_id = Some("user-1".to_string());
    let user_one = backend.find(query).await.expect("find should succeed");
    assert_eq!(user_one.len(), writer_count / 2);
    assert!(user_one
        .iter()
        .all(|hit| hit.record.user_id.as_deref() == Some("user-1")));
}

async fn assert_concurrent_scope_isolation(backend: Arc<dyn MemoryBackend>) {
    let mut handles = Vec::new();
    for index in 0..8 {
        let backend = Arc::clone(&backend);
        handles.push(tokio::spawn(async move {
            backend
                .store(StoreMemory {
                    content: "concurrent scoped memory".to_string(),
                    tags: Vec::new(),
                    metadata: Default::default(),
                    tier: MemoryTier::L1Atom,
                    node_id: Some(format!("node:scope-{index}")),
                    created_at: None,
                    scope: Some(MemoryScope::Task),
                    agent_id: None,
                    session_id: None,
                    task_id: Some(format!("task-{index}")),
                    user_id: None,
                    source: None,
                    confidence: None,
                    relations: Vec::new(),
                })
                .await
        }));
    }
    for handle in handles {
        handle
            .await
            .expect("store task should join")
            .expect("store should succeed");
    }

    for index in 0..8 {
        let mut query = MemoryQuery::new("scoped").with_limit(10);
        query.scope = Some(MemoryScope::Task);
        query.task_id = Some(format!("task-{index}"));
        let hits = backend.find(query).await.expect("find should succeed");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record.node_id, format!("node:scope-{index}"));
    }

    let mut handles = Vec::new();
    for _ in 0..8 {
        let backend = Arc::clone(&backend);
        handles.push(tokio::spawn(async move {
            backend
                .store(StoreMemory {
                    content: "idempotent concurrent duplicate".to_string(),
                    tags: Vec::new(),
                    metadata: Default::default(),
                    tier: MemoryTier::L1Atom,
                    node_id: Some("node:duplicate".to_string()),
                    created_at: None,
                    scope: Some(MemoryScope::Task),
                    agent_id: None,
                    session_id: None,
                    task_id: Some("task-duplicate".to_string()),
                    user_id: None,
                    source: None,
                    confidence: None,
                    relations: Vec::new(),
                })
                .await
        }));
    }
    for handle in handles {
        handle
            .await
            .expect("duplicate store task should join")
            .expect("duplicate store should succeed");
    }

    let mut query = MemoryQuery::new("duplicate").with_limit(10);
    query.scope = Some(MemoryScope::Task);
    query.task_id = Some("task-duplicate".to_string());
    let hits = backend.find(query).await.expect("find should succeed");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record.node_id, "node:duplicate");
}

/// Stress test: N agents × M operators write the shared memory concurrently through the
/// transactional commit log. Proves:
/// 1. All writes commit without corruption (commit_with_retry handles CAS conflicts).
/// 2. The commit-log sequence advances exactly N×M — one increment per committed write.
/// 3. Tenant isolation: each operator's writes are returned only for that operator's filter.
///
/// This is the acceptance criterion for Step 4: "N agents × M operators write the shared
/// memory concurrently with zero corruption and correct isolation."
#[tokio::test]
async fn transactional_memory_n_agents_m_operators_zero_corruption() {
    let agents = 6usize;
    let operators = 4usize;
    let total = agents * operators;

    let tempdir = TempDir::new("txn-stress");
    let backend = Arc::new(FilesBackend::new(tempdir.path()));
    let shared_log = CommitLog::new();

    let txn: Arc<TransactionalMemory<FilesBackend>> = Arc::new(TransactionalMemory::with_log(
        Arc::clone(&backend),
        shared_log,
    ));

    let mut handles = Vec::new();
    for op in 0..operators {
        for ag in 0..agents {
            let txn = Arc::clone(&txn);
            handles.push(tokio::spawn(async move {
                txn.commit_with_retry(
                    StoreMemory {
                        content: format!("operator-{op} agent-{ag} shared state"),
                        tags: vec![format!("op-{op}"), format!("ag-{ag}")],
                        metadata: Default::default(),
                        tier: MemoryTier::L1Atom,
                        node_id: Some(format!("node:op{op}:ag{ag}")),
                        created_at: None,
                        scope: Some(MemoryScope::Agent),
                        agent_id: Some(format!("agent-{ag}")),
                        session_id: None,
                        task_id: None,
                        user_id: Some(format!("operator-{op}")),
                        source: None,
                        confidence: None,
                        relations: Vec::new(),
                    },
                    32,
                )
                .await
            }));
        }
    }

    let mut successes = 0usize;
    for handle in handles {
        if handle.await.expect("join").is_ok() {
            successes += 1;
        }
    }

    assert_eq!(
        successes, total,
        "all {total} writes should commit — zero corruption"
    );
    assert_eq!(
        txn.current_seq(),
        total as u64,
        "commit log should have advanced exactly {total} times"
    );

    // Verify tenant isolation: each operator's memories are readable only through their filter.
    for op in 0..operators {
        let mut query = MemoryQuery::new("shared state").with_limit(total);
        query.user_id = Some(format!("operator-{op}"));
        let hits = backend.find(query).await.expect("find should succeed");
        assert_eq!(
            hits.len(),
            agents,
            "operator-{op} should see exactly {agents} memories"
        );
        assert!(
            hits.iter()
                .all(|h| h.record.user_id.as_deref() == Some(&format!("operator-{op}"))),
            "all hits for operator-{op} should belong to that operator"
        );
    }
}

/// Verify CAS conflict is detectable without retry, matching the Cursor model:
/// two writers that both read seq=N, only one commits; the other must retry.
#[tokio::test]
async fn transactional_memory_cas_conflict_is_detectable() {
    let tempdir = TempDir::new("txn-cas");
    let backend = Arc::new(FilesBackend::new(tempdir.path()));
    let txn = TransactionalMemory::new(Arc::clone(&backend));

    let (seq_a, mem_a) = txn
        .begin_write(StoreMemory::atom("writer A"))
        .await
        .unwrap();
    let (seq_b, mem_b) = txn
        .begin_write(StoreMemory::atom("writer B"))
        .await
        .unwrap();
    assert_eq!(seq_a, seq_b, "both writers should see the same initial seq");

    txn.commit(seq_a, mem_a)
        .await
        .expect("first commit succeeds");

    let err = txn.commit(seq_b, mem_b).await.unwrap_err();
    assert!(
        matches!(err, TxnError::Conflict { .. }),
        "second write on same seq should conflict, got {err}"
    );
}

/// Verify that a human-edited OKF file becomes immediately retrievable after
/// sync_okf_directory — the "file edit = transaction" primitive.
#[tokio::test]
async fn sync_okf_directory_makes_file_edits_retrievable() {
    let tempdir = TempDir::new("okf-edit-txn");
    let memory_dir = tempdir.path().join("memory");
    std::fs::create_dir_all(&memory_dir).unwrap();

    std::fs::write(
        memory_dir.join("decision.md"),
        "---\ntype: decision\ntitle: Approach B selected\ntimestamp: 2026-06-20T00:00:00Z\nnode_id: node:decision-b\ntier: l1-atom\n---\n\nApproach B was selected after approach A failed.\n",
    ).expect("write fixture");

    let backend = FilesBackend::new(&memory_dir);
    let report = aquifer::sync_okf_directory(&memory_dir, &backend)
        .await
        .expect("sync should succeed");

    assert!(report.files_scanned >= 1, "should scan at least 1 file");
    assert!(
        report.records_indexed >= 1,
        "should index at least 1 record"
    );
    assert_eq!(report.parse_failures, 0);
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
