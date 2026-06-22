// SPDX-License-Identifier: Apache-2.0

use aquifer::{
    recover_after_compaction, AnchorAnchorStore, FilesBackend, MemoryBackend, MemoryTier,
    SessionAnchor, SessionKey, StoreMemory,
};
use artesian_test_support::TempDir;

#[tokio::test]
async fn anchor_anchor_round_trips_through_log() {
    let tempdir = TempDir::new("anchor-roundtrip");
    let store = AnchorAnchorStore::new(tempdir.path());
    let mut anchor = SessionAnchor::new("implement task store", "write contract tests");
    anchor.plan_pointer = Some("docs/task-tracking.md#taskstore".to_string());
    anchor.last_decisions = vec!["single mutation authority".to_string()];

    let written = store.set(anchor).await.expect("anchor set should succeed");
    let loaded = store
        .get()
        .await
        .expect("anchor get should succeed")
        .expect("anchor should exist");

    assert_eq!(loaded.current_task, "implement task store");
    assert_eq!(
        loaded.plan_pointer,
        Some("docs/task-tracking.md#taskstore".to_string())
    );
    assert_eq!(loaded.last_decisions, vec!["single mutation authority"]);
    assert_eq!(loaded.next_step, "write contract tests");
    assert_eq!(loaded.updated_at, written.updated_at);
}

#[tokio::test]
async fn keyed_anchor_does_not_replace_default_anchor() {
    let tempdir = TempDir::new("anchor-keyed");
    let store = AnchorAnchorStore::new(tempdir.path());
    let key = SessionKey::new(
        Some("user-a".to_string()),
        Some("session-a".to_string()),
        Some("task-a".to_string()),
    );

    store
        .set(SessionAnchor::new("default task", "default next"))
        .await
        .expect("default anchor should write");
    store
        .set_for_session(&key, SessionAnchor::new("keyed task", "keyed next"))
        .await
        .expect("keyed anchor should write");

    let default_anchor = store
        .get()
        .await
        .expect("default anchor should read")
        .expect("default anchor should exist");
    assert_eq!(default_anchor.current_task, "default task");

    let keyed_anchor = store
        .get_for_session(&key)
        .await
        .expect("keyed anchor should read")
        .expect("keyed anchor should exist");
    assert_eq!(keyed_anchor.current_task, "keyed task");

    let other_key = SessionKey::new(
        Some("user-a".to_string()),
        Some("session-a".to_string()),
        Some("other-task".to_string()),
    );
    assert!(store
        .get_for_session(&other_key)
        .await
        .expect("other key should read")
        .is_none());
}

#[tokio::test]
async fn simulated_compaction_replays_anchor_and_targeted_memory() {
    let tempdir = TempDir::new("anchor-recovery");
    let backend = FilesBackend::new(tempdir.path());
    backend
        .store(StoreMemory {
            content: "write contract tests for simulated compaction recovery".to_string(),
            tags: Vec::new(),
            metadata: Default::default(),
            tier: MemoryTier::L1Atom,
            node_id: Some("node:recovery".to_string()),
            created_at: None,
            scope: None,
            agent_id: None,
            session_id: None,
            task_id: None,
            user_id: None,
            source: None,
            confidence: None,
            relations: Vec::new(),
        })
        .await
        .expect("memory store should succeed");

    let anchors = AnchorAnchorStore::new(tempdir.path());
    anchors
        .set(SessionAnchor::new(
            "simulated compaction recovery",
            "write contract tests",
        ))
        .await
        .expect("anchor set should succeed");

    let recovered = recover_after_compaction(&anchors, &backend, 5)
        .await
        .expect("recovery should succeed")
        .expect("anchor should exist");

    assert_eq!(
        recovered.anchor.current_task,
        "simulated compaction recovery"
    );
    assert!(
        recovered
            .hits
            .iter()
            .any(|hit| hit.record.node_id == "node:recovery"),
        "targeted memory.find should replay recovery context, got {:?}",
        recovered.hits
    );
}
