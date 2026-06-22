// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use aquifer::{FilesBackend, Session, SessionKey, SessionListFilter, SessionStore};
use artesian_test_support::TempDir;
use serde_json::json;

fn session_record(key: SessionKey, handed_off_from: &str, content: &str) -> Session {
    Session::new(
        key.clone(),
        json!({
            "ocf_version": "0.1",
            "agent_id": handed_off_from,
            "session": {
                "user_id": key.user_id,
                "session_id": key.session_id,
                "task_id": key.task_id,
                "handed_off_from": handed_off_from,
            },
            "created": "2026-06-22T00:00:00Z",
            "unit_source": "inline",
            "unit_refs": [],
        }),
        json!({
            "ocf_version": "0.1",
            "slots": [{"name": "task-state"}],
            "budget_tokens": 4096,
            "eviction": "lowest-score",
        }),
        json!({
            "budget_tokens": 4096,
            "token_count": 1,
            "saturation": 0.0,
            "entries": [{
                "id": "anchor-task",
                "slot": "task-state",
                "content": content,
                "tokens": 1,
                "score": 1.0,
                "resolution": "full",
                "committed_at": "2026-06-22T00:00:00Z",
            }],
        }),
        Vec::new(),
    )
}

#[tokio::test]
async fn sessions_are_loaded_by_exact_user_session_task_tuple() {
    let tempdir = TempDir::new("session-store");
    let store = SessionStore::new(Arc::new(FilesBackend::new(tempdir.path())));
    let key_a = SessionKey::new(
        Some("user-a".to_string()),
        Some("session-1".to_string()),
        Some("task-1".to_string()),
    );
    let key_b = SessionKey::new(
        Some("user-b".to_string()),
        Some("session-1".to_string()),
        Some("task-1".to_string()),
    );

    store
        .store(session_record(key_a.clone(), "codex", "state for user a"))
        .await
        .expect("session A should store");
    store
        .store(session_record(key_b.clone(), "claude", "state for user b"))
        .await
        .expect("session B should store");

    let loaded_a = store
        .load(&key_a)
        .await
        .expect("session A should load")
        .expect("session A should exist");
    let loaded_b = store
        .load(&key_b)
        .await
        .expect("session B should load")
        .expect("session B should exist");

    assert_eq!(
        loaded_a.snapshot["entries"][0]["content"],
        "state for user a"
    );
    assert_eq!(
        loaded_b.snapshot["entries"][0]["content"],
        "state for user b"
    );

    let summaries = store
        .list(SessionListFilter {
            user_id: Some("user-a".to_string()),
            ..SessionListFilter::default()
        })
        .await
        .expect("sessions should list");
    assert_eq!(summaries.len(), 1);
    assert_eq!(summaries[0].key, key_a);
}

#[tokio::test]
async fn default_session_key_round_trips_when_identity_is_unset() {
    let tempdir = TempDir::new("session-default");
    let store = SessionStore::new(Arc::new(FilesBackend::new(tempdir.path())));
    let key = SessionKey::new(None, None, None);

    store
        .store(session_record(
            key.clone(),
            "codex",
            "default session state",
        ))
        .await
        .expect("default session should store");

    let loaded = store
        .load(&key)
        .await
        .expect("default session should load")
        .expect("default session should exist");
    assert_eq!(loaded.key, key);
    assert_eq!(
        loaded.snapshot["entries"][0]["content"],
        "default session state"
    );
}
