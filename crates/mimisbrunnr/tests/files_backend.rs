// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use brunnr_test_support::TempDir;
use mimisbrunnr::{FilesBackend, MemoryBackend, MemoryQuery, MemoryTier, SearchHit, StoreMemory};
use tokio::fs;

#[tokio::test]
async fn files_backend_stores_date_tagged_markdown_and_finds_it() {
    let tempdir = TempDir::new("files-store");
    let backend = FilesBackend::new(tempdir.path());

    let stored = backend
        .store(StoreMemory {
            content: "Files backend keeps memory readable".to_string(),
            tags: vec!["files".to_string()],
            metadata: BTreeMap::new(),
            tier: MemoryTier::L1Atom,
            node_id: Some("node:files".to_string()),
            created_at: None,
        })
        .await
        .expect("store should succeed");

    let date_tag = stored.created_at.format("%Y-%m-%d").to_string();
    let memory_dir = tempdir.join(["memory", &date_tag].iter().collect::<std::path::PathBuf>());
    let path = std::fs::read_dir(memory_dir)
        .expect("memory date dir should exist")
        .map(|entry| entry.expect("record entry should be readable").path())
        .find(|path| {
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .is_some_and(|stem| stem == stored.id.as_str())
        })
        .expect("record file should exist");
    let rendered = fs::read_to_string(path)
        .await
        .expect("record should be readable");
    let hits = backend
        .find(MemoryQuery::new("readable"))
        .await
        .expect("find should succeed");

    assert!(rendered.contains(&format!("[{date_tag}] Files backend keeps memory readable")));
    assert_eq!(hits, vec![SearchHit::keyword(stored, 1.0)]);
}

#[tokio::test]
async fn files_backend_drills_down_by_node_id() {
    let tempdir = TempDir::new("files-node");
    let backend = FilesBackend::new(tempdir.path());
    let stored = backend
        .store(StoreMemory {
            content: "Ground truth evidence".to_string(),
            tags: Vec::new(),
            metadata: BTreeMap::new(),
            tier: MemoryTier::L0Raw,
            node_id: Some("node:evidence".to_string()),
            created_at: None,
        })
        .await
        .expect("store should succeed");

    assert_eq!(
        backend
            .get_node("node:evidence")
            .await
            .expect("get_node should succeed"),
        Some(stored)
    );
}
