// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "qdrant")]

use std::{collections::BTreeMap, env};

use chrono::Utc;
use mimisbrunnr::{
    MemoryBackend, MemoryQuery, MemoryTier, QdrantVectorStore, QdrantVectorStoreConfig, RrfOptions,
    StoreMemory, VectorMemoryBackend, VectorMemoryConfig, PINNED_FASTEMBED_DIMENSIONS,
    PINNED_FASTEMBED_MODEL,
};

#[test]
fn qdrant_vector_backend_pins_fastembed_model_and_dimensions() {
    let config = QdrantVectorStoreConfig::new("http://127.0.0.1:6333");

    assert_eq!(config.url, "http://127.0.0.1:6333");
    assert_eq!(PINNED_FASTEMBED_MODEL, "intfloat/multilingual-e5-small");
    assert_eq!(PINNED_FASTEMBED_DIMENSIONS, 384);
}

#[tokio::test]
#[ignore = "requires a local Qdrant instance and QDRANT_URL"]
async fn live_qdrant_vector_backend_satisfies_memory_contract() {
    let Ok(url) = env::var("QDRANT_URL") else {
        eprintln!("QDRANT_URL is not set; skipping live Qdrant test");
        return;
    };
    let mut config = QdrantVectorStoreConfig::new(url);
    config.api_key = env::var("QDRANT_API_KEY").ok();
    let store = QdrantVectorStore::connect(config).expect("Qdrant store should connect");
    let backend = VectorMemoryBackend::new(
        store,
        VectorMemoryConfig::new(format!("brunnr_test_{}", Utc::now().timestamp_millis())),
    )
    .expect("backend should construct");

    let stored = backend
        .store(StoreMemory {
            content: "Qdrant stores durable multilingual context".to_string(),
            tags: vec!["qdrant".to_string()],
            metadata: BTreeMap::new(),
            tier: MemoryTier::L1Atom,
            node_id: Some("node:qdrant".to_string()),
            created_at: None,
        })
        .await
        .expect("store should succeed");
    backend
        .store(StoreMemory {
            content: "hybrid vector keyword retrieval".to_string(),
            tags: Vec::new(),
            metadata: BTreeMap::new(),
            tier: MemoryTier::L1Atom,
            node_id: Some("node:qdrant-rrf".to_string()),
            created_at: None,
        })
        .await
        .expect("second store should succeed");

    let found = backend
        .find(MemoryQuery::new("durable context").with_limit(5))
        .await
        .expect("find should succeed");
    let drill_down = backend
        .get_node("node:qdrant")
        .await
        .expect("get_node should succeed");
    let hybrid = backend
        .hybrid_rrf(
            MemoryQuery::new("hybrid").with_limit(5),
            MemoryQuery::new("retrieval").with_limit(5),
            RrfOptions {
                limit: 5,
                ..RrfOptions::default()
            },
        )
        .await
        .expect("hybrid should succeed");

    assert!(
        found.iter().any(|hit| hit.record.node_id == "node:qdrant"),
        "find should return Qdrant record, got {found:?}"
    );
    assert_eq!(drill_down, Some(stored));
    assert!(
        hybrid
            .iter()
            .any(|hit| hit.record.node_id == "node:qdrant-rrf"),
        "hybrid should return Qdrant RRF record, got {hybrid:?}"
    );
}
