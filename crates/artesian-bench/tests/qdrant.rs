// SPDX-License-Identifier: Apache-2.0

use std::{collections::BTreeMap, env};

use aquifer::{
    MemoryBackend, MemoryQuery, MemoryTier, QdrantVectorStore, QdrantVectorStoreConfig,
    StoreMemory, VectorMemoryBackend, VectorMemoryConfig,
};
use chrono::Utc;

#[tokio::test]
#[ignore = "requires a live Qdrant instance and QDRANT_URL"]
async fn live_qdrant_benchmark_path_uses_real_vector_backend() {
    let Ok(url) = env::var("QDRANT_URL") else {
        eprintln!("QDRANT_URL is not set; skipping live Qdrant benchmark smoke");
        return;
    };
    let mut config = QdrantVectorStoreConfig::new(url);
    config.rest_url = env::var("QDRANT_REST_URL").ok();
    config.api_key = env::var("QDRANT_API_KEY").ok();
    let collection = format!("artesian_bench_smoke_{}", Utc::now().timestamp_millis());
    let backend = VectorMemoryBackend::new(
        QdrantVectorStore::connect(config).expect("Qdrant store should connect"),
        VectorMemoryConfig::new(collection.clone()),
    )
    .expect("Qdrant memory backend should construct");

    let test_result = async {
        backend
            .store(StoreMemory {
                content: "The benchmark smoke answer lives in a real Qdrant vector collection."
                    .to_string(),
                tags: vec!["benchmark".to_string()],
                metadata: BTreeMap::new(),
                tier: MemoryTier::L1Atom,
                node_id: Some("node:qdrant-bench-smoke".to_string()),
                created_at: None,
                scope: None,
                agent_id: None,
                session_id: None,
                task_id: None,
                user_id: None,
                source: None,
                confidence: None,
            })
            .await
            .map_err(|error| format!("store should succeed: {error}"))?;

        let hits = backend
            .find(MemoryQuery::new("benchmark smoke answer").with_limit(3))
            .await
            .map_err(|error| format!("find should succeed: {error}"))?;
        if hits
            .iter()
            .any(|hit| hit.record.node_id == "node:qdrant-bench-smoke")
        {
            Ok(())
        } else {
            Err(format!(
                "live Qdrant arm should retrieve the stored smoke record, got {hits:?}"
            ))
        }
    }
    .await;

    backend
        .vector_store()
        .client()
        .delete_collection(collection.clone())
        .await
        .expect("smoke cleanup should delete its temporary collection");
    let exists_after_cleanup = backend
        .vector_store()
        .client()
        .collection_exists(collection.clone())
        .await
        .expect("smoke cleanup should be verifiable");
    assert!(
        !exists_after_cleanup,
        "smoke cleanup left temporary collection {collection}"
    );

    test_result.expect("live Qdrant smoke should pass");
}
