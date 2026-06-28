// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use chrono::Utc;

use aquifer::{
    reciprocal_rank_fusion, MemoryId, MemoryRecord, MemoryState, MemoryTier, RrfOptions, SearchHit,
};

fn record(id: &str, node_id: &str) -> MemoryRecord {
    MemoryRecord {
        id: MemoryId::new(id),
        node_id: node_id.to_string(),
        content: id.to_string(),
        tags: Vec::new(),
        metadata: BTreeMap::new(),
        tier: MemoryTier::L1Atom,
        created_at: Utc::now(),
        scope: None,
        agent_id: None,
        session_id: None,
        task_id: None,
        user_id: None,
        project: None,
        source: None,
        confidence: None,
        relations: Vec::new(),
        last_access: None,
        access_count: 0,
        state: MemoryState::Active,
    }
}

#[test]
fn rrf_merges_duplicate_node_ids_across_channels() {
    let first = SearchHit::keyword(record("a", "node:a"), 10.0);
    let duplicate = SearchHit::keyword(record("a-copy", "node:a"), 4.0);
    let second = SearchHit::keyword(record("b", "node:b"), 3.0);

    let hits = reciprocal_rank_fusion(
        &[vec![first], vec![second, duplicate]],
        RrfOptions {
            rank_constant: 60.0,
            limit: 10,
        },
    );

    assert_eq!(
        hits.iter()
            .map(|hit| hit.record.node_id.as_str())
            .collect::<Vec<_>>(),
        vec!["node:a", "node:b"]
    );
    assert!(hits[0].score > hits[1].score);
}
