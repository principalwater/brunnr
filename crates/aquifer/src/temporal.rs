// SPDX-License-Identifier: Apache-2.0

use chrono::Utc;

use crate::{entity::EntityIndex, SearchHit};

/// Apply exponential recency decay to retrieval scores.
///
/// Each hit's score is multiplied by `exp(−lambda × age_in_days)`. `lambda = 0.0` is a no-op.
/// Higher lambda means stronger preference for recent records.
pub fn apply_recency_decay(mut hits: Vec<SearchHit>, lambda: f32) -> Vec<SearchHit> {
    if lambda == 0.0 {
        return hits;
    }
    let now = Utc::now();
    for hit in &mut hits {
        let age_days = (now - hit.record.created_at).num_seconds().max(0) as f32 / 86_400.0;
        hit.score *= (-lambda * age_days).exp();
    }
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.record.node_id.cmp(&b.record.node_id))
    });
    hits
}

/// Downrank older records that share entities with a newer record in the same hit set.
///
/// For each pair of hits sharing at least one entity, the older record's score is multiplied by
/// `supersession_penalty`. The newer record retains its score. This implements the
/// "knowledge-update" signal: a newer record about the same topic supersedes an older one for
/// retrieval purposes. The older record remains in the store for drill-down via `node_id`.
pub fn apply_knowledge_supersession(
    mut hits: Vec<SearchHit>,
    entity_index: &EntityIndex,
    supersession_penalty: f32,
) -> Vec<SearchHit> {
    let hit_entities: Vec<Vec<String>> = hits
        .iter()
        .map(|hit| {
            entity_index
                .record_entities(&hit.record.node_id)
                .unwrap_or(&[])
                .to_vec()
        })
        .collect();

    let mut penalty_mask = vec![false; hits.len()];
    for i in 0..hits.len() {
        for j in (i + 1)..hits.len() {
            let shares_entity = hit_entities[i].iter().any(|e| hit_entities[j].contains(e));
            if shares_entity {
                let created_i = hits[i].record.created_at;
                let created_j = hits[j].record.created_at;
                if created_i < created_j {
                    penalty_mask[i] = true;
                } else if created_j < created_i {
                    penalty_mask[j] = true;
                }
            }
        }
    }

    for (i, penalized) in penalty_mask.into_iter().enumerate() {
        if penalized {
            hits[i].score *= supersession_penalty;
        }
    }

    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.record.node_id.cmp(&b.record.node_id))
    });
    hits
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use chrono::{Duration, Utc};

    use crate::{MemoryId, MemoryRecord, MemoryTier, SearchHit, SearchSource};

    use super::*;

    fn hit(node_id: &str, score: f32, days_old: i64) -> SearchHit {
        SearchHit {
            record: MemoryRecord {
                id: MemoryId::new(format!("id:{node_id}")),
                node_id: node_id.to_string(),
                content: format!("content for {node_id}"),
                tags: Vec::new(),
                metadata: BTreeMap::new(),
                tier: MemoryTier::L1Atom,
                created_at: Utc::now() - Duration::days(days_old),
                scope: None,
                agent_id: None,
                session_id: None,
                task_id: None,
                user_id: None,
                source: None,
                confidence: None,
            },
            score,
            source: SearchSource::Hybrid,
        }
    }

    #[test]
    fn recency_decay_penalises_older_records() {
        let hits = vec![
            hit("node:old", 1.0, 365), // 1 year old
            hit("node:new", 1.0, 1),   // 1 day old
        ];
        let decayed = apply_recency_decay(hits, 0.01);
        assert!(
            decayed[0].record.node_id == "node:new",
            "newer record should rank first after decay"
        );
        assert!(
            decayed[0].score > decayed[1].score,
            "newer record should have higher score after decay"
        );
    }

    #[test]
    fn recency_decay_zero_lambda_is_noop() {
        let hits = vec![hit("node:a", 0.5, 100), hit("node:b", 0.8, 1)];
        let original_top = hits[0].record.node_id.clone();
        let decayed = apply_recency_decay(hits, 0.0);
        assert_eq!(decayed[0].record.node_id, original_top);
    }

    #[test]
    fn supersession_penalises_older_entity_overlap() {
        use crate::entity::EntityIndex;

        let old_hit = hit("node:old", 1.0, 200);
        let new_hit = hit("node:new", 0.8, 10);

        let mut index = EntityIndex::new();
        index.index_record({
            let mut r = old_hit.record.clone();
            r.tags = vec!["RateLimit".to_string()];
            r
        });
        index.index_record({
            let mut r = new_hit.record.clone();
            r.tags = vec!["RateLimit".to_string()];
            r
        });

        let hits = vec![old_hit, new_hit];
        let result = apply_knowledge_supersession(hits, &index, 0.3);

        let new_score = result
            .iter()
            .find(|h| h.record.node_id == "node:new")
            .map(|h| h.score)
            .unwrap();
        let old_score = result
            .iter()
            .find(|h| h.record.node_id == "node:old")
            .map(|h| h.score)
            .unwrap();
        assert!(
            new_score > old_score,
            "newer record should outscore older after supersession: new={new_score} old={old_score}"
        );
    }
}
