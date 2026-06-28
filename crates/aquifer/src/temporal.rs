// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use chrono::{DateTime, Utc};

use crate::{
    entity::{extract_entities, EntityIndex},
    event::Event,
    MemoryRecord, SearchHit,
};

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

/// Return the temporal profile of an entity: all records that mention it, ordered by
/// `created_at` ascending (oldest first).
///
/// A record "mentions" the entity when the entity string (case-insensitive) appears in the
/// record's tags or in the set of entities extracted from its content by [`extract_entities`].
///
/// The input slice may contain records from any backend; those without temporal data or without
/// matching entities are silently skipped (backward-compatible).
pub fn entity_timeline(records: &[MemoryRecord], entity: &str) -> Vec<MemoryRecord> {
    let entity_lower = entity.trim().to_lowercase();
    if entity_lower.is_empty() {
        return Vec::new();
    }

    let mut matching: Vec<MemoryRecord> = records
        .iter()
        .filter(|record| {
            // Fast path: tags (already stored, no extraction cost)
            if record.tags.iter().any(|t| t.to_lowercase() == entity_lower) {
                return true;
            }
            // Slow path: deterministic entity extraction from content
            extract_entities(&record.content)
                .iter()
                .any(|e| e.to_lowercase() == entity_lower)
        })
        .cloned()
        .collect();

    matching.sort_by_key(|r| r.created_at);
    matching
}

/// Re-order retrieval hits by event membership time.
///
/// Hits that belong to a known [`Event`] are placed in event-start-time order (oldest event
/// first), stable within each event group (original relative order preserved). Hits not covered
/// by any event are appended after the event-ordered block, retaining their original relative
/// order.
///
/// When `events` is empty this is a no-op (returns `hits` unchanged), so the default relevance
/// order of a normal `find` call is unaffected.
pub fn sort_hits_by_event_time(hits: Vec<SearchHit>, events: &[Event]) -> Vec<SearchHit> {
    if events.is_empty() {
        return hits;
    }

    // Build node_id → event start time (earliest created_at in event)
    let node_to_start: HashMap<&str, DateTime<Utc>> = events
        .iter()
        .flat_map(|event| {
            event
                .member_node_ids
                .iter()
                .map(move |nid| (nid.as_str(), event.time_range.0))
        })
        .collect();

    // Partition: event-covered hits carry their event start; others go to the tail
    let mut event_hits: Vec<(DateTime<Utc>, usize, SearchHit)> = Vec::new();
    let mut other_hits: Vec<(usize, SearchHit)> = Vec::new();

    for (index, hit) in hits.into_iter().enumerate() {
        match node_to_start.get(hit.record.node_id.as_str()) {
            Some(&start) => event_hits.push((start, index, hit)),
            None => other_hits.push((index, hit)),
        }
    }

    // Sort event-covered hits by event start time; stable original-index tiebreak
    event_hits.sort_by(|(start_a, idx_a, _), (start_b, idx_b, _)| {
        start_a.cmp(start_b).then(idx_a.cmp(idx_b))
    });

    let mut result: Vec<SearchHit> = event_hits.into_iter().map(|(_, _, hit)| hit).collect();
    result.extend(other_hits.into_iter().map(|(_, hit)| hit));
    result
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
                project: None,
                source: None,
                confidence: None,
                relations: Vec::new(),
                last_access: None,
                access_count: 0,
                state: crate::MemoryState::Active,
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

    // ── entity_timeline tests ─────────────────────────────────────────────────

    fn record_with_tag(node_id: &str, tag: &str, days_old: i64) -> MemoryRecord {
        MemoryRecord {
            id: MemoryId::new(format!("id:{node_id}")),
            node_id: node_id.to_string(),
            content: format!("content for {node_id}"),
            tags: vec![tag.to_string()],
            metadata: BTreeMap::new(),
            tier: MemoryTier::L1Atom,
            created_at: Utc::now() - Duration::days(days_old),
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
            state: crate::MemoryState::Active,
        }
    }

    #[test]
    fn entity_timeline_returns_records_ordered_by_time() {
        let records = vec![
            record_with_tag("node:newest", "RateLimit", 1),
            record_with_tag("node:oldest", "RateLimit", 30),
            record_with_tag("node:middle", "RateLimit", 15),
            record_with_tag("node:unrelated", "DatabaseMigration", 5),
        ];
        let timeline = entity_timeline(&records, "RateLimit");
        assert_eq!(
            timeline.len(),
            3,
            "only RateLimit records should be returned"
        );
        // Verify chronological order (oldest first)
        assert_eq!(timeline[0].node_id, "node:oldest");
        assert_eq!(timeline[1].node_id, "node:middle");
        assert_eq!(timeline[2].node_id, "node:newest");
    }

    #[test]
    fn entity_timeline_case_insensitive_tag_matching() {
        let records = vec![
            record_with_tag("node:a", "RateLimit", 10),
            record_with_tag("node:b", "ratelimit", 5), // different case
        ];
        let timeline = entity_timeline(&records, "RATELIMIT");
        assert_eq!(
            timeline.len(),
            2,
            "case-insensitive matching should find both"
        );
    }

    #[test]
    fn entity_timeline_empty_entity_returns_empty() {
        let records = vec![record_with_tag("node:a", "RateLimit", 1)];
        assert!(entity_timeline(&records, "").is_empty());
        assert!(entity_timeline(&records, "   ").is_empty());
    }

    #[test]
    fn entity_timeline_no_match_returns_empty() {
        let records = vec![record_with_tag("node:a", "SomeOtherThing", 1)];
        assert!(entity_timeline(&records, "RateLimit").is_empty());
    }

    #[test]
    fn entity_timeline_empty_records_returns_empty() {
        assert!(entity_timeline(&[], "RateLimit").is_empty());
    }

    #[test]
    fn entity_timeline_matches_content_entity() {
        // A record with a CamelCase entity in content (no tag)
        let record = MemoryRecord {
            id: MemoryId::new("id:content-match"),
            node_id: "node:content-match".to_string(),
            content: "BackgroundJobRetryPolicy exceeded max retries.".to_string(),
            tags: Vec::new(),
            metadata: BTreeMap::new(),
            tier: MemoryTier::L1Atom,
            created_at: Utc::now() - Duration::days(5),
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
            state: crate::MemoryState::Active,
        };
        let timeline = entity_timeline(&[record], "BackgroundJobRetryPolicy");
        assert_eq!(
            timeline.len(),
            1,
            "entity extracted from content should match"
        );
    }

    // ── sort_hits_by_event_time tests ─────────────────────────────────────────

    #[test]
    fn sort_hits_by_event_time_orders_by_event_start() {
        use crate::event::Event;

        // Two hits from different events; the older event should come first
        let old_hit = hit("node:old-event", 0.9, 30); // 30 days old
        let new_hit = hit("node:new-event", 1.0, 5); // 5 days old (higher score)

        let events = vec![
            Event {
                id: "ev-old".to_string(),
                title: "OldEvent".to_string(),
                time_range: (
                    Utc::now() - chrono::Duration::days(30),
                    Utc::now() - chrono::Duration::days(29),
                ),
                entities: vec!["OldEvent".to_string()],
                member_node_ids: vec!["node:old-event".to_string()],
            },
            Event {
                id: "ev-new".to_string(),
                title: "NewEvent".to_string(),
                time_range: (
                    Utc::now() - chrono::Duration::days(5),
                    Utc::now() - chrono::Duration::days(4),
                ),
                entities: vec!["NewEvent".to_string()],
                member_node_ids: vec!["node:new-event".to_string()],
            },
        ];

        // Input: new_hit first (higher relevance score), old_hit second
        let hits = vec![new_hit, old_hit];
        let sorted = sort_hits_by_event_time(hits, &events);

        assert_eq!(
            sorted[0].record.node_id, "node:old-event",
            "oldest event should come first after temporal sort"
        );
        assert_eq!(sorted[1].record.node_id, "node:new-event");
    }

    #[test]
    fn sort_hits_by_event_time_empty_events_is_noop() {
        let hits = vec![hit("node:a", 1.0, 5), hit("node:b", 0.5, 1)];
        let original_order: Vec<String> = hits.iter().map(|h| h.record.node_id.clone()).collect();
        let sorted = sort_hits_by_event_time(hits, &[]);
        let sorted_order: Vec<String> = sorted.iter().map(|h| h.record.node_id.clone()).collect();
        assert_eq!(original_order, sorted_order, "empty events → no-op");
    }

    #[test]
    fn sort_hits_by_event_time_uncovered_hits_appended_after() {
        use crate::event::Event;

        let event_hit = hit("node:in-event", 0.5, 20);
        let orphan_hit = hit("node:orphan", 1.0, 1); // higher score, no event

        let events = vec![Event {
            id: "ev1".to_string(),
            title: "SomeEvent".to_string(),
            time_range: (
                Utc::now() - chrono::Duration::days(20),
                Utc::now() - chrono::Duration::days(19),
            ),
            entities: vec!["SomeEvent".to_string()],
            member_node_ids: vec!["node:in-event".to_string()],
        }];

        let hits = vec![orphan_hit, event_hit];
        let sorted = sort_hits_by_event_time(hits, &events);

        assert_eq!(
            sorted[0].record.node_id, "node:in-event",
            "event-covered hit should precede uncovered hits"
        );
        assert_eq!(
            sorted[1].record.node_id, "node:orphan",
            "uncovered hit should be appended last"
        );
    }
}
