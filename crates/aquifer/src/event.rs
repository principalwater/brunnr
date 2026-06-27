// SPDX-License-Identifier: Apache-2.0

//! Temporal event envelopes — grouping atomic memories into coherent happenings.
//!
//! An [`Event`] is a derived, lightweight view over a set of [`MemoryRecord`]s that share at
//! least one entity and were created within a configurable time window of each other.
//!
//! Events are computed deterministically from stored records (no LLM required). They augment
//! the atomic-fact layer with a mid-level structure: a "happening" that clusters related facts
//! about the same thing(s) across a time span.

use std::{collections::HashMap, time::Duration};

use chrono::{DateTime, Utc};

use crate::{entity::extract_entities, MemoryRecord};

/// A coherent temporal group of related atomic memory records.
///
/// Events are derived — not stored separately — and are assembled on demand from a slice of
/// [`MemoryRecord`]s by [`assemble_events`]. An event represents a "happening": a cluster of
/// facts that share at least one entity and were recorded within a configurable time window of
/// each other.
///
/// # Backward compatibility
///
/// Events are a derived view. Records that have no tags and no extractable entities never become
/// event members; the store and all existing records remain unmodified.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
    /// Stable deterministic identifier derived from sorted member `node_id`s.
    ///
    /// Computed via FNV-1a over the concatenated, newline-separated node ids.
    pub id: String,
    /// Short human-readable title: the most-frequently occurring entity across member records.
    pub title: String,
    /// Temporal span: `(earliest created_at, latest created_at)` across member records.
    pub time_range: (DateTime<Utc>, DateTime<Utc>),
    /// Union of all entities extracted from member records (sorted, deduplicated).
    pub entities: Vec<String>,
    /// `node_id`s of the member records (sorted lexicographically).
    pub member_node_ids: Vec<String>,
}

/// Assemble temporal event envelopes from a slice of [`MemoryRecord`]s.
///
/// Two records are placed in the same event when they share at least one entity (extracted from
/// tags and content via [`extract_entities`]) *and* the time between consecutive same-entity
/// records does not exceed `window`. Transitivity applies via union-find: if A→B and B→C are
/// each within `window` and share an entity, then A, B, C form one event even if A and C are
/// more than `window` apart.
///
/// Events with only a single member record are omitted (an event requires at least two records).
/// The returned events are ordered by their start time (earliest `created_at` across members).
///
/// This function is deterministic and requires no LLM.
///
/// # Example
///
/// ```
/// use aquifer::event::{assemble_events, Event};
/// use aquifer::{MemoryId, MemoryRecord, MemoryState, MemoryTier};
/// use std::{collections::BTreeMap, time::Duration};
/// use chrono::{Duration as ChronoDuration, Utc};
///
/// let make = |node_id: &str, tags: Vec<&str>, days_old: i64| MemoryRecord {
///     id: MemoryId::new(format!("id:{node_id}")),
///     node_id: node_id.to_string(),
///     content: format!("{node_id} event"),
///     tags: tags.into_iter().map(str::to_string).collect(),
///     metadata: BTreeMap::new(),
///     tier: MemoryTier::L1Atom,
///     created_at: Utc::now() - ChronoDuration::days(days_old),
///     scope: None, agent_id: None, session_id: None, task_id: None,
///     user_id: None, source: None, confidence: None, relations: Vec::new(),
///     last_access: None, access_count: 0, state: MemoryState::Active,
/// };
/// let records = vec![
///     make("a", vec!["RateLimit"], 3),
///     make("b", vec!["RateLimit"], 2),
///     make("c", vec!["DatabaseMigration"], 100),
/// ];
/// let events = assemble_events(&records, Duration::from_secs(2 * 24 * 3600));
/// assert_eq!(events.len(), 1);
/// assert!(events[0].member_node_ids.contains(&"a".to_string()));
/// ```
pub fn assemble_events(records: &[MemoryRecord], window: Duration) -> Vec<Event> {
    if records.is_empty() {
        return Vec::new();
    }

    let n = records.len();
    let window_secs = window.as_secs() as i64;

    // ── Step 1: Extract entities for each record (tags + content) ────────────
    let record_entities: Vec<Vec<String>> = records
        .iter()
        .map(|record| {
            let mut entities = record.tags.clone();
            entities.extend(extract_entities(&record.content));
            entities.sort();
            entities.dedup();
            entities
        })
        .collect();

    // ── Step 2: Build entity → record-index list ──────────────────────────────
    let mut entity_to_indices: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, entities) in record_entities.iter().enumerate() {
        for entity in entities {
            entity_to_indices.entry(entity.clone()).or_default().push(i);
        }
    }

    // ── Step 3: Union-find ────────────────────────────────────────────────────
    // For each entity, sort its records by `created_at` and union consecutive
    // pairs that are within `window` of each other.
    let mut parent: Vec<usize> = (0..n).collect();

    for indices in entity_to_indices.values_mut() {
        indices.sort_by_key(|&i| records[i].created_at);
        for pair in indices.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let diff = (records[b].created_at - records[a].created_at)
                .num_seconds()
                .abs();
            if diff <= window_secs {
                uf_union(&mut parent, a, b);
            }
        }
    }

    // Path compression pass: compute all roots first (immutable borrow), then apply
    let roots: Vec<usize> = (0..n).map(|i| uf_find(&parent, i)).collect();
    parent
        .iter_mut()
        .zip(roots)
        .for_each(|(elem, root)| *elem = root);

    // ── Step 4: Collect components ────────────────────────────────────────────
    let mut components: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &root) in parent.iter().enumerate() {
        components.entry(root).or_default().push(i);
    }

    // ── Step 5: Build Event per component with ≥ 2 members ───────────────────
    let mut events: Vec<Event> = Vec::new();

    for member_indices in components.into_values() {
        if member_indices.len() < 2 {
            continue;
        }

        // Sort members by `created_at` for deterministic output
        let mut member_indices = member_indices;
        member_indices.sort_by_key(|&i| records[i].created_at);

        let earliest = records[*member_indices.first().unwrap()].created_at;
        let latest = records[*member_indices.last().unwrap()].created_at;

        // Union of all entities across members
        let mut all_entities: Vec<String> = member_indices
            .iter()
            .flat_map(|&i| record_entities[i].iter().cloned())
            .collect();
        all_entities.sort();
        all_entities.dedup();

        // Title: the most-frequent entity; alphabetic tiebreak for determinism
        let mut entity_freq: HashMap<String, usize> = HashMap::new();
        for &i in &member_indices {
            for entity in &record_entities[i] {
                *entity_freq.entry(entity.clone()).or_insert(0) += 1;
            }
        }
        let title = entity_freq
            .into_iter()
            .max_by(|(e1, c1), (e2, c2)| c1.cmp(c2).then_with(|| e1.cmp(e2)))
            .map(|(entity, _)| entity)
            .unwrap_or_else(|| "event".to_string());

        // Deterministic ID from sorted node_ids
        let mut node_ids: Vec<String> = member_indices
            .iter()
            .map(|&i| records[i].node_id.clone())
            .collect();
        node_ids.sort();
        let id = event_id_fnv1a(&node_ids);

        events.push(Event {
            id,
            title,
            time_range: (earliest, latest),
            entities: all_entities,
            member_node_ids: node_ids,
        });
    }

    // Sort events by start time; stable (sort is stable in Rust)
    events.sort_by_key(|e| e.time_range.0);
    events
}

// ── Union-find helpers (no path-compression to avoid borrow conflicts) ────────

fn uf_find(parent: &[usize], mut x: usize) -> usize {
    while parent[x] != x {
        x = parent[x];
    }
    x
}

fn uf_union(parent: &mut [usize], x: usize, y: usize) {
    let rx = uf_find(parent, x);
    let ry = uf_find(parent, y);
    if rx != ry {
        parent[rx] = ry;
    }
}

// ── FNV-1a 64-bit hash over sorted node_ids ───────────────────────────────────

fn event_id_fnv1a(sorted_node_ids: &[String]) -> String {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x00000100000001b3;
    let mut hash: u64 = FNV_OFFSET;
    for node_id in sorted_node_ids {
        for byte in node_id.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        // Separator between node_ids
        hash ^= b'\n' as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{hash:016x}")
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, time::Duration};

    use chrono::{Duration as ChronoDuration, Utc};

    use crate::{MemoryId, MemoryRecord, MemoryState, MemoryTier};

    use super::*;

    fn make_record(node_id: &str, content: &str, tags: Vec<&str>, days_old: i64) -> MemoryRecord {
        MemoryRecord {
            id: MemoryId::new(format!("id:{node_id}")),
            node_id: node_id.to_string(),
            content: content.to_string(),
            tags: tags.into_iter().map(str::to_string).collect(),
            metadata: BTreeMap::new(),
            tier: MemoryTier::L1Atom,
            created_at: Utc::now() - ChronoDuration::days(days_old),
            scope: None,
            agent_id: None,
            session_id: None,
            task_id: None,
            user_id: None,
            source: None,
            confidence: None,
            relations: Vec::new(),
            last_access: None,
            access_count: 0,
            state: MemoryState::Active,
        }
    }

    #[test]
    fn events_assemble_from_time_proximate_same_entity_records() {
        // Two records sharing "RateLimit" within a 2-day window → one event
        let records = vec![
            make_record(
                "node:a",
                "RateLimit exceeded on the API.",
                vec!["RateLimit"],
                5,
            ),
            make_record(
                "node:b",
                "RateLimit relaxed after fix.",
                vec!["RateLimit"],
                4,
            ),
            // Unrelated record: different entity, far in time
            make_record(
                "node:c",
                "DatabaseMigration completed.",
                vec!["DatabaseMigration"],
                100,
            ),
        ];
        let events = assemble_events(&records, Duration::from_secs(2 * 24 * 3600));
        assert_eq!(
            events.len(),
            1,
            "expected 1 event, got {:?}",
            events.iter().map(|e| &e.title).collect::<Vec<_>>()
        );
        assert!(
            events[0].member_node_ids.contains(&"node:a".to_string()),
            "node:a should be in event"
        );
        assert!(
            events[0].member_node_ids.contains(&"node:b".to_string()),
            "node:b should be in event"
        );
        assert!(
            !events[0].member_node_ids.contains(&"node:c".to_string()),
            "node:c (unrelated) should not be in event"
        );
    }

    #[test]
    fn events_do_not_group_records_outside_window() {
        // Same entity but 100 days apart → no event (1-day window)
        let records = vec![
            make_record("node:a", "RateLimit exceeded.", vec!["RateLimit"], 100),
            make_record("node:b", "RateLimit relaxed.", vec!["RateLimit"], 1),
        ];
        let events = assemble_events(&records, Duration::from_secs(24 * 3600));
        assert!(
            events.is_empty(),
            "records too far apart should not form an event"
        );
    }

    #[test]
    fn events_empty_input_returns_empty() {
        let events = assemble_events(&[], Duration::from_secs(3600));
        assert!(events.is_empty());
    }

    #[test]
    fn events_no_shared_entities_returns_empty() {
        // Close in time but no shared entities
        let records = vec![
            make_record("node:a", "Foo event happened.", vec!["Foo"], 1),
            make_record("node:b", "Bar event happened.", vec!["Bar"], 1),
        ];
        let events = assemble_events(&records, Duration::from_secs(7 * 24 * 3600));
        assert!(events.is_empty(), "no shared entities → no event");
    }

    #[test]
    fn event_id_is_deterministic() {
        let records = vec![
            make_record("node:x", "SharedEntity action.", vec!["SharedEntity"], 3),
            make_record("node:y", "SharedEntity followup.", vec!["SharedEntity"], 2),
        ];
        let window = Duration::from_secs(5 * 24 * 3600);
        let events1 = assemble_events(&records, window);
        let events2 = assemble_events(&records, window);
        assert!(!events1.is_empty(), "expected at least one event");
        assert_eq!(
            events1[0].id, events2[0].id,
            "event id must be deterministic across calls"
        );
    }

    #[test]
    fn events_ordered_by_start_time() {
        // Three pairs, creating three events; they should come out oldest-first
        let records = vec![
            make_record("node:a1", "AlphaEvent start.", vec!["AlphaEvent"], 30),
            make_record("node:a2", "AlphaEvent end.", vec!["AlphaEvent"], 29),
            make_record("node:b1", "BetaEvent start.", vec!["BetaEvent"], 10),
            make_record("node:b2", "BetaEvent end.", vec!["BetaEvent"], 9),
        ];
        let events = assemble_events(&records, Duration::from_secs(3 * 24 * 3600));
        assert_eq!(events.len(), 2);
        assert!(
            events[0].time_range.0 < events[1].time_range.0,
            "events should be ordered by start time (oldest first)"
        );
    }

    #[test]
    fn backward_compat_records_with_no_entities_are_graceful() {
        // Records with no tags and no extractable entities → no event (no shared entity)
        let records = vec![
            make_record("node:a", "plain text without entities", vec![], 1),
            make_record("node:b", "more plain text here", vec![], 1),
        ];
        let events = assemble_events(&records, Duration::from_secs(3 * 24 * 3600));
        assert!(
            events.is_empty(),
            "records with no entities should not form events (backward-compat)"
        );
    }

    #[test]
    fn event_title_is_most_frequent_entity() {
        // "Shared" appears in both records, "Unique" only in one → title should be "Shared"
        let records = vec![
            make_record(
                "node:a",
                "Shared something happened.",
                vec!["Shared", "Unique"],
                2,
            ),
            make_record("node:b", "Shared again.", vec!["Shared"], 1),
        ];
        let events = assemble_events(&records, Duration::from_secs(5 * 24 * 3600));
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].title, "Shared");
    }

    #[test]
    fn single_member_groups_are_not_events() {
        // Two records with different entities, close in time — each would be a singleton component
        let records = vec![
            make_record("node:a", "OnlyA happened.", vec!["OnlyA"], 1),
            make_record("node:b", "OnlyB happened.", vec!["OnlyB"], 1),
        ];
        let events = assemble_events(&records, Duration::from_secs(10 * 24 * 3600));
        assert!(
            events.is_empty(),
            "singleton components should not produce events"
        );
    }
}
