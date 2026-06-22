// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::{MemoryRecord, SearchHit, SearchSource};

/// In-memory entity → records index.
///
/// Built incrementally at write time by [`EntityIndex::index_record`]; queried at read time by
/// [`EntityIndex::entity_hits`]. Session-scoped — accumulates records stored in the current
/// process, not persisted separately from the vector store.
#[derive(Debug, Default)]
pub struct EntityIndex {
    by_entity: HashMap<String, Vec<String>>,
    by_node: HashMap<String, (MemoryRecord, Vec<String>)>,
}

impl EntityIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Index a stored record, extracting entities from its content and tags.
    pub fn index_record(&mut self, record: MemoryRecord) {
        let mut entities = record.tags.clone();
        entities.extend(extract_entities(&record.content));
        entities.sort();
        entities.dedup();
        let node_id = record.node_id.clone();
        for entity in &entities {
            self.by_entity
                .entry(entity.clone())
                .or_default()
                .push(node_id.clone());
        }
        self.by_node.insert(node_id, (record, entities));
    }

    /// Returns hits scored by entity overlap between `query` and stored records.
    pub fn entity_hits(&self, query: &str, limit: usize) -> Vec<SearchHit> {
        let mut query_entities = extract_entities(query);
        query_entities.sort();
        query_entities.dedup();

        if query_entities.is_empty() {
            return Vec::new();
        }

        let mut overlap_counts: HashMap<&str, usize> = HashMap::new();
        for entity in &query_entities {
            if let Some(node_ids) = self.by_entity.get(entity) {
                for node_id in node_ids {
                    *overlap_counts.entry(node_id.as_str()).or_insert(0) += 1;
                }
            }
        }

        let query_entity_count = query_entities.len() as f32;
        let mut hits: Vec<SearchHit> = overlap_counts
            .into_iter()
            .filter_map(|(node_id, count)| {
                let (record, _) = self.by_node.get(node_id)?;
                Some(SearchHit {
                    record: record.clone(),
                    score: count as f32 / query_entity_count,
                    source: SearchSource::Keyword,
                })
            })
            .collect();

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.record.node_id.cmp(&b.record.node_id))
        });
        hits.truncate(limit);
        hits
    }

    /// Returns the entity list for a stored record (used by supersession logic).
    pub fn record_entities(&self, node_id: &str) -> Option<&[String]> {
        self.by_node
            .get(node_id)
            .map(|(_, entities)| entities.as_slice())
    }
}

/// Extract named entities from text deterministically (no LLM required).
///
/// Captures:
/// - Backtick-quoted identifiers: `` `identifier` ``
/// - Double-quoted short phrases: `"Term"`
/// - CamelCase / PascalCase identifiers (contains uppercase after a lowercase letter)
/// - ALL-CAPS acronyms (3+ characters)
pub fn extract_entities(text: &str) -> Vec<String> {
    let mut entities = Vec::new();
    extract_backtick_quoted(text, &mut entities);
    extract_double_quoted(text, &mut entities);
    extract_word_entities(text, &mut entities);
    entities
}

fn extract_backtick_quoted(text: &str, entities: &mut Vec<String>) {
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '`' {
            let mut term = String::new();
            for inner in chars.by_ref() {
                if inner == '`' {
                    break;
                }
                term.push(inner);
            }
            let trimmed = term.trim().to_string();
            if trimmed.len() >= 2 {
                entities.push(trimmed);
            }
        }
    }
}

fn extract_double_quoted(text: &str, entities: &mut Vec<String>) {
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '"' {
            let mut term = String::new();
            for inner in chars.by_ref() {
                if inner == '"' {
                    break;
                }
                term.push(inner);
            }
            let trimmed = term.trim().to_string();
            // Only keep short phrases that don't span sentences
            if trimmed.len() >= 2 && trimmed.len() <= 60 && !trimmed.contains('\n') {
                entities.push(trimmed);
            }
        }
    }
}

fn extract_word_entities(text: &str, entities: &mut Vec<String>) {
    for word in text.split(|c: char| !c.is_alphanumeric() && c != '_') {
        let word = word.trim_matches('_');
        if word.len() < 3 {
            continue;
        }
        if is_camel_or_pascal(word) || is_all_caps(word) {
            entities.push(word.to_string());
        }
    }
}

fn is_camel_or_pascal(word: &str) -> bool {
    let chars: Vec<char> = word.chars().collect();
    if chars.len() < 2 {
        return false;
    }
    // Must have at least one lowercase letter (not all-caps) and at least one uppercase
    // after position 0 (PascalCase/camelCase, not plain capitalized word).
    let has_lowercase = chars.iter().any(|c| c.is_ascii_lowercase());
    let has_inner_upper = chars[1..].iter().any(|c| c.is_ascii_uppercase());
    has_lowercase && has_inner_upper
}

fn is_all_caps(word: &str) -> bool {
    word.len() >= 3
        && word
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
        && word.chars().any(|c| c.is_ascii_uppercase())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use chrono::Utc;

    use crate::{MemoryId, MemoryTier};

    use super::*;

    fn make_record(node_id: &str, content: &str, tags: Vec<&str>) -> MemoryRecord {
        MemoryRecord {
            id: MemoryId::new(format!("id:{node_id}")),
            node_id: node_id.to_string(),
            content: content.to_string(),
            tags: tags.into_iter().map(str::to_string).collect(),
            metadata: BTreeMap::new(),
            tier: MemoryTier::L1Atom,
            created_at: Utc::now(),
            scope: None,
            agent_id: None,
            session_id: None,
            task_id: None,
            user_id: None,
            source: None,
            confidence: None,
            relations: Vec::new(),
        }
    }

    #[test]
    fn extract_backtick_terms() {
        let entities = extract_entities("Use `PostgreSQLConnectionPool` for connections.");
        assert!(
            entities.contains(&"PostgreSQLConnectionPool".to_string()),
            "entities: {entities:?}"
        );
    }

    #[test]
    fn extract_camel_case() {
        let entities = extract_entities("BackgroundJobRetryPolicy uses exponential backoff.");
        assert!(
            entities.contains(&"BackgroundJobRetryPolicy".to_string()),
            "entities: {entities:?}"
        );
    }

    #[test]
    fn extract_all_caps() {
        let entities = extract_entities("The API rate limit is controlled by the TTL policy.");
        assert!(
            entities.contains(&"API".to_string()),
            "entities: {entities:?}"
        );
        assert!(
            entities.contains(&"TTL".to_string()),
            "entities: {entities:?}"
        );
    }

    #[test]
    fn extract_double_quoted() {
        let entities = extract_entities("The \"ProductionDatabase\" uses PostgreSQL.");
        assert!(
            entities.contains(&"ProductionDatabase".to_string()),
            "entities: {entities:?}"
        );
    }

    #[test]
    fn entity_index_scores_overlap() {
        let record = make_record(
            "node:1",
            "BackgroundJobRetryPolicy uses exponential backoff with 3 max attempts.",
            vec!["retry", "background-jobs"],
        );
        let mut index = EntityIndex::new();
        index.index_record(record);

        let hits = index.entity_hits("BackgroundJobRetryPolicy retry policy", 10);
        assert!(!hits.is_empty(), "expected entity overlap hits");
        assert_eq!(hits[0].record.node_id, "node:1");
    }

    #[test]
    fn entity_index_disambiguates_similar_terms() {
        let record_a = make_record(
            "node:a",
            "BackgroundJobRetryPolicy: 3 max attempts for queue workers.",
            vec!["background-jobs"],
        );
        let record_b = make_record(
            "node:b",
            "OutboundCallRetryPolicy: 5 max attempts for HTTP calls.",
            vec!["outbound"],
        );
        let mut index = EntityIndex::new();
        index.index_record(record_a);
        index.index_record(record_b);

        let hits = index.entity_hits("BackgroundJobRetryPolicy", 10);
        assert_eq!(
            hits.len(),
            1,
            "entity filter should return only the matching record"
        );
        assert_eq!(hits[0].record.node_id, "node:a");
    }
}
