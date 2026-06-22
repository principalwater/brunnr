// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use futures_util::{future::BoxFuture, FutureExt};
use serde::{Deserialize, Serialize};

use crate::{
    entity::extract_entities, MemoryBackend, MemoryRecord, MemoryResult, SearchHit, SearchSource,
};

pub const DEFAULT_GRAPH_HOPS: usize = 1;
pub const MAX_GRAPH_HOPS: usize = 3;
pub const MAX_RELATIONS_PER_RECORD: usize = 16;
pub const GRAPH_SCAN_LIMIT: usize = 512;
pub const GRAPH_EXPANSION_LIMIT: usize = 64;

const MAX_ENTITY_CHARS: usize = 128;
const MAX_PREDICATE_CHARS: usize = 64;
const EXTRACTED_RELATION_LIMIT: usize = 8;

/// Explicit entity-relation edge attached to the record that asserted it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Relation {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub source_node_id: String,
}

impl Relation {
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
        source_node_id: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
            source_node_id: source_node_id.into(),
        }
    }
}

pub fn normalize_relations(
    relations: impl IntoIterator<Item = Relation>,
    source_node_id: &str,
) -> Vec<Relation> {
    let mut seen = BTreeSet::new();
    let mut normalized = Vec::new();
    for relation in relations {
        let Some(relation) = normalize_relation(relation, source_node_id) else {
            continue;
        };
        if seen.insert(relation.clone()) {
            normalized.push(relation);
        }
        if normalized.len() >= MAX_RELATIONS_PER_RECORD {
            break;
        }
    }
    normalized
}

pub fn extract_entity_relations(
    content: &str,
    tags: &[String],
    source_node_id: &str,
) -> Vec<Relation> {
    let mut entities = tags.to_vec();
    entities.extend(extract_entities(content));
    entities.sort();
    entities.dedup();
    normalize_relations(
        entities
            .into_iter()
            .take(EXTRACTED_RELATION_LIMIT)
            .map(|entity| Relation::new(source_node_id, "mentions", entity, source_node_id)),
        source_node_id,
    )
}

pub fn by_entity_node_ids(records: &[MemoryRecord], entity: &str) -> Vec<String> {
    let entity = entity.trim();
    if entity.is_empty() {
        return Vec::new();
    }

    let mut seen = BTreeSet::new();
    let mut node_ids = Vec::new();
    for record in records {
        for relation in &record.relations {
            if relation_mentions_entity(relation, entity)
                && seen.insert(relation.source_node_id.clone())
            {
                node_ids.push(relation.source_node_id.clone());
                if node_ids.len() >= GRAPH_EXPANSION_LIMIT {
                    return node_ids;
                }
            }
        }
    }
    node_ids
}

pub fn neighbor_node_ids(records: &[MemoryRecord], node_id: &str, hops: usize) -> Vec<String> {
    if node_id.trim().is_empty() || hops == 0 {
        return Vec::new();
    }

    let hops = hops.min(MAX_GRAPH_HOPS);
    let mut source_to_entities: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut entity_to_sources: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for record in records {
        for relation in &record.relations {
            for entity in relation_entities(relation) {
                push_unique(
                    source_to_entities
                        .entry(relation.source_node_id.clone())
                        .or_default(),
                    entity.clone(),
                );
                push_unique(
                    entity_to_sources.entry(entity).or_default(),
                    relation.source_node_id.clone(),
                );
            }
        }
    }

    let mut visited = BTreeSet::from([node_id.to_string()]);
    let mut frontier = vec![node_id.to_string()];
    let mut output = Vec::new();
    for _ in 0..hops {
        let mut next = Vec::new();
        for source in frontier {
            let Some(entities) = source_to_entities.get(&source) else {
                continue;
            };
            for entity in entities {
                let Some(candidates) = entity_to_sources.get(entity) else {
                    continue;
                };
                for candidate in candidates {
                    if visited.insert(candidate.clone()) {
                        output.push(candidate.clone());
                        next.push(candidate.clone());
                        if output.len() >= GRAPH_EXPANSION_LIMIT {
                            return output;
                        }
                    }
                }
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }
    output
}

pub fn records_by_node_ids(records: &[MemoryRecord], node_ids: Vec<String>) -> Vec<MemoryRecord> {
    node_ids
        .into_iter()
        .filter_map(|node_id| {
            records
                .iter()
                .find(|record| record.node_id == node_id || record.id.as_str() == node_id)
                .cloned()
        })
        .collect()
}

pub fn expand_hits_with_neighbors<'a, B: MemoryBackend + ?Sized>(
    backend: &'a B,
    hits: Vec<SearchHit>,
    hops: usize,
) -> BoxFuture<'a, MemoryResult<Vec<SearchHit>>> {
    async move {
        let mut seen: BTreeSet<String> =
            hits.iter().map(|hit| hit.record.node_id.clone()).collect();
        let seeds: Vec<String> = hits.iter().map(|hit| hit.record.node_id.clone()).collect();
        let mut expanded = hits;
        let mut appended = 0usize;
        for seed in seeds {
            let neighbors = backend.neighbors(&seed, hops).await?;
            for record in neighbors {
                if seen.insert(record.node_id.clone()) {
                    expanded.push(SearchHit {
                        record,
                        score: 0.0,
                        source: SearchSource::Keyword,
                    });
                    appended += 1;
                    if appended >= GRAPH_EXPANSION_LIMIT {
                        return Ok(expanded);
                    }
                }
            }
        }
        Ok(expanded)
    }
    .boxed()
}

fn normalize_relation(relation: Relation, source_node_id: &str) -> Option<Relation> {
    let subject = bounded_trim(relation.subject, MAX_ENTITY_CHARS);
    let predicate = bounded_trim(relation.predicate, MAX_PREDICATE_CHARS);
    let object = bounded_trim(relation.object, MAX_ENTITY_CHARS);
    let source_node_id = bounded_trim(source_node_id.to_string(), MAX_ENTITY_CHARS);
    if subject.is_empty() || predicate.is_empty() || object.is_empty() || source_node_id.is_empty()
    {
        return None;
    }
    Some(Relation {
        subject,
        predicate,
        object,
        source_node_id,
    })
}

fn bounded_trim(value: String, max_chars: usize) -> String {
    value.trim().chars().take(max_chars).collect()
}

fn relation_mentions_entity(relation: &Relation, entity: &str) -> bool {
    relation.subject == entity || relation.object == entity
}

fn relation_entities(relation: &Relation) -> [String; 2] {
    [relation.subject.clone(), relation.object.clone()]
}

fn push_unique(values: &mut Vec<String>, value: String) {
    if !values.contains(&value) {
        values.push(value);
    }
}
