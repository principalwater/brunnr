// SPDX-License-Identifier: Apache-2.0

use std::{
    cmp::Reverse,
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use chrono::{DateTime, Utc};
use futures_util::{future::BoxFuture, FutureExt};
use serde::{Deserialize, Serialize};
use tokio::fs;

use crate::{
    identity::stable_memory_id, MemoryBackend, MemoryError, MemoryId, MemoryQuery, MemoryRecord,
    MemoryResult, MemoryTier, SearchHit, SearchSource, StoreMemory,
};

#[derive(Debug, Clone)]
pub struct FilesBackend {
    root: PathBuf,
}

impl FilesBackend {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn memory_dir(&self) -> PathBuf {
        self.root.join("memory")
    }

    fn record_path(&self, date_tag: &str, id: &MemoryId) -> PathBuf {
        self.memory_dir().join(date_tag).join(format!("{id}.md"))
    }

    fn load_records(&self) -> MemoryResult<Vec<MemoryRecord>> {
        let memory_dir = self.memory_dir();
        if !memory_dir.exists() {
            return Ok(Vec::new());
        }

        let mut records = Vec::new();
        collect_records(&memory_dir, &mut records)?;
        records.sort_by_key(|record| Reverse(record.created_at));
        Ok(records)
    }
}

impl MemoryBackend for FilesBackend {
    fn find(&self, query: MemoryQuery) -> BoxFuture<'_, MemoryResult<Vec<SearchHit>>> {
        async move {
            let terms = query_terms(&query.text);
            let mut hits = self
                .load_records()?
                .into_iter()
                .filter(|record| matches_node_filter(record, query.node_id.as_deref()))
                .filter(|record| matches_tags(record, &query.tags))
                .filter_map(|record| score_record(record, &terms))
                .collect::<Vec<_>>();

            hits.sort_by(|left, right| {
                right
                    .score
                    .partial_cmp(&left.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| right.record.created_at.cmp(&left.record.created_at))
            });
            hits.truncate(query.limit);
            Ok(hits)
        }
        .boxed()
    }

    fn store(&self, memory: StoreMemory) -> BoxFuture<'_, MemoryResult<MemoryRecord>> {
        async move {
            let id = stable_memory_id(&memory);
            let existing_path = find_existing_record_path(&self.memory_dir(), &id)?;
            if let Some(path) = existing_path {
                let text = fs::read_to_string(path).await?;
                return parse_record(&text);
            }

            let now = memory.created_at.unwrap_or_else(Utc::now);
            let date_tag = now.format("%Y-%m-%d").to_string();
            let node_id = memory.node_id.unwrap_or_else(|| format!("node:{id}"));
            let record = MemoryRecord {
                id,
                node_id,
                content: memory.content,
                tags: memory.tags,
                metadata: memory.metadata,
                tier: memory.tier,
                created_at: now,
            };
            let path = self.record_path(&date_tag, &record.id);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).await?;
            }
            fs::write(path, render_record(&record)?).await?;
            Ok(record)
        }
        .boxed()
    }

    fn get_node(&self, node_id: &str) -> BoxFuture<'_, MemoryResult<Option<MemoryRecord>>> {
        let node_id = node_id.to_string();
        async move {
            Ok(self
                .load_records()?
                .into_iter()
                .find(|record| record.node_id == node_id || record.id.as_str() == node_id))
        }
        .boxed()
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct FileHeader {
    id: MemoryId,
    node_id: String,
    tier: MemoryTier,
    tags: Vec<String>,
    metadata: BTreeMap<String, String>,
    created_at: DateTime<Utc>,
}

fn collect_records(dir: &Path, records: &mut Vec<MemoryRecord>) -> MemoryResult<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_records(&path, records)?;
        } else if path.extension().is_some_and(|extension| extension == "md") {
            let text = std::fs::read_to_string(&path)?;
            records.push(parse_record(&text)?);
        }
    }
    Ok(())
}

fn render_record(record: &MemoryRecord) -> MemoryResult<String> {
    let header = FileHeader {
        id: record.id.clone(),
        node_id: record.node_id.clone(),
        tier: record.tier,
        tags: record.tags.clone(),
        metadata: record.metadata.clone(),
        created_at: record.created_at,
    };
    let date_tag = record.created_at.format("%Y-%m-%d");
    Ok(format!(
        "+++\n{}+++\n\n[{date_tag}] {}\n",
        toml::to_string(&header)?,
        record.content
    ))
}

pub(crate) fn parse_record(text: &str) -> MemoryResult<MemoryRecord> {
    let rest = text
        .strip_prefix("+++\n")
        .ok_or_else(|| MemoryError::InvalidFile("missing TOML front matter".to_string()))?;
    let (header, body) = rest
        .split_once("\n+++\n")
        .ok_or_else(|| MemoryError::InvalidFile("unterminated TOML front matter".to_string()))?;
    let header: FileHeader = toml::from_str(header)?;
    let date_tag = header.created_at.format("%Y-%m-%d").to_string();
    let content = body
        .trim()
        .strip_prefix(&format!("[{date_tag}] "))
        .unwrap_or_else(|| body.trim())
        .to_string();

    Ok(MemoryRecord {
        id: header.id,
        node_id: header.node_id,
        content,
        tags: header.tags,
        metadata: header.metadata,
        tier: header.tier,
        created_at: header.created_at,
    })
}

fn find_existing_record_path(root: &Path, id: &MemoryId) -> MemoryResult<Option<PathBuf>> {
    if !root.exists() {
        return Ok(None);
    }
    let mut paths = Vec::new();
    collect_paths(root, &mut paths)?;
    Ok(paths
        .into_iter()
        .find(|path| path.file_stem().and_then(|stem| stem.to_str()) == Some(id.as_str())))
}

fn collect_paths(dir: &Path, paths: &mut Vec<PathBuf>) -> MemoryResult<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_paths(&path, paths)?;
        } else if path.extension().is_some_and(|extension| extension == "md") {
            paths.push(path);
        }
    }
    Ok(())
}

fn query_terms(input: &str) -> Vec<String> {
    input
        .split_whitespace()
        .map(str::to_ascii_lowercase)
        .filter(|term| !term.is_empty())
        .collect()
}

fn matches_node_filter(record: &MemoryRecord, node_id: Option<&str>) -> bool {
    node_id.is_none_or(|node_id| record.node_id == node_id || record.id.as_str() == node_id)
}

fn matches_tags(record: &MemoryRecord, tags: &[String]) -> bool {
    tags.iter().all(|tag| record.tags.contains(tag))
}

fn score_record(record: MemoryRecord, terms: &[String]) -> Option<SearchHit> {
    if terms.is_empty() {
        return Some(SearchHit {
            record,
            score: 1.0,
            source: SearchSource::Keyword,
        });
    }

    let haystack = format!(
        "{} {} {}",
        record.content.to_ascii_lowercase(),
        record.tags.join(" ").to_ascii_lowercase(),
        record
            .metadata
            .values()
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
            .to_ascii_lowercase()
    );
    let score = terms
        .iter()
        .map(|term| haystack.matches(term).count() as f32)
        .sum::<f32>();
    (score > 0.0).then_some(SearchHit {
        record,
        score,
        source: SearchSource::Keyword,
    })
}
