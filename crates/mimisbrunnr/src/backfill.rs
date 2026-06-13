// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use chrono::{DateTime, NaiveDate, Utc};

use crate::{
    files::parse_record, identity::stable_memory_id, MemoryBackend, MemoryRecord, MemoryResult,
    MemoryTier, StoreMemory,
};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BackfillStats {
    pub scanned: usize,
    pub imported: usize,
    pub skipped_duplicates: usize,
}

pub async fn backfill_directory(
    backend: &dyn MemoryBackend,
    directory: impl AsRef<Path>,
) -> MemoryResult<BackfillStats> {
    let mut paths = Vec::new();
    collect_memory_paths(directory.as_ref(), &mut paths)?;
    paths.sort();

    let mut stats = BackfillStats::default();
    for path in paths {
        stats.scanned += 1;
        let memory = parse_memory_path(&path)?;
        let id = stable_memory_id(&memory);
        if backend.get_node(id.as_str()).await?.is_some() {
            stats.skipped_duplicates += 1;
            continue;
        }
        backend.store(memory).await?;
        stats.imported += 1;
    }
    Ok(stats)
}

fn collect_memory_paths(directory: &Path, paths: &mut Vec<PathBuf>) -> MemoryResult<()> {
    if !directory.exists() {
        return Ok(());
    }
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_memory_paths(&path, paths)?;
        } else if path.extension().is_some_and(|extension| {
            extension.eq_ignore_ascii_case("md") || extension.eq_ignore_ascii_case("json")
        }) {
            paths.push(path);
        }
    }
    Ok(())
}

fn parse_memory_path(path: &Path) -> MemoryResult<StoreMemory> {
    let text = std::fs::read_to_string(path)?;
    match path.extension().and_then(|extension| extension.to_str()) {
        Some(extension) if extension.eq_ignore_ascii_case("json") => parse_json_memory(&text, path),
        _ => parse_markdown_memory(&text, path),
    }
}

fn parse_json_memory(text: &str, path: &Path) -> MemoryResult<StoreMemory> {
    if let Ok(record) = serde_json::from_str::<MemoryRecord>(text) {
        return Ok(StoreMemory {
            content: record.content,
            tags: record.tags,
            metadata: record.metadata,
            tier: record.tier,
            node_id: Some(record.node_id),
            created_at: Some(record.created_at),
        });
    }

    let mut memory = serde_json::from_str::<StoreMemory>(text)?;
    memory
        .metadata
        .entry("source_path".to_string())
        .or_insert_with(|| path.display().to_string());
    Ok(memory)
}

fn parse_markdown_memory(text: &str, path: &Path) -> MemoryResult<StoreMemory> {
    if let Ok(record) = parse_record(text) {
        return Ok(StoreMemory {
            content: record.content,
            tags: record.tags,
            metadata: record.metadata,
            tier: record.tier,
            node_id: Some(record.node_id),
            created_at: Some(record.created_at),
        });
    }

    let trimmed = text.trim();
    let (created_at, content) = parse_date_tagged_body(trimmed);
    let mut metadata = BTreeMap::new();
    metadata.insert("source_path".to_string(), path.display().to_string());
    Ok(StoreMemory {
        content,
        tags: Vec::new(),
        metadata,
        tier: MemoryTier::L1Atom,
        node_id: None,
        created_at,
    })
}

fn parse_date_tagged_body(text: &str) -> (Option<DateTime<Utc>>, String) {
    let Some(rest) = text.strip_prefix('[') else {
        return (None, text.to_string());
    };
    let Some((date_tag, content)) = rest.split_once("] ") else {
        return (None, text.to_string());
    };
    let created_at = NaiveDate::parse_from_str(date_tag, "%Y-%m-%d")
        .ok()
        .and_then(|date| date.and_hms_opt(0, 0, 0))
        .map(|datetime| DateTime::<Utc>::from_naive_utc_and_offset(datetime, Utc));
    (created_at, content.to_string())
}
