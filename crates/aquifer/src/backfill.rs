// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};

use crate::{
    files::parse_record, identity::stable_memory_id, MemoryBackend, MemoryRecord, MemoryResult,
    MemoryTier, StoreMemory,
};

const MAX_SECTION_CHARS: usize = 8_000;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackfillStats {
    pub scanned: usize,
    pub imported: usize,
    pub skipped_duplicates: usize,
    pub failed: Vec<BackfillFailure>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackfillFailure {
    pub file: PathBuf,
    pub reason: String,
}

pub async fn backfill_directory(
    backend: &dyn MemoryBackend,
    directory: impl AsRef<Path>,
) -> MemoryResult<BackfillStats> {
    let mut paths = collect_memory_paths(directory.as_ref())?;
    paths.sort();

    let mut stats = BackfillStats::default();
    for path in paths {
        stats.scanned += 1;
        let memories = match parse_memory_path(&path) {
            Ok(memories) => memories,
            Err(error) => {
                stats.failed.push(BackfillFailure {
                    file: path,
                    reason: error.to_string(),
                });
                continue;
            }
        };
        for memory in memories {
            let id = stable_memory_id(&memory);
            match backend.get_node(id.as_str()).await {
                Ok(Some(_)) => {
                    stats.skipped_duplicates += 1;
                    continue;
                }
                Ok(None) => {}
                Err(error) => {
                    stats.failed.push(BackfillFailure {
                        file: path.clone(),
                        reason: error.to_string(),
                    });
                    continue;
                }
            }
            if let Err(error) = backend.store(memory).await {
                stats.failed.push(BackfillFailure {
                    file: path.clone(),
                    reason: error.to_string(),
                });
                continue;
            }
            stats.imported += 1;
        }
    }
    Ok(stats)
}

pub fn collect_memory_paths(directory: &Path) -> MemoryResult<Vec<PathBuf>> {
    let mut paths = Vec::new();
    collect_memory_paths_into(directory, &mut paths)?;
    Ok(paths)
}

fn collect_memory_paths_into(directory: &Path, paths: &mut Vec<PathBuf>) -> MemoryResult<()> {
    if !directory.exists() {
        return Ok(());
    }
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            // Never descend into hidden dirs (.git/.claude/.fastembed_cache/.artesian/…) or common
            // vendor/build/cache dirs — they hold tooling config and model caches, not memory notes.
            if !is_skippable_dir(&path) {
                collect_memory_paths_into(&path, paths)?;
            }
        } else if path.extension().is_some_and(|extension| {
            extension.eq_ignore_ascii_case("md") || extension.eq_ignore_ascii_case("json")
        }) && !is_reserved_okf_file(&path)
        {
            paths.push(path);
        }
    }
    Ok(())
}

/// Directories that never hold user memory notes: hidden dirs (any name starting with `.`, e.g.
/// `.git`, `.claude`, `.fastembed_cache`, `.artesian`, `.venv`) and common vendor/build/cache dirs.
/// Skipping them keeps the import from trying to parse tooling config and model-cache files (e.g. a
/// million-line `tokenizer.json`) as memory.
fn is_skippable_dir(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    name.starts_with('.')
        || matches!(
            name,
            "node_modules" | "target" | "__pycache__" | "venv" | "dist" | "build" | "vendor"
        )
}

pub fn parse_memory_path(path: &Path) -> MemoryResult<Vec<StoreMemory>> {
    let text = std::fs::read_to_string(path)?;
    match path.extension().and_then(|extension| extension.to_str()) {
        Some(extension) if extension.eq_ignore_ascii_case("json") => parse_json_memory(&text, path),
        _ => parse_markdown_memory(&text, path),
    }
}

fn parse_json_memory(text: &str, path: &Path) -> MemoryResult<Vec<StoreMemory>> {
    if let Ok(record) = serde_json::from_str::<MemoryRecord>(text) {
        return Ok(vec![StoreMemory {
            content: record.content,
            tags: record.tags,
            metadata: record.metadata,
            tier: record.tier,
            node_id: Some(record.node_id),
            created_at: Some(record.created_at),
            scope: record.scope,
            agent_id: record.agent_id,
            session_id: record.session_id,
            task_id: record.task_id,
            user_id: record.user_id,
            source: record.source,
            confidence: record.confidence,
            relations: record.relations,
        }]);
    }

    let mut memory = serde_json::from_str::<StoreMemory>(text)?;
    memory
        .metadata
        .entry("source_path".to_string())
        .or_insert_with(|| path.display().to_string());
    Ok(vec![memory])
}

fn parse_markdown_memory(text: &str, path: &Path) -> MemoryResult<Vec<StoreMemory>> {
    if let Ok(record) = parse_record(text) {
        let mut metadata = record.metadata;
        metadata
            .entry("okf_type".to_string())
            .or_insert_with(|| "memory".to_string());
        let memory = StoreMemory {
            content: record.content,
            tags: record.tags,
            metadata,
            tier: record.tier,
            node_id: Some(record.node_id),
            created_at: Some(record.created_at),
            scope: record.scope,
            agent_id: record.agent_id,
            session_id: record.session_id,
            task_id: record.task_id,
            user_id: record.user_id,
            source: record.source,
            confidence: record.confidence,
            relations: record.relations,
        };
        return Ok(section_aware_chunks(memory, path));
    }

    let trimmed = text.trim();
    let (created_at, content) = parse_date_tagged_body(trimmed);
    let mut metadata = BTreeMap::new();
    metadata.insert("source_path".to_string(), path.display().to_string());
    metadata.insert("okf_type".to_string(), "memory".to_string());
    let memory = StoreMemory {
        content,
        tags: Vec::new(),
        metadata,
        tier: MemoryTier::L1Atom,
        node_id: None,
        created_at,
        scope: None,
        agent_id: None,
        session_id: None,
        task_id: None,
        user_id: None,
        source: None,
        confidence: None,
        relations: Vec::new(),
    };
    Ok(section_aware_chunks(memory, path))
}

fn section_aware_chunks(memory: StoreMemory, path: &Path) -> Vec<StoreMemory> {
    let chunks = split_markdown_sections(&memory.content);
    if chunks.len() <= 1 {
        return vec![memory];
    }
    let count = chunks.len();
    chunks
        .into_iter()
        .enumerate()
        .map(|(index, chunk)| {
            let mut chunk_memory = memory.clone();
            chunk_memory.content = chunk.content;
            chunk_memory
                .metadata
                .entry("source_path".to_string())
                .or_insert_with(|| path.display().to_string());
            chunk_memory
                .metadata
                .insert("chunk_index".to_string(), (index + 1).to_string());
            chunk_memory
                .metadata
                .insert("chunk_count".to_string(), count.to_string());
            if let Some(heading) = chunk.heading {
                chunk_memory.metadata.insert("heading".to_string(), heading);
            }
            if let Some(node_id) = &memory.node_id {
                chunk_memory.node_id = Some(format!("{node_id}#chunk-{}", index + 1));
            }
            chunk_memory
        })
        .collect()
}

#[derive(Debug)]
struct MarkdownChunk {
    heading: Option<String>,
    content: String,
}

fn split_markdown_sections(text: &str) -> Vec<MarkdownChunk> {
    let mut sections = Vec::new();
    let mut current = Vec::new();
    let mut heading = None;

    for line in text.lines() {
        if is_markdown_heading(line) {
            if !current.is_empty() {
                push_sized_section(&mut sections, heading.take(), current.join("\n"));
                current.clear();
            }
            heading = Some(line.trim_start_matches('#').trim().to_string());
        }
        current.push(line.to_string());
    }

    if !current.is_empty() {
        push_sized_section(&mut sections, heading, current.join("\n"));
    }

    if sections.is_empty() {
        push_sized_section(&mut sections, None, text.to_string());
    }

    sections
        .into_iter()
        .filter(|chunk| !chunk.content.trim().is_empty())
        .collect()
}

fn is_markdown_heading(line: &str) -> bool {
    let without_markers = line.trim_start_matches('#');
    without_markers.len() < line.len() && without_markers.starts_with(' ')
}

fn push_sized_section(sections: &mut Vec<MarkdownChunk>, heading: Option<String>, content: String) {
    if content.len() <= MAX_SECTION_CHARS {
        sections.push(MarkdownChunk { heading, content });
        return;
    }

    let mut part = String::new();
    let mut part_index = 1usize;
    for paragraph in content.split("\n\n") {
        if !part.is_empty() && part.len() + paragraph.len() + 2 > MAX_SECTION_CHARS {
            sections.push(MarkdownChunk {
                heading: heading
                    .as_ref()
                    .map(|heading| format!("{heading} part {part_index}")),
                content: part.trim().to_string(),
            });
            part.clear();
            part_index += 1;
        }
        if !part.is_empty() {
            part.push_str("\n\n");
        }
        part.push_str(paragraph);
    }
    if !part.trim().is_empty() {
        sections.push(MarkdownChunk {
            heading: heading.map(|heading| format!("{heading} part {part_index}")),
            content: part.trim().to_string(),
        });
    }
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

fn is_reserved_okf_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| matches!(name, "index.md" | "log.md"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collect_memory_paths_skips_hidden_and_vendor_dirs() {
        let tmp = std::env::temp_dir().join(format!("artesian-collect-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("sub")).unwrap();
        std::fs::write(tmp.join("notes.md"), "# note").unwrap();
        std::fs::write(tmp.join("sub").join("more.md"), "# more").unwrap();
        // Tooling config / model cache / vendor dirs that must NOT be scanned as memory.
        for (dir, file) in [
            (".claude", "settings.local.json"),
            (".fastembed_cache", "tokenizer.json"),
            (".git", "config"),
            ("node_modules", "pkg.md"),
            ("target", "build.json"),
        ] {
            std::fs::create_dir_all(tmp.join(dir)).unwrap();
            std::fs::write(tmp.join(dir).join(file), "{}").unwrap();
        }
        let mut got: Vec<String> = collect_memory_paths(&tmp)
            .unwrap()
            .into_iter()
            .map(|path| {
                path.strip_prefix(&tmp)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/")
            })
            .collect();
        got.sort();
        assert_eq!(got, vec!["notes.md".to_string(), "sub/more.md".to_string()]);
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
