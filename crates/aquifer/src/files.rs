// SPDX-License-Identifier: Apache-2.0

use std::{
    cmp::Reverse,
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use chrono::{DateTime, Utc};
use futures_util::{future::BoxFuture, FutureExt};
use serde::{Deserialize, Serialize};
use tokio::{fs, io::AsyncWriteExt};

use crate::{
    identity::stable_memory_id, MemoryBackend, MemoryError, MemoryId, MemoryQuery, MemoryRecord,
    MemoryResult, MemoryScope, MemoryTier, SearchHit, SearchSource, SessionLaneLock, StoreMemory,
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
        let read_root = if memory_dir.exists() {
            memory_dir
        } else if self.root.exists() {
            self.root.clone()
        } else {
            return Ok(Vec::new());
        };

        let mut records = Vec::new();
        collect_records(&read_root, &mut records)?;
        records.sort_by_key(|record| Reverse(record.created_at));
        Ok(records)
    }

    async fn ensure_reserved_files(&self) -> MemoryResult<()> {
        let memory_dir = self.memory_dir();
        fs::create_dir_all(&memory_dir).await?;
        let index = memory_dir.join("index.md");
        if !index.exists() {
            fs::write(
                &index,
                "---\ntype: index\ntitle: Artesian Memory Index\n---\n\n# Artesian Memory Index\n",
            )
            .await?;
        }
        let log = memory_dir.join("log.md");
        if !log.exists() {
            fs::write(
                &log,
                "---\ntype: log\ntitle: Artesian Memory Log\n---\n\n# Log\n",
            )
            .await?;
        }
        Ok(())
    }

    async fn append_update_log(&self, record: &MemoryRecord) -> MemoryResult<()> {
        let mut log = fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(self.memory_dir().join("log.md"))
            .await?;
        log.write_all(
            format!(
                "\n- {} stored memory `{}` node `{}`\n",
                record.created_at.to_rfc3339(),
                record.id,
                record.node_id
            )
            .as_bytes(),
        )
        .await?;
        log.flush().await?;
        Ok(())
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
                .filter(|record| matches_tenancy(record, &query))
                // A tag filter is an explicit selection (e.g. always-inject invariants), so keep
                // tag-matched records even with zero term overlap — relevance-ordered, not dropped.
                .filter_map(|record| score_record(record, &terms, !query.tags.is_empty()))
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
            let collection = self.root.display().to_string();
            let _lane_guard = SessionLaneLock::default_rooted()
                .acquire(&collection, memory.session_id.as_deref())
                .await?;
            let id = stable_memory_id(&memory);
            let existing_path = find_existing_record_path(&self.memory_dir(), &id)?;
            if let Some(path) = existing_path {
                let text = fs::read_to_string(path).await?;
                return parse_record(&text);
            }

            self.ensure_reserved_files().await?;
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
                scope: memory.scope,
                agent_id: memory.agent_id,
                session_id: memory.session_id,
                task_id: memory.task_id,
                user_id: memory.user_id,
            };
            let path = self.record_path(&date_tag, &record.id);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).await?;
            }
            fs::write(path, render_record(&record)?).await?;
            self.append_update_log(&record).await?;
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
    #[serde(default)]
    scope: Option<MemoryScope>,
    #[serde(default)]
    agent_id: Option<String>,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    task_id: Option<String>,
    #[serde(default)]
    user_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OkfHeader {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<MemoryId>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tags: Vec<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    timestamp: Option<DateTime<Utc>>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    node_id: Option<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    tier: Option<MemoryTier>,
    #[serde(default)]
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    metadata: BTreeMap<String, String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    scope: Option<MemoryScope>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    agent_id: Option<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    session_id: Option<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    task_id: Option<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    user_id: Option<String>,
    #[serde(flatten)]
    unknown: BTreeMap<String, serde_yaml::Value>,
}

fn collect_records(dir: &Path, records: &mut Vec<MemoryRecord>) -> MemoryResult<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_records(&path, records)?;
        } else if path.extension().is_some_and(|extension| extension == "md")
            && !is_reserved_okf_file(&path)
        {
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
        scope: record.scope,
        agent_id: record.agent_id.clone(),
        session_id: record.session_id.clone(),
        task_id: record.task_id.clone(),
        user_id: record.user_id.clone(),
    };
    Ok(format!(
        "---\n{}---\n\n{}\n",
        render_okf_header(header),
        record.content
    ))
}

fn render_okf_header(header: FileHeader) -> String {
    let okf = OkfHeader {
        kind: "memory".to_string(),
        id: Some(header.id),
        title: None,
        description: None,
        tags: header.tags,
        timestamp: Some(header.created_at),
        node_id: Some(header.node_id),
        tier: Some(header.tier),
        metadata: header.metadata,
        scope: header.scope,
        agent_id: header.agent_id,
        session_id: header.session_id,
        task_id: header.task_id,
        user_id: header.user_id,
        unknown: BTreeMap::new(),
    };
    serde_yaml::to_string(&okf).expect("OKF header serialization should be infallible")
}

pub(crate) fn parse_record(text: &str) -> MemoryResult<MemoryRecord> {
    if text.starts_with("---\n") {
        return parse_okf_record(text);
    }

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
        scope: header.scope,
        agent_id: header.agent_id,
        session_id: header.session_id,
        task_id: header.task_id,
        user_id: header.user_id,
    })
}

fn parse_okf_record(text: &str) -> MemoryResult<MemoryRecord> {
    let rest = text
        .strip_prefix("---\n")
        .ok_or_else(|| MemoryError::InvalidFile("missing OKF front matter".to_string()))?;
    let (header, body) = rest
        .split_once("\n---\n")
        .ok_or_else(|| MemoryError::InvalidFile("unterminated OKF front matter".to_string()))?;
    let header: OkfHeader = serde_yaml::from_str(header)?;
    let kind = header.kind.to_ascii_lowercase();
    if !matches!(
        kind.as_str(),
        "memory" | "decision" | "runbook" | "reference" | "incident" | "feedback" | "user"
    ) {
        return Err(MemoryError::InvalidFile(format!(
            "unsupported OKF record type: {}",
            header.kind
        )));
    }

    let created_at = header.timestamp.unwrap_or_else(Utc::now);
    let content = body.trim().to_string();
    let mut metadata = header.metadata;
    if kind != "memory" {
        metadata.insert("okf_type".to_string(), kind);
    }
    if let Some(title) = header.title {
        metadata.insert("title".to_string(), title);
    }
    if let Some(description) = header.description {
        metadata.insert("description".to_string(), description);
    }
    for (key, value) in header.unknown {
        if let Some(value) = scalar_to_string(value) {
            metadata.entry(key).or_insert(value);
        }
    }

    let store_memory = StoreMemory {
        content: content.clone(),
        tags: header.tags.clone(),
        metadata: metadata.clone(),
        tier: header.tier.unwrap_or(MemoryTier::L1Atom),
        node_id: header.node_id.clone(),
        created_at: Some(created_at),
        scope: header.scope,
        agent_id: header.agent_id.clone(),
        session_id: header.session_id.clone(),
        task_id: header.task_id.clone(),
        user_id: header.user_id.clone(),
    };
    let id = header.id.unwrap_or_else(|| stable_memory_id(&store_memory));
    let node_id = header.node_id.unwrap_or_else(|| format!("node:{id}"));
    Ok(MemoryRecord {
        id,
        node_id,
        content,
        tags: header.tags,
        metadata,
        tier: store_memory.tier,
        created_at,
        scope: store_memory.scope,
        agent_id: header.agent_id,
        session_id: header.session_id,
        task_id: header.task_id,
        user_id: header.user_id,
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
        } else if path.extension().is_some_and(|extension| extension == "md")
            && !is_reserved_okf_file(&path)
        {
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

fn matches_tenancy(record: &MemoryRecord, query: &MemoryQuery) -> bool {
    query.scope.is_none_or(|scope| record.scope == Some(scope))
        && query
            .agent_id
            .as_ref()
            .is_none_or(|agent_id| record.agent_id.as_ref() == Some(agent_id))
        && query
            .session_id
            .as_ref()
            .is_none_or(|session_id| record.session_id.as_ref() == Some(session_id))
        && query
            .task_id
            .as_ref()
            .is_none_or(|task_id| record.task_id.as_ref() == Some(task_id))
        && query
            .user_id
            .as_ref()
            .is_none_or(|user_id| record.user_id.as_ref() == Some(user_id))
}

fn score_record(record: MemoryRecord, terms: &[String], keep_zero: bool) -> Option<SearchHit> {
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
    (score > 0.0 || keep_zero).then_some(SearchHit {
        record,
        score,
        source: SearchSource::Keyword,
    })
}

fn is_reserved_okf_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| matches!(name, "index.md" | "log.md"))
}

fn scalar_to_string(value: serde_yaml::Value) -> Option<String> {
    match value {
        serde_yaml::Value::String(value) => Some(value),
        serde_yaml::Value::Number(value) => Some(value.to_string()),
        serde_yaml::Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}
