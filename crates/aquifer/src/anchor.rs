// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::{fs, io::AsyncWriteExt};

use crate::{MemoryBackend, MemoryError, MemoryQuery, MemoryResult, SearchHit, SessionKey};

const ANCHOR_START: &str = "<!-- artesian:anchor -->";
const ANCHOR_END: &str = "<!-- /artesian:anchor -->";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionAnchor {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session: Option<SessionKey>,
    pub current_task: String,
    pub plan_pointer: Option<String>,
    pub last_decisions: Vec<String>,
    pub next_step: String,
    pub updated_at: DateTime<Utc>,
}

impl SessionAnchor {
    pub fn new(current_task: impl Into<String>, next_step: impl Into<String>) -> Self {
        Self {
            session: None,
            current_task: current_task.into(),
            plan_pointer: None,
            last_decisions: Vec::new(),
            next_step: next_step.into(),
            updated_at: Utc::now(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AnchorAnchorStore {
    log_path: PathBuf,
}

impl AnchorAnchorStore {
    pub fn new(root: impl AsRef<Path>) -> Self {
        let root = root.as_ref();
        let log_path = if root.join("log.md").exists() {
            root.join("log.md")
        } else {
            root.join("memory").join("log.md")
        };
        Self { log_path }
    }

    pub fn from_log_path(log_path: impl Into<PathBuf>) -> Self {
        Self {
            log_path: log_path.into(),
        }
    }

    pub fn log_path(&self) -> &Path {
        &self.log_path
    }

    pub async fn set(&self, mut anchor: SessionAnchor) -> MemoryResult<SessionAnchor> {
        anchor.session = None;
        self.write(anchor).await
    }

    pub async fn set_for_session(
        &self,
        key: &SessionKey,
        mut anchor: SessionAnchor,
    ) -> MemoryResult<SessionAnchor> {
        anchor.session = (!key.is_default()).then(|| key.clone());
        self.write(anchor).await
    }

    async fn write(&self, mut anchor: SessionAnchor) -> MemoryResult<SessionAnchor> {
        anchor.updated_at = Utc::now();
        if let Some(parent) = self.log_path.parent() {
            fs::create_dir_all(parent).await?;
        }
        if !self.log_path.exists() {
            fs::write(
                &self.log_path,
                "---\ntype: log\ntitle: Artesian Memory Log\n---\n\n# Log\n",
            )
            .await?;
        }
        let mut file = fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&self.log_path)
            .await?;
        let payload = serde_json::to_string_pretty(&anchor)?;
        file.write_all(
            format!("\n{ANCHOR_START}\n```json\n{payload}\n```\n{ANCHOR_END}\n").as_bytes(),
        )
        .await?;
        file.flush().await?;
        Ok(anchor)
    }

    pub async fn get(&self) -> MemoryResult<Option<SessionAnchor>> {
        self.get_for_session(&SessionKey::default_session()).await
    }

    pub async fn get_for_session(&self, key: &SessionKey) -> MemoryResult<Option<SessionAnchor>> {
        if !self.log_path.exists() {
            return Ok(None);
        }
        let text = fs::read_to_string(&self.log_path).await?;
        latest_anchor_for(&text, key)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecoveryContext {
    pub anchor: SessionAnchor,
    pub hits: Vec<SearchHit>,
}

pub async fn recover_after_compaction(
    anchors: &AnchorAnchorStore,
    backend: &dyn MemoryBackend,
    limit: usize,
) -> MemoryResult<Option<RecoveryContext>> {
    let Some(anchor) = anchors.get().await? else {
        return Ok(None);
    };
    let query_text = format!("{} {}", anchor.current_task, anchor.next_step);
    let hits = backend
        .find(MemoryQuery::new(query_text).with_limit(limit))
        .await?;
    Ok(Some(RecoveryContext { anchor, hits }))
}

fn latest_anchor_for(text: &str, key: &SessionKey) -> MemoryResult<Option<SessionAnchor>> {
    let anchors = parse_anchors(text)?;
    Ok(anchors
        .into_iter()
        .rev()
        .find(|anchor| anchor.session.as_ref().cloned().unwrap_or_default() == *key))
}

fn parse_anchors(text: &str) -> MemoryResult<Vec<SessionAnchor>> {
    let mut anchors = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find(ANCHOR_START) {
        let after_start = &rest[start + ANCHOR_START.len()..];
        let Some(end) = after_start.find(ANCHOR_END) else {
            return Err(MemoryError::InvalidFile(
                "unterminated Anchor anchor".to_string(),
            ));
        };
        let block = &after_start[..end];
        anchors.push(parse_anchor_block(block)?);
        rest = &after_start[end + ANCHOR_END.len()..];
    }
    Ok(anchors)
}

fn parse_anchor_block(block: &str) -> MemoryResult<SessionAnchor> {
    let json = block
        .split_once("```json")
        .and_then(|(_, rest)| rest.split_once("```").map(|(json, _)| json.trim()))
        .ok_or_else(|| MemoryError::InvalidFile("missing Anchor anchor JSON".to_string()))?;
    Ok(serde_json::from_str(json)?)
}
