// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::Relation;

pub type MemoryResult<T> = Result<T, MemoryError>;

#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to encode metadata: {0}")]
    Encode(#[from] toml::ser::Error),
    #[error("failed to decode metadata: {0}")]
    Decode(#[from] toml::de::Error),
    #[error("failed to decode OKF front matter: {0}")]
    YamlDecode(#[from] serde_yaml::Error),
    #[error("failed to convert memory payload: {0}")]
    Payload(#[from] serde_json::Error),
    #[error("invalid memory file: {0}")]
    InvalidFile(String),
    #[error("confidence must be within 0.0..=1.0, got {0}")]
    InvalidConfidence(f32),
    #[error("database error: {0}")]
    Database(String),
    #[error("backend is not available in this build: {0}")]
    BackendUnavailable(String),
    #[error("collection embedding metadata mismatch: collection={collection_model}/{collection_dimensions}, configured={configured_model}/{configured_dimensions}; run artesian migrate before reading or writing")]
    CompatMismatch {
        collection_model: String,
        collection_dimensions: usize,
        configured_model: String,
        configured_dimensions: usize,
    },
    #[error("timed out acquiring session lane lock for {lane} after {timeout_millis}ms")]
    LaneLockTimeout { lane: String, timeout_millis: u128 },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct MemoryId(String);

impl MemoryId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for MemoryId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MemoryTier {
    L0Raw,
    L1Atom,
    L2Scenario,
    L3Project,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MemoryScope {
    Shared,
    Agent,
    Session,
    Task,
}

impl MemoryScope {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Shared => "shared",
            Self::Agent => "agent",
            Self::Session => "session",
            Self::Task => "task",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: MemoryId,
    pub node_id: String,
    pub content: String,
    pub tags: Vec<String>,
    pub metadata: BTreeMap<String, String>,
    pub tier: MemoryTier,
    pub created_at: DateTime<Utc>,
    #[serde(default)]
    pub scope: Option<MemoryScope>,
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub task_id: Option<String>,
    #[serde(default)]
    pub user_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relations: Vec<Relation>,
}

impl Eq for MemoryRecord {}

impl MemoryRecord {
    pub fn new(
        id: MemoryId,
        node_id: impl Into<String>,
        content: impl Into<String>,
        tags: Vec<String>,
        metadata: BTreeMap<String, String>,
        tier: MemoryTier,
    ) -> Self {
        Self {
            id,
            node_id: node_id.into(),
            content: content.into(),
            tags,
            metadata,
            tier,
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
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoreMemory {
    pub content: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
    pub tier: MemoryTier,
    #[serde(default)]
    pub node_id: Option<String>,
    #[serde(default)]
    pub created_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub scope: Option<MemoryScope>,
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub task_id: Option<String>,
    #[serde(default)]
    pub user_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relations: Vec<Relation>,
}

impl Eq for StoreMemory {}

impl StoreMemory {
    pub fn atom(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            tags: Vec::new(),
            metadata: BTreeMap::new(),
            tier: MemoryTier::L1Atom,
            node_id: None,
            created_at: None,
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

    pub fn validate_confidence(&self) -> MemoryResult<()> {
        if let Some(confidence) = self.confidence {
            if !(0.0..=1.0).contains(&confidence) {
                return Err(MemoryError::InvalidConfidence(confidence));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryQuery {
    pub text: String,
    pub limit: usize,
    pub tags: Vec<String>,
    pub node_id: Option<String>,
    #[serde(default)]
    pub scope: Option<MemoryScope>,
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub task_id: Option<String>,
    #[serde(default)]
    pub user_id: Option<String>,
}

impl MemoryQuery {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            limit: 10,
            tags: Vec::new(),
            node_id: None,
            scope: None,
            agent_id: None,
            session_id: None,
            task_id: None,
            user_id: None,
        }
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SearchSource {
    Keyword,
    Vector,
    Hybrid,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchHit {
    pub record: MemoryRecord,
    pub score: f32,
    pub source: SearchSource,
}

impl SearchHit {
    pub fn keyword(record: MemoryRecord, score: f32) -> Self {
        Self {
            record,
            score,
            source: SearchSource::Keyword,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RrfOptions {
    pub rank_constant: f32,
    pub limit: usize,
}

impl Default for RrfOptions {
    fn default() -> Self {
        Self {
            rank_constant: 60.0,
            limit: 10,
        }
    }
}
