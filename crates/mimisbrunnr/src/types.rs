// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub type MemoryResult<T> = Result<T, MemoryError>;

#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to encode metadata: {0}")]
    Encode(#[from] toml::ser::Error),
    #[error("failed to decode metadata: {0}")]
    Decode(#[from] toml::de::Error),
    #[error("failed to convert memory payload: {0}")]
    Payload(#[from] serde_json::Error),
    #[error("invalid memory file: {0}")]
    InvalidFile(String),
    #[error("database error: {0}")]
    Database(String),
    #[error("backend is not available in this build: {0}")]
    BackendUnavailable(String),
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: MemoryId,
    pub node_id: String,
    pub content: String,
    pub tags: Vec<String>,
    pub metadata: BTreeMap<String, String>,
    pub tier: MemoryTier,
    pub created_at: DateTime<Utc>,
}

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
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
}

impl StoreMemory {
    pub fn atom(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            tags: Vec::new(),
            metadata: BTreeMap::new(),
            tier: MemoryTier::L1Atom,
            node_id: None,
            created_at: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryQuery {
    pub text: String,
    pub limit: usize,
    pub tags: Vec<String>,
    pub node_id: Option<String>,
}

impl MemoryQuery {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            limit: 10,
            tags: Vec::new(),
            node_id: None,
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
