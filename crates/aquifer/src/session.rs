// SPDX-License-Identifier: Apache-2.0

use std::{
    cmp::Reverse,
    collections::{btree_map::Entry, BTreeMap},
    sync::Arc,
};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{
    MemoryBackend, MemoryQuery, MemoryRecord, MemoryResult, MemoryScope, MemoryTier, StoreMemory,
};

pub const DEFAULT_SESSION_COMPONENT: &str = "default";
pub const SESSION_RECORD_TAG: &str = "artesian-session";
pub const SESSION_RECORD_SOURCE: &str = "artesian-session";

/// Agent-agnostic address for a resumable session.
///
/// `agent_id` is intentionally absent: producers record themselves in the OCF manifest's
/// `session.handed_off_from`, while consumers resolve by `(user_id, session_id, task_id)`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SessionKey {
    pub user_id: String,
    pub session_id: String,
    pub task_id: String,
}

impl SessionKey {
    pub fn new(
        user_id: Option<String>,
        session_id: Option<String>,
        task_id: Option<String>,
    ) -> Self {
        Self {
            user_id: normalize_component(user_id),
            session_id: normalize_component(session_id),
            task_id: normalize_component(task_id),
        }
    }

    pub fn default_session() -> Self {
        Self::default()
    }

    pub fn is_default(&self) -> bool {
        self.user_id == DEFAULT_SESSION_COMPONENT
            && self.session_id == DEFAULT_SESSION_COMPONENT
            && self.task_id == DEFAULT_SESSION_COMPONENT
    }

    pub fn node_id(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.user_id.as_bytes());
        hasher.update([0]);
        hasher.update(self.session_id.as_bytes());
        hasher.update([0]);
        hasher.update(self.task_id.as_bytes());
        format!("session:{:x}", hasher.finalize())
    }
}

impl Default for SessionKey {
    fn default() -> Self {
        Self {
            user_id: DEFAULT_SESSION_COMPONENT.to_string(),
            session_id: DEFAULT_SESSION_COMPONENT.to_string(),
            task_id: DEFAULT_SESSION_COMPONENT.to_string(),
        }
    }
}

/// Persisted OCF handoff bundle for one [`SessionKey`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Session {
    pub key: SessionKey,
    pub manifest: Value,
    pub schema: Value,
    pub snapshot: Value,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub qualify: Vec<Value>,
    pub updated_at: DateTime<Utc>,
}

impl Session {
    pub fn new(
        key: SessionKey,
        manifest: Value,
        schema: Value,
        snapshot: Value,
        qualify: Vec<Value>,
    ) -> Self {
        Self {
            key,
            manifest,
            schema,
            snapshot,
            qualify,
            updated_at: Utc::now(),
        }
    }

    pub fn handed_off_from(&self) -> Option<String> {
        self.manifest
            .get("session")
            .and_then(|session| session.get("handed_off_from"))
            .and_then(Value::as_str)
            .map(ToString::to_string)
    }

    pub fn entry_count(&self) -> usize {
        self.snapshot
            .get("entries")
            .and_then(Value::as_array)
            .map_or(0, Vec::len)
    }

    pub fn token_count(&self) -> Option<usize> {
        self.snapshot
            .get("token_count")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionListFilter {
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub task_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionSummary {
    pub key: SessionKey,
    pub updated_at: DateTime<Utc>,
    pub handed_off_from: Option<String>,
    pub entry_count: usize,
    pub token_count: Option<usize>,
}

impl From<&Session> for SessionSummary {
    fn from(session: &Session) -> Self {
        Self {
            key: session.key.clone(),
            updated_at: session.updated_at,
            handed_off_from: session.handed_off_from(),
            entry_count: session.entry_count(),
            token_count: session.token_count(),
        }
    }
}

#[derive(Clone)]
pub struct SessionStore {
    backend: Arc<dyn MemoryBackend>,
}

impl SessionStore {
    pub fn new(backend: Arc<dyn MemoryBackend>) -> Self {
        Self { backend }
    }

    pub async fn store(&self, mut session: Session) -> MemoryResult<SessionSummary> {
        session.updated_at = Utc::now();
        let key = session.key.clone();
        let content = serde_json::to_string(&session)?;
        let mut metadata = BTreeMap::new();
        metadata.insert("record_type".to_string(), SESSION_RECORD_TAG.to_string());
        self.backend
            .store(StoreMemory {
                content,
                tags: vec![SESSION_RECORD_TAG.to_string()],
                metadata,
                tier: MemoryTier::L2Scenario,
                node_id: Some(key.node_id()),
                created_at: Some(session.updated_at),
                scope: Some(MemoryScope::Session),
                agent_id: None,
                session_id: Some(key.session_id.clone()),
                task_id: Some(key.task_id.clone()),
                user_id: Some(key.user_id.clone()),
                source: Some(SESSION_RECORD_SOURCE.to_string()),
                confidence: Some(1.0),
            })
            .await?;
        Ok(SessionSummary::from(&session))
    }

    pub async fn load(&self, key: &SessionKey) -> MemoryResult<Option<Session>> {
        let mut sessions = self
            .find_sessions(SessionListFilter {
                user_id: Some(key.user_id.clone()),
                session_id: Some(key.session_id.clone()),
                task_id: Some(key.task_id.clone()),
            })
            .await?;
        if let Some(session) = sessions.drain(..).find(|candidate| candidate.key == *key) {
            return Ok(Some(session));
        }

        let Some(record) = self.backend.get_node(&key.node_id()).await? else {
            return Ok(None);
        };
        let session = session_from_record(&record)?;
        Ok((session.key == *key).then_some(session))
    }

    pub async fn list(&self, filter: SessionListFilter) -> MemoryResult<Vec<SessionSummary>> {
        let sessions = self.find_sessions(filter).await?;
        Ok(sessions.iter().map(SessionSummary::from).collect())
    }

    async fn find_sessions(&self, filter: SessionListFilter) -> MemoryResult<Vec<Session>> {
        let mut query = MemoryQuery::new(SESSION_RECORD_TAG).with_limit(1000);
        query.tags = vec![SESSION_RECORD_TAG.to_string()];
        query.scope = Some(MemoryScope::Session);
        query.user_id = normalize_filter_component(filter.user_id.clone());
        query.session_id = normalize_filter_component(filter.session_id.clone());
        query.task_id = normalize_filter_component(filter.task_id.clone());

        let mut hits = self.backend.find(query).await?;
        hits.sort_by_key(|hit| Reverse(hit.record.created_at));

        let mut latest_by_key = BTreeMap::new();
        for hit in hits {
            let session = session_from_record(&hit.record)?;
            if !matches_filter(&session.key, &filter) {
                continue;
            }
            match latest_by_key.entry(session.key.clone()) {
                Entry::Vacant(entry) => {
                    entry.insert(session);
                }
                Entry::Occupied(_) => {}
            }
        }

        Ok(latest_by_key.into_values().collect())
    }
}

fn session_from_record(record: &MemoryRecord) -> MemoryResult<Session> {
    serde_json::from_str(&record.content).map_err(Into::into)
}

fn normalize_component(value: Option<String>) -> String {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_SESSION_COMPONENT.to_string())
}

fn normalize_filter_component(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn matches_filter(key: &SessionKey, filter: &SessionListFilter) -> bool {
    filter
        .user_id
        .as_ref()
        .is_none_or(|user_id| key.user_id == normalize_component(Some(user_id.clone())))
        && filter.session_id.as_ref().is_none_or(|session_id| {
            key.session_id == normalize_component(Some(session_id.clone()))
        })
        && filter
            .task_id
            .as_ref()
            .is_none_or(|task_id| key.task_id == normalize_component(Some(task_id.clone())))
}
