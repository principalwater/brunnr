// SPDX-License-Identifier: Apache-2.0

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::Role;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EventType {
    TaskAnnounced,
    TaskClaimed,
    Ask,
    Result,
    Review,
    Done,
    Verdict,
    Blocked,
    Status,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventSender {
    pub role: Role,
    pub agent_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventEnvelope {
    pub id: String,
    pub correlation_id: String,
    pub timestamp: DateTime<Utc>,
    pub sender: EventSender,
    pub protocol_version: String,
    #[serde(rename = "type")]
    pub event_type: EventType,
    pub payload: Value,
}

impl EventEnvelope {
    pub fn new(
        id: impl Into<String>,
        correlation_id: impl Into<String>,
        sender: EventSender,
        event_type: EventType,
        payload: Value,
    ) -> Self {
        Self {
            id: id.into(),
            correlation_id: correlation_id.into(),
            timestamp: Utc::now(),
            sender,
            protocol_version: "0.1".to_string(),
            event_type,
            payload,
        }
    }
}
