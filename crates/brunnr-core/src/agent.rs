// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;

use futures_core::Stream;
use futures_util::{future::BoxFuture, FutureExt};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::Role;

pub type AgentResult<T> = Result<T, AgentError>;
pub type AgentEventStream = Pin<Box<dyn Stream<Item = AgentResult<AgentEvent>> + Send>>;

/// Pluggable adapter seam for a concrete coding or reasoning agent CLI.
///
/// Implementations own process/session details. The core orchestration layer only relies on this
/// narrow spawn/send/stream/capabilities contract, so new agents do not require core changes.
///
/// ```
/// # use futures_util::{future::BoxFuture, FutureExt, stream};
/// # use brunnr_core::{
/// #     Agent, AgentCapabilities, AgentEvent, AgentEventStream, AgentMessage, AgentResponse,
/// #     AgentModel, AgentResult, AgentSession, Role, SpawnRequest,
/// # };
/// struct EchoAgent;
///
/// impl Agent for EchoAgent {
///     fn spawn(&self, request: SpawnRequest) -> BoxFuture<'_, AgentResult<AgentSession>> {
///         async move {
///             Ok(AgentSession {
///                 id: "session-1".to_string(),
///                 role: request.role,
///                 agent: request.agent,
///             })
///         }
///         .boxed()
///     }
///
///     fn send(
///         &self,
///         _session: &AgentSession,
///         message: AgentMessage,
///     ) -> BoxFuture<'_, AgentResult<AgentResponse>> {
///         async move { Ok(AgentResponse { content: message.content }) }.boxed()
///     }
///
///     fn stream(
///         &self,
///         _session: &AgentSession,
///         message: AgentMessage,
///     ) -> BoxFuture<'_, AgentResult<AgentEventStream>> {
///         async move {
///             Ok(Box::pin(stream::iter([
///                 Ok(AgentEvent::Text(message.content)),
///                 Ok(AgentEvent::Done),
///             ])) as AgentEventStream)
///         }
///         .boxed()
///     }
///
///     fn capabilities(&self) -> AgentCapabilities {
///         AgentCapabilities {
///             streaming: true,
///             tools: false,
///             mcp: true,
///         }
///     }
///
///     fn list_models(&self) -> BoxFuture<'_, AgentResult<Vec<AgentModel>>> {
///         async move {
///             Ok(vec![AgentModel {
///                 id: "echo-small".to_string(),
///                 reachable: true,
///                 source: "static".to_string(),
///             }])
///         }
///         .boxed()
///     }
/// }
///
/// let agent = EchoAgent;
/// let _capabilities = agent.capabilities();
/// ```
pub trait Agent: Send + Sync {
    fn spawn(&self, request: SpawnRequest) -> BoxFuture<'_, AgentResult<AgentSession>>;

    fn send(
        &self,
        session: &AgentSession,
        message: AgentMessage,
    ) -> BoxFuture<'_, AgentResult<AgentResponse>>;

    fn stream(
        &self,
        session: &AgentSession,
        message: AgentMessage,
    ) -> BoxFuture<'_, AgentResult<AgentEventStream>>;

    fn capabilities(&self) -> AgentCapabilities;

    fn list_models(&self) -> BoxFuture<'_, AgentResult<Vec<AgentModel>>> {
        async move { Ok(Vec::new()) }.boxed()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SpawnRequest {
    pub role: Role,
    pub agent: String,
    pub model: Option<String>,
    pub working_dir: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AgentSession {
    pub id: String,
    pub role: Role,
    pub agent: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AgentMessage {
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AgentResponse {
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", content = "payload", rename_all = "kebab-case")]
pub enum AgentEvent {
    Text(String),
    ToolCall { name: String },
    Done,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AgentCapabilities {
    pub streaming: bool,
    pub tools: bool,
    pub mcp: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AgentModel {
    pub id: String,
    pub reachable: bool,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AgentCatalogEntry {
    pub agent: String,
    pub command: Option<String>,
    pub reachable: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unreachable_reason: Option<AgentUnreachableReason>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_checked: Option<String>,
    pub models: Vec<AgentModel>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
pub struct AgentCatalog {
    pub generated_at: Option<String>,
    pub agents: Vec<AgentCatalogEntry>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub roles: Vec<AgentRoleDefinition>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum AgentUnreachableReason {
    NoCommand,
    NoCredentials,
    Quota,
    Network,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AgentRoleDefinition {
    pub name: String,
    pub kind: Role,
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allow_tools: Vec<String>,
    pub source: String,
}

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("agent is unavailable: {0}")]
    Unavailable(String),
    #[error("agent session failed: {0}")]
    Session(String),
}
