// SPDX-License-Identifier: Apache-2.0

//! Core role, agent-adapter, queue, and configuration seams for Artesian.

mod agent;
mod config;
mod coordination;
mod event;
mod roles;

pub use agent::{
    Agent, AgentCapabilities, AgentCatalog, AgentCatalogEntry, AgentError, AgentEvent,
    AgentEventStream, AgentMessage, AgentModel, AgentResponse, AgentResult, AgentRoleDefinition,
    AgentSession, AgentUnreachableReason, SpawnRequest,
};
pub use config::{
    AccConfig, AccLlmConfig, AgentBinding, ArtesianConfig, CoordinationConfig, MemoryBackendKind,
    MemoryConfig, Mode, ResourceQuotaConfig, SemanticCacheConfig, VerifierCommandConfig,
    DEFAULT_RERANK_CANDIDATES,
};
pub use coordination::{Barrier, ResourceQuota, TokenAccounting};
pub use event::{EventEnvelope, EventSender, EventType};
pub use roles::{CompletedJob, Job, JobStatus, Queue, Role, RoleParseError};
