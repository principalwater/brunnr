// SPDX-License-Identifier: Apache-2.0

//! Core role, agent-adapter, queue, and configuration seams for Brunnr.

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
    AgentBinding, BrunnrConfig, CoordinationConfig, MemoryBackendKind, MemoryConfig, Mode,
    ResourceQuotaConfig, VerifierCommandConfig,
};
pub use coordination::{Barrier, ResourceQuota, TokenAccounting};
pub use event::{EventEnvelope, EventSender, EventType};
pub use roles::{Erindi, ErindiStatus, Galdr, Role, RoleParseError, Thing};
