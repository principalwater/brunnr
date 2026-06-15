// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

use crate::Role;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Mode {
    Memory,
    Orchestrate,
    Full,
    Advanced,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MemoryBackendKind {
    Files,
    SqliteVec,
    Qdrant,
    TencentDb,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub backend: MemoryBackendKind,
    pub root: String,
    #[serde(default = "default_memory_collection")]
    pub collection: String,
    #[serde(default)]
    pub qdrant_url: Option<String>,
    #[serde(default)]
    pub qdrant_rest_url: Option<String>,
    #[serde(default)]
    pub qdrant_api_key_env: Option<String>,
    #[serde(default = "default_local_rerank_enabled")]
    pub local_rerank_enabled: bool,
    #[serde(default)]
    pub hyde_enabled: bool,
    #[serde(default)]
    pub multi_query_enabled: bool,
    #[serde(default)]
    pub debate_enabled: bool,
    #[serde(default)]
    pub llm_consolidation_enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentBinding {
    pub role: Role,
    pub agent: String,
    pub model: Option<String>,
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub timeout_seconds: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct CoordinationConfig {
    #[serde(default)]
    pub router_enabled: bool,
    #[serde(default)]
    pub quotas: Vec<ResourceQuotaConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub concurrency_limit: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_retries: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_backoff_millis: Option<u64>,
    #[serde(default)]
    pub verifiers: Vec<VerifierCommandConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topology: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spawn_registry_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_concurrent_spawns: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spawn_max_lifetime_seconds: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spawn_shutdown_grace_millis: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ResourceQuotaConfig {
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub user_id: Option<String>,
    #[serde(default)]
    pub max_prompt_tokens: Option<u64>,
    #[serde(default)]
    pub max_requests_per_minute: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct VerifierCommandConfig {
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BrunnrConfig {
    pub mode: Mode,
    pub memory: MemoryConfig,
    pub agents: Vec<AgentBinding>,
    #[serde(default)]
    pub coordination: CoordinationConfig,
}

impl BrunnrConfig {
    pub fn memory_files(root: impl Into<String>, agents: Vec<AgentBinding>) -> Self {
        Self {
            mode: Mode::Memory,
            memory: MemoryConfig {
                backend: MemoryBackendKind::Files,
                root: root.into(),
                collection: default_memory_collection(),
                qdrant_url: None,
                qdrant_rest_url: None,
                qdrant_api_key_env: None,
                local_rerank_enabled: default_local_rerank_enabled(),
                hyde_enabled: false,
                multi_query_enabled: false,
                debate_enabled: false,
                llm_consolidation_enabled: false,
            },
            agents,
            coordination: CoordinationConfig::default(),
        }
    }

    pub fn from_toml(input: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(input)
    }

    pub fn to_toml(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }
}

fn default_memory_collection() -> String {
    "brunnr-memory".to_string()
}

fn default_local_rerank_enabled() -> bool {
    true
}
