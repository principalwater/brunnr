// SPDX-License-Identifier: Apache-2.0

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::Role;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum Mode {
    Memory,
    Orchestrate,
    Full,
    Advanced,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum MemoryBackendKind {
    Files,
    SqliteVec,
    Qdrant,
    TencentDb,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
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
    /// Semantic query cache over a vector backend (no effect on the files backend).
    #[serde(default)]
    pub semantic_cache: SemanticCacheConfig,
}

/// Settings for the semantic query cache (see `aquifer::SemanticCache`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SemanticCacheConfig {
    /// Enable caching of vector-backend `find` results by query-embedding similarity.
    #[serde(default)]
    pub enabled: bool,
    /// Maximum cached queries (LRU eviction).
    #[serde(default = "default_cache_capacity")]
    pub capacity: usize,
    /// Cosine threshold above which a prior query counts as a cache hit.
    #[serde(default = "default_cache_min_similarity")]
    pub min_similarity: f32,
    /// Optional time-to-live for cache entries, in seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_seconds: Option<u64>,
}

impl Default for SemanticCacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            capacity: default_cache_capacity(),
            min_similarity: default_cache_min_similarity(),
            ttl_seconds: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
pub struct VerifierCommandConfig {
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
}

/// ACC (Agent Cognitive Compressor) control-plane settings, read by the CLI `memory commit`
/// command and the MCP `memory.commit` tool. All fields have sensible defaults, so the block
/// is optional in `artesian.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AccConfig {
    /// Token budget for the committed state (the saturation bound).
    #[serde(default = "default_acc_budget_tokens")]
    pub budget_tokens: usize,
    /// How many recall candidates to pull per cycle.
    #[serde(default = "default_acc_recall_limit")]
    pub recall_limit: usize,
    /// Minimum candidate score to qualify (recall-store-relative scale).
    #[serde(default = "default_acc_min_score")]
    pub min_score: f32,
    /// Token-overlap at or above which a candidate is rejected as redundant.
    #[serde(default = "default_acc_redundancy_threshold")]
    pub redundancy_threshold: f32,
    /// Compress an admitted candidate to fit remaining headroom instead of rejecting it.
    #[serde(default = "default_acc_compress_on_saturation")]
    pub compress_on_saturation: bool,
    /// Optional LLM judge-eval gate (drift / hallucination scoring). Requires a build with the
    /// `llm` feature; otherwise the deterministic gate is used and this is ignored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge: Option<AccLlmConfig>,
    /// Optional LLM compressor. Requires the `llm` feature; otherwise the extractive
    /// compressor is used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compressor: Option<AccLlmConfig>,
}

impl Default for AccConfig {
    fn default() -> Self {
        Self {
            budget_tokens: default_acc_budget_tokens(),
            recall_limit: default_acc_recall_limit(),
            min_score: default_acc_min_score(),
            redundancy_threshold: default_acc_redundancy_threshold(),
            compress_on_saturation: default_acc_compress_on_saturation(),
            judge: None,
            compressor: None,
        }
    }
}

/// LLM endpoint config for the ACC judge gate or compressor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AccLlmConfig {
    /// `openai` (OpenAI-compatible `/chat/completions`) or `command` (agent CLI subprocess).
    pub provider: String,
    /// API root including the version segment, e.g. `http://localhost:11434/v1` (Ollama).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Name of an environment variable holding the bearer API key (the key is never stored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    /// For `provider = "command"`: the executable to run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    /// For `provider = "command"`: its arguments ({prompt}/{system} placeholders supported).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ArtesianConfig {
    pub mode: Mode,
    pub memory: MemoryConfig,
    pub agents: Vec<AgentBinding>,
    #[serde(default)]
    pub coordination: CoordinationConfig,
    #[serde(default)]
    pub acc: AccConfig,
}

impl ArtesianConfig {
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
                semantic_cache: SemanticCacheConfig::default(),
            },
            agents,
            coordination: CoordinationConfig::default(),
            acc: AccConfig::default(),
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
    "artesian-memory".to_string()
}

fn default_local_rerank_enabled() -> bool {
    true
}

fn default_acc_budget_tokens() -> usize {
    2048
}

fn default_acc_recall_limit() -> usize {
    16
}

fn default_acc_min_score() -> f32 {
    0.2
}

fn default_acc_redundancy_threshold() -> f32 {
    0.8
}

fn default_acc_compress_on_saturation() -> bool {
    true
}

fn default_cache_capacity() -> usize {
    256
}

fn default_cache_min_similarity() -> f32 {
    0.95
}
