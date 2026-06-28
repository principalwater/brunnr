// SPDX-License-Identifier: Apache-2.0

use std::{env, fs, path::PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::Role;

pub const DEFAULT_RERANK_CANDIDATES: usize = 50;

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    #[serde(default)]
    pub qdrant_url: Option<String>,
    #[serde(default)]
    pub qdrant_rest_url: Option<String>,
    #[serde(default)]
    pub qdrant_api_key_env: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qdrant_api_key_file: Option<String>,
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
    /// Enable the neural cross-encoder reranker for vector-backed retrieval.
    ///
    /// Disabled by default because the model has load and inference cost.
    #[serde(default)]
    pub rerank: bool,
    /// Candidate pool fused before neural reranking. `0` means "use the default pool" when
    /// `rerank = true`, and no reranking when `rerank = false`.
    #[serde(default)]
    pub rerank_candidates: usize,
    /// Semantic query cache over a vector backend (no effect on the files backend).
    #[serde(default)]
    pub semantic_cache: SemanticCacheConfig,
    /// When `true` (the default), `find` bumps `access_count` and `last_access` on returned
    /// records (best-effort writeback). Set to `false` to reduce write amplification at the
    /// cost of losing reinforcement signals for dreams/decay.
    #[serde(default = "default_track_access")]
    pub track_access: bool,
    /// When `true` (the default), targeted recalls append one JSON line to
    /// `~/.artesian/token_savings.jsonl` and update `~/.artesian/token_savings.json` with
    /// cumulative totals.  Path overridable via `ARTESIAN_STATS_DIR`.  Set to `false` to
    /// disable stats collection entirely; failures are always silent and non-blocking.
    #[serde(default = "default_track_savings")]
    pub track_savings: bool,
}

impl MemoryConfig {
    pub fn effective_rerank_candidates(&self) -> usize {
        if self.rerank {
            match self.rerank_candidates {
                0 => DEFAULT_RERANK_CANDIDATES,
                candidates => candidates,
            }
        } else {
            0
        }
    }

    /// Resolve the configured Qdrant API key without logging or exposing the secret.
    ///
    /// Precedence is:
    /// 1. the environment variable named by `qdrant_api_key_env`, when present and non-empty;
    /// 2. the file named by `qdrant_api_key_file`, supporting either a plain key file or an
    ///    env-style file containing the configured variable name.
    pub fn resolve_qdrant_api_key(&self) -> Option<String> {
        resolve_qdrant_api_key_from(
            self.qdrant_api_key_env.as_deref(),
            self.qdrant_api_key_file.as_deref(),
            |name| env::var(name).ok(),
            env::home_dir,
        )
    }
}

fn resolve_qdrant_api_key_from<E, H>(
    env_name: Option<&str>,
    file: Option<&str>,
    getenv: E,
    home_dir: H,
) -> Option<String>
where
    E: Fn(&str) -> Option<String>,
    H: Fn() -> Option<PathBuf>,
{
    let env_name = env_name.and_then(non_empty_trimmed);
    if let Some(env_name) = env_name {
        if let Some(value) = getenv(env_name).filter(|value| !value.is_empty()) {
            return Some(value);
        }
    }

    let path = file
        .and_then(non_empty_trimmed)
        .and_then(|path| expand_qdrant_api_key_path(path, home_dir))?;
    let contents = fs::read_to_string(path).ok()?;
    qdrant_api_key_from_file_contents(&contents, env_name)
}

fn non_empty_trimmed(value: &str) -> Option<&str> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then_some(trimmed)
}

fn expand_qdrant_api_key_path<H>(path: &str, home_dir: H) -> Option<PathBuf>
where
    H: Fn() -> Option<PathBuf>,
{
    if path == "~" || path == "$HOME" {
        return home_dir();
    }
    if let Some(rest) = path.strip_prefix("~/") {
        return home_dir().map(|home| home.join(rest));
    }
    if let Some(rest) = path.strip_prefix("$HOME/") {
        return home_dir().map(|home| home.join(rest));
    }
    Some(PathBuf::from(path))
}

fn qdrant_api_key_from_file_contents(contents: &str, env_name: Option<&str>) -> Option<String> {
    if let Some(env_name) = env_name {
        for line in contents.lines() {
            if let Some((name, value)) = parse_env_assignment(line) {
                if name == env_name {
                    return clean_file_api_key_value(value);
                }
            }
        }
    }

    let mut non_empty_lines = contents
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'));
    let first = non_empty_lines.next()?;
    if non_empty_lines.next().is_some() {
        return None;
    }
    if let Some((_, value)) = parse_env_assignment(first) {
        if env_name.is_none() {
            clean_file_api_key_value(value)
        } else {
            None
        }
    } else {
        clean_file_api_key_value(first)
    }
}

fn parse_env_assignment(line: &str) -> Option<(&str, &str)> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return None;
    }
    let assignment = if let Some(rest) = trimmed.strip_prefix("export") {
        if rest.chars().next().is_some_and(char::is_whitespace) {
            rest.trim_start()
        } else {
            trimmed
        }
    } else {
        trimmed
    };
    let (name, value) = assignment.split_once('=')?;
    let name = name.trim();
    (!name.is_empty()).then_some((name, value))
}

fn clean_file_api_key_value(value: &str) -> Option<String> {
    let trimmed = value.trim();
    let unquoted = trimmed
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .or_else(|| {
            trimmed
                .strip_prefix('\'')
                .and_then(|value| value.strip_suffix('\''))
        })
        .unwrap_or(trimmed)
        .trim();
    (!unquoted.is_empty()).then(|| unquoted.to_string())
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
    /// When `true`, the Claude `PreCompact` hook spawns an offline dream consolidation
    /// pass **after** the synchronous session checkpoint completes and returns.  The
    /// dream runs fully detached (fire-and-forget) so it never delays the compaction or
    /// the hook's prompt return.  Default `false` because the dream is heavy and may
    /// consume LLM tokens.
    ///
    /// Can also be enabled per-invocation via the `ARTESIAN_DREAM_ON_COMPACT=1` env var
    /// without changing the config file.
    #[serde(default)]
    pub dream_on_compact: bool,
}

impl ArtesianConfig {
    pub fn memory_files(root: impl Into<String>, agents: Vec<AgentBinding>) -> Self {
        Self {
            mode: Mode::Memory,
            memory: MemoryConfig {
                backend: MemoryBackendKind::Files,
                root: root.into(),
                collection: default_memory_collection(),
                project: None,
                qdrant_url: None,
                qdrant_rest_url: None,
                qdrant_api_key_env: None,
                qdrant_api_key_file: None,
                local_rerank_enabled: default_local_rerank_enabled(),
                hyde_enabled: false,
                multi_query_enabled: false,
                debate_enabled: false,
                llm_consolidation_enabled: false,
                rerank: false,
                rerank_candidates: 0,
                semantic_cache: SemanticCacheConfig::default(),
                track_access: default_track_access(),
                track_savings: default_track_savings(),
            },
            agents,
            coordination: CoordinationConfig::default(),
            acc: AccConfig::default(),
            dream_on_compact: false,
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

fn default_track_access() -> bool {
    true
}

fn default_track_savings() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::resolve_qdrant_api_key_from;
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    #[test]
    fn qdrant_api_key_reads_plain_file_with_tilde_and_trims_quotes() {
        let home = temp_dir("artesian-core-qdrant-home");
        fs::create_dir_all(&home).expect("create home");
        fs::write(home.join("plain.key"), "  \"plain-file-key\" \n").expect("write key");

        let key = resolve_qdrant_api_key_from(
            Some("QDRANT__SERVICE__API_KEY"),
            Some("~/plain.key"),
            |_| None,
            || Some(home.clone()),
        );

        assert_eq!(key.as_deref(), Some("plain-file-key"));
        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn qdrant_api_key_reads_env_style_file_by_configured_name_with_home() {
        let home = temp_dir("artesian-core-qdrant-env-home");
        fs::create_dir_all(home.join(".macray")).expect("create key dir");
        fs::write(
            home.join(".macray").join("qdrant.env"),
            "OTHER_KEY=ignored\nexport QDRANT__SERVICE__API_KEY='env-file-key'\n",
        )
        .expect("write env file");

        let key = resolve_qdrant_api_key_from(
            Some("QDRANT__SERVICE__API_KEY"),
            Some("$HOME/.macray/qdrant.env"),
            |_| None,
            || Some(home.clone()),
        );

        assert_eq!(key.as_deref(), Some("env-file-key"));
        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn qdrant_api_key_env_var_wins_over_file() {
        let home = temp_dir("artesian-core-qdrant-precedence-home");
        fs::create_dir_all(&home).expect("create home");
        fs::write(home.join("plain.key"), "file-key\n").expect("write key");

        let key = resolve_qdrant_api_key_from(
            Some("QDRANT__SERVICE__API_KEY"),
            Some("~/plain.key"),
            |name| (name == "QDRANT__SERVICE__API_KEY").then(|| "env-key".to_string()),
            || Some(home.clone()),
        );

        assert_eq!(key.as_deref(), Some("env-key"));
        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn qdrant_api_key_missing_file_or_absent_key_resolves_to_none() {
        let home = temp_dir("artesian-core-qdrant-missing-home");
        fs::create_dir_all(&home).expect("create home");
        fs::write(home.join("qdrant.env"), "OTHER_KEY=ignored\n").expect("write env file");

        let missing = resolve_qdrant_api_key_from(
            Some("QDRANT__SERVICE__API_KEY"),
            Some("~/missing.env"),
            |_| None,
            || Some(home.clone()),
        );
        let absent = resolve_qdrant_api_key_from(
            Some("QDRANT__SERVICE__API_KEY"),
            Some("~/qdrant.env"),
            |_| None,
            || Some(home.clone()),
        );

        assert_eq!(missing, None);
        assert_eq!(absent, None);
        let _ = fs::remove_dir_all(&home);
    }

    fn temp_dir(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("{name}-{}-{unique}", std::process::id()))
    }
}
