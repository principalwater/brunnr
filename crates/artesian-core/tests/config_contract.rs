// SPDX-License-Identifier: Apache-2.0

use artesian_core::{
    AccConfig, AccLlmConfig, AgentBinding, ArtesianConfig, Role, DEFAULT_RERANK_CANDIDATES,
};

#[test]
fn config_round_trips_through_toml() {
    let mut config = ArtesianConfig::memory_files(
        ".artesian",
        vec![AgentBinding {
            role: Role::Master,
            agent: "claude-code".to_string(),
            model: Some("default".to_string()),
            command: Some("claude".to_string()),
            args: vec!["--print".to_string()],
            timeout_seconds: Some(120),
        }],
    );
    config.acc.budget_tokens = 4096;
    config.acc.min_score = 0.3;
    config.acc.judge = Some(AccLlmConfig {
        provider: "openai".to_string(),
        base_url: Some("http://localhost:11434/v1".to_string()),
        model: Some("llama3".to_string()),
        api_key_env: None,
        command: None,
        args: Vec::new(),
    });

    let encoded = config.to_toml().expect("config should encode");
    let decoded = ArtesianConfig::from_toml(&encoded).expect("config should decode");

    assert_eq!(decoded, config);
    assert_eq!(decoded.acc.budget_tokens, 4096);
    assert_eq!(
        decoded.acc.judge.expect("judge").model.as_deref(),
        Some("llama3")
    );
}

#[test]
fn acc_block_is_optional_and_defaults_apply() {
    let toml = r#"
mode = "memory"

[memory]
backend = "files"
root = ".artesian"
collection = "artesian-memory"

[[agents]]
role = "master"
agent = "claude-code"
"#;
    let config = ArtesianConfig::from_toml(toml).expect("config without [acc] should decode");
    assert_eq!(config.acc, AccConfig::default());
    assert_eq!(config.acc.budget_tokens, 2048);
    assert!(config.acc.compress_on_saturation);
    // The semantic cache also defaults (disabled) when its block is absent.
    assert!(!config.memory.semantic_cache.enabled);
    assert_eq!(config.memory.semantic_cache.capacity, 256);
}

#[test]
fn semantic_cache_config_round_trips() {
    let mut config = ArtesianConfig::memory_files(".artesian", Vec::new());
    config.memory.semantic_cache.enabled = true;
    config.memory.semantic_cache.min_similarity = 0.9;
    config.memory.semantic_cache.ttl_seconds = Some(300);

    let encoded = config.to_toml().expect("encode");
    let decoded = ArtesianConfig::from_toml(&encoded).expect("decode");

    assert_eq!(decoded, config);
    assert!(decoded.memory.semantic_cache.enabled);
    assert_eq!(decoded.memory.semantic_cache.ttl_seconds, Some(300));
}

#[test]
fn neural_rerank_config_defaults_and_round_trips() {
    let toml = r#"
mode = "memory"

[memory]
backend = "sqlite-vec"
root = ".artesian"
collection = "artesian-memory"
rerank = true

[[agents]]
role = "master"
agent = "claude-code"
"#;
    let config = ArtesianConfig::from_toml(toml).expect("config should decode");

    assert!(config.memory.rerank);
    assert_eq!(config.memory.rerank_candidates, 0);
    assert_eq!(
        config.memory.effective_rerank_candidates(),
        DEFAULT_RERANK_CANDIDATES
    );

    let mut config = ArtesianConfig::memory_files(".artesian", Vec::new());
    config.memory.rerank = true;
    config.memory.rerank_candidates = 64;
    let encoded = config.to_toml().expect("encode");
    let decoded = ArtesianConfig::from_toml(&encoded).expect("decode");

    assert_eq!(decoded, config);
    assert!(encoded.contains("rerank = true"));
    assert!(encoded.contains("rerank_candidates = 64"));
    assert_eq!(decoded.memory.effective_rerank_candidates(), 64);
}
