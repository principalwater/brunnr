// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use aquifer::{FilesBackend, MemoryBackend, StoreMemory};
use artesian_test_support::TempDir;
use headgate::{Headgate, HeadgateConfig, MemoryRecallStore, RecallItem, StaticRecallStore};

#[tokio::test]
async fn cycle_admits_qualifying_memories_from_files_backend() {
    let tempdir = TempDir::new("headgate-files");
    let backend = FilesBackend::new(tempdir.path());
    for content in [
        "the team chose Rust for the core crates",
        "the deployment runs nightly on the kubernetes cluster",
        "the team chose Rust for the core crates", // duplicate content
    ] {
        backend
            .store(StoreMemory::atom(content))
            .await
            .expect("store should succeed");
    }

    let recall = Arc::new(MemoryRecallStore::new(backend));
    let mut headgate = Headgate::new(recall, HeadgateConfig::default());
    let metrics = headgate
        .cycle("team rust deployment")
        .await
        .expect("cycle should succeed");

    assert!(metrics.candidates >= 2, "both distinct memories recalled");
    assert!(metrics.admitted >= 1, "at least one memory committed");
    assert!(
        metrics.footprint_tokens <= metrics.budget_tokens,
        "footprint stays within budget"
    );
    let rendered = headgate.render();
    assert!(
        rendered.contains("Rust") || rendered.contains("deployment"),
        "committed context surfaces recalled knowledge: {rendered:?}"
    );
}

#[tokio::test]
async fn second_cycle_does_not_recommit_the_same_knowledge() {
    let recall = Arc::new(StaticRecallStore::new(vec![
        RecallItem::new("n1", "the team chose Rust for the core crates", 1.0),
        RecallItem::new("n2", "deployments run nightly on kubernetes", 1.0),
    ]));
    let mut headgate = Headgate::new(recall, HeadgateConfig::default());

    let first = headgate.cycle("query").await.expect("first cycle");
    assert_eq!(first.admitted, 2);

    let second = headgate.cycle("query").await.expect("second cycle");
    assert_eq!(
        second.admitted, 0,
        "already-committed knowledge is not re-admitted"
    );
    assert_eq!(second.rejected_redundant, 2);
    assert_eq!(headgate.ccs().len(), 2);
}

#[tokio::test]
async fn tiny_budget_bounds_footprint_via_eviction() {
    let recall = Arc::new(StaticRecallStore::new(vec![
        RecallItem::new("low", "alpha beta gamma delta epsilon zeta eta theta", 0.3),
        RecallItem::new(
            "high",
            "one two three four five six seven eight nine ten",
            0.9,
        ),
    ]));
    // Budget large enough for one of the two entries but not both, compression off.
    let config = HeadgateConfig {
        budget_tokens: 12,
        compress_on_saturation: false,
        ..HeadgateConfig::default()
    };
    let mut headgate = Headgate::new(recall, config);
    let metrics = headgate.cycle("query").await.expect("cycle");

    assert!(
        headgate.ccs().token_count() <= 12,
        "committed footprint never exceeds the budget"
    );
    // The higher-scored entry should win a slot, evicting or excluding the lower one.
    assert!(metrics.admitted >= 1);
    assert!(metrics.evicted + metrics.rejected_saturated >= 1);
}

#[tokio::test]
async fn compression_fits_an_oversized_candidate() {
    let long = "First clause here. Second clause here. Third clause here. \
                Fourth clause here. Fifth clause here. Sixth clause here.";
    let recall = Arc::new(StaticRecallStore::new(vec![RecallItem::new(
        "big", long, 0.9,
    )]));
    let config = HeadgateConfig {
        budget_tokens: 10,
        compress_on_saturation: true,
        ..HeadgateConfig::default()
    };
    let mut headgate = Headgate::new(recall, config);
    let metrics = headgate.cycle("query").await.expect("cycle");

    assert_eq!(metrics.compressed, 1);
    assert_eq!(metrics.admitted, 1);
    assert!(headgate.ccs().token_count() <= 10);
}

#[cfg(feature = "llm")]
#[tokio::test]
async fn judge_gate_drives_a_cycle_and_rejects_drift() {
    use headgate::{JudgeQualifyGate, StaticLlmClient};

    // A judge that flags every candidate as high-drift rejects them all.
    let recall = Arc::new(StaticRecallStore::new(vec![
        RecallItem::new("n1", "the team chose Rust", 1.0),
        RecallItem::new("n2", "the team chose Go", 1.0),
    ]));
    let client = Arc::new(StaticLlmClient::new(
        "{\"relevance\":0.9,\"novelty\":0.9,\"drift\":0.9,\"reason\":\"contradiction\"}",
    ));
    let gate = Arc::new(JudgeQualifyGate::new(client));
    let mut headgate = Headgate::new(recall, HeadgateConfig::default()).with_gate(gate);

    let metrics = headgate.cycle("which language").await.expect("cycle");
    assert_eq!(
        metrics.admitted, 0,
        "high-drift candidates are rejected by the judge"
    );
    assert!(headgate.ccs().is_empty());
}

#[cfg(feature = "llm")]
#[tokio::test]
async fn judge_gate_admits_clean_candidate() {
    use headgate::{JudgeQualifyGate, StaticLlmClient};

    let recall = Arc::new(StaticRecallStore::new(vec![RecallItem::new(
        "n1",
        "the team chose Rust for the core crates",
        1.0,
    )]));
    let client = Arc::new(StaticLlmClient::new(
        "{\"relevance\":0.95,\"novelty\":0.9,\"drift\":0.05,\"slot\":\"decision\",\"reason\":\"ok\"}",
    ));
    let gate = Arc::new(JudgeQualifyGate::new(client));
    let mut headgate = Headgate::new(recall, HeadgateConfig::default()).with_gate(gate);

    let metrics = headgate.cycle("language choice").await.expect("cycle");
    assert_eq!(metrics.admitted, 1);
    assert!(headgate.render().contains("chose Rust"));
}

fn make_temp_dir(tag: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("{tag}-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

/// Verify that qualify.reject records a savings entry with returned=0 and saved=baseline.
/// Uses `with_savings_dir` to redirect I/O to a controlled temp directory, avoiding any
/// reliance on ARTESIAN_STATS_DIR env var (which is not safe to mutate in parallel tests).
#[tokio::test]
async fn qualify_reject_records_savings_entry() {
    let stats_dir = make_temp_dir("artesian-qualify-reject");

    // All candidates score below min_score=0.5 → all rejected_relevance.
    let recall = Arc::new(StaticRecallStore::new(vec![
        RecallItem::new("n1", "the team chose Rust for the core crates", 0.1),
        RecallItem::new("n2", "deployment runs on kubernetes every night", 0.1),
    ]));
    let config = HeadgateConfig {
        min_score: 0.5,
        ..HeadgateConfig::default()
    };
    let mut headgate = Headgate::new(recall, config)
        .with_savings("test-col", true)
        .with_savings_dir(stats_dir.clone());
    let metrics = headgate.cycle("rust kubernetes").await.expect("cycle");

    assert_eq!(
        metrics.rejected_relevance, 2,
        "both candidates rejected as below threshold"
    );
    assert_eq!(metrics.admitted, 0);

    // Verify a qualify.reject savings entry was written for each rejected unit.
    let log_path = stats_dir.join("token_savings.jsonl");
    assert!(
        log_path.exists(),
        "savings JSONL must be created on rejection"
    );
    let content = std::fs::read_to_string(&log_path).expect("read savings log");
    let lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(lines.len(), 2, "one savings entry per rejected candidate");
    for line in &lines {
        let v: serde_json::Value = serde_json::from_str(line).expect("valid JSON");
        assert_eq!(v["op"], "qualify.reject", "op must be qualify.reject");
        assert_eq!(
            v["returned_tokens"], 0,
            "returned_tokens must be 0 for a reject"
        );
        assert!(
            v["baseline_tokens"].as_u64().unwrap_or(0) > 0,
            "baseline_tokens must be > 0 (content has tokens)"
        );
        assert_eq!(
            v["saved_tokens"], v["baseline_tokens"],
            "saved_tokens must equal baseline_tokens when returned=0"
        );
        assert_eq!(v["collection"], "test-col");
    }

    let _ = std::fs::remove_dir_all(&stats_dir);
}

/// Verify that with_savings(..., false) disables qualify.reject recording.
#[tokio::test]
async fn qualify_reject_track_savings_false_writes_nothing() {
    let stats_dir = make_temp_dir("artesian-qualify-reject-off");

    let recall = Arc::new(StaticRecallStore::new(vec![RecallItem::new(
        "n1",
        "rust is great",
        0.05, // below threshold
    )]));
    let config = HeadgateConfig {
        min_score: 0.5,
        ..HeadgateConfig::default()
    };
    // track=false → should not write even though a dir is provided.
    let mut headgate = Headgate::new(recall, config)
        .with_savings("test-col", false)
        .with_savings_dir(stats_dir.clone());
    headgate.cycle("rust").await.expect("cycle");

    let log_path = stats_dir.join("token_savings.jsonl");
    assert!(
        !log_path.exists(),
        "no savings log when track_savings=false"
    );

    let _ = std::fs::remove_dir_all(&stats_dir);
}

/// Verify that without with_savings(), no qualify.reject savings are recorded (opt-in).
#[tokio::test]
async fn qualify_reject_no_savings_config_writes_nothing() {
    let stats_dir = make_temp_dir("artesian-qualify-no-savings");

    let recall = Arc::new(StaticRecallStore::new(vec![RecallItem::new(
        "n1",
        "rust is great",
        0.05,
    )]));
    let config = HeadgateConfig {
        min_score: 0.5,
        ..HeadgateConfig::default()
    };
    // No with_savings call → savings_collection is empty string → opt-out.
    // with_savings_dir is also provided to prove nothing is written there.
    let mut headgate = Headgate::new(recall, config).with_savings_dir(stats_dir.clone());
    headgate.cycle("rust").await.expect("cycle");

    let log_path = stats_dir.join("token_savings.jsonl");
    assert!(
        !log_path.exists(),
        "no savings log when with_savings() was never called"
    );

    let _ = std::fs::remove_dir_all(&stats_dir);
}

#[cfg(feature = "llm")]
#[tokio::test]
async fn llm_compressor_falls_back_when_model_overflows() {
    use headgate::{LlmCompressor, StaticLlmClient};

    // The "model" returns text that still overflows the budget, forcing the extractive fallback.
    let long = "First clause here. Second clause here. Third clause here. \
                Fourth clause here. Fifth clause here. Sixth clause here.";
    let recall = Arc::new(StaticRecallStore::new(vec![RecallItem::new(
        "big", long, 0.9,
    )]));
    let client = Arc::new(StaticLlmClient::new(long)); // no real compression
    let compressor = Arc::new(LlmCompressor::new(client));
    let config = HeadgateConfig {
        budget_tokens: 10,
        compress_on_saturation: true,
        ..HeadgateConfig::default()
    };
    let mut headgate = Headgate::new(recall, config).with_compressor(compressor);

    let metrics = headgate.cycle("query").await.expect("cycle");
    assert_eq!(metrics.admitted, 1);
    assert!(headgate.ccs().token_count() <= 10);
}
