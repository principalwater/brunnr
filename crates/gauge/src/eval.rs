// SPDX-License-Identifier: Apache-2.0

//! Question-answering evaluation harness for competitor-comparable benchmarks
//! (LoCoMo / LongMemEval), the format mem0 and the agent-memory literature report.
//!
//! For each [`QaCase`] the runner: stores the conversation facts as recall candidates,
//! runs one ACC cycle to build a bounded committed context for the question, asks an
//! [`LlmClient`](headgate::LlmClient) to answer **only from that context**, then grades the
//! answer against the gold answer with the same LLM (LLM-as-judge, the standard protocol).
//! It reports accuracy plus the committed **tokens/query** — the token-efficiency number
//! directly comparable to mem0's published per-query token budget.
//!
//! mem0 itself is not re-run here (it is Python + needs cloud LLM calls); its numbers are
//! cited from the published papers in `benchmarks/comparison/README.md`. This harness produces
//! the Artesian side on the same public datasets so the comparison is honest.
//!
//! Retrieval uses a lexical (term-overlap) recall over the case facts by default — deterministic
//! and dependency-free; a vector backend can be substituted for a production run.

#[cfg(feature = "llm")]
mod runner {
    use std::sync::Arc;

    use headgate::{
        count_tokens, Headgate, HeadgateConfig, LlmClient, LlmRequest, RecallItem, RecallStore,
        StaticRecallStore,
    };
    use serde::{Deserialize, Serialize};

    use crate::eval::QaCase;

    /// Aggregate results of a QA evaluation run.
    #[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
    pub struct EvalSummary {
        pub dataset: String,
        pub cases: usize,
        pub correct: usize,
        pub graded: usize,
        pub mean_committed_tokens: f32,
        pub mean_raw_recall_tokens: f32,
        pub accuracy: f32,
        pub footprint_ratio: f32,
    }

    /// Per-case detail, useful for inspecting or dumping failures.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct CaseOutcome {
        pub id: String,
        pub correct: bool,
        pub committed_tokens: usize,
        pub raw_recall_tokens: usize,
        pub answer: String,
    }

    /// Lexical term-overlap score of a fact against the question (case-insensitive word set).
    fn lexical_score(fact: &str, question: &str) -> f32 {
        let question_terms: std::collections::BTreeSet<String> = question
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 2)
            .map(str::to_string)
            .collect();
        if question_terms.is_empty() {
            return 0.0;
        }
        let fact_lower = fact.to_lowercase();
        let hits = question_terms
            .iter()
            .filter(|term| fact_lower.contains(term.as_str()))
            .count();
        hits as f32
    }

    fn recall_items(case: &QaCase) -> Vec<RecallItem> {
        let mut scored: Vec<(f32, usize, &String)> = case
            .facts
            .iter()
            .enumerate()
            .map(|(index, fact)| (lexical_score(fact, &case.question), index, fact))
            .collect();
        // Highest lexical score first; stable by original order on ties.
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        scored
            .into_iter()
            .map(|(score, index, fact)| RecallItem::new(format!("f{index}"), fact.clone(), score))
            .collect()
    }

    fn answer_prompt(committed: &str, question: &str) -> String {
        format!(
            "Answer the question using ONLY the context. If the context lacks the answer, say \
\"unknown\". Reply with just the answer, no explanation.\n\nContext:\n{committed}\n\nQuestion: \
{question}\nAnswer:"
        )
    }

    fn grade_prompt(question: &str, gold: &str, predicted: &str) -> String {
        format!(
            "You grade a predicted answer against the gold answer for the same question. Reply \
with ONLY `yes` if the prediction is correct (same meaning as gold), otherwise `no`.\n\n\
Question: {question}\nGold: {gold}\nPredicted: {predicted}\nCorrect (yes/no):"
        )
    }

    fn parse_grade(reply: &str) -> bool {
        reply.trim().to_lowercase().starts_with("yes")
    }

    /// Run one QA case against a pre-built recall store: ACC cycle → answer → grade.
    pub async fn run_case(
        case: &QaCase,
        recall: Arc<dyn RecallStore>,
        client: &dyn LlmClient,
        config: HeadgateConfig,
    ) -> headgate::HeadgateResult<CaseOutcome> {
        let raw_recall_tokens = case.facts.iter().map(|fact| count_tokens(fact)).sum();
        let mut headgate = Headgate::new(recall, config);
        headgate.cycle(&case.question).await?;
        let committed = headgate.render();
        let committed_tokens = count_tokens(&committed);

        let answer = client
            .complete(
                LlmRequest::new(answer_prompt(&committed, &case.question)).with_temperature(0.0),
            )
            .await?;
        let grade = client
            .complete(
                LlmRequest::new(grade_prompt(&case.question, &case.gold_answer, &answer))
                    .with_temperature(0.0),
            )
            .await?;

        Ok(CaseOutcome {
            id: case.id.clone(),
            correct: parse_grade(&grade),
            committed_tokens,
            raw_recall_tokens,
            answer: answer.trim().to_string(),
        })
    }

    /// Run a full dataset and aggregate. Cases that error (recall build or LLM unreachable) are
    /// skipped and excluded from accuracy, but counted in `cases`.
    pub async fn run_qa_eval(
        dataset: impl Into<String>,
        cases: &[QaCase],
        recall_factory: &dyn RecallFactory,
        client: &dyn LlmClient,
        config: HeadgateConfig,
    ) -> (EvalSummary, Vec<CaseOutcome>) {
        let mut outcomes = Vec::new();
        let mut correct = 0usize;
        let mut committed_total = 0usize;
        let mut raw_total = 0usize;
        for case in cases {
            let Ok(recall) = recall_factory.build(case).await else {
                continue;
            };
            match run_case(case, recall, client, config.clone()).await {
                Ok(outcome) => {
                    if outcome.correct {
                        correct += 1;
                    }
                    committed_total += outcome.committed_tokens;
                    raw_total += outcome.raw_recall_tokens;
                    outcomes.push(outcome);
                }
                Err(_) => continue,
            }
        }
        let graded = outcomes.len();
        let summary = EvalSummary {
            dataset: dataset.into(),
            cases: cases.len(),
            correct,
            graded,
            mean_committed_tokens: mean(committed_total, graded),
            mean_raw_recall_tokens: mean(raw_total, graded),
            accuracy: ratio(correct, graded),
            footprint_ratio: ratio(committed_total, raw_total),
        };
        (summary, outcomes)
    }

    fn mean(total: usize, count: usize) -> f32 {
        if count == 0 {
            0.0
        } else {
            total as f32 / count as f32
        }
    }

    fn ratio(numerator: usize, denominator: usize) -> f32 {
        if denominator == 0 {
            0.0
        } else {
            numerator as f32 / denominator as f32
        }
    }

    /// Builds the recall store for a case — the seam that swaps retrieval strategy.
    pub trait RecallFactory: Send + Sync {
        fn build<'a>(
            &'a self,
            case: &'a QaCase,
        ) -> futures_util::future::BoxFuture<'a, headgate::HeadgateResult<Arc<dyn RecallStore>>>;
    }

    /// Deterministic lexical (term-overlap) recall — no embedder, dependency-free.
    pub struct LexicalRecall;

    impl RecallFactory for LexicalRecall {
        fn build<'a>(
            &'a self,
            case: &'a QaCase,
        ) -> futures_util::future::BoxFuture<'a, headgate::HeadgateResult<Arc<dyn RecallStore>>>
        {
            use futures_util::FutureExt;
            let store: Arc<dyn RecallStore> = Arc::new(StaticRecallStore::new(recall_items(case)));
            async move { Ok(store) }.boxed()
        }
    }

    /// Vector recall: embeds each case's facts into a fresh sqlite-vec collection and recalls
    /// with the real `VectorMemoryBackend` retrieval (embedding + small-to-big + RRF). The
    /// embedder is loaded once and shared across cases.
    #[cfg(feature = "vector")]
    pub struct VectorRecall {
        embedder: Arc<dyn aquifer::TextEmbedder>,
    }

    #[cfg(feature = "vector")]
    impl VectorRecall {
        pub fn new() -> headgate::HeadgateResult<Self> {
            let embedder = aquifer::FastembedTextEmbedder::new()
                .map_err(|error| headgate::HeadgateError::Recall(error.to_string()))?;
            Ok(Self {
                embedder: Arc::new(embedder),
            })
        }
    }

    #[cfg(feature = "vector")]
    impl RecallFactory for VectorRecall {
        fn build<'a>(
            &'a self,
            case: &'a QaCase,
        ) -> futures_util::future::BoxFuture<'a, headgate::HeadgateResult<Arc<dyn RecallStore>>>
        {
            use aquifer::MemoryBackend;
            use futures_util::FutureExt;
            use headgate::{HeadgateError, MemoryRecallStore};

            let embedder = self.embedder.clone();
            let facts = case.facts.clone();
            async move {
                let dir = std::env::temp_dir().join(format!(
                    "gauge-eval-{}-{}",
                    std::process::id(),
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_nanos())
                        .unwrap_or(0)
                ));
                std::fs::create_dir_all(&dir)
                    .map_err(|error| HeadgateError::Recall(error.to_string()))?;
                let store = aquifer::SqliteVecVectorStore::open(
                    aquifer::SqliteVecVectorStoreConfig::new(dir.join("eval.sqlite3")),
                )?;
                let backend = aquifer::VectorMemoryBackend::with_embedder(
                    store,
                    aquifer::VectorMemoryConfig::new("eval"),
                    embedder,
                )?;
                for fact in &facts {
                    backend
                        .store(aquifer::StoreMemory::atom(fact.clone()))
                        .await?;
                }
                Ok(Arc::new(MemoryRecallStore::new(backend)) as Arc<dyn RecallStore>)
            }
            .boxed()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::eval::QaCase;
        use futures_util::{future::BoxFuture, FutureExt};
        use headgate::HeadgateResult;

        /// Answers with a canned string and grades every answer "yes".
        struct PerfectClient;
        impl LlmClient for PerfectClient {
            fn complete(&self, request: LlmRequest) -> BoxFuture<'_, HeadgateResult<String>> {
                let reply = if request.prompt.contains("Correct (yes/no):") {
                    "yes"
                } else {
                    "the team chose Rust"
                };
                let reply = reply.to_string();
                async move { Ok(reply) }.boxed()
            }
        }

        fn case() -> QaCase {
            QaCase {
                id: "c1".to_string(),
                facts: vec![
                    "the team chose Rust for the core crates".to_string(),
                    "the office coffee machine was replaced".to_string(),
                ],
                question: "what language did the team choose".to_string(),
                gold_answer: "Rust".to_string(),
                category: None,
            }
        }

        async fn lexical_store(case: &QaCase) -> Arc<dyn RecallStore> {
            LexicalRecall.build(case).await.expect("recall builds")
        }

        #[tokio::test]
        async fn run_case_answers_and_grades() {
            let case = case();
            let recall = lexical_store(&case).await;
            let outcome = run_case(&case, recall, &PerfectClient, HeadgateConfig::default())
                .await
                .expect("case runs");
            assert!(outcome.correct);
            assert!(outcome.committed_tokens > 0);
            // (On large corpora the budget makes committed << raw; a 2-fact toy is dominated
            // by the slot-markdown scaffolding, so no ordering is asserted here.)
            assert!(outcome.raw_recall_tokens > 0);
        }

        #[tokio::test]
        async fn run_qa_eval_aggregates() {
            let cases = vec![case(), case()];
            let (summary, outcomes) = run_qa_eval(
                "demo",
                &cases,
                &LexicalRecall,
                &PerfectClient,
                HeadgateConfig::default(),
            )
            .await;
            assert_eq!(summary.cases, 2);
            assert_eq!(summary.graded, 2);
            assert_eq!(summary.correct, 2);
            assert!((summary.accuracy - 1.0).abs() < 1e-6);
            assert!(summary.mean_committed_tokens > 0.0);
            assert_eq!(outcomes.len(), 2);
        }
    }
}

#[cfg(all(feature = "llm", feature = "vector"))]
pub use runner::VectorRecall;
#[cfg(feature = "llm")]
pub use runner::{run_case, run_qa_eval, CaseOutcome, EvalSummary, LexicalRecall, RecallFactory};

use serde::{Deserialize, Serialize};

/// A normalized question-answering case: conversation memory + a question and gold answer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QaCase {
    pub id: String,
    /// Conversation memory the agent may recall from (one entry per turn or extracted fact).
    pub facts: Vec<String>,
    pub question: String,
    pub gold_answer: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
}

/// Outcome of parsing a dataset file: the cases plus a count of skipped malformed entries.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LoadReport {
    pub cases: Vec<QaCase>,
    pub skipped: usize,
}

/// Parse the LoCoMo dataset (`locomo10.json`): a list of samples, each with a `conversation`
/// of numbered `session_N` turn lists and a `qa` list. Tolerant of the dataset's mixed shapes;
/// malformed QA entries are skipped and counted.
pub fn load_locomo(json: &str) -> Result<LoadReport, serde_json::Error> {
    let root: serde_json::Value = serde_json::from_str(json)?;
    let samples = root.as_array().cloned().unwrap_or_default();
    let mut report = LoadReport::default();
    for (sample_index, sample) in samples.iter().enumerate() {
        let facts = locomo_facts(sample.get("conversation"));
        let Some(qa_list) = sample.get("qa").and_then(|qa| qa.as_array()) else {
            continue;
        };
        for (qa_index, qa) in qa_list.iter().enumerate() {
            let question = qa.get("question").and_then(|v| v.as_str());
            let answer = qa.get("answer").map(value_to_string);
            match (question, answer) {
                (Some(question), Some(gold_answer)) if !gold_answer.is_empty() => {
                    report.cases.push(QaCase {
                        id: format!("locomo-{sample_index}-{qa_index}"),
                        facts: facts.clone(),
                        question: question.to_string(),
                        gold_answer,
                        category: qa.get("category").map(value_to_string),
                    });
                }
                _ => report.skipped += 1,
            }
        }
    }
    Ok(report)
}

fn locomo_facts(conversation: Option<&serde_json::Value>) -> Vec<String> {
    let Some(object) = conversation.and_then(|c| c.as_object()) else {
        return Vec::new();
    };
    let mut sessions: Vec<(&String, &serde_json::Value)> = object
        .iter()
        .filter(|(key, _)| key.starts_with("session_") && !key.ends_with("date_time"))
        .collect();
    sessions.sort_by_key(|(key, _)| key.as_str().to_string());
    let mut facts = Vec::new();
    for (_, session) in sessions {
        if let Some(turns) = session.as_array() {
            for turn in turns {
                let speaker = turn.get("speaker").and_then(|v| v.as_str()).unwrap_or("");
                let text = turn.get("text").and_then(|v| v.as_str()).unwrap_or("");
                if !text.is_empty() {
                    facts.push(format!("{speaker}: {text}").trim().to_string());
                }
            }
        }
    }
    facts
}

/// Parse the LongMemEval dataset (`longmemeval_s.json` / `_oracle.json`): a list of instances
/// with `question`, `answer`, and `haystack_sessions` (list of sessions, each a list of
/// role/content turns). Malformed instances are skipped and counted.
pub fn load_longmemeval(json: &str) -> Result<LoadReport, serde_json::Error> {
    let root: serde_json::Value = serde_json::from_str(json)?;
    let instances = root.as_array().cloned().unwrap_or_default();
    let mut report = LoadReport::default();
    for (index, instance) in instances.iter().enumerate() {
        let question = instance.get("question").and_then(|v| v.as_str());
        let answer = instance.get("answer").map(value_to_string);
        let facts = longmemeval_facts(instance.get("haystack_sessions"));
        match (question, answer) {
            (Some(question), Some(gold_answer)) if !gold_answer.is_empty() => {
                report.cases.push(QaCase {
                    id: instance
                        .get("question_id")
                        .and_then(|v| v.as_str())
                        .map(str::to_string)
                        .unwrap_or_else(|| format!("longmemeval-{index}")),
                    facts,
                    question: question.to_string(),
                    gold_answer,
                    category: instance.get("question_type").map(value_to_string),
                });
            }
            _ => report.skipped += 1,
        }
    }
    Ok(report)
}

fn longmemeval_facts(haystack: Option<&serde_json::Value>) -> Vec<String> {
    let Some(sessions) = haystack.and_then(|h| h.as_array()) else {
        return Vec::new();
    };
    let mut facts = Vec::new();
    for session in sessions {
        if let Some(turns) = session.as_array() {
            for turn in turns {
                let role = turn.get("role").and_then(|v| v.as_str()).unwrap_or("");
                let content = turn.get("content").and_then(|v| v.as_str()).unwrap_or("");
                if !content.is_empty() {
                    facts.push(format!("{role}: {content}").trim().to_string());
                }
            }
        }
    }
    facts
}

fn value_to_string(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(string) => string.clone(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod loader_tests {
    use super::*;

    #[test]
    fn loads_locomo_shape() {
        let json = r#"[
          {
            "conversation": {
              "session_1_date_time": "2pm on 1 May",
              "session_1": [
                {"speaker": "Alice", "text": "I started learning Rust"},
                {"speaker": "Bob", "text": "nice, for the backend?"}
              ]
            },
            "qa": [
              {"question": "What did Alice start learning?", "answer": "Rust", "category": "single-hop"},
              {"question": "broken", "answer": ""}
            ]
          }
        ]"#;
        let report = load_locomo(json).expect("parse");
        assert_eq!(report.cases.len(), 1);
        assert_eq!(report.skipped, 1);
        let case = &report.cases[0];
        assert_eq!(case.gold_answer, "Rust");
        assert_eq!(case.facts.len(), 2);
        assert!(case.facts[0].contains("Alice"));
    }

    #[test]
    fn loads_longmemeval_shape() {
        let json = r#"[
          {
            "question_id": "q1",
            "question_type": "single-session-user",
            "question": "What is my dog's name?",
            "answer": "Rex",
            "haystack_sessions": [
              [
                {"role": "user", "content": "My dog Rex loves walks"},
                {"role": "assistant", "content": "Rex sounds lovely"}
              ]
            ]
          }
        ]"#;
        let report = load_longmemeval(json).expect("parse");
        assert_eq!(report.cases.len(), 1);
        assert_eq!(report.skipped, 0);
        assert_eq!(report.cases[0].id, "q1");
        assert_eq!(report.cases[0].gold_answer, "Rex");
        assert_eq!(report.cases[0].facts.len(), 2);
    }

    #[test]
    fn integer_answer_is_stringified() {
        let json = r#"[{"conversation":{},"qa":[{"question":"how many?","answer":3}]}]"#;
        let report = load_locomo(json).expect("parse");
        assert_eq!(report.cases.len(), 1);
        assert_eq!(report.cases[0].gold_answer, "3");
    }
}
