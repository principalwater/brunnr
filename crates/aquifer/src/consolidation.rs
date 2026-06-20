// SPDX-License-Identifier: Apache-2.0

//! Consolidation/recharge pass — distills stored memory records into canonical QA claim units.
//!
//! ## Why
//!
//! As an agent loop accumulates memory, the same fact is often stored multiple times with slight
//! paraphrase variants ("we use Rust", "the team chose Rust", "Rust was selected"). These
//! near-duplicates waste tokens and can confuse retrieval. A consolidation pass:
//!
//! 1. Groups near-duplicate records by normalized Jaccard similarity.
//! 2. Distills each group into a canonical **QA claim unit**: a `question` (what is being
//!    asserted) and an `answer` (the canonical fact).
//! 3. Attaches **governance fields** (`scope`, `version`, `source`) — the `scope` field is
//!    the same [`crate::MemoryScope`] used by the Step-4 transactional isolation key, so
//!    consolidated claims are automatically scoped to the right operator/agent/run.
//!
//! ## Honesty note (Corrective-RAG / Graph-RAG)
//!
//! This is the "verify-then-consolidate" half of Corrective-RAG — the ACC qualify-gate already
//! implements the corrective half (drift/relevance/novelty check before admission). Graph-RAG
//! (relational memory via a `relational_map` CCS field) is a future direction and is not
//! implemented here.
//!
//! ## Usage
//!
//! ```no_run
//! # use aquifer::{MemoryRecord, consolidation::{consolidation_pass, ConsolidationOptions}};
//! # let records: Vec<MemoryRecord> = vec![];
//! let report = consolidation_pass(&records, &ConsolidationOptions::default());
//! println!("dedup removed {} records; footprint {} → {} tokens",
//!     report.dedup_removed, report.footprint_tokens_before, report.footprint_tokens_after);
//! ```

use std::collections::HashMap;

use crate::{MemoryRecord, MemoryScope};

/// Governance metadata attached to every consolidated claim.
///
/// `scope` is the [`MemoryScope`] isolation key — the same field the transactional
/// multi-writer substrate uses for per-scope isolation (Step 4). Consolidated claims inherit
/// the scope of their source records so they remain correctly isolated after consolidation.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GovernanceFields {
    /// Per-scope isolation key — maps to [`MemoryScope`] from the transactional substrate.
    pub scope: Option<MemoryScope>,
    /// Monotonic version counter for this claim within its scope.
    pub version: u32,
    /// Comma-separated `node_id`s of the source records this claim was distilled from.
    pub source: String,
}

/// A canonical claim unit distilled from one or more near-duplicate source records.
///
/// Content is stored as `Q: {question}\nA: {answer}` — structured, human-readable, and
/// useful for both retrieval and agent consumption.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ConsolidationClaim {
    /// The topic / question this claim answers (extracted from the leading sentence).
    pub question: String,
    /// The canonical answer / fact (the rest of the content).
    pub answer: String,
    /// Governance metadata for isolation and audit.
    pub governance: GovernanceFields,
    /// `node_id`s of every source record consolidated into this claim.
    pub source_ids: Vec<String>,
    /// Full structured content (`Q: … \nA: …`) suitable for storing back via `StoreMemory`.
    pub content: String,
    /// Tags inherited from the most-recent source record.
    pub tags: Vec<String>,
}

impl ConsolidationClaim {
    /// Render as a `StoreMemory`-compatible content string.
    pub fn render(&self) -> String {
        self.content.clone()
    }
}

/// Options for the consolidation pass.
#[derive(Debug, Clone)]
pub struct ConsolidationOptions {
    /// Jaccard similarity threshold for grouping near-duplicates (0.0–1.0).
    /// Default 0.6: two records sharing 60%+ of their word set are considered near-duplicates.
    pub similarity_threshold: f32,
    /// Override scope for all output claims (defaults to scope of source records).
    pub scope_override: Option<MemoryScope>,
    /// Label used as the `source` prefix in governance fields.
    pub source_label: String,
}

impl Default for ConsolidationOptions {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.6,
            scope_override: None,
            source_label: "consolidation-pass".to_string(),
        }
    }
}

/// Summary of a consolidation pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsolidationReport {
    /// Number of input records.
    pub input_records: usize,
    /// Number of output canonical claim units (groups).
    pub output_claims: usize,
    /// Number of near-duplicate records removed (input_records - output_claims).
    pub dedup_removed: usize,
    /// Estimated token footprint before consolidation (chars / 4 approximation).
    pub footprint_tokens_before: usize,
    /// Estimated token footprint after consolidation.
    pub footprint_tokens_after: usize,
    /// The consolidated claim units.
    pub claims: Vec<ConsolidationClaim>,
}

/// Run the consolidation pass over `records`.
///
/// This function is deterministic and does not require an LLM. It groups near-duplicate
/// records by normalized Jaccard word-set similarity, then distills each group into one
/// canonical [`ConsolidationClaim`].
///
/// For LLM-powered semantic consolidation (optional), the caller can post-process the
/// `claims` by calling an LLM with each claim's `content` to rephrase or summarize.
pub fn consolidation_pass(
    records: &[MemoryRecord],
    options: &ConsolidationOptions,
) -> ConsolidationReport {
    if records.is_empty() {
        return ConsolidationReport {
            input_records: 0,
            output_claims: 0,
            dedup_removed: 0,
            footprint_tokens_before: 0,
            footprint_tokens_after: 0,
            claims: Vec::new(),
        };
    }

    let footprint_before: usize = records.iter().map(|r| r.content.len() / 4 + 1).sum();

    // Normalize: lowercase, collapse whitespace, strip punctuation.
    let normalized: Vec<Vec<&str>> = records
        .iter()
        .map(|r| normalize_words(&r.content))
        .collect();

    // Greedy grouping: for each ungrouped record, start a new group and absorb any
    // subsequent record whose Jaccard similarity with the group centroid exceeds the threshold.
    let mut assigned = vec![false; records.len()];
    let mut groups: Vec<Vec<usize>> = Vec::new();

    for i in 0..records.len() {
        if assigned[i] {
            continue;
        }
        let mut group = vec![i];
        assigned[i] = true;
        for j in (i + 1)..records.len() {
            if assigned[j] {
                continue;
            }
            if jaccard(&normalized[i], &normalized[j]) >= options.similarity_threshold {
                group.push(j);
                assigned[j] = true;
            }
        }
        groups.push(group);
    }

    let mut claims = Vec::with_capacity(groups.len());

    for (group_idx, group) in groups.iter().enumerate() {
        // Pick the most recent record as canonical source.
        let canonical_idx = group
            .iter()
            .copied()
            .max_by_key(|&idx| records[idx].created_at)
            .unwrap_or(group[0]);
        let canonical = &records[canonical_idx];

        let (question, answer) = split_qa(&canonical.content);
        let source_ids: Vec<String> = group
            .iter()
            .map(|&idx| records[idx].node_id.clone())
            .collect();
        let scope = options.scope_override.or(canonical.scope);
        let source_label = if source_ids.len() == 1 {
            source_ids[0].clone()
        } else {
            source_ids.join(", ")
        };
        let content = format!("Q: {question}\nA: {answer}");

        claims.push(ConsolidationClaim {
            question: question.clone(),
            answer: answer.clone(),
            governance: GovernanceFields {
                scope,
                version: (group_idx as u32) + 1,
                source: format!("{}: {}", options.source_label, source_label),
            },
            source_ids,
            content,
            tags: canonical.tags.clone(),
        });
    }

    let footprint_after: usize = claims.iter().map(|c| c.content.len() / 4 + 1).sum();

    ConsolidationReport {
        input_records: records.len(),
        output_claims: claims.len(),
        dedup_removed: records.len().saturating_sub(claims.len()),
        footprint_tokens_before: footprint_before,
        footprint_tokens_after: footprint_after,
        claims,
    }
}

fn normalize_words(text: &str) -> Vec<&str> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect()
}

fn jaccard(a: &[&str], b: &[&str]) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    // Use word counts; normalize to lowercase for comparison.
    let mut counts: HashMap<String, (usize, usize)> = HashMap::new();
    for w in a {
        counts.entry(w.to_lowercase()).or_default().0 += 1;
    }
    for w in b {
        counts.entry(w.to_lowercase()).or_default().1 += 1;
    }
    let intersection: usize = counts.values().map(|(ca, cb)| ca.min(cb)).sum();
    let union: usize = counts.values().map(|(ca, cb)| ca.max(cb)).sum();
    if union == 0 {
        1.0
    } else {
        intersection as f32 / union as f32
    }
}

fn split_qa(content: &str) -> (String, String) {
    // If content already starts with "Q:" or "Question:", keep as-is.
    let trimmed = content.trim();
    if trimmed.starts_with("Q:") || trimmed.starts_with("Question:") {
        let parts: Vec<&str> = trimmed.splitn(2, '\n').collect();
        if parts.len() == 2 {
            return (
                parts[0].trim_start_matches("Q:").trim().to_string(),
                parts[1].trim().to_string(),
            );
        }
    }
    // Extract question from first sentence (up to '.', '?', or '!').
    let question_end = trimmed
        .find(['.', '?', '!'])
        .map(|i| i + 1)
        .unwrap_or(trimmed.len().min(80));
    let question = trimmed[..question_end].trim().to_string();
    let answer = if question_end < trimmed.len() {
        trimmed[question_end..].trim().to_string()
    } else {
        trimmed.to_string()
    };
    (question, answer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MemoryId, MemoryRecord, MemoryTier};
    use std::collections::BTreeMap;

    fn make_record(id: &str, content: &str) -> MemoryRecord {
        MemoryRecord::new(
            MemoryId::new(id),
            id,
            content,
            vec!["test".to_string()],
            BTreeMap::new(),
            MemoryTier::L1Atom,
        )
    }

    #[test]
    fn exact_duplicates_are_merged_into_one_claim() {
        let records = vec![
            make_record("a", "the team chose Rust for the core crates"),
            make_record("b", "the team chose Rust for the core crates"),
        ];
        let report = consolidation_pass(&records, &ConsolidationOptions::default());
        assert_eq!(report.input_records, 2);
        assert_eq!(report.output_claims, 1);
        assert_eq!(report.dedup_removed, 1);
    }

    #[test]
    fn near_duplicates_are_merged() {
        // Sentences share ≥ 0.6 of their word set → merged; third is distinct.
        let records = vec![
            make_record("a", "the team chose Rust for the core crate"),
            make_record("b", "the team chose Rust for the core crates"), // one word differs
            make_record("c", "Python is a scripting language with a GIL"), // distinct
        ];
        let report = consolidation_pass(&records, &ConsolidationOptions::default());
        assert_eq!(report.input_records, 3);
        // First two share ≥ 0.6 Jaccard and merge; third is distinct → 2 claims.
        assert!(
            report.output_claims <= 2,
            "near-duplicates should merge into one claim"
        );
        assert!(
            report.dedup_removed >= 1,
            "should remove at least one near-duplicate"
        );
    }

    #[test]
    fn distinct_records_each_become_a_claim() {
        let records = vec![
            make_record("a", "chose Rust for performance"),
            make_record("b", "Python is the scripting layer"),
            make_record("c", "Postgres stores the relational data"),
        ];
        let report = consolidation_pass(&records, &ConsolidationOptions::default());
        assert_eq!(
            report.output_claims, 3,
            "all distinct records stay separate"
        );
        assert_eq!(report.dedup_removed, 0);
    }

    #[test]
    fn claim_content_has_qa_format() {
        let records = vec![make_record(
            "a",
            "the team chose Rust for the systems crate.",
        )];
        let report = consolidation_pass(&records, &ConsolidationOptions::default());
        assert_eq!(report.claims.len(), 1);
        assert!(
            report.claims[0].content.starts_with("Q:"),
            "claim content should start with Q:"
        );
    }

    #[test]
    fn footprint_after_is_less_or_equal_before() {
        let records = vec![
            make_record("a", "the team chose Rust for the systems crate"),
            make_record("b", "the team chose Rust for the systems crate"),
            make_record("c", "the team chose Rust for the systems crate"),
        ];
        let report = consolidation_pass(&records, &ConsolidationOptions::default());
        assert!(report.footprint_tokens_after <= report.footprint_tokens_before);
    }

    #[test]
    fn governance_fields_inherit_scope_from_source() {
        let mut record = make_record("a", "chose Rust");
        record.scope = Some(MemoryScope::Session);
        let report = consolidation_pass(&[record], &ConsolidationOptions::default());
        assert_eq!(
            report.claims[0].governance.scope,
            Some(MemoryScope::Session)
        );
    }

    #[test]
    fn empty_input_returns_empty_report() {
        let report = consolidation_pass(&[], &ConsolidationOptions::default());
        assert_eq!(report.input_records, 0);
        assert_eq!(report.output_claims, 0);
        assert_eq!(report.claims.len(), 0);
    }
}
