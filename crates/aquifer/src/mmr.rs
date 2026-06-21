// SPDX-License-Identifier: Apache-2.0
//! Maximal Marginal Relevance (MMR) re-ranking for diverse recall.
//!
//! Plain top-k recall can return several near-duplicate hits (e.g. successive loop-turn commits),
//! crowding out distinct, useful context. MMR greedily selects the hit that best balances
//! **relevance** with **novelty** versus what is already selected, trimming redundancy. Similarity
//! is content-token Jaccard, so it works on any backend without needing the candidate embeddings.

use std::collections::BTreeSet;

use crate::SearchHit;

/// Default relevance/novelty trade-off — favors relevance while still shedding duplicates.
pub const MMR_DEFAULT_LAMBDA: f32 = 0.7;

fn word_set(text: &str) -> BTreeSet<String> {
    text.to_lowercase()
        .split(|character: char| !character.is_alphanumeric())
        .filter(|word| word.len() > 2)
        .map(str::to_string)
        .collect()
}

fn jaccard(left: &BTreeSet<String>, right: &BTreeSet<String>) -> f32 {
    let union = left.union(right).count();
    if union == 0 {
        return 0.0;
    }
    left.intersection(right).count() as f32 / union as f32
}

/// Re-rank `hits` with MMR and return the top `limit`. `lambda` in `[0, 1]` trades relevance
/// (`1.0`) against novelty (`0.0`); see [`MMR_DEFAULT_LAMBDA`]. Relevance is scaled by the largest
/// score so the trade-off is meaningful regardless of the backend's score scale.
pub fn mmr_diversify(hits: Vec<SearchHit>, limit: usize, lambda: f32) -> Vec<SearchHit> {
    if hits.len() <= 1 || limit <= 1 {
        return hits.into_iter().take(limit).collect();
    }
    let lambda = lambda.clamp(0.0, 1.0);
    // Scale relevance by the largest magnitude rather than min-max: min-max maps the least
    // relevant hit to exactly 0, over-penalizing it; dividing by |max| preserves the score
    // ratios so the trade-off stays meaningful (and tolerates reranker logits).
    let max = hits.iter().map(|hit| hit.score).fold(f32::MIN, f32::max);
    let denom = max.abs().max(f32::EPSILON);
    let relevance: Vec<f32> = hits.iter().map(|hit| hit.score / denom).collect();
    let word_sets: Vec<BTreeSet<String>> = hits
        .iter()
        .map(|hit| word_set(&hit.record.content))
        .collect();

    let mut selected: Vec<usize> = Vec::with_capacity(limit.min(hits.len()));
    let mut remaining: Vec<usize> = (0..hits.len()).collect();
    while selected.len() < limit && !remaining.is_empty() {
        let mut best = remaining[0];
        let mut best_score = f32::MIN;
        for &candidate in &remaining {
            let max_sim = selected
                .iter()
                .map(|&chosen| jaccard(&word_sets[candidate], &word_sets[chosen]))
                .fold(0.0_f32, f32::max);
            let mmr = lambda * relevance[candidate] - (1.0 - lambda) * max_sim;
            if mmr > best_score {
                best_score = mmr;
                best = candidate;
            }
        }
        selected.push(best);
        remaining.retain(|&index| index != best);
    }

    let mut slots: Vec<Option<SearchHit>> = hits.into_iter().map(Some).collect();
    selected
        .into_iter()
        .filter_map(|index| slots[index].take())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MemoryId, MemoryRecord, MemoryTier};
    use std::collections::BTreeMap;

    fn hit(id: &str, content: &str, score: f32) -> SearchHit {
        let record = MemoryRecord::new(
            MemoryId::new(id),
            format!("node:{id}"),
            content,
            Vec::new(),
            BTreeMap::new(),
            MemoryTier::L1Atom,
        );
        SearchHit::keyword(record, score)
    }

    #[test]
    fn diversifies_near_duplicates() {
        // Two near-identical top hits and one distinct lower hit. With diversity weighting the
        // distinct hit should be chosen over the duplicate for the second slot.
        let hits = vec![
            hit("a", "deploy the service to production cluster", 1.0),
            hit("b", "deploy the service to production cluster now", 0.95),
            hit("c", "rotate the database credentials quarterly", 0.6),
        ];
        let out = mmr_diversify(hits, 2, 0.5);
        let ids: Vec<String> = out.iter().map(|h| h.record.id.to_string()).collect();
        assert_eq!(ids[0], "a", "most relevant stays first");
        assert_eq!(
            ids[1], "c",
            "second slot favors the novel hit over the duplicate"
        );
    }

    #[test]
    fn lambda_one_preserves_relevance_order() {
        let hits = vec![
            hit("a", "alpha alpha alpha", 1.0),
            hit("b", "alpha alpha beta", 0.9),
            hit("c", "gamma delta epsilon", 0.5),
        ];
        let out = mmr_diversify(hits, 3, 1.0);
        let ids: Vec<String> = out.iter().map(|h| h.record.id.to_string()).collect();
        assert_eq!(ids, vec!["a", "b", "c"], "lambda=1 is pure relevance order");
    }
}
