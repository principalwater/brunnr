// SPDX-License-Identifier: Apache-2.0

//! Tests for import-time deterministic relation extraction (`relation_extraction = true`).
//!
//! All tests use the in-memory SQLite-vec backend — no external Qdrant, no live LLM.

use std::sync::Arc;

use aquifer::{
    MemoryBackend, MemoryQuery, MemoryResult, MemoryTier, SqliteVecVectorStore, StoreMemory,
    TextEmbedder, VectorMemoryBackend, VectorMemoryConfig,
};

// ── helpers ──────────────────────────────────────────────────────────────────

const TEST_DIMS: usize = 8;

struct TestEmbedder;

impl TextEmbedder for TestEmbedder {
    fn embed_query(&self, text: &str) -> MemoryResult<Vec<f32>> {
        Ok(test_vec(text))
    }
    fn embed_passage(&self, text: &str) -> MemoryResult<Vec<f32>> {
        Ok(test_vec(text))
    }
}

fn test_vec(text: &str) -> Vec<f32> {
    let mut v = vec![0.0f32; TEST_DIMS];
    for token in text.split_whitespace() {
        let idx = token
            .bytes()
            .fold(0usize, |h, b| h.wrapping_mul(31).wrapping_add(b as usize))
            % TEST_DIMS;
        v[idx] += 1.0;
    }
    let mag = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag > 0.0 {
        for x in &mut v {
            *x /= mag;
        }
    }
    v
}

/// Build a backend with `relation_extraction` set to the given value.
fn make_backend(
    collection: &str,
    relation_extraction: bool,
) -> VectorMemoryBackend<SqliteVecVectorStore> {
    let store = SqliteVecVectorStore::in_memory().expect("in-memory sqlite-vec");
    VectorMemoryBackend::with_embedder(
        store,
        VectorMemoryConfig {
            collection: collection.to_string(),
            dimensions: TEST_DIMS,
            ..VectorMemoryConfig::new(collection)
        }
        .with_relation_extraction(relation_extraction),
        Arc::new(TestEmbedder),
    )
    .expect("backend should construct")
}

fn atom_with_tags(content: &str, tags: Vec<&str>) -> StoreMemory {
    StoreMemory {
        content: content.to_string(),
        tags: tags.into_iter().map(str::to_string).collect(),
        metadata: Default::default(),
        tier: MemoryTier::L1Atom,
        node_id: None,
        created_at: None,
        scope: None,
        agent_id: None,
        session_id: None,
        task_id: None,
        user_id: None,
        project: None,
        source: None,
        confidence: None,
        relations: Vec::new(),
    }
}

fn atom(content: &str) -> StoreMemory {
    atom_with_tags(content, Vec::new())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// With `relation_extraction = true`, storing a memory with a camelCase entity in its content
/// should produce `mentions` relations visible on retrieval, so `by_entity` returns the record.
#[tokio::test]
async fn import_with_relation_extraction_makes_by_entity_return_links() {
    let backend = make_backend("link-by-entity", true);

    // Content contains `RustLang` (PascalCase entity), tags contain `backend`.
    let memory = atom_with_tags("We use RustLang for all backend systems", vec!["backend"]);
    backend.store(memory).await.expect("store should succeed");

    // `by_entity` should find the record via the `RustLang` entity extracted from content.
    let by_rust = backend
        .by_entity("RustLang")
        .await
        .expect("by_entity should succeed");
    assert!(
        !by_rust.is_empty(),
        "by_entity('RustLang') should return the stored record when relation_extraction is on"
    );

    // Also reachable by tag-derived entity.
    let by_backend = backend
        .by_entity("backend")
        .await
        .expect("by_entity should succeed");
    assert!(
        !by_backend.is_empty(),
        "by_entity('backend') should return the stored record via tag-derived relation"
    );
}

/// With `relation_extraction = true`, two records that share an entity should be reachable from
/// each other via `neighbors` (1-hop through the shared relation).
#[tokio::test]
async fn import_with_relation_extraction_makes_neighbors_return_linked_records() {
    let backend = make_backend("link-neighbors", true);

    // Both records mention `AquiferDB` — extraction gives both a `mentions AquiferDB` relation.
    let r1 = backend
        .store(atom_with_tags(
            "AquiferDB is our primary vector store",
            vec!["AquiferDB"],
        ))
        .await
        .expect("store r1");
    let r2 = backend
        .store(atom_with_tags(
            "AquiferDB supports multi-user tenancy",
            vec!["AquiferDB"],
        ))
        .await
        .expect("store r2");

    // Neighbors of r1 should include r2 (they share the `AquiferDB` entity edge).
    let neighbors = backend
        .neighbors(&r1.node_id, 1)
        .await
        .expect("neighbors should succeed");
    let neighbor_ids: Vec<_> = neighbors.iter().map(|r| r.node_id.clone()).collect();
    assert!(
        neighbor_ids.contains(&r2.node_id),
        "neighbors of r1 should include r2 when both share an extracted entity relation; \
         got {:?}",
        neighbor_ids
    );
}

/// With `relation_extraction = false` (`--no-link`), no deterministic relations are extracted,
/// so `by_entity` returns nothing for entities that were only in the content (not explicit relations).
#[tokio::test]
async fn import_without_relation_extraction_by_entity_returns_empty() {
    let backend = make_backend("no-link-by-entity", false);

    // Content has `RustLang` but no explicit relations were attached.
    backend
        .store(atom("We use RustLang for all backend services"))
        .await
        .expect("store should succeed");

    let by_rust = backend
        .by_entity("RustLang")
        .await
        .expect("by_entity should succeed");
    assert!(
        by_rust.is_empty(),
        "by_entity('RustLang') should return nothing when relation_extraction is off"
    );
}

/// Re-importing identical content is idempotent even with relation extraction enabled.
/// The second import must not create duplicate records or spurious relation entries.
#[tokio::test]
async fn import_with_relation_extraction_is_idempotent() {
    let backend = make_backend("link-idempotent", true);

    let memory = atom_with_tags("GraphQL powers our API layer", vec!["GraphQL"]);

    // First import.
    let r1 = backend
        .store(memory.clone())
        .await
        .expect("first store should succeed");
    // Second import of identical content.
    let r2 = backend
        .store(memory)
        .await
        .expect("second store should succeed");

    // Content-hash IDs must be the same (idempotent).
    assert_eq!(
        r1.id, r2.id,
        "second import of same content should return the same id"
    );

    // `by_entity` still returns exactly one hit, not two.
    let hits = backend
        .find(MemoryQuery::new("API layer").with_limit(20))
        .await
        .expect("find should succeed");
    let unique_ids: std::collections::BTreeSet<_> =
        hits.iter().map(|h| h.record.node_id.clone()).collect();
    assert_eq!(
        unique_ids.len(),
        1,
        "idempotent re-import should leave exactly one record"
    );
}

/// Bulk import (`bulk_store`) with `relation_extraction = true` also produces linked records.
#[tokio::test]
async fn bulk_import_with_relation_extraction_links_entities() {
    let backend = make_backend("bulk-link", true);

    let memories = vec![
        atom_with_tags("PostgreSQL stores our relational data", vec!["PostgreSQL"]),
        atom_with_tags(
            "PostgreSQL supports JSONB for document storage",
            vec!["PostgreSQL"],
        ),
    ];
    let report = backend.bulk_store(memories, 128).await;
    assert_eq!(report.stored, 2, "both records should be stored");
    assert!(report.failures.is_empty(), "no failures expected");

    // Both records are linked via the `PostgreSQL` entity.
    let by_pg = backend
        .by_entity("PostgreSQL")
        .await
        .expect("by_entity should succeed");
    assert_eq!(
        by_pg.len(),
        2,
        "bulk_store with relation_extraction should produce two linked records"
    );
}

/// Guard for the LLM consolidation path: calling `consolidation_pass` with zero records and
/// no LLM must not fail and must produce a zeroed report.  This covers the no-LLM branch
/// that `consolidate_after_import` hits when no [acc.compressor/judge] is configured.
#[test]
fn consolidation_pass_with_no_records_returns_zeroed_report() {
    let records = vec![];
    let report = aquifer::consolidation_pass(&records, &aquifer::ConsolidationOptions::default());
    assert_eq!(report.input_records, 0);
    assert_eq!(report.output_claims, 0);
    assert_eq!(report.dedup_removed, 0);
}
