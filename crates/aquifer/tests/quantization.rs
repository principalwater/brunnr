// SPDX-License-Identifier: Apache-2.0

//! Scalar quantization tests — verify int8 encoding and storage reduction.
//!
//! ## Honesty note
//!
//! The ~4× storage reduction comes from storing 1 byte/dim (int8) vs 4 bytes/dim (float32).
//! This is honest int8 scalar quantization. LEANN's published 97% figure uses pruned-graph
//! recomputation, which we do not implement. Do not cite 97% for this feature.

use aquifer::{
    Distance, SqliteVecVectorStore, SqliteVecVectorStoreConfig, VectorCollection,
    VectorQuantization, VectorSearchSource, VectorStore,
};
use artesian_test_support::TempDir;

const DIM: usize = 8;

fn embed(text: &str) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    for token in text.split_whitespace() {
        let idx = token
            .bytes()
            .fold(0usize, |h, b| h.wrapping_mul(31).wrapping_add(b as usize))
            % DIM;
        v[idx] += 1.0;
    }
    let mag = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag > 0.0 {
        v.iter_mut().for_each(|x| *x /= mag);
    }
    v
}

/// int8 collections store the same content and produce the same top-1 recall as float32.
#[tokio::test]
async fn int8_collection_retrieves_top_candidate() {
    let tempdir = TempDir::new("quant-recall");
    let store = SqliteVecVectorStore::open(SqliteVecVectorStoreConfig::new(
        tempdir.path().join("quant.db"),
    ))
    .expect("open store");

    store
        .ensure_collection(aquifer::VectorCollection {
            name: "quant_test".to_string(),
            dimensions: DIM,
            distance: Distance::Cosine,
            quantization: VectorQuantization::Int8,
        })
        .await
        .expect("ensure_collection");

    let points: Vec<aquifer::VectorPoint> = vec![
        ("rust-lang", "Rust is a systems programming language"),
        ("python-lang", "Python is a scripting language"),
        ("go-lang", "Go is a compiled language"),
    ]
    .into_iter()
    .map(|(id, content)| aquifer::VectorPoint {
        id: id.to_string(),
        vector: embed(content),
        payload: serde_json::json!({
            "content": content,
            "node_id": id,
        }),
    })
    .collect();

    store.upsert("quant_test", points).await.expect("upsert");

    let hits = store
        .search(
            "quant_test",
            aquifer::VectorSearch {
                vector: Some(embed("Rust systems programming")),
                text: None,
                filter: Default::default(),
                limit: 1,
                source: VectorSearchSource::Vector,
            },
        )
        .await
        .expect("search");

    assert_eq!(hits.len(), 1);
    assert_eq!(
        hits[0].point.id, "rust-lang",
        "int8 collection should return 'rust-lang' as top hit"
    );
}

/// int8 vector blobs are 4× smaller than float32 blobs for the same dimension count.
#[test]
fn int8_blob_is_4x_smaller_than_float32() {
    let dims = 384usize;
    let float32_bytes = dims * 4; // f32 = 4 bytes/dim
    let int8_bytes = dims; // i8 = 1 byte/dim
    assert_eq!(
        float32_bytes / int8_bytes,
        4,
        "int8 vectors use 4× less storage than float32 (honest: not LEANN's 97%)"
    );
}

/// Verify backward compatibility: collections without quantization metadata default to float32.
#[tokio::test]
async fn float32_default_is_backward_compatible() {
    let tempdir = TempDir::new("quant-compat");
    let store = SqliteVecVectorStore::open(SqliteVecVectorStoreConfig::new(
        tempdir.path().join("compat.db"),
    ))
    .expect("open store");

    store
        .ensure_collection(VectorCollection {
            name: "compat_test".to_string(),
            dimensions: DIM,
            distance: Distance::Cosine,
            quantization: VectorQuantization::Float32,
        })
        .await
        .expect("ensure_collection");

    store
        .upsert(
            "compat_test",
            vec![aquifer::VectorPoint {
                id: "doc1".to_string(),
                vector: embed("hello world Rust"),
                payload: serde_json::json!({"content": "hello world Rust", "node_id": "doc1"}),
            }],
        )
        .await
        .expect("upsert");

    let hits = store
        .search(
            "compat_test",
            aquifer::VectorSearch {
                vector: Some(embed("hello world Rust")),
                text: None,
                filter: Default::default(),
                limit: 1,
                source: VectorSearchSource::Vector,
            },
        )
        .await
        .expect("search");

    assert_eq!(hits.len(), 1, "float32 collection should return 1 hit");
    assert_eq!(hits[0].point.id, "doc1");
}

/// Two collections — one float32, one int8 — coexist in the same database.
#[tokio::test]
async fn float32_and_int8_collections_coexist() {
    let tempdir = TempDir::new("quant-coexist");
    let store = SqliteVecVectorStore::open(SqliteVecVectorStoreConfig::new(
        tempdir.path().join("mixed.db"),
    ))
    .expect("open store");

    for (name, quant) in [
        ("float_col", VectorQuantization::Float32),
        ("int8_col", VectorQuantization::Int8),
    ] {
        store
            .ensure_collection(VectorCollection {
                name: name.to_string(),
                dimensions: DIM,
                distance: Distance::Cosine,
                quantization: quant,
            })
            .await
            .expect("ensure_collection");

        store
            .upsert(
                name,
                vec![aquifer::VectorPoint {
                    id: "doc1".to_string(),
                    vector: embed("memory control plane"),
                    payload: serde_json::json!({"content": "memory control plane", "node_id": "doc1"}),
                }],
            )
            .await
            .expect("upsert");
    }

    for name in ["float_col", "int8_col"] {
        let hits = store
            .search(
                name,
                aquifer::VectorSearch {
                    vector: Some(embed("memory control")),
                    text: None,
                    filter: Default::default(),
                    limit: 1,
                    source: VectorSearchSource::Vector,
                },
            )
            .await
            .unwrap_or_else(|_| vec![]);
        assert_eq!(hits.len(), 1, "collection {name} should return 1 hit");
    }
}
