// SPDX-License-Identifier: Apache-2.0

//! Mímisbrunnr memory API and local backends.

mod backend;
mod backfill;
mod files;
mod identity;
#[cfg(feature = "qdrant")]
mod qdrant;
mod rrf;
#[cfg(feature = "sqlite-vec")]
mod sqlite_vec;
mod types;
#[cfg(feature = "vector")]
mod vector;
#[cfg(feature = "vector")]
mod vector_memory;

pub use backend::MemoryBackend;
pub use backfill::{backfill_directory, BackfillStats};
pub use files::FilesBackend;
#[cfg(feature = "qdrant")]
pub use qdrant::{QdrantBackend, QdrantVectorStore, QdrantVectorStoreConfig};
pub use rrf::reciprocal_rank_fusion;
#[cfg(feature = "sqlite-vec")]
pub use sqlite_vec::{SqliteVecBackend, SqliteVecVectorStore, SqliteVecVectorStoreConfig};
pub use types::{
    MemoryError, MemoryId, MemoryQuery, MemoryRecord, MemoryResult, MemoryTier, RrfOptions,
    SearchHit, SearchSource, StoreMemory,
};
#[cfg(feature = "vector")]
pub use vector::{
    Distance, Filter, FilterCondition, FilterValue, PayloadIndex, RangeFilter, VectorCollection,
    VectorPoint, VectorSearch, VectorSearchHit, VectorSearchSource, VectorStore,
    VectorStoreCapabilities,
};
#[cfg(feature = "vector")]
pub use vector_memory::{
    FastembedTextEmbedder, TextEmbedder, VectorMemoryBackend, VectorMemoryConfig,
    PINNED_FASTEMBED_DIMENSIONS, PINNED_FASTEMBED_MODEL,
};
