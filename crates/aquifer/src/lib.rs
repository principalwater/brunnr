// SPDX-License-Identifier: Apache-2.0

//! Aquifer memory API and local backends.

mod anchor;
mod backend;
mod backfill;
mod chunking;
mod compat;
pub mod entity;
pub mod episode;
mod files;
mod identity;
mod lane_lock;
mod mmr;
#[cfg(feature = "pgvector")]
mod pgvector;
#[cfg(feature = "qdrant")]
mod qdrant;
mod retrieval;
mod rrf;
mod semantic_cache;
#[cfg(feature = "sqlite-vec")]
mod sqlite_vec;
pub mod temporal;
pub mod txn;
mod types;
mod upgrade;
#[cfg(feature = "vector")]
mod vector;
#[cfg(feature = "vector")]
mod vector_memory;
mod working;

pub use anchor::{recover_after_compaction, AnchorAnchorStore, RecoveryContext, SessionAnchor};
pub use backend::MemoryBackend;
pub use backfill::{
    backfill_directory, collect_memory_paths, parse_memory_path, BackfillFailure, BackfillStats,
};
pub use chunking::{chunk_text, Chunk, ChunkConfig};
pub use compat::{CollectionCompat, COMPAT_POINT_ID, OKF_VERSION};
pub use entity::{extract_entities, EntityIndex};
pub use episode::EpisodeIndex;
pub use files::FilesBackend;
pub use identity::stable_memory_id;
pub use lane_lock::{SessionLaneGuard, SessionLaneLock};
pub use mmr::{mmr_diversify, MMR_DEFAULT_LAMBDA};
#[cfg(feature = "pgvector")]
pub use pgvector::{PgVectorBackend, PgVectorStore};
#[cfg(feature = "qdrant")]
pub use qdrant::{
    preflight_qdrant, replicate_collection, QdrantBackend, QdrantEndpoints, QdrantPreflightReport,
    QdrantVectorStore, QdrantVectorStoreConfig,
};
#[cfg(feature = "vector")]
pub use retrieval::FastembedReranker;
pub use retrieval::{LocalLexicalReranker, Reranker};
pub use rrf::reciprocal_rank_fusion;
#[cfg(feature = "vector")]
pub use semantic_cache::EmbedderVectorizer;
pub use semantic_cache::{cosine_similarity, CachingMemoryBackend, QueryVectorizer, SemanticCache};
#[cfg(feature = "sqlite-vec")]
pub use sqlite_vec::{SqliteVecBackend, SqliteVecVectorStore, SqliteVecVectorStoreConfig};
pub use temporal::{apply_knowledge_supersession, apply_recency_decay};
pub use txn::{sync_okf_directory, CommitLog, SyncReport, TransactionalMemory, TxnError, TxnSeq};
pub use types::{
    MemoryError, MemoryId, MemoryQuery, MemoryRecord, MemoryResult, MemoryScope, MemoryTier,
    RrfOptions, SearchHit, SearchSource, StoreMemory,
};
pub use upgrade::{
    default_migration_collection, export_okf_bundle, migrate_okf_bundle, migration_manifest_path,
    rechunk_oversized_sqlite, verify_okf_bundle, MigrationPlan, MigrationReport, OkfExportReport,
    OkfVerifyReport, RechunkReport, SnapshotReport, VectorCollectionAdmin,
};
#[cfg(feature = "vector")]
pub use vector::{
    Distance, Filter, FilterCondition, FilterValue, PayloadIndex, RangeFilter, VectorCollection,
    VectorPoint, VectorQuantization, VectorSearch, VectorSearchHit, VectorSearchSource,
    VectorStore, VectorStoreCapabilities,
};
#[cfg(feature = "vector")]
pub use vector_memory::{
    FastembedTextEmbedder, TextEmbedder, VectorMemoryBackend, VectorMemoryConfig,
    PINNED_FASTEMBED_DIMENSIONS, PINNED_FASTEMBED_MODEL,
};
pub use working::{
    InMemoryWorkingMemory, WorkingMemory, WorkingMemoryMode, WorkingMemoryView, WorkingTurn,
};

pub mod consolidation;
pub use consolidation::{
    consolidation_pass, ConsolidationClaim, ConsolidationOptions, ConsolidationReport,
    GovernanceFields,
};
