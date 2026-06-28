// SPDX-License-Identifier: Apache-2.0

use futures_util::{future::BoxFuture, FutureExt};

use crate::{
    reciprocal_rank_fusion, MemoryQuery, MemoryRecord, MemoryResult, RrfOptions, SearchHit,
    StoreMemory,
};

/// Aggregate counts returned by `MemoryBackend::bulk_store`.
#[derive(Debug, Clone, Default)]
pub struct BulkStoreReport {
    /// Number of chunks newly written.
    pub stored: usize,
    /// Number of chunks that already existed and were skipped.
    pub skipped: usize,
    /// IDs of chunks that failed to store, with error messages.
    pub failures: Vec<(String, String)>,
}

/// Pluggable memory backend contract.
///
/// Backends must support storing durable memories, finding relevant memories, hybrid RRF fusion,
/// and deterministic drill-down by `node_id` so summaries stay traceable to ground-truth records.
/// Backends may also expose an entity-relation layer; the default graph methods return empty so
/// older or non-indexing backends remain valid.
///
/// ```
/// # use futures_util::{future::BoxFuture, FutureExt};
/// # use aquifer::{
/// #     MemoryBackend, MemoryQuery, MemoryRecord, MemoryResult, RrfOptions, SearchHit,
/// #     StoreMemory,
/// # };
/// # struct EmptyBackend;
/// # impl MemoryBackend for EmptyBackend {
/// #     fn find(&self, _: MemoryQuery) -> BoxFuture<'_, MemoryResult<Vec<SearchHit>>> {
/// #         async { Ok(Vec::new()) }.boxed()
/// #     }
/// #     fn store(&self, _: StoreMemory) -> BoxFuture<'_, MemoryResult<MemoryRecord>> {
/// #         async { unimplemented!("example backend") }.boxed()
/// #     }
/// #     fn get_node(&self, _: &str) -> BoxFuture<'_, MemoryResult<Option<MemoryRecord>>> {
/// #         async { Ok(None) }.boxed()
/// #     }
/// # }
/// let backend = EmptyBackend;
/// let query = MemoryQuery::new("project convention");
/// # let _ = (backend, query, RrfOptions::default());
/// ```
pub trait MemoryBackend: Send + Sync {
    fn find(&self, query: MemoryQuery) -> BoxFuture<'_, MemoryResult<Vec<SearchHit>>>;

    fn store(&self, memory: StoreMemory) -> BoxFuture<'_, MemoryResult<MemoryRecord>>;

    fn hybrid_rrf(
        &self,
        keyword_query: MemoryQuery,
        vector_query: MemoryQuery,
        options: RrfOptions,
    ) -> BoxFuture<'_, MemoryResult<Vec<SearchHit>>> {
        async move {
            let keyword_hits = self.find(keyword_query).await?;
            let vector_hits = self.find(vector_query).await?;
            Ok(reciprocal_rank_fusion(
                &[keyword_hits, vector_hits],
                options,
            ))
        }
        .boxed()
    }

    fn get_node(&self, node_id: &str) -> BoxFuture<'_, MemoryResult<Option<MemoryRecord>>>;

    fn neighbors(
        &self,
        _node_id: &str,
        _hops: usize,
    ) -> BoxFuture<'_, MemoryResult<Vec<MemoryRecord>>> {
        async { Ok(Vec::new()) }.boxed()
    }

    fn by_entity(&self, _entity: &str) -> BoxFuture<'_, MemoryResult<Vec<MemoryRecord>>> {
        async { Ok(Vec::new()) }.boxed()
    }

    fn projects(&self) -> BoxFuture<'_, MemoryResult<Vec<String>>> {
        async { Ok(Vec::new()) }.boxed()
    }

    /// Store many memories in bulk. The default implementation is a sequential loop over
    /// `store()`, which is correct for all backends. Backends that support batch upsert (e.g.
    /// `VectorMemoryBackend<QdrantVectorStore>`) override this to batch the upserts and skip the
    /// per-chunk existence round-trip, making large imports dramatically faster.
    ///
    /// Content-hash IDs are deterministic so re-importing identical content is idempotent:
    /// the `skipped` count reflects duplicates detected by a single up-front bulk ID check;
    /// `stored` counts newly written chunks.
    fn bulk_store<'a>(
        &'a self,
        memories: Vec<StoreMemory>,
        _batch_size: usize,
    ) -> BoxFuture<'a, BulkStoreReport> {
        async move {
            let mut report = BulkStoreReport::default();
            for memory in memories {
                let id = crate::identity::stable_memory_id(&memory);
                match self.store(memory).await {
                    Ok(_) => report.stored += 1,
                    Err(error) => report.failures.push((id.to_string(), error.to_string())),
                }
            }
            report
        }
        .boxed()
    }
}

/// Delegating impl so a type-erased `Arc<dyn MemoryBackend>` can be used wherever a
/// `MemoryBackend` is expected (e.g. wrapping a runtime-selected backend in an adapter).
impl MemoryBackend for std::sync::Arc<dyn MemoryBackend> {
    fn find(&self, query: MemoryQuery) -> BoxFuture<'_, MemoryResult<Vec<SearchHit>>> {
        (**self).find(query)
    }

    fn store(&self, memory: StoreMemory) -> BoxFuture<'_, MemoryResult<MemoryRecord>> {
        (**self).store(memory)
    }

    fn get_node(&self, node_id: &str) -> BoxFuture<'_, MemoryResult<Option<MemoryRecord>>> {
        (**self).get_node(node_id)
    }

    fn neighbors(
        &self,
        node_id: &str,
        hops: usize,
    ) -> BoxFuture<'_, MemoryResult<Vec<MemoryRecord>>> {
        (**self).neighbors(node_id, hops)
    }

    fn by_entity(&self, entity: &str) -> BoxFuture<'_, MemoryResult<Vec<MemoryRecord>>> {
        (**self).by_entity(entity)
    }

    fn projects(&self) -> BoxFuture<'_, MemoryResult<Vec<String>>> {
        (**self).projects()
    }

    fn bulk_store<'a>(
        &'a self,
        memories: Vec<StoreMemory>,
        batch_size: usize,
    ) -> BoxFuture<'a, BulkStoreReport> {
        (**self).bulk_store(memories, batch_size)
    }
}
