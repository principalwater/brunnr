// SPDX-License-Identifier: Apache-2.0

use futures_util::{future::BoxFuture, FutureExt};

use crate::{
    reciprocal_rank_fusion, MemoryQuery, MemoryRecord, MemoryResult, RrfOptions, SearchHit,
    StoreMemory,
};

/// Pluggable memory backend contract.
///
/// Backends must support storing durable memories, finding relevant memories, hybrid RRF fusion,
/// and deterministic drill-down by `node_id` so summaries stay traceable to ground-truth records.
///
/// ```
/// # use futures_util::{future::BoxFuture, FutureExt};
/// # use mimisbrunnr::{
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
}
