// SPDX-License-Identifier: Apache-2.0

//! Semantic query cache for retrieval hardening.
//!
//! A semantic cache serves a query from a previous *similar* query's results instead of
//! re-running search, keyed by embedding cosine similarity rather than exact text match. This
//! cuts repeated embedding + ANN work for paraphrased or re-asked queries — common in agent
//! loops that recall around the same task.
//!
//! [`SemanticCache`] is the pure primitive (bounded, LRU, optional TTL), testable without any
//! embedder. [`CachingMemoryBackend`] wraps any [`MemoryBackend`] with a [`QueryVectorizer`];
//! under feature `vector` the existing `TextEmbedder` is adapted to `QueryVectorizer` in a few
//! lines. Writes clear the cache, so reads stay consistent with append-mostly memory.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use futures_util::{future::BoxFuture, FutureExt};

use crate::{MemoryBackend, MemoryQuery, MemoryRecord, MemoryResult, SearchHit, StoreMemory};

/// Produces a query embedding for semantic-cache keying.
pub trait QueryVectorizer: Send + Sync {
    fn vectorize(&self, text: &str) -> Vec<f32>;
}

struct CacheEntry {
    embedding: Vec<f32>,
    limit: usize,
    hits: Vec<SearchHit>,
    inserted: Instant,
}

/// Bounded, LRU semantic cache of query results keyed by embedding similarity.
pub struct SemanticCache {
    entries: Mutex<VecDeque<CacheEntry>>,
    capacity: usize,
    min_similarity: f32,
    ttl: Option<Duration>,
}

impl SemanticCache {
    /// `capacity` caps stored entries (LRU eviction); `min_similarity` is the cosine threshold
    /// above which a stored query counts as a hit.
    pub fn new(capacity: usize, min_similarity: f32) -> Self {
        Self {
            entries: Mutex::new(VecDeque::new()),
            capacity: capacity.max(1),
            min_similarity,
            ttl: None,
        }
    }

    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    /// Return cached hits for a query whose embedding is within `min_similarity` of a stored
    /// entry that covers `limit` results and has not expired. On a hit the entry is moved to
    /// the front (most-recently-used) and the result is truncated to `limit`.
    pub fn lookup(&self, embedding: &[f32], limit: usize) -> Option<Vec<SearchHit>> {
        let mut entries = self.entries.lock().ok()?;
        self.expire_locked(&mut entries);
        let now = Instant::now();
        let mut best: Option<(usize, f32)> = None;
        for (index, entry) in entries.iter().enumerate() {
            if entry.limit < limit {
                continue;
            }
            if let Some(ttl) = self.ttl {
                if now.duration_since(entry.inserted) > ttl {
                    continue;
                }
            }
            let similarity = cosine_similarity(embedding, &entry.embedding);
            if similarity >= self.min_similarity && best.is_none_or(|(_, b)| similarity > b) {
                best = Some((index, similarity));
            }
        }
        let (index, _) = best?;
        let entry = entries.remove(index)?;
        let hits = entry.hits.iter().take(limit).cloned().collect();
        entries.push_front(entry);
        Some(hits)
    }

    /// Store `hits` for a query embedding, evicting the least-recently-used entry past capacity.
    pub fn insert(&self, embedding: Vec<f32>, limit: usize, hits: Vec<SearchHit>) {
        let Ok(mut entries) = self.entries.lock() else {
            return;
        };
        entries.push_front(CacheEntry {
            embedding,
            limit,
            hits,
            inserted: Instant::now(),
        });
        while entries.len() > self.capacity {
            entries.pop_back();
        }
    }

    pub fn clear(&self) {
        if let Ok(mut entries) = self.entries.lock() {
            entries.clear();
        }
    }

    pub fn len(&self) -> usize {
        self.entries
            .lock()
            .map(|entries| entries.len())
            .unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn expire_locked(&self, entries: &mut VecDeque<CacheEntry>) {
        if let Some(ttl) = self.ttl {
            let now = Instant::now();
            entries.retain(|entry| now.duration_since(entry.inserted) <= ttl);
        }
    }
}

/// Cosine similarity of two equal-length vectors; `0.0` for empty, mismatched, or zero vectors.
pub fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    if left.is_empty() || left.len() != right.len() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut left_norm = 0.0f32;
    let mut right_norm = 0.0f32;
    for (a, b) in left.iter().zip(right.iter()) {
        dot += a * b;
        left_norm += a * a;
        right_norm += b * b;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        return 0.0;
    }
    dot / (left_norm.sqrt() * right_norm.sqrt())
}

/// A [`MemoryBackend`] decorator that serves `find` from a [`SemanticCache`] on a similar
/// prior query. Only plain text queries are cached; queries carrying a `node_id` or tenancy
/// filter bypass the cache (their results depend on more than the query text). `store` clears
/// the cache so a new write is never hidden by a stale cached read.
pub struct CachingMemoryBackend<B, V> {
    inner: B,
    vectorizer: V,
    cache: SemanticCache,
}

impl<B, V> CachingMemoryBackend<B, V> {
    pub fn new(inner: B, vectorizer: V, cache: SemanticCache) -> Self {
        Self {
            inner,
            vectorizer,
            cache,
        }
    }

    pub fn cache(&self) -> &SemanticCache {
        &self.cache
    }
}

fn is_cacheable(query: &MemoryQuery) -> bool {
    query.node_id.is_none()
        && query.scope.is_none()
        && query.agent_id.is_none()
        && query.session_id.is_none()
        && query.task_id.is_none()
        && query.user_id.is_none()
        && query.tags.is_empty()
}

impl<B: MemoryBackend, V: QueryVectorizer> MemoryBackend for CachingMemoryBackend<B, V> {
    fn find(&self, query: MemoryQuery) -> BoxFuture<'_, MemoryResult<Vec<SearchHit>>> {
        if !is_cacheable(&query) {
            return self.inner.find(query);
        }
        let embedding = self.vectorizer.vectorize(&query.text);
        let limit = query.limit;
        if let Some(hits) = self.cache.lookup(&embedding, limit) {
            return async move { Ok(hits) }.boxed();
        }
        async move {
            let hits = self.inner.find(query).await?;
            self.cache.insert(embedding, limit, hits.clone());
            Ok(hits)
        }
        .boxed()
    }

    fn store(&self, memory: StoreMemory) -> BoxFuture<'_, MemoryResult<MemoryRecord>> {
        async move {
            let record = self.inner.store(memory).await?;
            self.cache.clear();
            Ok(record)
        }
        .boxed()
    }

    fn get_node(&self, node_id: &str) -> BoxFuture<'_, MemoryResult<Option<MemoryRecord>>> {
        self.inner.get_node(node_id)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use crate::{MemoryId, MemoryTier, SearchSource};

    use super::*;

    fn hit(id: &str) -> SearchHit {
        SearchHit {
            record: MemoryRecord::new(
                MemoryId::new(id),
                id,
                format!("content {id}"),
                Vec::new(),
                BTreeMap::new(),
                MemoryTier::L1Atom,
            ),
            score: 1.0,
            source: SearchSource::Keyword,
        }
    }

    #[test]
    fn lookup_hits_on_similar_vector_and_misses_below_threshold() {
        let cache = SemanticCache::new(8, 0.9);
        cache.insert(vec![1.0, 0.0, 0.0], 5, vec![hit("a")]);
        // Near-parallel vector -> hit.
        let near = cache.lookup(&[0.99, 0.01, 0.0], 5);
        assert!(near.is_some());
        // Orthogonal vector -> miss.
        assert!(cache.lookup(&[0.0, 1.0, 0.0], 5).is_none());
    }

    #[test]
    fn lookup_respects_limit_coverage_and_truncates() {
        let cache = SemanticCache::new(8, 0.9);
        cache.insert(vec![1.0, 0.0], 3, vec![hit("a"), hit("b"), hit("c")]);
        // Asking for fewer than cached -> hit, truncated.
        let some = cache.lookup(&[1.0, 0.0], 2).expect("hit");
        assert_eq!(some.len(), 2);
        // Asking for more than cached -> miss.
        assert!(cache.lookup(&[1.0, 0.0], 5).is_none());
    }

    #[test]
    fn capacity_evicts_least_recently_used() {
        let cache = SemanticCache::new(2, 0.99);
        cache.insert(vec![1.0, 0.0], 1, vec![hit("a")]);
        cache.insert(vec![0.0, 1.0], 1, vec![hit("b")]);
        // Touch "a" so it becomes most-recently-used.
        assert!(cache.lookup(&[1.0, 0.0], 1).is_some());
        // Insert a third -> evicts the LRU ("b").
        cache.insert(vec![1.0, 1.0], 1, vec![hit("c")]);
        assert!(cache.lookup(&[0.0, 1.0], 1).is_none(), "b evicted");
        assert!(cache.lookup(&[1.0, 0.0], 1).is_some(), "a retained");
    }

    #[test]
    fn ttl_expires_entries() {
        let cache = SemanticCache::new(8, 0.9).with_ttl(Duration::from_millis(20));
        cache.insert(vec![1.0, 0.0], 1, vec![hit("a")]);
        assert!(cache.lookup(&[1.0, 0.0], 1).is_some());
        std::thread::sleep(Duration::from_millis(40));
        assert!(cache.lookup(&[1.0, 0.0], 1).is_none(), "entry expired");
    }

    struct CountingBackend {
        calls: Arc<AtomicUsize>,
    }

    impl MemoryBackend for CountingBackend {
        fn find(&self, _query: MemoryQuery) -> BoxFuture<'_, MemoryResult<Vec<SearchHit>>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            async move { Ok(vec![hit("x")]) }.boxed()
        }

        fn store(&self, memory: StoreMemory) -> BoxFuture<'_, MemoryResult<MemoryRecord>> {
            async move {
                Ok(MemoryRecord::new(
                    MemoryId::new("new"),
                    "new",
                    memory.content,
                    Vec::new(),
                    BTreeMap::new(),
                    MemoryTier::L1Atom,
                ))
            }
            .boxed()
        }

        fn get_node(&self, _node_id: &str) -> BoxFuture<'_, MemoryResult<Option<MemoryRecord>>> {
            async move { Ok(None) }.boxed()
        }
    }

    /// Identical query text always maps to the same vector.
    struct IdentityVectorizer;
    impl QueryVectorizer for IdentityVectorizer {
        fn vectorize(&self, text: &str) -> Vec<f32> {
            // A trivial deterministic embedding: char-code sums in 4 buckets.
            let mut buckets = [0.0f32; 4];
            for (index, byte) in text.bytes().enumerate() {
                buckets[index % 4] += byte as f32;
            }
            buckets.to_vec()
        }
    }

    #[tokio::test]
    async fn caching_backend_serves_repeat_query_and_invalidates_on_store() {
        let calls = Arc::new(AtomicUsize::new(0));
        let backend = CountingBackend {
            calls: calls.clone(),
        };
        let caching =
            CachingMemoryBackend::new(backend, IdentityVectorizer, SemanticCache::new(16, 0.999));

        let _ = caching.find(MemoryQuery::new("same query")).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        // Repeat -> served from cache, inner not called again.
        let _ = caching.find(MemoryQuery::new("same query")).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        // A write clears the cache.
        let _ = caching.store(StoreMemory::atom("new fact")).await.unwrap();
        let _ = caching.find(MemoryQuery::new("same query")).await.unwrap();
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "store invalidated the cache"
        );
    }

    #[tokio::test]
    async fn caching_backend_bypasses_filtered_queries() {
        let calls = Arc::new(AtomicUsize::new(0));
        let backend = CountingBackend {
            calls: calls.clone(),
        };
        let caching =
            CachingMemoryBackend::new(backend, IdentityVectorizer, SemanticCache::new(16, 0.999));

        let mut query = MemoryQuery::new("filtered");
        query.node_id = Some("node:1".to_string());
        let _ = caching.find(query.clone()).await.unwrap();
        let _ = caching.find(query).await.unwrap();
        // Both bypass the cache.
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert!(caching.cache().is_empty());
    }
}
