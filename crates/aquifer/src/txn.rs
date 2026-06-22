// SPDX-License-Identifier: Apache-2.0

//! Transactional multi-writer substrate — optimistic concurrency for agent memory.
//!
//! ## Why this exists
//!
//! The Oracle analysis (*File Systems vs Databases for Agent Memory*) and Cursor's agent-scaling
//! post both document the same failure mode: naive file coordination silently corrupts under
//! concurrent writes; the moment you add locking + atomic writes + indexing you are rebuilding a
//! database. Cursor moved 20 agents from the throughput of 2–3 (flat file-lock contention) to full
//! concurrency by switching to **optimistic concurrency** — read free, write fails if state changed.
//!
//! This module implements that model as a thin, additive layer over any [`MemoryBackend`]:
//!
//! - **[`CommitLog`]** — a monotonic `u64` sequence number per memory scope. Reads are free (no
//!   lock, no contention). Writes succeed only when the caller presents the sequence number they
//!   observed before the write — if anyone else committed between the read and the write, the write
//!   fails with [`TxnError::Conflict`] and the caller retries. This is CAS at the memory-substrate
//!   level.
//!
//! - **[`TransactionalMemory`]** — wraps any `B: MemoryBackend`, adding [`begin_write`] +
//!   [`commit`] CAS semantics on top of the existing `store` / `find` API. The underlying backend
//!   handles durability and retrieval; the commit log governs sequential consistency.
//!
//! - **[`sync_okf_directory`]** — re-indexes every OKF markdown file in a directory as a
//!   transactional write. This makes a human edit to an OKF file a first-class memory transaction:
//!   edit the file → call `sync_okf_directory` (or run the file-watcher daemon) → the new
//!   content is immediately retrievable.
//!
//! ## Architecture note
//!
//! The commit log does NOT replace the underlying vector store's own serialization guarantees. It
//! adds an explicit, application-visible concurrency control layer so agents can reason about
//! "what sequence was the memory in when I started this write?" rather than relying on opaque DB
//! internals. The two layers compose: the DB prevents corruption; the commit log prevents logical
//! races (two agents updating the "current plan" simultaneously).

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use crate::{
    backfill::{collect_memory_paths, parse_memory_path},
    MemoryBackend, MemoryError, MemoryQuery, MemoryRecord, MemoryResult, SearchHit, StoreMemory,
};
use futures_util::future::BoxFuture;

/// A monotonic sequence number for a memory scope.
pub type TxnSeq = u64;

/// Error returned when a CAS write fails because the sequence changed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TxnError {
    /// Another writer committed between the caller's read and write.
    Conflict { expected: TxnSeq, actual: TxnSeq },
    /// Underlying storage error.
    Storage(String),
}

impl std::fmt::Display for TxnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Conflict { expected, actual } => write!(
                f,
                "transaction conflict: expected seq {expected}, actual {actual}"
            ),
            Self::Storage(msg) => write!(f, "storage error: {msg}"),
        }
    }
}

impl std::error::Error for TxnError {}

/// An in-process commit log backed by an atomic `u64`.
///
/// For multi-process or multi-host deployments, replace this with a DB-backed sequence (e.g. a
/// SQLite row or a Qdrant payload field) — the trait contract is the same.
#[derive(Debug, Clone, Default)]
pub struct CommitLog {
    seq: Arc<AtomicU64>,
}

impl CommitLog {
    pub fn new() -> Self {
        Self::default()
    }

    /// Read the current sequence number — free, no lock.
    pub fn current_seq(&self) -> TxnSeq {
        self.seq.load(Ordering::Acquire)
    }

    /// Attempt a CAS commit: succeed iff `current_seq() == expected_seq`.
    ///
    /// On success returns the **new** sequence number. On failure returns
    /// [`TxnError::Conflict`] with the expected and actual values so the caller
    /// can retry with a fresh read.
    pub fn try_commit(&self, expected_seq: TxnSeq) -> Result<TxnSeq, TxnError> {
        match self.seq.compare_exchange(
            expected_seq,
            expected_seq + 1,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(old) => Ok(old + 1),
            Err(actual) => Err(TxnError::Conflict {
                expected: expected_seq,
                actual,
            }),
        }
    }
}

/// A [`MemoryBackend`] wrapper with optimistic-concurrency CAS semantics.
///
/// ```no_run
/// # use std::sync::Arc;
/// # use aquifer::FilesBackend;
/// # use aquifer::txn::{TransactionalMemory, TxnError};
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let backend = Arc::new(FilesBackend::new("/tmp/memory"));
/// let txn = TransactionalMemory::new(backend);
///
/// // Begin: capture the current sequence — the "snapshot" for this write.
/// let (seq, mem) = txn.begin_write(aquifer::StoreMemory::atom("plan: use approach B")).await?;
///
/// // Commit: succeed iff nobody else wrote between begin and now.
/// match txn.commit(seq, mem).await {
///     Ok(_new_seq) => println!("committed"),
///     Err(TxnError::Conflict { expected, actual }) => {
///         println!("conflict: seq {expected} → {actual}, retry");
///     }
///     Err(e) => return Err(e.into()),
/// }
/// # Ok(())
/// # }
/// ```
pub struct TransactionalMemory<B: MemoryBackend> {
    inner: Arc<B>,
    log: CommitLog,
}

impl<B: MemoryBackend> TransactionalMemory<B> {
    pub fn new(backend: Arc<B>) -> Self {
        Self {
            inner: backend,
            log: CommitLog::new(),
        }
    }

    pub fn with_log(backend: Arc<B>, log: CommitLog) -> Self {
        Self {
            inner: backend,
            log,
        }
    }

    /// Current commit-log sequence — free read.
    pub fn current_seq(&self) -> TxnSeq {
        self.log.current_seq()
    }

    /// Begin a write: snapshot the current sequence. The caller proceeds to prepare the write
    /// value and then calls [`commit`](Self::commit) with the snapshotted seq.
    pub async fn begin_write(&self, memory: StoreMemory) -> MemoryResult<(TxnSeq, StoreMemory)> {
        Ok((self.log.current_seq(), memory))
    }

    /// Commit a write: CAS on the sequence, then store to the underlying backend.
    ///
    /// Returns the new sequence on success, [`TxnError::Conflict`] if another writer committed
    /// between `begin_write` and `commit`.
    pub async fn commit(
        &self,
        expected_seq: TxnSeq,
        memory: StoreMemory,
    ) -> Result<TxnSeq, TxnError> {
        let new_seq = self.log.try_commit(expected_seq)?;
        self.inner
            .store(memory)
            .await
            .map_err(|e| TxnError::Storage(e.to_string()))
            .map(|_| new_seq)
    }

    /// Convenience: commit with automatic retry on conflict (up to `max_retries`).
    pub async fn commit_with_retry(
        &self,
        memory: StoreMemory,
        max_retries: usize,
    ) -> Result<TxnSeq, TxnError> {
        let mut attempts = 0;
        loop {
            let (seq, mem) = self
                .begin_write(memory.clone())
                .await
                .map_err(|e| TxnError::Storage(e.to_string()))?;
            match self.commit(seq, mem).await {
                Ok(new_seq) => return Ok(new_seq),
                Err(TxnError::Conflict { .. }) if attempts < max_retries => {
                    attempts += 1;
                    tokio::task::yield_now().await;
                }
                Err(e) => return Err(e),
            }
        }
    }

    pub fn inner(&self) -> &B {
        &self.inner
    }
}

impl<B: MemoryBackend> MemoryBackend for TransactionalMemory<B> {
    fn store(&self, memory: StoreMemory) -> BoxFuture<'_, MemoryResult<MemoryRecord>> {
        use futures_util::FutureExt;
        async move {
            let seq = self.log.current_seq();
            self.log
                .try_commit(seq)
                .map_err(|e| MemoryError::Database(e.to_string()))?;
            self.inner.store(memory).await
        }
        .boxed()
    }

    fn find(&self, query: MemoryQuery) -> BoxFuture<'_, MemoryResult<Vec<SearchHit>>> {
        self.inner.find(query)
    }

    fn get_node(&self, node_id: &str) -> BoxFuture<'_, MemoryResult<Option<MemoryRecord>>> {
        self.inner.get_node(node_id)
    }

    fn neighbors(
        &self,
        node_id: &str,
        hops: usize,
    ) -> BoxFuture<'_, MemoryResult<Vec<MemoryRecord>>> {
        self.inner.neighbors(node_id, hops)
    }

    fn by_entity(&self, entity: &str) -> BoxFuture<'_, MemoryResult<Vec<MemoryRecord>>> {
        self.inner.by_entity(entity)
    }
}

/// Report from a directory sync.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SyncReport {
    /// Number of files scanned.
    pub files_scanned: usize,
    /// Number of records successfully re-indexed.
    pub records_indexed: usize,
    /// Number of files that failed to parse.
    pub parse_failures: usize,
}

/// Re-index every OKF markdown file in `dir` as a transactional write to `backend`.
///
/// This is the "human file edit = transaction" primitive: run this after editing an OKF file
/// and the new content is immediately retrievable through `memory.find`. A periodic cron or a
/// file-watcher daemon calls this at whatever cadence fits the deployment.
pub async fn sync_okf_directory(
    dir: &std::path::Path,
    backend: &dyn MemoryBackend,
) -> MemoryResult<SyncReport> {
    let paths = collect_memory_paths(dir)?;
    let mut report = SyncReport::default();
    for path in paths {
        report.files_scanned += 1;
        match parse_memory_path(&path) {
            Ok(memories) => {
                for memory in memories {
                    backend.store(memory).await?;
                    report.records_indexed += 1;
                }
            }
            Err(_) => {
                report.parse_failures += 1;
            }
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FilesBackend;
    use artesian_test_support::TempDir;

    #[test]
    fn commit_log_cas_detects_conflict() {
        let log = CommitLog::new();
        assert_eq!(log.current_seq(), 0);

        let seq = log.current_seq();
        // First writer commits at seq 0 → new seq is 1.
        assert_eq!(log.try_commit(seq).unwrap(), 1);
        assert_eq!(log.current_seq(), 1);

        // Second writer tried to commit with the old seq → conflict.
        let err = log.try_commit(seq).unwrap_err();
        assert!(matches!(
            err,
            TxnError::Conflict {
                expected: 0,
                actual: 1
            }
        ));
    }

    #[test]
    fn commit_log_sequential_commits_increment() {
        let log = CommitLog::new();
        for expected in 0u64..5 {
            let new_seq = log
                .try_commit(expected)
                .expect("sequential commit should succeed");
            assert_eq!(new_seq, expected + 1);
        }
        assert_eq!(log.current_seq(), 5);
    }

    #[tokio::test]
    async fn transactional_memory_serializes_concurrent_writes() {
        let tempdir = TempDir::new("txn-memory");
        let backend = Arc::new(FilesBackend::new(tempdir.path()));
        let txn = Arc::new(TransactionalMemory::new(backend));

        let mut handles = Vec::new();
        for i in 0..8 {
            let txn = Arc::clone(&txn);
            handles.push(tokio::spawn(async move {
                txn.commit_with_retry(StoreMemory::atom(format!("concurrent write {i}")), 16)
                    .await
            }));
        }

        let mut successes = 0;
        for handle in handles {
            if handle.await.expect("join").is_ok() {
                successes += 1;
            }
        }
        assert_eq!(successes, 8, "all 8 writers should commit with retry");
        assert_eq!(txn.current_seq(), 8, "commit log should advance to 8");
    }

    #[tokio::test]
    async fn transactional_memory_detects_conflict_without_retry() {
        let tempdir = TempDir::new("txn-conflict");
        let backend = Arc::new(FilesBackend::new(tempdir.path()));
        let txn = TransactionalMemory::new(backend);

        // Both writers read seq=0.
        let (seq_a, mem_a) = txn
            .begin_write(StoreMemory::atom("writer A"))
            .await
            .unwrap();
        let (seq_b, mem_b) = txn
            .begin_write(StoreMemory::atom("writer B"))
            .await
            .unwrap();
        assert_eq!(seq_a, 0);
        assert_eq!(seq_b, 0);

        // Writer A commits first.
        let new_seq = txn
            .commit(seq_a, mem_a)
            .await
            .expect("first commit succeeds");
        assert_eq!(new_seq, 1);

        // Writer B now sees a conflict.
        let err = txn.commit(seq_b, mem_b).await.unwrap_err();
        assert!(
            matches!(
                err,
                TxnError::Conflict {
                    expected: 0,
                    actual: 1
                }
            ),
            "B should see conflict, got {err}"
        );
    }

    #[tokio::test]
    async fn sync_okf_directory_re_indexes_edited_files() {
        let tempdir = TempDir::new("okf-sync");
        std::fs::write(
            tempdir.join("plan.md"),
            r#"---
type: decision
title: Use approach B
timestamp: 2026-06-20T00:00:00Z
node_id: node:plan-b
tier: l1-atom
---

Use approach B after approach A failed in session 1.
"#,
        )
        .expect("write fixture");

        let backend = FilesBackend::new(tempdir.path());
        let report = sync_okf_directory(tempdir.path(), &backend)
            .await
            .expect("sync should succeed");

        assert!(report.files_scanned >= 1, "should scan at least 1 file");
        assert!(
            report.records_indexed >= 1,
            "should index at least 1 record"
        );
        assert_eq!(report.parse_failures, 0, "no parse failures expected");
    }
}
