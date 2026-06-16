// SPDX-License-Identifier: Apache-2.0

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use chrono::Utc;
use futures_util::future::BoxFuture;
use serde::{Deserialize, Serialize};

use crate::{
    backfill_directory, chunking::ChunkConfig, files::parse_record, CollectionCompat,
    MemoryBackend, MemoryError, MemoryResult, SqliteVecVectorStore, StoreMemory, TextEmbedder,
    VectorMemoryBackend, VectorMemoryConfig, VectorStore,
};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OkfVerifyReport {
    pub files: usize,
    pub records: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OkfExportReport {
    pub copied_files: usize,
    pub verified_records: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MigrationPlan {
    pub okf_root: PathBuf,
    pub alias: String,
    pub new_collection: String,
    pub retention_days: u32,
    pub config: VectorMemoryConfig,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MigrationReport {
    pub alias: String,
    pub old_collection: Option<String>,
    pub new_collection: String,
    pub scanned: usize,
    pub imported: usize,
    pub skipped_duplicates: usize,
    pub retained_old_collection: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotReport {
    pub collection: String,
    pub snapshot_name: String,
    pub path: PathBuf,
    pub size_bytes: Option<u64>,
    pub checksum: Option<String>,
}

pub trait VectorCollectionAdmin: VectorStore {
    fn active_collection(&self, alias: &str) -> BoxFuture<'_, MemoryResult<Option<String>>>;

    fn swap_alias(
        &self,
        alias: &str,
        old_collection: Option<&str>,
        new_collection: &str,
    ) -> BoxFuture<'_, MemoryResult<()>>;

    fn snapshot_collection(
        &self,
        collection: &str,
        target_dir: &Path,
    ) -> BoxFuture<'_, MemoryResult<SnapshotReport>>;
}

impl<T: VectorCollectionAdmin + ?Sized> VectorCollectionAdmin for &T {
    fn active_collection(&self, alias: &str) -> BoxFuture<'_, MemoryResult<Option<String>>> {
        (**self).active_collection(alias)
    }

    fn swap_alias(
        &self,
        alias: &str,
        old_collection: Option<&str>,
        new_collection: &str,
    ) -> BoxFuture<'_, MemoryResult<()>> {
        (**self).swap_alias(alias, old_collection, new_collection)
    }

    fn snapshot_collection(
        &self,
        collection: &str,
        target_dir: &Path,
    ) -> BoxFuture<'_, MemoryResult<SnapshotReport>> {
        (**self).snapshot_collection(collection, target_dir)
    }
}

pub async fn migrate_okf_bundle<A: VectorCollectionAdmin>(
    admin: &A,
    plan: MigrationPlan,
    embedder: Arc<dyn TextEmbedder>,
) -> MemoryResult<MigrationReport> {
    let mut rebuild_config = plan.config.clone();
    rebuild_config.collection = plan.new_collection.clone();
    let backend = VectorMemoryBackend::with_embedder(admin, rebuild_config, embedder)?;
    let stats = backfill_directory(&backend, &plan.okf_root).await?;
    let old_collection = admin.active_collection(&plan.alias).await?;
    admin
        .swap_alias(&plan.alias, old_collection.as_deref(), &plan.new_collection)
        .await?;
    Ok(MigrationReport {
        alias: plan.alias,
        old_collection: old_collection.clone(),
        new_collection: plan.new_collection,
        scanned: stats.scanned,
        imported: stats.imported,
        skipped_duplicates: stats.skipped_duplicates,
        retained_old_collection: old_collection.is_some(),
    })
}

pub fn default_migration_collection(alias: &str, compat: &CollectionCompat) -> String {
    let model = compat
        .embedding_model
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string();
    format!("{alias}__{model}__{}", compat.dimensions)
}

pub fn verify_okf_bundle(root: impl AsRef<Path>) -> MemoryResult<OkfVerifyReport> {
    let mut paths = Vec::new();
    collect_okf_paths(root.as_ref(), &mut paths)?;
    let mut report = OkfVerifyReport::default();
    for path in paths {
        report.files += 1;
        let text = std::fs::read_to_string(path)?;
        parse_record(&text)?;
        report.records += 1;
    }
    Ok(report)
}

pub fn export_okf_bundle(
    source: impl AsRef<Path>,
    target: impl AsRef<Path>,
) -> MemoryResult<OkfExportReport> {
    let source = source.as_ref();
    let target = target.as_ref();
    let verify = verify_okf_bundle(source)?;
    let mut paths = Vec::new();
    collect_export_paths(source, &mut paths)?;
    for path in &paths {
        let relative = path
            .strip_prefix(source)
            .map_err(|error| MemoryError::InvalidFile(error.to_string()))?;
        let target_path = target.join(relative);
        if let Some(parent) = target_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::copy(path, target_path)?;
    }
    Ok(OkfExportReport {
        copied_files: paths.len(),
        verified_records: verify.records,
    })
}

pub fn migration_manifest_path(root: impl AsRef<Path>) -> PathBuf {
    root.as_ref().join(format!(
        "brunnr-migration-{}.json",
        Utc::now().format("%Y%m%d%H%M%S")
    ))
}

fn collect_okf_paths(dir: &Path, paths: &mut Vec<PathBuf>) -> MemoryResult<()> {
    if !dir.exists() {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_okf_paths(&path, paths)?;
        } else if path.extension().is_some_and(|extension| extension == "md")
            && !is_reserved_okf_file(&path)
        {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(())
}

fn collect_export_paths(dir: &Path, paths: &mut Vec<PathBuf>) -> MemoryResult<()> {
    if !dir.exists() {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_export_paths(&path, paths)?;
        } else if path.extension().is_some_and(|extension| {
            extension.eq_ignore_ascii_case("md") || extension.eq_ignore_ascii_case("json")
        }) {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(())
}

fn is_reserved_okf_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| matches!(name, "index.md" | "log.md"))
}

/// Report returned by [`rechunk_oversized_sqlite`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RechunkReport {
    /// Total records scanned.
    pub scanned: usize,
    /// Records that were oversized (content > chunk_max_chars) and lacked `parent_node`.
    pub oversized: usize,
    /// Oversized records that were successfully re-stored as chunks and deleted.
    pub rechunked: usize,
}

/// Re-chunk oversized whole-file records in a SQLite-vec collection.
///
/// Scans all records in `collection`. For each record whose content exceeds
/// `ChunkConfig::default().max_chars` and that has no `parent_node` metadata
/// (i.e. was stored before chunking was introduced), re-stores the content via
/// `backend.store()` (which now splits into bounded chunks) then deletes the
/// original oversized record.
///
/// This is the sqlite-vec migration path. For Qdrant, rebuild via `migrate_okf_bundle`.
pub async fn rechunk_oversized_sqlite(
    store: &SqliteVecVectorStore,
    backend: &(impl MemoryBackend + ?Sized),
    collection: &str,
) -> MemoryResult<RechunkReport> {
    let max_chars = ChunkConfig::default().max_chars;
    let payloads = store.scan_all_records(collection)?;
    let mut report = RechunkReport {
        scanned: payloads.len(),
        ..RechunkReport::default()
    };

    let mut to_delete: Vec<String> = Vec::new();

    for payload in payloads {
        let content = payload
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or_default();

        let has_parent = payload
            .get("metadata")
            .and_then(|m| m.get("parent_node"))
            .is_some();

        if content.len() <= max_chars || has_parent {
            continue;
        }

        report.oversized += 1;

        let id = payload
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();

        let tags: Vec<String> = payload
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|t| t.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();

        let metadata: std::collections::BTreeMap<String, String> = payload
            .get("metadata")
            .and_then(|v| v.as_object())
            .map(|obj| {
                obj.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
            .unwrap_or_default();

        let mut mem = StoreMemory::atom(content);
        mem.tags = tags;
        mem.metadata = metadata;

        backend.store(mem).await?;
        to_delete.push(id);
        report.rechunked += 1;
    }

    let ids: Vec<&str> = to_delete.iter().map(String::as_str).collect();
    store.delete_records(collection, &ids)?;

    Ok(report)
}
