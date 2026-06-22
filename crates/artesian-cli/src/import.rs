// SPDX-License-Identifier: Apache-2.0

use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::Result;
use aquifer::{
    collect_memory_paths, parse_memory_path, stable_memory_id, BackfillFailure, FilesBackend,
    MemoryBackend, MemoryScope, StoreMemory,
};
use headrace::{FilesTaskStore, VectorTaskStore};
use serde::Serialize;

#[derive(Debug, Clone, Default, Serialize)]
pub struct ImportReport {
    pub scanned: usize,
    pub memory_imported: usize,
    pub memory_skipped_duplicates: usize,
    pub task_imported: usize,
    pub task_skipped_duplicates: usize,
    pub failed: Vec<BackfillFailure>,
    pub index_path: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct ImportOptions {
    pub directory: PathBuf,
    pub okf_root: PathBuf,
    pub user_id: Option<String>,
    /// Emit per-file progress to stderr (stdout stays reserved for the machine-readable summary).
    pub progress: bool,
}

#[derive(Debug, Clone)]
struct CatalogEntry {
    kind: CatalogKind,
    path: String,
    title: String,
    chunks: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CatalogKind {
    Memory,
    Task,
}

impl CatalogKind {
    const fn heading(self) -> &'static str {
        match self {
            Self::Memory => "Memory",
            Self::Task => "Tasks",
        }
    }
}

pub async fn import_directory(
    options: ImportOptions,
    primary_memory: Arc<dyn MemoryBackend>,
    write_okf_copy: bool,
    task_store: &VectorTaskStore,
) -> Result<ImportReport> {
    let mut paths = collect_memory_paths(&options.directory)?;
    paths.sort();

    let okf_memory = write_okf_copy.then(|| Arc::new(FilesBackend::new(&options.okf_root)));
    let mut report = ImportReport::default();
    let mut catalog = Vec::new();

    let total = paths.len();
    if options.progress {
        eprintln!(
            "importing {total} file(s) from {}",
            options.directory.display()
        );
    }

    for (idx, path) in paths.iter().enumerate() {
        report.scanned += 1;
        let before_imported = report.memory_imported + report.task_imported;
        let before_dups = report.memory_skipped_duplicates + report.task_skipped_duplicates;
        let before_failed = report.failed.len();
        let is_task = FilesTaskStore::is_task_like_path(path);
        if is_task {
            import_task_path(
                &options.directory,
                path,
                task_store,
                &mut report,
                &mut catalog,
            )
            .await;
        } else {
            import_memory_path(
                &options.directory,
                path,
                primary_memory.as_ref(),
                okf_memory
                    .as_deref()
                    .map(|backend| backend as &dyn MemoryBackend),
                options.user_id.as_deref(),
                &mut report,
                &mut catalog,
            )
            .await;
        }
        if options.progress {
            let imported = (report.memory_imported + report.task_imported) - before_imported;
            let dups =
                (report.memory_skipped_duplicates + report.task_skipped_duplicates) - before_dups;
            let outcome = if report.failed.len() > before_failed {
                let reason = report
                    .failed
                    .last()
                    .map(|failure| failure.reason.as_str())
                    .unwrap_or("error");
                format!("FAILED: {}", reason.chars().take(100).collect::<String>())
            } else if is_task {
                if imported > 0 {
                    "task imported".to_string()
                } else {
                    "task (duplicate)".to_string()
                }
            } else if imported > 0 && dups > 0 {
                format!("{imported} imported, {dups} duplicate")
            } else if imported > 0 {
                format!("{imported} imported")
            } else if dups > 0 {
                format!("{dups} duplicate")
            } else {
                "no records".to_string()
            };
            eprintln!(
                "[{}/{}] {} — {}",
                idx + 1,
                total,
                catalog_path(&options.directory, path),
                outcome
            );
        }
    }

    if !catalog.is_empty() {
        report.index_path = Some(write_index(&options.okf_root, &catalog)?);
    }

    Ok(report)
}

async fn import_task_path(
    source_root: &Path,
    path: &Path,
    task_store: &VectorTaskStore,
    report: &mut ImportReport,
    catalog: &mut Vec<CatalogEntry>,
) {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) => {
            push_failure(report, path, error);
            return;
        }
    };
    let task = match FilesTaskStore::parse_task_like_markdown(path, &text) {
        Ok(task) => task,
        Err(error) => {
            push_failure(report, path, error);
            return;
        }
    };
    match task_store.import_task(task).await {
        Ok(outcome) => {
            if outcome.imported() {
                report.task_imported += 1;
            } else {
                report.task_skipped_duplicates += 1;
            }
            catalog.push(CatalogEntry {
                kind: CatalogKind::Task,
                path: catalog_path(source_root, path),
                title: outcome.task().title.clone(),
                chunks: 1,
            });
        }
        Err(error) => push_failure(report, path, error),
    }
}

async fn import_memory_path(
    source_root: &Path,
    path: &Path,
    primary_memory: &dyn MemoryBackend,
    okf_memory: Option<&dyn MemoryBackend>,
    user_id: Option<&str>,
    report: &mut ImportReport,
    catalog: &mut Vec<CatalogEntry>,
) {
    let memories = match parse_memory_path(path) {
        Ok(memories) => memories,
        Err(error) => {
            push_failure(report, path, error);
            return;
        }
    };
    let chunk_count = memories.len();
    let title = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .map(|stem| stem.replace(['-', '_'], " "))
        .unwrap_or_else(|| "Imported memory".to_string());

    for memory in memories {
        let memory = with_user(memory, user_id);
        if let Some(okf_memory) = okf_memory {
            if let Err(error) = store_if_missing(okf_memory, memory.clone()).await {
                push_failure(report, path, error);
                continue;
            }
        }
        match store_if_missing(primary_memory, memory).await {
            Ok(StoreOutcome::Imported) => report.memory_imported += 1,
            Ok(StoreOutcome::SkippedDuplicate) => report.memory_skipped_duplicates += 1,
            Err(error) => push_failure(report, path, error),
        }
    }

    catalog.push(CatalogEntry {
        kind: CatalogKind::Memory,
        path: catalog_path(source_root, path),
        title,
        chunks: chunk_count,
    });
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StoreOutcome {
    Imported,
    SkippedDuplicate,
}

async fn store_if_missing(
    backend: &dyn MemoryBackend,
    memory: StoreMemory,
) -> Result<StoreOutcome, aquifer::MemoryError> {
    let id = stable_memory_id(&memory);
    if backend.get_node(id.as_str()).await?.is_some() {
        return Ok(StoreOutcome::SkippedDuplicate);
    }
    backend.store(memory).await?;
    Ok(StoreOutcome::Imported)
}

fn with_user(mut memory: StoreMemory, user_id: Option<&str>) -> StoreMemory {
    if let Some(user_id) = user_id {
        if memory.user_id.is_none() {
            memory.user_id = Some(user_id.to_string());
        }
        if memory.scope.is_none() {
            memory.scope = Some(MemoryScope::Shared);
        }
    }
    memory
}

fn write_index(root: &Path, catalog: &[CatalogEntry]) -> Result<PathBuf> {
    let memory_dir = root.join("memory");
    fs::create_dir_all(&memory_dir)?;
    let path = memory_dir.join("index.md");
    let mut output = String::from(
        "---\ntype: index\ntitle: Artesian Memory Index\n---\n\n# Artesian Memory Index\n\nRead this catalog first, then drill into the listed OKF records or task files as needed.\n",
    );

    for kind in [CatalogKind::Memory, CatalogKind::Task] {
        let entries = catalog
            .iter()
            .filter(|entry| entry.kind == kind)
            .collect::<Vec<_>>();
        if entries.is_empty() {
            continue;
        }
        output.push_str(&format!("\n## {}\n\n", kind.heading()));
        for entry in entries {
            output.push_str(&format!(
                "- `{}` — {} (chunks: {})\n",
                entry.path, entry.title, entry.chunks
            ));
        }
    }

    fs::write(&path, output)?;
    Ok(path)
}

fn catalog_path(source_root: &Path, path: &Path) -> String {
    path.strip_prefix(source_root)
        .unwrap_or(path)
        .display()
        .to_string()
}

fn push_failure(report: &mut ImportReport, path: &Path, error: impl std::fmt::Display) {
    report.failed.push(BackfillFailure {
        file: path.to_path_buf(),
        reason: error.to_string(),
    });
}
