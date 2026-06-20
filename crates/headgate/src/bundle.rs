// SPDX-License-Identifier: Apache-2.0
//! Portable working-context bundle — the committed-state + lifecycle LAYER.
//!
//! Most agent-memory formats standardize the memory *unit* (facts / records). This module
//! standardizes the layer none of them do: a portable snapshot of the agent's **committed
//! working context** — what it holds in force right now, as bounded, typed slots with a
//! saturation budget — plus a declarative **lifecycle log** of what is active / superseded /
//! deprecated and *why*. Together they let an agent (any model, any runtime) *resume* work,
//! not just retrieve facts.
//!
//! The bundle is deliberately memory-unit-agnostic. It does **not** re-define the unit layer;
//! it *references* it (`unit_source` / `unit_ref`), so a bundle composes with existing unit
//! formats (e.g. Portable AI Memory, AMP, plain files) rather than competing with them.
//!
//! The on-disk shape is documented in `docs/kit-format.md`; this module is its reference
//! implementation. Types are name-agnostic on purpose so the layer can later be promoted to a
//! standalone open specification without code churn.

use std::path::Path;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::ccs::{CcsSchema, CommittedContextState, CommittedEntry};
use crate::metrics::count_tokens;

/// Stable format identifier written into every manifest.
pub const BUNDLE_FORMAT: &str = "artesian.working-context";
/// Format version. Readers accept any bundle sharing this major version (`0`).
pub const BUNDLE_VERSION: &str = "0.1";

const MANIFEST_FILE: &str = "manifest.json";
const SNAPSHOT_FILE: &str = "snapshot.json";
const LIFECYCLE_FILE: &str = "lifecycle.jsonl";
const SNAPSHOT_MD_FILE: &str = "snapshot.md";

/// Errors reading, writing, or validating a working-context bundle.
#[derive(Debug, thiserror::Error)]
pub enum BundleError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unrecognized bundle format: {0}")]
    Format(String),
    #[error("incompatible bundle version: {0}")]
    Version(String),
    #[error("snapshot token_count {declared} disagrees with summed entries {actual}")]
    TokenMismatch { declared: usize, actual: usize },
}

/// How a snapshot entry's content is currently represented (after ClawVM's multi-resolution
/// pages): the full text, a token-reduced summary, or a resolvable pointer into the unit layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Resolution {
    /// Verbatim content.
    #[default]
    Full,
    /// Token-reduced summary; the full text is recoverable from the unit layer.
    Compressed,
    /// Only a reference; content lives in the unit store (`unit_ref`).
    Pointer,
}

/// One committed entry in the working-context snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SnapshotEntry {
    pub id: String,
    /// Schema slot this entry is filed under.
    pub slot: String,
    /// Current representation (may be empty when `resolution = pointer`).
    pub content: String,
    pub tokens: usize,
    pub score: f32,
    #[serde(default)]
    pub resolution: Resolution,
    /// Reference into the unit layer (id / content-hash in the external store), when composed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit_ref: Option<String>,
    pub committed_at: DateTime<Utc>,
}

impl SnapshotEntry {
    /// A full-resolution entry committed now (tokens counted from `content`).
    pub fn now(
        id: impl Into<String>,
        slot: impl Into<String>,
        content: impl Into<String>,
        score: f32,
    ) -> Self {
        let content = content.into();
        let tokens = count_tokens(&content);
        Self {
            id: id.into(),
            slot: slot.into(),
            content,
            tokens,
            score,
            resolution: Resolution::Full,
            unit_ref: None,
            committed_at: Utc::now(),
        }
    }
}

/// The bounded, schema-governed snapshot of the agent's committed working context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkingContextSnapshot {
    /// Slot names, in render order (the schema that governs the committed state).
    pub schema: Vec<String>,
    /// Saturation bound, in tokens.
    pub budget_tokens: usize,
    /// Current footprint, in tokens (equals the sum of entry tokens; checked by `validate`).
    pub token_count: usize,
    pub entries: Vec<SnapshotEntry>,
}

impl WorkingContextSnapshot {
    /// Serialize a live committed state into a portable snapshot.
    pub fn from_ccs(ccs: &CommittedContextState) -> Self {
        let entries = ccs
            .entries()
            .iter()
            .map(|entry| SnapshotEntry {
                id: entry.id.clone(),
                slot: entry.slot.clone(),
                content: entry.content.clone(),
                tokens: entry.tokens,
                score: entry.score,
                resolution: Resolution::Full,
                unit_ref: None,
                committed_at: entry.committed_at,
            })
            .collect();
        Self {
            schema: ccs.schema().slots.clone(),
            budget_tokens: ccs.budget_tokens(),
            token_count: ccs.token_count(),
            entries,
        }
    }

    /// Human-readable mirror, grouped by schema slot (interface ≠ substrate).
    pub fn render_markdown(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        for slot in &self.schema {
            let mut wrote = false;
            for entry in self.entries.iter().filter(|entry| &entry.slot == slot) {
                if !wrote {
                    let _ = writeln!(out, "## {slot}");
                    wrote = true;
                }
                let _ = writeln!(out, "- {}", entry.content);
            }
            if wrote {
                out.push('\n');
            }
        }
        let mut wrote_other = false;
        for entry in self
            .entries
            .iter()
            .filter(|entry| !self.schema.iter().any(|slot| slot == &entry.slot))
        {
            if !wrote_other {
                out.push_str("## other\n");
                wrote_other = true;
            }
            let _ = writeln!(out, "- {}", entry.content);
        }
        out.trim_end().to_string()
    }
}

/// The transition a lifecycle event records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Decision {
    Commit,
    Evict,
    Supersede,
    Deprecate,
}

/// The standing of an entry after a lifecycle event — what an importing runtime trusts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Hypothesis,
    Active,
    Validated,
    Deprecated,
    Superseded,
}

/// The qualify-gate signals behind a commit decision, carried so the importer sees *why*.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LifecycleReason {
    pub relevance: f32,
    pub novelty: f32,
    pub drift: f32,
}

/// One declarative lifecycle decision. The log of these travels *with* the snapshot so an
/// importing agent knows what is in force now and how it got there.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LifecycleEntry {
    pub ts: DateTime<Utc>,
    pub entry_id: String,
    pub decision: Decision,
    pub status: Status,
    /// The entry id this one replaces (for `supersede`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supersedes: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<LifecycleReason>,
}

impl LifecycleEntry {
    /// A plain "this entry was committed and is active" event.
    pub fn commit(entry_id: impl Into<String>) -> Self {
        Self {
            ts: Utc::now(),
            entry_id: entry_id.into(),
            decision: Decision::Commit,
            status: Status::Active,
            supersedes: None,
            reason: None,
        }
    }
}

/// Manifest that ties a bundle together and records which unit layer it composes with.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundleManifest {
    pub format: String,
    pub version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    pub created_at: DateTime<Utc>,
    /// Where the underlying memory units live: `"inline"` (entries carry their own content) or
    /// an external unit format/store this bundle references (e.g. `"pam"`, `"amp"`, `"files"`).
    pub unit_source: String,
    /// Pointer/URI to the external unit store, when `unit_source != "inline"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit_ref: Option<String>,
}

impl BundleManifest {
    pub fn inline() -> Self {
        Self {
            format: BUNDLE_FORMAT.to_string(),
            version: BUNDLE_VERSION.to_string(),
            agent_id: None,
            created_at: Utc::now(),
            unit_source: "inline".to_string(),
            unit_ref: None,
        }
    }
}

/// A complete, portable working-context bundle: manifest + snapshot + lifecycle log.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkingContextBundle {
    pub manifest: BundleManifest,
    pub snapshot: WorkingContextSnapshot,
    pub lifecycle: Vec<LifecycleEntry>,
}

impl WorkingContextBundle {
    /// Build an inline bundle from a snapshot and its lifecycle log.
    pub fn new(snapshot: WorkingContextSnapshot, lifecycle: Vec<LifecycleEntry>) -> Self {
        Self {
            manifest: BundleManifest::inline(),
            snapshot,
            lifecycle,
        }
    }

    /// Check the manifest format/version and the snapshot's token accounting.
    pub fn validate(&self) -> Result<(), BundleError> {
        if self.manifest.format != BUNDLE_FORMAT {
            return Err(BundleError::Format(self.manifest.format.clone()));
        }
        if !version_compatible(&self.manifest.version) {
            return Err(BundleError::Version(self.manifest.version.clone()));
        }
        let summed: usize = self.snapshot.entries.iter().map(|entry| entry.tokens).sum();
        if summed != self.snapshot.token_count {
            return Err(BundleError::TokenMismatch {
                declared: self.snapshot.token_count,
                actual: summed,
            });
        }
        Ok(())
    }

    /// Write the bundle as a human-readable directory (manifest.json, snapshot.json,
    /// lifecycle.jsonl, snapshot.md).
    pub fn write_dir(&self, dir: &Path) -> Result<(), BundleError> {
        std::fs::create_dir_all(dir)?;
        std::fs::write(
            dir.join(MANIFEST_FILE),
            serde_json::to_string_pretty(&self.manifest)?,
        )?;
        std::fs::write(
            dir.join(SNAPSHOT_FILE),
            serde_json::to_string_pretty(&self.snapshot)?,
        )?;
        let mut jsonl = String::new();
        for event in &self.lifecycle {
            jsonl.push_str(&serde_json::to_string(event)?);
            jsonl.push('\n');
        }
        std::fs::write(dir.join(LIFECYCLE_FILE), jsonl)?;
        std::fs::write(dir.join(SNAPSHOT_MD_FILE), self.render_markdown())?;
        Ok(())
    }

    /// Read and validate a bundle directory written by [`write_dir`](Self::write_dir).
    pub fn read_dir(dir: &Path) -> Result<Self, BundleError> {
        let manifest: BundleManifest =
            serde_json::from_str(&std::fs::read_to_string(dir.join(MANIFEST_FILE))?)?;
        let snapshot: WorkingContextSnapshot =
            serde_json::from_str(&std::fs::read_to_string(dir.join(SNAPSHOT_FILE))?)?;
        let lifecycle_path = dir.join(LIFECYCLE_FILE);
        let lifecycle = if lifecycle_path.exists() {
            std::fs::read_to_string(&lifecycle_path)?
                .lines()
                .filter(|line| !line.trim().is_empty())
                .map(serde_json::from_str)
                .collect::<Result<Vec<_>, _>>()?
        } else {
            Vec::new()
        };
        let bundle = Self {
            manifest,
            snapshot,
            lifecycle,
        };
        bundle.validate()?;
        Ok(bundle)
    }

    /// Rehydrate the snapshot into a live committed state — the "resume" path.
    pub fn to_ccs(&self) -> CommittedContextState {
        let mut ccs = CommittedContextState::new(
            CcsSchema::new(self.snapshot.schema.clone()),
            self.snapshot.budget_tokens,
        );
        for entry in &self.snapshot.entries {
            ccs.admit(CommittedEntry {
                id: entry.id.clone(),
                slot: entry.slot.clone(),
                content: entry.content.clone(),
                tokens: entry.tokens,
                score: entry.score,
                committed_at: entry.committed_at,
            });
        }
        ccs
    }

    /// Manifest header + the snapshot mirror + a one-line lifecycle summary.
    pub fn render_markdown(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        let _ = writeln!(
            out,
            "<!-- {} v{} · units: {} · {} -->",
            self.manifest.format,
            self.manifest.version,
            self.manifest.unit_source,
            self.manifest.created_at.to_rfc3339()
        );
        let _ = writeln!(
            out,
            "# Working context ({} entries, {} / {} tokens)\n",
            self.snapshot.entries.len(),
            self.snapshot.token_count,
            self.snapshot.budget_tokens
        );
        out.push_str(&self.snapshot.render_markdown());
        if !self.lifecycle.is_empty() {
            let _ = write!(out, "\n\n_lifecycle: {} event(s)_", self.lifecycle.len());
        }
        out
    }
}

// ---- OCF (Open Cognitive Format) ---------------------------------------------------------------
//
// OCF (github.com/aquifer-labs/ocf) is a sibling on-disk layout of the same committed working
// state, split into four files — manifest / schema / snapshot / qualify — so the schema is a
// checkable contract and the qualify log carries admit/reject decisions. Artesian is the reference
// implementation.

const OCF_VERSION: &str = "0.1";
const OCF_MANIFEST_FILE: &str = "manifest.json";
const OCF_SCHEMA_FILE: &str = "schema.json";
const OCF_SNAPSHOT_FILE: &str = "snapshot.json";
const OCF_QUALIFY_FILE: &str = "qualify.jsonl";

#[derive(Serialize, Deserialize)]
struct OcfManifest {
    ocf_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    agent_id: Option<String>,
    created: DateTime<Utc>,
    unit_source: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    unit_refs: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct OcfSlot {
    name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    description: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct OcfSchemaFile {
    ocf_version: String,
    slots: Vec<OcfSlot>,
    budget_tokens: usize,
    eviction: String,
}

#[derive(Serialize, Deserialize)]
struct OcfSnapshotFile {
    budget_tokens: usize,
    token_count: usize,
    saturation: f32,
    entries: Vec<SnapshotEntry>,
}

/// One line of `qualify.jsonl`: an admit/reject decision with its reason — the governance trail
/// that travels with an OCF bundle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualifyRecord {
    pub ts: DateTime<Utc>,
    pub unit_ref: String,
    pub admitted: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slot: Option<String>,
    pub score: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl WorkingContextBundle {
    /// Write the bundle in the Open Cognitive Format (OCF) four-file layout.
    pub fn write_ocf_dir(&self, dir: &Path) -> Result<(), BundleError> {
        std::fs::create_dir_all(dir)?;
        let manifest = OcfManifest {
            ocf_version: OCF_VERSION.to_string(),
            agent_id: self.manifest.agent_id.clone(),
            created: self.manifest.created_at,
            unit_source: self.manifest.unit_source.clone(),
            unit_refs: self.manifest.unit_ref.clone().into_iter().collect(),
        };
        let schema = OcfSchemaFile {
            ocf_version: OCF_VERSION.to_string(),
            slots: self
                .snapshot
                .schema
                .iter()
                .map(|name| OcfSlot {
                    name: name.clone(),
                    description: None,
                })
                .collect(),
            budget_tokens: self.snapshot.budget_tokens,
            eviction: "lowest-score".to_string(),
        };
        let saturation = if self.snapshot.budget_tokens > 0 {
            self.snapshot.token_count as f32 / self.snapshot.budget_tokens as f32
        } else {
            0.0
        };
        let snapshot = OcfSnapshotFile {
            budget_tokens: self.snapshot.budget_tokens,
            token_count: self.snapshot.token_count,
            saturation,
            entries: self.snapshot.entries.clone(),
        };
        // Admit record per committed entry (carrying slot + score), plus any reject/deprecation
        // events recorded in the lifecycle log.
        let mut qualify: Vec<QualifyRecord> = self
            .snapshot
            .entries
            .iter()
            .map(|entry| QualifyRecord {
                ts: entry.committed_at,
                unit_ref: entry.id.clone(),
                admitted: true,
                slot: Some(entry.slot.clone()),
                score: entry.score,
                reason: Some("qualified".to_string()),
            })
            .collect();
        for event in &self.lifecycle {
            if matches!(event.decision, Decision::Evict | Decision::Deprecate) {
                qualify.push(QualifyRecord {
                    ts: event.ts,
                    unit_ref: event.entry_id.clone(),
                    admitted: false,
                    slot: None,
                    score: 0.0,
                    reason: Some(format!("{:?}", event.decision).to_lowercase()),
                });
            }
        }
        std::fs::write(
            dir.join(OCF_MANIFEST_FILE),
            serde_json::to_string_pretty(&manifest)?,
        )?;
        std::fs::write(
            dir.join(OCF_SCHEMA_FILE),
            serde_json::to_string_pretty(&schema)?,
        )?;
        std::fs::write(
            dir.join(OCF_SNAPSHOT_FILE),
            serde_json::to_string_pretty(&snapshot)?,
        )?;
        let mut jsonl = String::new();
        for record in &qualify {
            jsonl.push_str(&serde_json::to_string(record)?);
            jsonl.push('\n');
        }
        std::fs::write(dir.join(OCF_QUALIFY_FILE), jsonl)?;
        Ok(())
    }

    /// Read an OCF four-file bundle back into a working-context bundle (the "resume" path).
    pub fn read_ocf_dir(dir: &Path) -> Result<Self, BundleError> {
        let manifest: OcfManifest =
            serde_json::from_str(&std::fs::read_to_string(dir.join(OCF_MANIFEST_FILE))?)?;
        if !version_compatible(&manifest.ocf_version) {
            return Err(BundleError::Version(manifest.ocf_version));
        }
        let schema: OcfSchemaFile =
            serde_json::from_str(&std::fs::read_to_string(dir.join(OCF_SCHEMA_FILE))?)?;
        let snapshot: OcfSnapshotFile =
            serde_json::from_str(&std::fs::read_to_string(dir.join(OCF_SNAPSHOT_FILE))?)?;
        let working = WorkingContextSnapshot {
            schema: schema.slots.into_iter().map(|slot| slot.name).collect(),
            budget_tokens: snapshot.budget_tokens,
            token_count: snapshot.token_count,
            entries: snapshot.entries,
        };
        let bundle_manifest = BundleManifest {
            format: BUNDLE_FORMAT.to_string(),
            version: BUNDLE_VERSION.to_string(),
            agent_id: manifest.agent_id,
            created_at: manifest.created,
            unit_source: manifest.unit_source,
            unit_ref: manifest.unit_refs.into_iter().next(),
        };
        let bundle = Self {
            manifest: bundle_manifest,
            snapshot: working,
            lifecycle: Vec::new(),
        };
        bundle.validate()?;
        Ok(bundle)
    }
}

/// Accept any bundle sharing this reader's major version.
fn version_compatible(found: &str) -> bool {
    let major = |version: &str| version.split('.').next().unwrap_or_default().to_string();
    major(found) == major(BUNDLE_VERSION)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixed_ts() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2026-06-21T08:30:00Z")
            .unwrap()
            .with_timezone(&Utc)
    }

    fn sample_snapshot() -> WorkingContextSnapshot {
        let entries = vec![
            SnapshotEntry {
                id: "a".into(),
                slot: "decision".into(),
                content: "ship the working-context bundle first".into(),
                tokens: 7,
                score: 1.0,
                resolution: Resolution::Full,
                unit_ref: None,
                committed_at: fixed_ts(),
            },
            SnapshotEntry {
                id: "b".into(),
                slot: "task-state".into(),
                content: "wire export/import into the CLI".into(),
                tokens: 6,
                score: 0.5,
                resolution: Resolution::Full,
                unit_ref: Some("pam://entry/9f86".into()),
                committed_at: fixed_ts(),
            },
        ];
        WorkingContextSnapshot {
            schema: vec!["decision".into(), "constraint".into(), "task-state".into()],
            budget_tokens: 4096,
            token_count: 13,
            entries,
        }
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "artesian_bundle_{}_{}_{}",
            tag,
            std::process::id(),
            nanos
        ))
    }

    #[test]
    fn snapshot_from_ccs_preserves_typed_slots_and_budget() {
        let mut ccs = CommittedContextState::new(CcsSchema::default(), 1000);
        ccs.admit(CommittedEntry::new("a", "decision", "chose Rust", 1.0));
        ccs.admit(CommittedEntry::new("b", "fact", "uses cargo", 0.5));
        let snapshot = WorkingContextSnapshot::from_ccs(&ccs);
        assert_eq!(snapshot.schema, ccs.schema().slots);
        assert_eq!(snapshot.budget_tokens, 1000);
        assert_eq!(snapshot.token_count, ccs.token_count());
        assert_eq!(snapshot.entries.len(), 2);
    }

    #[test]
    fn bundle_round_trips_through_a_directory() {
        let bundle = WorkingContextBundle::new(
            sample_snapshot(),
            vec![LifecycleEntry {
                ts: fixed_ts(),
                entry_id: "a".into(),
                decision: Decision::Commit,
                status: Status::Active,
                supersedes: None,
                reason: Some(LifecycleReason {
                    relevance: 0.9,
                    novelty: 0.5,
                    drift: 0.0,
                }),
            }],
        );
        let dir = temp_dir("roundtrip");
        bundle.write_dir(&dir).expect("write");
        let read = WorkingContextBundle::read_dir(&dir).expect("read");
        assert_eq!(read.snapshot, bundle.snapshot);
        assert_eq!(read.lifecycle, bundle.lifecycle);
        assert_eq!(read.manifest.unit_source, "inline");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn validate_rejects_token_mismatch_and_bad_format() {
        let mut snapshot = sample_snapshot();
        snapshot.token_count = 999; // lie about the footprint
        let bundle = WorkingContextBundle::new(snapshot, Vec::new());
        assert!(matches!(
            bundle.validate(),
            Err(BundleError::TokenMismatch { .. })
        ));

        let mut bad = WorkingContextBundle::new(sample_snapshot(), Vec::new());
        bad.manifest.format = "something.else".into();
        assert!(matches!(bad.validate(), Err(BundleError::Format(_))));
    }

    #[test]
    fn to_ccs_rehydrates_entries_for_resume() {
        let bundle = WorkingContextBundle::new(sample_snapshot(), Vec::new());
        let ccs = bundle.to_ccs();
        assert_eq!(ccs.len(), 2);
        assert_eq!(ccs.budget_tokens(), 4096);
        assert!(ccs.contains("a"));
        assert!(ccs.contains("b"));
        // Rehydrated render is grouped by the snapshot's schema slots.
        let rendered = ccs.render();
        assert!(rendered.contains("## decision"));
        assert!(rendered.contains("ship the working-context bundle first"));
    }

    #[test]
    fn version_compatibility_accepts_same_major() {
        assert!(version_compatible("0.1"));
        assert!(version_compatible("0.9"));
        assert!(!version_compatible("1.0"));
    }

    #[test]
    fn ocf_round_trips_through_a_directory() {
        let bundle =
            WorkingContextBundle::new(sample_snapshot(), vec![LifecycleEntry::commit("a")]);
        let dir = temp_dir("ocf");
        bundle.write_ocf_dir(&dir).expect("write ocf");
        for file in [
            "manifest.json",
            "schema.json",
            "snapshot.json",
            "qualify.jsonl",
        ] {
            assert!(dir.join(file).exists(), "missing {file}");
        }
        let read = WorkingContextBundle::read_ocf_dir(&dir).expect("read ocf");
        assert_eq!(read.snapshot, bundle.snapshot);
        assert_eq!(read.manifest.unit_source, "inline");
        std::fs::remove_dir_all(&dir).ok();
    }
}
