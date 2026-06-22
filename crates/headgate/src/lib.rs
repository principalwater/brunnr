// SPDX-License-Identifier: Apache-2.0

//! Headgate — the Agent Cognitive Compressor (ACC) control plane.
//!
//! Headgate implements the ACC model (arXiv:2601.11653): it separates the **recall** channel
//! (read candidates from any retrieval store via [`RecallStore`]) from the **commit** channel
//! (what enters the bounded, schema-governed [`CommittedContextState`]), with a
//! [`QualifyGate`] between them as the trust boundary. The [`Headgate`] controller runs the
//! commit-loop, compressing or evicting under saturation, and reports [`GaugeMetrics`].
//!
//! The crate is fully usable offline: the default gate is deterministic and the default
//! [`Compressor`] is extractive. LLM-backed gates (judge-eval of drift / hallucination) and
//! LLM compressors are drop-in replacements via [`Headgate::with_gate`] /
//! [`Headgate::with_compressor`].
//!
//! ```
//! use std::sync::Arc;
//! use headgate::{Headgate, HeadgateConfig, RecallItem, StaticRecallStore};
//!
//! # async fn demo() -> headgate::HeadgateResult<()> {
//! let store = Arc::new(StaticRecallStore::new(vec![
//!     RecallItem::new("n1", "the team chose Rust for the core crates", 1.0),
//!     RecallItem::new("n2", "the team chose Rust for the core crates", 1.0), // duplicate
//! ]));
//! let mut headgate = Headgate::new(store, HeadgateConfig::default());
//! let metrics = headgate.cycle("which language").await?;
//! assert_eq!(metrics.admitted, 1); // the redundant duplicate is rejected
//! assert!(headgate.render().contains("chose Rust"));
//! # Ok(())
//! # }
//! ```

mod bundle;
mod ccs;
mod compressor;
mod controller;
#[cfg(feature = "llm")]
mod council;
#[cfg(feature = "llm")]
mod fact;
mod gate;
#[cfg(feature = "headroom")]
mod headroom;
#[cfg(feature = "llm")]
mod judge;
#[cfg(feature = "llm")]
mod llm;
mod metrics;
mod recall;

pub use bundle::{
    BundleError, BundleManifest, Decision, LifecycleEntry, LifecycleReason, OcfSession,
    QualifyRecord, Resolution, SnapshotEntry, Status, WorkingContextBundle, WorkingContextSnapshot,
    BUNDLE_FORMAT, BUNDLE_VERSION,
};
pub use ccs::{CcsSchema, CommittedContextState, CommittedEntry};
pub use compressor::{Compressor, ExtractiveCompressor, NoopCompressor};
pub use controller::{Headgate, HeadgateConfig};
pub use gate::{DefaultQualifyGate, QualifyDecision, QualifyGate};
pub use metrics::{count_tokens, GaugeMetrics};
pub use recall::{MemoryRecallStore, RecallItem, RecallStore, StaticRecallStore};

#[cfg(feature = "llm")]
pub use compressor::LlmCompressor;
#[cfg(feature = "llm")]
pub use council::CouncilJudge;
#[cfg(feature = "llm")]
pub use fact::extract_atomic_facts;
#[cfg(feature = "headroom")]
pub use headroom::HeadroomCompressor;
#[cfg(feature = "llm")]
pub use judge::{JudgeQualifyGate, JudgeVerdict};
#[cfg(feature = "llm")]
pub use llm::{
    llm_client_from_config, CommandLlmClient, LlmClient, LlmRequest, OpenAiCompatibleClient,
    StaticLlmClient,
};

use thiserror::Error;

pub type HeadgateResult<T> = Result<T, HeadgateError>;

/// Errors surfaced by the ACC control plane.
#[derive(Debug, Error)]
pub enum HeadgateError {
    #[error("recall store error: {0}")]
    Recall(String),
    #[error("compressor error: {0}")]
    Compress(String),
    #[error("llm error: {0}")]
    Llm(String),
    #[error("memory backend error: {0}")]
    Memory(#[from] aquifer::MemoryError),
}
