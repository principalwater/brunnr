// SPDX-License-Identifier: Apache-2.0

//! Local, token-free coding-agent quota awareness.
//!
//! The reader is deliberately local-only: it reads small quota/usage/rate-limit files from agent
//! homes and never starts an agent, calls an LLM, or touches the network.

use std::{
    collections::BTreeMap,
    env, fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    time::SystemTime,
};

use aquifer::{
    MemoryBackend, MemoryScope, MemoryTier, Session, SessionAnchor, SessionKey, SessionSummary,
    StoreMemory, SESSION_RECORD_SOURCE, SESSION_RECORD_TAG,
};
use chrono::{DateTime, TimeZone, Utc};
use headgate::{SnapshotEntry, WorkingContextBundle, WorkingContextSnapshot};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const DEFAULT_QUOTA_WARN_PCT: f64 = 80.0;
pub const DEFAULT_QUOTA_HIGH_PCT: f64 = 95.0;
pub const QUOTA_CONTINUATION_NOTE: &str =
    "quota near limit - resume with another agent via `artesian session resume`";

const PERCENT_LIMIT: f64 = 100.0;
const MAX_QUOTA_FILE_BYTES: u64 = 2 * 1024 * 1024;
const CODEX_SESSIONS_DIR_ENV: &str = "CODEX_SESSIONS_DIR";
const CODEX_ROLLOUT_SCAN_DEPTH: usize = 8;
const MAX_CODEX_ROLLOUT_CANDIDATES: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum QuotaWindow {
    FiveHour,
    Weekly,
}

impl QuotaWindow {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FiveHour => "5h",
            Self::Weekly => "weekly",
        }
    }

    pub const fn display(self) -> &'static str {
        match self {
            Self::FiveHour => "5-hour",
            Self::Weekly => "weekly",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum QuotaStatusKind {
    Known,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuotaWindowStatus {
    pub agent: String,
    pub window: String,
    pub status: QuotaStatusKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub used: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pct: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resets_at: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

impl QuotaWindowStatus {
    fn known(
        agent: &str,
        window: QuotaWindow,
        reading: ParsedWindow,
        source: Option<PathBuf>,
    ) -> Self {
        Self {
            agent: agent.to_string(),
            window: window.as_str().to_string(),
            status: QuotaStatusKind::Known,
            used: reading.used,
            limit: reading.limit,
            pct: reading.pct,
            resets_at: reading.resets_at,
            source,
            message: None,
        }
    }

    fn unknown(agent: &str, window: QuotaWindow, message: impl Into<String>) -> Self {
        Self {
            agent: agent.to_string(),
            window: window.as_str().to_string(),
            status: QuotaStatusKind::Unknown,
            used: None,
            limit: None,
            pct: None,
            resets_at: None,
            source: None,
            message: Some(message.into()),
        }
    }

    pub fn threshold_message(&self, threshold_pct: f64) -> Option<String> {
        let pct = self.pct?;
        (pct >= threshold_pct).then(|| {
            let reset = self
                .resets_at
                .map(|value| value.to_rfc3339())
                .unwrap_or_else(|| "unknown reset".to_string());
            format!(
                "quota warning: {} {} window at {:.1}% used (threshold {:.1}%, resets {})",
                self.agent, self.window, pct, threshold_pct, reset
            )
        })
    }
}

#[derive(Debug, Clone)]
pub struct QuotaReadOptions {
    pub codex_home: Option<PathBuf>,
    pub claude_roots: Vec<PathBuf>,
}

impl QuotaReadOptions {
    pub fn from_env() -> Self {
        let home = home_dir();
        let codex_home = env::var_os("CODEX_HOME")
            .map(PathBuf::from)
            .or_else(|| home.as_ref().map(|home| home.join(".codex")));
        let mut claude_roots = env::var("CLAUDE_CONFIG_DIR")
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .map(str::trim)
                    .filter(|part| !part.is_empty())
                    .map(PathBuf::from)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if let Some(home) = home {
            claude_roots.push(home.join(".claude"));
            claude_roots.push(home.join(".config").join("claude"));
        }
        claude_roots.sort();
        claude_roots.dedup();
        Self {
            codex_home,
            claude_roots,
        }
    }
}

impl Default for QuotaReadOptions {
    fn default() -> Self {
        Self::from_env()
    }
}

#[derive(Debug, Clone)]
pub struct QuotaLoopConfig {
    pub warn_pct: f64,
    pub high_pct: f64,
    pub checkpoint_on_quota: bool,
    pub reader: QuotaReadOptions,
}

impl Default for QuotaLoopConfig {
    fn default() -> Self {
        Self {
            warn_pct: DEFAULT_QUOTA_WARN_PCT,
            high_pct: DEFAULT_QUOTA_HIGH_PCT,
            checkpoint_on_quota: false,
            reader: QuotaReadOptions::from_env(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuotaThresholdEvent {
    pub status: QuotaWindowStatus,
    pub threshold_pct: f64,
    pub high: bool,
}

#[derive(Debug, Clone)]
pub struct QuotaContinuation {
    pub key: SessionKey,
    pub anchor: SessionAnchor,
    pub resume_packet: Option<Value>,
}

#[derive(Debug, Clone)]
pub struct QuotaContinuationContext<'a> {
    pub run_id: &'a str,
    pub goal: &'a str,
    pub worker_cmd: Option<&'a str>,
    pub turn: u32,
    pub run_log_path: &'a Path,
    pub last_failed_check: Option<&'a str>,
    pub event: &'a QuotaThresholdEvent,
}

#[derive(Debug, Clone)]
struct ParsedWindow {
    window: QuotaWindow,
    used: Option<f64>,
    limit: Option<f64>,
    pct: Option<f64>,
    resets_at: Option<DateTime<Utc>>,
}

pub fn read_local_quota() -> Vec<QuotaWindowStatus> {
    read_local_quota_with_options(&QuotaReadOptions::from_env())
}

pub fn read_local_quota_with_options(options: &QuotaReadOptions) -> Vec<QuotaWindowStatus> {
    let mut statuses = Vec::new();
    statuses.extend(read_codex_quota(options.codex_home.as_deref()));
    statuses.extend(read_claude_quota(&options.claude_roots));
    statuses
}

pub fn quota_threshold_events(
    statuses: &[QuotaWindowStatus],
    warn_pct: f64,
    high_pct: f64,
) -> Vec<QuotaThresholdEvent> {
    statuses
        .iter()
        .filter_map(|status| {
            let pct = status.pct?;
            (pct >= warn_pct).then(|| QuotaThresholdEvent {
                status: status.clone(),
                threshold_pct: warn_pct,
                high: pct >= high_pct,
            })
        })
        .collect()
}

pub fn quota_session_key(run_id: &str, goal: &str) -> SessionKey {
    SessionKey::new(
        None,
        Some(format!("quota-{run_id}")),
        Some(format!("quota-loop-{}", slugify_goal(goal))),
    )
}

pub async fn write_quota_continuation(
    anchor_store: &aquifer::AnchorAnchorStore,
    backend: Option<&dyn MemoryBackend>,
    context: &QuotaContinuationContext<'_>,
) -> anyhow::Result<QuotaContinuation> {
    let key = quota_session_key(context.run_id, context.goal);
    let quota_line = format!(
        "{} {} quota at {:.1}% used",
        context.event.status.agent,
        context.event.status.window,
        context.event.status.pct.unwrap_or_default()
    );
    let current_task = format!("Loop goal: {}", context.goal);
    let next_step = match context.worker_cmd {
        Some(worker) => format!(
            "{QUOTA_CONTINUATION_NOTE}. Continue `{worker}` for goal `{}`.",
            context.goal
        ),
        None => format!(
            "{QUOTA_CONTINUATION_NOTE}. Continue verifying goal `{}`.",
            context.goal
        ),
    };
    let mut anchor = SessionAnchor::new(current_task, next_step);
    anchor.plan_pointer = Some(context.run_log_path.display().to_string());
    anchor.last_decisions = vec![
        QUOTA_CONTINUATION_NOTE.to_string(),
        quota_line,
        format!("checkpoint written before loop turn {}", context.turn),
    ];
    let anchor = anchor_store.set_for_session(&key, anchor).await?;
    let bundle = quota_continuation_bundle(&anchor, context);
    let session = bundle.to_ocf_session(&key, Some(context.event.status.agent.clone()))?;
    let resume_packet = WorkingContextBundle::resume_packet_from_session(&session).ok();
    if let Some(backend) = backend {
        store_session_checkpoint(backend, session).await?;
    }
    Ok(QuotaContinuation {
        key,
        anchor,
        resume_packet,
    })
}

fn read_codex_quota(root: Option<&Path>) -> Vec<QuotaWindowStatus> {
    if let Some((source, parsed)) = read_codex_transcript_quota(root) {
        return materialize_agent_windows("codex", Some(source), parsed);
    }

    read_agent_quota(
        "codex",
        root.into_iter().map(Path::to_path_buf).collect(),
        &[
            "rate_limits.json",
            "rate-limits.json",
            "rateLimits.json",
            "usage.json",
            "quota.json",
            "state/rate_limits.json",
            "state/usage.json",
            "account/rate_limits.json",
            "account/rateLimits.json",
            "account/usage.json",
        ],
    )
}

fn read_claude_quota(roots: &[PathBuf]) -> Vec<QuotaWindowStatus> {
    read_agent_quota(
        "claude",
        roots.to_vec(),
        &[
            "stats-cache.json",
            "usage.json",
            "quota.json",
            "rate_limits.json",
            "rate-limits.json",
            "limits.json",
            "usage_limits.json",
            "usage-limits.json",
            "state/usage.json",
            "state/rate_limits.json",
            "stats/usage.json",
        ],
    )
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct CodexRolloutCandidate {
    path: PathBuf,
    rollout_key: Option<String>,
    modified: Option<SystemTime>,
}

fn read_codex_transcript_quota(root: Option<&Path>) -> Option<(PathBuf, Vec<ParsedWindow>)> {
    let mut candidates = codex_rollout_candidates(&codex_session_roots(root));
    candidates.sort_by(|left, right| {
        right
            .rollout_key
            .cmp(&left.rollout_key)
            .then_with(|| right.modified.cmp(&left.modified))
            .then_with(|| right.path.cmp(&left.path))
    });

    for candidate in candidates.into_iter().take(MAX_CODEX_ROLLOUT_CANDIDATES) {
        if let Ok(Some(parsed)) = parse_codex_rollout_file(&candidate.path) {
            if parsed.iter().any(|window| window.pct.is_some()) {
                return Some((candidate.path, parsed));
            }
        }
    }
    None
}

fn codex_session_roots(root: Option<&Path>) -> Vec<PathBuf> {
    let mut roots = env::var_os(CODEX_SESSIONS_DIR_ENV)
        .map(|value| env::split_paths(&value).collect::<Vec<_>>())
        .unwrap_or_default();

    if let Some(root) = root {
        roots.push(root.join("sessions"));
        roots.push(root.join("archived_sessions"));
    }

    roots.sort();
    roots.dedup();
    roots
}

fn codex_rollout_candidates(roots: &[PathBuf]) -> Vec<CodexRolloutCandidate> {
    let mut candidates = Vec::new();
    for root in roots.iter().filter(|root| root.exists()) {
        collect_codex_rollout_candidates(root, 0, &mut candidates);
    }
    candidates
}

fn collect_codex_rollout_candidates(
    root: &Path,
    depth: usize,
    candidates: &mut Vec<CodexRolloutCandidate>,
) {
    if depth > CODEX_ROLLOUT_SCAN_DEPTH {
        return;
    }

    let Ok(entries) = fs::read_dir(root) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };

        if file_type.is_dir() {
            collect_codex_rollout_candidates(&path, depth + 1, candidates);
        } else if file_type.is_file() && is_codex_rollout_file(&path) {
            candidates.push(CodexRolloutCandidate {
                rollout_key: codex_rollout_sort_key(&path),
                path,
                modified: entry
                    .metadata()
                    .and_then(|metadata| metadata.modified())
                    .ok(),
            });
        }
    }
}

fn codex_rollout_sort_key(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    let timestamp = name.strip_prefix("rollout-")?.get(..19).filter(|value| {
        value.as_bytes().get(4) == Some(&b'-')
            && value.as_bytes().get(7) == Some(&b'-')
            && value.as_bytes().get(10) == Some(&b'T')
            && value.as_bytes().get(13) == Some(&b'-')
            && value.as_bytes().get(16) == Some(&b'-')
    })?;
    Some(timestamp.to_string())
}

fn is_codex_rollout_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with("rollout-") && name.ends_with(".jsonl"))
}

fn parse_codex_rollout_file(path: &Path) -> anyhow::Result<Option<Vec<ParsedWindow>>> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    Ok(parse_quota_jsonl(reader))
}

fn read_agent_quota(
    agent: &str,
    roots: Vec<PathBuf>,
    candidate_names: &[&str],
) -> Vec<QuotaWindowStatus> {
    let missing_message = if roots.is_empty() {
        format!("{agent} home is not configured")
    } else {
        format!("no local {agent} quota file found")
    };
    let mut last_error = None;
    for root in roots.iter().filter(|root| root.exists()) {
        for relative in candidate_names {
            let path = root.join(relative);
            if !path.exists() {
                continue;
            }
            match parse_quota_file(agent, &path) {
                Ok(parsed) if parsed.iter().any(|window| window.pct.is_some()) => {
                    return materialize_agent_windows(agent, Some(path), parsed);
                }
                Ok(_) => {
                    last_error = Some(format!("{} did not contain quota windows", path.display()));
                }
                Err(error) => {
                    last_error = Some(format!("{}: {error}", path.display()));
                }
            }
        }
    }
    unknown_agent_windows(agent, last_error.unwrap_or(missing_message))
}

fn materialize_agent_windows(
    agent: &str,
    source: Option<PathBuf>,
    parsed: Vec<ParsedWindow>,
) -> Vec<QuotaWindowStatus> {
    let mut by_window: BTreeMap<QuotaWindow, ParsedWindow> = BTreeMap::new();
    for window in parsed {
        by_window.entry(window.window).or_insert(window);
    }
    [QuotaWindow::FiveHour, QuotaWindow::Weekly]
        .into_iter()
        .map(|window| {
            by_window.remove(&window).map_or_else(
                || QuotaWindowStatus::unknown(agent, window, "window not present in quota file"),
                |reading| QuotaWindowStatus::known(agent, window, reading, source.clone()),
            )
        })
        .collect()
}

fn unknown_agent_windows(agent: &str, message: String) -> Vec<QuotaWindowStatus> {
    [QuotaWindow::FiveHour, QuotaWindow::Weekly]
        .into_iter()
        .map(|window| QuotaWindowStatus::unknown(agent, window, message.clone()))
        .collect()
}

fn parse_quota_file(agent: &str, path: &Path) -> anyhow::Result<Vec<ParsedWindow>> {
    let metadata = fs::metadata(path)?;
    if metadata.len() > MAX_QUOTA_FILE_BYTES {
        anyhow::bail!("quota file is too large");
    }
    let raw = fs::read_to_string(path)?;
    parse_quota_text(agent, &raw)
}

fn parse_quota_text(agent: &str, raw: &str) -> anyhow::Result<Vec<ParsedWindow>> {
    let json_text = strip_line_comments(raw);
    if let Ok(value) = serde_json::from_str::<Value>(&json_text) {
        let parsed = parse_quota_json(&value);
        if !parsed.is_empty() {
            return Ok(parsed);
        }
    }
    if json_text.contains("\"rate_limits\"") {
        if let Some(parsed) = parse_quota_jsonl(json_text.as_bytes()) {
            return Ok(parsed);
        }
    }
    Ok(parse_status_text(agent, raw))
}

fn parse_quota_json(value: &Value) -> Vec<ParsedWindow> {
    let mut parsed = Vec::new();
    collect_windows(value, None, 0, false, &mut parsed);
    parsed
}

fn parse_quota_jsonl(reader: impl BufRead) -> Option<Vec<ParsedWindow>> {
    let mut last = None;
    for line in reader.lines().map_while(Result::ok) {
        if !line.contains("\"rate_limits\"") {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let parsed = parse_quota_json(&value);
        if !parsed.is_empty() {
            last = Some(parsed);
        }
    }
    last
}

fn collect_windows(
    value: &Value,
    key_hint: Option<&str>,
    depth: usize,
    parent_unlimited: bool,
    parsed: &mut Vec<ParsedWindow>,
) {
    if depth > 5 {
        return;
    }
    let unlimited = value
        .as_object()
        .and_then(|object| object.get("unlimited"))
        .and_then(bool_value)
        .unwrap_or(parent_unlimited);
    if let Some(window) = parse_window_object(value, key_hint, unlimited) {
        parsed.push(window);
    }
    match value {
        Value::Object(object) => {
            for (key, child) in object {
                collect_windows(child, Some(key), depth + 1, unlimited, parsed);
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_windows(item, key_hint, depth + 1, unlimited, parsed);
            }
        }
        _ => {}
    }
}

fn parse_window_object(
    value: &Value,
    key_hint: Option<&str>,
    parent_unlimited: bool,
) -> Option<ParsedWindow> {
    let object = value.as_object()?;
    let window = classify_window(key_hint, value)?;
    if object
        .get("unlimited")
        .and_then(bool_value)
        .unwrap_or(parent_unlimited)
    {
        return Some(ParsedWindow {
            window,
            used: Some(0.0),
            limit: None,
            pct: Some(0.0),
            resets_at: reset_datetime(object),
        });
    }
    let pct = used_percent(object)?;
    let limit = first_number(
        object,
        &[
            "limit",
            "quota",
            "max",
            "total",
            "limit_percent",
            "limitPercent",
        ],
    )
    .unwrap_or(PERCENT_LIMIT);
    let used = first_number(object, &["used", "consumed"]).unwrap_or(pct * limit / PERCENT_LIMIT);
    Some(ParsedWindow {
        window,
        used: Some(round_one(used)),
        limit: Some(round_one(limit)),
        pct: Some(round_one(pct)),
        resets_at: reset_datetime(object),
    })
}

fn classify_window(key_hint: Option<&str>, value: &Value) -> Option<QuotaWindow> {
    if let Some(minutes) = window_minutes(value) {
        if (minutes - 300.0).abs() < 0.5 {
            return Some(QuotaWindow::FiveHour);
        }
        if (minutes - 10080.0).abs() < 0.5 {
            return Some(QuotaWindow::Weekly);
        }
    }
    let label = value
        .get("window")
        .and_then(Value::as_str)
        .or_else(|| value.get("name").and_then(Value::as_str))
        .or(key_hint)?;
    window_from_label(label)
}

fn window_from_label(label: &str) -> Option<QuotaWindow> {
    let normalized = label
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase();
    if normalized.contains("fivehour")
        || normalized.contains("5h")
        || normalized.contains("session")
        || normalized == "primary"
        || normalized == "primarywindow"
    {
        return Some(QuotaWindow::FiveHour);
    }
    if normalized.contains("sevenday")
        || normalized.contains("7day")
        || normalized.contains("weekly")
        || normalized.contains("week")
        || normalized == "secondary"
        || normalized == "secondarywindow"
    {
        return Some(QuotaWindow::Weekly);
    }
    None
}

fn used_percent(object: &serde_json::Map<String, Value>) -> Option<f64> {
    if let Some(pct) = first_number(
        object,
        &[
            "pct",
            "percent",
            "used_percent",
            "usedPercent",
            "percent_used",
            "percentUsed",
            "pct_used",
            "pctUsed",
            "utilization",
            "usage_percent",
            "usagePercent",
        ],
    ) {
        return Some(clamp_percent(pct));
    }
    if let Some(left) = first_number(
        object,
        &[
            "remaining_percent",
            "remainingPercent",
            "percent_left",
            "percentLeft",
            "pct_left",
            "pctLeft",
        ],
    ) {
        return Some(clamp_percent(PERCENT_LIMIT - left));
    }
    let used = first_number(object, &["used", "consumed"])?;
    let limit = first_number(object, &["limit", "quota", "max", "total"])?;
    (limit > 0.0).then(|| clamp_percent(used / limit * PERCENT_LIMIT))
}

fn window_minutes(value: &Value) -> Option<f64> {
    let object = value.as_object()?;
    first_number(
        object,
        &[
            "window_minutes",
            "windowMinutes",
            "windowDurationMins",
            "window_duration_mins",
        ],
    )
    .or_else(|| {
        first_number(
            object,
            &[
                "limit_window_seconds",
                "limitWindowSeconds",
                "window_seconds",
                "windowSeconds",
                "windowDurationSecs",
            ],
        )
        .map(|seconds| seconds / 60.0)
    })
}

fn reset_datetime(object: &serde_json::Map<String, Value>) -> Option<DateTime<Utc>> {
    [
        "resets_at",
        "resetsAt",
        "reset_at",
        "resetAt",
        "reset_time",
        "resetTime",
        "resets",
    ]
    .into_iter()
    .find_map(|key| object.get(key).and_then(parse_datetime_value))
}

fn parse_datetime_value(value: &Value) -> Option<DateTime<Utc>> {
    match value {
        Value::Number(number) => number.as_f64().and_then(datetime_from_unix),
        Value::String(text) => parse_datetime_str(text),
        _ => None,
    }
}

fn parse_datetime_str(text: &str) -> Option<DateTime<Utc>> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    DateTime::parse_from_rfc3339(trimmed)
        .map(|date| date.with_timezone(&Utc))
        .ok()
        .or_else(|| trimmed.parse::<f64>().ok().and_then(datetime_from_unix))
}

fn datetime_from_unix(value: f64) -> Option<DateTime<Utc>> {
    if !value.is_finite() || value <= 0.0 {
        return None;
    }
    let seconds = if value > 10_000_000_000.0 {
        value / 1000.0
    } else {
        value
    };
    let whole = seconds.trunc() as i64;
    let nanos = ((seconds.fract() * 1_000_000_000.0).round() as u32).min(999_999_999);
    Utc.timestamp_opt(whole, nanos).single()
}

fn first_number(object: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<f64> {
    keys.iter()
        .find_map(|key| object.get(*key).and_then(number_value))
}

fn number_value(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.trim().replace(',', "").parse::<f64>().ok(),
        _ => None,
    }
}

fn bool_value(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(value) => Some(*value),
        Value::String(text) => match text.trim().to_ascii_lowercase().as_str() {
            "true" | "yes" | "1" => Some(true),
            "false" | "no" | "0" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

fn parse_status_text(agent: &str, text: &str) -> Vec<ParsedWindow> {
    text.lines()
        .filter_map(|line| parse_status_line(agent, line))
        .collect()
}

fn parse_status_line(agent: &str, line: &str) -> Option<ParsedWindow> {
    let lower = line.to_ascii_lowercase();
    let window = match agent {
        "codex" if lower.contains("5h limit") => QuotaWindow::FiveHour,
        "codex" if lower.contains("weekly limit") => QuotaWindow::Weekly,
        "claude" if lower.contains("current session") => QuotaWindow::FiveHour,
        "claude" if lower.contains("current week") => QuotaWindow::Weekly,
        _ => return None,
    };
    let value = first_percent_token(line)?;
    let pct =
        if lower.contains("left") || lower.contains("remaining") || lower.contains("available") {
            PERCENT_LIMIT - value
        } else {
            value
        };
    Some(ParsedWindow {
        window,
        used: Some(round_one(pct)),
        limit: Some(PERCENT_LIMIT),
        pct: Some(round_one(clamp_percent(pct))),
        resets_at: None,
    })
}

fn first_percent_token(line: &str) -> Option<f64> {
    let percent = line.find('%')?;
    let before = &line[..percent];
    let start = before
        .rfind(|ch: char| !(ch.is_ascii_digit() || ch == '.'))
        .map_or(0, |index| index + 1);
    before[start..].trim().parse::<f64>().ok()
}

fn strip_line_comments(raw: &str) -> String {
    raw.lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn quota_continuation_bundle(
    anchor: &SessionAnchor,
    context: &QuotaContinuationContext<'_>,
) -> WorkingContextBundle {
    let mut entries = vec![
        SnapshotEntry::now(
            "anchor-task",
            "task-state",
            anchor.current_task.clone(),
            1.0,
        ),
        SnapshotEntry::now("anchor-next", "task-state", anchor.next_step.clone(), 1.0),
        SnapshotEntry::now(
            "quota-note",
            "decision",
            QUOTA_CONTINUATION_NOTE.to_string(),
            1.0,
        ),
        SnapshotEntry::now(
            "quota-status",
            "task-state",
            serde_json::to_string(&context.event.status).unwrap_or_else(|_| {
                format!(
                    "{} {} quota crossed threshold",
                    context.event.status.agent, context.event.status.window
                )
            }),
            1.0,
        ),
        SnapshotEntry::now(
            "run-log",
            "task-state",
            context.run_log_path.display().to_string(),
            0.8,
        ),
    ];
    if let Some(plan_pointer) = &anchor.plan_pointer {
        entries.push(SnapshotEntry::now(
            "anchor-plan",
            "task-state",
            plan_pointer.clone(),
            1.0,
        ));
    }
    for (index, decision) in anchor.last_decisions.iter().enumerate() {
        entries.push(SnapshotEntry::now(
            format!("anchor-decision-{index}"),
            "decision",
            decision.clone(),
            1.0,
        ));
    }
    if let Some(last_failed_check) = context.last_failed_check {
        entries.push(SnapshotEntry::now(
            "last-failed-check",
            "task-state",
            last_failed_check.to_string(),
            0.9,
        ));
    }
    let token_count = entries.iter().map(|entry| entry.tokens).sum();
    WorkingContextBundle::new(
        WorkingContextSnapshot {
            schema: vec!["task-state".to_string(), "decision".to_string()],
            budget_tokens: 4096,
            token_count,
            entries,
        },
        Vec::new(),
    )
}

async fn store_session_checkpoint(
    backend: &dyn MemoryBackend,
    mut session: Session,
) -> anyhow::Result<SessionSummary> {
    session.updated_at = Utc::now();
    let key = session.key.clone();
    let content = serde_json::to_string(&session)?;
    let mut metadata = BTreeMap::new();
    metadata.insert("record_type".to_string(), SESSION_RECORD_TAG.to_string());
    backend
        .store(StoreMemory {
            content,
            tags: vec![SESSION_RECORD_TAG.to_string()],
            metadata,
            tier: MemoryTier::L2Scenario,
            node_id: Some(key.node_id()),
            created_at: Some(session.updated_at),
            scope: Some(MemoryScope::Session),
            agent_id: None,
            session_id: Some(key.session_id.clone()),
            task_id: Some(key.task_id.clone()),
            user_id: Some(key.user_id.clone()),
            project: None,
            source: Some(SESSION_RECORD_SOURCE.to_string()),
            confidence: Some(1.0),
            relations: Vec::new(),
        })
        .await?;
    Ok(SessionSummary::from(&session))
}

fn slugify_goal(goal: &str) -> String {
    let mut slug = String::new();
    let mut last_dash = false;
    for ch in goal.chars().flat_map(char::to_lowercase) {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch);
            last_dash = false;
        } else if !last_dash && !slug.is_empty() {
            slug.push('-');
            last_dash = true;
        }
        if slug.len() >= 64 {
            break;
        }
    }
    while slug.ends_with('-') {
        slug.pop();
    }
    if slug.is_empty() {
        "task".to_string()
    } else {
        slug
    }
}

fn clamp_percent(value: f64) -> f64 {
    value.clamp(0.0, PERCENT_LIMIT)
}

fn round_one(value: f64) -> f64 {
    (value * 10.0).round() / 10.0
}

fn home_dir() -> Option<PathBuf> {
    env::var_os("HOME").map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use artesian_test_support::TempDir;

    #[test]
    fn parses_codex_rollout_rate_limit_shape() {
        let parsed = parse_quota_text(
            "codex",
            include_str!("../tests/fixtures/codex-rate-limits.json"),
        )
        .expect("fixture should parse");
        let statuses = materialize_agent_windows("codex", None, parsed);
        let five = statuses
            .iter()
            .find(|status| status.window == "5h")
            .expect("5h status");
        let weekly = statuses
            .iter()
            .find(|status| status.window == "weekly")
            .expect("weekly status");
        assert_eq!(five.status, QuotaStatusKind::Known);
        assert_eq!(five.pct, Some(82.0));
        assert_eq!(five.used, Some(82.0));
        assert_eq!(five.limit, Some(100.0));
        assert_eq!(weekly.pct, Some(41.0));
        assert_eq!(
            five.resets_at.map(|value| value.to_rfc3339()),
            Some("2026-07-01T00:00:00+00:00".to_string())
        );
        assert_eq!(
            weekly.resets_at.map(|value| value.to_rfc3339()),
            Some("2026-07-06T00:00:00+00:00".to_string())
        );
    }

    #[test]
    fn claude_stats_cache_without_limit_windows_is_unknown() {
        let parsed = parse_quota_text(
            "claude",
            include_str!("../tests/fixtures/claude-usage.json"),
        )
        .expect("fixture should parse");
        let statuses = materialize_agent_windows("claude", None, parsed);
        let five = statuses
            .iter()
            .find(|status| status.window == "5h")
            .expect("5h status");
        let weekly = statuses
            .iter()
            .find(|status| status.window == "weekly")
            .expect("weekly status");
        assert_eq!(five.status, QuotaStatusKind::Unknown);
        assert_eq!(weekly.status, QuotaStatusKind::Unknown);
        assert_eq!(five.pct, None);
        assert_eq!(weekly.resets_at, None);
    }

    #[test]
    fn reads_codex_quota_from_latest_session_rollout() {
        let tempdir = TempDir::new("quota-codex-sessions");
        let codex_home = tempdir.join("codex");
        let rollout_dir = codex_home
            .join("sessions")
            .join("2026")
            .join("06")
            .join("28");
        std::fs::create_dir_all(&rollout_dir).unwrap();
        let rollout = rollout_dir.join("rollout-2026-06-28T14-49-17-fixture.jsonl");
        std::fs::write(
            &rollout,
            include_str!("../tests/fixtures/codex-rate-limits.json"),
        )
        .unwrap();

        let statuses = read_local_quota_with_options(&QuotaReadOptions {
            codex_home: Some(codex_home),
            claude_roots: vec![tempdir.join("missing-claude")],
        });
        let five = statuses
            .iter()
            .find(|status| status.agent == "codex" && status.window == "5h")
            .expect("5h status");
        let weekly = statuses
            .iter()
            .find(|status| status.agent == "codex" && status.window == "weekly")
            .expect("weekly status");

        assert_eq!(five.status, QuotaStatusKind::Known);
        assert_eq!(five.pct, Some(82.0));
        assert_eq!(weekly.status, QuotaStatusKind::Known);
        assert_eq!(weekly.pct, Some(41.0));
        assert_eq!(
            five.resets_at.map(|value| value.to_rfc3339()),
            Some("2026-07-01T00:00:00+00:00".to_string())
        );
        assert_eq!(
            weekly.resets_at.map(|value| value.to_rfc3339()),
            Some("2026-07-06T00:00:00+00:00".to_string())
        );
        assert_eq!(five.source.as_deref(), Some(rollout.as_path()));
    }

    #[test]
    fn unlimited_codex_windows_are_known_without_threshold_events() {
        let parsed = parse_quota_text(
            "codex",
            r#"{"timestamp":"2026-06-28T14:49:17.000Z","type":"event_msg","payload":{"rate_limits":{"primary":{"unlimited":true,"window_minutes":300,"resets_at":1782864000},"secondary":{"unlimited":true,"window_minutes":10080,"resets_at":1783296000}}}}"#,
        )
        .expect("fixture should parse");
        let statuses = materialize_agent_windows("codex", None, parsed);

        assert!(statuses
            .iter()
            .all(|status| status.status == QuotaStatusKind::Known));
        assert!(statuses.iter().all(|status| status.pct == Some(0.0)));
        assert!(quota_threshold_events(&statuses, 1.0, 95.0).is_empty());
    }

    #[test]
    fn missing_files_are_unknown_not_errors() {
        let options = QuotaReadOptions {
            codex_home: Some(PathBuf::from("/definitely/missing/codex")),
            claude_roots: vec![PathBuf::from("/definitely/missing/claude")],
        };
        let statuses = read_local_quota_with_options(&options);
        assert_eq!(statuses.len(), 4);
        assert!(statuses
            .iter()
            .all(|status| status.status == QuotaStatusKind::Unknown));
    }

    #[test]
    fn threshold_events_fire_at_or_above_threshold() {
        let statuses = vec![
            QuotaWindowStatus::known(
                "codex",
                QuotaWindow::FiveHour,
                ParsedWindow {
                    window: QuotaWindow::FiveHour,
                    used: Some(80.0),
                    limit: Some(100.0),
                    pct: Some(80.0),
                    resets_at: None,
                },
                None,
            ),
            QuotaWindowStatus::known(
                "claude",
                QuotaWindow::Weekly,
                ParsedWindow {
                    window: QuotaWindow::Weekly,
                    used: Some(79.9),
                    limit: Some(100.0),
                    pct: Some(79.9),
                    resets_at: None,
                },
                None,
            ),
        ];
        let events = quota_threshold_events(&statuses, 80.0, 95.0);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].status.agent, "codex");
        assert!(!events[0].high);
    }
}
