// SPDX-License-Identifier: Apache-2.0

//! Deterministic, structure-aware recursive chunking.
//!
//! Long records are split into bounded, coherent chunks so retrieval returns a
//! small relevant slice (top-k chunks) instead of whole documents — the
//! standard RAG granularity (cf. recursive character splitting). Splitting is
//! deterministic and requires no LLM, preserving Brunnr's zero-cost local
//! default. Boundaries are tried in order — markdown headings, blank lines,
//! line breaks, sentences, then words — so a chunk stays semantically coherent;
//! a small overlap carries context across chunk boundaries.

/// Chunking bounds. `max_chars` ≈ 4× the target token count (~400 tokens here).
#[derive(Debug, Clone, Copy)]
pub struct ChunkConfig {
    pub max_chars: usize,
    pub overlap_chars: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max_chars: 1_600,
            overlap_chars: 200,
        }
    }
}

/// One produced chunk: its text, its 1-based index, and the nearest heading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub content: String,
    pub index: usize,
    pub heading: Option<String>,
}

const SEPARATORS: &[&str] = &["\n\n", "\n", ". ", " "];

/// Split `content` into bounded, coherent chunks. Content already within
/// `max_chars` returns a single chunk unchanged.
pub fn chunk_text(content: &str, cfg: &ChunkConfig) -> Vec<Chunk> {
    let max_chars = cfg.max_chars.max(1);
    let overlap = cfg.overlap_chars.min(max_chars / 2);

    let mut chunks = Vec::new();
    for (heading, body) in markdown_sections(content) {
        let pieces = recursive_split(&body, max_chars);
        for text in merge_with_overlap(pieces, max_chars, overlap) {
            chunks.push(Chunk {
                content: text,
                index: chunks.len() + 1,
                heading: heading.clone(),
            });
        }
    }
    if chunks.is_empty() {
        chunks.push(Chunk {
            content: content.to_string(),
            index: 1,
            heading: None,
        });
    }
    chunks
}

/// Split into `(heading, body)` sections at markdown headings, keeping the
/// heading line inside its body so context is not lost.
fn markdown_sections(text: &str) -> Vec<(Option<String>, String)> {
    let mut sections: Vec<(Option<String>, String)> = Vec::new();
    let mut heading: Option<String> = None;
    let mut current: Vec<&str> = Vec::new();
    for line in text.lines() {
        if is_heading(line) {
            if !current.is_empty() {
                sections.push((heading.take(), current.join("\n")));
                current.clear();
            }
            heading = Some(line.trim_start_matches('#').trim().to_string());
        }
        current.push(line);
    }
    if !current.is_empty() {
        sections.push((heading, current.join("\n")));
    }
    sections
        .into_iter()
        .filter(|(_, body)| !body.trim().is_empty())
        .collect()
}

fn is_heading(line: &str) -> bool {
    let rest = line.trim_start_matches('#');
    rest.len() < line.len() && rest.starts_with(' ')
}

/// Recursively split `text` until every piece is within `max_chars`, trying each
/// separator in turn and falling back to a hard char window.
fn recursive_split(text: &str, max_chars: usize) -> Vec<String> {
    if char_len(text) <= max_chars {
        return vec![text.to_string()];
    }
    for sep in SEPARATORS {
        if text.contains(sep) {
            let mut out = Vec::new();
            for part in text.split(sep) {
                if char_len(part) <= max_chars {
                    if !part.trim().is_empty() {
                        out.push(part.to_string());
                    }
                } else {
                    out.extend(recursive_split(part, max_chars));
                }
            }
            return out;
        }
    }
    // No separator helps (e.g. one very long token): hard char windows.
    let chars: Vec<char> = text.chars().collect();
    chars
        .chunks(max_chars)
        .map(|window| window.iter().collect())
        .collect()
}

/// Greedily pack pieces into chunks up to `max_chars`, seeding each new chunk
/// with the tail `overlap` characters of the previous one for continuity.
fn merge_with_overlap(pieces: Vec<String>, max_chars: usize, overlap: usize) -> Vec<String> {
    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();
    for piece in pieces {
        let piece = piece.trim();
        if piece.is_empty() {
            continue;
        }
        let joined = if current.is_empty() { 0 } else { 1 };
        if !current.is_empty() && char_len(&current) + joined + char_len(piece) > max_chars {
            let finished = std::mem::take(&mut current);
            if overlap > 0 {
                current = tail(&finished, overlap);
            }
            chunks.push(finished);
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(piece);
    }
    if !current.trim().is_empty() {
        chunks.push(current.trim().to_string());
    }
    chunks
}

fn char_len(s: &str) -> usize {
    s.chars().count()
}

fn tail(s: &str, n: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    let start = chars.len().saturating_sub(n);
    chars[start..].iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_content_is_one_chunk() {
        let chunks = chunk_text("a small note", &ChunkConfig::default());
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "a small note");
    }

    #[test]
    fn large_content_is_bounded_and_covers_everything() {
        let cfg = ChunkConfig {
            max_chars: 400,
            overlap_chars: 40,
        };
        let needle = "the answer is 42.";
        let body = format!("{} {needle} {}", "filler. ".repeat(300), "tail. ".repeat(300));
        let chunks = chunk_text(&body, &cfg);
        assert!(chunks.len() > 1, "large content must split");
        for chunk in &chunks {
            assert!(
                chunk.content.chars().count() <= cfg.max_chars,
                "chunk exceeds max: {}",
                chunk.content.chars().count()
            );
        }
        assert!(
            chunks.iter().any(|c| c.content.contains("the answer is 42")),
            "the relevant passage must survive in some chunk"
        );
    }

    #[test]
    fn a_single_huge_token_is_hard_windowed() {
        let cfg = ChunkConfig {
            max_chars: 100,
            overlap_chars: 0,
        };
        let blob = "x".repeat(1_000); // no separators at all
        let chunks = chunk_text(&blob, &cfg);
        assert!(chunks.len() >= 10);
        for chunk in &chunks {
            assert!(chunk.content.chars().count() <= cfg.max_chars);
        }
    }

    #[test]
    fn headings_are_attached() {
        let text = "# Caching\nTTL is 90 seconds.\n\n# Auth\nTokens expire in 15 minutes.";
        let chunks = chunk_text(text, &ChunkConfig::default());
        assert!(chunks.iter().any(|c| c.heading.as_deref() == Some("Caching")));
        assert!(chunks.iter().any(|c| c.heading.as_deref() == Some("Auth")));
    }
}
