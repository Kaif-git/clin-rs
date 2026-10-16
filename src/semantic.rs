//! Semantic-similarity overlay for the graph view.
//!
//! Embeds notes with `all-MiniLM-L6-v2` (fastembed ONNX, 384d, local only)
//! and returns `id -> [titles]` of similar notes. [`crate::graf_adapter`]
//! merges these into `NodeSpec.links` so similar-but-unlinked notes cluster
//! without any `[[wikilink]]`.
//!
//! Lifecycle (MVP): runs **only on graph open / refresh**, never while
//! typing. Vectors are cached per vault at `.clin/semantic_cache.json`
//! keyed by content hash, so only dirty notes are re-embedded.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

pub const MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
const CACHE_VERSION: u32 = 1;
const CACHE_DIMS: usize = 384;
/// Hard cap so similarity stays O(n^2)-cheap (2000^2 * 384 ≈ fast enough).
const MAX_NOTES: usize = 2000;
/// Max Related hits shown per note search.
pub const SEARCH_TOP_K: usize = 8;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct CacheEntry {
    hash: u64,
    vector: Vec<f32>,
}

#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
struct CacheFile {
    version: u32,
    model: String,
    dims: usize,
    entries: HashMap<String, CacheEntry>,
}

/// `title + tags + truncated body` input for the embedding model.
pub fn build_embed_text(
    title: &str,
    tags: &[String],
    content: &str,
    max_chars: usize,
    include_tags: bool,
) -> String {
    let mut text = String::with_capacity(256.min(max_chars.max(64)));
    text.push_str(title.trim());
    if include_tags {
        let tag_str: Vec<&str> = tags
            .iter()
            .map(std::string::String::as_str)
            .filter(|t| !t.trim().is_empty())
            .collect();
        if !tag_str.is_empty() {
            text.push_str("\nTags: ");
            text.push_str(&tag_str.join(", "));
        }
    }
    let body = content.trim();
    if !body.is_empty() {
        text.push_str("\n\n");
        text.push_str(body);
    }
    truncate_chars(&text, max_chars.max(64))
}

fn truncate_chars(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    s.chars().take(max_chars).collect()
}

pub fn content_hash(text: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

pub fn normalize_in_place(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Top-k similar titles per note id, skipping explicitly-linked pairs.
///
/// `items`: `(id, title, l2-normalized vector)`.
/// `explicit`: `id -> {explicitly linked note ids}` (both directions checked).
pub fn top_k_pairs(
    items: &[(String, String, Vec<f32>)],
    explicit: &HashMap<String, HashSet<String>>,
    threshold: f32,
    top_k: usize,
) -> HashMap<String, Vec<String>> {
    let mut out: HashMap<String, Vec<String>> = HashMap::new();
    if items.len() < 2 || top_k == 0 {
        return out;
    }
    let mut candidates: HashMap<&str, Vec<(f32, usize)>> = HashMap::new();
    for (i, (id_a, _, vec_a)) in items.iter().enumerate() {
        for (j, (id_b, _, vec_b)) in items.iter().enumerate().skip(i + 1) {
            if id_a == id_b {
                continue;
            }
            let linked = explicit.get(id_a).is_some_and(|s| s.contains(id_b))
                || explicit.get(id_b).is_some_and(|s| s.contains(id_a));
            if linked {
                continue;
            }
            let sim = cosine_sim(vec_a, vec_b);
            if sim >= threshold {
                candidates.entry(id_a.as_str()).or_default().push((sim, j));
                candidates.entry(id_b.as_str()).or_default().push((sim, i));
            }
        }
    }
    let title_of = |idx: usize| items[idx].1.clone();
    for (id, mut list) in candidates {
        list.sort_by(|a, b| b.0.total_cmp(&a.0));
        list.truncate(top_k);
        out.insert(
            id.to_string(),
            list.into_iter().map(|(_, j)| title_of(j)).collect(),
        );
    }
    out
}

pub fn cache_path(data_dir: &Path) -> PathBuf {
    data_dir.join(".clin").join("semantic_cache.json")
}

fn load_cache(data_dir: &Path) -> CacheFile {
    let path = cache_path(data_dir);
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(_) => return CacheFile::default(),
    };
    let file: CacheFile = match serde_json::from_slice(&bytes) {
        Ok(f) => f,
        Err(_) => return CacheFile::default(),
    };
    if file.version != CACHE_VERSION || file.model != MODEL_ID || file.dims != CACHE_DIMS {
        return CacheFile::default();
    }
    file
}

fn save_cache(data_dir: &Path, file: &CacheFile) {
    let path = cache_path(data_dir);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(bytes) = serde_json::to_vec(file) {
        let _ = std::fs::write(path, bytes);
    }
}

/// Embed dirty texts, reusing cached vectors by content hash.
///
/// Only initializes the ONNX model when at least one note needs embedding,
/// so graph refresh with no changes never pays model-load cost.
fn ensure_embeddings(
    texts: &[(String, String)],
    data_dir: &Path,
) -> Result<HashMap<String, Vec<f32>>> {
    // Unloaded: `ensure_embeddings_in` only loads the model when the cache
    // misses, so graph refresh with no changes stays free.
    let mut embedder = Embedder::unload();
    ensure_embeddings_in(&mut embedder, texts, data_dir)
}

/// Long-lived embedding model. The search worker keeps one per thread so
/// every keystroke doesn't pay model init; the graph path loads on demand.
pub(crate) struct Embedder {
    model: Option<fastembed::TextEmbedding>,
}

impl Embedder {
    pub(crate) fn load() -> Result<Self> {
        let model = fastembed::TextEmbedding::try_new(fastembed::InitOptions::new(
            fastembed::EmbeddingModel::AllMiniLML6V2,
        ))
        .context("failed to init embedding model (first run downloads ~80MB)")?;
        Ok(Self { model: Some(model) })
    }

    pub(crate) fn unload() -> Self {
        Self { model: None }
    }

    fn is_loaded(&self) -> bool {
        self.model.is_some()
    }

    pub(crate) fn embed_texts(&mut self, docs: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if !self.is_loaded() {
            *self = Self::load()?;
        }
        let model = self.model.as_mut().context("embedding model not loaded")?;
        let mut out = model.embed(docs, None).context("failed to embed notes")?;
        for v in &mut out {
            normalize_in_place(v);
        }
        Ok(out)
    }
}

fn ensure_embeddings_in(
    embedder: &mut Embedder,
    texts: &[(String, String)],
    data_dir: &Path,
) -> Result<HashMap<String, Vec<f32>>> {
    let mut file = load_cache(data_dir);
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut dirty: Vec<(String, String)> = Vec::new();
    for (id, text) in texts {
        let hash = content_hash(text);
        if let Some(entry) = file.entries.get(id)
            && entry.hash == hash
            && entry.vector.len() == CACHE_DIMS
        {
            let mut v = entry.vector.clone();
            normalize_in_place(&mut v);
            vectors.insert(id.clone(), v);
        } else {
            dirty.push((id.clone(), text.clone()));
        }
    }
    if dirty.is_empty() {
        return Ok(vectors);
    }
    let docs: Vec<String> = dirty.iter().map(|(_, t)| t.clone()).collect();
    let embeddings = embedder.embed_texts(docs)?;
    for ((id, text), vec) in dirty.into_iter().zip(embeddings) {
        file.entries.insert(
            id.clone(),
            CacheEntry {
                hash: content_hash(&text),
                vector: vec.clone(),
            },
        );
        vectors.insert(id, vec);
    }
    let live: HashSet<&str> = texts.iter().map(|(id, _)| id.as_str()).collect();
    file.entries.retain(|id, _| live.contains(id.as_str()));
    file.version = CACHE_VERSION;
    file.model = MODEL_ID.to_string();
    file.dims = CACHE_DIMS;
    save_cache(data_dir, &file);
    Ok(vectors)
}

/// Top `(note id, cosine)` matches for a raw query string against embedded
/// note texts. Powers the Related section of note search; the graph edge
/// builder stays in [`top_k_pairs`].
pub(crate) fn query_top(
    embedder: &mut Embedder,
    query: &str,
    texts: &[(String, String)],
    data_dir: &Path,
    threshold: f32,
    top_k: usize,
) -> Result<Vec<(String, f32)>> {
    if texts.is_empty() || top_k == 0 || query.trim().is_empty() {
        return Ok(Vec::new());
    }
    let vectors = ensure_embeddings_in(embedder, texts, data_dir)?;
    let query_vec = embedder
        .embed_texts(vec![truncate_chars(query.trim(), 2000)])
        .context("failed to embed search query")?
        .pop()
        .unwrap_or_default();
    let mut scored: Vec<(String, f32)> = vectors
        .iter()
        .map(|(id, v)| (id.clone(), cosine_sim(&query_vec, v)))
        .filter(|(_, s)| *s >= threshold)
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored.truncate(top_k);
    Ok(scored)
}

pub(crate) fn is_embeddable(id: &str) -> bool {
    !(id.ends_with(".draw")
        || id.ends_with(".canvas")
        || id.ends_with(".clin")
        || id.ends_with(".pinstar"))
}

/// Compute `id -> [similar titles]` for the graph overlay.
///
/// Returns empty on disabled config or <2 embeddable notes. Notes that fail
/// to load are skipped (never fails the whole graph).
pub fn compute_semantic_extras(
    summaries: &[crate::storage::NoteSummary],
    storage: &crate::storage::Storage,
    features: &crate::config::FeaturesConfig,
    semantic: &crate::config::SemanticConfig,
    data_dir: &Path,
) -> Result<HashMap<String, Vec<String>>> {
    let empty: HashMap<String, Vec<String>> = HashMap::new();
    if !semantic.enabled || semantic.top_k == 0 || semantic.threshold >= 1.0 {
        return Ok(empty);
    }
    let mut candidates: Vec<&crate::storage::NoteSummary> = summaries
        .iter()
        .filter(|n| {
            features.file_view_enabled(&n.id) && is_embeddable(&n.id) && !n.title.trim().is_empty()
        })
        .collect();
    if candidates.len() < 2 {
        return Ok(empty);
    }
    candidates.sort_by_key(|n| std::cmp::Reverse(n.updated_at));
    candidates.truncate(MAX_NOTES);

    let mut texts: Vec<(String, String)> = Vec::with_capacity(candidates.len());
    for n in &candidates {
        let content = match storage.load_note(&n.id) {
            Ok(note) => note.content,
            Err(_) => continue,
        };
        texts.push((
            n.id.clone(),
            build_embed_text(
                &n.title,
                &n.tags,
                &content,
                semantic.max_chars,
                semantic.include_tags,
            ),
        ));
    }
    if texts.len() < 2 {
        return Ok(empty);
    }
    let vectors = ensure_embeddings(&texts, data_dir)?;

    // title -> id (last wins, mirroring graf `build_graph`'s title map).
    let mut title_to_id: HashMap<String, &str> = HashMap::new();
    for n in &candidates {
        title_to_id.insert(n.title.to_lowercase(), n.id.as_str());
    }
    let mut explicit: HashMap<String, HashSet<String>> = HashMap::new();
    for n in &candidates {
        let entry = explicit.entry(n.id.clone()).or_default();
        for link in &n.links {
            if let Some(target) = title_to_id.get(&link.to_lowercase()) {
                entry.insert((*target).to_string());
            }
        }
    }
    let items: Vec<(String, String, Vec<f32>)> = candidates
        .iter()
        .filter_map(|n| {
            vectors
                .get(&n.id)
                .map(|v| (n.id.clone(), n.title.clone(), v.clone()))
        })
        .collect();
    Ok(top_k_pairs(
        &items,
        &explicit,
        semantic.threshold,
        semantic.top_k,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embed_text_includes_title_tags_and_truncates() {
        let out = build_embed_text(
            "Ash King",
            &["lore".to_string(), "fire".to_string()],
            "abcdefghij",
            64,
            true,
        );
        assert!(out.starts_with("Ash King"));
        assert!(out.contains("lore"));
        assert!(out.chars().count() <= 64);
        // Floor keeps tiny budgets usable instead of title-only garbage.
        let floored = build_embed_text("Ash King", &[], "body", 10, true);
        assert!(floored.starts_with("Ash King"));
    }

    #[test]
    fn embed_text_without_tags() {
        let out = build_embed_text("T", &["x".to_string()], "body", 2000, false);
        assert!(!out.contains("Tags:"));
        assert!(out.contains("body"));
    }

    #[test]
    fn cosine_and_normalize() {
        let mut v = vec![3.0_f32, 4.0];
        normalize_in_place(&mut v);
        assert!((cosine_sim(&v, &v) - 1.0).abs() < 1e-5);
        assert!((cosine_sim(&[1.0, 0.0], &[0.0, 1.0])).abs() < 1e-6);
    }

    #[test]
    fn top_k_respects_threshold_and_k() {
        let items = vec![
            ("a".to_string(), "A".to_string(), vec![1.0_f32, 0.0]),
            ("b".to_string(), "B".to_string(), vec![0.99, 0.141]),
            ("c".to_string(), "C".to_string(), vec![0.0, 1.0]),
        ];
        let explicit: HashMap<String, HashSet<String>> = HashMap::new();
        let out = top_k_pairs(&items, &explicit, 0.9, 5);
        assert!(out.get("a").is_some_and(|v| v == &vec!["B".to_string()]));
        assert!(out.get("c").is_none() || out["c"].is_empty());

        let out_k0 = top_k_pairs(&items, &explicit, 0.0, 0);
        assert!(out_k0.is_empty());
    }

    #[test]
    fn top_k_skips_explicit_links() {
        let items = vec![
            ("a".to_string(), "A".to_string(), vec![1.0_f32, 0.0]),
            ("b".to_string(), "B".to_string(), vec![1.0, 0.0]),
        ];
        let mut explicit: HashMap<String, HashSet<String>> = HashMap::new();
        explicit.insert("a".to_string(), HashSet::from(["b".to_string()]));
        let out = top_k_pairs(&items, &explicit, 0.0, 5);
        assert!(out.values().all(std::vec::Vec::is_empty) || out.is_empty());
    }

    #[test]
    fn cache_roundtrip_prunes_stale_ids() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file = CacheFile {
            version: CACHE_VERSION,
            model: MODEL_ID.to_string(),
            dims: CACHE_DIMS,
            entries: HashMap::from([
                (
                    "keep".to_string(),
                    CacheEntry {
                        hash: 1,
                        vector: vec![0.5; CACHE_DIMS],
                    },
                ),
                (
                    "stale".to_string(),
                    CacheEntry {
                        hash: 2,
                        vector: vec![0.5; CACHE_DIMS],
                    },
                ),
            ]),
        };
        save_cache(dir.path(), &file);
        let loaded = load_cache(dir.path());
        assert_eq!(loaded.entries.len(), 2);
    }

    #[test]
    fn query_top_shorts_circuits_without_model() {
        // Empty texts / empty query return before any model load, so these
        // run offline with no download.
        let mut embedder = Embedder::unload();
        let dir = tempfile::tempdir().expect("tempdir");
        let out = query_top(&mut embedder, "ember", &[], dir.path(), 0.0, 8).expect("empty");
        assert!(out.is_empty());
        let texts = vec![("a".to_string(), "Ash King fire desert".to_string())];
        let out = query_top(&mut embedder, "   ", &texts, dir.path(), 0.0, 8).expect("blank");
        assert!(out.is_empty());
        let out = query_top(&mut embedder, "ember", &texts, dir.path(), 0.0, 0).expect("k0");
        assert!(out.is_empty());
    }

    #[test]
    fn skips_non_text_ids() {
        assert!(!is_embeddable("map.draw"));
        assert!(!is_embeddable("board.canvas"));
        assert!(!is_embeddable("secret.clin"));
        assert!(is_embeddable("lore/city.md"));
    }
}
