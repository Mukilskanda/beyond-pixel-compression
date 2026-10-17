"""Visual semantic cache — image-similarity function for cache-hit lookup
(new direction, this session): reuses this project's EXISTING saliency
machinery (pixelpath/saliency.py, previously validated and shown NOT to
reduce per-call cost on either tested provider — §9) as the similarity
function that makes call-elimination caching possible. The pixel-analysis
tools built for region selection are repurposed here, not discarded.

Motivation (paper/draft.md, new Related Work subsection on caching):
prompt caching only reuses exact repeated prefixes; semantic caching (reuse
an answer for a similar-but-not-identical query) exists only for text in
current systems - multimodal semantic caching is an explicitly open gap.
This module is the similarity half of a visual semantic cache: given a new
image, find the most similar previously-answered image in the cache store.

Similarity metric: cosine similarity between two images' saliency maps,
each resized to a fixed small grid (default 16x16) so images of different
resolutions are comparable and the comparison is cheap (no model inference).
This reuses saliency_map() as-is - no new visual feature extraction was
built for this.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pixelpath.saliency import saliency_map


def _saliency_signature(image_path: str | Path, grid: int = 16) -> np.ndarray:
    """Fixed-size, L2-normalized saliency signature for one image."""
    sal = saliency_map(image_path)
    small = cv2.resize(sal, (grid, grid), interpolation=cv2.INTER_AREA)
    flat = small.flatten().astype(np.float64)
    norm = np.linalg.norm(flat)
    return flat / norm if norm > 0 else flat


def image_similarity(image_path_a: str | Path, image_path_b: str | Path, grid: int = 16) -> float:
    """Cosine similarity in [-1, 1] (in practice [0, 1] for saliency maps,
    which are non-negative) between two images' saliency signatures."""
    sig_a = _saliency_signature(image_path_a, grid)
    sig_b = _saliency_signature(image_path_b, grid)
    return float(np.dot(sig_a, sig_b))


class QuestionSimilarity:
    """Text-embedding similarity for the query-aware cache signature
    (paper/draft.md §11.4). Uses TF-IDF (scikit-learn, already a project
    dependency - router/learned.py's planned learned router) rather than a
    downloaded embedding model, to stay genuinely zero-cost/zero-network:
    no model weights to fetch, no API call, no new dependency. TF-IDF is a
    real, standard text-similarity baseline, not a placeholder - it is
    reported as exactly that (a baseline) in draft.md, not oversold as a
    modern embedding.

    Must be fit on a corpus of questions before use (`fit`) - cosine
    similarity between TF-IDF vectors is only meaningful within one fitted
    vector space, so the corpus should be every question the cache will
    ever compare against (in this project's real-data experiment, the same
    40 DocVQA questions used for the image-similarity analysis).
    """

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._fitted = False
        self._question_to_vec: dict[str, np.ndarray] = {}

    def fit(self, questions: list[str]) -> None:
        matrix = self._vectorizer.fit_transform(questions)  # rows already L2-normalized (sklearn default)
        self._question_to_vec = {
            q: np.asarray(matrix[i].todense()).flatten() for i, q in enumerate(questions)
        }
        self._fitted = True

    def vector(self, question: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("QuestionSimilarity must be fit() on a question corpus before use")
        if question in self._question_to_vec:
            return self._question_to_vec[question]
        # question not in the fitted corpus - transform on the fly (still
        # valid, just not pre-cached)
        vec = self._vectorizer.transform([question])
        return np.asarray(vec.todense()).flatten()

    def similarity(self, question_a: str, question_b: str) -> float:
        va, vb = self.vector(question_a), self.vector(question_b)
        norm_a, norm_b = np.linalg.norm(va), np.linalg.norm(vb)
        if norm_a == 0 or norm_b == 0:
            return 0.0  # no shared vocabulary signal - treat as dissimilar, not an error
        return float(np.dot(va, vb) / (norm_a * norm_b))


@dataclass
class CacheEntry:
    image_path: str
    question: str
    answer: str
    signature: np.ndarray


class VisualSemanticCache:
    """Backing store of (image, question, answer) entries with signature-
    based similarity lookup. Deliberately simple (linear scan, no ANN index)
    - the cache stores are small (tens to low hundreds of entries) at the
    scale this project's real data supports; an ANN index would be
    premature optimization for a store this size.
    """

    def __init__(self, grid: int = 16):
        self.grid = grid
        self.entries: list[CacheEntry] = []

    def add(self, image_path: str | Path, question: str, answer: str) -> None:
        sig = _saliency_signature(image_path, self.grid)
        self.entries.append(CacheEntry(image_path=str(image_path), question=question, answer=answer, signature=sig))

    def best_match(self, image_path: str | Path) -> tuple[CacheEntry | None, float]:
        """Returns (most similar cache entry, similarity score), or
        (None, 0.0) if the cache is empty."""
        if not self.entries:
            return None, 0.0
        query_sig = _saliency_signature(image_path, self.grid)
        best_entry = None
        best_sim = -2.0  # below any real cosine similarity
        for entry in self.entries:
            sim = float(np.dot(query_sig, entry.signature))
            if sim > best_sim:
                best_sim = sim
                best_entry = entry
        return best_entry, best_sim

    def lookup(self, image_path: str | Path, threshold: float) -> tuple[CacheEntry | None, float]:
        """Returns (entry, similarity) if the best match clears `threshold`
        (a cache HIT - the caller should reuse entry.answer instead of
        calling the API), else (None, similarity) (a cache MISS)."""
        entry, sim = self.best_match(image_path)
        if entry is not None and sim >= threshold:
            return entry, sim
        return None, sim


@dataclass
class QueryCacheEntry:
    image_path: str
    question: str
    answer: str
    image_signature: np.ndarray


class QueryAwareCache:
    """§11.4: cache-hit requires BOTH image similarity AND question
    similarity to clear a threshold, unlike VisualSemanticCache above
    (image-only, §11.2-11.3's negative result). Deliberately a separate
    class rather than a modified VisualSemanticCache, so the original
    negative-result experiment stays exactly reproducible.
    """

    def __init__(self, question_similarity: QuestionSimilarity, grid: int = 16):
        self.question_similarity = question_similarity
        self.grid = grid
        self.entries: list[QueryCacheEntry] = []

    def add(self, image_path: str | Path, question: str, answer: str) -> None:
        sig = _saliency_signature(image_path, self.grid)
        self.entries.append(QueryCacheEntry(image_path=str(image_path), question=question, answer=answer, image_signature=sig))

    def best_match(self, image_path: str | Path, question: str) -> tuple[QueryCacheEntry | None, float, float]:
        """Returns (best entry by combined score, image_sim, question_sim)
        for that entry, or (None, 0.0, 0.0) if the cache is empty. "Best"
        is ranked by min(image_sim, question_sim) - the combined score a
        hit must clear on BOTH axes, not just on average."""
        if not self.entries:
            return None, 0.0, 0.0
        query_img_sig = _saliency_signature(image_path, self.grid)
        best_entry, best_img_sim, best_q_sim, best_combined = None, 0.0, 0.0, -2.0
        for entry in self.entries:
            img_sim = float(np.dot(query_img_sig, entry.image_signature))
            q_sim = self.question_similarity.similarity(question, entry.question)
            combined = min(img_sim, q_sim)
            if combined > best_combined:
                best_combined = combined
                best_entry, best_img_sim, best_q_sim = entry, img_sim, q_sim
        return best_entry, best_img_sim, best_q_sim

    def lookup(self, image_path: str | Path, question: str, threshold: float) -> tuple[QueryCacheEntry | None, float, float]:
        """Cache HIT only if BOTH image_sim >= threshold AND question_sim >=
        threshold for the best-matching entry (same shared threshold on
        both axes, for direct comparability with §11.2's image-only curve)."""
        entry, img_sim, q_sim = self.best_match(image_path, question)
        if entry is not None and img_sim >= threshold and q_sim >= threshold:
            return entry, img_sim, q_sim
        return None, img_sim, q_sim
