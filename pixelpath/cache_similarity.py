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
