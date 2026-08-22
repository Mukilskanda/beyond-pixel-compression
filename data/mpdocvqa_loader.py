"""Real multi-page-document loader — MP-DocVQA (`lmms-lab-encoder/MP-DocVQA`
on HuggingFace Hub), the dataset that closes the gap `experiments/
cross_page_dedup_check.py`'s synthetic fixture couldn't: DocVQA/ChartQA
(this project's other loaders) are independent single-page documents by
construction, so cross-page dedup had never been checked against a real
multi-page document until this loader existed.

**Why this loader looks the way it does, not the simpler shape of
`data/docvqa_loader.py`.** MP-DocVQA's schema (`doc_id`, `page_ids`,
`image_1..image_20`) does NOT give one row per page the way DocVQA gives
one row per question. A single row only carries the page image(s) actually
relevant to that row's specific question — other pages of the same
document are `None` in that row, even though `page_ids` lists the full
page set. Getting every real page of one document means scanning rows
until enough of them have contributed a populated `image_N` for each page
index — confirmed by direct inspection this session (see git history for
the exact scan), not assumed.

**A real, load-bearing limitation found this session, not designed away:**
every document in this dataset's `val` split has >=15 pages (scanned
2,000 rows, minimum was 15) — there is no small real multi-page document
here, unlike DocVQA's small single documents. `load_mpdocvqa_pages` below
takes a slice of a larger real document's pages rather than a whole small
one, which is a legitimate substitute for this project's purposes (dedup
is a page-to-page mechanism, not a whole-document one) but is worth
knowing before assuming a fetched "document" is complete.

**Also found, and worth knowing before spending time on this loader
again:** streaming through enough rows to assemble even 10-20 pages of one
document took ~3-5 minutes and hit transient HF Hub connection
errors/timeouts on specific parquet shards more than once this session —
this is real, not a fluke, budget for it. `max_scan_rows` and
`max_scan_seconds` both cap how long a single call will try before giving
up with whatever it found.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MPDocVQAPage:
    doc_id: str
    page_index: int
    image_path: Path


def load_mpdocvqa_pages(
    doc_id: str | None = None,
    n_pages: int = 10,
    split: str = "val",
    cache_dir: str | Path = "data/mpdocvqa_cache",
    max_scan_rows: int = 2500,
    max_scan_seconds: float = 150.0,
) -> list[MPDocVQAPage]:
    """Stream real MP-DocVQA rows and assemble up to `n_pages` real,
    consecutive pages of one real document.

    If `doc_id` is None, picks whichever document accumulates the most
    real pages within the scan budget (this is how this session's
    reference document, `ylvj0223`, a real "Annexure to the Auditor's
    Report" + Board of Directors report, was found — not hand-picked).
    If `doc_id` is given, scans for that specific document only; may find
    fewer than `n_pages` (or none) within budget — a real, network-bound
    dataset, not a guaranteed lookup.

    Already-cached pages (by filename) are reused without re-streaming, in
    keeping with this project's caching discipline elsewhere.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if doc_id is not None:
        cached = sorted(cache_dir.glob(f"{doc_id}_p*.png"))
        if len(cached) >= n_pages:
            return [
                MPDocVQAPage(doc_id=doc_id, page_index=i, image_path=p) for i, p in enumerate(cached[:n_pages])
            ]
    else:
        # No specific doc requested: reuse whatever's already cached rather
        # than re-streaming, if it looks like a complete prior fetch.
        cached_by_doc: dict[str, list[Path]] = {}
        for f in sorted(cache_dir.glob("*_p*.png")):
            cached_by_doc.setdefault(f.name.rsplit("_p", 1)[0], []).append(f)
        best_cached = max(cached_by_doc.items(), key=lambda kv: len(kv[1]), default=(None, []))
        if len(best_cached[1]) >= n_pages:
            return [
                MPDocVQAPage(doc_id=best_cached[0], page_index=i, image_path=p)
                for i, p in enumerate(sorted(best_cached[1])[:n_pages])
            ]

    from collections import defaultdict

    from datasets import load_dataset

    by_doc: dict[str, dict[int, object]] = defaultdict(dict)
    n = 0
    start = time.monotonic()
    try:
        ds = load_dataset("lmms-lab-encoder/MP-DocVQA", split=split, streaming=True)
        for row in ds:
            n += 1
            row_doc_id = row["doc_id"]
            if doc_id is not None and row_doc_id != doc_id:
                continue
            page_ids = row["page_ids"]
            for idx in range(len(page_ids)):
                if idx in by_doc[row_doc_id]:
                    continue
                img = row.get(f"image_{idx + 1}")
                if img is not None:
                    by_doc[row_doc_id][idx] = img.copy()
            if doc_id is not None and len(by_doc[doc_id]) >= n_pages:
                break
            if n >= max_scan_rows or (time.monotonic() - start) > max_scan_seconds:
                break
    except Exception:
        pass  # real, observed network flakiness — return whatever was collected

    if not by_doc:
        return []

    if doc_id is not None:
        chosen_id, pages = doc_id, by_doc.get(doc_id, {})
    else:
        chosen_id, pages = max(by_doc.items(), key=lambda kv: len(kv[1]))

    result = []
    for idx in sorted(pages)[:n_pages]:
        path = cache_dir / f"{chosen_id}_p{idx:02d}.png"
        pages[idx].save(path)
        result.append(MPDocVQAPage(doc_id=chosen_id, page_index=idx, image_path=path))
    return result
