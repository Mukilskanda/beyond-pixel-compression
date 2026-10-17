"""ChartQA loader via HuggingFace Hub (lmms-lab/ChartQA) — same
no-registration pattern as data/docvqa_loader.py. Structurally different
domain from DocVQA: chart images (bars/lines/axes) rather than scanned
text documents, so a real-billing text-route validation on this dataset
tests whether §12's finding generalizes beyond dense-text documents to a
domain where OCR is a much weaker signal (chart labels/axes/legend, not
paragraphs) — a chart's semantic content lives in shape/position, not
extractable text, which is exactly the DocVQA-vs-ChartQA contrast this
project's own §3.1 category list flagged as untested.

Note: dataset schema uses `answer` (singular) not `answers` (a list, as in
DocVQA) — normalized to a single-element list here so downstream code
(harness/judge.py, harness/metrics.py, both of which take
`ground_truths: list[str]`) doesn't need a ChartQA-specific branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ChartQASample:
    question_id: str
    image_path: Path
    question: str
    answers: list[str]


def load_chartqa(
    split: str = "test",
    limit: int | None = 20,
    cache_dir: str | Path = "data/chartqa_cache",
) -> list[ChartQASample]:
    """Stream real ChartQA samples from HuggingFace Hub and materialize
    images to disk, mirroring data.docvqa_loader.load_docvqa()'s pattern.
    `limit` defaults to 20, not None — this hits the network and writes
    files, so don't default to pulling the whole split unintentionally.

    ChartQA has no native question_id field (unlike DocVQA) - synthesizes
    one from the split name and row index instead, stable across repeated
    loads of the same split/limit since streaming order is deterministic.
    """
    from datasets import load_dataset

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("lmms-lab/ChartQA", split=split, streaming=True)

    samples = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break

        question_id = f"chartqa_{split}_{i}"
        image_path = cache_dir / f"{question_id}.png"
        if not image_path.exists():
            row["image"].save(image_path)

        samples.append(
            ChartQASample(
                question_id=question_id,
                image_path=image_path,
                question=row["question"],
                answers=[row["answer"]],
            )
        )

    return samples
