"""TextVQA loader via HuggingFace Hub (lmms-lab/textvqa) — same
no-registration streaming pattern as data/docvqa_loader.py and
data/chartqa_loader.py.

**Why this dataset, for what purpose.** DocVQA (scanned documents) and
ChartQA (charts) are this project's only two domains tested so far — both
are "designed to be read": dense paragraphs or labeled axes. TextVQA is a
real, standard, widely-cited benchmark (Singh et al. 2019) of natural
photographs (street signs, product packaging, camera brands, storefronts)
where text is incidental to the scene, not the point of it — stylized
fonts, rotated/partial text, cluttered backgrounds. This is the
generalization test: do §12's text-route and §13's tiling+batching
findings hold outside "text-dominant document" domains, or were they
specific to DocVQA/ChartQA's structure? Real OCR quality on this domain is
a genuine open question, not assumed — check `experiments/
textvqa_ocr_mechanism_check.py` before any real API spend.

Answers field has multiple human annotations per question (matches
DocVQA's `answers: list[str]` shape already expected by
`harness/judge.py`/`harness/metrics.py` — no dataset-specific branch
needed).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TextVQASample:
    question_id: str
    image_path: Path
    question: str
    answers: list[str]


def load_textvqa(
    split: str = "validation",
    limit: int | None = 20,
    cache_dir: str | Path = "data/textvqa_cache",
) -> list[TextVQASample]:
    """Stream real TextVQA samples from HuggingFace Hub and materialize
    images to disk, mirroring data.docvqa_loader.load_docvqa()'s pattern.
    `limit` defaults to 20, not None — hits the network and writes files,
    don't default to pulling the whole split unintentionally.
    """
    from datasets import load_dataset

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("lmms-lab/textvqa", split=split, streaming=True)

    samples = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break

        image_path = cache_dir / f"{row['question_id']}.png"
        if not image_path.exists():
            row["image"].convert("RGB").save(image_path)

        samples.append(
            TextVQASample(
                question_id=str(row["question_id"]),
                image_path=image_path,
                question=row["question"],
                answers=list(row["answers"]),
            )
        )
    return samples
