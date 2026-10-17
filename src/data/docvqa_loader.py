"""DocVQA loader via HuggingFace Hub (lmms-lab/DocVQA) — no manual registration
or account required, unlike the official rrc.cvc.uab.es download. Same
underlying dataset (document scans + questions + human answers), streamed
on demand and cached locally by `datasets` after first access.

Note: image field is a decoded PIL.Image, not a file path — samples are
materialized to disk under `data/docvqa_cache/` on first load so the rest of
the pipeline (which expects file paths) can use them unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DocVQASample:
    question_id: str
    image_path: Path
    question: str
    answers: list[str]


def load_docvqa(
    split: str = "validation",
    limit: int | None = 50,
    cache_dir: str | Path = "data/docvqa_cache",
) -> list[DocVQASample]:
    """Stream real DocVQA samples from HuggingFace Hub and materialize images
    to disk. `limit` defaults to 50, not None — this hits the network and
    writes files, so don't default to pulling the whole split unintentionally.
    """
    from datasets import load_dataset

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split=split, streaming=True)

    samples = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break

        image_path = cache_dir / f"{row['questionId']}.png"
        if not image_path.exists():
            row["image"].save(image_path)

        samples.append(
            DocVQASample(
                question_id=str(row["questionId"]),
                image_path=image_path,
                question=row["question"],
                answers=row.get("answers", []),
            )
        )

    return samples


# --- Legacy path: official rrc.cvc.uab.es manual download ------------------
# Kept for anyone who already has the official release extracted locally, or
# wants the exact original file layout for reproducibility citations. Not
# needed for normal use — load_docvqa() above is the recommended path.
def load_docvqa_official(
    root: str | Path = "data/docvqa", split: str = "val_v1.0", limit: int | None = None
) -> list[DocVQASample]:
    import json

    root = Path(root)
    ann_path = root / f"{split}.json"
    if not ann_path.exists():
        raise FileNotFoundError(
            f"{ann_path} not found. Download DocVQA from https://rrc.cvc.uab.es/?ch=17 "
            f"and extract images/ + {split}.json into {root}/ — or use load_docvqa() "
            f"instead, which needs no registration."
        )

    with open(ann_path, encoding="utf-8") as f:
        raw = json.load(f)

    samples = []
    for entry in raw["data"]:
        image_path = root / "images" / entry["image"]
        samples.append(
            DocVQASample(
                question_id=str(entry["questionId"]),
                image_path=image_path,
                question=entry["question"],
                answers=entry.get("answers", []),
            )
        )
        if limit is not None and len(samples) >= limit:
            break

    return samples
