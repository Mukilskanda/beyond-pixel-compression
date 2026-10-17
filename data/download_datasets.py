"""Download the DocVQA and ChartQA samples used in the paper into data/*_cache/.

Streams from the HuggingFace Hub (no account needed) via the loaders in
src/data/, in loader order, until every image referenced by a logged call in
results/results.sqlite is present (capped at 1,000 samples per dataset).

Usage (from the repository root): python data/download_datasets.py
"""
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from data.chartqa_loader import load_chartqa  # noqa: E402
from data.docvqa_loader import load_docvqa  # noqa: E402


def needed(folder: str) -> set[str]:
    rows = sqlite3.connect(ROOT / "results/results.sqlite").execute("SELECT DISTINCT image_path FROM runs").fetchall()
    return {Path(p.replace("\\", "/")).name for (p,) in rows if f"{folder}/" in p.replace("\\", "/")}


def fetch(loader, folder: str) -> None:
    want = needed(folder)
    limit = 50
    while True:
        have = {s.image_path.name for s in loader(limit=limit)}
        missing = want - have
        if not missing or limit >= 1000:
            break
        limit *= 2
    print(f"{folder}: {len(want) - len(missing)}/{len(want)} logged images present (loaded {limit} samples)")


if __name__ == "__main__":
    fetch(load_docvqa, "docvqa_cache")
    fetch(load_chartqa, "chartqa_cache")
