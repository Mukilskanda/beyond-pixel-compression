"""INDICON 2026 revision: Gemini 2.5 Flash bills thinking tokens as output,
but CallResult.output_tokens records only candidates_token_count. This
recomputes Table IV's Gemini output column with thoughts_token_count
added, from cached responses only (a cache miss raises instead of calling
the API).

Usage: python experiments/revision_gemini_thinking.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from baselines.baselines import GEMINI_LOW_DETAIL_SETTINGS
from data.docvqa_loader import load_docvqa
from experiments.batched_tile_qa_check import build_prompt
from harness.cache import ResponseCache

N = 4
MODEL = "gemini-2.5-flash"


def usage(cache: ResponseCache, image_path, prompt: str) -> tuple[int, int, int]:
    settings = {"prompt": prompt, **GEMINI_LOW_DETAIL_SETTINGS}
    hit = cache.get(Path(image_path).read_bytes(), MODEL, settings)
    if hit is None:
        raise RuntimeError(f"cache miss for {image_path}: refusing to call the API")
    um = (hit.get("raw") or {}).get("usage_metadata") or {}
    return hit["input_tokens"], um.get("candidates_token_count") or 0, um.get("thoughts_token_count") or 0


def main() -> None:
    cache = ResponseCache("results/cache.sqlite")
    agg = {"sep": [], "tiled": []}
    for off in (0, 4, 8, 12):
        samples = load_docvqa(limit=off + N)[off:off + N]
        sep = [usage(cache, s.image_path, s.question) for s in samples]
        grid = Path("data/scratch") / f"batched_tile_n{N}_gemini_docvqa_o{off}_base.png"
        if off == 0 and not grid.exists():
            grid = Path("data/scratch/batched_tile_n4_gemini.png")
        tiled = usage(cache, grid, build_prompt(N, [s.question for s in samples]))
        s_in, s_ans, s_th = (sum(x[i] for x in sep) for i in range(3))
        print(f"o{off:<3} sep in={s_in} answer={s_ans} thinking={s_th} | tiled in={tiled[0]} answer={tiled[1]} thinking={tiled[2]}")
        agg["sep"].append((s_in, s_ans, s_th))
        agg["tiled"].append(tiled)
    for k, v in agg.items():
        n = len(v)
        i, a, t = (sum(x[j] for x in v) / n for j in range(3))
        usd = (i * 0.30 + (a + t) * 2.50) / 1000
        print(f"{k:<6} mean in={i:.1f} answer={a:.1f} thinking={t:.1f} output(total)={a + t:.1f}  USD/1k={usd:.3f}")


if __name__ == "__main__":
    main()
