# blackbox-vtr

Client-side visual token reduction under black-box access to multimodal LLM APIs.

## Status: Week 1 (harness)

## Setup

```bash
pip install -r requirements.txt
```

Tesseract OCR binary is required for `textpath/` (separate from the `pytesseract` pip package):
- Windows: `winget install --id UB-Mannheim.TesseractOCR`
- macOS: `brew install tesseract`
- Linux: `apt install tesseract-ocr`

### DocVQA (no registration needed)

Loaded via HuggingFace Hub (`lmms-lab/DocVQA`) — same underlying dataset as the
official rrc.cvc.uab.es release, no account or manual download required:

```python
from data.docvqa_loader import load_docvqa
samples = load_docvqa(limit=50)  # streams + caches images to data/docvqa_cache/
```

The official manual-download path (`load_docvqa_official()` in the same file) is
kept for anyone who already has the original release, or wants the exact
citation-standard file layout — not needed for normal use.

Local dev/test model: install [Ollama](https://ollama.ai) and pull a vision model
(`ollama pull llava` or `ollama pull moondream`) for free, unlimited pipeline iteration.
Real measurement runs (the numbers that go in the paper) use
`harness/providers.py:AnthropicProvider` instead — token counts must come from the
real API's `usage` field, not an estimate.

## Before spending real API budget

**Never approve open-ended spend.** Approve a specific dollar cap for a specific
experiment (e.g. "$15 for the week-2 baseline comparison on 50 samples"), not a
blanket "go ahead."

Before any real run, prove the cache actually dedupes repeated calls — that's
what keeps re-runs and debugging free after the first pass:

```bash
python experiments/cache_validation.py --model claude-opus-5 --n 5
```

This calls the real API twice on the same 5 samples. Pass 1 is real cost (small —
5 calls). Pass 2 should be near-instant with zero additional spend. If it isn't,
something is wrong with the cache key construction — stop and fix it before
approving any larger run.

## Layout

See `blackbox-visual-token-reduction-plan.md` (parent directory) for the full project plan.

```
harness/       cache, providers, metrics, result DB, orchestration
profiler/      Stage 1: cheap local image features (<15ms, never calls the API)
router/        Stage 2: text/pixel/hybrid routing (rules v1, learned v2 later)
textpath/      Stage 3a: OCR -> structured markdown
pixelpath/     Stage 3b: saliency, region proposal, greedy/DP knapsack, tiling cost model
baselines/     All 6 non-trivial baselines from plan §3.2
experiments/   Per-experiment configs
results/       SQLite (cache.sqlite, results.sqlite) + generated plots
data/          Dataset loaders and local extractions (gitignored)
```

## Quick smoke test

```bash
python -c "
from PIL import Image, ImageDraw
img = Image.new('RGB', (800, 600), 'white')
ImageDraw.Draw(img).text((50,50), 'test invoice', fill='black')
img.save('data/scratch/test.png')

from profiler.features import profile_image
from router.rules import route
print(route(profile_image('data/scratch/test.png'), 'what is the total?'))
"
```
