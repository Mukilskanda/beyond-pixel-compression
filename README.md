# Beyond Pixel Compression: Measuring Cost-Reduction Levers in Black-Box Multimodal APIs

Code, call logs and paper source for the IEEE INDICON 2026 paper (provisionally accepted, Paper ID 4684).

**Question.** Visual-token-reduction methods (pruning, merging, optical compression) report savings in FLOPs and latency inside the model. Does reducing an image on the client lower the input tokens that a *hosted* API reports for billing?

**Method.** Every number comes from the provider's own usage field (`usage.prompt_tokens` on Groq, `usage_metadata.prompt_token_count` on Gemini), logged per call. Billed output is `completion_tokens` on Groq and answer plus thinking tokens (`thoughts_token_count`) on Gemini; `experiments/revision_gemini_thinking.py` and `experiments/revision_textroute_total.py` recompute the thinking-inclusive figures from the cache. All calls used free-tier access, so "cost" means reported tokens; dollar figures apply published paid-tier rates to those counts.

## Findings

| Lever | Result | Where |
|---|---|---|
| Downscale, JPEG, crop, low-detail flag | No change in reported image tokens on Gemini 2.5 Flash or Groq Qwen | Sec. IV-A, Table II |
| Gemini `media_resolution=LOW` | 258 → 74 tokens for the same image (3.5×) | Sec. IV-B |
| OCR text instead of the image | 11.24× fewer input tokens on Groq/DocVQA (2.4–2.7× counting output tokens; no total-token saving on Gemini), but 30 of 143 paired answers flipped from correct to wrong (2 the other way) | Sec. IV-C, Table III |
| 4 questions in one tiled call (N=4) | 3.24× fewer input tokens on Groq with unchanged accuracy; 1.43× on Gemini with lower observed accuracy (9/16 vs. 15/16), against directly measured separate calls | Sec. IV-D, Table IV |
| Caching (saliency similarity; provider prefix cache) | Unsafe or unreachable | Sec. IV-E |

Flat image billing is endpoint-specific: an OpenRouter model billed three images flat, while its replacement varied by a factor of 3.7 across the same images (Sec. VI). Re-measure before relying on any of these results; three of the models used here were withdrawn during the study.

## Repository layout

```
paper/indicon_revision/    Revised paper: main.tex, refs.bib, figures/, make_figures_rev.py, main.pdf
paper/indicon_submission/  Submitted (anonymised) version and the 100-row human audit of the judge
paper/draft.md             Full research notebook: every experiment, dead end and correction, in order
results/results.sqlite     One row per API call: tokens in/out, finish reason, response, timing
results/cache.sqlite       Content-hash response cache, including all judge verdicts
results/revision_table3.jsonl  Per-window results behind Table IV
harness/                   Provider wrappers, cache, judge, metrics, result logger
experiments/               One script per experiment (see table below)
textpath/  pixelpath/      OCR route; tiling and region selection
router/  profiler/  baselines/   Routing rules, image features, the six baselines
tests/                     212 unit tests (pytest), run in CI
extension/                 TileSaver: a browser extension that applies the tiling result to uploads
```

## Reproducing the paper

### 1. Setup

```bash
pip install -r requirements.txt
```

Tesseract is needed for the OCR route: `winget install UB-Mannheim.TesseractOCR` (Windows), `brew install tesseract` (macOS) or `apt install tesseract-ocr` (Linux).

### 2. Offline: recompute the reported numbers from the logs (no API key)

```bash
python experiments/revision_recount.py
```

Reads `results/results.sqlite` and the cached judge verdicts in `results/cache.sqlite`, and prints the paired fidelity counts (143 / 146 pairs, 30 / 32 forward flips, 2 reversals), absolute accuracies (Table III), the 101 cost-ratio pairs, and Cohen's kappa for the human audit (0.63, 86/100 agreement). No API calls are made.

```bash
cd paper/indicon_revision && python make_figures_rev.py
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Regenerates Figs. 1-3 from the logged data and builds the PDF.

### 3. Live: re-run the experiments (API keys required)

Copy `.env.example` to `.env` and fill in the keys you have (Gemini and Groq both offer free tiers). Datasets (DocVQA, ChartQA) are streamed from the HuggingFace Hub on first use and cached under `data/`.

| Paper element | Command |
|---|---|
| Table IV (batching, N=4) | `python experiments/revision_table3.py --provider groq --dataset docvqa --offsets 0,4,8,12,16`<br>`python experiments/revision_table3.py --provider gemini --dataset docvqa --offsets 0,4,8 --conditions sep,multi,tiled` |
| Table II (pixel-space baselines) | `python experiments/week2_comparison_groq.py` |
| Text route (Sec. IV-C) | `python experiments/text_route_real_validation.py`, then `python experiments/paired_fidelity_analysis.py` |
| Flip mechanism (25 DocVQA flips) | `python experiments/revision_flip_mechanism.py` |
| Bootstrap CIs | `python experiments/evidence_strength_report.py` |

Free-tier quotas bound how much can run per day (Gemini: 20 requests per model; Groq: 200,000 tokens). Every call is cached, so an interrupted run resumes without paying twice. Groq retired `qwen/qwen3.6-27b` on 2026-10-16; the scripts default to its successor, `qwen/qwen3.8-27b`.

## Fidelity judge

Answers are scored by `openai/gpt-oss-120b` on Groq (`harness/judge.py`), which sees only the question, the ground-truth answers and the candidate answer, never the condition that produced it. A 100-row human audit (`paper/indicon_submission/judge_validation_sample_scored.csv`) agrees on 86 rows (Cohen's kappa 0.63).

## Citation

```bibtex
@inproceedings{skanda2026beyondpixel,
  author    = {Mukil Skanda A},
  title     = {Beyond Pixel Compression: Measuring Cost-Reduction Levers in Black-Box Multimodal {APIs}},
  booktitle = {Proc. 23rd IEEE India Council International Conference (INDICON)},
  year      = {2026}
}
```

## License

Code is released under the MIT License (see `LICENSE`). DocVQA and ChartQA are not redistributed here; they are downloaded from their sources under their own terms.
