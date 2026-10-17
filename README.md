# Beyond Pixel Compression: Measuring Cost-Reduction Levers in Black-Box Multimodal APIs

[![tests](https://github.com/Mukilskanda/beyond-pixel-compression/actions/workflows/tests.yml/badge.svg)](https://github.com/Mukilskanda/beyond-pixel-compression/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Code and call logs for the IEEE INDICON 2026 paper (Paper ID 4684). The version described
in the paper is tag
[`v1.0-indicon2026`](https://github.com/Mukilskanda/beyond-pixel-compression/tree/v1.0-indicon2026).

Visual-token-reduction methods report savings in FLOPs and latency inside the model. This
project measures whether client-side image reduction lowers the input tokens that a hosted
API reports for billing, using the provider's own usage field on Google Gemini 2.5 Flash and
Groq-hosted Qwen models (free tier). It then measures which client-side levers do lower the
reported tokens, and what each costs in answer accuracy, judged by an LLM judge validated
against a 100-sample human audit.

## Key findings

| Finding | Headline numbers | Paper |
|---|---|---|
| Pixel reduction of a fixed image does not change reported tokens | Downscale, JPEG, crop and Groq's low-detail flag: token-identical to the unreduced image on Groq; Gemini flat at 258 image tokens from 10x10 to 8000x8000 px; Groq's per-image cost follows page shape, not pixel area (three levels across 172 images) | IV-A, Table II, Figs. 1-2 |
| Gemini's resolution mode is the one single-image setting that helps | 258 to 66 image tokens (3.9x) | IV-B |
| OCR text instead of the image | Groq: 11.24x fewer input tokens, 2.4 to 2.7x fewer total tokens; Gemini: no total-token saving (1.05x). 30 of 143 paired answers flip from correct to wrong, 2 the other way | IV-C, Table III, Fig. 3 |
| Four questions in one tiled call (N=4) | Groq: 3.24x fewer input tokens, 19/20 correct in both conditions; Gemini: 1.43x, 15/16 vs. 9/16 correct (Fisher p = 0.04) | IV-D, Table IV, Fig. 4 |
| Caching | Saliency-similarity cache correct on under 15% of hits; provider prefix caching unreachable | IV-E |

Flat image billing is endpoint-specific: on OpenRouter, a replacement model billed the same
three images with a 3.7x spread (Table VI).

## Repository layout

```
analysis/     offline scripts that regenerate every reported number and figure
audit/        100-sample human audit of the judge (seed 42) and the agreement script
prompts/      exact judge, batching and per-tile prompt templates
results/      results.sqlite (one row per API call), cache.sqlite (raw responses,
              judge verdicts, Gemini usage_metadata), revision_table3.jsonl (Table IV),
              docvqa_forward_flips.md (the 25 DocVQA flips), gemini_size_sweep.md (Fig. 1),
              groq_image_billing.csv (Fig. 2: size and billed tokens of 172 images)
experiments/  live experiment scripts, mapped to paper sections in experiments/README.md
src/          pipeline code: harness (providers, cache, judge, metrics, logger),
              router, textpath (OCR), pixelpath (tiling, region selection), profiler,
              baselines, data (dataset loaders)
data/         dataset download script and notes (datasets are not redistributed)
tests/        unit tests (pytest)
```

## Setup

Python 3.11 or newer (results were produced with 3.13).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

The OCR route also needs the Tesseract binary: `winget install UB-Mannheim.TesseractOCR`
(Windows), `brew install tesseract` (macOS) or `apt install tesseract-ocr` (Linux). It is
not needed for the offline reproduction.

## Reproduce the paper from the logged data (no API keys)

```bash
./run_all.sh tables     # regenerates every reported number and checks it against the paper (120 values)
./run_all.sh figures    # regenerates the paper's figures into figures/
./run_all.sh audit      # judge vs. human agreement and Cohen's kappa
./run_all.sh test       # unit tests
```

`make tables` and so on work as well. `analysis/reproduce_all.py` checks 126 values
(Tables II to IV and VI, Figs. 2 to 4, the token ratios and their bootstrap confidence
intervals, the McNemar and Fisher p-values, and the audit kappas) and reports MATCH or
MISMATCH for each. The paper rounds halves up.

One result is not in the result database and cannot be re-checked by the script: the
12-call Gemini image-size sweep (Fig. 1), which was run interactively; its values are
recorded in `results/gemini_size_sweep.md`. The Sec. IV-B resolution-mode figures (258 vs.
66 image tokens) are checked from the cached `usage_metadata` of every Gemini call.

### How tokens are counted

| Provider | Input | Billed output |
|---|---|---|
| Groq | `usage.prompt_tokens` | `usage.completion_tokens` (includes any reasoning text; Qwen3.8 returns no separate reasoning field) |
| Gemini | `usage_metadata.prompt_token_count` | `usage_metadata.candidates_token_count + thoughts_token_count` (thinking was left on, and Google bills it as output) |

The logger stored only `candidates_token_count` as Gemini output; the analysis scripts add
`thoughts_token_count` from the cached raw responses. Dollar figures apply published
paid-tier rates (Gemini 2.5 Flash $0.30/$2.50, Groq Qwen3.8-27B $0.80/$4.00 per 1M
input/output tokens, accessed 2026-10-16) to the logged counts; all calls ran on free tiers,
so no invoices exist.

### Which model produced which result

| Result | Model |
|---|---|
| Table II, text-route fidelity and token ratios (Sec. IV-C), N=6/8 ceiling | Groq `qwen/qwen3.6-27b` (withdrawn by Groq on 2026-10-16) |
| Table IV and Fig. 4 (batching) | Groq `qwen/qwen3.8-27b` |
| All Gemini results | `gemini-2.5-flash` |
| Fidelity judge | Groq `openai/gpt-oss-120b`, `reasoning_effort=low`, prompt in `prompts/judge_prompt.txt` |

Qwen3.6-27B can no longer be called, so the logged results are the record for those rows.

## Re-running live experiments (API keys needed)

Copy `.env.example` to `.env` and fill in the keys you need. `experiments/README.md` maps
each script to the paper section it produced. For example, Table IV:

```bash
python experiments/revision_table3.py --provider groq --dataset docvqa --offsets 0,4,8,12,16
python experiments/revision_table3.py --provider gemini --dataset docvqa --offsets 0,4,8,12 --conditions sep,multi,tiled
```

Every call is cached, so an interrupted run resumes without repeating calls. Free-tier
quotas limit how much runs per day (Gemini: 20 requests per model; Groq: 200,000 tokens per
rolling 24 hours). Hosted models are withdrawn without much notice: three were withdrawn
during this study (Groq Llama 4 Scout, an OpenRouter Nemotron model, and Groq Qwen3.6-27B),
so live re-runs may need a successor model and may not reproduce the logged numbers.

## Data and licensing

DocVQA and ChartQA are streamed from the HuggingFace Hub (`lmms-lab/DocVQA`,
`lmms-lab/ChartQA`) and are not redistributed here. The logs identify each sample by its
DocVQA `questionId` or ChartQA index; `python data/download_datasets.py` fetches the samples
used. Check the dataset cards for their terms before reuse.

Code is released under the MIT License (`LICENSE`).

## Limitations

- Two hosted APIs were measured in depth, plus OpenRouter as a partial check; results are
  specific to these endpoints and their behavior at the time of measurement.
- All calls used free tiers; paid-tier accounting may differ.
- Samples are small. Text-route fidelity: n = 99, 31 and 13 pairs per group (the ChartQA
  direction is not significant, McNemar p = 0.06). Table IV: 5 windows for Groq/DocVQA,
  4 for Gemini, 2 for Groq/ChartQA.
- The judge agrees with the human audit on 86/100 answers (kappa 0.63), less on DocVQA
  (kappa 0.41) than on ChartQA (0.74), and is slightly lenient.
- Gemini's dollar saving in Fig. 4 rests on thinking tokens, which vary by call (-17% to
  -51% across windows); treat it as indicative.

## Citation

```bibtex
@inproceedings{skanda2026beyondpixel,
  author    = {Mukil Skanda A and Mullunja Stuthi},
  title     = {Beyond Pixel Compression: Measuring Cost-Reduction Levers in Black-Box Multimodal {APIs}},
  booktitle = {Proc. 23rd IEEE India Council International Conference (INDICON)},
  address   = {Chennai, India},
  month     = dec,
  year      = {2026}
}
```

## Contact

Mukil Skanda A (corresponding author), mukil1336@gmail.com
Mullunja Stuthi, stuthi.mullunja@gmail.com

Dept. of Computer Science and Engineering, PES University, Bangalore, India.
