# Experiments

Live experiments call the provider APIs and log every call to `results/results.sqlite`
(responses cached in `results/cache.sqlite`). Run from the repository root. They need
API keys in `.env`; see the top-level README for quotas and model availability.

The offline analysis that regenerates the paper's numbers lives in `analysis/` and needs
none of these scripts to be re-run.

## Scripts behind the paper

| Paper | Script | Model(s) | Notes |
|---|---|---|---|
| IV-A, Table II | `week2_comparison_groq.py` | Groq Qwen3.6-27B | Pixel-space baselines on 3 DocVQA samples |
| IV-A, Table VI (Gemini column) | `week2_comparison.py` | Gemini 2.5 Flash | Same samples on Gemini |
| IV-A, Fig. 1; IV-B | none (ad hoc calls) | Gemini 2.5 Flash | The 12-call image-size sweep and the default vs. low resolution comparison were run interactively and not written to the result database; values and method are recorded in `archive/research_notebook.md` (Secs. 3.5, 7). Low-resolution image cost (66 image tokens per image) is visible in every cached low-resolution call. `packing_validation.py` is the related check of the local cost model |
| IV-C, token ratios | `text_route_real_validation.py`, `chartqa_text_route_validation.py` | Qwen3.6-27B, Gemini | Image route vs. OCR text route |
| IV-C, Table III, Fig. 2 | `judge_rescore.py`, `chartqa_judge_rescore.py`, `paired_fidelity_analysis.py` | judge: `openai/gpt-oss-120b` | Judge verdicts are cached; re-running makes no new calls |
| IV-C, bootstrap CIs | `evidence_strength_report.py` (uses `fit_text_route_advisor.py`) | none (offline) | Source of the 11.24x / 1.96x confidence intervals |
| IV-D, Table IV, Fig. 3 | `revision_table3.py` | Qwen3.8-27B, Gemini | Sep / Multi / Tiled / Per-tile, N=4 |
| IV-D, earlier tiling results | `tiling_mechanism_check.py`, `tiling_fidelity_check.py`, `batched_tile_qa_check.py`, `output_format_stacking_check.py` | Qwen3.6-27B, Gemini | N=2 to 12 composite costs and the N=6/8 ceiling |
| IV-E, caching | `cache_safety_analysis.py`, `cache_safety_analysis_query_aware.py`, `image_caching_viability_check.py` | none / Gemini, Groq | Saliency-similarity cache and provider prefix caching |
| VI-B, OpenRouter | `text_route_real_validation.py` (`--provider openrouter`) | nemotron models | Table VI OpenRouter columns |
| Limitations, cross-page dedup | `multipage_dedup_savings.py`, `cross_page_dedup_real_check.py` | none (offline) | n=1 document |

`cache_validation.py` is a pre-spend check that the response cache deduplicates calls.

## Exploratory scripts (not reported in the paper)

`cross_page_dedup_check.py`, `min_confidence_fidelity.py`, `ocr_compression_savings.py`,
`pdf_batched_text_route_check.py`, `rerun_region_ablations.py`, `textvqa_ocr_mechanism_check.py`,
`tiling_mode_compounding_check.py`, and the YAML configs. They are kept because the unit tests
import them and the research notebook refers to them.

## Model availability

Groq withdrew `qwen/qwen3.6-27b` on 2026-10-16. Every Qwen3.6 result (Table II, the text-route
fidelity of Sec. IV-C, and the N=6/8 ceiling) can no longer be re-run; the logged calls are the
record. Table IV was measured on the successor `qwen/qwen3.8-27b`, which `revision_table3.py`
uses by default.
