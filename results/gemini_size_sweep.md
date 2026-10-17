# Gemini 2.5 Flash image-size sweep (Fig. 1, Sec. IV-A)

These calls were made interactively and were not written to `results/results.sqlite`,
so `analysis/reproduce_all.py` cannot re-check them. Values as recorded at the time
(August 2026), from the provider's `usage_metadata`.

| Image | Reported tokens |
|---|---|
| 10x10 px | 258 image tokens |
| 32x32 px | 258 image tokens |
| 100x100 px | 266 total prompt tokens (258 image + 8 prompt) |
| 3000x4000 px | 266 total prompt tokens (258 image + 8 prompt) |
| 8000x8000 px | 258 image tokens |
| 1000x1000 px, blank white | identical to the random-noise image below |
| 1000x1000 px, dense random noise | identical to the blank image above |

The DocVQA pages in the cost-model validation (`experiments/packing_validation.py`,
sizes up to 1653x2339 px) also reported 258 image tokens each.

Client-side resizing was ruled out by reading the `google-genai` SDK: its PIL-to-`Part`
conversion saves the image at full resolution with no resize step.

Google's documentation at the time stated: "Images ≤384 pixels in both dimensions count
as 258 tokens. Larger images are tiled into 768x768 pixel tiles, each counting as 258
tokens." (https://ai.google.dev/gemini-api/docs/tokens). The measured cost did not
follow that rule.

The resolution-mode comparison of Sec. IV-B (258 default vs. 66 low-resolution image
tokens) is in the cached `usage_metadata` of every logged Gemini call and is checked
by `analysis/reproduce_all.py`.
