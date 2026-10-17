# Prompts

Exact prompt templates used in the paper, written out from the code that sent them.

| File | Used for | Source |
|---|---|---|
| `judge_prompt.txt` | Fidelity judge (`openai/gpt-oss-120b`, `reasoning_effort=low`); Sec. III-B | `src/harness/judge.py` (`JUDGE_PROMPT_TEMPLATE`) |
| `batching_prompt_tiled.txt` | One tiled call, four questions, JSON answer (Table IV "Tiled") | `experiments/batched_tile_qa_check.py` (`build_prompt`) |
| `batching_prompt_untiled.txt` | Four untiled images in one call (Table IV "Multi") | `experiments/revision_table3.py` (`multi_prompt`) |
| `per_tile_prompt.txt` | One question per call on the tiled composite (Table IV "Per-tile") | `experiments/revision_table3.py`, `experiments/tiling_fidelity_check.py` |

Single-image calls (Table IV "Sep.", unreduced baselines) send the dataset question verbatim with no template.
The judge strips any `<think>...</think>` block from the candidate before filling `{candidate}`.
