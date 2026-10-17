"""Judge-model fidelity scoring — the real implementation
`harness/metrics.answer_agreement()`'s docstring flagged as missing
("Replace with judge-model call for production use").

Why this exists: ANLS (harness/metrics.anls) penalizes a correct answer
wrapped in conversational filler ("The correct name is the **University
of California, San Diego**... Let me know if you need anything else!")
the same as a genuinely wrong one — found twice this session (§9.4's
"ITC Limited" case, and again in the text-route/min_confidence real-call
validation, paper/draft.md §12.4-12.5) even after stripping the qwen
<think> preamble. A judge model that's asked "does this answer the
question correctly, ignoring phrasing/verbosity" is not fooled by that.

Blind by construction: judge_correctness() only ever receives the
question, the ground-truth answers, and the candidate text — never which
condition (image-route vs text-route, raw vs filtered OCR) produced it,
so there is no route label for the judge to be biased by.
"""
from __future__ import annotations

from harness.metrics import _strip_think_block

JUDGE_PROMPT_TEMPLATE = """You are a strict but fair grading judge for a document visual question-answering benchmark.

Question: {question}
Ground-truth answer(s), any one of which counts as correct: {ground_truths}
Candidate answer: {candidate}

Judge whether the candidate answer is factually correct and would satisfy someone asking the question. Ignore differences in phrasing, verbosity, added commentary, capitalization, or formatting (e.g. bold markdown) - only the core factual content matters. A candidate that states the correct fact but wraps it in extra sentences is still CORRECT.

Respond with exactly one word: YES if correct, NO if incorrect."""


def judge_correctness(question: str, prediction: str, ground_truths: list[str], judge_provider) -> bool | None:
    """Ask `judge_provider` (any provider with call_text_only(), e.g.
    GeminiProvider) whether `prediction` correctly answers `question`
    against `ground_truths`. Returns None if the judge's response can't be
    parsed as YES/NO (surfaced as "unknown", not silently coerced to a
    score in either direction).

    Strips a closed <think> block from `prediction` before judging - the
    judge should see the candidate answer, not a reasoning-model's
    internal preamble (which can be long enough to dominate the judge
    prompt and isn't itself part of "the answer").

    Cached like any other call_text_only() call (same ResponseCache,
    keyed on the judge prompt's content hash) - re-scoring the same
    (question, prediction, ground_truths) tuple costs nothing after the
    first time.

    Assumes a Groq/OpenAI-shaped provider (passes reasoning_effort="low")
    - GeminiProvider's call_text_only would reject that kwarg. Only
    GroqProvider has been used as the judge so far; a Gemini judge would
    need this function's settings to branch on provider type.
    """
    cleaned_prediction = _strip_think_block(prediction) if prediction else prediction
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        ground_truths=" | ".join(ground_truths),
        candidate=cleaned_prediction,
    )
    # max_tokens=10 (the first thing tried) silently failed here: some Groq
    # judge models (openai/gpt-oss-120b included) burn hidden reasoning
    # tokens before the visible YES/NO, the same class of bug as qwen's
    # <think> preamble (§9.4) but on the OUTPUT budget of the judge itself
    # rather than the thing being judged - found when a full re-scoring
    # batch came back 60/60 unparseable (finish_reason=length, empty text).
    # Raising max_tokens to 100 alone still left ~18/60 unparseable (the
    # hidden reasoning budget is variable-length, not fixed) - the real fix
    # is reasoning_effort="low", which suppresses most of that hidden
    # reasoning outright (verified: one previously-failing case went from
    # 100 output tokens burned with empty text to 18 tokens with a clean
    # "YES"). max_tokens=100 is kept as a safety margin on top of that, not
    # load-bearing on its own.
    result = judge_provider.call_text_only("", prompt, max_tokens=100, reasoning_effort="low")
    verdict = (result.text or "").strip().upper()

    if verdict.startswith("YES"):
        return True
    if verdict.startswith("NO"):
        return False
    return None
