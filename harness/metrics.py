"""Fidelity metrics: ANLS, exact match, answer agreement."""
from __future__ import annotations

import re


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def exact_match(prediction: str, ground_truths: list[str]) -> bool:
    pred = _normalize(prediction)
    return any(pred == _normalize(gt) for gt in ground_truths)


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = curr
    return prev[-1]


def anls(prediction: str, ground_truths: list[str], threshold: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity — standard DocVQA metric."""
    pred = _normalize(prediction)
    best = 0.0
    for gt in ground_truths:
        gt_n = _normalize(gt)
        max_len = max(len(pred), len(gt_n))
        if max_len == 0:
            sim = 1.0
        else:
            sim = 1.0 - _levenshtein(pred, gt_n) / max_len
        best = max(best, sim)
    return best if best >= threshold else 0.0


def answer_agreement(reduced_answer: str, unreduced_answer: str) -> bool:
    """Cheap proxy: exact-match after normalization. Replace with judge-model call for production use."""
    return _normalize(reduced_answer) == _normalize(unreduced_answer)


# Severity-weighted error (plan §3.3): "a wrong number in a financial table
# != a slightly-off description." This taxonomy is a starting point — the
# plan calls for hand-labeling severity on a sample; these categories are
# what that labeling should choose among, not a substitute for it.
class Severity:
    CRITICAL = 3   # wrong number/fact that would change a decision (e.g. financial totals)
    MODERATE = 2   # wrong but non-decision-critical detail (e.g. minor description error)
    MINOR = 1      # phrasing/formatting difference, same substantive answer
    NONE = 0       # correct


def severity_weighted_score(is_correct: bool, severity_if_wrong: int = Severity.MODERATE) -> int:
    """Returns the penalty for one prediction. Aggregate (sum or mean) across
    a sample to get a severity-weighted error rate — lower is better, unlike
    ANLS/exact-match where higher is better. `severity_if_wrong` must come
    from hand-labeling (plan §3.3), not be inferred automatically — there is
    no reliable automatic signal for "this error would mislead a decision."
    """
    return Severity.NONE if is_correct else severity_if_wrong
