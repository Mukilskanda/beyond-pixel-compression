"""INDICON 2026 revision: offline recount of every disputed fidelity number
(Reviewer 1, point 2 and point 4). Zero API calls. Judge verdicts are read
straight out of results/cache.sqlite (openai/gpt-oss-120b entries), matched
to results/results.sqlite rows by (question, think-stripped candidate).

Reproduces the selection rule of experiments/evidence_strength_report.py
(most-complete non-truncated row per (image, method, model); text route =
text_compressed if judged, else text_raw) and reports, per group:
paired n, forward flips, reversals, absolute image/text accuracy.

Also computes Cohen's kappa for the 100-row human audit.

Usage: python experiments/revision_recount.py
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harness.metrics import _strip_think_block

Q_RE = re.compile(
    r"Question: (.*?)\nGround-truth answer\(s\), any one of which counts as correct: (.*?)\nCandidate answer: (.*)\n\nJudge whether",
    re.S,
)


def load_judge_cache() -> dict[tuple[str, str], tuple[bool | None, str]]:
    conn = sqlite3.connect("results/cache.sqlite")
    out = {}
    for settings_json, response_json in conn.execute(
        "SELECT settings_json, response_json FROM cache WHERE model = 'openai/gpt-oss-120b'"
    ):
        m = Q_RE.search(json.loads(settings_json)["prompt"])
        if not m:
            continue
        question, gts, candidate = m.groups()
        text = (json.loads(response_json).get("text") or "").strip().upper()
        verdict = True if text.startswith("YES") else False if text.startswith("NO") else None
        out[(question, candidate)] = (verdict, gts)
    return out


def best_rows():
    conn = sqlite3.connect("results/results.sqlite")
    rows = conn.execute(
        "SELECT dataset, image_path, method, model, prompt, response_text, input_tokens, output_tokens FROM runs "
        "WHERE method IN ('unreduced','text_raw','text_compressed') AND truncated = 0 "
        "AND dataset IN ('docvqa','chartqa') ORDER BY output_tokens DESC"
    ).fetchall()
    best = {}
    for r in rows:
        key = (r[1], r[2], r[3])
        if key not in best:
            best[key] = r
    return best


def wilson(x, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z / d) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return c - h, c + h


def main() -> None:
    judge = load_judge_cache()
    best = best_rows()
    verdicts = {}
    missing = 0
    for key, r in best.items():
        cand = _strip_think_block(r[5]) if r[5] else (r[5] or "")
        hit = judge.get((r[4], cand))
        if hit is None:
            missing += 1
            continue
        verdicts[key] = (hit[0], r[0])
    print(f"rows with a cached verdict: {len(verdicts)}  (no cached verdict: {missing})")

    groups = defaultdict(lambda: defaultdict(int))
    unknown = defaultdict(int)
    pairs = sorted({(k[0], k[2]) for k in verdicts})
    for image_path, model in pairs:
        ur = verdicts.get((image_path, "unreduced", model))
        tm = "text_compressed" if (image_path, "text_compressed", model) in verdicts else "text_raw"
        tr = verdicts.get((image_path, tm, model))
        if ur is None or tr is None:
            continue
        ds = ur[1]
        g = groups[(ds, model)]
        if ur[0] is None or tr[0] is None:
            unknown[(ds, model)] += 1
            continue
        g["n"] += 1
        g["img_ok"] += ur[0]
        g["txt_ok"] += tr[0]
        if ur[0] and not tr[0]:
            g["forward"] += 1
        elif tr[0] and not ur[0]:
            g["reverse"] += 1
        elif ur[0] and tr[0]:
            g["both_ok"] += 1
        else:
            g["both_wrong"] += 1

    tot = defaultdict(int)
    print(
        f"\n{'group':<34}{'n':>4}{'fwd':>5}{'rev':>5}{'both+':>6}{'both-':>6}{'img acc':>9}{'txt acc':>9}{'unk':>5}  fwd Wilson95"
    )
    for (ds, model), g in sorted(groups.items()):
        lo, hi = wilson(g["forward"], g["n"])
        print(
            f"{ds + '/' + model:<34}{g['n']:>4}{g['forward']:>5}{g['reverse']:>5}{g['both_ok']:>6}{g['both_wrong']:>6}"
            f"{100 * g['img_ok'] / g['n']:>8.1f}%{100 * g['txt_ok'] / g['n']:>8.1f}%{unknown[(ds, model)]:>5}  [{100 * lo:.1f}, {100 * hi:.1f}]"
        )
        for k, v in g.items():
            tot[k] += v
    print(
        f"{'TOTAL':<34}{tot['n']:>4}{tot['forward']:>5}{tot['reverse']:>5}{tot['both_ok']:>6}{tot['both_wrong']:>6}"
        f"  unknown pairs={sum(unknown.values())}"
    )

    # Token-reduction pairs (text route ratio): count per provider/dataset
    conn = sqlite3.connect("results/results.sqlite")
    by = defaultdict(dict)
    for ds, ip, m, mo, t in conn.execute(
        "SELECT dataset, image_path, method, model, input_tokens FROM runs "
        "WHERE method IN ('unreduced','text_compressed') AND truncated = 0"
    ):
        by[(ds, ip, mo)][m] = t
    cnt = defaultdict(int)
    for (ds, ip, mo), mm in by.items():
        if "unreduced" in mm and "text_compressed" in mm and mm["text_compressed"] > 0:
            cnt[(ds, mo)] += 1
    print("\ncost-ratio pairs (unreduced & text_compressed, non-truncated):", dict(cnt))

    # Cohen's kappa for human audit. CSV stores human agreement with the
    # judge (1/0); reconstruct the judge verdict from the cache, then human
    # label = judge verdict if agree else its negation.
    path = Path("audit/judge_validation_sample_scored.csv")
    with open(path, encoding="utf-8") as f:
        audit = list(csv.DictReader(f))
    conn = sqlite3.connect("results/results.sqlite")
    jv, hv = [], []
    unmatched = 0
    for a in audit:
        row = conn.execute("SELECT prompt, response_text FROM runs WHERE id = ?", (int(a["row_id"]),)).fetchone()
        if row is None:
            unmatched += 1
            continue
        cand = _strip_think_block(row[1]) if row[1] else (row[1] or "")
        hit = judge.get((row[0], cand))
        if hit is None or hit[0] is None:
            unmatched += 1
            continue
        j = hit[0]
        agree = a["human_correct_1_or_0"].strip() == "1"
        jv.append(j)
        hv.append(j if agree else (not j))
    n = len(jv)
    po = sum(1 for x, y in zip(jv, hv) if x == y) / n
    pj, ph = sum(jv) / n, sum(hv) / n
    pe = pj * ph + (1 - pj) * (1 - ph)
    kappa = (po - pe) / (1 - pe)
    print(f"\nHuman audit: rows={len(audit)} matched={n} unmatched={unmatched}")
    print(
        f"  agreement={100 * po:.1f}%  judge YES rate={100 * pj:.1f}%  human YES rate={100 * ph:.1f}%  kappa={kappa:.3f}"
    )
    tp = sum(1 for x, y in zip(jv, hv) if x and y)
    fp = sum(1 for x, y in zip(jv, hv) if x and not y)
    fn = sum(1 for x, y in zip(jv, hv) if not x and y)
    tn = sum(1 for x, y in zip(jv, hv) if not x and not y)
    print(f"  confusion (judge,human): YY={tp} YN={fp} NY={fn} NN={tn}")
    print(
        "  audit composition:",
        dict(
            sorted(
                defaultdict(
                    int,
                    {
                        k: sum(1 for a in audit if (a["dataset"], a["method"]) == k)
                        for k in {(a["dataset"], a["method"]) for a in audit}
                    },
                ).items()
            )
        ),
    )


if __name__ == "__main__":
    main()


def export_flips(path: str = "results/docvqa_forward_flips.md") -> None:
    """Write every DocVQA forward flip (image correct, text wrong) for manual review."""
    judge = load_judge_cache()
    best = best_rows()
    v = {}
    for key, r in best.items():
        cand = _strip_think_block(r[5]) if r[5] else (r[5] or "")
        hit = judge.get((r[4], cand))
        if hit:
            v[key] = (hit[0], hit[1], r[4], cand)
    out = ["# DocVQA forward flips (image-route correct, text-route wrong)\n"]
    k = 0
    for (ip, method, model), (ok, gts, q, cand) in sorted(v.items()):
        if method != "unreduced" or not ok or "chartqa" in ip:
            continue
        tm = "text_compressed" if (ip, "text_compressed", model) in v else "text_raw"
        t = v.get((ip, tm, model))
        if t is None or t[0] is not False:
            continue
        k += 1
        out.append(
            f"## {k}. {model} | {ip}\n- Q: {q}\n- GT: {gts}\n- Image route: {cand[:400]}\n- Text route ({tm}): {t[3][:400]}\n- Failure mode (fill in: LAYOUT / EXTRACTION): \n"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {k} flips to {path}")
