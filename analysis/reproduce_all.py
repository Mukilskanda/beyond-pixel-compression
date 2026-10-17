"""Regenerate every number reported in the paper from the logged data and
compare it with the value printed in the paper. No API calls.

Inputs:  results/results.sqlite      one row per API call (tokens, response)
         results/cache.sqlite        cached raw responses, incl. judge verdicts
                                     and Gemini usage_metadata
         results/revision_table3.jsonl  per-window Table IV records
         audit/judge_validation_sample_scored.csv  100-row human audit
Output:  a report on stdout, one line per checked value (MATCH / MISMATCH),
         and the paper figures via analysis/make_figures.py.

Output-token definition used for billing (see README):
  Groq   = usage.completion_tokens (includes any reasoning text)
  Gemini = usage_metadata.candidates_token_count + thoughts_token_count
           (thinking is on by default and billed as output)

Usage (from the repository root): python analysis/reproduce_all.py
"""

from __future__ import annotations

import csv
import json
import sqlite3
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "analysis")]

import numpy as np  # noqa: E402

from harness.metrics import _strip_think_block  # noqa: E402
from revision_recount import best_rows, load_judge_cache  # noqa: E402

RESULTS = ROOT / "results/results.sqlite"
CACHE = ROOT / "results/cache.sqlite"
JSONL = ROOT / "results/revision_table3.jsonl"
AUDIT = ROOT / "audit/judge_validation_sample_scored.csv"

checks: list[tuple[str, str, str, bool]] = []


def rhu(x: float, nd: int = 0):
    """Round half up, the convention used in the paper (Python's round() is half-to-even)."""
    from decimal import ROUND_HALF_UP, Decimal

    q = Decimal(1).scaleb(-nd)
    v = Decimal(repr(x)).quantize(q, rounding=ROUND_HALF_UP)
    return int(v) if nd == 0 else float(v)


def check(label: str, got, paper, tol: float = 0.0) -> None:
    if isinstance(paper, (int, float)) and isinstance(got, (int, float)):
        ok = abs(got - paper) <= tol
        g = f"{got:.4g}" if isinstance(got, float) else str(got)
    else:
        ok = str(got) == str(paper)
        g = str(got)
    checks.append((label, g, str(paper), ok))


def section(title: str) -> None:
    checks.append((f"## {title}", "", "", True))


# ---------------------------------------------------------------- Table II
def table2() -> None:
    section("Table II: Groq pixel-space baselines (Qwen3.6-27B)")
    c = sqlite3.connect(RESULTS)
    rows = c.execute("SELECT image_path, method, input_tokens FROM runs WHERE dataset='docvqa_week2_groq'").fetchall()
    by = defaultdict(dict)
    for ip, m, t in rows:
        by[Path(ip.replace("\\", "/")).stem][m] = t
    paper = {"49153": (1827, 1926), "24580": (1810, 1830), "57349": (1812, 2027)}
    for img, (unred, ocr) in paper.items():
        d = by[img]
        check(f"{img} unreduced", d["unreduced"], unred)
        for m in ("uniform_downscale_50", "jpeg_quality_50", "fixed_ratio_crop_60", "low_detail"):
            check(f"{img} {m} == unreduced", d[m], unred)
        check(f"{img} ocr_everything", d["ocr_everything"], ocr)
        check(f"{img} hybrid", d["full_pipeline_hybrid"], ocr)


# --------------------------------------------------------------- Table III
def table3_and_tests() -> None:
    section("Table III / Fig. 3: paired text-route fidelity")
    judge = load_judge_cache()
    verdicts = {}
    for key, r in best_rows().items():
        cand = _strip_think_block(r[5]) if r[5] else (r[5] or "")
        hit = judge.get((r[4], cand))
        if hit is not None:
            verdicts[key] = (hit[0], r[0])
    groups = defaultdict(lambda: defaultdict(int))
    for img, model in sorted({(k[0], k[2]) for k in verdicts}):
        ur = verdicts.get((img, "unreduced", model))
        tm = "text_compressed" if (img, "text_compressed", model) in verdicts else "text_raw"
        tr = verdicts.get((img, tm, model))
        if ur is None or tr is None or ur[0] is None or tr[0] is None:
            continue
        g = groups[(ur[1], model)]
        g["n"] += 1
        g["img"] += ur[0]
        g["txt"] += tr[0]
        g["fwd"] += bool(ur[0] and not tr[0])
        g["rev"] += bool(tr[0] and not ur[0])
    paper = {
        ("docvqa", "qwen/qwen3.6-27b"): (99, 97.0, 84.8, 14, 2),
        ("docvqa", "gemini-2.5-flash"): (31, 96.8, 61.3, 11, 0),
        ("chartqa", "qwen/qwen3.6-27b"): (13, 100.0, 61.5, 5, 0),
        ("chartqa", "gemini-2.5-flash"): (3, 100.0, 33.3, 2, 0),
    }
    tot = defaultdict(int)
    for key, (n, ia, ta, f, rv) in paper.items():
        g = groups[key]
        name = f"{key[0]}/{key[1].split('/')[-1]}"
        check(f"{name} n", g["n"], n)
        check(f"{name} image acc %", round(100 * g["img"] / g["n"], 1), ia)
        check(f"{name} text acc %", round(100 * g["txt"] / g["n"], 1), ta)
        check(f"{name} forward flips", g["fwd"], f)
        check(f"{name} reversals", g["rev"], rv)
        if key != ("chartqa", "gemini-2.5-flash"):
            for k in ("n", "fwd", "rev"):
                tot[k] += g[k]
    check("143-pair total n", tot["n"], 143)
    check("143-pair forward flips", tot["fwd"], 30)
    check("146-pair forward flips", tot["fwd"] + groups[("chartqa", "gemini-2.5-flash")]["fwd"], 32)

    section("Exact tests (Limitations, Sec. IV-D)")

    def mcnemar(b, c):
        n = b + c
        return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n)

    for key, label, p in [
        (("docvqa", "qwen/qwen3.6-27b"), "Groq/DocVQA", 0.004),
        (("docvqa", "gemini-2.5-flash"), "Gemini/DocVQA", 0.001),
        (("chartqa", "qwen/qwen3.6-27b"), "Groq/ChartQA", 0.06),
    ]:
        g = groups[key]
        got = mcnemar(g["fwd"], g["rev"])
        if label == "Gemini/DocVQA":
            check(f"McNemar {label} p < 0.001", got < 0.001, True)
        else:
            check(f"McNemar {label} p", round(got, 3 if p < 0.01 else 2), p)

    def fisher(a, b, c, d):
        n1, n2, k = a + b, c + d, a + c
        p = lambda x: comb(n1, x) * comb(n2, k - x) / comb(n1 + n2, k)  # noqa: E731
        p0 = p(a)
        return sum(p(x) for x in range(max(0, k - n2), min(k, n1) + 1) if p(x) <= p0 + 1e-12)

    check("Fisher Gemini Table IV 15/16 vs 9/16", round(fisher(15, 1, 9, 7), 2), 0.04)


# ------------------------------------------------------------- ratios / CIs
def text_route_ratios() -> None:
    section("Sec. IV-C: text-route token ratios (bootstrap 95% CI)")
    from experiments.evidence_strength_report import text_route_reduction_ci

    ci = text_route_reduction_ci()
    g = ci["docvqa/qwen/qwen3.6-27b"]
    check("Groq input-only n", g["n"], 101)
    # evidence_strength_report rounds the mean to 3 dp (11.235); use the exact mean (11.2351...).
    db = sqlite3.connect(RESULTS)
    by = defaultdict(dict)
    for ds, ip, m, mo, t in db.execute(
        "SELECT dataset, image_path, method, model, input_tokens FROM runs "
        "WHERE method IN ('unreduced','text_compressed') AND truncated=0"
    ):
        by[(ds, ip, mo)][m] = t
    exact = [
        v["unreduced"] / v["text_compressed"]
        for (ds, ip, mo), v in by.items()
        if ds == "docvqa" and mo == "qwen/qwen3.6-27b" and len(v) == 2 and v["text_compressed"] > 0
    ]
    check("Groq input-only mean", rhu(sum(exact) / len(exact), 2), 11.24)
    check("Groq input-only CI low", g["ci_low"], 9.08, 0.006)
    check("Groq input-only CI high", g["ci_high"], 13.64, 0.006)
    g = ci["docvqa/gemini-2.5-flash"]
    check("Gemini input-only n", g["n"], 31)
    check("Gemini input-only mean", g["mean"], 1.96, 0.005)
    check("Gemini input-only CI", (round(g["ci_low"], 2), round(g["ci_high"], 2)), (0.94, 3.38))

    import io
    from contextlib import redirect_stdout

    import revision_textroute_total as rtt

    rtt.RNG = np.random.default_rng(0)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rtt.main()
    out = buf.getvalue()
    import re

    tot = re.findall(r"total-token ratio mean ([\d.]+)\s+CI \[([\d.]+), ([\d.]+)\]", out)
    check("Groq total-token mean (CI)", f"{tot[0][0]} [{tot[0][1]}, {tot[0][2]}]", "2.72 [2.52, 2.93]")
    check("Gemini total-token mean (CI)", f"{tot[1][0]} [{tot[1][1]}, {tot[1][2]}]", "1.05 [0.89, 1.23]")
    means = re.findall(r"mean in (\d+)->(\d+)\s+out (\d+)->(\d+)", out)
    ui, ti, uo, to = map(int, means[0])
    check("Groq text-route output 289 -> 542", f"{uo} -> {to}", "289 -> 542")
    check("Groq pooled total ratio", round((ui + uo) / (ti + to), 2), 2.41)


# ---------------------------------------------------------------- Table IV
def gemini_thinking_by_text() -> dict[str, int]:
    out = {}
    for (r,) in sqlite3.connect(CACHE).execute("SELECT response_json FROM cache WHERE model='gemini-2.5-flash'"):
        rr = json.loads(r)
        um = (rr.get("raw") or {}).get("usage_metadata")
        if um is not None:
            out[rr.get("text") or ""] = um.get("thoughts_token_count") or 0
    return out


def table4() -> None:
    section("Table IV: batched queries, N=4 (mean per window)")
    last = {}
    for line in JSONL.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if "error" not in r:
            last[(r["provider"], r["dataset"], r["offset"], r["condition"])] = r
    agg = defaultdict(lambda: defaultdict(list))
    for (p, d, o, c), r in last.items():
        agg[(p, d)][c].append(r)

    # Gemini thinking tokens: logged per call in results.sqlite (rev_t3_* rows), recovered from the cache by response text.
    th = gemini_thinking_by_text()
    db = sqlite3.connect(RESULTS)
    gthink = defaultdict(list)
    for ds, txt in db.execute("SELECT dataset, response_text FROM runs WHERE dataset LIKE 'rev_t3_gemini_docvqa_o%'"):
        off, cond = ds.split("_o")[1].split("_", 1)
        gthink[(int(off), cond)].append(th.get(txt or ""))

    def mean(rows, f):
        return sum(f(r) for r in rows) / len(rows)

    paper = {
        ("groq", "docvqa"): {
            "sep": (6355, 438, "19/20"),
            "tiled": (1964, 46, "19/20"),
            "pertile": (7436, 329, "17/19"),
        },
        ("gemini", "docvqa"): {"sep": (311, 990, "15/16"), "tiled": (217, 621, "9/16"), "multi": (427, None, "10/12")},
        ("groq", "chartqa"): {"sep": (6761, 481, "8/8"), "tiled": (1961, 34, "6/8")},
    }
    for (prov, ds), conds in paper.items():
        a = agg[(prov, ds)]
        matched = sorted({r["offset"] for r in a["sep"]} & {r["offset"] for r in a["tiled"]})
        check(
            f"{prov}/{ds} matched windows",
            len(matched),
            {"groq/docvqa": 5, "gemini/docvqa": 4, "groq/chartqa": 2}[f"{prov}/{ds}"],
        )
        for cond, (pin, pout, pcor) in conds.items():
            rows = [r for r in a[cond] if r["offset"] in matched]
            check(f"{prov}/{ds} {cond} input", rhu(mean(rows, lambda r: r["input_tokens"])), pin)
            if pout is not None:
                out = mean(rows, lambda r: r["output_tokens"])
                if prov == "gemini":
                    out += mean(rows, lambda r: sum(gthink[(r["offset"], cond)]))
                check(f"{prov}/{ds} {cond} output (billed)", rhu(out), pout)
            check(
                f"{prov}/{ds} {cond} correct",
                f"{sum(r['correct'] for r in rows)}/{sum(r['correct'] + r['incorrect'] for r in rows)}",
                pcor,
            )
        s = {r["offset"]: r["input_tokens"] for r in a["sep"]}
        t = {r["offset"]: r["input_tokens"] for r in a["tiled"]}
        ratios = [s[o] / t[o] for o in matched]
        if (prov, ds) == ("groq", "docvqa"):
            check("Groq/DocVQA input reduction mean", round(sum(ratios) / len(ratios), 2), 3.24)
            check("Groq/DocVQA range", f"{min(ratios):.2f}-{max(ratios):.2f}", "2.67-3.72")
        if (prov, ds) == ("gemini", "docvqa"):
            check("Gemini input reduction mean", round(sum(ratios) / len(ratios), 2), 1.43)
            check("Gemini range", f"{min(ratios):.2f}-{max(ratios):.2f}", "1.43-1.45")
            g3 = {c: [r for r in a[c] if r["offset"] in (0, 4, 8)] for c in ("sep", "multi", "tiled")}
            check(
                "Gemini 3 matched windows sep/multi/tiled correct",
                "/".join(f"{sum(r['correct'] for r in v)}" for v in g3.values()),
                "11/10/8",
            )

    section("Fig. 4: USD per 1,000 requests (input + billed output)")
    import make_figures

    _, sep, tiled = make_figures.fig_cost_dollars()
    for lab, s, t, ps, pt, pr in [
        ("Groq/DocVQA", sep[0], tiled[0], 6.84, 1.76, 74),
        ("Groq/ChartQA", sep[1], tiled[1], 7.33, 1.70, 77),
        ("Gemini/DocVQA", sep[2], tiled[2], 2.57, 1.62, 37),
    ]:
        check(f"{lab} separate $", round(s, 2), ps)
        check(f"{lab} tiled $", round(t, 2), pt)
        check(f"{lab} reduction %", round(100 * (1 - t / s)), pr)


# ------------------------------------------------------------------ audit
def audit() -> None:
    section("Sec. VI-A: human audit of the judge")
    judge = load_judge_cache()
    db = sqlite3.connect(RESULTS)
    pairs = defaultdict(list)
    for a in csv.DictReader(open(AUDIT, encoding="utf-8")):
        prompt, resp = db.execute("SELECT prompt, response_text FROM runs WHERE id=?", (int(a["row_id"]),)).fetchone()
        j = judge[(prompt, _strip_think_block(resp) if resp else "")][0]
        h = j if a["human_correct_1_or_0"].strip() == "1" else (not j)
        ds = "docvqa" if a["dataset"].startswith("docvqa") else "chartqa"
        pairs[ds].append((j, h))
        pairs["all"].append((j, h))

    def kappa(v):
        n = len(v)
        po = sum(x == y for x, y in v) / n
        pj, ph = sum(x for x, _ in v) / n, sum(y for _, y in v) / n
        pe = pj * ph + (1 - pj) * (1 - ph)
        return sum(x == y for x, y in v), n, (po - pe) / (1 - pe)

    for ds, (agree, n, k) in [(d, kappa(pairs[d])) for d in ("all", "docvqa", "chartqa")]:
        exp = {"all": ("86/100", 0.63), "docvqa": ("40/48", 0.41), "chartqa": ("46/52", 0.74)}[ds]
        check(f"audit {ds} agreement", f"{agree}/{n}", exp[0])
        check(f"audit {ds} kappa", round(k, 2), exp[1])
    v = pairs["all"]
    check("judge accepted, human rejected", sum(1 for j, h in v if j and not h), 9)
    check("judge rejected, human accepted", sum(1 for j, h in v if not j and h), 5)


# ------------------------------------------------------- other text numbers
def misc() -> None:
    section("Other reported values")
    db = sqlite3.connect(RESULTS)
    by = defaultdict(dict)
    for ds, ip, m, mo, t in db.execute(
        "SELECT dataset, image_path, method, model, input_tokens FROM runs WHERE method IN ('unreduced','text_compressed') AND truncated=0"
    ):
        by[(ds, ip, mo)][m] = t
    n101 = sum(
        1
        for (ds, ip, mo), v in by.items()
        if ds == "docvqa" and mo == "qwen/qwen3.6-27b" and len(v) == 2 and v["text_compressed"] > 0
    )
    check("Groq/DocVQA cost-ratio pairs", n101, 101)

    toks = [
        t
        for (t,) in db.execute(
            "SELECT MIN(input_tokens) FROM runs WHERE method='unreduced' AND model='qwen/qwen3.6-27b' "
            "AND dataset IN ('docvqa','chartqa') AND truncated=0 GROUP BY image_path"
        )
    ]
    check("Groq natural images measured", len(toks), 172)
    check("Groq images at highest level (~1,830)", sum(1 for t in toks if t >= 1800), 154)

    r = db.execute(
        "SELECT MIN(input_tokens), MAX(input_tokens) FROM runs WHERE dataset='docvqa_tiled' AND model='gemini-2.5-flash'"
    ).fetchone()
    check("Gemini composite-call range N=2-12", f"{r[0]}-{r[1]}", "305-324")
    r = db.execute(
        "SELECT MIN(input_tokens), MAX(input_tokens) FROM runs WHERE dataset='docvqa_tiled' AND model='gemini-2.5-flash' AND method='tile_n4'"
    ).fetchone()
    check("Gemini per-tile N=4 range", f"{r[0]}-{r[1]}", "305-322")
    r = db.execute(
        "SELECT MIN(input_tokens), MAX(input_tokens) FROM runs WHERE dataset='docvqa_tiled' AND model='qwen/qwen3.6-27b' AND method='tile_n8'"
    ).fetchone()
    check("Groq composite-call range N=8", f"{r[0]}-{r[1]}", "1850-1867")

    # Gemini image tokens per resolution setting, from cached usage_metadata (single-image calls).
    img = defaultdict(set)
    for s_json, r in sqlite3.connect(CACHE).execute(
        "SELECT settings_json, response_json FROM cache WHERE model='gemini-2.5-flash'"
    ):
        mode = json.loads(s_json).get("media_resolution", "default")
        um = (json.loads(r).get("raw") or {}).get("usage_metadata") or {}
        for d in um.get("prompt_tokens_details") or []:
            if "IMAGE" in str(d.get("modality")):
                img[mode].add(d.get("token_count"))
    check("Gemini image tokens, default resolution (Sec. IV-B)", sorted(img["default"]), [258])
    check("Gemini image tokens, low resolution (Sec. IV-B)", sorted(img["MEDIA_RESOLUTION_LOW"]), [66])
    check("Gemini resolution-mode reduction", rhu(258 / 66, 1), 3.9)

    section("Fig. 2: Groq billed tokens vs. image area (results/groq_image_billing.csv)")
    with open(ROOT / "results/groq_image_billing.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_path = {Path(r["image"]).stem: int(r["billed_input_tokens"]) for r in rows}
    db_tok = {
        Path(ip.replace("\\", "/")).stem: t
        for ip, t in db.execute(
            "SELECT image_path, MIN(input_tokens) FROM runs WHERE method='unreduced' AND model='qwen/qwen3.6-27b' "
            "AND dataset IN ('docvqa','chartqa') AND truncated=0 GROUP BY image_path"
        )
    }
    check("Fig. 2 CSV tokens match the call log", by_path == db_tok, True)
    hw = [int(r["height_px"]) / int(r["width_px"]) for r in rows]
    tok = [int(r["billed_input_tokens"]) for r in rows]
    lvl = lambda t: 790 if t < 1000 else (1300 if t < 1600 else 1830)  # noqa: E731
    check(
        "Fig. 2 near-square (h/w 0.9-1.2) all at ~1,300",
        sorted({lvl(t) for t, h in zip(tok, hw) if 0.9 <= h <= 1.2}),
        [1300],
    )
    check("Fig. 2 tall (h/w >= 1.8) all at ~790", sorted({lvl(t) for t, h in zip(tok, hw) if h >= 1.8}), [790])
    top = [int(r["width_px"]) * int(r["height_px"]) / 1e6 for r in rows if lvl(int(r["billed_input_tokens"])) == 1830]
    check("Fig. 2 top-level area range (MP)", f"{min(top):.2f}-{max(top):.1f}", "0.13-8.1")

    # Sec. IV-A: Qwen3.8-27B bills the same three levels (its single-image calls in Table IV).
    q38 = set()
    for line in JSONL.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("provider") == "groq" and r.get("condition") == "sep":
            q38 |= {lvl(t) for t in r["per_call_input"]}
    check("Qwen3.8 single-image calls bill the same levels", sorted(q38), [790, 1300, 1830])
    s4 = [
        r
        for r in map(json.loads, JSONL.read_text(encoding="utf-8").splitlines())
        if r.get("provider") == "groq" and r.get("dataset") == "docvqa" and r.get("condition") == "sep"
    ]
    tall20 = sum(1 for r in s4 for t in r["per_call_input"] if lvl(t) == 790)
    check("Table IV Groq/DocVQA: tall pages among the 20", tall20, 4)

    section("Table VI: natural images, Gemini and OpenRouter")
    imgs = {"49153": (282, 3368, 3357), "24580": (265, 3351, 909), "57349": (267, 3353, 3291)}
    for img, (gem, or12, or30) in imgs.items():
        like = f"%{img}.png"
        g = db.execute(
            "SELECT input_tokens FROM runs WHERE dataset='docvqa_week2' AND method='unreduced' AND image_path LIKE ?",
            (like,),
        ).fetchone()
        a = db.execute(
            "SELECT input_tokens FROM runs WHERE model='nvidia/nemotron-nano-12b-v2-vl:free' AND image_path LIKE ?",
            (like,),
        ).fetchone()
        b = db.execute(
            "SELECT input_tokens FROM runs WHERE model='nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free' AND image_path LIKE ?",
            (like,),
        ).fetchone()
        check(f"{img} Gemini / OR-12B / OR-30B", f"{g[0]}/{a[0]}/{b[0]}", f"{gem}/{or12}/{or30}")
    check("OR-12B spread %", round(100 * (3368 - 3351) / 3351, 1), 0.5)


def main() -> None:
    import os

    os.chdir(ROOT)
    for step in (table2, table3_and_tests, text_route_ratios, table4, audit, misc):
        step()
    bad = 0
    for label, got, paper, ok in checks:
        if label.startswith("## "):
            print(f"\n{label[3:]}")
            continue
        bad += not ok
        print(f"  {'MATCH   ' if ok else 'MISMATCH'} {label:<48} got {got:<22} paper {paper}")
    n = sum(1 for c in checks if not c[0].startswith("## "))
    print(f"\n{n - bad}/{n} values match the paper; {bad} mismatch(es).")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
