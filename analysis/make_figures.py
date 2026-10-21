"""Figures for the INDICON 2026 revision. Sized at the IEEE column width
(3.5 in) so that 9-10 pt text stays legible in print. Every plotted value
is either a measured count (results/revision_table3.jsonl,
results/groq_image_billing.csv, analysis/revision_recount.py output) or a
published rate.

Usage (from the repository root): python analysis/make_figures.py
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
    }
)
W = 3.5
OUT = Path("figures")
OUT.mkdir(exist_ok=True)
JSONL = Path("results/revision_table3.jsonl")

# Published rates, USD per 1M tokens (input, output), accessed 2026-10-16.
RATES = {"gemini": (0.30, 2.50), "groq": (0.80, 4.00)}

# Mean thoughts_token_count per window over the 11 matched windows (0-12 from
# experiments/revision_gemini_thinking.py, 16-40 logged per row in
# results/revision_table3.jsonl). Billed as output by Gemini.
GEMINI_THINKING_MEAN = {"sep": 868.0, "tiled": 567.8}


def fig_groq_billing():
    """Groq (Qwen3.6-27B) billed input tokens for the unreduced image vs. pixel area,
    172 natural DocVQA/ChartQA images (results/groq_image_billing.csv, which records
    each logged image's size and its usage.prompt_tokens)."""
    import csv

    with open("results/groq_image_billing.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    area = np.array([int(r["width_px"]) * int(r["height_px"]) / 1e6 for r in rows])
    tok = np.array([int(r["billed_input_tokens"]) for r in rows])
    hw = np.array([int(r["height_px"]) / int(r["width_px"]) for r in rows])
    fig, ax = plt.subplots(figsize=(W, 1.7))
    tall = hw >= 1.8
    square = (hw >= 0.9) & (hw <= 1.2)
    other = ~(tall | square)
    for m, lab, col, mk in [
        (other, f"other ({other.sum()})", "#6b7280", "o"),
        (square, f"near-square ({square.sum()})", "#1d4ed8", "s"),
        (tall, f"tall ({tall.sum()})", "#b91c1c", "v"),
    ]:
        ax.scatter(
            area[m],
            tok[m],
            s=14 if mk == "o" else 26,
            c=col,
            marker=mk,
            alpha=0.7 if mk == "o" else 1.0,
            label=lab,
            linewidths=0,
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Image area (megapixels, log scale)")
    ax.set_ylabel("Billed input tokens")
    ax.set_ylim(600, 2000)
    ax.grid(True, alpha=0.25)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
        fontsize=8,
        handletextpad=0.1,
        columnspacing=1.2,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_groq_billing.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def wilson(x, n, z=1.96):
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return c - h, c + h


def fig_falsification():
    side = np.logspace(1, math.log10(8000), 400)
    tiles = np.where(side <= 384, 1, np.ceil(side / 768) ** 2)
    documented = 258 * tiles
    fig, ax = plt.subplots(figsize=(W, 1.7))
    ax.plot(side, documented, "--", color="#b91c1c", lw=1.6, label="Documented tiling rule")
    # Square sizes logged in results/gemini_size_sweep.jsonl (usage_metadata image tokens).
    sq = sorted({(r["width"], r["image_tokens"]) for r in map(json.loads, open("results/gemini_size_sweep.jsonl", encoding="utf-8"))
                 if r["width"] == r["height"]})
    pts = np.array([w for w, _ in sq], dtype=float)
    ax.plot(pts, np.array([t for _, t in sq], dtype=float), "o-", color="#047857", lw=1.8, ms=4, label="Measured usage field")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Image side length (px), square image")
    ax.set_ylabel("Image tokens")
    ax.set_ylim(150, 100000)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, which="major", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_falsification.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def fig_fidelity():
    groups = ["Groq\nDocVQA", "Gemini\nDocVQA", "Groq\nChartQA"]
    lost = [14, 11, 5]
    n = [99, 31, 13]
    pct = [100 * a / b for a, b in zip(lost, n)]
    ci = [wilson(a, b) for a, b in zip(lost, n)]
    lo = [p - 100 * c[0] for p, c in zip(pct, ci)]
    hi = [100 * c[1] - p for p, c in zip(pct, ci)]
    fig, ax = plt.subplots(figsize=(W, 1.7))
    bars = ax.bar(
        groups, pct, yerr=[lo, hi], capsize=4, width=0.55, color=["#b91c1c", "#1d4ed8", "#f87171"], ecolor="#111827"
    )
    for b, a, m, c in zip(bars, lost, n, ci):
        ax.text(b.get_x() + b.get_width() / 2, 100 * c[1] + 2, f"{a}/{m}", ha="center", fontsize=9)
    ax.set_ylabel("Answers lost (%)")
    ax.set_ylim(0, 80)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_fidelity.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def table3_rows():
    # Keep the last row per (provider, dataset, offset, condition): retries
    # after a quota stop re-record cached conditions.
    last = {}
    for line in JSONL.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if "error" in r:
            continue
        last[(r["provider"], r["dataset"], r["offset"], r["condition"])] = r
    agg = defaultdict(lambda: defaultdict(list))
    for (prov, ds, _, cond), r in last.items():
        agg[(prov, ds)][cond].append(r)
    return agg


def fig_cost_dollars():
    agg = table3_rows()
    cells = [
        ("groq", "docvqa", "Groq\nDocVQA"),
        ("groq", "chartqa", "Groq\nChartQA"),
        ("gemini", "docvqa", "Gemini\nDocVQA"),
    ]
    labels, sep, tiled = [], [], []
    for prov, ds, lab in cells:
        a = agg.get((prov, ds))
        if not a or not a.get("sep") or not a.get("tiled"):
            continue
        # Only windows where both conditions were measured.
        offs = {r["offset"] for r in a["sep"]} & {r["offset"] for r in a["tiled"]}
        rin, rout = RATES[prov]

        def usd(rows, cond):
            rows = [r for r in rows if r["offset"] in offs]
            i = np.mean([r["input_tokens"] for r in rows])
            o = np.mean([r["output_tokens"] for r in rows])
            if prov == "gemini":
                # output_tokens is answer text only; Gemini also bills thinking tokens.
                o += GEMINI_THINKING_MEAN[cond]
            return 1000 * (i * rin + o * rout) / 1e6

        labels.append(lab)
        sep.append(usd(a["sep"], "sep"))
        tiled.append(usd(a["tiled"], "tiled"))
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(W, 1.7))
    b1 = ax.bar(x - w / 2, sep, w, label="4 separate calls", color="#b91c1c")
    b2 = ax.bar(x + w / 2, tiled, w, label="1 tiled batched call", color="#15803d")
    ax.set_yscale("log")
    for bars, vals in ((b1, sep), (b2, tiled)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.2f}", ha="center", fontsize=8.5)
    for xi, s, t in zip(x, sep, tiled):
        ax.text(xi, max(s, t) * 2.3, f"$-${100 * (1 - t / s):.0f}%", ha="center", fontsize=9, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("USD / 1k req.")
    ax.set_ylim(min(tiled) * 0.4, max(sep) * 6)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cost_dollars.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    return labels, sep, tiled


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_falsification()
    fig_fidelity()
    fig_groq_billing()
    print(fig_cost_dollars())
