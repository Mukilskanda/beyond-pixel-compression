"""The headline artifact (plan §3.3): cost-fidelity curves, x-axis tokens,
y-axis fidelity, one curve per method, one panel per dataset category.
Not a single operating point — the Pareto frontier, with methods that cross
left visible, not hidden.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_runs(db_path: str | Path = "results/results.sqlite") -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM runs", conn)
    conn.close()
    return df


def bootstrap_ci(values: pd.Series, n_resamples: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    """Per plan §3.4: bootstrap CIs on all reported means, 1000 resamples."""
    import numpy as np

    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(0)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_resamples)]
    lower = (1 - ci) / 2 * 100
    upper = (1 - (1 - ci) / 2) * 100
    return (float(np.percentile(means, lower)), float(np.percentile(means, upper)))


def cost_fidelity_curve(
    df: pd.DataFrame,
    fidelity_col: str,
    dataset: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """One panel: x = mean input tokens per method, y = mean fidelity, with CI bars.
    Not a single point — this plots the whole Pareto frontier across methods.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    subset = df[df["dataset"] == dataset] if dataset else df

    summary = []
    for method, group in subset.groupby("method"):
        tokens = group["input_tokens"].dropna()
        fidelity = group[fidelity_col].dropna()
        if len(tokens) == 0 or len(fidelity) == 0:
            continue
        fid_lo, fid_hi = bootstrap_ci(fidelity)
        summary.append({
            "method": method,
            "mean_tokens": tokens.mean(),
            "mean_fidelity": fidelity.mean(),
            "fid_ci_lo": fid_lo,
            "fid_ci_hi": fid_hi,
            "n": len(group),
        })

    summary_df = pd.DataFrame(summary).sort_values("mean_tokens")

    for _, row in summary_df.iterrows():
        ax.errorbar(
            row["mean_tokens"], row["mean_fidelity"],
            yerr=[[row["mean_fidelity"] - row["fid_ci_lo"]], [row["fid_ci_hi"] - row["mean_fidelity"]]],
            marker="o", capsize=4, label=f"{row['method']} (n={int(row['n'])})",
        )

    ax.set_xlabel("Mean input tokens")
    ax.set_ylabel(f"Mean {fidelity_col}")
    ax.set_title(dataset or "All datasets")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def generate_headline_plot(
    db_path: str | Path = "results/results.sqlite",
    fidelity_col: str = "anls_score",
    output_path: str | Path = "results/cost_fidelity.png",
) -> None:
    """One panel per dataset category, per plan §3.3."""
    df = load_runs(db_path)
    if df.empty:
        raise ValueError(f"No runs found in {db_path} — run experiments first")

    datasets = sorted(df["dataset"].unique())
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)

    for ax, dataset in zip(axes[0], datasets):
        cost_fidelity_curve(df, fidelity_col=fidelity_col, dataset=dataset, ax=ax)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    generate_headline_plot()
