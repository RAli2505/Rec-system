"""
Generate publication-quality result figures from evaluation data.

Figure: MARS vs Baselines — 1×5 subplot bar chart.
Data source: results/tables/table1_comparison.csv, table3_significance.csv
Output: results/fig_mars_vs_baselines.{png,pdf}
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.plot_style import (
    setup_publication_style, save_figure,
    MARS_COLORS, DOUBLE_COL,
)

setup_publication_style()

# ── Load data ────────────────────────────────────────────────────────

df = pd.read_csv("results/tables/table1_comparison.csv")
sig = pd.read_csv("results/tables/table3_significance.csv")


def parse_mean_std(s):
    """Parse '0.1234 ± 0.0056' → (0.1234, 0.0056)."""
    m = re.match(r"([\d.]+)\s*[±+-]+\s*([\d.]+)", str(s))
    if m:
        return float(m.group(1)), float(m.group(2))
    return float(s), 0.0


# Metrics to plot (column name, display label)
METRICS = [
    ("NDCG@5",  "NDCG@5"),
    ("NDCG@10", "NDCG@10"),
    ("MAP@10",  "MAP@10"),
    ("MRR",     "MRR"),
    ("Coverage", "Coverage"),
]

# Parse mean/std for each method × metric
methods = df["Method"].tolist()
data = {}  # method → {metric: (mean, std)}
for _, row in df.iterrows():
    m = row["Method"]
    data[m] = {}
    for col, _ in METRICS:
        data[m][col] = parse_mean_std(row[col])

# Parse p-values: significance of MARS vs each baseline
pvals = {}  # baseline → {metric: p_value}
for _, row in sig.iterrows():
    bl = row["Baseline"]
    pvals[bl] = {}
    for col, _ in METRICS:
        p_col = f"{col}_t_p"
        if p_col in sig.columns:
            pvals[bl][col] = float(row[p_col])

# ── Significance star text ────────────────────────────────────────────

def _star(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ── Consistent method order (by average rank across metrics) ─────────

SHORT_NAMES = {
    "MARS (ours)": "MARS", "Content-only": "Content",
    "Binary-conf": "Bin-conf", "Monolithic": "Monolith",
}

avg_rank = {}
for m in methods:
    ranks = []
    for col, _ in METRICS:
        ranked = sorted(methods, key=lambda x: -data[x][col][0])
        ranks.append(ranked.index(m))
    avg_rank[m] = np.mean(ranks)
method_order = sorted(methods, key=lambda m: avg_rank[m])

# ── Figure: 1×5 subplots ────────────────────────────────────────────

fig, axes = plt.subplots(1, 5, figsize=(DOUBLE_COL[0], 3.2),
                         gridspec_kw={"wspace": 0.08})

MARS_BLUE = MARS_COLORS["mars_main"]
GREY = "#B0B0B0"
DARK_GREY = "#707070"

for panel_idx, (ax, (col, label)) in enumerate(zip(axes, METRICS)):
    means = [data[m][col][0] for m in method_order]
    stds = [data[m][col][1] for m in method_order]
    colors = [MARS_BLUE if "MARS" in m else GREY for m in method_order]

    y_pos = np.arange(len(method_order))
    ax.barh(y_pos, means, xerr=stds, height=0.7,
            color=colors, edgecolor="white", linewidth=0.3,
            capsize=1.5, error_kw={"linewidth": 0.7})

    ax.set_xlabel(label, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(left=0)

    # Y-axis labels only on first subplot (order is the same everywhere)
    short = [SHORT_NAMES.get(m, m) for m in method_order]
    ax.set_yticks(y_pos)
    if panel_idx == 0:
        ax.set_yticklabels(short, fontsize=7.5)
        for tl in ax.get_yticklabels():
            if "MARS" in tl.get_text():
                tl.set_fontweight("bold")
                tl.set_color(MARS_BLUE)
    else:
        ax.set_yticklabels([])

    # Value annotation: MARS bar + best baseline bar
    mars_idx = next(i for i, m in enumerate(method_order) if "MARS" in m)
    non_mars = [(i, means[i]) for i in range(len(method_order))
                if "MARS" not in method_order[i]]
    best_bl_idx, best_bl_val = max(non_mars, key=lambda x: x[1])
    best_bl_name = method_order[best_bl_idx]

    mars_mean, mars_std = means[mars_idx], stds[mars_idx]
    x_max = max(m + s for m, s in zip(means, stds))
    pad = x_max * 0.03

    # Significance star
    star = ""
    if best_bl_name in pvals and col in pvals[best_bl_name]:
        star = " " + _star(pvals[best_bl_name][col])

    ax.text(mars_mean + mars_std + pad, mars_idx,
            f"{mars_mean:.3f}{star}", va="center", ha="left",
            fontsize=6, fontweight="bold", color=MARS_BLUE)
    ax.text(best_bl_val + stds[best_bl_idx] + pad, best_bl_idx,
            f"{best_bl_val:.3f}", va="center", ha="left",
            fontsize=5.5, color=DARK_GREY)

    ax.set_xlim(right=x_max * 1.25)
    ax.grid(axis="x", alpha=0.2)
    ax.grid(axis="y", visible=False)

fig.suptitle("MARS vs Baselines", fontsize=11, fontweight="bold", y=1.01)

save_figure(fig, "fig_mars_vs_baselines", results_dir="results")
plt.close()
print("Done.")
