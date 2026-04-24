"""
Regenerate Fig. 8 — multi-metric radar comparison of MARS / DKT-LSTM / GRU.

Replaces the old radar that included BPR-MF (which is no longer in Table 3).
Uses the same 5 metrics as the updated Table 3 (no R@10):
    AUC-ROC, NDCG@10, Precision@10, MRR, Coverage.

Source : results/xes3g5m/tables/table_main_results.csv
Output : results/xes3g5m/figures/fig_radar_comparison.{png,pdf}
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.plot_style import setup_publication_style, save_figure, METHOD_COLORS

setup_publication_style()


def parse(v):
    s = str(v).split("+/-")[0].strip()
    return float(s)


df = pd.read_csv("results/xes3g5m/tables/table_main_results.csv").set_index("Metric")

# Metrics in the order shown around the radar (clockwise from top).
METRICS = ["AUC-ROC", "NDCG@10", "Precision@10", "MRR", "Coverage"]
METHODS = ["DKT (LSTM)", "GRU", "MARS (ours)"]

# Per-metric min-max normalisation so all axes share [0, 1].
# Coverage is clipped to [0, 1] (Random's 1.058 is pathological).
data = {}
for m in METRICS:
    vals = {meth: parse(df.loc[m, meth]) for meth in METHODS}
    if m == "Coverage":
        vals = {k: min(v, 1.0) for k, v in vals.items()}
    lo = min(vals.values())
    hi = max(vals.values())
    span = hi - lo if hi > lo else 1.0
    data[m] = {
        "raw":    vals,
        "scaled": {k: (v - lo) / span for k, v in vals.items()},
    }

# Angles for the radar
n = len(METRICS)
angles = [i / n * 2 * np.pi for i in range(n)]
angles_closed = angles + [angles[0]]

fig, ax = plt.subplots(figsize=(4.6, 4.6),
                        subplot_kw=dict(projection="polar"))

color_map = {
    "MARS (ours)": "#0173B2",
    "DKT (LSTM)":  "#DE8F05",
    "GRU":         "#029E73",
}
ls_map = {
    "MARS (ours)": "-",
    "DKT (LSTM)":  "--",
    "GRU":         ":",
}

for meth in METHODS:
    vals = [data[m]["scaled"][meth] for m in METRICS]
    vals_closed = vals + [vals[0]]
    ax.plot(angles_closed, vals_closed, label=meth,
            color=color_map[meth], linestyle=ls_map[meth],
            linewidth=2.0 if "MARS" in meth else 1.5,
            marker="o" if "MARS" in meth else "s",
            markersize=5)
    ax.fill(angles_closed, vals_closed,
            color=color_map[meth],
            alpha=0.18 if "MARS" in meth else 0.08)

ax.set_xticks(angles)
ax.set_xticklabels(METRICS, fontsize=9)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, color="#666")
ax.set_ylim(0, 1.05)
ax.set_rlabel_position(180 / n)   # axis labels off the metric spokes

ax.grid(True, linewidth=0.6, alpha=0.5)
ax.spines["polar"].set_color("#888")

ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18),
          ncol=3, frameon=False, fontsize=9)

ax.set_title("Multi-metric comparison on XES3G5M  (per-axis normalised)",
             pad=18, fontsize=10)

fig.tight_layout()
save_figure(fig, "fig_radar_comparison",
            results_dir="results/xes3g5m/figures")
plt.close(fig)

# Print raw values for caption / sanity check
print("\nRaw values used (per-axis min/max normalised for the plot):")
print(f"  {'Metric':<13s}  " + "  ".join(f"{m:>13s}" for m in METHODS))
for m in METRICS:
    row = data[m]["raw"]
    print(f"  {m:<13s}  " + "  ".join(f"{row[meth]:>13.4f}"
                                        for meth in METHODS))
