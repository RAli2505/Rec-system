"""
Generate the confidence-class distribution figure for the paper.

Source : results/seed_42/agent_metrics.json -> confidence.class_distribution
Output : results/figures/fig_confidence_class_distribution.{png,pdf}
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.plot_style import (
    setup_publication_style, save_figure,
    CONFIDENCE_COLORS, DOUBLE_COL,
)

setup_publication_style()

CLASS_ORDER = [
    "SOLID",
    "UNSURE_CORRECT",
    "FALSE_CONFIDENCE",
    "CLEAR_GAP",
    "DOUBT_CORRECT",
    "DOUBT_INCORRECT",
]

with open("results/seed_42/agent_metrics.json", encoding="utf-8") as f:
    metrics = json.load(f)

dist = metrics["confidence"]["class_distribution"]
counts = [dist[c] for c in CLASS_ORDER]
total = sum(counts)
pct = [c / total * 100 for c in counts]

fig, ax = plt.subplots(figsize=DOUBLE_COL)

bars = ax.bar(
    CLASS_ORDER, counts,
    color=CONFIDENCE_COLORS,
    edgecolor="black", linewidth=0.6,
)

ymax = max(counts)
for bar, c, p in zip(bars, counts, pct):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + ymax * 0.012,
        f"{c:,}\n({p:.1f}%)",
        ha="center", va="bottom", fontsize=8.5,
    )

ax.set_ylabel("Number of interactions")
ax.set_xlabel("Confidence class")
ax.set_title(f"Distribution of 6-class behavioural confidence  (N = {total:,})")
ax.set_ylim(0, ymax * 1.18)
ax.tick_params(axis="x", rotation=20)
ax.grid(axis="x", visible=False)

fig.tight_layout()
save_figure(fig, "fig_confidence_class_distribution", results_dir="results/figures")

print("\nClass distribution:")
for name, c, p in zip(CLASS_ORDER, counts, pct):
    print(f"  {name:<18s} {c:>10,}  ({p:5.2f}%)")
print(f"  {'TOTAL':<18s} {total:>10,}")
