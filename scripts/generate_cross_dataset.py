"""
Cross-dataset comparison of MARS on EdNet (TOEIC) vs XES3G5M (Math).
Reads the 5-seed result JSONs from each dataset and produces a grouped
bar chart with mean +/- std error bars across 6 metrics.

Output: results/xes3g5m/figures/fig_ednet_vs_xes3g5m.{png,pdf}
"""

import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.plot_style import setup_publication_style, save_figure

setup_publication_style()

SEEDS = [42, 123, 456, 789, 2024]

# ─── Load 5-seed results from each dataset ───────────────────────────

def load_dataset(label: str, glob_pattern: str, today_only: bool = False):
    """Load eval_metrics from each seed; pick newest run per seed."""
    out = {}
    for s in SEEDS:
        candidates = sorted(
            glob.glob(glob_pattern.replace("{SEED}", str(s))),
            key=os.path.getmtime, reverse=True,
        )
        if today_only:
            candidates = [c for c in candidates
                           if any(d in c for d in
                                   ["20260423_06", "20260423_07", "20260423_08",
                                    "20260423_09", "20260423_1"])]
        if not candidates:
            print(f"  [{label}] seed {s}: no run found")
            continue
        with open(Path(candidates[0]) / "metrics.json") as f:
            m = json.load(f)
        out[s] = m
        print(f"  [{label}] seed {s}: {Path(candidates[0]).name}")
    return out


print("Loading EdNet results:")
ednet = load_dataset("EdNet",
                      "results/ednet_comparable/ednet_comparable_s{SEED}_*")
print("\nLoading XES3G5M results (today's NUM_TAGS=865 runs):")
xes3g5m = load_dataset("XES3G5M",
                        "results/xes3g5m/xes3g5m_full_s{SEED}_*",
                        today_only=True)

# ─── Compute mean +/- std per metric ─────────────────────────────────

# Map: display label -> json key (eval_metrics)
METRICS = [
    ("AUC",          "lstm_auc"),
    ("NDCG@10",      "ndcg@10"),
    ("Precision@10", "precision@10"),
    ("MRR",          "mrr"),
    ("Coverage",     "tag_coverage"),
    ("Learning Gain", "learning_gain"),
]

def stats(runs, key):
    vals = []
    for s, m in runs.items():
        # Some EdNet/XES3G5M runs nest eval_metrics, others flatten
        e = m.get("eval_metrics", m)
        v = e.get(key)
        if v is not None:
            vals.append(float(v))
    if not vals:
        return None, None, 0
    arr = np.array(vals)
    std = arr.std(ddof=1) if len(arr) > 1 else 0.0
    return arr.mean(), std, len(arr)


print("\n=== Mean ± Std per metric ===")
print(f"  {'Metric':<14s}  {'EdNet':>20s}  {'XES3G5M':>20s}")
ednet_mean, ednet_std = [], []
xes_mean,   xes_std   = [], []
for label, key in METRICS:
    em, es, en = stats(ednet, key)
    xm, xs, xn = stats(xes3g5m, key)
    ednet_mean.append(em or 0); ednet_std.append(es or 0)
    xes_mean.append(xm or 0);   xes_std.append(xs or 0)
    print(f"  {label:<14s}  {em:>10.4f} ± {es:.4f} (n={en})  "
          f"{xm:>10.4f} ± {xs:.4f} (n={xn})")

# ─── Plot grouped bars with error bars ───────────────────────────────

fig, ax = plt.subplots(figsize=(7.0, 3.6))

x = np.arange(len(METRICS))
w = 0.36

bars_e = ax.bar(x - w/2, ednet_mean, w, yerr=ednet_std,
                 color="#E76F51", edgecolor="black", linewidth=0.6,
                 capsize=3, label="EdNet (TOEIC)")
bars_x = ax.bar(x + w/2, xes_mean, w, yerr=xes_std,
                 color="#0173B2", edgecolor="black", linewidth=0.6,
                 capsize=3, label="XES3G5M (Math)")

# Annotate each bar
for bars, vals, errs in [(bars_e, ednet_mean, ednet_std),
                          (bars_x, xes_mean,   xes_std)]:
    for bar, v, e in zip(bars, vals, errs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + e + 0.012,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([label for label, _ in METRICS], rotation=15, ha="right")
ax.set_ylabel("Score")
ax.set_title("MARS Cross-Dataset Comparison: EdNet (TOEIC) vs XES3G5M (Math)\n"
             "5 seeds each, mean ± std",
             fontsize=10)
ax.set_ylim(min(0, min(ednet_mean + xes_mean) - 0.05),
             max(ednet_mean + xes_mean) * 1.15)
ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
ax.grid(axis="y", linewidth=0.4, alpha=0.5)
ax.legend(loc="upper right", frameon=True, fontsize=9)

fig.tight_layout()
save_figure(fig, "fig_ednet_vs_xes3g5m",
             results_dir="results/xes3g5m/figures")
plt.close(fig)
print("\nSaved fig_ednet_vs_xes3g5m.{png,pdf}")
