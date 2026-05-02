"""
Bar-plot replacement for the methods-comparison heatmap (reviewer
G5). Heatmaps hide significance differences; grouped bar charts
with error bars are the publication standard.

Reads the same data sources as `generate_paper_alt_figures.py`'s
`fig_methods_heatmap` — `table_main_results.csv` for LSTM/GRU/MARS
(5-seed if present), and the attention_kt_baselines_*/baselines_s*
JSONs for SAINT/AKT/SimpleKT/DTransformer/SASRec.

Output: results/xes3g5m/figures/fig_methods_barplot.{png,pdf}
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.plot_style import setup_publication_style, save_figure, DOUBLE_COL

setup_publication_style()

OUT_DIR = "results/xes3g5m/figures"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)


# ─── Load data ─────────────────────────────────────────────────────

main = pd.read_csv("results/xes3g5m/tables/table_main_results.csv")


def parse_main(v: str) -> tuple[float, float]:
    """Parse "0.683 +/- 0.023" → (0.683, 0.023). Handles bare floats."""
    s = str(v)
    if "+/-" in s:
        a, b = s.split("+/-")
        return float(a.strip()), float(b.strip())
    return float(s), 0.0


def load_attn_kt() -> dict[str, dict[str, tuple[float, float]]]:
    runs: dict[str, list[dict]] = defaultdict(list)
    # Attention-KT baselines (SAINT/AKT/SimpleKT/DTransformer/SASRec).
    for jf in glob("results/xes3g5m/attention_kt_baselines_*/baselines_s*.json"):
        try:
            d = json.load(open(jf))
        except Exception:
            continue
        for name, m in d.items():
            if isinstance(m, dict) and "error" not in m:
                runs[name].append(m)
    # NCF baseline (separate output directory format: one model per
    # JSON, name carried in the dict).
    for jf in glob("results/xes3g5m/ncf_baseline_*/results_NCF_s*.json"):
        try:
            m = json.load(open(jf))
        except Exception:
            continue
        if isinstance(m, dict) and "error" not in m:
            runs[m.get("model", "NCF")].append(m)
    out: dict[str, dict[str, tuple[float, float]]] = {}
    keymap = {"AUC-ROC": "test_auc_macro",
                "NDCG@10": "ndcg@10",
                "Precision@10": "precision@10",
                "MRR": "mrr",
                "Coverage": "tag_coverage"}
    for name, rs in runs.items():
        out[name] = {}
        for label, k in keymap.items():
            vals = [r.get(k) for r in rs if r.get(k) is not None]
            if not vals:
                continue
            out[name][label] = (
                float(np.mean(vals)),
                float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            )
    return out


METHODS = ["DKT (LSTM)", "GRU", "NCF", "SAINT", "AKT", "SimpleKT",
            "DTransformer", "SASRec", "MARS (ours)"]
METRICS = ["AUC-ROC", "NDCG@10", "Precision@10", "MRR", "Coverage"]


attn = load_attn_kt()
data: dict[str, dict[str, tuple[float, float]]] = {}
for meth in METHODS:
    if meth in main.columns:
        data[meth] = {m: parse_main(main.loc[main["Metric"] == m, meth].values[0])
                       for m in METRICS}
    elif meth in attn:
        data[meth] = attn[meth]
    else:
        data[meth] = {}


# ─── Figure: 5-panel grouped bar plot ──────────────────────────────

fig, axes = plt.subplots(1, 5, figsize=(DOUBLE_COL[0] * 1.2, 3.4),
                          sharey=False)

# Soft palette: lstm/gru blue-ish, attention-kt teal/green, MARS orange
colors = {
    "DKT (LSTM)":   "#5B8FB9",
    "GRU":          "#7BA7C2",
    "NCF":          "#9CB1D5",
    "SAINT":        "#6FA694",
    "AKT":          "#5C9785",
    "SimpleKT":     "#4F8B79",
    "DTransformer": "#3F7B6D",
    "SASRec":       "#2D6E5E",
    "MARS (ours)":  "#D55E00",
}

for ax, metric in zip(axes, METRICS):
    means = []
    stds = []
    labels = []
    bar_colors = []
    for meth in METHODS:
        if metric in data.get(meth, {}):
            mean, std = data[meth][metric]
            means.append(mean)
            stds.append(std)
            labels.append(meth)
            bar_colors.append(colors[meth])

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=2.0, color=bar_colors,
                    edgecolor="#333", linewidth=0.5,
                    error_kw={"elinewidth": 0.7, "ecolor": "#333"})

    # Highlight MARS bar with thicker edge
    for i, lab in enumerate(labels):
        if lab == "MARS (ours)":
            bars[i].set_edgecolor("#D55E00")
            bars[i].set_linewidth(1.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_title(metric, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylim(0, max(means + [0]) * 1.18)
    ax.grid(axis="y", linestyle=":", alpha=0.35, zorder=0)

axes[0].set_ylabel("Score", fontsize=9)
fig.suptitle(
    "MARS vs. KT baselines on XES3G5M  "
    "(5-seed mean ± std where available; bars without error bars are 1-seed point estimates)",
    y=1.02, fontsize=10,
)
fig.tight_layout()
save_figure(fig, "fig_methods_barplot", results_dir=OUT_DIR)
plt.close(fig)
print("DONE")
