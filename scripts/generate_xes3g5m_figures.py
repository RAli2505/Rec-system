"""
Generate all figures for the XES3G5M section of the paper.

Reads from results/xes3g5m/*/metrics.json and baselines.json.
Outputs to results/xes3g5m/figures/.

Usage:
    python scripts/generate_xes3g5m_figures.py
"""

import json
import glob
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIGURES_DIR = ROOT / "results" / "xes3g5m" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
MARS_COLOR = "#2196F3"
BASELINE_COLORS = {"random": "#9E9E9E", "popularity": "#FF9800",
                   "dkt_lstm": "#4CAF50", "gru": "#9C27B0"}


def load_mars_seeds():
    """Load all MARS full pipeline seed results."""
    patterns = [
        str(ROOT / "results/xes3g5m/xes3g5m_full_s*/metrics.json"),
        str(ROOT / "results/xes3g5m/xes3g5m_s*/metrics.json"),
    ]
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))
    seeds = {}
    for f in files:
        m = json.load(open(f))
        s = m.get("seed", 0)
        seeds[s] = m.get("eval_metrics", {})
        # merge top-level metrics (learning_gain, etc.)
        for k in ["learning_gain", "learning_gain_std", "learning_gain_trimmed"]:
            if k in m:
                seeds[s][k] = m[k]
        seeds[s]["val_auc"] = m.get("agent_metrics", {}).get("prediction", {}).get("val_auc", 0)
    return seeds


def load_baselines():
    """Load baseline results."""
    files = sorted(glob.glob(str(ROOT / "results/xes3g5m/baselines_s*/baselines.json")))
    if not files:
        return {}
    return json.load(open(files[-1]))


def load_history(seed=42):
    """Load training history for a seed."""
    files = sorted(glob.glob(str(ROOT / f"results/xes3g5m/xes3g5m_full_s{seed}_*/history.json")))
    if not files:
        return None
    return json.load(open(files[-1]))


def fig1_training_curves():
    """Training curves: val_auc and train_loss per epoch."""
    hist = load_history(42)
    if not hist:
        print("No history.json found for seed 42")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(hist["val_auc"]) + 1)
    ax1.plot(epochs, hist["val_auc"], "o-", color=MARS_COLOR, linewidth=2, label="val_auc")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("AUC")
    ax1.set_title("Validation AUC per Epoch")
    ax1.legend()

    ax2.plot(epochs, hist["train_loss"], "s-", color="#F44336", linewidth=2, label="train_loss")
    ax2.plot(epochs, hist["val_loss"], "^-", color="#FF9800", linewidth=2, label="val_loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training / Validation Loss")
    ax2.legend()

    fig.suptitle("MARS Training on XES3G5M (seed=42)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_training_curves.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_training_curves.pdf", bbox_inches="tight")
    plt.close()
    print("✓ fig_training_curves")


def fig2_mars_vs_baselines():
    """Bar chart: MARS vs all baselines on key metrics."""
    seeds = load_mars_seeds()
    baselines = load_baselines()

    if not seeds or not baselines:
        print("Missing data for MARS vs baselines")
        return

    metrics = ["test_auc_macro", "ndcg@10", "precision@10", "mrr"]
    labels = ["AUC", "NDCG@10", "Precision@10", "MRR"]

    # MARS mean±std
    mars_means = []
    mars_stds = []
    for m in metrics:
        key = m if m != "test_auc_macro" else "lstm_auc"
        vals = [seeds[s].get(key, seeds[s].get(m, 0)) for s in seeds]
        mars_means.append(np.mean(vals))
        mars_stds.append(np.std(vals))

    methods = list(baselines.keys())
    n_methods = len(methods) + 1  # +1 for MARS
    n_metrics = len(metrics)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_metrics)
    width = 0.15

    # Baselines
    for i, method in enumerate(methods):
        vals = []
        for m in metrics:
            vals.append(baselines[method].get(m, 0))
        ax.bar(x + i * width, vals, width, label=method.upper(),
               color=BASELINE_COLORS.get(method, "#999"), alpha=0.8)

    # MARS
    ax.bar(x + len(methods) * width, mars_means, width, yerr=mars_stds,
           label="MARS (ours)", color=MARS_COLOR, alpha=0.9, capsize=4,
           edgecolor="black", linewidth=1.5)

    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("MARS vs Baselines on XES3G5M", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_mars_vs_baselines.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_mars_vs_baselines.pdf", bbox_inches="tight")
    plt.close()
    print("✓ fig_mars_vs_baselines")


def fig3_ednet_vs_xes3g5m():
    """Side-by-side comparison: EdNet Phase A vs XES3G5M full pipeline."""
    ednet = {
        "AUC": 0.7054, "NDCG@10": 0.3978, "Precision@10": 0.3552,
        "MRR": 0.4396, "Coverage": 0.3369, "Learning Gain": -0.0123,
    }
    xes_seeds = load_mars_seeds()
    if not xes_seeds:
        print("No XES3G5M results")
        return

    xes = {
        "AUC": np.mean([xes_seeds[s].get("lstm_auc", 0) for s in xes_seeds]),
        "NDCG@10": np.mean([xes_seeds[s].get("ndcg@10", 0) for s in xes_seeds]),
        "Precision@10": np.mean([xes_seeds[s].get("precision@10", 0) for s in xes_seeds]),
        "MRR": np.mean([xes_seeds[s].get("mrr", 0) for s in xes_seeds]),
        "Coverage": np.mean([xes_seeds[s].get("tag_coverage", 0) for s in xes_seeds]),
        "Learning Gain": np.mean([xes_seeds[s].get("learning_gain", 0) for s in xes_seeds]),
    }

    metrics = list(ednet.keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width/2, [ednet[m] for m in metrics], width,
           label="EdNet (TOEIC)", color="#FF7043", alpha=0.8)
    ax.bar(x + width/2, [xes[m] for m in metrics], width,
           label="XES3G5M (Math)", color=MARS_COLOR, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11, rotation=15)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("MARS Performance: EdNet vs XES3G5M", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_ednet_vs_xes3g5m.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_ednet_vs_xes3g5m.pdf", bbox_inches="tight")
    plt.close()
    print("✓ fig_ednet_vs_xes3g5m")


def fig4_radar_chart():
    """Radar chart: multi-metric comparison MARS vs DKT vs GRU."""
    seeds = load_mars_seeds()
    baselines = load_baselines()
    if not seeds or not baselines:
        print("Missing data for radar")
        return

    categories = ["AUC", "NDCG@10", "P@10", "MRR", "Coverage"]
    mars_vals = [
        np.mean([seeds[s].get("lstm_auc", 0) for s in seeds]),
        np.mean([seeds[s].get("ndcg@10", 0) for s in seeds]),
        np.mean([seeds[s].get("precision@10", 0) for s in seeds]),
        np.mean([seeds[s].get("mrr", 0) for s in seeds]),
        np.mean([seeds[s].get("tag_coverage", 0) for s in seeds]),
    ]

    methods = {"MARS (ours)": mars_vals}
    for name in ["dkt_lstm", "gru"]:
        if name in baselines:
            b = baselines[name]
            methods[name.upper()] = [
                b.get("test_auc_macro", 0), b.get("ndcg@10", 0),
                b.get("precision@10", 0), b.get("mrr", 0),
                b.get("tag_coverage", 0),
            ]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = {"MARS (ours)": MARS_COLOR, "DKT_LSTM": "#4CAF50", "GRU": "#9C27B0"}

    for name, vals in methods.items():
        values = vals + vals[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=name,
                color=colors.get(name, "#999"))
        ax.fill(angles, values, alpha=0.15, color=colors.get(name, "#999"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_title("Multi-Metric Comparison on XES3G5M", fontsize=14,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_radar_chart.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_radar_chart.pdf", bbox_inches="tight")
    plt.close()
    print("✓ fig_radar_chart")


def fig5_seed_stability():
    """Box plot: metric distribution across 5 seeds."""
    seeds = load_mars_seeds()
    if len(seeds) < 3:
        print("Need >=3 seeds for stability plot")
        return

    metrics = {"AUC": "lstm_auc", "NDCG@10": "ndcg@10",
               "P@10": "precision@10", "MRR": "mrr", "Coverage": "tag_coverage"}

    data = []
    labels = []
    for label, key in metrics.items():
        vals = [seeds[s].get(key, 0) for s in seeds]
        data.append(vals)
        labels.append(label)

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    for patch in bp["boxes"]:
        patch.set_facecolor(MARS_COLOR)
        patch.set_alpha(0.6)

    # Overlay individual points
    for i, vals in enumerate(data):
        ax.scatter([i+1]*len(vals), vals, color="#F44336", s=50, zorder=5)

    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("MARS Metric Stability Across 5 Seeds (XES3G5M)", fontsize=14,
                 fontweight="bold")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_seed_stability.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_seed_stability.pdf", bbox_inches="tight")
    plt.close()
    print("✓ fig_seed_stability")


def table_summary():
    """Print summary table for paper (mean±std)."""
    seeds = load_mars_seeds()
    baselines = load_baselines()

    print("\n" + "="*70)
    print("TABLE FOR PAPER: XES3G5M Results (5 seeds)")
    print("="*70)

    metrics = [
        ("AUC (macro)", "lstm_auc", "test_auc_macro"),
        ("NDCG@10", "ndcg@10", "ndcg@10"),
        ("Precision@10", "precision@10", "precision@10"),
        ("MRR", "mrr", "mrr"),
        ("Coverage", "tag_coverage", "tag_coverage"),
        ("Learning Gain", "learning_gain", None),
    ]

    header = f"{'Metric':<18} {'Random':>8} {'Popular':>8} {'DKT':>8} {'GRU':>8} {'MARS (ours)':>16}"
    print(header)
    print("-" * len(header))

    for name, mars_key, bl_key in metrics:
        vals = [seeds[s].get(mars_key, 0) for s in seeds]
        mars_str = f"{np.mean(vals):.4f}±{np.std(vals):.4f}"

        bl_vals = []
        for bl_name in ["random", "popularity", "dkt_lstm", "gru"]:
            if bl_key and bl_name in baselines:
                bl_vals.append(f"{baselines[bl_name].get(bl_key, 0):.4f}")
            else:
                bl_vals.append("   -   ")

        print(f"{name:<18} {bl_vals[0]:>8} {bl_vals[1]:>8} {bl_vals[2]:>8} {bl_vals[3]:>8} {mars_str:>16}")


if __name__ == "__main__":
    print("Generating XES3G5M figures...")
    fig1_training_curves()
    fig2_mars_vs_baselines()
    fig3_ednet_vs_xes3g5m()
    fig4_radar_chart()
    fig5_seed_stability()
    table_summary()
    print(f"\nAll figures saved to {FIGURES_DIR}")
