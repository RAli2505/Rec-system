"""
Alternative paper figures replacing Fig. 3 (grouped bars) and Fig. 4 (h-bars).

Outputs to results/xes3g5m/figures/:
  fig_methods_heatmap.{png,pdf}     -- methods x metrics heatmap (replaces Fig. 3)
  fig_ablation_heatmap.{png,pdf}    -- components x metrics delta heatmap (replaces Fig. 4)
  fig_cd_diagram.{png,pdf}          -- Critical Difference diagram across metrics
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
import matplotlib.colors as mcolors
from scipy.stats import friedmanchisquare, rankdata

from utils.plot_style import setup_publication_style, save_figure, DOUBLE_COL

setup_publication_style()

OUT_DIR = "results/xes3g5m/figures"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)


# ─── Load data ──────────────────────────────────────────────────────────

main = pd.read_csv("results/xes3g5m/tables/table_main_results.csv")
abl  = pd.read_csv("results/xes3g5m/tables/table_ablation.csv")


def parse(v):
    """Parse '0.9321 +/- 0.0025' or '0.9321' -> 0.9321."""
    s = str(v).split("+/-")[0].strip()
    return float(s)


# Use whichever methods are present in the CSV (extra baselines may have
# been added since first paper draft). Keep them in a deliberate order.
ALL_METHODS = ["Random", "Popularity", "BPR-MF", "CF-only", "Content-only",
               "DKT (LSTM)", "GRU", "MARS (ours)"]
methods = [m for m in ALL_METHODS if m in main.columns]
# R@10 dropped: MARS evaluates on a filtered candidate pool, so its recall
# is not directly comparable with full-tag-pool baselines.
metrics = ["AUC-ROC", "NDCG@10", "Precision@10", "MRR", "Coverage"]

M = np.array([[parse(main.loc[main["Metric"] == m, meth].values[0])
               for meth in methods]
              for m in metrics])  # shape (n_metrics, n_methods)


# ─────────────────────────────────────────────────────────────────────
# 1.  METHODS x METRICS HEATMAP
# ─────────────────────────────────────────────────────────────────────

def fig_methods_heatmap():
    # Per-row min-max normalization for fair colour comparison.
    # Coverage > 1 (Random=1.058) is clipped to 1.0 since values above 1
    # reflect a degenerate "recommend everything" pattern, not real coverage.
    M_norm = M.copy()
    M_norm[metrics.index("Coverage")] = np.clip(
        M_norm[metrics.index("Coverage")], 0, 1.0
    )
    row_min = M_norm.min(axis=1, keepdims=True)
    row_max = M_norm.max(axis=1, keepdims=True)
    M_scaled = (M_norm - row_min) / (row_max - row_min + 1e-12)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL[0], 3.6))

    im = ax.imshow(M_scaled, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)

    # cell text: raw value
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M[i, j]
            txt_color = "white" if M_scaled[i, j] > 0.55 else "#222"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color=txt_color)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics)
    ax.tick_params(axis="both", which="both", length=0)
    ax.grid(False)

    # Highlight MARS column
    mars_idx = methods.index("MARS (ours)")
    ax.add_patch(plt.Rectangle((mars_idx - 0.5, -0.5), 1, len(metrics),
                                fill=False, edgecolor="#D55E00", linewidth=1.8))

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Per-metric normalised score", fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title("MARS vs. baselines on XES3G5M (per-row normalisation)",
                 pad=10)

    fig.tight_layout()
    save_figure(fig, "fig_methods_heatmap", results_dir=OUT_DIR)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# 2.  ABLATION HEATMAP
# ─────────────────────────────────────────────────────────────────────

def fig_ablation_heatmap():
    abl_metrics = ["AUC-ROC", "NDCG@10", "P@10", "MRR", "Coverage"]
    full = abl.iloc[0]
    rows = abl.iloc[1:].reset_index(drop=True)
    full_vals = np.array([full[m] for m in abl_metrics], dtype=float)

    deltas = np.zeros((len(rows), len(abl_metrics)))
    raw    = np.zeros((len(rows), len(abl_metrics)))
    for i, (_, r) in enumerate(rows.iterrows()):
        for j, m in enumerate(abl_metrics):
            raw[i, j] = float(r[m])
            deltas[i, j] = float(r[m]) - full_vals[j]

    config_labels = [str(r).replace("- ", "− ") for r in rows["Configuration"]]

    # Symmetric divergent scale based on max abs delta
    vmax = max(abs(deltas.min()), abs(deltas.max()))

    fig, ax = plt.subplots(figsize=(DOUBLE_COL[0], 3.0))
    im = ax.imshow(deltas, cmap="RdBu", aspect="auto",
                   vmin=-vmax, vmax=vmax)

    for i in range(deltas.shape[0]):
        for j in range(deltas.shape[1]):
            d = deltas[i, j]
            sign = "+" if d > 0 else ""
            txt_color = "white" if abs(d) > vmax * 0.55 else "#222"
            ax.text(j, i, f"{sign}{d:.3f}", ha="center", va="center",
                    fontsize=8.5, color=txt_color)

    ax.set_xticks(range(len(abl_metrics)))
    ax.set_xticklabels(abl_metrics)
    ax.set_yticks(range(len(config_labels)))
    ax.set_yticklabels(config_labels)
    ax.tick_params(axis="both", which="both", length=0)
    ax.grid(False)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Δ from Full MARS", fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    full_str = ", ".join(f"{m}={full_vals[j]:.3f}"
                         for j, m in enumerate(abl_metrics))
    ax.set_title(f"Ablation: change relative to Full MARS\n({full_str})",
                 pad=10, fontsize=10)

    fig.tight_layout()
    save_figure(fig, "fig_ablation_heatmap", results_dir=OUT_DIR)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# 3.  CRITICAL DIFFERENCE DIAGRAM (across metrics as tasks)
# ─────────────────────────────────────────────────────────────────────

def fig_cd_diagram():
    # Treat each of the 6 metrics as an independent "task".
    # Rank methods per metric (1 = best). All metrics are higher-is-better.
    # For Coverage, clip Random's pathological >1 value (still ranks high).
    M_for_rank = M.copy()
    cov_idx = metrics.index("Coverage")
    M_for_rank[cov_idx] = np.clip(M_for_rank[cov_idx], 0, 1.0)

    # ranks: rank within each row, higher score => lower rank number (1 best)
    ranks = np.array([rankdata(-row, method="average") for row in M_for_rank])
    avg_ranks = ranks.mean(axis=0)  # one rank per method

    n_methods = len(methods)
    n_tasks = len(metrics)

    # Friedman test for overall difference among methods
    chi2, p_friedman = friedmanchisquare(*[ranks[:, j] for j in range(n_methods)])

    # Nemenyi critical difference at alpha = 0.05
    # CD = q_alpha * sqrt(k(k+1) / (6N))
    # Studentized range q for alpha=0.05, infinite df, k methods
    Q_ALPHA_05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
                  6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    q = Q_ALPHA_05[n_methods]
    CD = q * np.sqrt(n_methods * (n_methods + 1) / (6.0 * n_tasks))

    # ── Plot (Demšar-style: best on LEFT, worst on RIGHT) ────────────
    fig, ax = plt.subplots(figsize=(DOUBLE_COL[0], 3.2))

    rmin = 1
    rmax = n_methods

    # Sort methods by avg rank (best first)
    order = np.argsort(avg_ranks)
    sorted_methods = [methods[i] for i in order]
    sorted_ranks = avg_ranks[order]

    # Coord system: x = rank (1 left, n_methods right), y in [0, 10]
    pad_x = 1.2
    ax.set_xlim(rmin - pad_x, rmax + pad_x)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # ── Title (top) ──
    ax.text((rmin + rmax) / 2, 9.7,
            f"Critical Difference Diagram (Friedman χ²={chi2:.2f}, "
            f"p={p_friedman:.1e})",
            ha="center", va="top", fontsize=10, fontweight="bold")
    ax.text((rmin + rmax) / 2, 9.1,
            f"methods ranked across {n_tasks} metrics on XES3G5M",
            ha="center", va="top", fontsize=8.5, color="#555",
            style="italic")

    # ── CD bracket (between title and axis) ──
    cd_y = 7.7
    cd_xL = rmin
    cd_xR = rmin + CD
    ax.plot([cd_xL, cd_xR], [cd_y, cd_y], "k-", linewidth=1.4)
    ax.plot([cd_xL, cd_xL], [cd_y - 0.18, cd_y + 0.18], "k-", linewidth=1.4)
    ax.plot([cd_xR, cd_xR], [cd_y - 0.18, cd_y + 0.18], "k-", linewidth=1.4)
    ax.text((cd_xL + cd_xR) / 2, cd_y + 0.35,
            f"CD = {CD:.2f}  (Nemenyi, α=0.05)",
            ha="center", va="bottom", fontsize=8.5)

    # ── Rank axis ──
    y_axis = 6.5
    ax.plot([rmin, rmax], [y_axis, y_axis], "k-", linewidth=1.2)
    for r in range(rmin, rmax + 1):
        ax.plot([r, r], [y_axis, y_axis + 0.18], "k-", linewidth=1.0)
        ax.text(r, y_axis + 0.45, f"{r}", ha="center", va="bottom", fontsize=9)
    ax.text(rmin, y_axis - 0.55, "best",  ha="center", va="top",
            fontsize=8, style="italic", color="#555")
    ax.text(rmax, y_axis - 0.55, "worst", ha="center", va="top",
            fontsize=8, style="italic", color="#555")

    # ── Method labels: split left/right halves by rank ──
    half = (n_methods + 1) // 2  # left side gets the better-ranked half
    y_slots = [4.7, 3.9, 3.1, 2.3, 1.5]  # generous spacing

    label_pad = 0.7

    def hl_color(name):
        return "#D55E00" if "MARS" in name else "#333"

    # Left side: best-ranked methods, labelled on the LEFT
    for i in range(half):
        m = sorted_methods[i]
        r = sorted_ranks[i]
        y = y_slots[i]
        c = hl_color(m)
        ax.plot([r, r], [y_axis, y], color=c, linewidth=1.0)
        ax.plot([r, rmin - label_pad], [y, y], color=c, linewidth=1.0)
        ax.text(rmin - label_pad - 0.06, y, f"{m}  ({r:.2f})",
                ha="right", va="center", fontsize=9.5, color=c,
                fontweight="bold" if "MARS" in m else "normal")

    # Right side: worst-ranked methods, labelled on the RIGHT
    for i in range(n_methods - half):
        idx = half + i
        m = sorted_methods[idx]
        r = sorted_ranks[idx]
        y = y_slots[i]
        c = hl_color(m)
        ax.plot([r, r], [y_axis, y], color=c, linewidth=1.0)
        ax.plot([r, rmax + label_pad], [y, y], color=c, linewidth=1.0)
        ax.text(rmax + label_pad + 0.06, y, f"({r:.2f})  {m}",
                ha="left", va="center", fontsize=9.5, color=c,
                fontweight="bold" if "MARS" in m else "normal")

    # ── Cliques (groups not significantly different) ──
    sorted_r = avg_ranks[order]
    cliques = []
    for i in range(n_methods):
        j = i
        while j + 1 < n_methods and (sorted_r[j + 1] - sorted_r[i]) < CD:
            j += 1
        if j > i:
            cliques.append((sorted_r[i], sorted_r[j]))
    # Drop sub-cliques (keep only maximal ones)
    dedup = []
    for c in cliques:
        if not any((c[0] >= d[0] and c[1] <= d[1] and c != d) for d in cliques):
            dedup.append(c)

    clique_y0 = y_axis - 0.3
    for k, (a, b) in enumerate(dedup):
        y = clique_y0 - k * 0.32
        ax.plot([a - 0.07, b + 0.07], [y, y], "k-", linewidth=3.5,
                solid_capstyle="round")

    fig.tight_layout()
    save_figure(fig, "fig_cd_diagram", results_dir=OUT_DIR)
    plt.close(fig)

    # Print rank summary
    print("\n=== CD Diagram ranks ===")
    for m, r in sorted(zip(methods, avg_ranks), key=lambda x: x[1]):
        print(f"  {m:<14s} avg rank = {r:.3f}")
    print(f"\nFriedman: chi2={chi2:.3f}, p={p_friedman:.4g}")
    print(f"Nemenyi CD (alpha=0.05, k={n_methods}, N={n_tasks}) = {CD:.3f}")


# ─── Run ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fig_methods_heatmap()
    fig_ablation_heatmap()
    fig_cd_diagram()
    print(f"\nAll figures saved to: {Path(OUT_DIR).resolve()}")
