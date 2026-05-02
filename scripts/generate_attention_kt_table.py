"""
Generate the attention-KT comparison table and heatmap for MARS Table 4b.

Sources
-------
results/xes3g5m/attention_kt_baselines_*/baselines_s*.json
    A-style: SAINT, AKT, SimpleKT, DTransformer (14-dim, matched input,
    same eval as MARS — direct NDCG@10 / MRR / P@10 comparison).
results/xes3g5m/tables/table_seed_stability.csv
    MARS 5-seed reference numbers.

Outputs
-------
results/xes3g5m/tables/table_attention_kt.{csv,md,tex}
results/xes3g5m/figures/fig_attention_kt_heatmap.{png,pdf}
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.plot_style import setup_publication_style, save_figure

setup_publication_style()

OUT_TABLES = ROOT / "results/xes3g5m/tables"
OUT_FIGS = ROOT / "results/xes3g5m/figures"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)


# ─── Load attention-KT baselines (A-style) ─────────────────────────────

def aggregate_attention_kt() -> pd.DataFrame:
    rows = defaultdict(list)
    for jf in glob(str(ROOT / "results/xes3g5m/attention_kt_baselines_*/baselines_s*.json")):
        d = json.load(open(jf))
        for name, m in d.items():
            if isinstance(m, dict) and "error" not in m:
                rows[name].append(m)
    # NCF baseline (one model per JSON in a separate directory).
    for jf in glob(str(ROOT / "results/xes3g5m/ncf_baseline_*/results_NCF_s*.json")):
        try:
            m = json.load(open(jf))
        except Exception:
            continue
        if isinstance(m, dict) and "error" not in m:
            rows[m.get("model", "NCF")].append(m)

    summary = []
    metric_keys = {
        "AUC": "test_auc_macro",
        "NDCG@10": "ndcg@10",
        "MRR": "mrr",
        "P@10": "precision@10",
        "Tag Cov": "tag_coverage",
    }
    for model, runs in rows.items():
        row = {"Model": model, "n_seeds": len(runs),
               "params_M": runs[0].get("n_params", 0) / 1e6}
        for label, key in metric_keys.items():
            vals = [r.get(key) for r in runs if r.get(key) is not None]
            if vals:
                row[f"{label} mean"] = float(np.mean(vals))
                row[f"{label} std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary.append(row)
    return pd.DataFrame(summary)


def load_mars_reference() -> dict:
    """5-seed MARS numbers from the existing seed-stability table."""
    path = ROOT / "results/xes3g5m/tables/table_seed_stability.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        key = row["Metric"].strip()
        out[key] = (float(row["Mean"]), float(row["Std"]))
    return {
        "AUC mean":     out.get("lstm_auc",       (None, None))[0],
        "AUC std":      out.get("lstm_auc",       (None, None))[1],
        "NDCG@10 mean": out.get("ndcg@10",        (None, None))[0],
        "NDCG@10 std":  out.get("ndcg@10",        (None, None))[1],
        "MRR mean":     out.get("mrr",            (None, None))[0],
        "MRR std":      out.get("mrr",            (None, None))[1],
        "P@10 mean":    out.get("precision@10",   (None, None))[0],
        "P@10 std":     out.get("precision@10",   (None, None))[1],
        "Tag Cov mean": out.get("tag_coverage",   (None, None))[0],
        "Tag Cov std":  out.get("tag_coverage",   (None, None))[1],
    }


# ─── Build comparison table ─────────────────────────────────────────────

attn = aggregate_attention_kt()
attn = attn.sort_values("Model").reset_index(drop=True)

mars_ref = load_mars_reference()
mars_row = {
    "Model": "MARS (Full)",
    "n_seeds": 5,
    "params_M": 2.84,  # SAINT-inspired Transformer 4L/256d/8h
}
mars_row.update(mars_ref)
table = pd.concat([attn, pd.DataFrame([mars_row])], ignore_index=True)
# Keep MARS at the bottom for visual emphasis
table_order = [m for m in ["NCF", "SAINT", "AKT", "SimpleKT", "DTransformer", "SASRec"] if m in table["Model"].values] + ["MARS (Full)"]
table = table.set_index("Model").reindex(table_order).reset_index()

# ─── Save CSV ───────────────────────────────────────────────────────────

csv_cols = ["Model", "n_seeds", "params_M",
             "AUC mean", "AUC std",
             "NDCG@10 mean", "NDCG@10 std",
             "MRR mean", "MRR std",
             "P@10 mean", "P@10 std",
             "Tag Cov mean", "Tag Cov std"]
csv_cols = [c for c in csv_cols if c in table.columns]
table[csv_cols].to_csv(OUT_TABLES / "table_attention_kt.csv", index=False)
print(f"wrote {OUT_TABLES/'table_attention_kt.csv'}")


# ─── Markdown table ─────────────────────────────────────────────────────

def fmt(mean: float | None, std: float | None) -> str:
    if mean is None or pd.isna(mean):
        return "—"
    if std is None or pd.isna(std):
        return f"{mean:.3f}"
    return f"{mean:.3f}±{std:.3f}"


md_lines = [
    "# Attention-based KT comparison (5-seed mean ± std)",
    "",
    "All baselines (SAINT, AKT, SimpleKT, DTransformer) trained from "
    "scratch on XES3G5M with the same 14-dim per-step input as the "
    "MARS Prediction Agent, so the comparison isolates the attention "
    "mechanism and decoder choice from feature engineering. Metrics "
    "computed against the same multi-label failure-prediction task as "
    "MARS (Table 4 in the manuscript).",
    "",
    "| Model | Params (M) | AUC | NDCG@10 | MRR | P@10 | Tag Coverage |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for _, row in table.iterrows():
    md_lines.append(
        f"| {row['Model']} "
        f"| {row.get('params_M', 0):.2f} "
        f"| {fmt(row.get('AUC mean'),     row.get('AUC std'))} "
        f"| {fmt(row.get('NDCG@10 mean'), row.get('NDCG@10 std'))} "
        f"| {fmt(row.get('MRR mean'),     row.get('MRR std'))} "
        f"| {fmt(row.get('P@10 mean'),    row.get('P@10 std'))} "
        f"| {fmt(row.get('Tag Cov mean'), row.get('Tag Cov std'))} |"
    )
md_lines += [
    "",
    "Reading. Attention baselines lead on AUC by 3–4 percentage points, "
    "MARS dominates the ranking metrics (NDCG@10, MRR, P@10) by 12–41 "
    "percentage points. The dissociation supports the paper's claim "
    "that MARS's value comes from the multi-agent ranking pipeline, not "
    "from the choice of attention backbone.",
]
(OUT_TABLES / "table_attention_kt.md").write_text("\n".join(md_lines), encoding="utf-8")
print(f"wrote {OUT_TABLES/'table_attention_kt.md'}")


# ─── LaTeX (drop-in for sn-article.tex) ─────────────────────────────────

tex_lines = [
    "% Auto-generated by scripts/generate_attention_kt_table.py",
    "\\begin{table}[t]",
    "\\centering",
    "\\caption{Attention-based KT comparison on XES3G5M (5 seeds, "
    "mean$\\pm$std). All baselines (SAINT, AKT, SimpleKT, DTransformer) "
    "consume the same 14-dim per-step input as the MARS Prediction "
    "Agent and are trained from scratch with the same recipe (focal "
    "BCE + label smoothing, AdamW, $5\\!\\times\\!10^{-4}$ lr, "
    "patience 5). The comparison isolates the attention mechanism "
    "vs.\\ decoder choice from feature engineering. Attention baselines "
    "lead AUC by 3--4\\,pp; MARS dominates NDCG@10, MRR, and "
    "Precision@10 by 12--41\\,pp because MARS's value comes from the "
    "downstream multi-agent ranking pipeline (Thompson-sampled "
    "multi-strategy candidate generation, IRT-gated ZPD filtering, "
    "MMR diversification), not from the encoder.}",
    "\\label{tab:attention_kt}",
    "\\begin{tabular}{lrrrrrr}",
    "\\toprule",
    "Model & Params (M) & AUC & NDCG@10 & MRR & P@10 & Tag Cov.\\\\",
    "\\midrule",
]
for _, row in table.iterrows():
    name = row["Model"]
    if name == "MARS (Full)":
        tex_lines.append("\\midrule")
        name = "\\textbf{MARS (Full)}"
    tex_lines.append(
        f"{name} & {row.get('params_M', 0):.2f} & "
        f"{fmt(row.get('AUC mean'), row.get('AUC std'))} & "
        f"{fmt(row.get('NDCG@10 mean'), row.get('NDCG@10 std'))} & "
        f"{fmt(row.get('MRR mean'), row.get('MRR std'))} & "
        f"{fmt(row.get('P@10 mean'), row.get('P@10 std'))} & "
        f"{fmt(row.get('Tag Cov mean'), row.get('Tag Cov std'))}\\\\"
    )
tex_lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
(OUT_TABLES / "table_attention_kt.tex").write_text("\n".join(tex_lines), encoding="utf-8")
print(f"wrote {OUT_TABLES/'table_attention_kt.tex'}")


# ─── Heatmap figure ─────────────────────────────────────────────────────

metrics_for_heat = ["AUC", "NDCG@10", "MRR", "P@10", "Tag Cov"]
M = []
for _, row in table.iterrows():
    M.append([
        row.get("AUC mean", np.nan),
        row.get("NDCG@10 mean", np.nan),
        row.get("MRR mean", np.nan),
        row.get("P@10 mean", np.nan),
        row.get("Tag Cov mean", np.nan),
    ])
M = np.array(M, dtype=float)

# Per-column min-max normalisation so cells are visually comparable
norm_M = np.zeros_like(M)
for j in range(M.shape[1]):
    col = M[:, j]
    valid = ~np.isnan(col)
    if valid.sum() > 1:
        cmin = col[valid].min()
        cmax = col[valid].max()
        if cmax > cmin:
            norm_M[:, j] = (col - cmin) / (cmax - cmin)
        else:
            norm_M[:, j] = 0.5

fig, ax = plt.subplots(figsize=(7.2, 3.6))
im = ax.imshow(norm_M, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)

ax.set_xticks(np.arange(len(metrics_for_heat)))
ax.set_xticklabels(metrics_for_heat, fontsize=10)
ax.set_yticks(np.arange(len(table)))
ax.set_yticklabels(table["Model"].tolist(), fontsize=10)

# Highlight MARS row
mars_idx = table.index[table["Model"] == "MARS (Full)"].tolist()
if mars_idx:
    mi = mars_idx[0]
    ax.add_patch(plt.Rectangle((-0.5, mi - 0.5), len(metrics_for_heat), 1,
                                fill=False, edgecolor="#D55E00",
                                linewidth=2.0, zorder=10))

# Annotate cells with raw values
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        if np.isnan(v):
            continue
        bg = norm_M[i, j]
        color = "white" if bg > 0.55 else "black"
        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                fontsize=9, color=color)

ax.set_title("Attention-KT vs MARS on XES3G5M (per-column normalised)",
             fontsize=11)
fig.colorbar(im, ax=ax, label="per-metric normalised score",
              fraction=0.04, pad=0.02)
fig.tight_layout()
save_figure(fig, "fig_attention_kt_heatmap", results_dir=str(OUT_FIGS))
plt.close(fig)
print("DONE")
