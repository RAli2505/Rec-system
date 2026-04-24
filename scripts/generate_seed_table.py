"""
Generate per-seed stability table (replaces Fig. 7 box plots).

Source : results/xes3g5m/tables/table_seed_stability.csv
Outputs:
  results/xes3g5m/tables/table_seed_stability_full.tex   -- LaTeX, paper-ready
  results/xes3g5m/tables/table_seed_stability_full.md    -- Markdown preview
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

SEEDS = [42, 123, 456, 789, 2024]

# Map: paper metric label -> CSV row key
METRICS = [
    ("AUC-ROC",  "lstm_auc"),
    ("NDCG@10",  "ndcg@10"),
    ("P@10",     "precision@10"),
    ("MRR",      "mrr"),
    ("Coverage", "tag_coverage"),
]

src = pd.read_csv("results/xes3g5m/tables/table_seed_stability.csv")
src = src.set_index("Metric")

# Build per-seed value matrix
rows = []
for label, key in METRICS:
    seed_vals = [float(x) for x in str(src.loc[key, "Seeds"]).split()]
    if len(seed_vals) != len(SEEDS):
        raise ValueError(f"{key}: expected {len(SEEDS)} seeds, got {len(seed_vals)}")
    rows.append({"metric": label, "values": seed_vals})

# ─── LaTeX (Springer / booktabs style) ───────────────────────────────────

def fmt(v, dp=4):
    return f"{v:.{dp}f}"

lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\caption{Per-seed test metrics for MARS on XES3G5M across five "
             r"random seeds. CV denotes the coefficient of variation "
             r"(std/mean $\times$ 100).}")
lines.append(r"\label{tab:seed_stability_full}")
header_cols = " & ".join(["Seed"] + [m for m, _ in METRICS])
lines.append(r"\begin{tabular}{l" + "r" * len(METRICS) + "}")
lines.append(r"\toprule")
lines.append(header_cols + r" \\")
lines.append(r"\midrule")

# Per-seed rows
for i, seed in enumerate(SEEDS):
    cells = [str(seed)] + [fmt(r["values"][i]) for r in rows]
    lines.append(" & ".join(cells) + r" \\")

lines.append(r"\midrule")

# Mean ± Std row
mean_cells = ["Mean $\\pm$ Std"]
for r in rows:
    arr = np.array(r["values"])
    mean_cells.append(f"{arr.mean():.4f} $\\pm$ {arr.std(ddof=1):.4f}")
lines.append(" & ".join(mean_cells) + r" \\")

# Min / Max row
min_cells = ["Min / Max"]
for r in rows:
    arr = np.array(r["values"])
    min_cells.append(f"{arr.min():.4f} / {arr.max():.4f}")
lines.append(" & ".join(min_cells) + r" \\")

# CV row
cv_cells = ["CV (\\%)"]
for r in rows:
    arr = np.array(r["values"])
    cv = arr.std(ddof=1) / arr.mean() * 100
    cv_cells.append(f"{cv:.2f}")
lines.append(" & ".join(cv_cells) + r" \\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

tex = "\n".join(lines) + "\n"

tex_path = Path("results/xes3g5m/tables/table_seed_stability_full.tex")
tex_path.write_text(tex, encoding="utf-8")
print(f"Wrote {tex_path}")

# ─── Markdown preview ────────────────────────────────────────────────────

md = []
header = "| Seed | " + " | ".join(m for m, _ in METRICS) + " |"
sep    = "|------|" + "|".join("--------" for _ in METRICS) + "|"
md.append(header)
md.append(sep)

for i, seed in enumerate(SEEDS):
    cells = [str(seed)] + [fmt(r["values"][i]) for r in rows]
    md.append("| " + " | ".join(cells) + " |")

md.append(sep)

mean_cells = ["**Mean ± Std**"]
for r in rows:
    arr = np.array(r["values"])
    mean_cells.append(f"**{arr.mean():.4f} ± {arr.std(ddof=1):.4f}**")
md.append("| " + " | ".join(mean_cells) + " |")

min_cells = ["Min / Max"]
for r in rows:
    arr = np.array(r["values"])
    min_cells.append(f"{arr.min():.4f} / {arr.max():.4f}")
md.append("| " + " | ".join(min_cells) + " |")

cv_cells = ["CV (%)"]
for r in rows:
    arr = np.array(r["values"])
    cv = arr.std(ddof=1) / arr.mean() * 100
    cv_cells.append(f"{cv:.2f}")
md.append("| " + " | ".join(cv_cells) + " |")

md_path = Path("results/xes3g5m/tables/table_seed_stability_full.md")
md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
print(f"Wrote {md_path}")

print("\n---- preview ----")
print("\n".join(md))
