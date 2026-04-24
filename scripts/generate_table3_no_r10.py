"""
Generate paper Table 3 (Main results on XES3G5M test set) WITHOUT the R@10
column. Sources values from results/xes3g5m/tables/table_main_results.csv
(produced by aggregate_xes3g5m.py).

R@10 is dropped because MARS evaluates on a filtered candidate pool, so its
recall is not directly comparable with full-tag-pool baselines.

Outputs:
  results/xes3g5m/tables/table3_main_results_no_r10.tex
  results/xes3g5m/tables/table3_main_results_no_r10.md
"""

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

SRC = Path("results/xes3g5m/tables/table_main_results.csv")
if not SRC.exists():
    raise SystemExit(f"Missing {SRC} — run scripts/aggregate_xes3g5m.py first.")

df = pd.read_csv(SRC)
df = df.set_index("Metric")

# Display order: methods left -> right (worst -> best baseline -> ours).
# Skip any column not present in the source CSV (e.g. if BPR-MF/CF/Content
# extra baselines weren't run for this snapshot).
ALL_METHODS = ["Random", "Popularity", "BPR-MF", "CF-only", "Content-only",
               "DKT (LSTM)", "GRU", "MARS (ours)"]
METHODS = [m for m in ALL_METHODS if m in df.columns]
# Drop R@10 entirely. Retain AUC / NDCG / P@10 / MRR / Coverage.
METRICS = ["AUC-ROC", "NDCG@10", "Precision@10", "MRR", "Coverage"]
# Short labels for the table header
SHORT = {"AUC-ROC": "AUC", "NDCG@10": "NDCG@10",
         "Precision@10": "P@10", "MRR": "MRR", "Coverage": "Cov."}


def fmt_cell(raw):
    """Format a CSV cell for the table.
    Baselines: float -> "0.xxx"
    MARS: "0.xxxx +/- 0.xxxx"  ->  "0.xxx ± 0.xxx"
    """
    if pd.isna(raw):
        return "—"
    s = str(raw)
    m = re.match(r"^([\d.]+)\s*\+/-\s*([\d.]+)$", s)
    if m:
        return f"{float(m.group(1)):.3f} ± {float(m.group(2)):.3f}"
    try:
        return f"{float(s):.3f}"
    except ValueError:
        return s


def fmt_cell_tex(raw):
    """LaTeX version: ± becomes \\pm."""
    return fmt_cell(raw).replace("±", r"$\pm$")


# ─── LaTeX (booktabs, Springer/Elsevier-friendly) ────────────────────

lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\caption{Main results on XES3G5M test set across five random "
             r"seeds (mean $\pm$ std for MARS; baselines are deterministic "
             r"or single-seed). R@10 is omitted because MARS evaluates on a "
             r"filtered candidate pool, which makes recall not directly "
             r"comparable with full-tag-pool baselines. AUC-ROC is reported "
             r"only for sequence-based knowledge tracing models (DKT, GRU, "
             r"MARS); for non-KT baselines (Random, Popularity, BPR-MF, "
             r"CF-only, Content-only) the score vector is constant across "
             r"users, which makes per-concept macro-AUC degenerate to 0.5 "
             r"and not informative \textemdash{} those entries are marked "
             r"as \textemdash.}")
lines.append(r"\label{tab:main_results}")
lines.append(r"\begin{tabular}{l" + "r" * len(METHODS) + "}")
lines.append(r"\toprule")
lines.append("Metric & " + " & ".join(METHODS) + r" \\")
lines.append(r"\midrule")
NON_KT = {"Random", "Popularity", "BPR-MF", "CF-only", "Content-only"}
for m in METRICS:
    cells = []
    for meth in METHODS:
        if m == "AUC-ROC" and meth in NON_KT:
            cells.append(r"---")
        else:
            cells.append(fmt_cell_tex(df.loc[m, meth]))
    # Bold the MARS cell
    cells[-1] = r"\textbf{" + cells[-1] + r"}"
    lines.append(SHORT[m] + " & " + " & ".join(cells) + r" \\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

tex_path = Path("results/xes3g5m/tables/table3_main_results_no_r10.tex")
tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {tex_path}")

# ─── Markdown preview ────────────────────────────────────────────────

md = []
md.append("| Metric | " + " | ".join(METHODS) + " |")
md.append("|" + "|".join("---" for _ in range(len(METHODS) + 1)) + "|")
for m in METRICS:
    cells = []
    for meth in METHODS:
        if m == "AUC-ROC" and meth in NON_KT:
            cells.append("—")
        else:
            cells.append(fmt_cell(df.loc[m, meth]))
    cells[-1] = "**" + cells[-1] + "**"
    md.append("| " + SHORT[m] + " | " + " | ".join(cells) + " |")

md_path = Path("results/xes3g5m/tables/table3_main_results_no_r10.md")
md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
print(f"Wrote {md_path}")

print("\n---- preview ----")
print("\n".join(md))
