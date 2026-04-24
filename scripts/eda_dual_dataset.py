"""
Comparative EDA of the two benchmarks used in the paper — XES3G5M and
the 6000-user resample of EdNet KT2. Produces side-by-side statistics,
distribution plots, and a Markdown summary that can be dropped into an
appendix.

Uses the same loaders and the same 6000-user / min-20-interaction
sampling as the main pipeline, so the numbers match what the rest of
the paper reports.

Outputs
-------
results/xes3g5m/eda_dual/table_dual_stats.{md,csv,tex}
results/xes3g5m/eda_dual/fig_seq_length.{png,pdf}
results/xes3g5m/eda_dual/fig_accuracy_hist.{png,pdf}
results/xes3g5m/eda_dual/fig_elapsed_time.{png,pdf}
results/xes3g5m/eda_dual/fig_concept_coverage.{png,pdf}
results/xes3g5m/eda_dual/fig_correctness_per_user.{png,pdf}
results/xes3g5m/eda_dual/eda_summary.md
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.plot_style import setup_publication_style, save_figure, MARS_COLORS

from data.xes3g5m_loader import load_xes3g5m
from data.ednet_comparable_loader import load_ednet_comparable

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
logger = logging.getLogger("eda_dual")

setup_publication_style()

OUT_DIR = ROOT / "results/xes3g5m/eda_dual"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_STUDENTS = 6000
MIN_INTER = 20


# ── helpers ──────────────────────────────────────────────────────────

def summarize(name: str, train, val, test) -> dict:
    df = pd.concat([train, val, test], ignore_index=True)
    n_users = df["user_id"].nunique()
    n_rows = len(df)
    seq_lens = df.groupby("user_id").size()
    accuracy_per_user = df.groupby("user_id")["correct"].mean()
    tag_col = "tags" if "tags" in df.columns else ("tag" if "tag" in df.columns else None)
    n_concepts = None
    if tag_col is not None:
        tag_set: set = set()
        for t in df[tag_col].values:
            if isinstance(t, list):
                tag_set.update(int(x) for x in t if pd.notna(x))
            elif isinstance(t, (int, float)) and not pd.isna(t):
                tag_set.add(int(t))
            elif isinstance(t, str):
                for x in t.replace(";", ",").split(","):
                    if x.strip().isdigit():
                        tag_set.add(int(x.strip()))
        n_concepts = len(tag_set)
    n_questions = df["question_id"].nunique() if "question_id" in df.columns else None
    elapsed = None
    if "elapsed_time" in df.columns:
        elapsed = df["elapsed_time"].astype(float).values
        elapsed = elapsed[np.isfinite(elapsed) & (elapsed > 0)]
    changed_ratio = None
    if "changed_answer" in df.columns:
        changed_ratio = float(df["changed_answer"].astype(bool).mean())
    stats = {
        "dataset": name,
        "n_users": int(n_users),
        "n_interactions": int(n_rows),
        "avg_seq_len": float(seq_lens.mean()),
        "median_seq_len": float(seq_lens.median()),
        "std_seq_len": float(seq_lens.std(ddof=1)),
        "min_seq_len": int(seq_lens.min()),
        "max_seq_len": int(seq_lens.max()),
        "n_questions": int(n_questions) if n_questions is not None else None,
        "n_concepts": int(n_concepts) if n_concepts is not None else None,
        "overall_accuracy": float(df["correct"].astype(bool).mean()),
        "mean_user_accuracy": float(accuracy_per_user.mean()),
        "std_user_accuracy": float(accuracy_per_user.std(ddof=1)),
        "train_users": int(train["user_id"].nunique()),
        "val_users": int(val["user_id"].nunique()),
        "test_users": int(test["user_id"].nunique()),
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
    }
    if elapsed is not None and len(elapsed) > 0:
        stats["elapsed_median_ms"] = float(np.median(elapsed))
        stats["elapsed_p95_ms"] = float(np.percentile(elapsed, 95))
        stats["elapsed_mean_ms"] = float(np.mean(elapsed))
    if changed_ratio is not None:
        stats["changed_answer_ratio"] = changed_ratio
    return stats


def fmt(v, spec="{:,}"):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 1000 else f"{v:,.0f}"
    return spec.format(v)


# ── loading ──────────────────────────────────────────────────────────

logger.info("Loading XES3G5M (n=%d, seed=%d) ...", N_STUDENTS, SEED)
xes_train, xes_val, xes_test = load_xes3g5m(
    n_students=N_STUDENTS, min_interactions=MIN_INTER, seed=SEED,
)

logger.info("Loading EdNet-comparable (n=%d, seed=%d) ...", N_STUDENTS, SEED)
ed_train, ed_val, ed_test = load_ednet_comparable(
    n_students=N_STUDENTS, min_interactions=MIN_INTER, seed=SEED,
)

xes_stats = summarize("XES3G5M", xes_train, xes_val, xes_test)
ed_stats = summarize("EdNet KT2 (6k sample)", ed_train, ed_val, ed_test)

(OUT_DIR / "dual_stats.json").write_text(
    json.dumps({"XES3G5M": xes_stats, "EdNet": ed_stats}, indent=2),
    encoding="utf-8",
)


# ── table (csv, md, tex) ─────────────────────────────────────────────

rows = [
    ("Students (sampled)",          "n_users"),
    ("Interactions (total)",        "n_interactions"),
    ("Avg interactions / student",  "avg_seq_len"),
    ("Median interactions / student","median_seq_len"),
    ("Std interactions / student",  "std_seq_len"),
    ("Min / Max interactions",      None),  # special
    ("Unique questions",            "n_questions"),
    ("Unique concepts (KC)",        "n_concepts"),
    ("Overall accuracy",            "overall_accuracy"),
    ("Mean per-user accuracy",      "mean_user_accuracy"),
    ("Std per-user accuracy",       "std_user_accuracy"),
    ("Train / Val / Test users",    None),
    ("Train / Val / Test rows",     None),
    ("Median elapsed_time (ms)",    "elapsed_median_ms"),
    ("P95  elapsed_time (ms)",      "elapsed_p95_ms"),
    ("Answer-change rate",          "changed_answer_ratio"),
]

csv_lines = ["metric,XES3G5M,EdNet"]
md_lines = ["| Metric | XES3G5M | EdNet KT2 (6k sample) |", "|---|---:|---:|"]
tex_lines = [
    "% Auto-generated by scripts/eda_dual_dataset.py",
    "\\begin{table}[t]",
    "\\centering",
    "\\caption{Dataset statistics for the XES3G5M and EdNet KT2 benchmarks used in this paper. "
    "Both are sampled with the same protocol (6{,}000 users with $\\geq 20$ interactions) "
    "and share the 70/15/15 user-level split.}",
    "\\label{tab:dual_dataset_stats}",
    "\\begin{tabular}{lrr}",
    "\\toprule",
    "Metric & XES3G5M & EdNet KT2 (6k sample)\\\\",
    "\\midrule",
]

for label, key in rows:
    if label == "Min / Max interactions":
        a = f"{xes_stats['min_seq_len']} / {xes_stats['max_seq_len']}"
        b = f"{ed_stats['min_seq_len']} / {ed_stats['max_seq_len']}"
    elif label == "Train / Val / Test users":
        a = f"{xes_stats['train_users']} / {xes_stats['val_users']} / {xes_stats['test_users']}"
        b = f"{ed_stats['train_users']} / {ed_stats['val_users']} / {ed_stats['test_users']}"
    elif label == "Train / Val / Test rows":
        a = f"{xes_stats['train_rows']:,} / {xes_stats['val_rows']:,} / {xes_stats['test_rows']:,}"
        b = f"{ed_stats['train_rows']:,} / {ed_stats['val_rows']:,} / {ed_stats['test_rows']:,}"
    else:
        a = fmt(xes_stats.get(key))
        b = fmt(ed_stats.get(key))
    csv_lines.append(f'"{label}","{a}","{b}"')
    md_lines.append(f"| {label} | {a} | {b} |")
    tex_lines.append(f"{label} & {a} & {b}\\\\")

tex_lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

(OUT_DIR / "table_dual_stats.csv").write_text("\n".join(csv_lines), encoding="utf-8")
(OUT_DIR / "table_dual_stats.md").write_text("\n".join(md_lines), encoding="utf-8")
(OUT_DIR / "table_dual_stats.tex").write_text("\n".join(tex_lines), encoding="utf-8")
logger.info("wrote stats tables")


# ── figures ──────────────────────────────────────────────────────────

xes_df = pd.concat([xes_train, xes_val, xes_test], ignore_index=True)
ed_df  = pd.concat([ed_train,  ed_val,  ed_test ], ignore_index=True)

COL_XES = MARS_COLORS["primary"][0]
COL_EDN = MARS_COLORS["primary"][1]


def hist_side_by_side(xes_vals, ed_vals, xlabel, fname, bins=40, log_y=False,
                      title=None, xscale="linear"):
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    xes_vals = np.asarray(xes_vals, dtype=float)
    ed_vals = np.asarray(ed_vals, dtype=float)
    if xscale == "log":
        xes_vals = xes_vals[xes_vals > 0]
        ed_vals = ed_vals[ed_vals > 0]
        if len(xes_vals) == 0 or len(ed_vals) == 0:
            logger.warning("%s: empty values after positive filter", fname)
            return
        lo = min(xes_vals.min(), ed_vals.min())
        hi = max(xes_vals.max(), ed_vals.max())
        bin_edges = np.logspace(np.log10(lo), np.log10(hi), bins)
        ax.hist(xes_vals, bins=bin_edges, alpha=0.55, color=COL_XES, label="XES3G5M", density=True)
        ax.hist(ed_vals,  bins=bin_edges, alpha=0.55, color=COL_EDN, label="EdNet KT2", density=True)
        ax.set_xscale("log")
    else:
        lo = min(np.percentile(xes_vals, 0.5), np.percentile(ed_vals, 0.5))
        hi = max(np.percentile(xes_vals, 99.5), np.percentile(ed_vals, 99.5))
        bin_edges = np.linspace(lo, hi, bins)
        ax.hist(xes_vals, bins=bin_edges, alpha=0.55, color=COL_XES, label="XES3G5M", density=True)
        ax.hist(ed_vals,  bins=bin_edges, alpha=0.55, color=COL_EDN, label="EdNet KT2", density=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    if log_y:
        ax.set_yscale("log")
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, fname, results_dir=str(OUT_DIR))
    plt.close(fig)


# 1. sequence length
hist_side_by_side(
    xes_df.groupby("user_id").size().values,
    ed_df.groupby("user_id").size().values,
    xlabel="interactions per student (log scale)",
    fname="fig_seq_length",
    title="Sequence length distribution per student",
    xscale="log",
)

# 2. per-user accuracy
hist_side_by_side(
    xes_df.groupby("user_id")["correct"].mean().values,
    ed_df.groupby("user_id")["correct"].mean().values,
    xlabel="per-user accuracy",
    fname="fig_accuracy_hist",
    bins=30,
    title="Per-user accuracy distribution",
)

# 3. elapsed_time (if present)
if "elapsed_time" in xes_df.columns and "elapsed_time" in ed_df.columns:
    xes_elapsed = xes_df["elapsed_time"].astype(float).values
    ed_elapsed  = ed_df["elapsed_time"].astype(float).values
    xes_elapsed = xes_elapsed[np.isfinite(xes_elapsed) & (xes_elapsed > 0)]
    ed_elapsed  = ed_elapsed[np.isfinite(ed_elapsed) & (ed_elapsed > 0)]
    # clip heavy tails for display
    xes_elapsed = xes_elapsed[xes_elapsed < np.percentile(xes_elapsed, 99)]
    ed_elapsed  = ed_elapsed[ed_elapsed  < np.percentile(ed_elapsed,  99)]
    hist_side_by_side(
        xes_elapsed, ed_elapsed,
        xlabel="elapsed_time (ms, log scale)",
        fname="fig_elapsed_time",
        title="Response time distribution (elapsed_time)",
        xscale="log",
    )


# 4. concept coverage (how many concepts per user)
def concepts_per_user(df):
    tag_col = "tags" if "tags" in df.columns else "tag"
    out = []
    for uid, grp in df.groupby("user_id"):
        s: set = set()
        for t in grp[tag_col].values:
            if isinstance(t, list):
                s.update(int(x) for x in t if pd.notna(x))
            elif isinstance(t, (int, float)) and not pd.isna(t):
                s.add(int(t))
            elif isinstance(t, str):
                for x in t.replace(";", ",").split(","):
                    if x.strip().isdigit():
                        s.add(int(x.strip()))
        out.append(len(s))
    return np.array(out)


xes_cpu = concepts_per_user(xes_df)
ed_cpu  = concepts_per_user(ed_df)
hist_side_by_side(
    xes_cpu, ed_cpu,
    xlabel="unique concepts seen per student",
    fname="fig_concept_coverage",
    bins=40,
    title="Per-student concept coverage",
)


# 5. correctness panel (train/val/test overall + gap between datasets)
fig, ax = plt.subplots(figsize=(6.0, 3.0))
datasets = ["XES3G5M", "EdNet KT2"]
overall = [xes_stats["overall_accuracy"], ed_stats["overall_accuracy"]]
meanuser = [xes_stats["mean_user_accuracy"], ed_stats["mean_user_accuracy"]]
x = np.arange(len(datasets))
w = 0.35
ax.bar(x - w/2, overall,  w, color=COL_XES, label="overall accuracy")
ax.bar(x + w/2, meanuser, w, color=COL_EDN, label="mean per-user accuracy")
for i, v in enumerate(overall):
    ax.text(i - w/2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
for i, v in enumerate(meanuser):
    ax.text(i + w/2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylabel("accuracy")
ax.set_ylim(0, 1.0)
ax.set_title("Overall vs per-user accuracy")
ax.legend()
fig.tight_layout()
save_figure(fig, "fig_correctness_per_user", results_dir=str(OUT_DIR))
plt.close(fig)


# ── summary markdown ─────────────────────────────────────────────────

xa = xes_stats; ea = ed_stats
summary_lines = [
    "# Dual-dataset EDA (XES3G5M vs EdNet KT2)",
    "",
    "Both corpora are sampled with the same protocol — 6,000 students "
    "with at least 20 interactions each — and share the 70/15/15 "
    "user-level split. Full stats are in `table_dual_stats.{md,csv,tex}`.",
    "",
    "## Headline numbers",
    "",
    f"- **Interactions**: XES3G5M ≈ {xa['n_interactions']:,}; "
    f"EdNet ≈ {ea['n_interactions']:,} "
    f"(ratio ≈ {xa['n_interactions']/max(ea['n_interactions'],1):.2f}×).",
    f"- **Avg seq. length**: XES3G5M ≈ {xa['avg_seq_len']:.1f}, "
    f"EdNet ≈ {ea['avg_seq_len']:.1f}.",
    f"- **Unique concepts**: XES3G5M = {xa['n_concepts']}, "
    f"EdNet = {ea['n_concepts']}.",
    f"- **Unique questions**: XES3G5M = {xa['n_questions']:,}, "
    f"EdNet = {ea['n_questions']:,}.",
    f"- **Overall accuracy**: XES3G5M = {xa['overall_accuracy']:.3f}, "
    f"EdNet = {ea['overall_accuracy']:.3f}.",
    "",
    "## Behavioural signal availability",
    "",
]
if "changed_answer_ratio" in xa or "changed_answer_ratio" in ea:
    summary_lines.append(
        f"- **Answer-change rate**: XES3G5M = {xa.get('changed_answer_ratio', 0):.4f}, "
        f"EdNet = {ea.get('changed_answer_ratio', 0):.4f}. "
        "On XES3G5M this field is synthesised as zero (no native signal in the corpus), "
        "which is why the 6-class confidence taxonomy collapses to 2 populated classes "
        "(UNSURE_CORRECT, CLEAR_GAP). See `confidence_support_*.json`."
    )
if "elapsed_median_ms" in xa and "elapsed_median_ms" in ea:
    summary_lines.append(
        f"- **Median elapsed_time**: XES3G5M = {xa['elapsed_median_ms']:.0f} ms "
        f"(synthesised from timestamp deltas), EdNet = {ea['elapsed_median_ms']:.0f} ms "
        "(native field from the logging pipeline)."
    )

summary_lines += [
    "",
    "## Figures",
    "",
    "- `fig_seq_length.pdf` — sequence length distribution",
    "- `fig_accuracy_hist.pdf` — per-user accuracy distribution",
    "- `fig_elapsed_time.pdf` — response-time distribution (log scale)",
    "- `fig_concept_coverage.pdf` — per-student concept coverage",
    "- `fig_correctness_per_user.pdf` — overall vs per-user accuracy",
    "",
    "## Reproduction",
    "",
    "```bash",
    "python scripts/eda_dual_dataset.py",
    "```",
]

(OUT_DIR / "eda_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
logger.info("wrote summary")
logger.info("DONE -> %s", OUT_DIR)
