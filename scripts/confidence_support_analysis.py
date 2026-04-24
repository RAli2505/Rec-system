"""
Per-class support, confusion matrix, and honest reporting for the
6-class behavioural confidence taxonomy (reviewer A.1 #28).

The Confidence Agent is a *rule-based* classifier: its output is a
deterministic function of (is_correct, is_fast, changed_answer). On
data where both timing and answer-change are observed (EdNet), all six
classes are populated. On data where they are not, the scheme
degenerates to fewer distinct classes.

This script reports, for each dataset and for each of the 6 classes:

  - raw support count and percentage
  - precision / recall / f1 (trivially 1.0 when labels equal rules)
  - a 6x6 confusion matrix (diagonal by construction)

It then writes a short LaTeX block that can replace any
"F1 = 1.0" claim in the main text with an honest support-plus-
interpretation report.

No ML model is trained.

Outputs
-------
results/xes3g5m/confidence_support_xes3g5m.json
results/xes3g5m/confidence_support_ednet.json
results/xes3g5m/tables/table_confidence_support.md
results/xes3g5m/tables/table_confidence_support.tex
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLASS_NAMES = [
    "SOLID",
    "UNSURE_CORRECT",
    "FALSE_CONFIDENCE",
    "CLEAR_GAP",
    "DOUBT_CORRECT",
    "DOUBT_INCORRECT",
]


def confusion_from_metrics(class_distribution: dict[str, int]) -> np.ndarray:
    """6x6 confusion matrix; diagonal = support since classifier is rule-based."""
    mat = np.zeros((6, 6), dtype=int)
    for cls, count in class_distribution.items():
        if cls in CLASS_NAMES:
            i = CLASS_NAMES.index(cls)
            mat[i, i] = int(count)
    return mat


def per_class_report(dist: dict[str, int]) -> list[dict]:
    total = sum(dist.get(c, 0) for c in CLASS_NAMES)
    rows = []
    for c in CLASS_NAMES:
        support = int(dist.get(c, 0))
        frac = support / total if total > 0 else 0.0
        # Because classifier is rule-based:
        precision = 1.0 if support > 0 else float("nan")
        recall = 1.0 if support > 0 else float("nan")
        f1 = 1.0 if support > 0 else float("nan")
        rows.append({
            "class": c,
            "support": support,
            "pct_of_total": round(100 * frac, 2),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return rows


def load_distribution_from_metrics(metrics_path: Path) -> dict[str, int]:
    data = json.loads(metrics_path.read_text())
    cd = (data.get("agent_metrics", {}).get("confidence", {})
              .get("class_distribution", {}))
    return {k: int(v) for k, v in cd.items()}


def write_markdown(xes_dist, ednet_dist, out: Path) -> None:
    xes_rows = per_class_report(xes_dist)
    ed_rows = per_class_report(ednet_dist)

    lines = [
        "# Confidence Agent — Per-Class Support and Honest F1 Analysis",
        "",
        "The 6-class behavioural confidence classifier is **rule-based**: "
        "labels are a deterministic function of `(is_correct, is_fast, "
        "changed_answer)`. Reporting F1=1.0 against these same rule-"
        "generated labels is tautological; this document replaces that "
        "claim with a dataset-level support analysis.",
        "",
        "## Per-class support",
        "",
        "| Class | XES3G5M support | XES3G5M % | EdNet support | EdNet % |",
        "|---|---:|---:|---:|---:|",
    ]
    for x, e in zip(xes_rows, ed_rows):
        lines.append(
            f"| {x['class']} | {x['support']:,} | {x['pct_of_total']:.2f}% "
            f"| {e['support']:,} | {e['pct_of_total']:.2f}% |"
        )
    lines += [
        "",
        "## Confusion matrix (both datasets are trivially diagonal)",
        "",
        "Because the classifier **is** the rule set, every sample is "
        "classified to its own label by construction. The two matrices "
        "below therefore contain non-zero entries only on the diagonal; "
        "they are provided for transparency, not as evidence of "
        "predictive performance.",
        "",
        "### XES3G5M",
        "",
    ]

    def mat_md(mat: np.ndarray) -> list[str]:
        hdr = "| true \\\\ pred | " + " | ".join(CLASS_NAMES) + " |"
        sep = "|---|" + "|".join("---:" for _ in CLASS_NAMES) + "|"
        body = []
        for i, name in enumerate(CLASS_NAMES):
            row = [f"{mat[i, j]:,}" for j in range(6)]
            body.append(f"| {name} | " + " | ".join(row) + " |")
        return [hdr, sep, *body]

    xes_mat = confusion_from_metrics(xes_dist)
    ed_mat = confusion_from_metrics(ednet_dist)
    lines += mat_md(xes_mat)
    lines += ["", "### EdNet", ""]
    lines += mat_md(ed_mat)

    # Diagnose degeneracy
    xes_nonzero = sum(1 for r in xes_rows if r["support"] > 0)
    ed_nonzero = sum(1 for r in ed_rows if r["support"] > 0)
    lines += [
        "",
        "## Dataset-level diagnosis of the 6-class scheme",
        "",
        f"- XES3G5M activates **{xes_nonzero} / 6** classes "
        f"(support > 0). The remaining classes do not appear in this "
        f"corpus because XES3G5M lacks `changed_answer` and has a "
        f"compressed `elapsed_time` distribution.",
        f"- EdNet activates **{ed_nonzero} / 6** classes; all six are "
        f"populated with non-trivial support.",
        "",
        "**Implication for the paper.** The 6-class taxonomy is a "
        "property of the rule set, not of a learned model. Its value in "
        "MARS is interpretability (assigning a human-readable behavioural "
        "tag to every interaction) and downstream skill-delta signals, "
        "not classification accuracy. The per-seed ablation in "
        "`ablation_significance.md` shows that removing the class-derived "
        "skill-delta signal changes NDCG@10 by $\\approx 10^{-4}$ "
        "(Confidence column), consistent with the taxonomy being "
        "primarily an interpretability layer.",
        "",
        "## Drop-in replacement for the Confidence F1 claim",
        "",
        "Replace any text of the form",
        "",
        "> \"The behavioural confidence classifier achieves F1 = 1.0 on "
        "six-class labels.\"",
        "",
        "with:",
        "",
        "> \"The behavioural confidence classifier is rule-based: a "
        "deterministic function of `(is_correct, is_fast, changed_answer)`. "
        "Because predicted labels are the rules, classification F1 is "
        "trivially 1.0 and is not a measure of generalisation. We "
        "instead report per-class support (Appendix~S3.X): on EdNet "
        "all six classes are populated; on XES3G5M the taxonomy "
        "collapses to two populated classes due to missing "
        "answer-change signal. The 6-class scheme is therefore reported "
        "as an interpretability layer, not a predictive classifier, and "
        "its contribution is quantified via the corresponding row in "
        "Table~\\ref{tab:ablation_significance} rather than via F1.\"",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


def write_latex(xes_dist, ednet_dist, out: Path) -> None:
    xes_rows = per_class_report(xes_dist)
    ed_rows = per_class_report(ednet_dist)
    lines = [
        "% Auto-generated by scripts/confidence_support_analysis.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Per-class support for the six-class behavioural "
        "confidence taxonomy. Labels are assigned by a deterministic rule "
        "over \\texttt{(is\\_correct, is\\_fast, changed\\_answer)}; "
        "classification F1 against these same rule-generated labels is "
        "trivially 1.0 and is therefore not reported. Instead we report "
        "support, which reveals that the scheme activates all six classes "
        "on EdNet but collapses to two populated classes on XES3G5M due "
        "to the absence of \\texttt{changed\\_answer} signal in that "
        "corpus.}",
        "\\label{tab:confidence_support}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Class & XES3G5M \\# & XES3G5M \\% & EdNet \\# & EdNet \\%\\\\",
        "\\midrule",
    ]
    for x, e in zip(xes_rows, ed_rows):
        lines.append(
            f"\\texttt{{{x['class'].replace('_', r'\\_')}}} & "
            f"{x['support']:,} & {x['pct_of_total']:.2f} & "
            f"{e['support']:,} & {e['pct_of_total']:.2f}\\\\"
        )
    lines += [
        "\\bottomrule", "\\end{tabular}", "\\end{table}",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


def main() -> int:
    xes_src = ROOT / "results/xes3g5m/xes3g5m_full_s42_20260423_065329/metrics.json"
    ed_src  = ROOT / "results/ednet_comparable/ednet_comparable_s42_n6000_min20_20260416_204945/metrics.json"
    xes_dist = load_distribution_from_metrics(xes_src)
    ed_dist  = load_distribution_from_metrics(ed_src)

    (ROOT / "results/xes3g5m/confidence_support_xes3g5m.json").write_text(
        json.dumps({
            "class_distribution": xes_dist,
            "per_class": per_class_report(xes_dist),
            "confusion_matrix_classes": CLASS_NAMES,
            "confusion_matrix": confusion_from_metrics(xes_dist).tolist(),
            "note": "Rule-based classifier: matrix is diagonal by construction; "
                    "F1=1.0 is tautological.",
        }, indent=2))
    (ROOT / "results/xes3g5m/confidence_support_ednet.json").write_text(
        json.dumps({
            "class_distribution": ed_dist,
            "per_class": per_class_report(ed_dist),
            "confusion_matrix_classes": CLASS_NAMES,
            "confusion_matrix": confusion_from_metrics(ed_dist).tolist(),
            "note": "Rule-based classifier: matrix is diagonal by construction; "
                    "F1=1.0 is tautological.",
        }, indent=2))

    write_markdown(xes_dist, ed_dist, ROOT / "results/xes3g5m/tables/table_confidence_support.md")
    write_latex(xes_dist, ed_dist, ROOT / "results/xes3g5m/tables/table_confidence_support.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
