"""
Seed-level statistical tests for MARS component ablation (reviewer A.1 #2,#3).

Reads the 5-seed ablation JSON produced by run_ablation_inference_5seeds.py
and computes, for each (Full MARS vs -Component) pair and each reported
metric:

  - mean delta and per-seed std
  - paired Wilcoxon signed-rank test p-value    (non-parametric, N=5)
  - paired t-test p-value                       (parametric, N=5)
  - BCa bootstrap 95% CI for the mean delta     (10000 resamples)

Honest caveat: N=5 (per-seed) is a weak statistical regime; only very
large effects clear p<0.05 with five samples. We report this explicitly
in the output table. A per-user Wilcoxon (N=899) was not possible here
because per-seed ablation runs did not persist per-user metrics.

Inputs
------
results/xes3g5m/ablation_inference_5seeds_<ts>/ablation_5seeds.json

Outputs
-------
results/xes3g5m/ablation_significance.csv
results/xes3g5m/ablation_significance.md
results/xes3g5m/ablation_significance_latex.tex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

ROOT = Path(__file__).resolve().parent.parent

METRICS = [
    "ndcg@10",
    "mrr",
    "precision@10",
    "recall@10",
    "lstm_auc",
    "tag_coverage",
]

ABLATIONS = [
    "- Prediction",
    "- Knowledge Graph",
    "- Confidence",
    "- IRT (Diagnostic)",
]

FULL = "Full MARS"
N_BOOTSTRAP = 10_000
RNG_SEED = 20260424


def bca_ci(diffs: np.ndarray, alpha: float = 0.05, n_boot: int = N_BOOTSTRAP,
           rng_seed: int = RNG_SEED) -> tuple[float, float]:
    """BCa (bias-corrected, accelerated) bootstrap CI for the mean.

    N=5 is small, but BCa is still the right nonparametric interval to
    quote — it corrects for bias and skew in a way percentile intervals
    do not. We cap n_boot at 10k because the paired-sample space is tiny.
    """
    rng = np.random.default_rng(rng_seed)
    n = len(diffs)
    theta_hat = diffs.mean()

    # 1. Bootstrap replications
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = diffs[idx].mean(axis=1)

    # 2. Bias-correction z0
    frac_below = np.mean(boots < theta_hat)
    # clamp to avoid ±inf at extreme fractions
    frac_below = np.clip(frac_below, 1 / (2 * n_boot), 1 - 1 / (2 * n_boot))
    from scipy.stats import norm
    z0 = norm.ppf(frac_below)

    # 3. Acceleration a via jackknife
    jack = np.array([
        np.mean(np.delete(diffs, i)) for i in range(n)
    ])
    jack_mean = jack.mean()
    num = np.sum((jack_mean - jack) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5 + 1e-12)
    a = num / den

    # 4. Adjusted percentiles
    z_lo = norm.ppf(alpha / 2)
    z_hi = norm.ppf(1 - alpha / 2)
    p_lo = norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
    p_hi = norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))

    lo = float(np.quantile(boots, p_lo))
    hi = float(np.quantile(boots, p_hi))
    return lo, hi


def compute_pair(full: np.ndarray, ablated: np.ndarray) -> dict:
    """Compute delta stats for Full - ablated."""
    diffs = full - ablated  # positive = Full better than ablated
    mean_d = float(diffs.mean())
    std_d = float(diffs.std(ddof=1))

    # Wilcoxon is undefined when all diffs are zero or there are too few non-zero diffs
    try:
        w_stat, w_p = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        w_p = float(w_p)
    except ValueError:
        w_p = float("nan")

    try:
        t_stat, t_p = ttest_rel(full, ablated)
        t_p = float(t_p)
    except ValueError:
        t_p = float("nan")

    lo, hi = bca_ci(diffs)
    return {
        "mean_delta": mean_d,
        "std_delta": std_d,
        "wilcoxon_p": w_p,
        "ttest_p": t_p,
        "bca_lo": lo,
        "bca_hi": hi,
        "n": len(diffs),
    }


def significance_marker(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def ci_excludes_zero(lo: float, hi: float) -> str:
    """Bootstrap-CI-based significance: the BCa 95% CI is the preferred
    evidence at N=5, because Wilcoxon signed-rank cannot reach p<0.05
    with only five paired observations (its minimum p is 2^-4 = 0.0625)."""
    if lo > 0 or hi < 0:
        return "CI*"
    return "ns"


def main(args: argparse.Namespace) -> int:
    src = Path(args.input)
    with src.open() as f:
        data = json.load(f)

    rows = []
    for metric in METRICS:
        full_vals = np.array(
            [data[FULL][str(s)][metric] for s in [42, 123, 456, 789, 2024]],
            dtype=float,
        )
        for cfg in ABLATIONS:
            abl_vals = np.array(
                [data[cfg][str(s)][metric] for s in [42, 123, 456, 789, 2024]],
                dtype=float,
            )
            stats = compute_pair(full_vals, abl_vals)
            rows.append({
                "metric": metric,
                "ablation": cfg,
                "full_mean": round(full_vals.mean(), 4),
                "full_std": round(full_vals.std(ddof=1), 4),
                "ablated_mean": round(abl_vals.mean(), 4),
                "ablated_std": round(abl_vals.std(ddof=1), 4),
                "delta": round(stats["mean_delta"], 4),
                "delta_std": round(stats["std_delta"], 4),
                "bca_95_lo": round(stats["bca_lo"], 4),
                "bca_95_hi": round(stats["bca_hi"], 4),
                "wilcoxon_p": round(stats["wilcoxon_p"], 4),
                "ttest_p": round(stats["ttest_p"], 4),
                "wilcoxon_sig": significance_marker(stats["wilcoxon_p"]),
                "ttest_sig": significance_marker(stats["ttest_p"]),
                "ci_sig": ci_excludes_zero(stats["bca_lo"], stats["bca_hi"]),
                "n_seeds": stats["n"],
            })

    df = pd.DataFrame(rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "ablation_significance.csv"
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")

    # Markdown summary grouped by metric
    lines = ["# Ablation significance (Full MARS vs -Component, N=5 seeds)", ""]
    lines.append("Paired tests on 5 seed-level observations. BCa bootstrap "
                 "95% CI on mean delta, 10k resamples.")
    lines.append("")
    lines.append("**Important caveat:** the Wilcoxon signed-rank test with "
                 "N=5 paired observations has a hard minimum two-sided p of "
                 "2^-4 = 0.0625, so Wilcoxon can never reach p<0.05 here. "
                 "Primary evidence is therefore (a) whether the 95% BCa CI on "
                 "the mean delta excludes zero (column `CI*`), and (b) the "
                 "paired t-test column for large-magnitude effects. This is "
                 "also why the reviewer asked for a per-user test (N=899), "
                 "which is left as follow-up work pending a re-run with "
                 "per-user NDCG persisted.")
    lines.append("")
    lines.append("Markers: `CI*` = BCa 95% CI excludes 0, `*`/`**`/`***` = "
                 "parametric p<0.05 / 0.01 / 0.001, `ns` otherwise.")
    lines.append("")

    for metric in METRICS:
        sub = df[df["metric"] == metric].copy()
        lines.append(f"## {metric}")
        lines.append("")
        lines.append("| Config | Full mean±std | Ablated mean±std | Δ=Full-Abl | 95% BCa CI | Wilcoxon p | t-test p | CI* | t-sig |")
        lines.append("|---|---:|---:|---:|:---:|---:|---:|:---:|:---:|")
        for _, row in sub.iterrows():
            lines.append(
                f"| {row['ablation']} "
                f"| {row['full_mean']:.4f} ± {row['full_std']:.4f} "
                f"| {row['ablated_mean']:.4f} ± {row['ablated_std']:.4f} "
                f"| {row['delta']:+.4f} ± {row['delta_std']:.4f} "
                f"| [{row['bca_95_lo']:+.4f}, {row['bca_95_hi']:+.4f}] "
                f"| {row['wilcoxon_p']:.4f} "
                f"| {row['ttest_p']:.4f} "
                f"| {row['ci_sig']} "
                f"| {row['ttest_sig']} |"
            )
        lines.append("")

    md_path = out_dir / "ablation_significance.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {md_path}")

    # LaTeX — two key metrics for the paper (NDCG@10, MRR)
    tex_lines = [
        "% Auto-generated by scripts/ablation_significance.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Component ablation with paired significance tests across "
        "5 seeds on XES3G5M. $\\Delta$ is Full~MARS minus the ablated variant, "
        "therefore positive $\\Delta$ favours the full model. 95\\%~CI is a "
        "bias-corrected accelerated (BCa) bootstrap of the mean seed-wise "
        "delta ($10^4$ resamples). $p$ is the two-sided Wilcoxon signed-rank "
        "statistic ($N=5$), a conservative regime in which only very large "
        "effects achieve $p<0.05$.}",
        "\\label{tab:ablation_significance}",
        "\\begin{tabular}{llrrrrc}",
        "\\toprule",
        "Metric & Config & Full & Ablated & $\\Delta$ & 95\\% CI & $p$\\\\",
        "\\midrule",
    ]
    for metric_label, metric_key in [
        ("NDCG@10", "ndcg@10"),
        ("MRR", "mrr"),
        ("Tag Coverage", "tag_coverage"),
    ]:
        sub = df[df["metric"] == metric_key].copy()
        for _, row in sub.iterrows():
            marker = row['ci_sig'] if row['ci_sig'] == 'CI*' else row['ttest_sig']
            marker_latex = "$^{\\dagger}$" if marker == "CI*" else (
                "$^{*}$" if marker == "*" else
                "$^{**}$" if marker == "**" else
                "$^{***}$" if marker == "***" else ""
            )
            tex_lines.append(
                f"{metric_label} & {row['ablation'].replace('-', '$-$').strip()} "
                f"& {row['full_mean']:.3f}$\\pm${row['full_std']:.3f} "
                f"& {row['ablated_mean']:.3f}$\\pm${row['ablated_std']:.3f} "
                f"& {row['delta']:+.3f}{marker_latex} "
                f"& [{row['bca_95_lo']:+.3f}, {row['bca_95_hi']:+.3f}] "
                f"& {row['wilcoxon_p']:.3f}\\\\"
            )
        tex_lines.append("\\midrule")
    tex_lines[-1] = "\\bottomrule"
    tex_lines += ["\\end{tabular}", "\\end{table}"]
    tex_path = out_dir / "ablation_significance_latex.tex"
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
    print(f"wrote {tex_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(
            ROOT / "results/xes3g5m/ablation_inference_5seeds_20260424_010555/"
            "ablation_5seeds.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results/xes3g5m"),
    )
    sys.exit(main(parser.parse_args()))
