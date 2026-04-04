"""
Statistical significance tests for MARS vs baselines.

Reads multi-seed results and computes:
  - Wilcoxon signed-rank test (non-parametric, N=5)
  - Paired t-test (parametric alternative)
  - Cohen's d effect size
  - Bonferroni correction for multiple comparisons

Usage
-----
    python scripts/statistical_tests.py [--results-dir results] [--mars-method "MARS (full)"]

Input: results/aggregated/all_seeds_raw.csv (or per-seed JSONs)
Output: results/aggregated/statistical_tests.csv, statistical_tests_latex.tex
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
logger = logging.getLogger("stat_tests")


def compare_methods(
    mars_scores: list[float],
    baseline_scores: list[float],
    method_name: str,
    metric_name: str,
) -> dict:
    """
    Compare MARS vs one baseline on one metric across seeds.

    Parameters
    ----------
    mars_scores : list of float
        Metric values for MARS across seeds.
    baseline_scores : list of float
        Metric values for the baseline across seeds.

    Returns
    -------
    dict with test results.
    """
    mars = np.array(mars_scores, dtype=float)
    base = np.array(baseline_scores, dtype=float)
    diff = mars - base
    n = len(diff)

    result = {
        "method": method_name,
        "metric": metric_name,
        "mars_mean": round(float(np.mean(mars)), 4),
        "mars_std": round(float(np.std(mars, ddof=1)), 4) if n > 1 else 0.0,
        "baseline_mean": round(float(np.mean(base)), 4),
        "baseline_std": round(float(np.std(base, ddof=1)), 4) if n > 1 else 0.0,
        "diff_mean": round(float(np.mean(diff)), 4),
        "n_seeds": n,
    }

    # Cohen's d (effect size)
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 1e-10
    if std_diff < 1e-10:
        d = float("inf") if np.mean(diff) != 0 else 0.0
    else:
        d = float(np.mean(diff) / std_diff)

    result["cohens_d"] = round(d, 4)
    result["effect_size"] = (
        "large" if abs(d) > 0.8
        else "medium" if abs(d) > 0.5
        else "small"
    )

    # Wilcoxon signed-rank test (non-parametric)
    try:
        if n >= 5 and not np.all(diff == 0):
            stat, p_val = wilcoxon(mars, base)
            result["wilcoxon_stat"] = round(float(stat), 4)
            result["wilcoxon_p"] = round(float(p_val), 6)
        else:
            result["wilcoxon_stat"] = None
            result["wilcoxon_p"] = 1.0
    except ValueError:
        result["wilcoxon_stat"] = None
        result["wilcoxon_p"] = 1.0

    # Paired t-test (parametric)
    try:
        if n >= 2 and std_diff > 1e-10:
            t_stat, t_p = ttest_rel(mars, base)
            result["ttest_stat"] = round(float(t_stat), 4)
            result["ttest_p"] = round(float(t_p), 6)
        else:
            result["ttest_stat"] = None
            result["ttest_p"] = 1.0
    except ValueError:
        result["ttest_stat"] = None
        result["ttest_p"] = 1.0

    return result


def apply_bonferroni(results: list[dict], p_col: str = "wilcoxon_p") -> list[dict]:
    """Apply Bonferroni correction to p-values."""
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        logger.warning("statsmodels not installed — skipping Bonferroni correction")
        for r in results:
            r["corrected_p"] = r.get(p_col, 1.0)
            r["significant_corrected"] = r.get(p_col, 1.0) < 0.05
        return results

    p_values = [r.get(p_col, 1.0) for r in results]
    # Replace None with 1.0
    p_values = [p if p is not None else 1.0 for p in p_values]

    if len(p_values) == 0:
        return results

    reject, corrected_p, _, _ = multipletests(p_values, method="bonferroni")

    for r, cp, rej in zip(results, corrected_p, reject):
        r["corrected_p"] = round(float(cp), 6)
        r["significant_corrected"] = bool(rej)
        # Significance stars
        raw_p = r.get(p_col, 1.0) or 1.0
        if raw_p < 0.001:
            r["sig_stars"] = "***"
        elif raw_p < 0.01:
            r["sig_stars"] = "**"
        elif raw_p < 0.05:
            r["sig_stars"] = "*"
        else:
            r["sig_stars"] = "ns"

    return results


def load_multi_method_results(
    results_dir: Path,
    seeds: list[int],
) -> dict[str, pd.DataFrame]:
    """
    Load results for multiple methods/baselines.

    Looks for:
      - results/seed_{seed}/eval_metrics.json (MARS)
      - results/seed_{seed}/baselines/{method}.json (baselines)
      - results/aggregated/all_seeds_raw.csv (combined)
    """
    methods = {}

    # Try combined CSV first
    combined_path = results_dir / "aggregated" / "all_seeds_raw.csv"
    if combined_path.exists():
        df = pd.read_csv(combined_path)
        if "method" in df.columns:
            for method in df["method"].unique():
                methods[method] = df[df["method"] == method]
            return methods

    # Load MARS results from per-seed JSONs
    mars_rows = []
    for seed in seeds:
        eval_path = results_dir / f"seed_{seed}" / "eval_metrics.json"
        if eval_path.exists():
            with open(eval_path) as f:
                data = json.load(f)
                data["seed"] = seed
                data["method"] = "MARS (full)"
                mars_rows.append(data)

        # Load baselines if they exist
        baselines_dir = results_dir / f"seed_{seed}" / "baselines"
        if baselines_dir.exists():
            for bp in baselines_dir.glob("*.json"):
                with open(bp) as f:
                    bdata = json.load(f)
                    bdata["seed"] = seed
                    bdata["method"] = bp.stem
                    mars_rows.append(bdata)

    if mars_rows:
        df = pd.DataFrame(mars_rows)
        for method in df["method"].unique():
            methods[method] = df[df["method"] == method]

    return methods


def format_significance_table(
    results: list[dict],
    metrics: list[str] | None = None,
) -> str:
    """Format results as a readable Markdown table."""
    if not results:
        return "No comparison results."

    if metrics is None:
        metrics = sorted(set(r["metric"] for r in results))

    lines = [
        "## Statistical Significance (MARS vs Baselines)",
        "",
        "| Baseline | Metric | MARS | Baseline | Diff | Cohen's d | Effect | p-value | Sig |",
        "|----------|--------|------|----------|------|-----------|--------|---------|-----|",
    ]

    for r in results:
        sig = r.get("sig_stars", "")
        p = r.get("wilcoxon_p", 1.0)
        p_str = f"{p:.4f}" if p is not None else "N/A"
        lines.append(
            f"| {r['method']} | {r['metric']} | "
            f"{r['mars_mean']:.3f}±{r['mars_std']:.3f} | "
            f"{r['baseline_mean']:.3f}±{r['baseline_std']:.3f} | "
            f"{r['diff_mean']:+.3f} | {r['cohens_d']:.2f} | "
            f"{r['effect_size']} | {p_str} | {sig} |"
        )

    lines.append("")
    lines.append("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns: not significant")
    lines.append("P-values: Wilcoxon signed-rank test with Bonferroni correction")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="MARS statistical significance tests")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 2024])
    parser.add_argument("--mars-method", default="MARS (full)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    agg_dir = results_dir / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)

    # Load all methods
    methods = load_multi_method_results(results_dir, args.seeds)

    if args.mars_method not in methods:
        logger.error(
            "MARS results not found ('%s'). Available: %s",
            args.mars_method, list(methods.keys()),
        )
        logger.info("Run scripts/run_multi_seed.py first.")
        return

    mars_df = methods[args.mars_method]
    metric_cols = [
        c for c in mars_df.columns
        if c not in ("seed", "method", "eval_time_sec", "n_users_evaluated")
        and mars_df[c].dtype in ("float64", "int64", "float32")
    ]

    baseline_names = [m for m in methods if m != args.mars_method]

    if not baseline_names:
        logger.info(
            "No baselines found. Only MARS results available.\n"
            "Run baselines in 08_evaluation.ipynb and save per-seed results."
        )
        # Still output MARS-only summary
        print("\nMARS metrics across seeds:")
        for col in metric_cols:
            vals = mars_df[col].dropna().values
            if len(vals) > 0:
                print(f"  {col}: {np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}")
        return

    # Compare MARS vs each baseline on each metric
    all_comparisons = []
    for baseline_name in baseline_names:
        base_df = methods[baseline_name]
        for metric in metric_cols:
            if metric not in base_df.columns:
                continue

            mars_vals = mars_df.sort_values("seed")[metric].dropna().tolist()
            base_vals = base_df.sort_values("seed")[metric].dropna().tolist()

            # Align by seed count
            n = min(len(mars_vals), len(base_vals))
            if n < 2:
                continue

            result = compare_methods(
                mars_vals[:n], base_vals[:n],
                baseline_name, metric,
            )
            all_comparisons.append(result)

    # Apply Bonferroni
    all_comparisons = apply_bonferroni(all_comparisons)

    # Save
    comp_df = pd.DataFrame(all_comparisons)
    comp_df.to_csv(agg_dir / "statistical_tests.csv", index=False)

    # Display
    table = format_significance_table(all_comparisons)
    print(table)

    table_path = agg_dir / "statistical_tests.md"
    table_path.write_text(table, encoding="utf-8")
    logger.info("Results saved to %s", agg_dir)


if __name__ == "__main__":
    main()
