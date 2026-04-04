"""
Validate MARS pipeline scalability across different sample sizes.

Runs the data pipeline on 1K, 5K, 10K, and 50K users, measuring:
- Wall-clock time (load, preprocess, total)
- Peak memory usage
- Dataset statistics (rows, users after cleaning)
- Baseline metrics stability (accuracy rate, mean elapsed time)

Output: Markdown table for inclusion in Section 4 "Experimental Setup",
plus a representativeness test (KS test: sampled vs full population stats).

Usage
-----
    python validate_scaling.py [--data-dir data/raw] [--output results/scaling_report.md]
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.loader import EdNetLoader
from data.preprocessor import EdNetPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("validate_scaling")


def get_peak_memory_mb() -> float:
    """Return peak RSS in MB (cross-platform)."""
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def run_pipeline(
    data_dir: str,
    sample_users: int | None,
    stratified: bool = True,
) -> dict:
    """Run load + preprocess and return timing/stats."""
    gc.collect()
    mem_before = get_peak_memory_mb()

    loader = EdNetLoader(data_dir=data_dir)
    preprocessor = EdNetPreprocessor()

    # Load
    t0 = time.time()
    interactions = loader.load_interactions(
        sample_users=sample_users,
        stratified_sampling=stratified,
    )
    t_load = time.time() - t0

    n_raw_rows = len(interactions)
    n_raw_users = interactions["user_id"].nunique()

    # Preprocess
    t1 = time.time()
    chunked = sample_users is not None and sample_users > 5000
    splits = preprocessor.run(interactions, chunked=chunked)
    t_preprocess = time.time() - t1

    # Combine splits for stats
    all_data = pd.concat(splits.values(), ignore_index=True)
    n_clean_rows = len(all_data)
    n_clean_users = all_data["user_id"].nunique()

    # Basic stats
    accuracy_rate = float(all_data["correct"].mean()) if "correct" in all_data.columns else np.nan
    mean_elapsed = float(all_data["elapsed_time"].mean()) if "elapsed_time" in all_data.columns else np.nan
    median_elapsed = float(all_data["elapsed_time"].median()) if "elapsed_time" in all_data.columns else np.nan

    interactions_per_user = all_data.groupby("user_id").size()
    mean_interactions = float(interactions_per_user.mean())

    changed_pct = float(all_data["changed_answer"].mean()) * 100 if "changed_answer" in all_data.columns else np.nan

    mem_after = get_peak_memory_mb()

    result = {
        "sample_users": sample_users or "ALL",
        "load_time_s": round(t_load, 1),
        "preprocess_time_s": round(t_preprocess, 1),
        "total_time_s": round(t_load + t_preprocess, 1),
        "raw_rows": n_raw_rows,
        "raw_users": n_raw_users,
        "clean_rows": n_clean_rows,
        "clean_users": n_clean_users,
        "retention_pct": round(100 * n_clean_users / max(n_raw_users, 1), 1),
        "accuracy_rate": round(accuracy_rate, 4),
        "mean_elapsed_ms": round(mean_elapsed, 0),
        "median_elapsed_ms": round(median_elapsed, 0),
        "mean_interactions_per_user": round(mean_interactions, 1),
        "changed_answer_pct": round(changed_pct, 2),
        "peak_ram_mb": round(mem_after, 0),
        "ram_delta_mb": round(mem_after - mem_before, 0),
        # Keep distributions for KS tests
        "_interactions_per_user": interactions_per_user.values,
        "_elapsed_times": all_data["elapsed_time"].dropna().values if "elapsed_time" in all_data.columns else np.array([]),
        "_accuracy_per_user": all_data.groupby("user_id")["correct"].mean().values if "correct" in all_data.columns else np.array([]),
    }
    return result


def ks_test_representativeness(
    baseline: dict, sample: dict
) -> dict[str, float]:
    """
    Kolmogorov-Smirnov test between baseline (largest sample) and a smaller sample.

    Returns p-values for key distributions. p > 0.05 = representative.
    """
    results = {}

    for key, label in [
        ("_interactions_per_user", "interactions_per_user"),
        ("_elapsed_times", "elapsed_time"),
        ("_accuracy_per_user", "accuracy_per_user"),
    ]:
        a = baseline.get(key, np.array([]))
        b = sample.get(key, np.array([]))
        if len(a) > 0 and len(b) > 0:
            stat, pval = stats.ks_2samp(a, b)
            results[label] = {"statistic": round(stat, 4), "p_value": round(pval, 4)}
        else:
            results[label] = {"statistic": np.nan, "p_value": np.nan}

    return results


def format_report(
    results: list[dict],
    ks_results: dict | None = None,
) -> str:
    """Format results as a Markdown report."""
    lines = [
        "# MARS Pipeline Scaling Validation",
        "",
        "## Performance Table",
        "",
        "| N users | Load (s) | Preprocess (s) | Total (s) | "
        "Clean rows | Clean users | Retention % | RAM (MB) |",
        "|--------:|--------:|--------------:|--------:|----------:|-----------:|-----------:|--------:|",
    ]

    for r in results:
        lines.append(
            f"| {r['sample_users']:>7} | {r['load_time_s']:>7} | "
            f"{r['preprocess_time_s']:>14} | {r['total_time_s']:>7} | "
            f"{r['clean_rows']:>10,} | {r['clean_users']:>11,} | "
            f"{r['retention_pct']:>10} | {r['peak_ram_mb']:>7} |"
        )

    lines += [
        "",
        "## Metric Stability",
        "",
        "| N users | Accuracy | Mean elapsed (ms) | Median elapsed (ms) | "
        "Avg interactions/user | Changed answer % |",
        "|--------:|---------:|------------------:|--------------------:|"
        "---------------------:|-----------------:|",
    ]

    for r in results:
        lines.append(
            f"| {r['sample_users']:>7} | {r['accuracy_rate']:>8} | "
            f"{r['mean_elapsed_ms']:>17} | {r['median_elapsed_ms']:>19} | "
            f"{r['mean_interactions_per_user']:>20} | "
            f"{r['changed_answer_pct']:>16} |"
        )

    if ks_results:
        lines += [
            "",
            "## Representativeness (KS Test vs largest sample)",
            "",
            "| Sample | Distribution | KS statistic | p-value | Representative? |",
            "|-------:|:-------------|:------------:|--------:|:---------------:|",
        ]
        for sample_label, tests in ks_results.items():
            for dist_name, vals in tests.items():
                rep = "Yes" if vals["p_value"] > 0.05 else "**No**"
                lines.append(
                    f"| {sample_label} | {dist_name} | "
                    f"{vals['statistic']} | {vals['p_value']} | {rep} |"
                )

    lines += ["", "---", f"Generated by `validate_scaling.py`", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="MARS pipeline scaling validation")
    parser.add_argument("--data-dir", default="data/raw", help="Path to raw EdNet data")
    parser.add_argument("--output", default="results/scaling_report.md", help="Output report path")
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=[1000, 5000, 10000, 50000],
        help="Sample sizes to test",
    )
    args = parser.parse_args()

    results = []
    for n in args.sizes:
        logger.info("=" * 60)
        logger.info("Running pipeline with sample_users=%d", n)
        logger.info("=" * 60)
        try:
            r = run_pipeline(args.data_dir, sample_users=n)
            results.append(r)
            logger.info(
                "N=%d: %d clean users, %d rows, %.1fs total, %.0f MB RAM",
                n, r["clean_users"], r["clean_rows"],
                r["total_time_s"], r["peak_ram_mb"],
            )
        except Exception as e:
            logger.error("Failed for N=%d: %s", n, e)
            continue

    if len(results) < 2:
        logger.warning("Need at least 2 successful runs for KS tests")
        ks_results = None
    else:
        # Use the largest sample as baseline
        baseline = results[-1]
        ks_results = {}
        for r in results[:-1]:
            label = str(r["sample_users"])
            ks_results[label] = ks_test_representativeness(baseline, r)

    # Generate report
    report = format_report(results, ks_results)
    print(report)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Report saved to %s", out_path)


if __name__ == "__main__":
    main()
