"""
Aggregate multi-seed MARS results into publication-ready tables.

Reads results/seed_*/eval_metrics.json and produces:
  - results/aggregated/summary_table.csv  (mean +/- std per metric)
  - results/aggregated/summary_latex.tex  (LaTeX-ready table)
  - results/aggregated/per_agent_summary.csv

Usage
-----
    python scripts/aggregate_results.py [--results-dir results] [--seeds 42 123 456 789 2024]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
logger = logging.getLogger("aggregate")


def load_seed_results(
    results_dir: Path,
    seeds: list[int],
) -> tuple[list[dict], list[dict]]:
    """Load eval_metrics.json and agent_metrics.json for each seed."""
    eval_results = []
    agent_results = []

    for seed in seeds:
        seed_dir = results_dir / f"seed_{seed}"

        eval_path = seed_dir / "eval_metrics.json"
        if eval_path.exists():
            with open(eval_path) as f:
                data = json.load(f)
                data["seed"] = seed
                eval_results.append(data)
        else:
            logger.warning("Missing %s", eval_path)

        agent_path = seed_dir / "agent_metrics.json"
        if agent_path.exists():
            with open(agent_path) as f:
                data = json.load(f)
                data["seed"] = seed
                agent_results.append(data)

    logger.info(
        "Loaded %d eval results, %d agent results from %d seeds",
        len(eval_results), len(agent_results), len(seeds),
    )
    return eval_results, agent_results


def compute_summary(
    eval_results: list[dict],
    method_name: str = "MARS (full)",
) -> pd.DataFrame:
    """Compute mean, std, CI for each metric."""
    if not eval_results:
        return pd.DataFrame()

    df = pd.DataFrame(eval_results)
    metric_cols = [c for c in df.columns if c not in ("seed", "eval_time_sec", "n_users_evaluated")]

    rows = []
    n = len(df)
    for col in metric_cols:
        vals = df[col].dropna().values.astype(float)
        if len(vals) == 0:
            continue
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ci = 1.96 * std / np.sqrt(len(vals)) if len(vals) > 1 else 0.0

        rows.append({
            "method": method_name,
            "metric": col,
            "mean": round(mean, 4),
            "std": round(std, 4),
            "ci_95": round(ci, 4),
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
            "median": round(float(np.median(vals)), 4),
            "n_seeds": len(vals),
            "formatted": f"{mean:.3f}±{std:.3f}",
        })

    return pd.DataFrame(rows)


def compute_agent_summary(agent_results: list[dict]) -> pd.DataFrame:
    """Aggregate per-agent metrics across seeds."""
    if not agent_results:
        return pd.DataFrame()

    rows = []
    # Flatten agent metrics
    for agent_name in ["diagnostic", "confidence", "knowledge_graph", "prediction", "personalization"]:
        agent_vals = {}
        for result in agent_results:
            agent_data = result.get(agent_name, {})
            for key, val in agent_data.items():
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    if key not in agent_vals:
                        agent_vals[key] = []
                    agent_vals[key].append(float(val))

        for metric, vals in agent_vals.items():
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            rows.append({
                "agent": agent_name,
                "metric": metric,
                "mean": round(mean, 4),
                "std": round(std, 4),
                "formatted": f"{mean:.4f}±{std:.4f}",
                "n_seeds": len(vals),
            })

    return pd.DataFrame(rows)


def generate_latex_table(
    summary_df: pd.DataFrame,
    metrics_order: list[str] | None = None,
) -> str:
    """Generate a LaTeX-ready comparison table."""
    if summary_df.empty:
        return "% No data available"

    if metrics_order is None:
        metrics_order = summary_df["metric"].unique().tolist()

    # Pivot to method x metric
    methods = summary_df["method"].unique()
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Main Results (mean $\\pm$ std over 5 seeds)}")
    lines.append("\\label{tab:main_results}")

    # Column format
    n_metrics = len(metrics_order)
    col_fmt = "l" + "c" * n_metrics
    lines.append(f"\\begin{{tabular}}{{{col_fmt}}}")
    lines.append("\\toprule")

    # Header
    header = "Method"
    for m in metrics_order:
        # Format metric name: ndcg@10 -> NDCG@10
        display_name = m.upper().replace("_", " ").replace("@", "@")
        header += f" & {display_name}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Rows
    for method in methods:
        method_data = summary_df[summary_df["method"] == method]
        row = method
        for metric in metrics_order:
            match = method_data[method_data["metric"] == metric]
            if len(match) > 0:
                val = match.iloc[0]
                cell = f"${val['mean']:.3f} \\pm {val['std']:.3f}$"
            else:
                cell = "---"
            row += f" & {cell}"
        row += " \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed MARS results")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 2024])
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    agg_dir = results_dir / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)

    # Load
    eval_results, agent_results = load_seed_results(results_dir, args.seeds)

    if not eval_results:
        logger.error(
            "No eval results found. Run scripts/run_multi_seed.py first."
        )
        return

    # Aggregate evaluation metrics
    summary = compute_summary(eval_results, "MARS (full)")
    summary.to_csv(agg_dir / "summary_table.csv", index=False)
    logger.info("Summary table saved to %s", agg_dir / "summary_table.csv")

    # Print formatted results
    print("\n" + "=" * 70)
    print("MARS Multi-Seed Results Summary")
    print("=" * 70)
    for _, row in summary.iterrows():
        print(f"  {row['metric']:25s}: {row['formatted']}")
    print()

    # Agent-level summary
    agent_summary = compute_agent_summary(agent_results)
    if not agent_summary.empty:
        agent_summary.to_csv(agg_dir / "per_agent_summary.csv", index=False)
        print("Per-Agent Metrics:")
        for agent in agent_summary["agent"].unique():
            agent_data = agent_summary[agent_summary["agent"] == agent]
            print(f"\n  {agent}:")
            for _, row in agent_data.iterrows():
                print(f"    {row['metric']:30s}: {row['formatted']}")

    # LaTeX table
    latex = generate_latex_table(summary)
    latex_path = agg_dir / "summary_latex.tex"
    latex_path.write_text(latex, encoding="utf-8")
    logger.info("LaTeX table saved to %s", latex_path)
    print(f"\nLaTeX table:\n{latex}")


if __name__ == "__main__":
    main()
