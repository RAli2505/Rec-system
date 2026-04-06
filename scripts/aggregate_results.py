"""
Aggregate multi-seed MARS results into publication-ready tables.

Reads results/seed_*/eval_metrics.json and produces:
  - results/aggregated/summary_table.csv  (mean +/- std per metric)
  - results/aggregated/summary_latex.tex  (LaTeX-ready table)
  - results/aggregated/per_agent_summary.csv
  - results/summary.json                  (official compact summary)
  - results/tables/table1_comparison.csv  (official current comparison table)

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


def _format_mean_std(vals: list[float] | np.ndarray, decimals: int = 4) -> str:
    """Format a list of values as 'mean ± std'."""
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return ""
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


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


def load_baseline_results(results_dir: Path) -> dict[str, dict]:
    """Load baseline metrics saved by run_multi_seed.py."""
    baseline_path = results_dir / "aggregated" / "baseline_results.json"
    if not baseline_path.exists():
        logger.warning("Missing %s", baseline_path)
        return {}

    with open(baseline_path) as f:
        data = json.load(f)
    logger.info("Loaded baseline results from %s", baseline_path)
    return data


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


def build_current_comparison_table(
    eval_results: list[dict],
    baseline_results: dict[str, dict],
) -> pd.DataFrame:
    """
    Build the current paper-facing comparison table from the latest artifacts.

    Uses only metrics that are actually produced by the latest pipeline:
    recommendation metrics from eval_metrics + key component metrics for MARS.
    """
    if not eval_results:
        return pd.DataFrame()

    eval_df = pd.DataFrame(eval_results)
    mars_metrics = {
        "ndcg@10": eval_df["ndcg@10"].dropna().astype(float).tolist() if "ndcg@10" in eval_df else [],
        "mrr": eval_df["mrr"].dropna().astype(float).tolist() if "mrr" in eval_df else [],
        "precision@10": eval_df["precision@10"].dropna().astype(float).tolist() if "precision@10" in eval_df else [],
        "recall@10": eval_df["recall@10"].dropna().astype(float).tolist() if "recall@10" in eval_df else [],
        "coverage": eval_df["coverage"].dropna().astype(float).tolist() if "coverage" in eval_df else [],
    }

    rows: list[dict] = []

    baseline_name_map = {
        "random": "Random",
        "popularity": "Popularity",
        "bpr": "BPR",
    }
    for key in ("random", "popularity", "bpr"):
        bl = baseline_results.get(key, {})
        rows.append({
            "Method": baseline_name_map[key],
            "NDCG@10": _format_mean_std([bl.get("ndcg@10", np.nan)]),
            "NDCG@10_mean": bl.get("ndcg@10", np.nan),
            "MRR": _format_mean_std([bl.get("mrr", np.nan)]),
            "MRR_mean": bl.get("mrr", np.nan),
            "Precision@10": _format_mean_std([bl.get("precision@10", np.nan)]),
            "Precision@10_mean": bl.get("precision@10", np.nan),
            "Recall@10": _format_mean_std([bl.get("recall@10", np.nan)]),
            "Recall@10_mean": bl.get("recall@10", np.nan),
            "Coverage": _format_mean_std([bl.get("coverage", np.nan)]),
            "Coverage_mean": bl.get("coverage", np.nan),
            "Run Type": "baseline",
            "Seeds": 1,
        })

    rows.append({
        "Method": "MARS (ours)",
        "NDCG@10": _format_mean_std(mars_metrics["ndcg@10"]),
        "NDCG@10_mean": float(np.mean(mars_metrics["ndcg@10"])) if mars_metrics["ndcg@10"] else np.nan,
        "MRR": _format_mean_std(mars_metrics["mrr"]),
        "MRR_mean": float(np.mean(mars_metrics["mrr"])) if mars_metrics["mrr"] else np.nan,
        "Precision@10": _format_mean_std(mars_metrics["precision@10"]),
        "Precision@10_mean": float(np.mean(mars_metrics["precision@10"])) if mars_metrics["precision@10"] else np.nan,
        "Recall@10": _format_mean_std(mars_metrics["recall@10"]),
        "Recall@10_mean": float(np.mean(mars_metrics["recall@10"])) if mars_metrics["recall@10"] else np.nan,
        "Coverage": _format_mean_std(mars_metrics["coverage"]),
        "Coverage_mean": float(np.mean(mars_metrics["coverage"])) if mars_metrics["coverage"] else np.nan,
        "Run Type": "single-seed" if len(eval_results) == 1 else "multi-seed",
        "Seeds": len(eval_results),
    })

    return pd.DataFrame(rows)


def build_official_summary(
    eval_results: list[dict],
    agent_results: list[dict],
    baseline_results: dict[str, dict],
) -> dict:
    """Build the compact official summary.json from the freshest artifacts."""
    if not eval_results:
        return {}

    eval_df = pd.DataFrame(eval_results)
    latest = eval_results[-1]

    best_baseline_name = None
    best_baseline_ndcg = None
    for name, metrics in baseline_results.items():
        ndcg = metrics.get("ndcg@10")
        if ndcg is None:
            continue
        if best_baseline_ndcg is None or ndcg > best_baseline_ndcg:
            best_baseline_ndcg = float(ndcg)
            best_baseline_name = name

    best_baseline_display = {
        "random": "Random",
        "popularity": "Popularity",
        "bpr": "BPR",
    }.get(best_baseline_name, best_baseline_name)

    summary = {
        "run_type": "single_seed" if len(eval_results) == 1 else "multi_seed",
        "seeds": [int(r["seed"]) for r in eval_results if "seed" in r],
        "n_seeds_completed": len(eval_results),
        "n_users_evaluated": int(round(float(eval_df["n_users_evaluated"].mean()))) if "n_users_evaluated" in eval_df else None,
        "mars_ndcg_10": _format_mean_std(eval_df["ndcg@10"].dropna().astype(float).tolist()) if "ndcg@10" in eval_df else "",
        "mars_mrr": _format_mean_std(eval_df["mrr"].dropna().astype(float).tolist()) if "mrr" in eval_df else "",
        "mars_precision_10": _format_mean_std(eval_df["precision@10"].dropna().astype(float).tolist()) if "precision@10" in eval_df else "",
        "mars_recall_10": _format_mean_std(eval_df["recall@10"].dropna().astype(float).tolist()) if "recall@10" in eval_df else "",
        "mars_coverage": _format_mean_std(eval_df["coverage"].dropna().astype(float).tolist()) if "coverage" in eval_df else "",
        "lstm_auc": _format_mean_std(eval_df["lstm_auc"].dropna().astype(float).tolist()) if "lstm_auc" in eval_df else "",
        "lstm_auc_weighted": _format_mean_std(eval_df["lstm_auc_weighted"].dropna().astype(float).tolist()) if "lstm_auc_weighted" in eval_df else "",
        "lstm_f1_micro": _format_mean_std(eval_df["lstm_f1_micro"].dropna().astype(float).tolist()) if "lstm_f1_micro" in eval_df else "",
        "lstm_threshold": _format_mean_std(eval_df["lstm_threshold"].dropna().astype(float).tolist()) if "lstm_threshold" in eval_df else "",
        "learning_gain": _format_mean_std(eval_df["learning_gain"].dropna().astype(float).tolist()) if "learning_gain" in eval_df else "",
        "learning_gain_trimmed": _format_mean_std(eval_df["learning_gain_trimmed"].dropna().astype(float).tolist()) if "learning_gain_trimmed" in eval_df else "",
        "best_baseline": best_baseline_display,
        "best_baseline_ndcg_10": best_baseline_ndcg,
        "improvement_over_best_ndcg_10_pct": round(
            100.0 * ((float(np.mean(eval_df["ndcg@10"])) - best_baseline_ndcg) / best_baseline_ndcg),
            2,
        ) if best_baseline_ndcg and "ndcg@10" in eval_df else None,
        "latest_eval_metrics": latest,
        "baselines": baseline_results,
    }

    if agent_results:
        pred_vals = [r.get("prediction", {}).get("val_auc") for r in agent_results if r.get("prediction", {}).get("val_auc") is not None]
        conf_vals = [r.get("confidence", {}).get("full_f1_macro") for r in agent_results if r.get("confidence", {}).get("full_f1_macro") is not None]
        kg_nodes = [r.get("knowledge_graph", {}).get("n_nodes") for r in agent_results if r.get("knowledge_graph", {}).get("n_nodes") is not None]
        kg_edges = [r.get("knowledge_graph", {}).get("n_edges") for r in agent_results if r.get("knowledge_graph", {}).get("n_edges") is not None]

        summary["prediction_val_auc"] = _format_mean_std(pred_vals) if pred_vals else ""
        summary["confidence_f1"] = _format_mean_std(conf_vals) if conf_vals else ""
        summary["kg_nodes"] = int(round(float(np.mean(kg_nodes)))) if kg_nodes else None
        summary["kg_edges"] = int(round(float(np.mean(kg_edges)))) if kg_edges else None

    return summary


def refresh_official_artifacts(
    results_dir: Path,
    eval_results: list[dict],
    agent_results: list[dict],
    baseline_results: dict[str, dict],
) -> None:
    """Write current source-of-truth summary and comparison table."""
    summary = build_official_summary(eval_results, agent_results, baseline_results)
    if summary:
        summary_path = results_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Official summary saved to %s", summary_path)

    table = build_current_comparison_table(eval_results, baseline_results)
    if not table.empty:
        tables_dir = results_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        table_path = tables_dir / "table1_comparison.csv"
        table.to_csv(table_path, index=False)
        logger.info("Official comparison table saved to %s", table_path)


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
    baseline_results = load_baseline_results(results_dir)

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

    refresh_official_artifacts(results_dir, eval_results, agent_results, baseline_results)


if __name__ == "__main__":
    main()
