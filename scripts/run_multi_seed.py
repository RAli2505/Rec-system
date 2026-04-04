"""
Run the full MARS pipeline across multiple seeds for statistical robustness.

For each seed:
  1. Set global seed (affects model init, dropout, K-Means, SMOTE)
  2. Load & preprocess data (chronological split is deterministic)
  3. Train all 6 agents
  4. Evaluate via Orchestrator.batch_evaluation()
  5. Collect per-agent metrics
  6. Save to results/seed_{seed}/

Usage
-----
    python scripts/run_multi_seed.py [--config configs/config.yaml] [--seeds 42 123 456 789 2024]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.utils import set_global_seed, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("multi_seed")


def load_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load preprocessed splits or run pipeline."""
    splits_dir = ROOT / "data" / "splits"

    if (splits_dir / "train.parquet").exists():
        train = pd.read_parquet(splits_dir / "train.parquet")
        val = pd.read_parquet(splits_dir / "val.parquet")
        test = pd.read_parquet(splits_dir / "test.parquet")
        logger.info(
            "Loaded splits: train=%d, val=%d, test=%d",
            len(train), len(val), len(test),
        )
        return train, val, test

    # Run full pipeline
    logger.info("No preprocessed data found — running full pipeline...")
    from data.loader import EdNetLoader
    from data.preprocessor import EdNetPreprocessor

    data_cfg = config.get("data", {})
    loader = EdNetLoader(data_dir="data/raw")
    interactions = loader.load_interactions(
        sample_users=data_cfg.get("sample_users", 50000),
        stratified_sampling=data_cfg.get("stratified_sampling", True),
    )

    preprocessor = EdNetPreprocessor()
    splits = preprocessor.run(
        interactions,
        chunked=data_cfg.get("chunked_processing", True),
    )
    return splits["train"], splits["val"], splits["test"]


def train_all_agents(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    seed: int,
    config: dict,
) -> dict:
    """Train all 6 agents and return per-agent metrics."""
    from agents.diagnostic_agent import DiagnosticAgent
    from agents.confidence_agent import ConfidenceAgent
    from agents.kg_agent import KnowledgeGraphAgent
    from agents.prediction_agent import PredictionAgent
    from agents.recommendation_agent import RecommendationAgent
    from agents.personalization_agent import PersonalizationAgent, extract_user_features
    from data.loader import EdNetLoader

    set_global_seed(seed)
    agent_metrics = {}

    # 1. DiagnosticAgent (IRT calibration)
    logger.info("[seed=%d] Training DiagnosticAgent...", seed)
    t0 = time.time()
    diag = DiagnosticAgent(seed=seed)
    irt_params = diag.calibrate_from_interactions(train_df, min_answers_per_q=20)

    # Compute correlation with raw accuracy
    user_accuracy = train_df.groupby("user_id")["correct"].mean()
    thetas = {}
    for uid, grp in train_df.groupby("user_id"):
        responses = []
        for _, row in grp.head(50).iterrows():
            qid = str(row["question_id"])
            idx = diag._qid_to_idx.get(qid)
            if idx is not None:
                responses.append((idx, bool(row["correct"])))
        if responses:
            theta, se = diag.update_theta(responses)
            thetas[uid] = (theta, se)

    common = set(thetas.keys()) & set(user_accuracy.index)
    if len(common) > 2:
        from scipy.stats import pearsonr
        theta_vals = [thetas[u][0] for u in common]
        acc_vals = [user_accuracy[u] for u in common]
        r_val, _ = pearsonr(theta_vals, acc_vals)
        se_vals = [thetas[u][1] for u in common]
        agent_metrics["diagnostic"] = {
            "r_theta_accuracy": round(r_val, 4),
            "mean_sem": round(float(np.mean(se_vals)), 4),
            "n_items": len(irt_params),
            "n_users_estimated": len(thetas),
            "time_s": round(time.time() - t0, 1),
        }
    else:
        agent_metrics["diagnostic"] = {"r_theta_accuracy": 0.0, "time_s": round(time.time() - t0, 1)}

    # 2. ConfidenceAgent (XGBoost)
    logger.info("[seed=%d] Training ConfidenceAgent...", seed)
    t0 = time.time()
    conf = ConfidenceAgent()
    conf_cfg = config.get("confidence", {})
    cv_results = conf.train(
        train_df,
        irt_params=irt_params,
        use_smote=conf_cfg.get("use_smote", False),
        use_class_weight=conf_cfg.get("use_class_weight", True),
    )
    agent_metrics["confidence"] = {
        **cv_results,
        "time_s": round(time.time() - t0, 1),
    }

    # 3. KGAgent (GraphSAGE)
    logger.info("[seed=%d] Training KGAgent...", seed)
    t0 = time.time()
    kg = KnowledgeGraphAgent()
    loader = EdNetLoader(data_dir="data/raw")
    questions_df = loader.questions
    lectures_df = loader.lectures
    kg.build_graph(questions_df, lectures_df)
    kg.build_prerequisites(train_df)

    kg_metrics_dict = {
        "n_nodes": kg.graph.number_of_nodes(),
        "n_edges": kg.graph.number_of_edges(),
        "time_s": round(time.time() - t0, 1),
    }
    # Train embeddings if method exists
    if hasattr(kg, "train_embeddings"):
        try:
            emb_metrics = kg.train_embeddings()
            kg_metrics_dict.update(emb_metrics)
        except Exception as e:
            logger.warning("GraphSAGE training failed: %s", e)
    agent_metrics["knowledge_graph"] = kg_metrics_dict

    # 4. PredictionAgent (LSTM)
    logger.info("[seed=%d] Training PredictionAgent...", seed)
    t0 = time.time()
    pred = PredictionAgent()
    pred_results = pred.train(train_df, val_df=val_df)
    agent_metrics["prediction"] = {
        **pred_results,
        "time_s": round(time.time() - t0, 1),
    }

    # 5. RecommendationAgent (TS + LambdaMART)
    logger.info("[seed=%d] Training RecommendationAgent...", seed)
    t0 = time.time()
    rec = RecommendationAgent(random_seed=seed)
    # Initialize with data if method exists
    if hasattr(rec, "initialize"):
        try:
            rec.initialize(
                questions_df=questions_df,
                lectures_df=lectures_df,
                interactions_df=train_df,
                train_user_ids=train_df["user_id"].unique().tolist(),
            )
        except Exception as e:
            logger.warning("RecommendationAgent init: %s", e)
    agent_metrics["recommendation"] = {
        "time_s": round(time.time() - t0, 1),
    }

    # 6. PersonalizationAgent (K-Means)
    logger.info("[seed=%d] Training PersonalizationAgent...", seed)
    t0 = time.time()
    pers = PersonalizationAgent()
    user_feats = extract_user_features(train_df)
    optimal_k = pers.train_clusters(user_feats)
    agent_metrics["personalization"] = {
        "optimal_k": optimal_k,
        "silhouette_scores": pers._silhouette_scores,
        "time_s": round(time.time() - t0, 1),
    }

    # Return agents + metrics for evaluation
    agents = {
        "diagnostic": diag,
        "confidence": conf,
        "knowledge_graph": kg,
        "prediction": pred,
        "recommendation": rec,
        "personalization": pers,
    }
    return agents, agent_metrics


def evaluate_pipeline(
    agents: dict,
    test_df: pd.DataFrame,
    seed: int,
) -> dict[str, float]:
    """Register agents with orchestrator and run batch evaluation."""
    from agents.orchestrator import Orchestrator

    set_global_seed(seed)
    orch = Orchestrator(seed=seed)

    for agent in agents.values():
        orch.register_agent(agent)

    metrics = orch.batch_evaluation(test_df, top_k=10)
    return metrics


def save_results(
    seed: int,
    agent_metrics: dict,
    eval_metrics: dict,
    output_dir: Path,
) -> None:
    """Save per-seed results to JSON."""
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Combined metrics
    combined = {
        "seed": seed,
        "agent_metrics": agent_metrics,
        "eval_metrics": eval_metrics,
    }

    with open(seed_dir / "metrics.json", "w") as f:
        json.dump(combined, f, indent=2, default=str)

    # Also save eval metrics separately for easy aggregation
    with open(seed_dir / "eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2, default=str)

    with open(seed_dir / "agent_metrics.json", "w") as f:
        json.dump(agent_metrics, f, indent=2, default=str)

    logger.info("Results saved to %s", seed_dir)


def main():
    parser = argparse.ArgumentParser(description="MARS multi-seed evaluation")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Seeds to evaluate (default: from config)",
    )
    parser.add_argument("--output", default="results", help="Output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    seeds = args.seeds or config.get("global", {}).get(
        "seeds_for_repeats", [42, 123, 456, 789, 2024]
    )
    output_dir = Path(args.output)

    logger.info("Seeds: %s", seeds)
    logger.info("Config: %s", args.config)

    # Load data once (chronological split is deterministic)
    train_df, val_df, test_df = load_data(config)

    all_eval_metrics = []
    total_t0 = time.time()

    for i, seed in enumerate(seeds):
        logger.info("=" * 70)
        logger.info("SEED %d/%d: %d", i + 1, len(seeds), seed)
        logger.info("=" * 70)

        seed_t0 = time.time()

        # Train
        agents, agent_metrics = train_all_agents(train_df, val_df, seed, config)

        # Evaluate
        eval_metrics = evaluate_pipeline(agents, test_df, seed)
        all_eval_metrics.append({"seed": seed, **eval_metrics})

        # Save
        save_results(seed, agent_metrics, eval_metrics, output_dir)

        seed_elapsed = time.time() - seed_t0
        logger.info(
            "Seed %d complete in %.1fs. Metrics: %s",
            seed, seed_elapsed,
            {k: v for k, v in eval_metrics.items() if not k.startswith("eval_")},
        )

    # Save summary
    summary_df = pd.DataFrame(all_eval_metrics)
    summary_path = output_dir / "aggregated" / "all_seeds_raw.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    total_elapsed = time.time() - total_t0
    logger.info("=" * 70)
    logger.info("ALL SEEDS COMPLETE in %.1fs", total_elapsed)
    logger.info("Raw results: %s", summary_path)

    # Quick summary
    metric_cols = [c for c in summary_df.columns if c != "seed"]
    for col in metric_cols:
        vals = summary_df[col].dropna()
        if len(vals) > 0:
            logger.info(
                "  %s: %.4f +/- %.4f",
                col, vals.mean(), vals.std(),
            )


if __name__ == "__main__":
    main()
