"""
5-seed ablation in INFERENCE mode (no Prediction Agent retraining).

Reuses the saved best.pt from each seed's main pipeline run instead of
training a fresh Transformer per ablation config. This converts the
ablation cost from ~3 hours/config to ~5 min/config — full 5-seed
ablation finishes in ~75 min instead of ~15 hours.

Per seed, runs 4 ablations: -Prediction, -KG, -Confidence, -IRT.
Full MARS metrics are loaded from the existing main-pipeline JSON.

Output:
  results/xes3g5m/ablation_inference_5seeds_<ts>/ablation_5seeds.json
  - one entry per (seed, config) with full eval_metrics
  - aggregated mean ± std across seeds for each config
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from agents.utils import set_global_seed
from agents.prediction_agent import (
    PredictionAgent, GapSequenceDataset, create_model, set_num_tags,
    DEVICE, NUM_CONF_CLASSES, NUM_WORKERS,
)
from data.xes3g5m_loader import load_xes3g5m
from scripts.run_xes3g5m_full import (
    build_xes3g5m_questions_df, build_xes3g5m_lectures_df,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("ablation_inference_5seeds")

SEEDS = [42, 123, 456, 789, 2024]
CONFIGS = [
    ("- Prediction",       {"disable_prediction": True}),
    ("- Knowledge Graph",  {"disable_kg":         True}),
    ("- Confidence",       {"disable_confidence": True}),
    ("- IRT (Diagnostic)", {"disable_irt":        True}),
]


def newest_main_run(seed: int) -> Path | None:
    """Return today's newest main-pipeline run dir for this seed."""
    paths = sorted(
        [p for p in glob.glob(f"results/xes3g5m/xes3g5m_full_s{seed}_*")
         if "20260423_06" in p or "20260423_07" in p
         or "20260423_08" in p or "20260423_09" in p
         or "20260423_1" in p],
        key=os.path.getmtime, reverse=True,
    )
    for p in paths:
        if (Path(p) / "best.pt").exists() and (Path(p) / "metrics.json").exists():
            return Path(p)
    return None


def run_one_config(
    config_name: str,
    seed: int,
    pretrained_pt: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    questions_df: pd.DataFrame,
    lectures_df: pd.DataFrame,
    disable_prediction: bool = False,
    disable_kg: bool = False,
    disable_confidence: bool = False,
    disable_irt: bool = False,
    save_per_user: bool = False,
) -> dict:
    """Same agent-disabling logic as run_xes3g5m_ablation.run_ablation_config,
    but loads best.pt instead of retraining the Prediction Agent."""

    from agents.diagnostic_agent import DiagnosticAgent
    from agents.confidence_agent import ConfidenceAgent
    from agents.kg_agent import KnowledgeGraphAgent
    from agents.recommendation_agent import RecommendationAgent
    from agents.personalization_agent import PersonalizationAgent, extract_user_features
    from agents.orchestrator import Orchestrator

    set_global_seed(seed)
    logger.info("─── seed=%d  %s ───", seed, config_name)

    # ── Diagnostic ──
    diag = DiagnosticAgent(seed=seed)
    diag.calibrate_from_interactions(train_df, min_answers_per_q=5)
    if disable_irt and diag.irt_params is not None:
        logger.info("  IRT DISABLED — zeroing irt_params")
        diag.irt_params.b[:] = 0.0
        diag.irt_params.a[:] = 1.0

    # ── Confidence ──
    conf = ConfidenceAgent()
    if not disable_confidence:
        conf.train(train_df, irt_params=diag.irt_params if hasattr(diag, "irt_params") else None)
    else:
        logger.info("  Confidence DISABLED")
        conf.train(train_df)

    # ── KG ──
    kg = KnowledgeGraphAgent()
    kg.build_graph(questions_df, lectures_df)
    if not disable_kg:
        kg.build_prerequisites(train_df, train_user_ids=set(train_df["user_id"].unique()))
    else:
        logger.info("  KG prerequisites DISABLED (graph still built for candidate retrieval)")

    # ── Prepare confidence labels for prediction input ──
    train_e = train_df.copy()
    val_e = val_df.copy()
    test_e = test_df.copy()
    if not disable_confidence:
        train_e["confidence_class"] = conf.classify_batch(interactions=train_e)["classes"]
        val_e["confidence_class"]   = conf.classify_batch(interactions=val_e)["classes"]
        test_e["confidence_class"]  = conf.classify_batch(interactions=test_e)["classes"]
    else:
        train_e["confidence_class"] = 0
        val_e["confidence_class"]   = 0
        test_e["confidence_class"]  = 0

    # ── Prediction (LOAD instead of train) ──
    pred = PredictionAgent()
    model = create_model(pred.model_type, num_conf_classes=NUM_CONF_CLASSES).to(DEVICE)
    if disable_prediction:
        logger.info("  Prediction DISABLED — using random-init model")
        # leave model un-trained = effectively random
    else:
        logger.info("  Loading pre-trained model from %s", pretrained_pt.name)
        state = torch.load(pretrained_pt, map_location=DEVICE)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
    model.eval()
    pred.model = model

    # ── Recommendation Agent ──
    rec = RecommendationAgent(random_seed=seed)
    if hasattr(rec, "initialize"):
        try:
            rec.initialize(
                questions_df=questions_df, lectures_df=lectures_df,
                interactions_df=train_e,
                train_user_ids=train_e["user_id"].unique().tolist(),
            )
        except Exception as e:
            logger.warning("RecommendationAgent.initialize failed: %s", e)

    if not disable_irt and diag.irt_params is not None:
        irt_difficulty = {
            str(qid): float(b)
            for qid, b in zip(diag.irt_params.question_ids, diag.irt_params.b)
        }
        rec.set_irt_params(irt_difficulty)

    # ── Personalization Agent ──
    pers = PersonalizationAgent()
    user_feats = extract_user_features(train_e)
    pers.train_clusters(user_feats)

    # ── Orchestrator eval ──
    orch = Orchestrator(seed=seed)
    for agent in (diag, conf, kg, pred, rec, pers):
        orch.register_agent(agent)
    metrics = orch.batch_evaluation(
        test_e, top_k=10, context_ratio=0.3, save_per_user=save_per_user,
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_students",       type=int, default=6000)
    parser.add_argument("--min_interactions", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / "xes3g5m" / f"ablation_inference_5seeds_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=== 5-SEED ABLATION (inference mode) — out_dir=%s ===", out_dir)

    all_results: dict[str, dict[int, dict]] = {
        name: {} for name, _ in CONFIGS
    }
    all_results["Full MARS"] = {}

    for seed in args.seeds:
        logger.info("\n" + "=" * 60)
        logger.info("SEED %d", seed)
        logger.info("=" * 60)

        # Find this seed's pretrained best.pt + metrics.json
        run = newest_main_run(seed)
        if run is None:
            logger.error("No main-pipeline run with best.pt for seed %d — skipping", seed)
            continue
        pretrained_pt = run / "best.pt"
        with open(run / "metrics.json") as f:
            full_run = json.load(f)
        full_eval = full_run.get("eval_metrics", {})
        full_eval["_source"] = run.name
        all_results["Full MARS"][seed] = full_eval
        logger.info("Full MARS reused from %s", run.name)

        # Load XES3G5M data once per seed
        train_df, val_df, test_df = load_xes3g5m(
            n_students=args.n_students,
            min_interactions=args.min_interactions,
            seed=seed,
        )
        for df in [train_df, val_df, test_df]:
            df["confidence_class"] = 0

        # Set NUM_TAGS dynamically
        train_max_id = max(int(t) for tags in train_df["tags"]
                            if isinstance(tags, list) and tags
                            for t in tags)
        n_tags = train_max_id + 1
        set_num_tags(n_tags)

        # Build questions + lectures (same for all configs)
        questions_df = build_xes3g5m_questions_df("data/xes3g5m/XES3G5M")
        if "bundle_id"      not in questions_df.columns: questions_df["bundle_id"] = questions_df["question_id"]
        if "correct_answer" not in questions_df.columns: questions_df["correct_answer"] = "A"
        if "deployed_at"    not in questions_df.columns: questions_df["deployed_at"] = 0
        lectures_df = pd.DataFrame({
            "lecture_id": [], "tags": [], "part_id": [],
            "type_of": [], "bundle_id": [],
        })

        # Run 4 ablation configs
        for cfg_name, cfg_kwargs in CONFIGS:
            t0 = time.time()
            try:
                metrics = run_one_config(
                    cfg_name, seed, pretrained_pt,
                    train_df, val_df, test_df,
                    questions_df, lectures_df,
                    **cfg_kwargs,
                )
                metrics["time_s"] = round(time.time() - t0, 1)
                all_results[cfg_name][seed] = metrics
                logger.info(
                    "  [seed=%d %s]  AUC=%.4f NDCG=%.4f Cov=%.4f  (%.1fs)",
                    seed, cfg_name,
                    metrics.get("lstm_auc", -1),
                    metrics.get("ndcg@10", -1),
                    metrics.get("tag_coverage", -1),
                    metrics["time_s"],
                )
            except Exception as e:
                logger.exception("FAILED seed=%d %s: %s", seed, cfg_name, e)
                all_results[cfg_name][seed] = {"error": str(e),
                                                "time_s": round(time.time() - t0, 1)}

        # Save after every seed (resilient to interruption)
        with open(out_dir / "ablation_5seeds.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # ── Aggregated mean ± std per config ──
    summary = {}
    for cfg_name, by_seed in all_results.items():
        per_seed = [m for m in by_seed.values() if isinstance(m, dict) and "error" not in m]
        if not per_seed:
            continue
        agg = {"n_seeds": len(per_seed)}
        for k in ("lstm_auc", "ndcg@10", "precision@10", "mrr",
                   "tag_coverage", "recall@10"):
            vals = [m[k] for m in per_seed if k in m and m[k] is not None]
            if vals:
                arr = np.asarray(vals, dtype=float)
                agg[f"{k}_mean"] = round(float(arr.mean()), 4)
                agg[f"{k}_std"]  = round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 4)
        summary[cfg_name] = agg

    with open(out_dir / "ablation_5seeds_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("\n=== SUMMARY ===")
    for cfg, agg in summary.items():
        logger.info("  %-22s n=%d  AUC=%.4f±%.4f  NDCG=%.4f±%.4f  Cov=%.4f±%.4f",
                     cfg, agg.get("n_seeds", 0),
                     agg.get("lstm_auc_mean", 0), agg.get("lstm_auc_std", 0),
                     agg.get("ndcg@10_mean", 0), agg.get("ndcg@10_std", 0),
                     agg.get("tag_coverage_mean", 0), agg.get("tag_coverage_std", 0))
    logger.info("Outputs: %s", out_dir)


if __name__ == "__main__":
    main()
