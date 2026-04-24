"""
Sensitivity analysis for the Recommendation Agent's scoring weights
(reviewer block B.2 — defends "arbitrary weights gap_weight=0.25, MMR
λ=0.80"). Inference-only: uses seed_42's saved best.pt.

Sweeps:
  gap_weight ∈ {0.0, 0.1, 0.25, 0.5, 0.75}
  mmr_lambda ∈ {0.5, 0.7, 0.8, 0.9, 1.0}
= 25 grid points × ~3-4 min per eval ≈ 90 min.

Output:
  results/xes3g5m/sensitivity_rec_weights_s42.{json,csv}
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import torch

from agents.utils import set_global_seed
from agents.prediction_agent import (
    PredictionAgent, create_model, set_num_tags,
    DEVICE, NUM_CONF_CLASSES,
)
from data.xes3g5m_loader import load_xes3g5m
from scripts.run_xes3g5m_full import (
    build_xes3g5m_questions_df, build_xes3g5m_lectures_df,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("rec_weights_sensitivity")

SEED = 42
GAP_WEIGHTS = [0.0, 0.1, 0.25, 0.5, 0.75]
MMR_LAMBDAS = [0.5, 0.7, 0.8, 0.9, 1.0]


def find_pretrained() -> Path:
    paths = sorted(
        [p for p in glob.glob(f"results/xes3g5m/xes3g5m_full_s{SEED}_*")
         if any(d in p for d in ["20260423_06","20260423_07","20260423_08"])],
        key=os.path.getmtime, reverse=True,
    )
    for p in paths:
        if (Path(p) / "best.pt").exists():
            return Path(p) / "best.pt"
    raise FileNotFoundError("No best.pt for seed_42 today")


def run_one_grid_point(
    gap_w: float, mmr_lambda: float,
    train_df, val_df, test_df, questions_df, lectures_df,
    pretrained_pt: Path,
) -> dict:
    from agents.diagnostic_agent import DiagnosticAgent
    from agents.confidence_agent import ConfidenceAgent
    from agents.kg_agent import KnowledgeGraphAgent
    from agents.recommendation_agent import RecommendationAgent
    from agents.personalization_agent import PersonalizationAgent, extract_user_features
    from agents.orchestrator import Orchestrator

    set_global_seed(SEED)

    diag = DiagnosticAgent(seed=SEED)
    diag.calibrate_from_interactions(train_df, min_answers_per_q=5)
    conf = ConfidenceAgent()
    conf.train(train_df, irt_params=diag.irt_params)
    kg = KnowledgeGraphAgent()
    kg.build_graph(questions_df, lectures_df)
    kg.build_prerequisites(train_df, train_user_ids=set(train_df["user_id"].unique()))

    train_e = train_df.copy()
    test_e = test_df.copy()
    train_e["confidence_class"] = conf.classify_batch(interactions=train_e)["classes"]
    test_e["confidence_class"]  = conf.classify_batch(interactions=test_e)["classes"]

    pred = PredictionAgent()
    model = create_model(pred.model_type, num_conf_classes=NUM_CONF_CLASSES).to(DEVICE)
    state = torch.load(pretrained_pt, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    pred.model = model

    rec = RecommendationAgent(random_seed=SEED)
    # Override scoring weights — try multiple attribute names depending
    # on the agent implementation
    for attr_name, value in [
        ("gap_weight",  gap_w), ("_gap_weight",  gap_w),
        ("w_gap",       gap_w),
        ("mmr_lambda",  mmr_lambda), ("_mmr_lambda",  mmr_lambda),
        ("mmr_lambda_param", mmr_lambda),
    ]:
        if hasattr(rec, attr_name):
            setattr(rec, attr_name, value)
    if hasattr(rec, "_config") and isinstance(rec._config, dict):
        rec._config["gap_weight"] = gap_w
        rec._config.setdefault("mmr", {})["lambda"] = mmr_lambda

    if hasattr(rec, "initialize"):
        try:
            rec.initialize(
                questions_df=questions_df, lectures_df=lectures_df,
                interactions_df=train_e,
                train_user_ids=train_e["user_id"].unique().tolist(),
            )
        except Exception as e:
            logger.warning("RecommendationAgent.initialize failed: %s", e)
    if diag.irt_params is not None:
        rec.set_irt_params({
            str(qid): float(b)
            for qid, b in zip(diag.irt_params.question_ids, diag.irt_params.b)
        })

    pers = PersonalizationAgent()
    pers.train_clusters(extract_user_features(train_e))

    orch = Orchestrator(seed=SEED)
    for agent in (diag, conf, kg, pred, rec, pers):
        orch.register_agent(agent)
    metrics = orch.batch_evaluation(test_e, top_k=10, context_ratio=0.3)

    return {
        "gap_weight":  gap_w,
        "mmr_lambda":  mmr_lambda,
        **{k: metrics.get(k) for k in
           ("lstm_auc", "ndcg@10", "precision@10", "mrr", "tag_coverage")},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_students",       type=int, default=6000)
    parser.add_argument("--min_interactions", type=int, default=20)
    args = parser.parse_args()

    pretrained_pt = find_pretrained()
    logger.info("Loaded pretrained model from %s", pretrained_pt)

    train_df, val_df, test_df = load_xes3g5m(
        n_students=args.n_students,
        min_interactions=args.min_interactions,
        seed=SEED,
    )
    for df in [train_df, val_df, test_df]:
        df["confidence_class"] = 0
    train_max_id = max(int(t) for tags in train_df["tags"]
                        if isinstance(tags, list) and tags
                        for t in tags)
    set_num_tags(train_max_id + 1)

    questions_df = build_xes3g5m_questions_df("data/xes3g5m/XES3G5M")
    if "bundle_id"      not in questions_df.columns: questions_df["bundle_id"] = questions_df["question_id"]
    if "correct_answer" not in questions_df.columns: questions_df["correct_answer"] = "A"
    if "deployed_at"    not in questions_df.columns: questions_df["deployed_at"] = 0
    lectures_df = pd.DataFrame({
        "lecture_id": [], "tags": [], "part_id": [],
        "type_of": [], "bundle_id": [],
    })

    grid = list(itertools.product(GAP_WEIGHTS, MMR_LAMBDAS))
    logger.info("Sweeping %d grid points (%d gap_weight x %d mmr_lambda)",
                len(grid), len(GAP_WEIGHTS), len(MMR_LAMBDAS))

    out_json = ROOT / "results" / "xes3g5m" / "sensitivity_rec_weights_s42.json"
    out_csv  = ROOT / "results" / "xes3g5m" / "sensitivity_rec_weights_s42.csv"

    rows = []
    for i, (gw, lam) in enumerate(grid, 1):
        t0 = time.time()
        try:
            r = run_one_grid_point(gw, lam, train_df, val_df, test_df,
                                     questions_df, lectures_df, pretrained_pt)
            r["time_s"] = round(time.time() - t0, 1)
            rows.append(r)
            logger.info("[%d/%d] gap=%.2f lam=%.1f  NDCG=%.4f  Cov=%.4f  MRR=%.4f  (%.1fs)",
                         i, len(grid), gw, lam,
                         r.get("ndcg@10", 0), r.get("tag_coverage", 0),
                         r.get("mrr", 0), r["time_s"])
        except Exception as e:
            logger.exception("[%d/%d] FAILED gap=%.2f lam=%.1f: %s", i, len(grid), gw, lam, e)
            rows.append({"gap_weight": gw, "mmr_lambda": lam,
                          "error": str(e), "time_s": round(time.time() - t0, 1)})
        with open(out_json, "w") as f:
            json.dump(rows, f, indent=2, default=str)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logger.info("Saved %s and %s", out_json, out_csv)


if __name__ == "__main__":
    main()
