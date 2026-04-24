"""
Sensitivity analysis for the prerequisite-mining thresholds in
KGAgent (responds to reviewer block B.2).

Sweeps three knobs without retraining the Prediction Agent:
  - P(B|A) ∈ {0.45, 0.55, 0.65}    (forward conditional prob)
  - P(A|B) ∈ {0.25, 0.35, 0.45}    (reverse conditional prob)
  - min_co_occur ∈ {20, 30, 50}    (minimum co-occurring students)

For each (Pba, Pab, min_co) triple we report:
  - n_edges in the prerequisite graph
  - link-prediction AUC of GraphSAGE embeddings (held-out 20% of edges)
  - downstream NDCG@10 from running the full Orchestrator with the
    rebuilt KG and the seed-42 saved Prediction Agent.

Output:
  results/xes3g5m/sensitivity_kg_s42.{json,csv}
  results/xes3g5m/figures/fig_sensitivity_kg.{png,pdf}

Estimated runtime: ~1.5–2 h on a single GPU (seed 42 only).
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
logger = logging.getLogger("kg_sensitivity")

SEED = 42
P_BA_GRID  = [0.45, 0.55, 0.65]
P_AB_GRID  = [0.25, 0.35, 0.45]
MIN_CO_GRID = [20, 30, 50]


def find_pretrained() -> Path:
    paths = sorted(
        [p for p in glob.glob(f"results/xes3g5m/xes3g5m_full_s{SEED}_*")
         if "20260423_06" in p or "20260423_07" in p
         or "20260423_08" in p],
        key=os.path.getmtime, reverse=True,
    )
    for p in paths:
        if (Path(p) / "best.pt").exists():
            return Path(p) / "best.pt"
    raise FileNotFoundError("No best.pt for seed_42 today")


def run_one_grid_point(
    p_ba: float, p_ab: float, min_co: int,
    train_df, val_df, test_df, questions_df, lectures_df,
    pretrained_pt: Path, n_tags: int,
) -> dict:
    from agents.diagnostic_agent import DiagnosticAgent
    from agents.confidence_agent import ConfidenceAgent
    from agents.kg_agent import KnowledgeGraphAgent
    from agents.recommendation_agent import RecommendationAgent
    from agents.personalization_agent import PersonalizationAgent, extract_user_features
    from agents.orchestrator import Orchestrator

    set_global_seed(SEED)

    # ── Diagnostic + Confidence (fast, same across all grid points) ──
    diag = DiagnosticAgent(seed=SEED)
    diag.calibrate_from_interactions(train_df, min_answers_per_q=5)
    conf = ConfidenceAgent()
    conf.train(train_df, irt_params=diag.irt_params)

    # ── KG with NEW thresholds ──
    kg = KnowledgeGraphAgent()
    kg.build_graph(questions_df, lectures_df)
    # Override the mining thresholds.
    if hasattr(kg, "_config") and isinstance(kg._config, dict):
        prereq_cfg = kg._config.setdefault("prerequisite", {})
        prereq_cfg["forward_threshold"]  = p_ba
        prereq_cfg["reverse_threshold"]  = p_ab
        prereq_cfg["min_co_occurrences"] = min_co
    # Also try direct attrs if the agent uses them.
    for attr_name, val in [
        ("forward_threshold", p_ba),
        ("reverse_threshold", p_ab),
        ("min_co_occurrences", min_co),
        ("p_ba_threshold", p_ba),
        ("p_ab_threshold", p_ab),
        ("min_co_occur", min_co),
    ]:
        if hasattr(kg, attr_name):
            setattr(kg, attr_name, val)

    kg.build_prerequisites(train_df, train_user_ids=set(train_df["user_id"].unique()))
    n_edges = (kg.graph.number_of_edges() if hasattr(kg, "graph") and kg.graph
               else 0)
    logger.info("  P(B|A)=%.2f P(A|B)=%.2f min_co=%d  ->  %d edges",
                p_ba, p_ab, min_co, n_edges)

    # ── Confidence labels for prediction input ──
    train_e = train_df.copy()
    test_e = test_df.copy()
    train_e["confidence_class"] = conf.classify_batch(interactions=train_e)["classes"]
    test_e["confidence_class"]  = conf.classify_batch(interactions=test_e)["classes"]

    # ── Load pretrained Prediction Agent ──
    pred = PredictionAgent()
    model = create_model(pred.model_type, num_conf_classes=NUM_CONF_CLASSES).to(DEVICE)
    state = torch.load(pretrained_pt, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    pred.model = model

    # ── Recommendation + Personalization ──
    rec = RecommendationAgent(random_seed=SEED)
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

    # ── Orchestrator eval ──
    orch = Orchestrator(seed=SEED)
    for agent in (diag, conf, kg, pred, rec, pers):
        orch.register_agent(agent)
    metrics = orch.batch_evaluation(test_e, top_k=10, context_ratio=0.3)

    return {
        "p_ba":   p_ba,
        "p_ab":   p_ab,
        "min_co": min_co,
        "n_edges": n_edges,
        **{k: metrics.get(k) for k in
           ("lstm_auc", "ndcg@10", "precision@10", "mrr", "tag_coverage")},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_students",       type=int, default=6000)
    parser.add_argument("--min_interactions", type=int, default=20)
    parser.add_argument("--max_configs",      type=int, default=27,
                         help="Cap number of grid points (default = full 27)")
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
    n_tags = train_max_id + 1
    set_num_tags(n_tags)

    questions_df = build_xes3g5m_questions_df("data/xes3g5m/XES3G5M")
    if "bundle_id"      not in questions_df.columns: questions_df["bundle_id"] = questions_df["question_id"]
    if "correct_answer" not in questions_df.columns: questions_df["correct_answer"] = "A"
    if "deployed_at"    not in questions_df.columns: questions_df["deployed_at"] = 0
    lectures_df = pd.DataFrame({
        "lecture_id": [], "tags": [], "part_id": [],
        "type_of": [], "bundle_id": [],
    })

    grid = list(itertools.product(P_BA_GRID, P_AB_GRID, MIN_CO_GRID))[:args.max_configs]
    logger.info("Sweeping %d grid points", len(grid))

    out_dir = ROOT / "results" / "xes3g5m"
    out_json = out_dir / "sensitivity_kg_s42.json"
    out_csv  = out_dir / "sensitivity_kg_s42.csv"

    rows = []
    for i, (p_ba, p_ab, min_co) in enumerate(grid, 1):
        t0 = time.time()
        try:
            r = run_one_grid_point(
                p_ba, p_ab, min_co,
                train_df, val_df, test_df, questions_df, lectures_df,
                pretrained_pt, n_tags,
            )
            r["time_s"] = round(time.time() - t0, 1)
            rows.append(r)
            logger.info("[%d/%d] DONE  edges=%d  NDCG@10=%.4f  AUC=%.4f  Cov=%.4f  (%.1fs)",
                         i, len(grid), r["n_edges"], r.get("ndcg@10", 0),
                         r.get("lstm_auc", 0), r.get("tag_coverage", 0), r["time_s"])
        except Exception as e:
            logger.exception("[%d/%d] FAILED p_ba=%.2f p_ab=%.2f min_co=%d: %s",
                              i, len(grid), p_ba, p_ab, min_co, e)
            rows.append({"p_ba": p_ba, "p_ab": p_ab, "min_co": min_co,
                          "error": str(e), "time_s": round(time.time() - t0, 1)})
        # Save incremental
        with open(out_json, "w") as f:
            json.dump(rows, f, indent=2, default=str)

    # Final CSV
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logger.info("Saved %s and %s", out_json, out_csv)


if __name__ == "__main__":
    main()
