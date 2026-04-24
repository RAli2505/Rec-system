"""
Ablation study for XES3G5M: run full pipeline with each agent disabled.

Usage:
    python scripts/run_xes3g5m_ablation.py [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.utils import set_global_seed
from data.xes3g5m_loader import load_xes3g5m
from scripts.run_xes3g5m_full import (
    build_xes3g5m_questions_df, build_xes3g5m_lectures_df,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("xes3g5m_ablation")


def run_ablation_config(
    config_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    questions_df: pd.DataFrame,
    lectures_df: pd.DataFrame,
    seed: int,
    run_dir: Path,
    disable_prediction: bool = False,
    disable_kg: bool = False,
    disable_confidence: bool = False,
    disable_irt: bool = False,
) -> dict:
    """Run one ablation configuration and return metrics."""
    from agents.diagnostic_agent import DiagnosticAgent
    from agents.confidence_agent import ConfidenceAgent
    from agents.kg_agent import KnowledgeGraphAgent
    from agents.prediction_agent import (
        PredictionAgent, GapSequenceDataset, create_model,
        DEVICE, NUM_CONF_CLASSES, NUM_WORKERS, LABEL_SMOOTHING,
    )
    from agents.recommendation_agent import RecommendationAgent
    from agents.personalization_agent import PersonalizationAgent, extract_user_features
    from agents.orchestrator import Orchestrator
    import torch.nn as nn

    set_global_seed(seed)
    logger.info("=== %s ===", config_name)

    # 1. Diagnostic
    diag = DiagnosticAgent(seed=seed)
    if not disable_irt:
        diag.calibrate_from_interactions(train_df, min_answers_per_q=5)
    else:
        logger.info("  IRT DISABLED")
        diag.calibrate_from_interactions(train_df, min_answers_per_q=5)
        # Zero out IRT params
        if diag.irt_params is not None:
            diag.irt_params.b[:] = 0.0
            diag.irt_params.a[:] = 1.0

    # 2. Confidence
    conf = ConfidenceAgent()
    if not disable_confidence:
        conf.train(train_df, irt_params=diag.irt_params if hasattr(diag, 'irt_params') else None)
    else:
        logger.info("  Confidence DISABLED")
        conf.train(train_df)

    # 3. KG
    kg = KnowledgeGraphAgent()
    if not disable_kg:
        kg.build_graph(questions_df, lectures_df)
        kg.build_prerequisites(train_df, train_user_ids=set(train_df["user_id"].unique()))
    else:
        logger.info("  KG DISABLED")
        kg.build_graph(questions_df, lectures_df)

    # 4. Prediction
    train_e = train_df.copy()
    val_e = val_df.copy()
    if not disable_confidence:
        train_e["confidence_class"] = conf.classify_batch(interactions=train_e)["classes"]
        val_e["confidence_class"] = conf.classify_batch(interactions=val_e)["classes"]
    else:
        train_e["confidence_class"] = 0
        val_e["confidence_class"] = 0

    pred = PredictionAgent()
    if not disable_prediction:
        # Custom training loop (same as run_xes3g5m_full.py)
        from sklearn.metrics import roc_auc_score
        train_ds = GapSequenceDataset(train_e)
        val_ds = GapSequenceDataset(val_e)
        model = create_model(pred.model_type, num_conf_classes=NUM_CONF_CLASSES).to(DEVICE)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        all_labels = np.stack(train_ds.labels)
        pos_rate = all_labels.mean(axis=0)
        pw = np.where(pos_rate > 0, (1.0 - pos_rate) / (pos_rate + 1e-8), 1.0)
        pw = np.clip(pw, 1.0, 50.0)
        pos_weight = torch.tensor(pw, dtype=torch.float32).to(DEVICE)

        def focal_bce(logits, targets):
            t = targets * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits, t, pos_weight=pos_weight, reduction="none")
            probs = torch.sigmoid(logits)
            p_t = probs * t + (1 - probs) * (1 - t)
            return ((1 - p_t) ** 2.0 * bce).mean()

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

        best_auc, best_state, no_imp = -1.0, None, 0
        for epoch in range(1, 51):
            model.train()
            for X, y in train_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                loss = focal_bce(model(X), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            model.eval()
            vp, vt = [], []
            with torch.no_grad():
                for X, y in val_loader:
                    vp.append(torch.sigmoid(model(X.to(DEVICE))).cpu().numpy())
                    vt.append(y.numpy())
            yp, yt = np.concatenate(vp), np.concatenate(vt)
            mask = yt.sum(axis=0) > 0
            try:
                val_auc = float(roc_auc_score(yt[:, mask], yp[:, mask], average="macro"))
            except ValueError:
                val_auc = 0.0
            scheduler.step(epoch)
            logger.info("  %s Epoch %d val_auc=%.4f", config_name, epoch, val_auc)
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= 5:
                    break
        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        pred.model = model
    else:
        logger.info("  Prediction DISABLED — using random predictions")
        # Create dummy model that outputs random
        model = create_model(pred.model_type, num_conf_classes=NUM_CONF_CLASSES).to(DEVICE)
        model.eval()
        pred.model = model  # untrained = random-ish

    # 5. Recommendation
    rec = RecommendationAgent(random_seed=seed)
    if hasattr(rec, "initialize"):
        try:
            rec.initialize(questions_df=questions_df, lectures_df=lectures_df,
                           interactions_df=train_df, train_user_ids=train_df["user_id"].unique().tolist())
        except Exception:
            pass
    if not disable_irt and diag.irt_params is not None:
        irt_diff = {str(q): float(b) for q, b in zip(diag.irt_params.question_ids, diag.irt_params.b)}
        rec.set_irt_params(irt_diff)

    # 6. Personalization
    pers = PersonalizationAgent()
    pers.train_clusters(extract_user_features(train_df))

    # Eval
    orch = Orchestrator(seed=seed)
    for a in [diag, conf, kg, pred, rec, pers]:
        orch.register_agent(a)
    metrics = orch.batch_evaluation(test_df, top_k=10, context_ratio=0.3)
    logger.info("  %s: AUC=%.4f NDCG=%.4f P@10=%.4f MRR=%.4f Cov=%.4f",
                config_name,
                metrics.get("lstm_auc", 0), metrics.get("ndcg@10", 0),
                metrics.get("precision@10", 0), metrics.get("mrr", 0),
                metrics.get("tag_coverage", 0))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_students", type=int, default=6000)
    parser.add_argument("--min_interactions", type=int, default=20)
    args = parser.parse_args()

    set_global_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "results" / "xes3g5m" / f"ablation_s{args.seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_xes3g5m(
        n_students=args.n_students, min_interactions=args.min_interactions, seed=args.seed)
    for df in [train_df, val_df, test_df]:
        df["confidence_class"] = 0

    # Configure NUM_TAGS dynamically from train data BEFORE any dataset/model build.
    from agents.prediction_agent import set_num_tags
    train_max_id = 0
    for tags in train_df["tags"]:
        if isinstance(tags, list) and tags:
            train_max_id = max(train_max_id, max(int(t) for t in tags))
    n_tags = train_max_id + 1
    logger.info("Concept-space: max_train_id=%d  ->  NUM_TAGS=%d", train_max_id, n_tags)
    set_num_tags(n_tags)

    questions_df = build_xes3g5m_questions_df("data/xes3g5m/XES3G5M")
    if "bundle_id" not in questions_df.columns:
        questions_df["bundle_id"] = questions_df["question_id"]
    if "correct_answer" not in questions_df.columns:
        questions_df["correct_answer"] = "A"
    if "deployed_at" not in questions_df.columns:
        questions_df["deployed_at"] = 0
    lectures_df = pd.DataFrame({"lecture_id": [], "tags": [], "part_id": [],
                                 "type_of": [], "bundle_id": []})

    # "Full MARS" is reused from the existing main-pipeline run for this seed
    # (run_xes3g5m_full.py output) to avoid retraining the full system here —
    # those metrics are identical given the same seed and same data split.
    # Only the agent-disabled configurations are trained from scratch.
    configs = [
        ("- Prediction", {"disable_prediction": True}),
        ("- Knowledge Graph", {"disable_kg": True}),
        ("- Confidence", {"disable_confidence": True}),
        ("- IRT (Diagnostic)", {"disable_irt": True}),
    ]

    results = {}

    # Load Full MARS metrics from the most recent main-pipeline run for this seed.
    import glob
    import os
    full_paths = sorted(
        glob.glob(str(ROOT / f"results/xes3g5m/xes3g5m_full_s{args.seed}_*/metrics.json")),
        key=os.path.getmtime, reverse=True,
    )
    if full_paths:
        with open(full_paths[0]) as f:
            full_run = json.load(f)
        full_eval = full_run.get("eval_metrics", {})
        # Tag the row with where it came from so the lineage is traceable.
        full_eval["_reused_from"] = str(Path(full_paths[0]).parent.name)
        full_eval["time_s"] = 0.0  # no time spent re-training in this script
        results["Full MARS"] = full_eval
        logger.info("Reusing Full MARS metrics from %s", full_paths[0])
    else:
        logger.warning(
            "No main-pipeline run found for seed=%d at "
            "results/xes3g5m/xes3g5m_full_s%d_* — Full MARS row will be missing. "
            "Run scripts/run_xes3g5m_full.py --seed %d first.",
            args.seed, args.seed, args.seed,
        )

    for name, kwargs in configs:
        t0 = time.time()
        metrics = run_ablation_config(
            name, train_df, val_df, test_df, questions_df, lectures_df,
            args.seed, run_dir, **kwargs)
        metrics["time_s"] = round(time.time() - t0, 1)
        results[name] = metrics

    # Summary
    logger.info("=" * 70)
    logger.info("ABLATION SUMMARY (XES3G5M, seed=%d)", args.seed)
    logger.info("=" * 70)
    logger.info("%-20s %8s %8s %8s %8s %8s", "Config", "AUC", "NDCG@10", "P@10", "MRR", "Cov")
    for name, m in results.items():
        logger.info("%-20s %8.4f %8.4f %8.4f %8.4f %8.4f",
                     name, m.get("lstm_auc", 0), m.get("ndcg@10", 0),
                     m.get("precision@10", 0), m.get("mrr", 0), m.get("tag_coverage", 0))

    with open(run_dir / "ablation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved to %s", run_dir / "ablation.json")


if __name__ == "__main__":
    main()
