"""
Full MARS multi-agent pipeline on XES3G5M.

Runs all 6 agents + Orchestrator batch_evaluation, same as EdNet pipeline.
Adapts each agent to XES3G5M's math domain:
  - DiagnosticAgent: IRT calibration on math questions
  - ConfidenceAgent: rule-based (works as-is)
  - KG Agent: build concept graph from kc_routes in questions.json
  - PredictionAgent: SAINT transformer (Phase A)
  - RecommendationAgent: Thompson Sampling on concepts
  - PersonalizationAgent: cluster users by accuracy/activity

Usage:
    python scripts/run_xes3g5m_full.py [--seed 42] [--n_students 6000]
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

from agents.utils import set_global_seed, load_config
from data.xes3g5m_loader import load_xes3g5m

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("xes3g5m_full")


def build_xes3g5m_questions_df(data_dir: str) -> pd.DataFrame:
    """Build a questions DataFrame compatible with EdNet format from XES3G5M metadata."""
    meta_path = Path(data_dir) / "metadata" / "questions.json"
    kc_map_path = Path(data_dir) / "metadata" / "kc_routes_map.json"

    questions = json.load(open(meta_path, encoding="utf-8"))
    kc_map = json.load(open(kc_map_path, encoding="utf-8"))

    rows = []
    for qid_str, qdata in questions.items():
        qid = int(qid_str)
        kc_routes = qdata.get("kc_routes", [])
        # Extract leaf concept IDs: last element of each route path
        tags = []
        for route in kc_routes:
            if isinstance(route, str):
                # Find concept ID by matching leaf name in kc_map
                leaf = route.split("----")[-1] if "----" in route else route
                for cid, cname in kc_map.items():
                    if cname == leaf:
                        tags.append(int(cid))
                        break
        if not tags:
            tags = [0]

        rows.append({
            "question_id": f"q{qid}",
            "tags": tags,
            "part_id": 1,  # single domain (math)
            "type": qdata.get("type", "unknown"),
        })

    df = pd.DataFrame(rows)
    logger.info("XES3G5M questions: %d questions, %d unique tags",
                len(df), len(set(t for tags in df["tags"] for t in tags)))
    return df


def build_xes3g5m_lectures_df() -> pd.DataFrame:
    """XES3G5M has no lectures — return empty DataFrame with EdNet schema."""
    return pd.DataFrame(columns=["lecture_id", "tags", "part_id"])


def train_all_agents(train_df, val_df, seed, questions_df, lectures_df, run_dir=None):
    """Train all 6 MARS agents on XES3G5M data."""
    from agents.diagnostic_agent import DiagnosticAgent
    from agents.confidence_agent import ConfidenceAgent
    from agents.kg_agent import KnowledgeGraphAgent
    from agents.prediction_agent import PredictionAgent
    from agents.recommendation_agent import RecommendationAgent
    from agents.personalization_agent import PersonalizationAgent, extract_user_features

    set_global_seed(seed)
    agent_metrics = {}

    # 1. DiagnosticAgent — IRT calibration
    logger.info("[seed=%d] Training DiagnosticAgent (IRT)...", seed)
    t0 = time.time()
    diag = DiagnosticAgent(seed=seed)
    irt_params = diag.calibrate_from_interactions(train_df, min_answers_per_q=5)
    agent_metrics["diagnostic"] = {
        "n_items": len(irt_params) if irt_params is not None else 0,
        "time_s": round(time.time() - t0, 1),
    }
    logger.info("  IRT: %d items calibrated in %.1fs",
                agent_metrics["diagnostic"]["n_items"],
                agent_metrics["diagnostic"]["time_s"])

    # 2. ConfidenceAgent — rule-based
    logger.info("[seed=%d] Training ConfidenceAgent...", seed)
    t0 = time.time()
    conf = ConfidenceAgent()
    conf_metrics = conf.train(train_df, irt_params=irt_params)
    agent_metrics["confidence"] = {
        **conf_metrics,
        "time_s": round(time.time() - t0, 1),
    }
    logger.info("  Confidence F1=%.4f in %.1fs",
                conf_metrics.get("full_f1_macro", 0),
                agent_metrics["confidence"]["time_s"])

    # 3. KG Agent — build graph from questions metadata
    logger.info("[seed=%d] Building KG...", seed)
    t0 = time.time()
    kg = KnowledgeGraphAgent()
    kg.build_graph(questions_df, lectures_df)
    kg.build_prerequisites(train_df, train_user_ids=set(train_df["user_id"].unique()))
    agent_metrics["knowledge_graph"] = {
        "n_nodes": kg.graph.number_of_nodes(),
        "n_edges": kg.graph.number_of_edges(),
        "time_s": round(time.time() - t0, 1),
    }
    logger.info("  KG: %d nodes, %d edges in %.1fs",
                agent_metrics["knowledge_graph"]["n_nodes"],
                agent_metrics["knowledge_graph"]["n_edges"],
                agent_metrics["knowledge_graph"]["time_s"])

    # 4. PredictionAgent — add confidence_class first
    if "confidence_class" not in train_df.columns:
        train_df = train_df.copy()
        train_df["confidence_class"] = conf.classify_batch(interactions=train_df)["classes"]
    if val_df is not None and "confidence_class" not in val_df.columns:
        val_df = val_df.copy()
        val_df["confidence_class"] = conf.classify_batch(interactions=val_df)["classes"]

    logger.info("[seed=%d] Training PredictionAgent (custom loop)...", seed)
    t0 = time.time()
    pred = PredictionAgent()

    # Custom training loop inline — same architecture as Phase A but with:
    #   - per-epoch val_auc logging
    #   - best checkpoint saved on every improvement (named, never overwrites)
    #   - early stopping patience kept as default (8)
    from agents.prediction_agent import (
        GapSequenceDataset, create_model, DEVICE, NUM_CONF_CLASSES,
        NUM_WORKERS, LABEL_SMOOTHING,
    )
    import torch.nn as nn

    train_dataset = GapSequenceDataset(train_df)
    val_dataset = GapSequenceDataset(val_df)
    logger.info("  Train: %d sequences, Val: %d sequences",
                len(train_dataset), len(val_dataset))

    model = create_model(pred.model_type, num_conf_classes=NUM_CONF_CLASSES).to(DEVICE)
    logger.info("  Model: %s — %.2fM params",
                model.__class__.__name__,
                sum(p.numel() for p in model.parameters()) / 1e6)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # Loss (same as Phase A: focal BCE + pos_weight)
    all_labels = np.stack(train_dataset.labels)
    pos_rate = all_labels.mean(axis=0)
    pw = np.where(pos_rate > 0, (1.0 - pos_rate) / (pos_rate + 1e-8), 1.0)
    pw = np.clip(pw, 1.0, 50.0)
    pos_weight = torch.tensor(pw, dtype=torch.float32).to(DEVICE)
    focal_gamma = 2.0

    def focal_bce(logits, targets):
        t = targets * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, t, pos_weight=pos_weight, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * t + (1 - probs) * (1 - t)
        return ((1 - p_t) ** focal_gamma * bce).mean()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    from sklearn.metrics import roc_auc_score
    best_val_auc, best_val_loss, best_epoch = -1.0, float("inf"), 0
    best_state = None
    epochs_no_improve = 0
    patience = 5
    if run_dir is None:
        run_dir = Path("results/xes3g5m/tmp_run")
    ckpts_dir = run_dir / "checkpoints"
    ckpts_dir.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    for epoch in range(1, 51):
        model.train()
        losses = []
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = focal_bce(model(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())
        avg_train = float(np.mean(losses))

        model.eval()
        vl, vp, vt = [], [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                out = model(X)
                vl.append(focal_bce(out, y).item())
                vp.append(torch.sigmoid(out).cpu().numpy())
                vt.append(y.cpu().numpy())
        avg_val = float(np.mean(vl))
        yp = np.concatenate(vp); yt = np.concatenate(vt)
        mask = yt.sum(axis=0) > 0
        try:
            val_auc = float(roc_auc_score(yt[:, mask], yp[:, mask], average="macro"))
        except ValueError:
            val_auc = 0.0

        scheduler.step(epoch)
        lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_auc"].append(val_auc)

        # Log EVERY epoch
        logger.info("  Epoch %d/50  train=%.4f  val=%.4f  val_auc=%.4f  lr=%.2e",
                     epoch, avg_train, avg_val, val_auc, lr)

        # Save checkpoint on improvement (never overwrites)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_loss = avg_val
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            ckpt = ckpts_dir / f"epoch{epoch:03d}_auc{val_auc:.4f}.pt"
            torch.save(best_state, ckpt)
            torch.save(best_state, run_dir / "best.pt")
            logger.info("    ↳ BEST checkpoint: %s", ckpt.name)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("  Early stopping at epoch %d (best=%d, auc=%.4f)",
                             epoch, best_epoch, best_val_auc)
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    pred.model = model

    # Save history
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f)

    pred_results = {
        "best_epoch": best_epoch,
        "val_auc": round(best_val_auc, 4),
        "val_loss": round(best_val_loss, 4),
        "total_epochs": len(history["train_loss"]),
    }
    agent_metrics["prediction"] = {
        **pred_results,
        "time_s": round(time.time() - t0, 1),
    }
    logger.info("  Prediction: val_auc=%.4f, best_epoch=%d in %.1fs",
                pred_results.get("val_auc", 0),
                pred_results.get("best_epoch", 0),
                agent_metrics["prediction"]["time_s"])

    # 5. RecommendationAgent
    logger.info("[seed=%d] Initializing RecommendationAgent...", seed)
    t0 = time.time()
    rec = RecommendationAgent(random_seed=seed)
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
    # Pass IRT difficulty
    if diag.irt_params is not None:
        irt_difficulty = {
            str(qid): float(b)
            for qid, b in zip(diag.irt_params.question_ids, diag.irt_params.b)
        }
        rec.set_irt_params(irt_difficulty)
    agent_metrics["recommendation"] = {
        "time_s": round(time.time() - t0, 1),
    }

    # 6. PersonalizationAgent
    logger.info("[seed=%d] Training PersonalizationAgent...", seed)
    t0 = time.time()
    pers = PersonalizationAgent()
    user_feats = extract_user_features(train_df)
    n_levels = pers.train_clusters(user_feats)
    agent_metrics["personalization"] = {
        "n_levels": n_levels,
        "time_s": round(time.time() - t0, 1),
    }

    agents = {
        "diagnostic": diag,
        "confidence": conf,
        "knowledge_graph": kg,
        "prediction": pred,
        "recommendation": rec,
        "personalization": pers,
    }
    return agents, agent_metrics


def evaluate_pipeline(agents, test_df, seed):
    """Run Orchestrator batch_evaluation — full multi-agent eval."""
    from agents.orchestrator import Orchestrator

    set_global_seed(seed)
    orch = Orchestrator(seed=seed)
    for agent in agents.values():
        orch.register_agent(agent)

    metrics = orch.batch_evaluation(test_df, top_k=10, context_ratio=0.3)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_students", type=int, default=6000)
    parser.add_argument("--min_interactions", type=int, default=20)
    args = parser.parse_args()

    set_global_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"xes3g5m_full_s{args.seed}_{timestamp}"
    run_dir = ROOT / "results" / "xes3g5m" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("MARS FULL PIPELINE on XES3G5M")
    logger.info("  Run: %s", run_name)
    logger.info("  Students: %d, Min interactions: %d, Seed: %d",
                args.n_students, args.min_interactions, args.seed)
    logger.info("=" * 70)

    # Load data
    t0_total = time.time()
    train_df, val_df, test_df = load_xes3g5m(
        n_students=args.n_students,
        min_interactions=args.min_interactions,
        seed=args.seed,
    )
    logger.info("Data loaded: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    # Build questions metadata — add EdNet-compatible dummy columns
    # that KG agent's build_graph expects but XES3G5M doesn't have.
    questions_df = build_xes3g5m_questions_df("data/xes3g5m/XES3G5M")
    if "bundle_id" not in questions_df.columns:
        questions_df["bundle_id"] = questions_df["question_id"]
    if "correct_answer" not in questions_df.columns:
        questions_df["correct_answer"] = "A"
    if "deployed_at" not in questions_df.columns:
        questions_df["deployed_at"] = 0
    lectures_df = build_xes3g5m_lectures_df()
    if "lecture_id" not in lectures_df.columns:
        lectures_df = pd.DataFrame({
            "lecture_id": [], "tags": [], "part_id": [],
            "type_of": [], "bundle_id": [],
        })

    # Train all agents
    agents, agent_metrics = train_all_agents(
        train_df, val_df, args.seed, questions_df, lectures_df, run_dir=run_dir
    )

    # Evaluate via Orchestrator
    logger.info("Running Orchestrator batch_evaluation...")
    t0 = time.time()
    eval_metrics = evaluate_pipeline(agents, test_df, args.seed)
    eval_time = time.time() - t0
    total_time = time.time() - t0_total

    # Report
    logger.info("=" * 70)
    logger.info("FULL PIPELINE RESULTS — %s", run_name)
    logger.info("=" * 70)
    for k, v in sorted(eval_metrics.items()):
        logger.info("  %-25s: %s", k, v)
    logger.info("  eval_time:               %.1fs", eval_time)
    logger.info("  total_time:              %.1fs", total_time)

    # Save
    combined = {
        "run_name": run_name,
        "dataset": "XES3G5M",
        "pipeline": "full_multi_agent",
        "seed": args.seed,
        "n_students": args.n_students,
        "min_interactions": args.min_interactions,
        "agent_metrics": agent_metrics,
        "eval_metrics": eval_metrics,
        "eval_time_s": round(eval_time, 1),
        "total_time_s": round(total_time, 1),
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(combined, f, indent=2, default=str)
    logger.info("Saved to %s", run_dir / "metrics.json")


if __name__ == "__main__":
    main()
