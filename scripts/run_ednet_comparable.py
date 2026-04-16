"""
Full MARS pipeline on EdNet with SAME sampling as XES3G5M for fair comparison.

6000 students, min 20 interactions, user-level split 70/15/15.
All 6 agents + Orchestrator batch_evaluation.
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
from data.ednet_comparable_loader import load_ednet_comparable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("ednet_comparable")


def train_all_agents(train_df, val_df, seed, questions_df, lectures_df, run_dir):
    from agents.diagnostic_agent import DiagnosticAgent
    from agents.confidence_agent import ConfidenceAgent
    from agents.kg_agent import KnowledgeGraphAgent
    from agents.prediction_agent import (
        PredictionAgent, GapSequenceDataset, create_model,
        DEVICE, NUM_CONF_CLASSES, NUM_WORKERS, LABEL_SMOOTHING,
    )
    from agents.recommendation_agent import RecommendationAgent
    from agents.personalization_agent import PersonalizationAgent, extract_user_features
    import torch.nn as nn

    set_global_seed(seed)
    agent_metrics = {}

    # 1. Diagnostic
    logger.info("[seed=%d] DiagnosticAgent...", seed); t0 = time.time()
    diag = DiagnosticAgent(seed=seed)
    diag.calibrate_from_interactions(train_df, min_answers_per_q=20)
    agent_metrics["diagnostic"] = {"time_s": round(time.time()-t0, 1)}

    # 2. Confidence
    logger.info("[seed=%d] ConfidenceAgent...", seed); t0 = time.time()
    conf = ConfidenceAgent()
    conf_m = conf.train(train_df, irt_params=diag.irt_params)
    agent_metrics["confidence"] = {**conf_m, "time_s": round(time.time()-t0, 1)}

    # 3. KG
    logger.info("[seed=%d] KG...", seed); t0 = time.time()
    kg = KnowledgeGraphAgent()
    kg.build_graph(questions_df, lectures_df)
    kg.build_prerequisites(train_df, train_user_ids=set(train_df["user_id"].unique()))
    agent_metrics["knowledge_graph"] = {
        "n_nodes": kg.graph.number_of_nodes(),
        "n_edges": kg.graph.number_of_edges(),
        "time_s": round(time.time()-t0, 1),
    }

    # 4. Prediction (custom training loop — per-epoch logging, patience=5)
    train_e = train_df.copy(); val_e = val_df.copy()
    train_e["confidence_class"] = conf.classify_batch(interactions=train_e)["classes"]
    val_e["confidence_class"] = conf.classify_batch(interactions=val_e)["classes"]

    logger.info("[seed=%d] PredictionAgent (custom loop)...", seed); t0 = time.time()
    pred = PredictionAgent()
    train_ds = GapSequenceDataset(train_e)
    val_ds = GapSequenceDataset(val_e)
    logger.info("  Train: %d seqs, Val: %d seqs", len(train_ds), len(val_ds))
    model = create_model(pred.model_type, num_conf_classes=NUM_CONF_CLASSES).to(DEVICE)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    all_labels = np.stack(train_ds.labels)
    pos_rate = all_labels.mean(axis=0)
    pw = np.where(pos_rate > 0, (1.0-pos_rate)/(pos_rate+1e-8), 1.0)
    pw = np.clip(pw, 1.0, 50.0)
    pos_weight = torch.tensor(pw, dtype=torch.float32).to(DEVICE)

    def focal_bce(logits, targets):
        t = targets*(1-LABEL_SMOOTHING) + 0.5*LABEL_SMOOTHING
        bce = nn.functional.binary_cross_entropy_with_logits(logits, t, pos_weight=pos_weight, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs*t + (1-probs)*(1-t)
        return ((1-p_t)**2.0 * bce).mean()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    from sklearn.metrics import roc_auc_score
    best_val_auc, best_val_loss, best_epoch = -1.0, float("inf"), 0
    best_state = None
    no_imp = 0
    patience = 5
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
        yp, yt = np.concatenate(vp), np.concatenate(vt)
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

        logger.info("  Epoch %d/50  train=%.4f  val=%.4f  val_auc=%.4f  lr=%.2e",
                    epoch, avg_train, avg_val, val_auc, lr)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_loss = avg_val
            best_epoch = epoch
            no_imp = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            ckpt = ckpts_dir / f"epoch{epoch:03d}_auc{val_auc:.4f}.pt"
            torch.save(best_state, ckpt)
            torch.save(best_state, run_dir / "best.pt")
            logger.info("    -> BEST checkpoint: %s", ckpt.name)
        else:
            no_imp += 1
            if no_imp >= patience:
                logger.info("  Early stop at epoch %d (best=%d, auc=%.4f)",
                            epoch, best_epoch, best_val_auc)
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    pred.model = model

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f)

    agent_metrics["prediction"] = {
        "best_epoch": best_epoch,
        "val_auc": round(best_val_auc, 4),
        "val_loss": round(best_val_loss, 4),
        "total_epochs": len(history["train_loss"]),
        "time_s": round(time.time()-t0, 1),
    }

    # 5. Recommendation
    logger.info("[seed=%d] RecommendationAgent...", seed); t0 = time.time()
    rec = RecommendationAgent(random_seed=seed)
    if hasattr(rec, "initialize"):
        try:
            rec.initialize(questions_df=questions_df, lectures_df=lectures_df,
                           interactions_df=train_df, train_user_ids=train_df["user_id"].unique().tolist())
        except Exception:
            pass
    if diag.irt_params is not None:
        irt_diff = {str(q): float(b) for q, b in zip(diag.irt_params.question_ids, diag.irt_params.b)}
        rec.set_irt_params(irt_diff)
    agent_metrics["recommendation"] = {"time_s": round(time.time()-t0, 1)}

    # 6. Personalization
    pers = PersonalizationAgent()
    pers.train_clusters(extract_user_features(train_df))
    agent_metrics["personalization"] = {}

    agents = {"diagnostic":diag, "confidence":conf, "knowledge_graph":kg,
              "prediction":pred, "recommendation":rec, "personalization":pers}
    return agents, agent_metrics


def evaluate(agents, test_df, seed):
    from agents.orchestrator import Orchestrator
    set_global_seed(seed)
    orch = Orchestrator(seed=seed)
    for a in agents.values():
        orch.register_agent(a)
    return orch.batch_evaluation(test_df, top_k=10, context_ratio=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_students", type=int, default=6000)
    parser.add_argument("--min_interactions", type=int, default=20)
    args = parser.parse_args()

    set_global_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ednet_comparable_s{args.seed}_n{args.n_students}_min{args.min_interactions}_{timestamp}"
    run_dir = ROOT / "results" / "ednet_comparable" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("MARS on EdNet (comparable to XES3G5M)")
    logger.info("  Run: %s", run_name)
    logger.info("="*70)

    t0 = time.time()
    train_df, val_df, test_df = load_ednet_comparable(
        n_students=args.n_students, min_interactions=args.min_interactions, seed=args.seed)
    logger.info("Data loaded in %.1fs", time.time() - t0)

    from data.loader import EdNetLoader
    loader = EdNetLoader(data_dir="data/raw")
    questions_df = loader.questions
    lectures_df = loader.lectures

    agents, agent_metrics = train_all_agents(
        train_df, val_df, args.seed, questions_df, lectures_df, run_dir)

    logger.info("Running Orchestrator batch_evaluation...")
    t0 = time.time()
    eval_metrics = evaluate(agents, test_df, args.seed)
    eval_time = time.time() - t0

    logger.info("="*70)
    logger.info("FINAL RESULTS — %s", run_name)
    logger.info("="*70)
    for k, v in sorted(eval_metrics.items()):
        logger.info("  %-25s: %s", k, v)

    combined = {
        "run_name": run_name, "dataset": "EdNet",
        "seed": args.seed, "n_students": args.n_students,
        "min_interactions": args.min_interactions,
        "agent_metrics": agent_metrics,
        "eval_metrics": eval_metrics,
        "eval_time_s": round(eval_time, 1),
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(combined, f, indent=2, default=str)
    logger.info("Saved to %s", run_dir / "metrics.json")


if __name__ == "__main__":
    main()
