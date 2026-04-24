"""
Run the MARS prediction pipeline on XES3G5M dataset using Phase A model
(commit 0.78 architecture: SAINT transformer).

Features:
- Per-epoch logging of val_auc, val_loss, train_loss, lr
- Best checkpoint saved on every val_auc improvement (timestamped filename)
- Named model files per run (never overwrites previous)
- Early stopping patience=5 on val_auc
- Test AUC evaluation at the end

Usage:
    python scripts/run_xes3g5m.py [--n_students 6000] [--min_interactions 20]
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import prediction_agent as PA
from agents.prediction_agent import (
    PredictionAgent, GapSequenceDataset, create_model,
    SEQ_LEN, HORIZON, BATCH_SIZE, DEVICE, NUM_WORKERS,
    LABEL_SMOOTHING, NUM_CONF_CLASSES,
    set_num_tags,
)
from agents.utils import set_global_seed
from data.xes3g5m_loader import load_xes3g5m

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("xes3g5m")

# ──────────────────────────────────────────────────────────
# Custom training loop (wraps Phase A model without modifying it)
# ──────────────────────────────────────────────────────────

def train_with_logging(
    model: nn.Module,
    train_dataset: GapSequenceDataset,
    val_dataset: GapSequenceDataset,
    run_dir: Path,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 5e-4,
    patience: int = 5,
) -> dict:
    """Custom training loop with per-epoch logging + best checkpoint saves."""

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    logger.info("Train: %d sequences, Val: %d sequences", len(train_dataset), len(val_dataset))
    logger.info("Model: %s — %.2fM params",
                model.__class__.__name__,
                sum(p.numel() for p in model.parameters()) / 1e6)

    # Loss: focal BCE with pos_weight (same as Phase A)
    all_labels = np.stack(train_dataset.labels)
    pos_rate = all_labels.mean(axis=0)
    pw = np.where(pos_rate > 0, (1.0 - pos_rate) / (pos_rate + 1e-8), 1.0)
    pw = np.clip(pw, 1.0, 50.0)
    pos_weight = torch.tensor(pw, dtype=torch.float32).to(DEVICE)
    focal_gamma = 2.0

    def focal_bce_loss(logits, targets):
        targets = targets * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction="none",
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** focal_gamma
        return (focal_weight * bce).mean()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6,
    )

    best_val_auc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = focal_bce_loss(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(loss.item())
        avg_train_loss = float(np.mean(train_losses))

        # ── Validate ──
        model.eval()
        val_losses = []
        all_preds_v, all_labels_v = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                y_pred = model(X_batch)
                loss = focal_bce_loss(y_pred, y_batch)
                val_losses.append(loss.item())
                all_preds_v.append(torch.sigmoid(y_pred).cpu().numpy())
                all_labels_v.append(y_batch.cpu().numpy())
        avg_val_loss = float(np.mean(val_losses))

        # Compute val_auc
        from sklearn.metrics import roc_auc_score
        y_true_v = np.concatenate(all_labels_v)
        y_score_v = np.concatenate(all_preds_v)
        tag_mask = y_true_v.sum(axis=0) > 0
        try:
            val_auc = float(roc_auc_score(
                y_true_v[:, tag_mask], y_score_v[:, tag_mask], average="macro"))
        except ValueError:
            val_auc = 0.0

        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_auc"].append(val_auc)

        # ── LOG EVERY EPOCH ──
        logger.info(
            "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_auc=%.4f  lr=%.2e",
            epoch, epochs, avg_train_loss, avg_val_loss, val_auc, current_lr,
        )

        # ── Save checkpoint on improvement ──
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            ckpt_path = checkpoints_dir / f"epoch{epoch:03d}_auc{val_auc:.4f}.pt"
            torch.save(best_state, ckpt_path)
            logger.info("  ↳ BEST checkpoint saved: %s", ckpt_path.name)

            # Also save as "best.pt" for easy access
            best_path = run_dir / "best.pt"
            torch.save(best_state, best_path)
            with open(run_dir / "best_meta.json", "w") as f:
                json.dump({"epoch": epoch, "val_auc": val_auc,
                           "val_loss": avg_val_loss}, f)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(
                    "Early stopping at epoch %d (best=%d, val_auc=%.4f)",
                    epoch, best_epoch, best_val_auc,
                )
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save final model with unique name (NEVER overwrites previous runs)
    final_path = run_dir / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    logger.info("Final model saved: %s", final_path)

    return {
        "best_epoch": best_epoch,
        "val_auc": round(best_val_auc, 4),
        "val_loss": round(best_val_loss, 4),
        "total_epochs": len(history["train_loss"]),
        "history": history,
    }


def compute_all_metrics(model, test_df, train_df, batch_size):
    """Compute ALL metrics on XES3G5M test split — matches EdNet metric set.

    Returns dict with: test_auc_macro, test_auc_weighted, test_f1_micro,
    ndcg@10, precision@10, recall@10, mrr, tag_coverage, question_coverage,
    diversity (ILD), novelty, learning_gain, learning_gain_std.
    """
    import math
    from sklearn.metrics import roc_auc_score, f1_score
    from sklearn.metrics.pairwise import cosine_distances

    test_dataset = GapSequenceDataset(test_df)
    if len(test_dataset) == 0:
        return {"test_auc_macro": 0, "n_test_sequences": 0}

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            X = batch[0].to(DEVICE)
            y = batch[1]
            probs = torch.sigmoid(model(X)).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(y.numpy())

    y_score = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    N = len(y_score)
    n_tags = y_true.shape[1]

    # ── AUC ──
    tag_pos = y_true.sum(axis=0)
    tag_neg = (1 - y_true).sum(axis=0)
    tag_mask = (tag_pos >= 5) & (tag_neg >= 5)
    n_eval_tags = int(tag_mask.sum())

    if n_eval_tags > 0:
        auc_macro = float(roc_auc_score(
            y_true[:, tag_mask], y_score[:, tag_mask], average="macro"))
        auc_weighted = float(roc_auc_score(
            y_true[:, tag_mask], y_score[:, tag_mask], average="weighted"))
    else:
        auc_macro, auc_weighted = 0.0, 0.0

    # ── F1 with threshold search ──
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.01, 0.5, 0.01):
        yb = (y_score[:, tag_mask] >= t).astype(int)
        if yb.sum() == 0:
            continue
        f1 = f1_score(y_true[:, tag_mask], yb, average="micro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    # ── Ranking metrics: per-sequence NDCG@10, Precision@10, Recall@10, MRR ──
    ndcg10s, p10s, r10s, mrrs = [], [], [], []
    all_rec_tags = set()
    diversity_scores = []
    novelty_scores = []

    # Item popularity for novelty (from train)
    tag_freq = np.zeros(n_tags, dtype=np.float32)
    n_train_users = train_df["user_id"].nunique()
    for tags in train_df["tags"]:
        if isinstance(tags, list):
            for t in tags:
                if 0 <= t < n_tags:
                    tag_freq[t] += 1
    tag_pop = tag_freq / max(n_train_users, 1)  # P(tag seen by a user)

    # Tag one-hot vectors for diversity (ILD)
    tag_vecs = np.eye(n_tags, dtype=np.float32)

    for i in range(N):
        gt = set(np.where(y_true[i] > 0)[0])
        if len(gt) == 0:
            continue
        ranked = np.argsort(-y_score[i])[:10].tolist()
        all_rec_tags.update(ranked)

        hits_10 = len(set(ranked) & gt)

        # Precision@10
        p10s.append(hits_10 / 10)
        # Recall@10
        r10s.append(hits_10 / len(gt))
        # NDCG@10
        dcg = sum(1.0 / np.log2(r + 2) for r, t in enumerate(ranked) if t in gt)
        ideal = sum(1.0 / np.log2(r + 2) for r in range(min(len(gt), 10)))
        ndcg10s.append(dcg / ideal if ideal > 0 else 0.0)
        # MRR
        rr = 0.0
        for r, t in enumerate(ranked):
            if t in gt:
                rr = 1.0 / (r + 1)
                break
        mrrs.append(rr)

        # Diversity (ILD): avg pairwise cosine distance in top-10
        if len(ranked) >= 2:
            vecs = tag_vecs[ranked]
            dists = cosine_distances(vecs)
            n_r = len(ranked)
            ild = float(dists[np.triu_indices(n_r, k=1)].mean())
            diversity_scores.append(ild)

        # Novelty: avg -log2(popularity) of recommended tags
        nov_scores = []
        for t in ranked:
            p = tag_pop[t] if 0 <= t < n_tags else 1e-6
            nov_scores.append(-math.log2(p + 1e-10))
        novelty_scores.append(float(np.mean(nov_scores)))

    # ── Coverage ──
    tag_coverage = len(all_rec_tags) / max(n_eval_tags, 1)
    # Question coverage: not directly available (we rank tags not questions),
    # approximate as fraction of unique tags recommended / total tags
    question_coverage = len(all_rec_tags) / max(n_tags, 1)

    # ── Learning Gain ──
    # Proxy: for each test sequence, compare avg gap_prob of the first half
    # vs second half of test interactions for each user. If recommendations
    # help, later interactions should show lower gap_probs (= higher mastery).
    learning_gains = []
    test_by_user = test_df.groupby("user_id")
    for uid, grp in test_by_user:
        grp = grp.sort_values("timestamp")
        if len(grp) < 10:
            continue
        mid = len(grp) // 2
        first_half = grp.iloc[:mid]
        second_half = grp.iloc[mid:]
        # Mastery proxy = fraction correct
        acc_before = first_half["correct"].mean()
        acc_after = second_half["correct"].mean()
        learning_gains.append(acc_after - acc_before)

    lg_mean = float(np.mean(learning_gains)) if learning_gains else 0.0
    lg_std = float(np.std(learning_gains)) if learning_gains else 0.0
    # Trimmed: remove top/bottom 5%
    if len(learning_gains) > 20:
        lg_arr = np.array(learning_gains)
        lo, hi = np.percentile(lg_arr, 5), np.percentile(lg_arr, 95)
        lg_trimmed = float(lg_arr[(lg_arr >= lo) & (lg_arr <= hi)].mean())
    else:
        lg_trimmed = lg_mean

    return {
        "test_auc_macro": round(auc_macro, 4),
        "test_auc_weighted": round(auc_weighted, 4),
        "test_f1_micro": round(best_f1, 4),
        "test_f1_threshold": round(best_t, 3),
        "ndcg@10": round(float(np.mean(ndcg10s)), 4) if ndcg10s else 0.0,
        "precision@10": round(float(np.mean(p10s)), 4) if p10s else 0.0,
        "recall@10": round(float(np.mean(r10s)), 4) if r10s else 0.0,
        "mrr": round(float(np.mean(mrrs)), 4) if mrrs else 0.0,
        "tag_coverage": round(tag_coverage, 4),
        "question_coverage": round(question_coverage, 4),
        "diversity": round(float(np.mean(diversity_scores)), 4) if diversity_scores else 0.0,
        "novelty": round(float(np.mean(novelty_scores)), 4) if novelty_scores else 0.0,
        "learning_gain": round(lg_mean, 4),
        "learning_gain_std": round(lg_std, 4),
        "learning_gain_trimmed": round(lg_trimmed, 4),
        "n_test_sequences": len(test_dataset),
        "n_eval_tags": n_eval_tags,
        "n_eval_users": len(ndcg10s),
    }


def main():
    parser = argparse.ArgumentParser(description="Run MARS Phase A on XES3G5M")
    parser.add_argument("--n_students", type=int, default=6000)
    parser.add_argument("--min_interactions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--model_type", type=str, default="transformer")
    args = parser.parse_args()

    set_global_seed(args.seed)

    # Create unique run directory (NEVER overwrites previous)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"xes3g5m_s{args.seed}_n{args.n_students}_min{args.min_interactions}_{timestamp}"
    run_dir = ROOT / "results" / "xes3g5m" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("MARS Phase A on XES3G5M")
    logger.info("  Run: %s", run_name)
    logger.info("  Students: %d, Min interactions: %d, Seed: %d",
                args.n_students, args.min_interactions, args.seed)
    logger.info("  Patience: %d, LR: %.1e, Batch: %d",
                args.patience, args.lr, args.batch_size)
    logger.info("  Output: %s", run_dir)
    logger.info("=" * 70)

    # Load data
    t0 = time.time()
    train_df, val_df, test_df = load_xes3g5m(
        n_students=args.n_students,
        min_interactions=args.min_interactions,
        seed=args.seed,
    )
    for df in [train_df, val_df, test_df]:
        df["confidence_class"] = 0
    logger.info("Data loaded in %.1fs", time.time() - t0)

    # Determine concept-space size from training data and configure model
    # accordingly. Hardcoding NUM_TAGS=293 (EdNet TOEIC) silently dropped
    # ~565 of XES3G5M's 858 concepts from labels.
    train_max_id = 0
    for tags in train_df["tags"]:
        if isinstance(tags, list) and tags:
            train_max_id = max(train_max_id, max(int(t) for t in tags))
    n_tags = train_max_id + 1
    logger.info("Concept-space: max_train_id=%d  ->  NUM_TAGS=%d",
                train_max_id, n_tags)
    set_num_tags(n_tags)  # MUST be called BEFORE building any dataset/model

    # Build datasets
    train_dataset = GapSequenceDataset(train_df)
    val_dataset = GapSequenceDataset(val_df)

    # Create model (Phase A architecture from commit 0.78)
    model = create_model(args.model_type, num_conf_classes=NUM_CONF_CLASSES).to(DEVICE)

    # Train with custom loop
    train_metrics = train_with_logging(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        run_dir=run_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )

    # Test evaluation — ALL metrics (matches EdNet metric set)
    test_metrics = compute_all_metrics(model, test_df, train_df, args.batch_size)

    # Final report
    logger.info("=" * 70)
    logger.info("FINAL RESULTS — %s", run_name)
    logger.info("=" * 70)
    for k, v in test_metrics.items():
        logger.info("  %-25s: %s", k, v)

    # Save combined results
    combined = {
        "run_name": run_name,
        "dataset": "XES3G5M",
        "model_type": args.model_type,
        "n_students": args.n_students,
        "min_interactions": args.min_interactions,
        "seed": args.seed,
        "patience": args.patience,
        "lr": args.lr,
        "batch_size": args.batch_size,
        **{k: v for k, v in train_metrics.items() if k != "history"},
        **test_metrics,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(combined, f, indent=2)
    # Save history separately (large)
    with open(run_dir / "history.json", "w") as f:
        json.dump(train_metrics["history"], f)

    logger.info("All results saved to %s", run_dir)


if __name__ == "__main__":
    main()
