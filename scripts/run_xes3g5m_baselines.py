"""
Baselines for XES3G5M: Random, Popularity, DKT (LSTM), GRU.

Computes the same 15 metrics as run_xes3g5m.py for fair comparison.

Usage:
    python scripts/run_xes3g5m_baselines.py [--seed 42] [--n_students 6000]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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

from agents.prediction_agent import (
    GapSequenceDataset, create_model, NUM_TAGS, DEVICE,
    NUM_CONF_CLASSES, LABEL_SMOOTHING, NUM_WORKERS,
)
from agents.utils import set_global_seed
from data.xes3g5m_loader import load_xes3g5m

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("xes3g5m_baselines")


# ──────────────────────────────────────────────────────────
# Shared eval function (same 15 metrics as run_xes3g5m.py)
# ──────────────────────────────────────────────────────────

def compute_all_metrics(y_score, y_true, train_df, test_df, n_tags):
    """Compute all 15 metrics from prediction scores and ground truth."""
    from sklearn.metrics import roc_auc_score, f1_score
    from sklearn.metrics.pairwise import cosine_distances

    N = len(y_score)
    tag_pos = y_true.sum(axis=0)
    tag_neg = (1 - y_true).sum(axis=0)
    tag_mask = (tag_pos >= 5) & (tag_neg >= 5)
    n_eval_tags = int(tag_mask.sum())

    # AUC
    if n_eval_tags > 0:
        auc_macro = float(roc_auc_score(
            y_true[:, tag_mask], y_score[:, tag_mask], average="macro"))
        auc_weighted = float(roc_auc_score(
            y_true[:, tag_mask], y_score[:, tag_mask], average="weighted"))
    else:
        auc_macro, auc_weighted = 0.0, 0.0

    # F1
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.01, 0.5, 0.01):
        yb = (y_score[:, tag_mask] >= t).astype(int)
        if yb.sum() == 0:
            continue
        f1 = f1_score(y_true[:, tag_mask], yb, average="micro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    # Ranking
    ndcg10s, p10s, r10s, mrrs = [], [], [], []
    all_rec_tags = set()
    diversity_scores, novelty_scores = [], []

    tag_freq = np.zeros(n_tags, dtype=np.float32)
    n_train_users = train_df["user_id"].nunique()
    for tags in train_df["tags"]:
        if isinstance(tags, list):
            for t in tags:
                if 0 <= t < n_tags:
                    tag_freq[t] += 1
    tag_pop = tag_freq / max(n_train_users, 1)
    tag_vecs = np.eye(n_tags, dtype=np.float32)

    for i in range(N):
        gt = set(np.where(y_true[i] > 0)[0])
        if len(gt) == 0:
            continue
        ranked = np.argsort(-y_score[i])[:10].tolist()
        all_rec_tags.update(ranked)

        hits = len(set(ranked) & gt)
        p10s.append(hits / 10)
        r10s.append(hits / len(gt))

        dcg = sum(1.0 / np.log2(r + 2) for r, t in enumerate(ranked) if t in gt)
        ideal = sum(1.0 / np.log2(r + 2) for r in range(min(len(gt), 10)))
        ndcg10s.append(dcg / ideal if ideal > 0 else 0.0)

        rr = 0.0
        for r, t in enumerate(ranked):
            if t in gt:
                rr = 1.0 / (r + 1)
                break
        mrrs.append(rr)

        if len(ranked) >= 2:
            vecs = tag_vecs[ranked]
            dists = cosine_distances(vecs)
            n_r = len(ranked)
            diversity_scores.append(float(dists[np.triu_indices(n_r, k=1)].mean()))

        nov = [- math.log2(tag_pop[t] + 1e-10) for t in ranked if 0 <= t < n_tags]
        if nov:
            novelty_scores.append(float(np.mean(nov)))

    tag_coverage = len(all_rec_tags) / max(n_eval_tags, 1)
    question_coverage = len(all_rec_tags) / max(n_tags, 1)

    # Learning gain
    learning_gains = []
    for uid, grp in test_df.groupby("user_id"):
        grp = grp.sort_values("timestamp")
        if len(grp) < 10:
            continue
        mid = len(grp) // 2
        lg = grp.iloc[mid:]["correct"].mean() - grp.iloc[:mid]["correct"].mean()
        learning_gains.append(lg)

    lg_mean = float(np.mean(learning_gains)) if learning_gains else 0.0
    lg_std = float(np.std(learning_gains)) if learning_gains else 0.0
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
        "n_eval_tags": n_eval_tags,
        "n_eval_users": len(ndcg10s),
    }


# ──────────────────────────────────────────────────────────
# Baselines
# ──────────────────────────────────────────────────────────

def baseline_random(test_dataset, train_df, test_df, n_tags, seed):
    """Random: uniform random scores per tag."""
    rng = np.random.RandomState(seed)
    y_true = np.stack(test_dataset.labels)
    y_score = rng.rand(len(y_true), n_tags).astype(np.float32)
    return compute_all_metrics(y_score, y_true, train_df, test_df, n_tags)


def baseline_popularity(test_dataset, train_df, test_df, n_tags):
    """Popularity: score = tag failure frequency in train."""
    all_labels = np.stack([GapSequenceDataset._all_tags_to_vec(l, n_tags)
                           if hasattr(GapSequenceDataset, '_all_tags_to_vec')
                           else l for l in test_dataset.labels])
    # Actually use pre-built labels directly
    y_true = np.stack(test_dataset.labels)

    # Tag failure frequency from train
    train_ds = GapSequenceDataset(train_df)
    if len(train_ds) > 0:
        train_labels = np.stack(train_ds.labels)
        tag_fail_freq = train_labels.mean(axis=0)  # (n_tags,)
    else:
        tag_fail_freq = np.ones(n_tags) / n_tags

    # Every test sample gets the same popularity-based score
    y_score = np.tile(tag_fail_freq, (len(y_true), 1)).astype(np.float32)
    return compute_all_metrics(y_score, y_true, train_df, test_df, n_tags)


def baseline_model(model_type, train_df, val_df, test_df, n_tags, seed,
                    epochs=30, batch_size=32, patience=5, lr=5e-4):
    """Train a model (LSTM/GRU) baseline and evaluate."""
    import torch.nn as nn

    set_global_seed(seed)
    train_dataset = GapSequenceDataset(train_df)
    val_dataset = GapSequenceDataset(val_df)
    test_dataset = GapSequenceDataset(test_df)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        return {"error": "no sequences"}

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = create_model(model_type, num_conf_classes=NUM_CONF_CLASSES).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("%s baseline: %.2fM params", model_type.upper(), n_params / 1e6)

    # Same loss as Phase A
    all_labels = np.stack(train_dataset.labels)
    pos_rate = all_labels.mean(axis=0)
    pw = np.where(pos_rate > 0, (1.0 - pos_rate) / (pos_rate + 1e-8), 1.0)
    pw = np.clip(pw, 1.0, 50.0)
    pos_weight = torch.tensor(pw, dtype=torch.float32).to(DEVICE)

    def bce_loss(logits, targets):
        targets = targets * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
        return nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_auc = -1.0
    best_state = None
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = bce_loss(model(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(DEVICE)
                all_p.append(torch.sigmoid(model(X)).cpu().numpy())
                all_l.append(y.numpy())
        yp = np.concatenate(all_p)
        yt = np.concatenate(all_l)
        mask = yt.sum(axis=0) > 0
        from sklearn.metrics import roc_auc_score
        try:
            val_auc = float(roc_auc_score(yt[:, mask], yp[:, mask], average="macro"))
        except ValueError:
            val_auc = 0.0

        logger.info("  %s Epoch %d/%d  loss=%.4f  val_auc=%.4f",
                     model_type.upper(), epoch, epochs, np.mean(losses), val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("  %s early stop at epoch %d (best=%d, auc=%.4f)",
                             model_type.upper(), epoch, best_epoch, best_val_auc)
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    # Test predictions
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    all_p, all_l = [], []
    with torch.no_grad():
        for batch in test_loader:
            X = batch[0].to(DEVICE)
            y = batch[1]
            all_p.append(torch.sigmoid(model(X)).cpu().numpy())
            all_l.append(y.numpy())

    y_score = np.concatenate(all_p)
    y_true = np.concatenate(all_l)

    metrics = compute_all_metrics(y_score, y_true, train_df, test_df, n_tags)
    metrics["val_auc"] = round(best_val_auc, 4)
    metrics["best_epoch"] = best_epoch
    metrics["n_params"] = n_params
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_students", type=int, default=6000)
    parser.add_argument("--min_interactions", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    set_global_seed(args.seed)
    logger.info("=" * 60)
    logger.info("XES3G5M BASELINES (seed=%d, n=%d)", args.seed, args.n_students)
    logger.info("=" * 60)

    train_df, val_df, test_df = load_xes3g5m(
        n_students=args.n_students,
        min_interactions=args.min_interactions,
        seed=args.seed,
    )
    for df in [train_df, val_df, test_df]:
        df["confidence_class"] = 0

    all_tags = set()
    for df in [train_df, val_df, test_df]:
        for tags in df["tags"]:
            if isinstance(tags, list):
                all_tags.update(tags)
    n_tags = NUM_TAGS  # use model's NUM_TAGS for consistency

    test_dataset = GapSequenceDataset(test_df)
    logger.info("Test sequences: %d", len(test_dataset))

    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / "xes3g5m" / f"baselines_s{args.seed}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Random
    logger.info("--- Random baseline ---")
    t0 = time.time()
    results["random"] = baseline_random(test_dataset, train_df, test_df, n_tags, args.seed)
    results["random"]["time_s"] = round(time.time() - t0, 1)
    logger.info("  Random AUC=%.4f NDCG=%.4f", results["random"]["test_auc_macro"],
                results["random"]["ndcg@10"])

    # 2. Popularity
    logger.info("--- Popularity baseline ---")
    t0 = time.time()
    results["popularity"] = baseline_popularity(test_dataset, train_df, test_df, n_tags)
    results["popularity"]["time_s"] = round(time.time() - t0, 1)
    logger.info("  Popularity AUC=%.4f NDCG=%.4f", results["popularity"]["test_auc_macro"],
                results["popularity"]["ndcg@10"])

    # 3. DKT (LSTM)
    logger.info("--- DKT (LSTM) baseline ---")
    t0 = time.time()
    results["dkt_lstm"] = baseline_model(
        "lstm", train_df, val_df, test_df, n_tags, args.seed,
        epochs=30, batch_size=args.batch_size, patience=5)
    results["dkt_lstm"]["time_s"] = round(time.time() - t0, 1)
    logger.info("  DKT AUC=%.4f NDCG=%.4f", results["dkt_lstm"]["test_auc_macro"],
                results["dkt_lstm"]["ndcg@10"])

    # 4. GRU
    logger.info("--- GRU baseline ---")
    t0 = time.time()
    results["gru"] = baseline_model(
        "gru", train_df, val_df, test_df, n_tags, args.seed,
        epochs=30, batch_size=args.batch_size, patience=5)
    results["gru"]["time_s"] = round(time.time() - t0, 1)
    logger.info("  GRU AUC=%.4f NDCG=%.4f", results["gru"]["test_auc_macro"],
                results["gru"]["ndcg@10"])

    # Summary
    logger.info("=" * 60)
    logger.info("BASELINE SUMMARY (seed=%d)", args.seed)
    logger.info("=" * 60)
    logger.info("%-12s %8s %8s %8s %8s %8s", "Method", "AUC", "NDCG@10", "P@10", "MRR", "Cov")
    for name, m in results.items():
        logger.info("%-12s %8.4f %8.4f %8.4f %8.4f %8.4f",
                     name,
                     m.get("test_auc_macro", 0),
                     m.get("ndcg@10", 0),
                     m.get("precision@10", 0),
                     m.get("mrr", 0),
                     m.get("tag_coverage", 0))

    # Save
    with open(out_dir / "baselines.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved to %s", out_dir / "baselines.json")


if __name__ == "__main__":
    main()
