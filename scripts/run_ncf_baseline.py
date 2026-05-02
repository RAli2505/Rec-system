"""
NCF (Neural Collaborative Filtering, He et al. 2017) baseline
for MARS Table 4 (reviewer item C1).

NCF is canonical user-item matrix factorisation extended with an
MLP. We adapt it to the MARS per-tag failure-prediction task by
treating (user, tag) as the latent matrix and predicting failure
probability per tag at the user's CURRENT step. Sequential context
is summarised by mean-pooling the user's recent interactions into
the user embedding via an attentional read-out, matching how MARS
folds historical context into its 14-dim per-step input.

Architecture:
  user_emb = Embedding(n_users, d) + mean-pool of recent (correct,
              log_elapsed) signals projected through a small MLP
              (gives per-step context conditioning without full
              sequential modelling)
  tag_emb  = Embedding(n_tags, d)
  NCF MLP: [user_emb || tag_emb || user_emb*tag_emb] → 256 → 128 → 1

Trained per-step on the GapSequenceDataset's last-step targets
(same multi-label per-tag failure target that the attention-KT
baselines use). One seed = ~25 minutes on a single GPU.

Output:
  results/xes3g5m/ncf_baseline_<ts>/results_s{seed}.json
  results/xes3g5m/ncf_baseline_<ts>/summary.csv
"""

from __future__ import annotations

import argparse
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from agents import prediction_agent as PA
from agents.prediction_agent import (
    GapSequenceDataset, set_num_tags, DEVICE,
)
from agents.utils import set_global_seed
from data.xes3g5m_loader import load_xes3g5m
from scripts.run_xes3g5m_baselines import compute_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("ncf")


class NCFBaseline(nn.Module):
    """NCF adapted to per-step multi-label per-tag failure prediction.

    The user's current state is represented by mean-pooled history
    statistics (correct, elapsed) projected into the same dim as the
    tag embedding, plus a learnable user-id embedding (used during
    training; falls back to mean-pool only at test time when the
    user-id is unseen — so the model stays usable without per-test-
    user fitting).
    """

    def __init__(self, n_tags: int, d_emb: int = 64, d_hidden: int = 256,
                 dropout: float = 0.2):
        super().__init__()
        self.n_tags = n_tags
        self.d_emb = d_emb
        # No user-id embedding (test users are disjoint from train);
        # we represent users by their pooled history.
        self.tag_emb = nn.Embedding(n_tags + 1, d_emb, padding_idx=0)
        # History → user embedding projector (input: 14-dim per-step
        # vector mean-pooled over time, output: d_emb).
        self.user_proj = nn.Sequential(
            nn.Linear(14, d_emb),
            nn.GELU(),
            nn.LayerNorm(d_emb),
            nn.Linear(d_emb, d_emb),
        )
        # NCF MLP head: [user || tag || user*tag] → score.
        self.mlp = nn.Sequential(
            nn.Linear(3 * d_emb, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 14) — same input format as the attention-KT
        baselines. We mean-pool over T to get a static user
        representation and score each tag."""
        # Mean-pool over time: (B, 14)
        user_h = x.mean(dim=1)
        u = self.user_proj(user_h)                             # (B, d_emb)
        # Score every tag: build (B, n_tags, d_emb) via broadcast.
        tag_idx = torch.arange(1, self.n_tags + 1,
                                 device=x.device)               # (n_tags,)
        t = self.tag_emb(tag_idx)                                # (n_tags, d_emb)
        # Cross-product features
        u_exp = u.unsqueeze(1).expand(-1, self.n_tags, -1)      # (B, n_tags, d_emb)
        t_exp = t.unsqueeze(0).expand(u.size(0), -1, -1)        # (B, n_tags, d_emb)
        feat = torch.cat([u_exp, t_exp, u_exp * t_exp], dim=-1) # (B, n_tags, 3*d)
        logits = self.mlp(feat).squeeze(-1)                      # (B, n_tags)
        return logits


def train_one_seed(seed: int, args: argparse.Namespace,
                    out_dir: Path) -> dict:
    set_global_seed(seed)

    train_df, val_df, test_df = load_xes3g5m(
        n_students=args.n_students,
        min_interactions=args.min_interactions,
        seed=seed,
    )
    for df in [train_df, val_df, test_df]:
        df["confidence_class"] = 0
    train_max_id = max(int(t) for tags in train_df["tags"]
                        if isinstance(tags, list) and tags
                        for t in tags)
    n_tags = train_max_id + 1
    set_num_tags(n_tags)

    train_ds = GapSequenceDataset(train_df)
    val_ds = GapSequenceDataset(val_df)
    test_ds = GapSequenceDataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                shuffle=True, num_workers=0,
                                pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    model = NCFBaseline(n_tags=n_tags).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("NCF seed=%d params=%.2fM", seed, n_params / 1e6)

    # Pos-weight to counter class imbalance (same as attention-KT).
    all_labels = np.stack(train_ds.labels)
    pos_rate = all_labels.mean(axis=0)
    pw = np.where(pos_rate > 0, (1.0 - pos_rate) / (pos_rate + 1e-8), 1.0)
    pw = np.clip(pw, 1.0, 50.0)
    pos_weight = torch.tensor(pw, dtype=torch.float32).to(DEVICE)

    def loss_fn(logits, targets):
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    from sklearn.metrics import roc_auc_score

    best_val_auc, best_state, no_improve = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())

        model.eval()
        ps, ls = [], []
        with torch.no_grad():
            for X, y in val_loader:
                ps.append(torch.sigmoid(model(X.to(DEVICE))).cpu().numpy())
                ls.append(y.numpy())
        yp = np.concatenate(ps)
        yt = np.concatenate(ls)
        mask = yt.sum(axis=0) > 0
        try:
            val_auc = float(roc_auc_score(yt[:, mask], yp[:, mask],
                                            average="macro"))
        except ValueError:
            val_auc = 0.0
        logger.info("  NCF seed=%d ep=%2d loss=%.4f val_auc=%.4f",
                     seed, ep, float(np.mean(losses)), val_auc)
        if val_auc > best_val_auc:
            best_val_auc, no_improve = val_auc, 0
            best_state = {k: v.cpu().clone()
                            for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info("  NCF seed=%d early stop ep=%d", seed, ep)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                shuffle=False)
    ps, ls = [], []
    with torch.no_grad():
        for X, y in test_loader:
            ps.append(torch.sigmoid(model(X.to(DEVICE))).cpu().numpy())
            ls.append(y.numpy())
    y_score = np.concatenate(ps)
    y_true = np.concatenate(ls)

    metrics = compute_all_metrics(y_score, y_true, train_df, test_df, n_tags)
    metrics.update({
        "model": "NCF",
        "seed":  int(seed),
        "n_params": int(n_params),
        "val_auc": round(best_val_auc, 4),
    })

    out_path = out_dir / f"results_NCF_s{seed}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("[NCF s%d] saved %s NDCG@10=%.4f MRR=%.4f Cov=%.4f",
                 seed, out_path.name, metrics["ndcg@10"],
                 metrics["mrr"], metrics["tag_coverage"])
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int,
                          default=[42, 123, 456, 789, 2024])
    parser.add_argument("--n_students", type=int, default=6000)
    parser.add_argument("--min_interactions", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results/xes3g5m" / f"ncf_baseline_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("NCF output dir: %s", out_dir)

    rows = []
    for seed in args.seeds:
        try:
            rows.append(train_one_seed(seed, args, out_dir))
        except Exception as e:
            logger.exception("seed %d failed: %s", seed, e)

    # Summary
    if rows:
        keys = ["test_auc_macro", "ndcg@10", "precision@10",
                  "recall@10", "mrr", "tag_coverage"]
        summary = {"model": "NCF", "n_seeds": len(rows)}
        for k in keys:
            vals = [r.get(k) for r in rows if r.get(k) is not None]
            if vals:
                summary[f"{k}_mean"] = round(float(np.mean(vals)), 4)
                summary[f"{k}_std"]  = (
                    round(float(np.std(vals, ddof=1)), 4)
                    if len(vals) > 1 else 0.0
                )
        pd.DataFrame([summary]).to_csv(out_dir / "summary.csv",
                                          index=False)
        logger.info("Saved summary: %s", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
