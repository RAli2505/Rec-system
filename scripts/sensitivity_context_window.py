"""
Inference-only sweep over the input context window (reviewer C.2).
Uses seed_42's saved best.pt and feeds it different `seq_len` truncations:
context = {25, 50, 75, 100} most recent interactions.

Note: the model was trained with seq_len=100. Smaller context just means
the model's positional encoding sees fewer non-padding positions; this is
how MARS would behave at inference for a user with shorter recent
history. No retraining is needed.

Output:
  results/xes3g5m/sensitivity_context_window_s42.{json,csv}
"""

from __future__ import annotations

import argparse
import glob
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
from torch.utils.data import DataLoader

from agents.utils import set_global_seed
from agents.prediction_agent import (
    PredictionAgent, GapSequenceDataset, create_model, set_num_tags,
    DEVICE, NUM_CONF_CLASSES, HORIZON,
)
from data.xes3g5m_loader import load_xes3g5m

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("context_window_sensitivity")

SEED = 42
WINDOWS = [25, 50, 75, 100]


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


def evaluate_window(
    seq_len: int, test_df, pretrained_pt: Path, n_tags: int,
) -> dict:
    """Compute prediction-only ranking metrics with a given seq_len truncation."""
    set_global_seed(SEED)

    pred = PredictionAgent()
    model = create_model(pred.model_type, num_conf_classes=NUM_CONF_CLASSES).to(DEVICE)
    state = torch.load(pretrained_pt, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    # Build dataset with the same seq_len + horizon (override default)
    ds = GapSequenceDataset(test_df, seq_len=seq_len, horizon=HORIZON)
    if len(ds) == 0:
        return {"seq_len": seq_len, "n_sequences": 0, "error": "empty"}

    loader = DataLoader(ds, batch_size=64, shuffle=False)
    all_score, all_true = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            # Pad / truncate the seq dim to match model's expected SEQ_LEN=100.
            target_len = 100
            cur_len = X.shape[1]
            if cur_len < target_len:
                # Left-pad with zeros so the model still sees seq_len=100
                pad = torch.zeros(X.shape[0], target_len - cur_len,
                                   X.shape[2], device=X.device)
                X = torch.cat([pad, X], dim=1)
            elif cur_len > target_len:
                X = X[:, -target_len:, :]
            p = torch.sigmoid(model(X)).cpu().numpy()
            all_score.append(p)
            all_true.append(y.numpy())
    y_score = np.concatenate(all_score)
    y_true  = np.concatenate(all_true)

    # AUC-ROC macro on tags with >=5 pos & >=5 neg
    from sklearn.metrics import roc_auc_score
    tag_mask = (y_true.sum(axis=0) >= 5) & ((1 - y_true).sum(axis=0) >= 5)
    if tag_mask.sum() > 1:
        auc = float(roc_auc_score(y_true[:, tag_mask], y_score[:, tag_mask],
                                   average="macro"))
    else:
        auc = 0.0

    # NDCG@10 / MRR / P@10 per sequence
    ndcg_list, mrr_list, p_list = [], [], []
    for i in range(len(y_score)):
        gt = set(np.where(y_true[i] > 0)[0])
        if not gt:
            continue
        ranked = np.argsort(-y_score[i])[:10].tolist()
        hits = len(set(ranked) & gt)
        p_list.append(hits / 10)
        dcg = sum(1.0 / np.log2(r + 2) for r, t in enumerate(ranked) if t in gt)
        ideal = sum(1.0 / np.log2(r + 2) for r in range(min(len(gt), 10)))
        ndcg_list.append(dcg / ideal if ideal > 0 else 0.0)
        rr = 0.0
        for r, t in enumerate(ranked):
            if t in gt:
                rr = 1.0 / (r + 1); break
        mrr_list.append(rr)

    return {
        "seq_len":      seq_len,
        "n_sequences":  len(y_score),
        "lstm_auc":     round(auc, 4),
        "ndcg@10":      round(float(np.mean(ndcg_list)), 4) if ndcg_list else 0.0,
        "precision@10": round(float(np.mean(p_list)),    4) if p_list    else 0.0,
        "mrr":          round(float(np.mean(mrr_list)),  4) if mrr_list  else 0.0,
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

    rows = []
    for w in WINDOWS:
        t0 = time.time()
        try:
            r = evaluate_window(w, test_df, pretrained_pt, train_max_id + 1)
            r["time_s"] = round(time.time() - t0, 1)
            rows.append(r)
            logger.info("seq_len=%d  AUC=%.4f  NDCG=%.4f  MRR=%.4f  P@10=%.4f  (n=%d)",
                         w, r.get("lstm_auc", 0), r.get("ndcg@10", 0),
                         r.get("mrr", 0), r.get("precision@10", 0),
                         r.get("n_sequences", 0))
        except Exception as e:
            logger.exception("FAILED seq_len=%d: %s", w, e)
            rows.append({"seq_len": w, "error": str(e),
                          "time_s": round(time.time() - t0, 1)})

    out_json = ROOT / "results" / "xes3g5m" / "sensitivity_context_window_s42.json"
    out_csv  = ROOT / "results" / "xes3g5m" / "sensitivity_context_window_s42.csv"
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logger.info("Saved %s + %s", out_json, out_csv)


if __name__ == "__main__":
    main()
