"""
Backfill lstm_auc / lstm_auc_weighted / lstm_f1_micro / lstm_f1_macro
in seed runs whose orchestrator silently dropped them due to a hardcoded
NUM_TAGS=293 check (see orchestrator.py:599).

Loads best.pt from each affected seed run, re-runs the prediction model on
the test set, computes the missing AUC/F1 metrics, patches metrics.json
in place. Idempotent — safe to re-run.

Usage:
    python scripts/backfill_lstm_auc.py [--seeds 42 123]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import DataLoader

from agents import prediction_agent as PA
from agents.prediction_agent import (
    GapSequenceDataset, create_model, set_num_tags,
    DEVICE, NUM_CONF_CLASSES,
)
from data.xes3g5m_loader import load_xes3g5m


def newest_run_for_seed(seed: int) -> Path | None:
    pat = f"results/xes3g5m/xes3g5m_full_s{seed}_*"
    runs = sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)
    return Path(runs[0]) if runs else None


def backfill(seed: int, n_students: int = 6000, min_inter: int = 20) -> None:
    run_dir = newest_run_for_seed(seed)
    if run_dir is None:
        print(f"[seed {seed}] no run found — skip")
        return
    metrics_path = run_dir / "metrics.json"
    best_pt = run_dir / "best.pt"
    if not metrics_path.exists() or not best_pt.exists():
        print(f"[seed {seed}] missing metrics.json or best.pt — skip")
        return

    with open(metrics_path) as f:
        m = json.load(f)
    if "lstm_auc" in m.get("eval_metrics", {}):
        print(f"[seed {seed}] lstm_auc already present — skip")
        return

    print(f"[seed {seed}] backfilling AUC/F1 from {run_dir.name}")
    t0 = time.time()

    # Reload data with the same seed used for the original run
    train_df, val_df, test_df = load_xes3g5m(
        n_students=n_students, min_interactions=min_inter, seed=seed,
    )
    for df in [train_df, val_df, test_df]:
        df["confidence_class"] = 0

    # Configure NUM_TAGS from train data (must match what was used during training)
    train_max_id = 0
    for tags in train_df["tags"]:
        if isinstance(tags, list) and tags:
            train_max_id = max(train_max_id, max(int(t) for t in tags))
    n_tags = train_max_id + 1
    set_num_tags(n_tags)
    print(f"  NUM_TAGS = {n_tags}")

    # Build test dataset and load model checkpoint
    test_ds = GapSequenceDataset(test_df)
    print(f"  test sequences: {len(test_ds)}")

    # Default model_type for run_xes3g5m_full is the PredictionAgent default;
    # the saved checkpoint dictates the architecture indirectly via state-dict
    # shape, so try transformer first then fallback to lstm.
    state = torch.load(best_pt, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Detect arch from state dict keys
    keys = list(state.keys()) if hasattr(state, "keys") else []
    if any("transformer" in k for k in keys):
        model_type = "transformer"
    elif any("gru" in k.lower() for k in keys):
        model_type = "gru"
    else:
        model_type = "lstm"
    print(f"  detected model_type = {model_type}")

    model = create_model(model_type, num_conf_classes=NUM_CONF_CLASSES).to(DEVICE)
    model.load_state_dict(state)
    model.eval()

    # Run forward pass
    loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    all_p, all_y = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            p = torch.sigmoid(model(X)).cpu().numpy()
            all_p.append(p)
            all_y.append(y.numpy())
    y_score = np.concatenate(all_p)
    y_true  = np.concatenate(all_y)

    # AUC + F1 (same logic as orchestrator.batch_evaluation)
    tag_pos = y_true.sum(axis=0)
    tag_neg = (1 - y_true).sum(axis=0)
    tag_mask = (tag_pos >= 5) & (tag_neg >= 5)
    out = {}
    if tag_mask.sum() > 1:
        out["lstm_auc"] = round(float(roc_auc_score(
            y_true[:, tag_mask], y_score[:, tag_mask], average="macro")), 4)
        out["lstm_auc_weighted"] = round(float(roc_auc_score(
            y_true[:, tag_mask], y_score[:, tag_mask], average="weighted")), 4)
    # Best-threshold F1
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.005, 0.51, 0.005):
        yb = (y_score[:, tag_mask] >= t).astype(int)
        if yb.sum() == 0:
            continue
        f1 = f1_score(y_true[:, tag_mask], yb, average="micro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    out["lstm_f1_micro"] = round(best_f1, 4)
    out["lstm_threshold"] = round(float(best_t), 4)
    yb = (y_score[:, tag_mask] >= best_t).astype(int)
    out["lstm_f1_macro"] = round(float(f1_score(
        y_true[:, tag_mask], yb, average="macro", zero_division=0)), 4)
    out["n_eval_tags"] = int(tag_mask.sum())

    # Patch metrics.json
    m["eval_metrics"].update(out)
    m["_backfilled"] = {"lstm_auc_etc": True, "n_tags_used": n_tags}
    with open(metrics_path, "w") as f:
        json.dump(m, f, indent=2, default=str)

    print(f"  patched in {time.time()-t0:.1f}s: " +
          ", ".join(f"{k}={v}" for k, v in out.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    parser.add_argument("--n_students", type=int, default=6000)
    parser.add_argument("--min_interactions", type=int, default=20)
    args = parser.parse_args()
    for s in args.seeds:
        backfill(s, args.n_students, args.min_interactions)
    print("\nBackfill complete.")
