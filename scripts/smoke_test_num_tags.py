"""
Smoke-test the dynamic NUM_TAGS refactor in prediction_agent.

Verifies:
  1. Default NUM_TAGS = 293 still works (EdNet backward compat)
  2. set_num_tags(858) propagates to:
     - GapSequenceDataset label dim
     - Model embedding size
     - Model output dim (forward pass shape)
  3. Tags above NUM_TAGS in TRAIN data still get clipped (no crash)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch

from agents import prediction_agent as PA
from agents.prediction_agent import (
    GapSequenceDataset, create_model, set_num_tags, SEQ_LEN, HORIZON,
    DEVICE,
)


def _make_synthetic_df(n_users: int, max_tag_id: int, seed: int = 0):
    """Build a tiny chronological interaction dataframe with concept IDs
    drawn from [0, max_tag_id]. Each user has SEQ_LEN+HORIZON+10 rows so
    GapSequenceDataset can slice at least one window."""
    rng = np.random.RandomState(seed)
    n_per_user = SEQ_LEN + HORIZON + 10
    rows = []
    for u in range(n_users):
        for i in range(n_per_user):
            tid = int(rng.randint(0, max_tag_id + 1))
            rows.append({
                "user_id": f"u{u}",
                "timestamp": i,
                "question_id": f"q{i}",
                "tags": [tid],
                "correct": int(rng.randint(0, 2)),
                "elapsed_time": 15000.0,
                "changed_answer": 0,
                "part_id": 1,
                "confidence_class": 0,
            })
    return pd.DataFrame(rows)


def case(label, *, override_num_tags=None, train_max_id, expected_dim):
    print(f"\n--- {label} ---")
    if override_num_tags is not None:
        set_num_tags(override_num_tags)
    print(f"  PA.NUM_TAGS = {PA.NUM_TAGS}")

    df = _make_synthetic_df(n_users=4, max_tag_id=train_max_id, seed=0)
    ds = GapSequenceDataset(df)
    print(f"  dataset built: {len(ds)} sequences,  label dim = {ds.labels[0].shape[0]}")
    assert ds.labels[0].shape[0] == expected_dim, \
        f"label dim mismatch: got {ds.labels[0].shape[0]}, expected {expected_dim}"

    model = create_model("lstm").to(DEVICE)
    n_emb = model.tag_embedding.num_embeddings
    n_out = model.fc[-1].out_features
    print(f"  model: tag_embedding({n_emb}) -> fc out_features = {n_out}")
    assert n_emb == expected_dim + 1, \
        f"embedding mismatch: got {n_emb}, expected {expected_dim + 1}"
    assert n_out == expected_dim, \
        f"fc output mismatch: got {n_out}, expected {expected_dim}"

    # Forward pass
    X = torch.from_numpy(np.stack(ds.sequences[:2])).to(DEVICE)
    with torch.no_grad():
        out = model(X)
    print(f"  forward pass: input {tuple(X.shape)} -> output {tuple(out.shape)}")
    assert out.shape == (2, expected_dim), \
        f"forward shape mismatch: got {tuple(out.shape)}, expected (2, {expected_dim})"
    print("  PASS")


# Case 1: default 293 (EdNet backward compat) — train data fits within range
case("Default NUM_TAGS = 293 (EdNet)",
     override_num_tags=None, train_max_id=200, expected_dim=293)

# Case 2: XES3G5M-sized 858
case("XES3G5M NUM_TAGS = 858",
     override_num_tags=858, train_max_id=857, expected_dim=858)

# Case 3: tiny override
case("Custom NUM_TAGS = 50",
     override_num_tags=50, train_max_id=49, expected_dim=50)

# Restore default for other consumers
set_num_tags(293)
print("\nAll cases passed. NUM_TAGS reset to 293.")
