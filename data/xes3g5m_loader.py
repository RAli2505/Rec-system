"""
XES3G5M dataset loader for MARS pipeline.

Converts the XES3G5M Chinese math KT dataset into the same DataFrame
format that EdNet uses (user_id, timestamp, tags, correct, elapsed_time,
changed_answer, part_id, question_id, confidence_class) so the existing
PredictionAgent / Orchestrator can process it without code changes.

Usage:
    from data.xes3g5m_loader import load_xes3g5m
    train_df, val_df, test_df = load_xes3g5m(
        data_dir="data/xes3g5m/XES3G5M",
        n_students=6000,
        min_interactions=20,
        seed=42,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("mars.data.xes3g5m")


def load_xes3g5m(
    data_dir: str = "data/xes3g5m/XES3G5M",
    n_students: int = 6000,
    min_interactions: int = 20,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and preprocess XES3G5M into MARS-compatible DataFrames.

    Returns (train_df, val_df, test_df) with per-user chronological splits.
    """
    rng = np.random.RandomState(seed)
    kc_path = Path(data_dir) / "kc_level" / "train_valid_sequences.csv"
    logger.info("Loading XES3G5M from %s", kc_path)

    raw = pd.read_csv(kc_path)
    logger.info("Raw: %d students", len(raw))

    # Expand sequences into per-interaction rows
    rows = []
    for _, seq in raw.iterrows():
        uid = str(seq["uid"])
        questions = str(seq["questions"]).split(",")
        concepts = str(seq["concepts"]).split(",")
        responses = str(seq["responses"]).split(",")
        timestamps = str(seq["timestamps"]).split(",")

        n = min(len(questions), len(concepts), len(responses), len(timestamps))
        valid_count = 0
        for i in range(n):
            r = responses[i].strip()
            ts = timestamps[i].strip()
            if r in ("-1", "") or ts in ("-1", ""):
                continue
            valid_count += 1

        if valid_count < min_interactions:
            continue

        # Parse valid timestamps for elapsed_time computation
        valid_indices = []
        valid_ts = []
        for i in range(n):
            r = responses[i].strip()
            ts = timestamps[i].strip()
            if r in ("-1", "") or ts in ("-1", ""):
                continue
            valid_indices.append(i)
            valid_ts.append(int(ts))

        for vi_pos, i in enumerate(valid_indices):
            r = responses[i].strip()
            ts = timestamps[i].strip()
            q = questions[i].strip()
            c = concepts[i].strip()

            # Synthesize elapsed_time from consecutive timestamps:
            # elapsed[i] = timestamp[i+1] - timestamp[i]
            # Cap at 300000ms (5 min) — anything longer is a session break.
            # Last interaction gets median fallback (15000ms).
            if vi_pos + 1 < len(valid_ts):
                elapsed = valid_ts[vi_pos + 1] - valid_ts[vi_pos]
                elapsed = max(1000, min(elapsed, 300000))  # clamp [1s, 5min]
            else:
                elapsed = 15000  # last interaction: median fallback

            rows.append({
                "user_id": uid,
                "timestamp": int(ts),
                "question_id": f"q{q}",
                "tags": [int(c)] if c.isdigit() else [0],
                "correct": int(r),
                "elapsed_time": float(elapsed),
                "changed_answer": 0,      # not available in XES3G5M
                "part_id": 1,             # single domain (math)
            })

    df = pd.DataFrame(rows)
    logger.info("Expanded: %d interactions from %d students",
                len(df), df["user_id"].nunique())

    # Sample n_students
    all_users = df["user_id"].unique()
    if len(all_users) > n_students:
        sampled_users = rng.choice(all_users, size=n_students, replace=False)
        df = df[df["user_id"].isin(sampled_users)].reset_index(drop=True)
    logger.info("Sampled: %d interactions from %d students",
                len(df), df["user_id"].nunique())

    # User-level split: entire users go to train/val/test (not per-user
    # temporal slicing). This ensures val/test users have FULL interaction
    # histories (avg ~156 steps) which is enough for GapSequenceDataset
    # to build sequences (needs SEQ_LEN + HORIZON = 120 min). Per-user
    # temporal split would give val only ~23 steps/user → nearly zero
    # valid sequences.
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    all_users = list(df["user_id"].unique())  # convert from ArrowStringArray to list
    rng.shuffle(all_users)
    n_train = int(len(all_users) * train_ratio)
    n_val = int(len(all_users) * val_ratio)
    train_users = set(all_users[:n_train])
    val_users = set(all_users[n_train:n_train + n_val])
    test_users = set(all_users[n_train + n_val:])

    train = df[df["user_id"].isin(train_users)].reset_index(drop=True)
    val = df[df["user_id"].isin(val_users)].reset_index(drop=True)
    test = df[df["user_id"].isin(test_users)].reset_index(drop=True)

    logger.info(
        "Splits: train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%)",
        len(train), 100 * len(train) / len(df),
        len(val), 100 * len(val) / len(df),
        len(test), 100 * len(test) / len(df),
    )

    # Stats
    n_tags = len(set(t for tags in df["tags"] for t in tags))
    n_questions = df["question_id"].nunique()
    logger.info("XES3G5M stats: %d concepts, %d questions, %.1f avg interactions/student",
                n_tags, n_questions, len(df) / df["user_id"].nunique())

    return train, val, test
