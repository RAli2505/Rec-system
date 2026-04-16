"""
EdNet loader matching XES3G5M sampling protocol for fair comparison.

Samples 6000 users with >=20 interactions (same as XES3G5M), applies
user-level split 70/15/15, returns train/val/test DataFrames.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("mars.data.ednet_comparable")


def load_ednet_comparable(
    data_dir: str = "data/raw",
    n_students: int = 6000,
    min_interactions: int = 20,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load EdNet with identical sampling to XES3G5M for cross-dataset comparison."""
    from data.loader import EdNetLoader
    from data.preprocessor import EdNetPreprocessor

    # Load interactions — oversample because we filter by min_interactions after
    loader = EdNetLoader(data_dir=data_dir)
    # Load 2x n_students to have enough after filtering
    raw = loader.load_interactions(
        sample_users=n_students * 2,
        stratified_sampling=True,
    )
    logger.info("EdNet raw: %d interactions, %d users", len(raw), raw["user_id"].nunique())

    # Clean via preprocessor
    pp = EdNetPreprocessor()
    df = pp.clean(raw)
    df = pp.engineer_features(df)
    logger.info("EdNet cleaned: %d interactions", len(df))

    # Filter users with >=min_interactions
    user_counts = df.groupby("user_id").size()
    qualified = user_counts[user_counts >= min_interactions].index
    df = df[df["user_id"].isin(qualified)]
    logger.info("After filter (>=%d interactions): %d users, %d interactions",
                min_interactions, df["user_id"].nunique(), len(df))

    # Sample exactly n_students users (deterministic with seed)
    rng = np.random.RandomState(seed)
    all_users = list(df["user_id"].unique())
    if len(all_users) > n_students:
        rng.shuffle(all_users)
        sampled_users = all_users[:n_students]
        df = df[df["user_id"].isin(sampled_users)].reset_index(drop=True)

    logger.info("EdNet sampled: %d interactions from %d students (seed=%d)",
                len(df), df["user_id"].nunique(), seed)

    # User-level split (same as XES3G5M)
    all_users = list(df["user_id"].unique())
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
        "EdNet splits: train=%d (%d users) / val=%d (%d users) / test=%d (%d users)",
        len(train), len(train_users), len(val), len(val_users), len(test), len(test_users),
    )
    return train, val, test
