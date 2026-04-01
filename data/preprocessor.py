"""
EdNet data preprocessor for MARS project.

Cleans interactions, engineers features, and creates
chronological train/val/test splits saved to Parquet.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Thresholds
MIN_ANSWERS_PER_USER = 10
ELAPSED_MIN_MS = 1_000
ELAPSED_MAX_MS = 300_000
SESSION_GAP_MS = 30 * 60 * 1_000  # 30 minutes
ROLLING_WINDOW = 20

# Chronological split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


class EdNetPreprocessor:
    """Clean, feature-engineer, and split EdNet interactions."""

    def __init__(
        self,
        output_dir: str | Path = "data/processed",
        splits_dir: str | Path = "data/splits",
    ):
        self.output_dir = Path(output_dir)
        self.splits_dir = Path(splits_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(self, interactions: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Full preprocessing pipeline.

        Returns dict with keys 'train', 'val', 'test'.
        """
        logger.info("Starting preprocessing on %d rows", len(interactions))

        df = self.clean(interactions)
        df = self.engineer_features(df)
        splits = self.chronological_split(df)

        # Save
        self._save(df, splits)

        return splits

    # ------------------------------------------------------------------
    # Step 1: Cleaning
    # ------------------------------------------------------------------

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        n0 = len(df)

        # Remove extreme elapsed times
        if "elapsed_time" in df.columns:
            mask = df["elapsed_time"].between(ELAPSED_MIN_MS, ELAPSED_MAX_MS)
            df = df[mask | df["elapsed_time"].isna()].copy()
            logger.info("Elapsed time filter: %d → %d rows", n0, len(df))

        # Remove users with fewer than MIN_ANSWERS_PER_USER answers
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= MIN_ANSWERS_PER_USER].index
        n1 = len(df)
        df = df[df["user_id"].isin(valid_users)].copy()
        logger.info(
            "Min-answers filter (%d): %d → %d rows, %d → %d users",
            MIN_ANSWERS_PER_USER, n1, len(df),
            len(user_counts), len(valid_users),
        )

        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Step 2: Feature Engineering
    # ------------------------------------------------------------------

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

        df = self._add_tag_accuracy(df)
        df = self._add_avg_elapsed_by_tag(df)
        df = self._add_session_id(df)
        df = self._add_rolling_accuracy(df)
        df = self._add_time_since_last(df)

        logger.info("Feature engineering complete. Columns: %s", list(df.columns))
        return df

    @staticmethod
    def _add_tag_accuracy(df: pd.DataFrame) -> pd.DataFrame:
        """Per-user, per-tag cumulative accuracy up to (but not including) current row."""
        if "tags" not in df.columns:
            return df

        # Keep original index for mapping back
        df = df.reset_index(drop=True)
        exploded = df[["user_id", "correct", "tags"]].copy()
        exploded["_orig_idx"] = exploded.index
        exploded = exploded.explode("tags").dropna(subset=["tags"])
        exploded["tags"] = exploded["tags"].astype(int)

        # Cumulative accuracy per (user, tag) — use transform to keep aligned
        exploded = exploded.sort_values(["user_id", "tags"]).reset_index(drop=True)
        exploded["_tag_acc"] = (
            exploded
            .groupby(["user_id", "tags"])["correct"]
            .transform(lambda s: s.expanding().mean())
        )

        # Average tag accuracy across all tags for the original row
        mean_tag_acc = exploded.groupby("_orig_idx")["_tag_acc"].mean()
        df["tag_accuracy"] = df.index.map(mean_tag_acc)
        df["tag_accuracy"] = df["tag_accuracy"].fillna(0.5)  # prior for cold-start

        return df

    @staticmethod
    def _add_avg_elapsed_by_tag(df: pd.DataFrame) -> pd.DataFrame:
        """Z-score normalised average elapsed time per tag (global)."""
        if "tags" not in df.columns or "elapsed_time" not in df.columns:
            return df

        exploded = df[["elapsed_time", "tags"]].explode("tags").dropna(subset=["tags", "elapsed_time"])
        exploded["tags"] = exploded["tags"].astype(int)

        tag_mean = exploded.groupby("tags")["elapsed_time"].mean()
        global_mean = tag_mean.mean()
        global_std = tag_mean.std()
        if global_std == 0 or pd.isna(global_std):
            global_std = 1.0

        tag_z = ((tag_mean - global_mean) / global_std).to_dict()

        # Map back: average z-score across tags per row
        def _avg_z(tags):
            if not isinstance(tags, list) or len(tags) == 0:
                return 0.0
            vals = [tag_z.get(int(t), 0.0) for t in tags]
            return float(np.mean(vals))

        df["avg_elapsed_by_tag"] = df["tags"].apply(_avg_z)
        return df

    @staticmethod
    def _add_session_id(df: pd.DataFrame) -> pd.DataFrame:
        """Assign session_id: gap > 30 min = new session."""
        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        time_diff = df.groupby("user_id")["timestamp"].diff()
        new_session = (time_diff > SESSION_GAP_MS) | time_diff.isna()
        df["session_id"] = new_session.groupby(df["user_id"]).cumsum().astype(int)
        return df

    @staticmethod
    def _add_rolling_accuracy(df: pd.DataFrame) -> pd.DataFrame:
        """Rolling accuracy over the last ROLLING_WINDOW answers per user."""
        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        df["rolling_accuracy"] = (
            df.groupby("user_id")["correct"]
            .transform(lambda s: s.rolling(ROLLING_WINDOW, min_periods=1).mean())
        )
        return df

    @staticmethod
    def _add_time_since_last(df: pd.DataFrame) -> pd.DataFrame:
        """Time (ms) since previous answer for the same user."""
        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        df["time_since_last"] = df.groupby("user_id")["timestamp"].diff()
        df["time_since_last"] = df["time_since_last"].fillna(0).astype(np.int64)
        return df

    # ------------------------------------------------------------------
    # Step 3: Chronological Split
    # ------------------------------------------------------------------

    def chronological_split(
        self, df: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        """
        Per-user chronological split: first 70% train, next 15% val, last 15% test.
        This avoids data leakage from future interactions.
        """
        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

        # Assign split per-user based on cumulative position
        def _assign_split(group: pd.DataFrame) -> pd.Series:
            n = len(group)
            positions = np.arange(n) / n
            splits = pd.Series("train", index=group.index)
            splits[positions >= TRAIN_RATIO] = "val"
            splits[positions >= TRAIN_RATIO + VAL_RATIO] = "test"
            return splits

        df["split"] = df.groupby("user_id", group_keys=False).apply(_assign_split)

        train = df[df["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
        val = df[df["split"] == "val"].drop(columns=["split"]).reset_index(drop=True)
        test = df[df["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)

        logger.info(
            "Split sizes — train: %d (%.1f%%), val: %d (%.1f%%), test: %d (%.1f%%)",
            len(train), 100 * len(train) / len(df),
            len(val), 100 * len(val) / len(df),
            len(test), 100 * len(test) / len(df),
        )

        return {"train": train, "val": val, "test": test}

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def _save(
        self,
        full: pd.DataFrame,
        splits: dict[str, pd.DataFrame],
    ) -> None:
        # Tags as lists can't go into parquet directly — convert to strings
        for frame in [full, *splits.values()]:
            if "tags" in frame.columns:
                frame["tags"] = frame["tags"].apply(
                    lambda x: ";".join(str(t) for t in x) if isinstance(x, list) else str(x)
                )

        full_path = self.output_dir / "interactions.parquet"
        full.to_parquet(full_path, index=False, engine="pyarrow")
        logger.info("Saved full dataset → %s (%d rows)", full_path, len(full))

        for name, frame in splits.items():
            path = self.splits_dir / f"{name}.parquet"
            frame.to_parquet(path, index=False, engine="pyarrow")
            logger.info("Saved %s → %s (%d rows)", name, path, len(frame))
