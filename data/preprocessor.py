"""
EdNet data preprocessor for MARS project.

Cleans interactions, engineers features, and creates
chronological train/val/test splits saved to Parquet.

Supports chunked processing for large datasets (50K+ users)
with progress reporting.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    # Fallback: no-op progress bar if tqdm not installed
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable=None, **kwargs):
            self._it = iterable
        def __iter__(self):
            return iter(self._it)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def set_postfix_str(self, s):
            pass
        def update(self, n=1):
            pass

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
        chunk_size: int = 5_000,
    ):
        self.output_dir = Path(output_dir)
        self.splits_dir = Path(splits_dir)
        self.chunk_size = chunk_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        interactions: pd.DataFrame,
        chunked: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        Full preprocessing pipeline.

        Parameters
        ----------
        interactions : pd.DataFrame
            Raw interactions from EdNetLoader.
        chunked : bool
            If True, process feature engineering in user-batches
            of ``self.chunk_size`` to reduce peak RAM usage.
            Recommended for datasets with >10K users.

        Returns dict with keys 'train', 'val', 'test'.
        """
        t0 = time.time()
        n_users = interactions["user_id"].nunique()
        logger.info(
            "Starting preprocessing: %d rows, %d users (chunked=%s)",
            len(interactions), n_users, chunked,
        )

        df = self.clean(interactions)

        if chunked and n_users > self.chunk_size:
            df = self._engineer_features_chunked(df)
        else:
            df = self.engineer_features(df)

        splits = self.chronological_split(df)

        # Save
        self._save(df, splits)

        t_total = time.time() - t0
        logger.info("Preprocessing complete in %.1fs", t_total)

        return splits

    # ------------------------------------------------------------------
    # Step 1: Cleaning
    # ------------------------------------------------------------------

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        n0 = len(df)
        n_users_0 = df["user_id"].nunique()

        # Remove extreme elapsed times
        if "elapsed_time" in df.columns:
            mask = df["elapsed_time"].between(ELAPSED_MIN_MS, ELAPSED_MAX_MS)
            df = df[mask | df["elapsed_time"].isna()].copy()
            n_elapsed_removed = n0 - len(df)
            logger.info(
                "Elapsed time filter [%d, %d]ms: %d → %d rows (removed %d, %.1f%%)",
                ELAPSED_MIN_MS, ELAPSED_MAX_MS, n0, len(df),
                n_elapsed_removed, 100 * n_elapsed_removed / max(n0, 1),
            )

        # Remove users with fewer than MIN_ANSWERS_PER_USER answers
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= MIN_ANSWERS_PER_USER].index
        n1 = len(df)
        n_users_before = len(user_counts)
        df = df[df["user_id"].isin(valid_users)].copy()
        n_users_removed = n_users_before - len(valid_users)
        logger.info(
            "Min-answers filter (>=%d): %d → %d rows, %d → %d users "
            "(removed %d users, %.1f%%)",
            MIN_ANSWERS_PER_USER, n1, len(df),
            n_users_before, len(valid_users),
            n_users_removed, 100 * n_users_removed / max(n_users_before, 1),
        )
        logger.info(
            "Cleaning summary: %d → %d rows (%.1f%% retained), "
            "%d → %d users (%.1f%% retained)",
            n0, len(df), 100 * len(df) / max(n0, 1),
            n_users_0, len(valid_users), 100 * len(valid_users) / max(n_users_0, 1),
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

    def _engineer_features_chunked(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature engineering in user-batches to limit peak RAM.

        Global statistics (tag means, median elapsed) are computed once
        on the full dataset, then per-user features are computed in chunks.
        """
        t0 = time.time()
        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

        # --- Global stats needed across chunks ---
        # Tag-level average elapsed time (z-scored)
        tag_z = {}
        if "tags" in df.columns and "elapsed_time" in df.columns:
            exploded = df[["elapsed_time", "tags"]].explode("tags").dropna(
                subset=["tags", "elapsed_time"]
            )
            exploded["tags"] = exploded["tags"].astype(int)
            tag_mean = exploded.groupby("tags")["elapsed_time"].mean()
            global_mean = tag_mean.mean()
            global_std = tag_mean.std()
            if global_std == 0 or pd.isna(global_std):
                global_std = 1.0
            tag_z = ((tag_mean - global_mean) / global_std).to_dict()

        # --- Chunked processing ---
        users = df["user_id"].unique()
        n_chunks = int(np.ceil(len(users) / self.chunk_size))
        logger.info(
            "Chunked feature engineering: %d users in %d chunks of %d",
            len(users), n_chunks, self.chunk_size,
        )

        chunks_out = []
        for ci in tqdm(range(n_chunks), desc="Feature engineering"):
            chunk_users = users[ci * self.chunk_size : (ci + 1) * self.chunk_size]
            chunk_df = df[df["user_id"].isin(set(chunk_users))].copy()

            # Per-user features
            chunk_df = self._add_tag_accuracy(chunk_df)
            chunk_df = self._add_avg_elapsed_by_tag_precomputed(chunk_df, tag_z)
            chunk_df = self._add_session_id(chunk_df)
            chunk_df = self._add_rolling_accuracy(chunk_df)
            chunk_df = self._add_time_since_last(chunk_df)

            chunks_out.append(chunk_df)
            if (ci + 1) % 5 == 0 or ci == n_chunks - 1:
                logger.info(
                    "Chunk %d/%d done (%.1fs elapsed)",
                    ci + 1, n_chunks, time.time() - t0,
                )

        result = pd.concat(chunks_out, ignore_index=True)
        result = result.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        logger.info(
            "Chunked feature engineering complete: %d rows in %.1fs",
            len(result), time.time() - t0,
        )
        return result

    @staticmethod
    def _add_avg_elapsed_by_tag_precomputed(
        df: pd.DataFrame, tag_z: dict
    ) -> pd.DataFrame:
        """Map pre-computed tag z-scores (used in chunked mode)."""
        if not tag_z or "tags" not in df.columns:
            df["avg_elapsed_by_tag"] = 0.0
            return df

        def _avg_z(tags):
            if not isinstance(tags, list) or len(tags) == 0:
                return 0.0
            vals = [tag_z.get(int(t), 0.0) for t in tags]
            return float(np.mean(vals))

        df["avg_elapsed_by_tag"] = df["tags"].apply(_avg_z)
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
