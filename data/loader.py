"""
EdNet data loader for MARS project.

Loads EdNet KT2 interactions + metadata (questions, lectures),
computes derived features (correct, elapsed_time, changed_answer, response_count),
and returns a unified interactions DataFrame.

Supports stratified sampling by user activity quintiles for
representative subsets (important for publication-grade experiments).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# EdNet KT2 column schema
KT2_COLUMNS = ["timestamp", "action_type", "item_id", "source", "user_answer"]


class EdNetLoader:
    """Load and join EdNet KT2 data with question/lecture metadata."""

    def __init__(self, data_dir: str | Path = "data/raw"):
        self.data_dir = Path(data_dir)
        self._questions: Optional[pd.DataFrame] = None
        self._lectures: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def load_questions(self) -> pd.DataFrame:
        """Load questions.csv; parse semicolon-separated tags into lists."""
        path = self.data_dir / "questions.csv"
        logger.info("Loading questions from %s", path)
        df = pd.read_csv(path)

        # Normalise column names (EdNet uses 'question_id' already)
        df.columns = df.columns.str.strip().str.lower()

        # Parse tags: "tag1;tag2;tag3" → list[int]
        df["tags"] = (
            df["tags"]
            .fillna("")
            .apply(lambda s: [int(t) for t in s.split(";") if t.strip()])
        )

        # Rename 'part' → 'part_id' for consistency
        if "part" in df.columns and "part_id" not in df.columns:
            df = df.rename(columns={"part": "part_id"})

        self._questions = df
        logger.info("Loaded %d questions, %d unique tags", len(df), df["tags"].explode().nunique())
        return df

    def load_lectures(self) -> pd.DataFrame:
        """Load lectures.csv; parse tags."""
        path = self.data_dir / "lectures.csv"
        logger.info("Loading lectures from %s", path)
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()

        # Tags in lectures.csv can be a single int or semicolon-separated string
        def _parse_tags(val):
            if pd.isna(val):
                return []
            if isinstance(val, (int, float)):
                return [int(val)]
            return [int(t) for t in str(val).split(";") if t.strip()]

        df["tags"] = df["tags"].apply(_parse_tags)

        # Rename 'part' → 'part_id' for consistency
        if "part" in df.columns and "part_id" not in df.columns:
            df = df.rename(columns={"part": "part_id"})

        self._lectures = df
        logger.info("Loaded %d lectures", len(df))
        return df

    @property
    def questions(self) -> pd.DataFrame:
        if self._questions is None:
            self.load_questions()
        return self._questions  # type: ignore[return-value]

    @property
    def lectures(self) -> pd.DataFrame:
        if self._lectures is None:
            self.load_lectures()
        return self._lectures  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # KT2 interactions
    # ------------------------------------------------------------------

    def _list_kt2_users(self) -> list[str]:
        """Return sorted list of user CSV filenames inside KT2/."""
        kt2_dir = self.data_dir / "KT2"
        if not kt2_dir.exists():
            raise FileNotFoundError(f"KT2 directory not found: {kt2_dir}")
        files = sorted(kt2_dir.glob("*.csv"))
        return files

    def _load_single_user(self, path: Path) -> pd.DataFrame:
        """Load one user's KT2 CSV and return raw DataFrame."""
        user_id = path.stem  # filename without .csv
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        df["user_id"] = user_id
        return df

    def load_interactions(
        self,
        sample_users: Optional[int] = 50_000,
        random_seed: int = 42,
        stratified_sampling: bool = True,
    ) -> pd.DataFrame:
        """
        Load KT2 files, join with questions metadata, and compute derived features.

        Parameters
        ----------
        sample_users : int, optional
            Number of students to sample. Default 50,000 for publication-grade
            experiments. Use 1000 for quick debug runs. If None, load all.
        random_seed : int
            Seed for reproducible user sampling.
        stratified_sampling : bool
            If True, sample proportionally from 5 activity quintiles
            (by number of file lines) to ensure representativeness.
            If False, use uniform random sampling (legacy behaviour).

        Returns
        -------
        pd.DataFrame
            Columns: user_id, timestamp, question_id, bundle_id, part_id,
            tags (list), correct (bool), elapsed_time (ms),
            changed_answer (bool), response_count (int), source (str)
        """
        t_start = time.time()

        # Ensure questions are loaded (needed for correct_answer)
        questions = self.questions

        # Discover user files
        user_files = self._list_kt2_users()
        n_total = len(user_files)
        logger.info("Found %d user files in KT2/", n_total)

        if sample_users is not None and sample_users < n_total:
            if stratified_sampling:
                user_files = self._stratified_sample(
                    user_files, sample_users, random_seed
                )
            else:
                rng = np.random.RandomState(random_seed)
                idx = rng.choice(n_total, size=sample_users, replace=False)
                user_files = [user_files[i] for i in sorted(idx)]
            logger.info(
                "Sampled %d / %d users (%.1f%%) — stratified=%s",
                len(user_files), n_total,
                100 * len(user_files) / n_total, stratified_sampling,
            )

        # Load all selected users
        t_load = time.time()
        dfs = []
        n_skipped = 0
        for i, fpath in enumerate(user_files):
            try:
                dfs.append(self._load_single_user(fpath))
            except Exception as e:
                logger.warning("Skipping %s: %s", fpath.name, e)
                n_skipped += 1
            if (i + 1) % 5000 == 0:
                elapsed = time.time() - t_load
                logger.info(
                    "Loaded %d / %d users (%.1fs elapsed)",
                    i + 1, len(user_files), elapsed,
                )

        if n_skipped > 0:
            logger.warning(
                "Skipped %d / %d files (%.1f%%)",
                n_skipped, len(user_files), 100 * n_skipped / len(user_files),
            )

        raw = pd.concat(dfs, ignore_index=True)
        t_concat = time.time()
        logger.info(
            "Raw KT2 rows: %d from %d users (load: %.1fs)",
            len(raw), len(dfs), t_concat - t_load,
        )

        # Derive features from action sequences
        interactions = self._derive_features(raw, questions)
        t_end = time.time()
        logger.info(
            "Final interactions: %d rows, %d users "
            "(derive: %.1fs, total: %.1fs)",
            len(interactions), interactions["user_id"].nunique(),
            t_end - t_concat, t_end - t_start,
        )
        return interactions

    # ------------------------------------------------------------------
    # Stratified sampling
    # ------------------------------------------------------------------

    def _stratified_sample(
        self,
        user_files: list[Path],
        n_sample: int,
        seed: int,
    ) -> list[Path]:
        """
        Sample users proportionally from 5 activity quintiles.

        Activity is estimated by file size (proxy for number of
        interactions) to avoid reading every CSV.
        """
        # Estimate activity from file size (fast — no I/O reads)
        sizes = np.array([f.stat().st_size for f in user_files])
        quintiles = pd.qcut(sizes, q=5, labels=False, duplicates="drop")
        n_quintiles = int(quintiles.max()) + 1

        rng = np.random.RandomState(seed)
        selected_indices = []

        for q in range(n_quintiles):
            q_mask = np.where(quintiles == q)[0]
            # Proportional allocation
            q_n = max(1, int(round(n_sample * len(q_mask) / len(user_files))))
            q_n = min(q_n, len(q_mask))
            chosen = rng.choice(q_mask, size=q_n, replace=False)
            selected_indices.extend(chosen.tolist())
            logger.info(
                "Quintile %d: %d users available, sampled %d "
                "(file size range: %d–%d bytes)",
                q + 1, len(q_mask), q_n,
                int(sizes[q_mask].min()), int(sizes[q_mask].max()),
            )

        selected_indices.sort()
        result = [user_files[i] for i in selected_indices]
        logger.info(
            "Stratified sampling: requested %d, selected %d from %d quintiles",
            n_sample, len(result), n_quintiles,
        )
        return result

    # ------------------------------------------------------------------
    # Feature derivation
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_features(raw: pd.DataFrame, questions: pd.DataFrame) -> pd.DataFrame:
        """
        From raw KT2 action logs derive per-question interaction rows.

        KT2 action sequence per bundle:
          enter bXXXX → respond qXXXX (1+) → submit bXXXX

        For each encounter we compute:
        - correct: whether user_answer matches correct_answer
        - elapsed_time: timestamp(submit) - timestamp(enter)
        - changed_answer: more than 1 'respond' action before 'submit'
        - response_count: number of 'respond' actions before 'submit'
        """
        raw = raw.copy()
        raw["item_id"] = raw["item_id"].astype(str)

        # Sort chronologically per user
        raw = raw.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

        # --- Group actions into encounters per (user) ---
        # Each encounter starts with 'enter' (item_id starts with 'b').
        raw["is_enter"] = (raw["action_type"] == "enter") & raw["item_id"].str.startswith("b")
        raw["encounter_id"] = raw.groupby("user_id")["is_enter"].cumsum()

        # Build question_id: respond rows have 'qXXXX', enter/submit have 'bXXXX'
        q_mask = raw["item_id"].str.startswith("q")
        raw.loc[q_mask, "question_id"] = raw.loc[q_mask, "item_id"].str[1:].astype(int)

        logger.info("Aggregating encounters (vectorized)...")

        # ---- Vectorized aggregation per (user_id, encounter_id) ----
        groups = raw.groupby(["user_id", "encounter_id"])

        # Timestamps for elapsed_time
        enter_ts = raw[raw["action_type"] == "enter"].groupby(["user_id", "encounter_id"])["timestamp"].min()
        submit_ts = raw[raw["action_type"] == "submit"].groupby(["user_id", "encounter_id"])["timestamp"].max()

        # Respond-level stats
        respond_mask = raw["action_type"] == "respond"
        responds = raw[respond_mask]
        # Last respond per encounter (for user_answer)
        last_respond = responds.groupby(["user_id", "encounter_id"]).last()
        response_counts = responds.groupby(["user_id", "encounter_id"]).size()

        # Source from first row of each encounter
        sources = groups["source"].first()

        # Question_id from first respond in encounter
        question_ids = responds.groupby(["user_id", "encounter_id"])["question_id"].first()

        # Assemble
        interactions = pd.DataFrame({
            "ts_enter": enter_ts,
            "ts_submit": submit_ts,
            "question_id": question_ids,
            "user_answer": last_respond["user_answer"] if "user_answer" in last_respond.columns else np.nan,
            "response_count": response_counts,
            "source": sources,
        })

        interactions["elapsed_time"] = interactions["ts_submit"] - interactions["ts_enter"]
        interactions["timestamp"] = interactions["ts_submit"].fillna(interactions["ts_enter"])
        interactions["changed_answer"] = interactions["response_count"] > 1

        # Drop encounters without a valid timestamp or question
        interactions = interactions.dropna(subset=["timestamp", "question_id"]).reset_index()
        interactions["question_id"] = interactions["question_id"].astype(int)

        # --- Join with questions metadata ---
        # Build question_id column in metadata matching format (numeric)
        q_meta = questions[["question_id", "bundle_id", "correct_answer", "part_id", "tags"]].copy()
        # question_id in metadata is like "q123" — extract numeric
        q_meta["question_id"] = (
            q_meta["question_id"].astype(str).str.replace("q", "", regex=False).astype(int)
        )
        interactions = interactions.merge(q_meta, on="question_id", how="left")

        # Compute correctness
        interactions["correct"] = (
            interactions["user_answer"].astype(str).str.strip().str.lower()
            == interactions["correct_answer"].astype(str).str.strip().str.lower()
        )

        # Type conversions
        interactions["timestamp"] = interactions["timestamp"].astype(np.int64)
        interactions["changed_answer"] = interactions["changed_answer"].astype(bool)
        interactions["response_count"] = interactions["response_count"].fillna(0).astype(int)

        # Select and order final columns
        cols = [
            "user_id", "timestamp", "question_id", "bundle_id", "part_id",
            "tags", "correct", "elapsed_time", "changed_answer",
            "response_count", "source",
        ]
        interactions = interactions[[c for c in cols if c in interactions.columns]]

        return interactions.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
