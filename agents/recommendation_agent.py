"""
Recommendation Agent for MARS.

Combines three strategies via Thompson Sampling, then re-ranks
with LambdaMART (LightGBM) for the final recommendation list.

Strategies
----------
1. Knowledge-Based: gap analysis from KnowledgeGraphAgent
2. Content-Based: Sentence-BERT (MiniLM-L6-v2) + FAISS nearest neighbours
3. Collaborative: ALS matrix factorisation on user×tag accuracy

Selection: Thompson Sampling with Beta(α, β) posteriors.
Ranking: LightGBM lambdarank on 12 features per (user, item) pair.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

from .base_agent import BaseAgent
from .confidence_agent import (
    DEFAULT_CONFIDENCE_N_CLASSES,
    count_risk_confidence_events,
)

logger = logging.getLogger("mars.agent.recommendation")

# ──────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────

@dataclass
class Rec:
    """A single recommendation."""
    item_id: str
    item_type: str          # "lecture" or "question"
    score: float
    strategy: str           # which strategy produced this
    related_tags: list[int] = field(default_factory=list)


# ──────────────────────────────────────────────────────────
# Thompson Sampling priors
# ──────────────────────────────────────────────────────────

DEFAULT_STRATEGY_PRIORS = {
    "knowledge_based": {"alpha": 10, "beta": 1},
    "content_based":   {"alpha": 5,  "beta": 1},
    "collaborative":   {"alpha": 1,  "beta": 1},
}

LAMBDAMART_FEATURES = [
    "gap_by_tag",
    "user_accuracy_on_part",
    "avg_elapsed_time_tag",
    "changed_answer_rate",
    "tag_difficulty",
    "als_score",
    "lecture_coverage",
    "kg_score",
    "cb_score",
    "cf_score",
    "false_confidence_count",
    "prerequisite_completion",
]

CF_MIN_INTERACTIONS = 20  # minimum answers before enabling collaborative


class RecommendationAgent(BaseAgent):
    """
    Multi-strategy recommendation agent with Thompson Sampling
    exploration and LambdaMART re-ranking.
    """

    name = "recommendation"

    def __init__(
        self,
        random_seed: int | None = None,
        bandit_strategy: str | None = None,
    ) -> None:
        super().__init__()
        # Seed fixing
        _seed = random_seed if random_seed is not None else self._config.get("seed", 42)
        from .utils import set_global_seed
        set_global_seed(_seed)

        # Bandit strategy: "thompson", "ucb1", "epsilon_greedy", "fixed_kb", "fixed_round_robin"
        self._bandit_strategy = (
            bandit_strategy
            or self._config.get("bandit_strategy", "thompson")
        )
        self._round_robin_step: dict[str, int] = {}  # per-user step counter
        self._ucb_counts: dict[str, dict[str, int]] = {}
        self._ucb_rewards: dict[str, dict[str, float]] = {}
        self._ucb_total_steps: dict[str, int] = {}
        self._epsilon = float(self._config.get("epsilon", 0.1))

        # Config-driven parameters
        self._cf_min_interactions = self._config.get("cf_min_interactions", CF_MIN_INTERACTIONS)
        ts_cfg = self._config.get("thompson_sampling", {})
        self._strategy_priors = ts_cfg.get("priors", DEFAULT_STRATEGY_PRIORS)
        als_cfg = self._config.get("als", {})
        self._als_n_factors = als_cfg.get("n_factors", 64)
        sbert_cfg = self._config.get("sbert", {})
        self._sbert_model_name = sbert_cfg.get("model_name", "all-MiniLM-L6-v2")
        self._sbert_local_files_only = sbert_cfg.get("local_files_only", True)
        lm_cfg = self._config.get("lambdamart", {})
        self._lm_num_leaves = lm_cfg.get("num_leaves", 31)
        self._lm_min_child = lm_cfg.get("min_child_samples", 10)
        self._lm_lr = lm_cfg.get("learning_rate", 0.1)
        self._lm_n_estimators = lm_cfg.get("n_estimators", 100)
        self._n_confidence_classes = int(
            self._config.get("n_confidence_classes", DEFAULT_CONFIDENCE_N_CLASSES)
        )

        self._models_dir = Path("models")
        self._rng = np.random.RandomState(_seed)

        # Thompson Sampling state per user
        self._ts_params: dict[str, dict[str, dict[str, float]]] = {}

        # Content-based
        self._sbert_model = None
        self._faiss_index: faiss.IndexFlatIP | None = None
        self._item_embeddings: np.ndarray | None = None
        self._item_ids: list[str] = []
        self._item_meta: list[dict] = []

        # Collaborative
        self._als_user_factors: np.ndarray | None = None
        self._als_item_factors: np.ndarray | None = None
        self._als_user_map: dict[str, int] = {}
        self._als_tag_map: dict[int, int] = {}
        self._als_tag_reverse: dict[int, int] = {}

        # User profiles (for scoring)
        self._user_profiles: dict[str, dict] = {}

        # LambdaMART ranker
        self._ranker: lgb.LGBMRanker | None = None

        # KG agent reference (set by orchestrator)
        self._kg_agent = None

    def set_confidence_schema(self, n_classes: int) -> None:
        """Update the active confidence schema for downstream ranking features."""
        self._n_confidence_classes = int(max(2, n_classes))

    # ──────────────────────────────────────────────────────
    # BaseAgent interface
    # ──────────────────────────────────────────────────────

    def initialize(self, **kwargs: Any) -> None:
        if "lectures_df" in kwargs:
            self.build_content_index(kwargs["lectures_df"], kwargs.get("questions_df"))
        if "interactions_df" in kwargs:
            self.train_collaborative(kwargs["interactions_df"])
        if "kg_agent" in kwargs:
            self._kg_agent = kwargs["kg_agent"]

    # ──────────────────────────────────────────────────────
    # 1. Knowledge-Based
    # ──────────────────────────────────────────────────────

    def get_knowledge_based(
        self,
        user_id: str,
        kg_profile: dict | None = None,
        n: int = 10,
        **kwargs: Any,
    ) -> list[Rec]:
        """
        Gap-driven recommendations via the Knowledge Graph.

        Uses gap_tags from KG profile, finds lectures covering gaps,
        prioritises prerequisites.
        """
        if kg_profile is None:
            return []

        recs_raw = kg_profile.get("recommendations", [])
        results = []
        for r in recs_raw[:n]:
            results.append(Rec(
                item_id=r.get("item_id", ""),
                item_type=r.get("item_type", "lecture"),
                score=float(r.get("priority", 1.0)),
                strategy="knowledge_based",
                related_tags=r.get("related_tags", []),
            ))

        return results

    # ──────────────────────────────────────────────────────
    # 2. Content-Based (Sentence-BERT + FAISS)
    # ──────────────────────────────────────────────────────

    def build_content_index(
        self,
        lectures_df: pd.DataFrame,
        questions_df: pd.DataFrame | None = None,
    ) -> None:
        """
        Build FAISS index from lecture/question text descriptions
        encoded with MiniLM-L6-v2.
        """
        from sentence_transformers import SentenceTransformer

        logger.info("Loading %s for content-based encoding ...", self._sbert_model_name)
        self._sbert_model = SentenceTransformer(
            self._sbert_model_name,
            local_files_only=self._sbert_local_files_only,
        )

        # Build descriptions for lectures
        items = []
        descriptions = []

        l_df = lectures_df.copy()
        part_col = "part_id" if "part_id" in l_df.columns else "part"

        def _parse_tags(val):
            if isinstance(val, list):
                return val
            if isinstance(val, (int, float)) and not pd.isna(val):
                return [int(val)]
            if isinstance(val, str):
                return [int(t) for t in val.split(";") if t.strip()]
            return []

        for _, row in l_df.iterrows():
            lid = str(row["lecture_id"])
            pid = int(row[part_col]) if pd.notna(row[part_col]) else 0
            tags = _parse_tags(row.get("tags", []))
            desc = f"Lecture {lid}, Part: {pid}, Tags: {tags}"
            items.append({"item_id": lid, "item_type": "lecture",
                          "part_id": pid, "tags": tags})
            descriptions.append(desc)

        if questions_df is not None:
            q_df = questions_df.copy()
            qpart_col = "part_id" if "part_id" in q_df.columns else "part"
            # Only add a sample of questions (too many for full index)
            sample = q_df.sample(min(2000, len(q_df)), random_state=42)
            for _, row in sample.iterrows():
                qid = str(row["question_id"])
                pid = int(row[qpart_col])
                tags = _parse_tags(row.get("tags", []))
                diff = row.get("difficulty", 0.5)
                desc = f"Question {qid}, Part: {pid}, Difficulty: {diff:.2f}, Tags: {tags}"
                items.append({"item_id": qid, "item_type": "question",
                              "part_id": pid, "tags": tags})
                descriptions.append(desc)

        # Encode
        logger.info("Encoding %d items with MiniLM ...", len(descriptions))
        embeddings = self._sbert_model.encode(descriptions, show_progress_bar=False,
                                               normalize_embeddings=True)
        self._item_embeddings = embeddings.astype(np.float32)
        self._item_ids = [it["item_id"] for it in items]
        self._item_meta = items

        # Build FAISS index (inner product = cosine on normalised vectors)
        dim = embeddings.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dim)
        self._faiss_index.add(self._item_embeddings)

        logger.info("FAISS index built: %d items, dim=%d", len(items), dim)

    def get_content_based(
        self,
        user_id: str,
        gap_tags: list[int] | None = None,
        n: int = 10,
        **kwargs: Any,
    ) -> list[Rec]:
        """
        Find items similar to the user's gap profile via FAISS.

        Builds a query embedding from gap tag descriptions, then
        retrieves nearest neighbours.
        """
        if self._faiss_index is None or self._sbert_model is None:
            return []

        # Build query from gap tags
        if gap_tags:
            query_text = f"Student needs help with tags: {gap_tags}"
        else:
            query_text = "General TOEIC practice across all parts"

        query_vec = self._sbert_model.encode([query_text], normalize_embeddings=True)
        query_vec = query_vec.astype(np.float32)

        # Search
        k = min(n * 3, self._faiss_index.ntotal)  # oversample then dedup
        scores, indices = self._faiss_index.search(query_vec, k)

        results = []
        seen = set()
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._item_ids):
                continue
            iid = self._item_ids[idx]
            if iid in seen:
                continue
            seen.add(iid)
            meta = self._item_meta[idx]
            results.append(Rec(
                item_id=iid,
                item_type=meta["item_type"],
                score=float(score),
                strategy="content_based",
                related_tags=meta.get("tags", []),
            ))
            if len(results) >= n:
                break

        return results

    # ──────────────────────────────────────────────────────
    # 3. Collaborative Filtering (ALS via SVD)
    # ──────────────────────────────────────────────────────

    def train_collaborative(
        self,
        interactions_df: pd.DataFrame,
        n_factors: int = 64,
    ) -> None:
        """
        Train ALS-style collaborative filtering on user×tag accuracy matrix.

        Uses TruncatedSVD on the sparse accuracy matrix as a lightweight
        ALS approximation suitable for the research prototype.
        """
        df = interactions_df.copy()
        if "tags" not in df.columns:
            logger.warning("No tags column — skipping collaborative training")
            return

        # Parse tags
        if df["tags"].dtype == object and isinstance(df["tags"].iloc[0], str):
            df["tags"] = df["tags"].apply(
                lambda s: [int(t) for t in str(s).split(";") if t.strip()]
            )

        # Build user × tag accuracy matrix
        records = []
        for _, row in df.iterrows():
            uid = str(row["user_id"])
            tags = row["tags"] if isinstance(row["tags"], list) else []
            correct = float(row["correct"])
            for tid in tags:
                records.append((uid, tid, correct))

        if not records:
            return

        rec_df = pd.DataFrame(records, columns=["user_id", "tag_id", "correct"])
        agg = rec_df.groupby(["user_id", "tag_id"])["correct"].mean().reset_index()

        # Create mappings
        users = sorted(agg["user_id"].unique())
        tags = sorted(agg["tag_id"].unique())
        self._als_user_map = {u: i for i, u in enumerate(users)}
        self._als_tag_map = {t: i for i, t in enumerate(tags)}
        self._als_tag_reverse = {i: t for t, i in self._als_tag_map.items()}

        # Sparse matrix
        rows = [self._als_user_map[u] for u in agg["user_id"]]
        cols = [self._als_tag_map[t] for t in agg["tag_id"]]
        vals = agg["correct"].values
        mat = sp.csr_matrix((vals, (rows, cols)),
                            shape=(len(users), len(tags)))

        # SVD (ALS-equivalent factorisation)
        n_components = min(n_factors, min(mat.shape) - 1)
        if n_components < 2:
            logger.warning("Too few components for SVD — skipping")
            return

        svd = TruncatedSVD(n_components=n_components, random_state=42)
        user_factors = svd.fit_transform(mat)  # (n_users, n_factors)
        item_factors = svd.components_.T       # (n_tags, n_factors)

        self._als_user_factors = user_factors
        self._als_item_factors = item_factors

        logger.info("Collaborative model trained: %d users × %d tags, %d factors",
                     len(users), len(tags), n_components)

    def get_collaborative(
        self,
        user_id: str,
        n: int = 10,
        **kwargs: Any,
    ) -> list[Rec]:
        """
        Get collaborative filtering recommendations based on similar users.
        """
        if self._als_user_factors is None or user_id not in self._als_user_map:
            return []

        uid_idx = self._als_user_map[user_id]
        user_vec = self._als_user_factors[uid_idx]

        # Score all tags
        scores = self._als_item_factors @ user_vec  # (n_tags,)
        top_indices = np.argsort(scores)[::-1][:n]

        results = []
        for idx in top_indices:
            tag_id = self._als_tag_reverse.get(int(idx))
            if tag_id is None:
                continue
            results.append(Rec(
                item_id=f"tag_{tag_id}",
                item_type="tag",
                score=float(scores[idx]),
                strategy="collaborative",
                related_tags=[tag_id],
            ))

        return results

    # ──────────────────────────────────────────────────────
    # 4. Thompson Sampling
    # ──────────────────────────────────────────────────────

    def _get_ts_params(self, user_id: str) -> dict[str, dict[str, float]]:
        if user_id not in self._ts_params:
            import copy
            self._ts_params[user_id] = copy.deepcopy(DEFAULT_STRATEGY_PRIORS)
        return self._ts_params[user_id]

    def select_strategy(self, user_id: str, n_interactions: int = 0) -> str:
        """
        Select a recommendation strategy using the configured bandit policy.

        Supported policies:
          - "thompson"         — Thompson Sampling with Beta(α, β) posteriors
          - "ucb1"             — UCB1: mean + sqrt(2·ln(t) / n_i)
          - "epsilon_greedy"   — ε-greedy (ε=0.1): random 10% of the time
          - "fixed_kb"         — always knowledge-based (no exploration)
          - "fixed_round_robin" — KB → CB → CF → KB → … (deterministic)

        Collaborative is only eligible when n_interactions >= CF_MIN_INTERACTIONS.
        """
        eligible = ["knowledge_based", "content_based"]
        if n_interactions >= self._cf_min_interactions:
            eligible.append("collaborative")

        bs = self._bandit_strategy

        # ── Fixed: always knowledge-based ──
        if bs == "fixed_kb":
            return "knowledge_based"

        # ── Fixed: round-robin ──
        if bs == "fixed_round_robin":
            step = self._round_robin_step.get(user_id, 0)
            strategy = eligible[step % len(eligible)]
            self._round_robin_step[user_id] = step + 1
            return strategy

        # ── ε-greedy ──
        if bs == "epsilon_greedy":
            if self._rng.random() < self._epsilon:
                return eligible[self._rng.randint(len(eligible))]
            # Exploit: pick strategy with highest average reward
            params = self._get_ts_params(user_id)
            best_s, best_v = eligible[0], -1.0
            for s in eligible:
                ab = params.get(s, {"alpha": 1, "beta": 1})
                avg = ab["alpha"] / (ab["alpha"] + ab["beta"])
                if avg > best_v:
                    best_v = avg
                    best_s = s
            return best_s

        # ── UCB1 ──
        if bs == "ucb1":
            if user_id not in self._ucb_counts:
                self._ucb_counts[user_id] = {s: 0 for s in eligible}
                self._ucb_rewards[user_id] = {s: 0.0 for s in eligible}
                self._ucb_total_steps[user_id] = 0

            counts = self._ucb_counts[user_id]
            rewards = self._ucb_rewards[user_id]
            total = self._ucb_total_steps[user_id] + 1

            # Ensure new strategies get at least one try
            for s in eligible:
                if counts.get(s, 0) == 0:
                    return s

            best_s, best_v = eligible[0], -1.0
            for s in eligible:
                n_i = counts.get(s, 1)
                avg = rewards.get(s, 0.0) / max(n_i, 1)
                exploration = np.sqrt(2.0 * np.log(total) / n_i)
                v = avg + exploration
                if v > best_v:
                    best_v = v
                    best_s = s
            return best_s

        # ── Thompson Sampling (default) ──
        params = self._get_ts_params(user_id)
        best_strategy = "knowledge_based"
        best_sample = -1.0

        for strategy, ab in params.items():
            if strategy not in eligible:
                continue
            sample = self._rng.beta(ab["alpha"], ab["beta"])
            if sample > best_sample:
                best_sample = sample
                best_strategy = strategy

        return best_strategy

    def update_reward(
        self,
        user_id: str,
        strategy: str,
        reward: float,
        **kwargs: Any,
    ) -> None:
        """
        Update bandit posterior / statistics for the selected strategy.

        reward=1 if the recommended tag accuracy improved, else 0.
        Works for all bandit policies (TS posteriors, UCB1 counts, ε-greedy averages).
        """
        # Thompson Sampling / ε-greedy (share the same Beta params)
        params = self._get_ts_params(user_id)
        if strategy in params:
            params[strategy]["alpha"] += reward
            params[strategy]["beta"] += (1.0 - reward)
            logger.debug("TS update %s/%s: alpha=%.1f, beta=%.1f",
                         user_id, strategy, params[strategy]["alpha"], params[strategy]["beta"])

        # UCB1 counts
        if user_id in self._ucb_counts:
            self._ucb_counts[user_id][strategy] = self._ucb_counts[user_id].get(strategy, 0) + 1
            self._ucb_rewards[user_id][strategy] = self._ucb_rewards[user_id].get(strategy, 0.0) + reward
            self._ucb_total_steps[user_id] = self._ucb_total_steps.get(user_id, 0) + 1

    def get_ts_weights(self, user_id: str) -> dict[str, float]:
        """Return current expected weight for each strategy."""
        params = self._get_ts_params(user_id)
        weights = {}
        for s, ab in params.items():
            weights[s] = ab["alpha"] / (ab["alpha"] + ab["beta"])
        return weights

    # ──────────────────────────────────────────────────────
    # 5. LambdaMART Re-ranking
    # ──────────────────────────────────────────────────────

    def train_ranker(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        train_groups: list[int],
        val_features: np.ndarray | None = None,
        val_labels: np.ndarray | None = None,
        val_groups: list[int] | None = None,
    ) -> lgb.LGBMRanker:
        """
        Train LambdaMART ranker on (user, item) feature pairs.

        12 features per pair as defined in LAMBDAMART_FEATURES.
        """
        self._ranker = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            eval_at=[5, 10],
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=10,
            random_state=42,
            verbosity=-1,
        )

        callbacks = [lgb.log_evaluation(period=100)]
        eval_set = []
        eval_group = []
        eval_names = []
        if val_features is not None and val_labels is not None and val_groups is not None:
            eval_set = [(val_features, val_labels)]
            eval_group = [val_groups]
            eval_names = ["val"]
            callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=True))

        self._ranker.fit(
            train_features, train_labels,
            group=train_groups,
            eval_set=eval_set if eval_set else None,
            eval_group=eval_group if eval_group else None,
            eval_names=eval_names if eval_names else None,
            callbacks=callbacks,
        )

        # Save
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._ranker.booster_.save_model(str(self._models_dir / "lambdamart.txt"))
        logger.info("LambdaMART ranker trained and saved")

        return self._ranker

    def _build_ranking_features(
        self,
        user_id: str,
        candidates: list[Rec],
        user_profile: dict | None = None,
    ) -> np.ndarray:
        """
        Build the 12-feature vector for each (user, candidate) pair.
        """
        profile = user_profile or self._user_profiles.get(user_id, {})
        n = len(candidates)
        feats = np.zeros((n, 12), dtype=np.float32)

        for i, cand in enumerate(candidates):
            tags = cand.related_tags
            tag_id = tags[0] if tags else -1

            # 1. gap_by_tag: 1 if tag is in user's gaps
            feats[i, 0] = 1.0 if tag_id in profile.get("gap_tags", []) else 0.0
            # 2. user_accuracy_on_part
            feats[i, 1] = profile.get("part_accuracy", {}).get(str(tag_id), 0.5)
            # 3. avg_elapsed_time_tag (normalised)
            feats[i, 2] = profile.get("tag_elapsed", {}).get(str(tag_id), 0.5)
            # 4. changed_answer_rate
            feats[i, 3] = profile.get("changed_rate", 0.1)
            # 5. tag_difficulty
            feats[i, 4] = profile.get("tag_difficulty", {}).get(str(tag_id), 0.5)
            # 6. als_score (from collaborative)
            feats[i, 5] = cand.score if cand.strategy == "collaborative" else 0.0
            # 7. lecture_coverage: how many tags does this lecture cover
            feats[i, 6] = min(len(tags) / 5.0, 1.0)
            # 8. kg_score
            feats[i, 7] = cand.score if cand.strategy == "knowledge_based" else 0.0
            # 9. cb_score (content-based)
            feats[i, 8] = cand.score if cand.strategy == "content_based" else 0.0
            # 10. cf_score (collaborative)
            feats[i, 9] = cand.score if cand.strategy == "collaborative" else 0.0
            # 11. false_confidence_count
            feats[i, 10] = profile.get("false_confidence_count", 0) / 10.0
            # 12. prerequisite_completion
            feats[i, 11] = profile.get("prereq_completion", {}).get(str(tag_id), 0.5)

        return feats

    def rank_candidates(
        self,
        user_id: str,
        candidates: list[Rec],
        user_profile: dict | None = None,
    ) -> list[Rec]:
        """
        Re-rank candidates using LambdaMART or fall back to score sorting.
        """
        if not candidates:
            return []

        if self._ranker is not None:
            feats = self._build_ranking_features(user_id, candidates, user_profile)
            scores = self._ranker.predict(feats)
            for i, s in enumerate(scores):
                candidates[i].score = float(s)

        candidates.sort(key=lambda r: r.score, reverse=True)
        return candidates

    # ──────────────────────────────────────────────────────
    # 6. Main recommend()
    # ──────────────────────────────────────────────────────

    def recommend(
        self,
        user_id: str,
        kg_profile: dict | None = None,
        confidence: dict | None = None,
        predictions: dict | None = None,
        n: int = 10,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate recommendations using Thompson Sampling strategy selection
        and optional LambdaMART re-ranking.

        Returns dict compatible with orchestrator pipelines.
        """
        self._set_processing()

        # Count user's interactions for CF eligibility
        n_interactions = self._user_profiles.get(user_id, {}).get("n_interactions", 0)

        # Select strategy via Thompson Sampling
        strategy = self.select_strategy(user_id, n_interactions=n_interactions)
        logger.debug("Strategy for %s: %s", user_id, strategy)

        # Gather candidates from all strategies (primary gets more slots)
        candidates: list[Rec] = []

        # Knowledge-based (always available)
        kb_recs = self.get_knowledge_based(user_id, kg_profile=kg_profile, n=n)
        candidates.extend(kb_recs)

        # Content-based
        gap_tags = (kg_profile or {}).get("gap_tags", [])
        cb_recs = self.get_content_based(user_id, gap_tags=gap_tags, n=n)
        candidates.extend(cb_recs)

        # Collaborative (if eligible)
        if n_interactions >= CF_MIN_INTERACTIONS:
            cf_recs = self.get_collaborative(user_id, n=n)
            candidates.extend(cf_recs)

        # Boost scores for the selected strategy
        for c in candidates:
            if c.strategy == strategy:
                c.score *= 1.5

        # Deduplicate by item_id (keep highest score)
        seen: dict[str, Rec] = {}
        for c in candidates:
            if c.item_id not in seen or c.score > seen[c.item_id].score:
                seen[c.item_id] = c
        candidates = list(seen.values())

        # Re-rank
        ranked = self.rank_candidates(user_id, candidates)[:n]

        self._set_idle()

        return {
            "user_id": user_id,
            "items": [{"item_id": r.item_id, "item_type": r.item_type,
                        "score": round(r.score, 4), "strategy": r.strategy,
                        "related_tags": r.related_tags}
                       for r in ranked],
            "strategy_selected": strategy,
            "strategy_weights": self.get_ts_weights(user_id),
            "n_candidates": len(candidates),
        }

    def rerank(
        self,
        user_id: str,
        confidence: dict | None = None,
        prediction: dict | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Re-rank existing recommendations (continuous pipeline)."""
        # For now, delegate to recommend with cached profile
        return self.recommend(user_id=user_id, **kwargs)

    # ──────────────────────────────────────────────────────
    # User profile management
    # ──────────────────────────────────────────────────────

    def update_user_profile(
        self,
        user_id: str,
        interactions: pd.DataFrame | None = None,
        confidence_result: dict | None = None,
    ) -> None:
        """Cache user statistics for ranking features."""
        if interactions is None or len(interactions) == 0:
            return

        profile = self._user_profiles.get(user_id, {
            "n_interactions": 0,
            "gap_tags": [],
            "part_accuracy": {},
            "tag_elapsed": {},
            "changed_rate": 0.1,
            "tag_difficulty": {},
            "false_confidence_count": 0,
            "prereq_completion": {},
        })

        profile["n_interactions"] = len(interactions)
        profile["changed_rate"] = float(interactions["changed_answer"].mean()) if "changed_answer" in interactions.columns else 0.1

        if "part_id" in interactions.columns:
            pa = interactions.groupby("part_id")["correct"].mean()
            profile["part_accuracy"] = {str(k): round(float(v), 3) for k, v in pa.items()}

        if confidence_result and "class_names" in confidence_result:
            profile["false_confidence_count"] = count_risk_confidence_events(
                confidence_result["class_names"],
                n_classes=confidence_result.get("n_classes", self._n_confidence_classes),
            )

        self._user_profiles[user_id] = profile
