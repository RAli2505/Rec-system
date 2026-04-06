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
        _seed = random_seed if random_seed is not None else self.global_seed
        from .utils import set_global_seed
        set_global_seed(_seed)
        self._seed = _seed

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

        # KG agent reference (set by orchestrator)
        self._kg_agent = None

        # IRT item difficulty map: item_id/tag_id → difficulty (b parameter)
        self._item_difficulty: dict[str, float] = {}

        # Ablation switches (config-driven)
        abl = self._config.get("ablation", {})
        self._use_prediction_boost = abl.get("use_prediction_boost", True)
        self._prediction_boost_weight = float(abl.get("prediction_boost_weight", 0.25))
        self._use_mmr = abl.get("use_mmr", True)
        self._mmr_lambda = float(abl.get("mmr_lambda", 0.80))
        self._use_zpd_bonus = abl.get("use_zpd_bonus", True)
        self._use_learner_level = abl.get("use_learner_level", True)

    def set_irt_params(self, item_params: dict[str, float]) -> None:
        """Set IRT difficulty parameters for ZPD filtering.

        Parameters
        ----------
        item_params : dict
            Mapping of item_id or tag_id → difficulty (IRT b parameter).
        """
        self._item_difficulty = {str(k): float(v) for k, v in item_params.items()}
        logger.info("Set IRT params for %d items", len(self._item_difficulty))

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
            # Index more questions for better coverage and item-level matches
            sample = q_df.sample(min(10000, len(q_df)), random_state=self._seed)
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

        svd = TruncatedSVD(n_components=n_components, random_state=self._seed)
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

        Maps top predicted tags to actual items in the content index
        (questions/lectures) so that recommendations are concrete items
        rather than abstract tag IDs.
        """
        if self._als_user_factors is None or user_id not in self._als_user_map:
            return []

        uid_idx = self._als_user_map[user_id]
        user_vec = self._als_user_factors[uid_idx]

        # Score all tags
        scores = self._als_item_factors @ user_vec  # (n_tags,)
        top_indices = np.argsort(scores)[::-1][:n * 3]  # oversample to find items

        results = []
        seen_items: set = set()
        for idx in top_indices:
            tag_id = self._als_tag_reverse.get(int(idx))
            if tag_id is None:
                continue
            cf_score = float(scores[idx])

            # Try to map tag to actual items in the content index
            mapped = False
            if self._item_meta:
                for item_idx, meta in enumerate(self._item_meta):
                    if tag_id in meta.get("tags", []):
                        iid = self._item_ids[item_idx]
                        if iid not in seen_items:
                            seen_items.add(iid)
                            results.append(Rec(
                                item_id=iid,
                                item_type=meta["item_type"],
                                score=cf_score,
                                strategy="collaborative",
                                related_tags=meta.get("tags", [tag_id]),
                            ))
                            mapped = True
                            if len(results) >= n:
                                break
                if len(results) >= n:
                    break

            # Fallback: emit tag-level rec if no items match
            if not mapped:
                results.append(Rec(
                    item_id=f"tag_{tag_id}",
                    item_type="tag",
                    score=cf_score,
                    strategy="collaborative",
                    related_tags=[tag_id],
                ))

            if len(results) >= n:
                break

        return results[:n]

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
    # 5. Weighted Linear Scoring (interpretable ranking)
    # ──────────────────────────────────────────────────────
    #
    # Replaces LambdaMART with a transparent, theory-grounded formula:
    #   score = w_gap * gap_relevance       (Vygotsky ZPD — target gaps)
    #         + w_diff * difficulty_match    (item–ability proximity)
    #         + w_recency * recency_need     (Ebbinghaus spacing effect)
    #         + w_strategy * strategy_score  (strategy-specific relevance)
    #
    # Weights are interpretable and defensible in a dissertation.

    # Default weights (can be overridden via config)
    # Tuned for tag-based relevance evaluation:
    #   - gap_relevance drives Precision/Recall (hit rate on failed tags)
    #   - difficulty_match improves NDCG (right items ranked higher)
    #   - strategy_score preserves signal from CB/CF/KB models
    LINEAR_WEIGHTS = {
        "gap_relevance": 0.40,       # target known gaps — primary signal for recall
        "difficulty_match": 0.10,    # ZPD proximity (lower: noisy without per-user IRT)
        "recency_need": 0.05,        # spacing / forgetting curve (low: rarely available)
        "strategy_score": 0.45,      # CB/CF/KB original score — strongest reliable signal
    }

    def rank_candidates(
        self,
        user_id: str,
        candidates: list[Rec],
        user_profile: dict | None = None,
    ) -> list[Rec]:
        """
        Re-rank candidates using weighted linear scoring.

        Interpretable ranking formula with 4 theory-grounded components:
        - gap_relevance: 1 if item targets a known gap tag, 0 otherwise
        - difficulty_match: 1 − |item_difficulty − user_ability| / 4
        - recency_need: inverse of how recently the tag was practiced
        - strategy_score: normalized original score from CB/CF/KB strategy
        """
        if not candidates:
            return []

        profile = user_profile or self._user_profiles.get(user_id, {})
        gap_tags = set(profile.get("gap_tags", []))
        user_theta = profile.get("theta", 0.0)
        tag_difficulty = profile.get("tag_difficulty", {})
        tag_last_seen = profile.get("tag_last_seen", {})  # timestamp of last practice

        w = {
            "gap_relevance": 0.50,
            "difficulty_match": 0.15,
            "recency_need": 0.10,
            "strategy_score": 0.25,
        }

        # Normalize strategy scores to [0, 1]
        raw_scores = [c.score for c in candidates]
        max_raw = max(raw_scores) if raw_scores else 1.0
        min_raw = min(raw_scores) if raw_scores else 0.0
        score_range = max_raw - min_raw if max_raw > min_raw else 1.0

        for c in candidates:
            tag_id = c.related_tags[0] if c.related_tags else -1

            # 1. Gap relevance: does this item target a known gap?
            gap_rel = 1.0 if tag_id in gap_tags else 0.0
            # Bonus: if ANY of the item's tags is a gap
            if not gap_rel and c.related_tags:
                gap_rel = len(set(c.related_tags) & gap_tags) / len(c.related_tags)

            # 2. Difficulty match: how close is item difficulty to user ability?
            diff = float(tag_difficulty.get(str(tag_id), 0.0))
            diff_match = max(0.0, 1.0 - abs(diff - user_theta) / 4.0)

            # 3. Recency need: tags not practiced recently score higher
            last_seen = tag_last_seen.get(str(tag_id), 0)
            if last_seen > 0:
                # Normalize: more time since last → higher need (capped at 1)
                recency = min(1.0, last_seen / 1_000_000)  # rough normalization
            else:
                recency = 1.0  # never practiced → high need

            # 4. Strategy score (normalized)
            strat_score = (c.score - min_raw) / score_range

            # Weighted linear combination
            c.score = (
                w["gap_relevance"] * gap_rel
                + w["difficulty_match"] * diff_match
                + w["recency_need"] * recency
                + w["strategy_score"] * strat_score
            )

        candidates.sort(key=lambda r: r.score, reverse=True)
        return candidates

    # ──────────────────────────────────────────────────────
    # 5b. MMR Diversification
    # ──────────────────────────────────────────────────────

    @staticmethod
    def _tag_similarity(a: Rec, b: Rec) -> float:
        """Jaccard similarity between two items based on tags."""
        tags_a = set(a.related_tags)
        tags_b = set(b.related_tags)
        if not tags_a or not tags_b:
            return 0.0
        return len(tags_a & tags_b) / len(tags_a | tags_b)

    def _mmr_rerank(
        self,
        candidates: list[Rec],
        n: int = 10,
        lam: float = 0.7,
    ) -> list[Rec]:
        """
        Maximal Marginal Relevance re-ranking.

        Greedily selects items that maximize:
            λ * relevance(item) - (1-λ) * max_similarity(item, selected)

        This increases tag diversity while preserving relevance.
        """
        if len(candidates) <= n:
            return candidates

        # Normalize scores to [0, 1]
        scores = [c.score for c in candidates]
        max_score = max(scores) if scores else 1.0
        min_score = min(scores) if scores else 0.0
        score_range = max_score - min_score if max_score > min_score else 1.0

        selected: list[Rec] = []
        remaining = list(candidates)

        for _ in range(n):
            if not remaining:
                break

            best_idx = 0
            best_mmr = -float("inf")

            for i, cand in enumerate(remaining):
                rel = (cand.score - min_score) / score_range

                if selected:
                    max_sim = max(self._tag_similarity(cand, s) for s in selected)
                else:
                    max_sim = 0.0

                mmr = lam * rel - (1 - lam) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

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
        learner_level: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate recommendations using Thompson Sampling strategy selection
        and weighted linear re-ranking.

        Parameters
        ----------
        learner_level : str, optional
            Learner level from PersonalizationAgent (e.g. "beginner",
            "intermediate", "advanced"). Used to adjust scoring weights.

        Returns dict compatible with orchestrator pipelines.
        """
        self._set_processing()

        profile = self._user_profiles.setdefault(user_id, {})
        if kg_profile:
            if "gap_tags" in kg_profile:
                profile["gap_tags"] = list(dict.fromkeys(kg_profile.get("gap_tags", [])))
            if "theta" in kg_profile:
                profile["theta"] = float(kg_profile.get("theta", 0.0))

        if predictions:
            pred_gap_tags = [
                int(g["tag_id"])
                for g in predictions.get("gaps", [])
                if isinstance(g, dict) and "tag_id" in g
            ]
            if pred_gap_tags:
                existing_gap_tags = list(profile.get("gap_tags", []))
                profile["gap_tags"] = list(dict.fromkeys(existing_gap_tags + pred_gap_tags))
            profile["predicted_gap_probabilities"] = predictions.get("gap_probabilities", [])

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
        gap_tags = profile.get("gap_tags", (kg_profile or {}).get("gap_tags", []))
        cb_recs = self.get_content_based(user_id, gap_tags=gap_tags, n=n)
        candidates.extend(cb_recs)

        # Collaborative (if eligible)
        if n_interactions >= self._cf_min_interactions:
            cf_recs = self.get_collaborative(user_id, n=n)
            candidates.extend(cf_recs)

        # Boost scores for the selected strategy
        for c in candidates:
            if c.strategy == strategy:
                c.score *= 1.5

        # Directly inject PredictionAgent signal into candidate scores.
        if self._use_prediction_boost:
            gap_probs = profile.get("predicted_gap_probabilities", []) or []
            if gap_probs:
                for c in candidates:
                    if c.related_tags:
                        pred_scores = [
                            float(gap_probs[int(t)])
                            for t in c.related_tags
                            if isinstance(t, (int, np.integer)) and 0 <= int(t) < len(gap_probs)
                        ]
                        if pred_scores:
                            c.score += self._prediction_boost_weight * max(pred_scores)

        # ── Diagnostic: per-strategy candidate counts before dedup ──
        pre_dedup_total = len(candidates)
        strategy_counts = defaultdict(int)
        for c in candidates:
            strategy_counts[c.strategy] += 1

        # Deduplicate by item_id (keep highest score)
        seen: dict[str, Rec] = {}
        for c in candidates:
            if c.item_id not in seen or c.score > seen[c.item_id].score:
                seen[c.item_id] = c
        candidates = list(seen.values())
        post_dedup_total = len(candidates)
        dedup_overlap = pre_dedup_total - post_dedup_total

        logger.debug(
            "Candidates for %s: pre_dedup=%d, post_dedup=%d, overlap=%d, "
            "by_strategy={KB=%d, CB=%d, CF=%d}",
            user_id, pre_dedup_total, post_dedup_total, dedup_overlap,
            strategy_counts.get("knowledge_based", 0),
            strategy_counts.get("content_based", 0),
            strategy_counts.get("collaborative", 0),
        )

        # ZPD soft scoring: smooth Gaussian bonus centred on delta=+0.5
        # (slightly above student level = ideal challenge).
        # No penalty — items outside ZPD keep their original score.
        user_theta = float((kg_profile or {}).get("theta", 0.0))
        if self._use_zpd_bonus and self._item_difficulty:
            import math
            zpd_centre = 0.5   # ideal delta (difficulty - theta)
            zpd_sigma = 1.5    # wide bell — very soft
            max_bonus = 0.15   # small bonus, never dominates
            for c in candidates:
                tag_id = c.related_tags[0] if c.related_tags else -1
                difficulty = self._item_difficulty.get(str(tag_id),
                             self._item_difficulty.get(c.item_id, 0.0))
                delta = difficulty - user_theta
                bonus = max_bonus * math.exp(-0.5 * ((delta - zpd_centre) / zpd_sigma) ** 2)
                c.score += bonus

        # Learner-level personalization: adjust item type preference
        if self._use_learner_level and learner_level and candidates:
            level_lower = str(learner_level).lower()
            for c in candidates:
                if level_lower in ("struggling", "developing"):
                    # Prefer lectures for weaker learners (build foundation)
                    if c.item_type == "lecture":
                        c.score += 0.10
                elif level_lower in ("improving", "advanced"):
                    # Prefer questions for stronger learners (challenge)
                    if c.item_type == "question":
                        c.score += 0.05

        # Re-rank via weighted linear scoring
        ranked = self.rank_candidates(user_id, candidates)

        # ── Diagnostic: pre-MMR tag diversity ──
        pre_mmr_tags = set()
        for r in ranked[:n]:
            pre_mmr_tags.update(r.related_tags)

        # MMR diversification: greedily select items balancing relevance + diversity
        if self._use_mmr:
            ranked = self._mmr_rerank(ranked, n=n, lam=self._mmr_lambda)

        # ── Diagnostic: post-MMR tag diversity ──
        post_mmr_tags = set()
        for r in ranked[:n]:
            post_mmr_tags.update(r.related_tags)
        if self._use_mmr:
            logger.debug(
                "MMR for %s: pre_tags=%d, post_tags=%d, diversity_delta=%+d",
                user_id, len(pre_mmr_tags), len(post_mmr_tags),
                len(post_mmr_tags) - len(pre_mmr_tags),
            )

        self._set_idle()

        # ── Diagnostic: strategy distribution in final output ──
        final_strategies = defaultdict(int)
        for r in ranked:
            final_strategies[r.strategy] += 1

        return {
            "user_id": user_id,
            "items": [{"item_id": r.item_id, "item_type": r.item_type,
                        "score": round(r.score, 4), "strategy": r.strategy,
                        "related_tags": r.related_tags}
                       for r in ranked],
            "strategy_selected": strategy,
            "strategy_weights": self.get_ts_weights(user_id),
            "n_candidates": post_dedup_total,
            "diagnostics": {
                "pre_dedup_total": pre_dedup_total,
                "post_dedup_total": post_dedup_total,
                "dedup_overlap": dedup_overlap,
                "strategy_counts": dict(strategy_counts),
                "pre_mmr_unique_tags": len(pre_mmr_tags),
                "post_mmr_unique_tags": len(post_mmr_tags),
                "final_strategy_distribution": dict(final_strategies),
            },
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

        if "tags" in interactions.columns and "timestamp" in interactions.columns:
            tag_last_seen: dict[str, int] = {}
            for _, row in interactions.iterrows():
                tags = row.get("tags", [])
                if isinstance(tags, str):
                    tags = [int(x) for x in tags.replace(";", ",").split(",") if x.strip().isdigit()]
                if isinstance(tags, list):
                    for tag in tags:
                        tag_last_seen[str(int(tag))] = int(row["timestamp"])
            profile["tag_last_seen"] = tag_last_seen

        if confidence_result and "class_names" in confidence_result:
            profile["false_confidence_count"] = count_risk_confidence_events(
                confidence_result["class_names"],
                n_classes=confidence_result.get("n_classes", self._n_confidence_classes),
            )

        self._user_profiles[user_id] = profile
