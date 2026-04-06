"""
Baseline recommenders for MARS evaluation.

Three baselines required for academic paper comparison:
  1. Random       — uniformly random items from catalogue
  2. Popularity   — most frequently answered questions globally
  3. BPR          — Bayesian Personalised Ranking (matrix factorisation)

All baselines share the same interface as RecommendationAgent:
    baseline.recommend(user_id, n=10) -> dict with "items" list

Usage in run_multi_seed.py:
    from agents.baselines import RandomBaseline, PopularityBaseline, BPRBaseline
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger("mars.baselines")


# ──────────────────────────────────────────────────────────
# 1. Random Baseline
# ──────────────────────────────────────────────────────────

class RandomBaseline:
    """Recommends n uniformly random items from the catalogue."""

    name = "random"

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.RandomState(seed)
        self._item_ids: list[str] = []

    def fit(self, interactions_df: pd.DataFrame, **kwargs) -> None:
        """Build item catalogue from interactions."""
        self._item_ids = [
            str(x) for x in interactions_df["question_id"].unique()
        ]
        logger.info("RandomBaseline: catalogue = %d items", len(self._item_ids))

    def recommend(self, user_id: str, n: int = 10, **kwargs) -> dict[str, Any]:
        if not self._item_ids:
            return {"user_id": user_id, "items": []}
        chosen = self._rng.choice(
            self._item_ids,
            size=min(n, len(self._item_ids)),
            replace=False,
        )
        return {
            "user_id": user_id,
            "items": [
                {"item_id": iid, "score": 1.0, "strategy": "random", "related_tags": []}
                for iid in chosen
            ],
        }


# ──────────────────────────────────────────────────────────
# 2. Popularity Baseline
# ──────────────────────────────────────────────────────────

class PopularityBaseline:
    """
    Recommends the globally most-answered questions.

    Grounded in the long-tail distribution of educational content:
    popular items are seen by more students and have more evidence
    of pedagogical value (Lops et al., 2011).
    """

    name = "popularity"

    def __init__(self) -> None:
        self._top_items: list[str] = []   # sorted by frequency desc
        self._top_scores: list[float] = []

    def fit(self, interactions_df: pd.DataFrame, **kwargs) -> None:
        """Compute global item frequency ranking."""
        counts = (
            interactions_df["question_id"]
            .astype(str)
            .value_counts()
        )
        total = counts.sum()
        self._top_items = counts.index.tolist()
        # Normalize to [0, 1] for compatibility with scoring
        self._top_scores = (counts.values / total).tolist()
        logger.info(
            "PopularityBaseline: %d items, top=%s (freq=%.4f)",
            len(self._top_items),
            self._top_items[0] if self._top_items else "—",
            self._top_scores[0] if self._top_scores else 0.0,
        )

    def recommend(self, user_id: str, n: int = 10, **kwargs) -> dict[str, Any]:
        items = [
            {
                "item_id": iid,
                "score": round(float(score), 6),
                "strategy": "popularity",
                "related_tags": [],
            }
            for iid, score in zip(self._top_items[:n], self._top_scores[:n])
        ]
        return {"user_id": user_id, "items": items}


# ──────────────────────────────────────────────────────────
# 3. BPR Baseline (Bayesian Personalised Ranking)
# ──────────────────────────────────────────────────────────

class BPRBaseline:
    """
    Bayesian Personalised Ranking via SGD matrix factorisation.

    BPR optimises for pairwise ranking loss:
        L = Σ -log σ(x_ui - x_uj) + λ||Θ||²
    where x_ui = P[u] · Q[i]ᵀ, and (u, i) is observed,
    (u, j) is unobserved (Rendle et al., 2009, UAI).

    Implemented as lightweight SGD — no external dependency required.
    For large datasets falls back to SVD (TruncatedSVD from sklearn).
    """

    name = "bpr"

    def __init__(
        self,
        n_factors: int = 64,
        n_epochs: int = 20,
        lr: float = 0.01,
        reg: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self._rng = np.random.RandomState(seed)

        self._user_factors: np.ndarray | None = None   # (n_users, k)
        self._item_factors: np.ndarray | None = None   # (n_items, k)
        self._user_map: dict[str, int] = {}
        self._item_map: dict[str, int] = {}
        self._item_ids: list[str] = []

    def fit(self, interactions_df: pd.DataFrame, **kwargs) -> None:
        """
        Fit BPR via implicit SGD on positive interactions.

        Uses TruncatedSVD on the user×item binary matrix as a fast
        initialisation, then refines with BPR pairwise SGD updates.
        """
        df = interactions_df.copy()
        df["user_id"] = df["user_id"].astype(str)
        df["question_id"] = df["question_id"].astype(str)

        users = df["user_id"].unique().tolist()
        items = df["question_id"].unique().tolist()

        self._user_map = {u: i for i, u in enumerate(users)}
        self._item_map = {it: i for i, it in enumerate(items)}
        self._item_ids = items

        n_users = len(users)
        n_items = len(items)

        # Build sparse binary matrix (user × item)
        rows = df["user_id"].map(self._user_map).values
        cols = df["question_id"].map(self._item_map).values
        data = np.ones(len(df), dtype=np.float32)
        R = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

        logger.info(
            "BPRBaseline: %d users × %d items, density=%.4f%%",
            n_users, n_items, 100 * len(df) / (n_users * n_items),
        )

        # Initialise via SVD (fast, good starting point for BPR)
        k = min(self.n_factors, min(n_users, n_items) - 1)
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=k, random_state=42)
        U = svd.fit_transform(R)                  # (n_users, k)
        V = svd.components_.T                      # (n_items, k)

        # Normalize
        U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-8)
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)

        # BPR SGD refinement
        R_csr = R.tocsr()
        n_samples = min(len(df) * 2, 500_000)  # cap for speed

        logger.info("BPR SGD: %d epochs, %d samples/epoch", self.n_epochs, n_samples)

        for epoch in range(self.n_epochs):
            # Sample triplets (u, i, j): i=positive, j=negative
            u_idx = self._rng.randint(0, n_users, n_samples)
            i_idx = np.array([
                R_csr.indices[R_csr.indptr[u]:R_csr.indptr[u + 1]]
                [self._rng.randint(0, max(1, R_csr.indptr[u + 1] - R_csr.indptr[u]))]
                if R_csr.indptr[u + 1] > R_csr.indptr[u] else 0
                for u in u_idx
            ])
            j_idx = self._rng.randint(0, n_items, n_samples)

            # BPR update
            xu = (U[u_idx] * V[i_idx]).sum(axis=1)  # positive score
            xj = (U[u_idx] * V[j_idx]).sum(axis=1)  # negative score
            delta = xu - xj
            sigmoid = 1.0 / (1.0 + np.exp(-delta.clip(-10, 10)))
            grad = (1.0 - sigmoid)[:, None]           # (n_samples, 1)

            lr = self.lr
            reg = self.reg

            # Vectorised update (approximate, no per-sample loop)
            np.add.at(U, u_idx,  lr * (grad * (V[i_idx] - V[j_idx]) - reg * U[u_idx]))
            np.add.at(V, i_idx,  lr * (grad * U[u_idx] - reg * V[i_idx]))
            np.add.at(V, j_idx, -lr * (grad * U[u_idx] + reg * V[j_idx]))

            if (epoch + 1) % 5 == 0 or epoch == 0:
                auc_approx = float((sigmoid > 0.5).mean())
                logger.info(
                    "BPR epoch %d/%d  pairwise_acc=%.4f",
                    epoch + 1, self.n_epochs, auc_approx,
                )

        self._user_factors = U
        self._item_factors = V
        logger.info("BPRBaseline: training complete")

    def recommend(self, user_id: str, n: int = 10, **kwargs) -> dict[str, Any]:
        uid = self._user_map.get(str(user_id))
        if uid is None or self._user_factors is None:
            # Cold-start: fall back to random
            chosen = self._rng.choice(len(self._item_ids), size=min(n, len(self._item_ids)), replace=False)
            return {
                "user_id": user_id,
                "items": [{"item_id": self._item_ids[i], "score": 0.0,
                            "strategy": "bpr_coldstart", "related_tags": []} for i in chosen],
            }

        # Score all items
        scores = self._user_factors[uid] @ self._item_factors.T  # (n_items,)

        # Top-n by score (exclude already seen — optional for ablation)
        top_idx = np.argpartition(scores, -min(n, len(scores)))[-n:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        items = [
            {
                "item_id": self._item_ids[i],
                "score": round(float(scores[i]), 6),
                "strategy": "bpr",
                "related_tags": [],
            }
            for i in top_idx
        ]
        return {"user_id": user_id, "items": items}
