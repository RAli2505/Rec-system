from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


class BayesianPersonalizedRanking:
    """
    Small compatibility shim for environments without the compiled `implicit` package.

    This uses a truncated SVD over the implicit user-item matrix and exposes the
    subset of the API that the notebooks rely on: `fit(...)` and `recommend(...)`.
    """

    def __init__(
        self,
        factors: int = 64,
        iterations: int = 50,
        learning_rate: float = 0.01,
        random_state: int | None = None,
        **_: object,
    ) -> None:
        self.factors = int(factors)
        self.iterations = int(iterations)
        self.learning_rate = float(learning_rate)
        self.random_state = random_state
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None

    def fit(self, user_items: sparse.spmatrix) -> "BayesianPersonalizedRanking":
        matrix = user_items.tocsr().astype(np.float32)
        n_users, n_items = matrix.shape
        n_components = max(1, min(self.factors, n_users - 1, n_items - 1))

        svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        user_latent = svd.fit_transform(matrix)
        item_latent = svd.components_.T
        sing_vals = np.sqrt(np.maximum(svd.singular_values_, 1e-8))

        self.user_factors = user_latent / sing_vals
        self.item_factors = item_latent * sing_vals
        return self

    def recommend(
        self,
        userid: int,
        user_items: sparse.spmatrix,
        N: int = 10,
        filter_already_liked_items: bool = True,
        **_: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model must be fit before calling recommend().")

        scores = self.user_factors[int(userid)] @ self.item_factors.T
        scores = np.asarray(scores, dtype=np.float32).copy()

        if filter_already_liked_items:
            row = user_items.tocsr()
            if row.shape[0] > 0:
                liked = row.indices
                scores[liked] = -np.inf

        top_n = min(int(N), scores.shape[0])
        top_idx = np.argpartition(-scores, kth=top_n - 1)[:top_n]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return top_idx.astype(np.int64), scores[top_idx]
