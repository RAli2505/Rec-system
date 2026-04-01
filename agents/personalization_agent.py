"""
Personalization Agent for MARS.

Clusters students into learner types using K-Means on 8 behavioural
features extracted from interaction history.  The optimal K is chosen
via Silhouette Score (K = 3..8, expecting 4-6 clusters).

Features (8 conceptual, 14 dimensions after one-hot)
-----------------------------------------------------
1. avg_elapsed_time      — mean response time, normalised by global median
2. accuracy_rate         — fraction of correct answers
3. changed_answer_rate   — fraction of answers changed before submit
4. dominant_part         — one-hot (7) of the student's most-practised part
5. session_frequency     — study sessions per week
6. avg_questions_per_session — mean questions answered per session
7. false_confidence_rate — fraction of FALSE_CONFIDENCE class responses
8. learning_speed        — slope of rolling accuracy over last 50 questions

Cluster labels are automatically named by interpreting centroid feature
values (e.g. "Fast Learner", "Struggling", "Steady", "Cramming").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .base_agent import BaseAgent

logger = logging.getLogger("mars.agent.personalization")

# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────

NUM_PARTS = 7
SESSION_GAP_MS = 30 * 60 * 1_000   # 30 minutes — same as preprocessor
K_RANGE = range(3, 9)               # test K = 3..8
RANDOM_STATE = 42
ROLLING_WINDOW = 50                  # for learning_speed

FEATURE_NAMES_SCALAR = [
    "avg_elapsed_time",
    "accuracy_rate",
    "changed_answer_rate",
    "session_frequency",
    "avg_questions_per_session",
    "false_confidence_rate",
    "learning_speed",
]
FEATURE_NAMES_ONEHOT = [f"dominant_part_{p}" for p in range(1, NUM_PARTS + 1)]
FEATURE_NAMES = FEATURE_NAMES_SCALAR + FEATURE_NAMES_ONEHOT  # 14 total

# Archetype label templates (mapped by centroid analysis)
ARCHETYPE_LABELS = {
    "fast_high":      "Fast Learner",
    "slow_high":      "Methodical",
    "fast_low":       "Struggling Guesser",
    "slow_low":       "Struggling",
    "improving":      "Improving",
    "cramming":       "Cramming",
    "steady":         "Steady",
    "default":        "Learner",
}


# ──────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────

@dataclass
class ClusterProfile:
    """Result of cluster assignment for a single user."""
    cluster_id: int
    cluster_name: str
    user_type: str
    feature_vector: list[float]
    centroid: list[float]
    distance_to_centroid: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "cluster_name": self.cluster_name,
            "user_type": self.user_type,
            "feature_vector": self.feature_vector,
            "centroid": self.centroid,
            "distance_to_centroid": round(self.distance_to_centroid, 4),
        }


# ──────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────

def extract_user_features(interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Build one feature row per user from their interaction history.

    Parameters
    ----------
    interactions : pd.DataFrame
        Must contain: user_id, timestamp, correct, elapsed_time.
        Optional: changed_answer, part_id, tags.

    Returns
    -------
    pd.DataFrame indexed by user_id with 14 feature columns.
    """
    df = interactions.copy()
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Global median elapsed for normalisation
    global_median_elapsed = df["elapsed_time"].median()
    if global_median_elapsed == 0 or pd.isna(global_median_elapsed):
        global_median_elapsed = 15_000.0

    # ── Per-user aggregates ──
    user_groups = df.groupby("user_id")

    feats = pd.DataFrame(index=user_groups.groups.keys())
    feats.index.name = "user_id"

    # 1. avg_elapsed_time (normalised)
    feats["avg_elapsed_time"] = (
        user_groups["elapsed_time"].mean() / global_median_elapsed
    ).clip(0, 10).fillna(1.0)

    # 2. accuracy_rate
    feats["accuracy_rate"] = user_groups["correct"].mean().fillna(0.5)

    # 3. changed_answer_rate
    if "changed_answer" in df.columns:
        feats["changed_answer_rate"] = (
            user_groups["changed_answer"].mean().fillna(0.0)
        )
    else:
        feats["changed_answer_rate"] = 0.0

    # 4. session_frequency (sessions per week)
    #    Session = gap > 30 min between consecutive answers
    def _session_freq(grp: pd.DataFrame) -> float:
        if len(grp) < 2:
            return 1.0
        ts = grp["timestamp"].values
        gaps = np.diff(ts)
        n_sessions = 1 + int((gaps > SESSION_GAP_MS).sum())
        span_weeks = max((ts[-1] - ts[0]) / (7 * 24 * 3600 * 1000), 1.0)
        return n_sessions / span_weeks

    feats["session_frequency"] = user_groups.apply(_session_freq).fillna(1.0)

    # 5. avg_questions_per_session
    def _avg_q_per_session(grp: pd.DataFrame) -> float:
        if len(grp) < 2:
            return float(len(grp))
        ts = grp["timestamp"].values
        gaps = np.diff(ts)
        n_sessions = 1 + int((gaps > SESSION_GAP_MS).sum())
        return len(grp) / max(n_sessions, 1)

    feats["avg_questions_per_session"] = (
        user_groups.apply(_avg_q_per_session).fillna(1.0)
    )

    # 6. false_confidence_rate
    #    FALSE_CONFIDENCE = fast AND incorrect AND NOT changed
    if "elapsed_time" in df.columns and "correct" in df.columns:
        med_by_q = df.groupby("question_id")["elapsed_time"].transform("median")
        is_fast = df["elapsed_time"] < med_by_q
        is_incorrect = ~df["correct"].astype(bool)
        not_changed = ~df["changed_answer"].astype(bool) if "changed_answer" in df.columns else True
        df["_is_false_conf"] = (is_fast & is_incorrect & not_changed).astype(float)
        feats["false_confidence_rate"] = (
            df.groupby("user_id")["_is_false_conf"].mean().fillna(0.0)
        )
        df.drop(columns=["_is_false_conf"], inplace=True)
    else:
        feats["false_confidence_rate"] = 0.0

    # 7. learning_speed (slope of rolling accuracy over last 50 questions)
    def _learning_speed(grp: pd.DataFrame) -> float:
        correct = grp["correct"].astype(float).values
        if len(correct) < 10:
            return 0.0
        # Rolling mean with window, take last ROLLING_WINDOW
        tail = correct[-ROLLING_WINDOW:] if len(correct) >= ROLLING_WINDOW else correct
        if len(tail) < 5:
            return 0.0
        # Linear regression slope: accuracy vs time-index
        x = np.arange(len(tail), dtype=float)
        x -= x.mean()
        y = tail - tail.mean()
        denom = (x ** 2).sum()
        if denom < 1e-10:
            return 0.0
        slope = (x * y).sum() / denom
        return float(slope)

    feats["learning_speed"] = user_groups.apply(_learning_speed).fillna(0.0)

    # 8. dominant_part (one-hot 7)
    if "part_id" in df.columns:
        dominant = user_groups["part_id"].agg(
            lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else 1
        ).astype(int).clip(1, NUM_PARTS)
    else:
        dominant = pd.Series(1, index=feats.index)

    for p in range(1, NUM_PARTS + 1):
        feats[f"dominant_part_{p}"] = (dominant == p).astype(float)

    # Ensure column order
    feats = feats[FEATURE_NAMES]

    logger.info(
        "Extracted features for %d users (%d dimensions)",
        len(feats), feats.shape[1],
    )
    return feats


# ──────────────────────────────────────────────────────────
# Cluster naming heuristic
# ──────────────────────────────────────────────────────────

def _name_clusters(centroids_raw: np.ndarray) -> dict[int, str]:
    """
    Assign unique human-readable archetype names to all clusters at once.

    Uses relative ranking of centroid features (z-scored across clusters)
    to ensure differentiation even when absolute values are similar.
    """
    n = len(centroids_raw)
    feat_idx = {name: i for i, name in enumerate(FEATURE_NAMES)}

    acc = centroids_raw[:, feat_idx["accuracy_rate"]]
    speed = centroids_raw[:, feat_idx["avg_elapsed_time"]]
    learn = centroids_raw[:, feat_idx["learning_speed"]]
    false_conf = centroids_raw[:, feat_idx["false_confidence_rate"]]
    sess_freq = centroids_raw[:, feat_idx["session_frequency"]]
    q_per_sess = centroids_raw[:, feat_idx["avg_questions_per_session"]]

    # Z-score each feature across clusters for relative comparison
    def _z(arr: np.ndarray) -> np.ndarray:
        s = arr.std()
        return (arr - arr.mean()) / s if s > 1e-10 else np.zeros_like(arr)

    z_acc = _z(acc)
    z_speed = _z(speed)
    z_learn = _z(learn)
    z_false = _z(false_conf)
    z_freq = _z(sess_freq)
    z_qps = _z(q_per_sess)

    # Score each archetype for each cluster; pick the best unique assignment
    # Priority-ordered candidate rules (label, scoring function)
    candidates = [
        ("Fast Learner",       lambda i: z_acc[i] + z_learn[i] - z_speed[i]),
        ("Improving",          lambda i: 2 * z_learn[i]),
        ("Methodical",         lambda i: z_acc[i] + z_speed[i] - z_false[i]),
        ("Cramming",           lambda i: z_freq[i] + z_qps[i]),
        ("Struggling Guesser", lambda i: -z_acc[i] - z_speed[i] + z_false[i]),
        ("Struggling",         lambda i: -z_acc[i] - z_learn[i]),
        ("Steady",             lambda i: -abs(z_acc[i]) - abs(z_speed[i])),
    ]

    names: dict[int, str] = {}
    used_labels: set[str] = set()

    # For each cluster, score all unused labels and pick the best
    # Process clusters ordered by most extreme features first (easier to name)
    extremity = [sum(abs(z) for z in [z_acc[i], z_speed[i], z_learn[i]])
                 for i in range(n)]
    order = sorted(range(n), key=lambda i: -extremity[i])

    for cid in order:
        best_label = "Steady"
        best_score = -999.0
        for label, score_fn in candidates:
            if label in used_labels:
                continue
            s = score_fn(cid)
            if s > best_score:
                best_score = s
                best_label = label
        names[cid] = best_label
        used_labels.add(best_label)

    return names


# ──────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────

class PersonalizationAgent(BaseAgent):
    """
    K-Means-based student clustering for personalization.

    Extracts 8 behavioural features per user, finds the optimal K
    via Silhouette Score, and assigns each student to a learner archetype.
    """

    name = "personalization"

    def __init__(self) -> None:
        super().__init__()
        self.model: KMeans | None = None
        self.scaler: StandardScaler | None = None
        self.optimal_k: int = 0
        self._models_dir = Path("models")
        self._user_features: pd.DataFrame | None = None  # cached feature matrix
        self._user_clusters: dict[str, int] = {}          # user_id → cluster_id
        self._cluster_names: dict[int, str] = {}          # cluster_id → name
        self._cluster_centroids_raw: np.ndarray | None = None  # unscaled centroids
        self._silhouette_scores: dict[int, float] = {}
        self._inertias: dict[int, float] = {}

    # ──────────────────────────────────────────────────────
    # BaseAgent interface
    # ──────────────────────────────────────────────────────

    def initialize(self, **kwargs: Any) -> None:
        if "interactions_df" in kwargs:
            features = extract_user_features(kwargs["interactions_df"])
            self.train_clusters(features)

    def receive_message(self, message):
        super().receive_message(message)
        action = message.data.get("action")
        if action == "assign_cluster":
            return self.assign_cluster(
                user_id=message.data.get("user_id"),
                diagnostic=message.data.get("diagnostic"),
                confidence=message.data.get("confidence"),
            )
        if action == "get_user_type":
            return self.get_user_type(message.data.get("user_id"))
        return None

    # ──────────────────────────────────────────────────────
    # 1. Feature extraction from interactions
    # ──────────────────────────────────────────────────────

    def build_features(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Extract per-user feature matrix and cache it."""
        self._user_features = extract_user_features(interactions_df)
        return self._user_features

    # ──────────────────────────────────────────────────────
    # 2. Train clusters
    # ──────────────────────────────────────────────────────

    def train_clusters(
        self,
        user_features_df: pd.DataFrame,
        k_range: range = K_RANGE,
    ) -> int:
        """
        Find optimal K via Silhouette Score and fit final K-Means.

        Parameters
        ----------
        user_features_df : pd.DataFrame
            One row per user, columns = FEATURE_NAMES (14 dims).
        k_range : range
            Values of K to evaluate (default 3..8).

        Returns
        -------
        int — the optimal K.
        """
        self._set_processing()
        self._user_features = user_features_df

        X_raw = user_features_df[FEATURE_NAMES].values.astype(np.float64)

        # Handle NaN/Inf
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=5.0, neginf=-5.0)

        # Scale
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X_raw)

        # ── Silhouette search ──
        best_k = min(k_range)
        best_sil = -1.0
        self._silhouette_scores = {}
        self._inertias = {}

        for k in k_range:
            if k >= len(X):
                break
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, max_iter=300)
            labels = km.fit_predict(X)
            sil = silhouette_score(X, labels)
            self._silhouette_scores[k] = round(float(sil), 4)
            self._inertias[k] = round(float(km.inertia_), 2)
            logger.info("K=%d  silhouette=%.4f  inertia=%.1f", k, sil, km.inertia_)
            if sil > best_sil:
                best_sil = sil
                best_k = k

        self.optimal_k = best_k
        logger.info("Optimal K=%d (silhouette=%.4f)", best_k, best_sil)

        # ── Fit final model ──
        self.model = KMeans(
            n_clusters=best_k, random_state=RANDOM_STATE,
            n_init=10, max_iter=300,
        )
        labels = self.model.fit_predict(X)

        # Inverse-transform centroids for interpretation
        self._cluster_centroids_raw = self.scaler.inverse_transform(
            self.model.cluster_centers_
        )

        # Name each cluster (unique names via relative ranking)
        self._cluster_names = _name_clusters(self._cluster_centroids_raw)

        # Store per-user assignments
        user_ids = user_features_df.index.tolist()
        for uid, label in zip(user_ids, labels):
            self._user_clusters[str(uid)] = int(label)

        # Save model artifacts
        self._save_model()

        logger.info(
            "Clusters: %s",
            {cid: f"{name} (n={int((labels == cid).sum())})"
             for cid, name in self._cluster_names.items()},
        )

        self._set_idle()
        return best_k

    # ──────────────────────────────────────────────────────
    # 3. Assign cluster
    # ──────────────────────────────────────────────────────

    def assign_cluster(
        self,
        user_id: str,
        interactions: pd.DataFrame | None = None,
        diagnostic: dict | None = None,
        confidence: dict | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Assign a user to a cluster.

        If the user already has cached features, use those.
        If interactions are provided, compute features on the fly.
        If only diagnostic/confidence are available (cold start),
        build a partial feature vector.

        Returns dict compatible with Orchestrator pipeline.
        """
        self._set_processing()

        # Already assigned?
        if user_id in self._user_clusters and self.model is not None:
            cid = self._user_clusters[user_id]
            profile = self._build_profile(user_id, cid)
            self._set_idle()
            return profile.to_dict()

        # Try to build features from interactions
        if interactions is not None and len(interactions) > 0:
            feat_df = extract_user_features(interactions)
            if user_id in feat_df.index:
                vec = feat_df.loc[user_id, FEATURE_NAMES].values.astype(np.float64)
            else:
                # User might have different ID format
                vec = feat_df.iloc[0][FEATURE_NAMES].values.astype(np.float64)
        elif diagnostic is not None:
            # Cold start: build partial vector from diagnostic + confidence
            vec = self._cold_start_features(diagnostic, confidence)
        else:
            # No data at all — assign to nearest cluster from default vector
            vec = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
            vec[FEATURE_NAMES.index("accuracy_rate")] = 0.5
            vec[FEATURE_NAMES.index("avg_elapsed_time")] = 1.0
            vec[FEATURE_NAMES.index("session_frequency")] = 1.0
            vec[FEATURE_NAMES.index("avg_questions_per_session")] = 10.0

        if self.model is None or self.scaler is None:
            self._set_idle()
            return ClusterProfile(
                cluster_id=0,
                cluster_name="Unknown",
                user_type="Unknown",
                feature_vector=vec.tolist(),
                centroid=[],
                distance_to_centroid=0.0,
            ).to_dict()

        # Predict cluster
        vec = np.nan_to_num(vec, nan=0.0, posinf=5.0, neginf=-5.0)
        X_scaled = self.scaler.transform(vec.reshape(1, -1))
        cid = int(self.model.predict(X_scaled)[0])
        self._user_clusters[user_id] = cid

        # Cache features
        if self._user_features is not None:
            self._user_features.loc[user_id] = vec

        profile = self._build_profile(user_id, cid, feature_vector=vec)
        self._set_idle()
        return profile.to_dict()

    # ──────────────────────────────────────────────────────
    # 4. Personalize (assessment pipeline)
    # ──────────────────────────────────────────────────────

    def personalize(
        self,
        user_id: str,
        diagnostic: dict | None = None,
        confidence: dict | None = None,
        recommendations: dict | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Refine cluster assignment after an assessment session.

        Called from Orchestrator.assessment_pipeline (step 6).
        Returns updated cluster profile + personalization adjustments.
        """
        # Re-assign cluster with latest data
        profile = self.assign_cluster(
            user_id=user_id,
            diagnostic=diagnostic,
            confidence=confidence,
        )

        cluster_name = profile.get("cluster_name", "Unknown")

        # Personalization adjustments based on cluster
        adjustments = self._cluster_adjustments(
            profile.get("cluster_id", 0), diagnostic, confidence,
        )

        return {
            **profile,
            "adjustments": adjustments,
        }

    # ──────────────────────────────────────────────────────
    # 5. Get user type
    # ──────────────────────────────────────────────────────

    def get_user_type(self, user_id: str) -> str:
        """Return the human-readable cluster name for a user."""
        cid = self._user_clusters.get(user_id)
        if cid is None:
            return "Unknown"
        return self._cluster_names.get(cid, f"Cluster_{cid}")

    # ──────────────────────────────────────────────────────
    # 6. Cluster analysis (for paper)
    # ──────────────────────────────────────────────────────

    def cluster_summary(self) -> pd.DataFrame:
        """
        Build a summary table of cluster characteristics.

        Returns DataFrame: cluster_id, name, size, and mean of each feature.
        """
        if self._user_features is None or self.model is None:
            return pd.DataFrame()

        df = self._user_features.copy()
        df["cluster"] = [
            self._user_clusters.get(str(uid), -1) for uid in df.index
        ]
        df = df[df["cluster"] >= 0]

        rows = []
        for cid in range(self.optimal_k):
            mask = df["cluster"] == cid
            cluster_data = df[mask]
            row = {
                "cluster_id": cid,
                "cluster_name": self._cluster_names.get(cid, f"Cluster_{cid}"),
                "size": int(mask.sum()),
                "pct": round(100 * mask.sum() / len(df), 1),
            }
            for feat in FEATURE_NAMES_SCALAR:
                row[f"mean_{feat}"] = round(float(cluster_data[feat].mean()), 4)
            rows.append(row)

        return pd.DataFrame(rows)

    def get_centroids_df(self) -> pd.DataFrame:
        """Return centroids as a DataFrame (unscaled, original feature space)."""
        if self._cluster_centroids_raw is None:
            return pd.DataFrame()
        df = pd.DataFrame(
            self._cluster_centroids_raw,
            columns=FEATURE_NAMES,
        )
        df.insert(0, "cluster_name", [
            self._cluster_names.get(i, f"Cluster_{i}") for i in range(len(df))
        ])
        return df

    # ──────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────

    def _build_profile(
        self,
        user_id: str,
        cluster_id: int,
        feature_vector: np.ndarray | None = None,
    ) -> ClusterProfile:
        """Build a ClusterProfile for a user."""
        name = self._cluster_names.get(cluster_id, f"Cluster_{cluster_id}")

        if feature_vector is None and self._user_features is not None:
            if user_id in self._user_features.index:
                feature_vector = self._user_features.loc[user_id].values

        fv = feature_vector.tolist() if feature_vector is not None else []
        centroid = (
            self._cluster_centroids_raw[cluster_id].tolist()
            if self._cluster_centroids_raw is not None else []
        )

        # Distance to centroid
        dist = 0.0
        if feature_vector is not None and self.scaler is not None and self.model is not None:
            x_scaled = self.scaler.transform(feature_vector.reshape(1, -1))
            c_scaled = self.model.cluster_centers_[cluster_id]
            dist = float(np.linalg.norm(x_scaled - c_scaled))

        return ClusterProfile(
            cluster_id=cluster_id,
            cluster_name=name,
            user_type=name,
            feature_vector=fv,
            centroid=centroid,
            distance_to_centroid=dist,
        )

    def _cold_start_features(
        self,
        diagnostic: dict | None,
        confidence: dict | None,
    ) -> np.ndarray:
        """Build a partial feature vector from diagnostic + confidence results."""
        vec = np.zeros(len(FEATURE_NAMES), dtype=np.float64)

        # Defaults
        vec[FEATURE_NAMES.index("avg_elapsed_time")] = 1.0
        vec[FEATURE_NAMES.index("session_frequency")] = 1.0
        vec[FEATURE_NAMES.index("avg_questions_per_session")] = 10.0

        if diagnostic:
            responses = diagnostic.get("responses", [])
            if responses:
                n_correct = sum(1 for r in responses if r.get("correct", False))
                vec[FEATURE_NAMES.index("accuracy_rate")] = n_correct / len(responses)
                # Dominant part
                parts = [r.get("part_id", 1) for r in responses]
                if parts:
                    dominant = max(set(parts), key=parts.count)
                    dominant = min(max(dominant, 1), NUM_PARTS)
                    vec[FEATURE_NAMES.index(f"dominant_part_{dominant}")] = 1.0

        if confidence:
            class_names = confidence.get("class_names", [])
            if class_names:
                n_false_conf = sum(1 for c in class_names if c == "FALSE_CONFIDENCE")
                vec[FEATURE_NAMES.index("false_confidence_rate")] = (
                    n_false_conf / len(class_names)
                )

        return vec

    def _cluster_adjustments(
        self,
        cluster_id: int,
        diagnostic: dict | None,
        confidence: dict | None,
    ) -> dict[str, Any]:
        """
        Return personalisation adjustments based on the cluster archetype.

        These can be used by RecommendationAgent to tune difficulty,
        pacing, and content mix.
        """
        name = self._cluster_names.get(cluster_id, "default")

        # Default adjustments
        adj: dict[str, Any] = {
            "difficulty_adjustment": 0.0,
            "pacing": "normal",
            "content_mix": "balanced",
            "review_frequency": "standard",
        }

        if name == "Fast Learner":
            adj["difficulty_adjustment"] = +0.3
            adj["pacing"] = "accelerated"
            adj["content_mix"] = "challenge_heavy"
            adj["review_frequency"] = "low"
        elif name == "Methodical":
            adj["difficulty_adjustment"] = +0.1
            adj["pacing"] = "normal"
            adj["content_mix"] = "balanced"
            adj["review_frequency"] = "standard"
        elif name == "Struggling Guesser":
            adj["difficulty_adjustment"] = -0.3
            adj["pacing"] = "slow"
            adj["content_mix"] = "review_heavy"
            adj["review_frequency"] = "high"
        elif name == "Struggling":
            adj["difficulty_adjustment"] = -0.2
            adj["pacing"] = "slow"
            adj["content_mix"] = "foundational"
            adj["review_frequency"] = "high"
        elif name == "Improving":
            adj["difficulty_adjustment"] = +0.1
            adj["pacing"] = "adaptive"
            adj["content_mix"] = "progressive"
            adj["review_frequency"] = "standard"
        elif name == "Cramming":
            adj["difficulty_adjustment"] = 0.0
            adj["pacing"] = "normal"
            adj["content_mix"] = "spaced_review"
            adj["review_frequency"] = "high"
        elif name == "Steady":
            adj["difficulty_adjustment"] = 0.0
            adj["pacing"] = "normal"
            adj["content_mix"] = "balanced"
            adj["review_frequency"] = "standard"

        return adj

    def _save_model(self) -> None:
        """Save model artifacts to disk."""
        import pickle

        self._models_dir.mkdir(parents=True, exist_ok=True)
        artifacts = {
            "model": self.model,
            "scaler": self.scaler,
            "optimal_k": self.optimal_k,
            "cluster_names": self._cluster_names,
            "centroids_raw": self._cluster_centroids_raw,
            "silhouette_scores": self._silhouette_scores,
            "inertias": self._inertias,
        }
        path = self._models_dir / "personalization_kmeans.pkl"
        with open(path, "wb") as f:
            pickle.dump(artifacts, f)
        logger.info("Saved personalization model → %s", path)

    def load_model(self) -> bool:
        """Load model artifacts from disk. Returns True if successful."""
        import pickle

        path = self._models_dir / "personalization_kmeans.pkl"
        if not path.exists():
            return False
        with open(path, "rb") as f:
            artifacts = pickle.load(f)
        self.model = artifacts["model"]
        self.scaler = artifacts["scaler"]
        self.optimal_k = artifacts["optimal_k"]
        self._cluster_names = artifacts["cluster_names"]
        self._cluster_centroids_raw = artifacts["centroids_raw"]
        self._silhouette_scores = artifacts["silhouette_scores"]
        self._inertias = artifacts["inertias"]
        logger.info("Loaded personalization model (K=%d)", self.optimal_k)
        return True

    @property
    def silhouette_scores(self) -> dict[int, float]:
        return self._silhouette_scores

    @property
    def inertias(self) -> dict[int, float]:
        return self._inertias
