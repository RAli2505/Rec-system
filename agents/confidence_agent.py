"""
Confidence Agent for MARS.

Classifies each student response into one of 6 confidence classes
using an XGBoost classifier trained on behavioural features.

Classes encode *how* the student answered — not just whether they
were correct — enabling finer-grained knowledge updates.

| Class              | ID               | Condition                                | Δ skill |
|--------------------|------------------|------------------------------------------|---------|
| Solid knowledge    | SOLID            | fast AND correct AND NOT changed         | +0.15   |
| Unsure correct     | UNSURE_CORRECT   | slow AND correct                         | +0.05   |
| False confidence   | FALSE_CONFIDENCE | fast AND incorrect AND NOT changed       | -0.10   |
| Clear gap          | CLEAR_GAP        | slow AND incorrect                       | -0.15   |
| Doubt + correct    | DOUBT_CORRECT    | changed AND correct                      | +0.03   |
| Doubt + incorrect  | DOUBT_INCORRECT  | changed AND incorrect                    | -0.12   |
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold

from .base_agent import BaseAgent

logger = logging.getLogger("mars.agent.confidence")


# ──────────────────────────────────────────────────────────
# Confidence classes
# ──────────────────────────────────────────────────────────

class ConfidenceClass(IntEnum):
    SOLID = 0
    UNSURE_CORRECT = 1
    FALSE_CONFIDENCE = 2
    CLEAR_GAP = 3
    DOUBT_CORRECT = 4
    DOUBT_INCORRECT = 5


CLASS_NAMES = [
    "SOLID",
    "UNSURE_CORRECT",
    "FALSE_CONFIDENCE",
    "CLEAR_GAP",
    "DOUBT_CORRECT",
    "DOUBT_INCORRECT",
]

SKILL_DELTAS = {
    ConfidenceClass.SOLID: +0.15,
    ConfidenceClass.UNSURE_CORRECT: +0.05,
    ConfidenceClass.FALSE_CONFIDENCE: -0.10,
    ConfidenceClass.CLEAR_GAP: -0.15,
    ConfidenceClass.DOUBT_CORRECT: +0.03,
    ConfidenceClass.DOUBT_INCORRECT: -0.12,
}

# Feature names for the XGBoost model (order matters)
FEATURE_NAMES = [
    "elapsed_time_normalized",
    "changed_answer",
    "response_count",
    "is_correct",
    "question_difficulty",
    "time_of_day_sin",
    "time_of_day_cos",
    "speed_vs_difficulty",
    "user_rolling_accuracy",
    "tag_familiarity",
    "part_1", "part_2", "part_3", "part_4",
    "part_5", "part_6", "part_7",
]


@dataclass
class ClassificationResult:
    """Result for a single interaction."""
    confidence_class: int
    class_name: str
    skill_delta: float
    probabilities: list[float]


# ──────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────

class ConfidenceAgent(BaseAgent):
    """
    XGBoost-based 6-class confidence classifier.

    Labels are auto-generated from behavioural signals (elapsed time,
    changed answer, correctness). The trained model generalises these
    rule-based labels using richer features including IRT difficulty,
    rolling accuracy, and time-of-day patterns.
    """

    name = "confidence"

    def __init__(self) -> None:
        super().__init__()
        self.model: xgb.XGBClassifier | None = None
        self._models_dir = Path("models")
        self._median_elapsed: dict[str, float] = {}   # question_id → median elapsed
        self._global_median_elapsed: float = 15_000.0
        self._irt_difficulty: dict[str, float] = {}    # question_id → b
        self._cv_results: dict[str, float] = {}

    # ──────────────────────────────────────────────────────
    # BaseAgent interface
    # ──────────────────────────────────────────────────────

    def initialize(self, **kwargs: Any) -> None:
        if "interactions_df" in kwargs:
            self.train(kwargs["interactions_df"],
                       irt_params=kwargs.get("irt_params"))

    def receive_message(self, message):
        super().receive_message(message)
        action = message.data.get("action")
        if action == "classify":
            return self.classify(message.data["interaction"])
        return None

    # ──────────────────────────────────────────────────────
    # 1. Auto-labelling
    # ──────────────────────────────────────────────────────

    def _assign_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Assign 6-class labels from behavioural signals.

        Priority: changed_answer checks first, then speed × correctness.
        """
        # Per-question median elapsed time
        med_elapsed = df.groupby("question_id")["elapsed_time"].transform("median")
        is_fast = df["elapsed_time"] < med_elapsed
        is_correct = df["correct"].astype(bool)
        changed = df["changed_answer"].astype(bool)

        labels = pd.Series(ConfidenceClass.CLEAR_GAP, index=df.index, dtype=int)

        # Changed answer takes priority
        labels[changed & is_correct] = ConfidenceClass.DOUBT_CORRECT
        labels[changed & ~is_correct] = ConfidenceClass.DOUBT_INCORRECT

        # Non-changed: speed × correctness
        nc = ~changed
        labels[nc & is_fast & is_correct] = ConfidenceClass.SOLID
        labels[nc & ~is_fast & is_correct] = ConfidenceClass.UNSURE_CORRECT
        labels[nc & is_fast & ~is_correct] = ConfidenceClass.FALSE_CONFIDENCE
        labels[nc & ~is_fast & ~is_correct] = ConfidenceClass.CLEAR_GAP

        return labels

    # ──────────────────────────────────────────────────────
    # 2. Feature engineering
    # ──────────────────────────────────────────────────────

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the 17-dimensional feature matrix (10 base + 7 part one-hot).
        """
        feats = pd.DataFrame(index=df.index)

        # 1. elapsed_time_normalized: elapsed / per-question median
        med = df.groupby("question_id")["elapsed_time"].transform("median")
        med = med.replace(0, self._global_median_elapsed)
        feats["elapsed_time_normalized"] = (df["elapsed_time"] / med).clip(0, 10).fillna(1.0)

        # 2. changed_answer
        feats["changed_answer"] = df["changed_answer"].astype(int)

        # 3. response_count
        feats["response_count"] = df["response_count"].fillna(1).astype(int) if "response_count" in df.columns else 1

        # 4. is_correct
        feats["is_correct"] = df["correct"].astype(int)

        # 5. question_difficulty (from IRT, 0 if unknown)
        feats["question_difficulty"] = df["question_id"].astype(str).map(self._irt_difficulty).fillna(0.0)

        # 6-7. time_of_day (sin/cos from timestamp)
        if "timestamp" in df.columns:
            # Convert ms timestamp to hour-of-day
            seconds_in_day = 86400
            hour_frac = (df["timestamp"] / 1000 % seconds_in_day) / seconds_in_day
            feats["time_of_day_sin"] = np.sin(2 * np.pi * hour_frac)
            feats["time_of_day_cos"] = np.cos(2 * np.pi * hour_frac)
        else:
            feats["time_of_day_sin"] = 0.0
            feats["time_of_day_cos"] = 0.0

        # 8. speed_vs_difficulty: elapsed / (difficulty + 1) normalized
        diff = feats["question_difficulty"] + 4  # shift so always positive
        feats["speed_vs_difficulty"] = (feats["elapsed_time_normalized"] / diff.clip(0.1)).clip(0, 10)

        # 9. user_rolling_accuracy (window=20)
        if "rolling_accuracy" in df.columns:
            feats["user_rolling_accuracy"] = df["rolling_accuracy"].fillna(0.5)
        elif "user_id" in df.columns and "timestamp" in df.columns:
            feats["user_rolling_accuracy"] = (
                df.sort_values(["user_id", "timestamp"])
                .groupby("user_id")["correct"]
                .transform(lambda s: s.rolling(20, min_periods=1).mean())
                .fillna(0.5)
            )
        else:
            feats["user_rolling_accuracy"] = df["correct"].expanding().mean().fillna(0.5)

        # 10. tag_familiarity: cumulative count of questions answered with same tags
        if "tags" in df.columns and "user_id" in df.columns:
            feats["tag_familiarity"] = (
                df.groupby("user_id").cumcount().clip(0, 200) / 200.0
            )
        else:
            feats["tag_familiarity"] = 0.0

        # 11-17. part_id one-hot
        if "part_id" in df.columns:
            for p in range(1, 8):
                feats[f"part_{p}"] = (df["part_id"] == p).astype(int)
        else:
            for p in range(1, 8):
                feats[f"part_{p}"] = 0

        return feats[FEATURE_NAMES]

    # ──────────────────────────────────────────────────────
    # 3. Training
    # ──────────────────────────────────────────────────────

    def train(
        self,
        interactions_df: pd.DataFrame,
        irt_params: Any | None = None,
        use_smote: bool = True,
        n_folds: int = 5,
    ) -> dict[str, float]:
        """
        Train the XGBoost 6-class confidence classifier.

        1. Auto-label interactions from behavioural rules.
        2. Build feature matrix.
        3. Optionally apply SMOTE for class balance.
        4. Stratified 5-fold CV.
        5. Train final model on full data.

        Returns dict of metrics.
        """
        self._set_processing()

        df = interactions_df.copy()
        df = df.dropna(subset=["elapsed_time", "correct", "changed_answer"])

        # Store medians for inference
        self._global_median_elapsed = float(df["elapsed_time"].median())
        med_by_q = df.groupby("question_id")["elapsed_time"].median()
        self._median_elapsed = med_by_q.to_dict()

        # IRT difficulty map
        if irt_params is not None:
            for qid, b_val in zip(irt_params.question_ids, irt_params.b):
                self._irt_difficulty[str(qid)] = float(b_val)

        # Auto-label
        labels = self._assign_labels(df)
        logger.info("Label distribution:\n%s", labels.value_counts().sort_index().to_string())

        # Features
        X = self._build_features(df)
        y = labels.values

        # ── SMOTE if imbalanced ──
        class_counts = pd.Series(y).value_counts()
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / max(min_count, 1)

        X_train_full = X.values.astype(np.float32)
        y_train_full = y

        if use_smote and imbalance_ratio > 5:
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=42, k_neighbors=min(5, min_count - 1) if min_count > 1 else 1)
                X_train_full, y_train_full = smote.fit_resample(X_train_full, y_train_full)
                logger.info("SMOTE applied: %d → %d samples (imbalance ratio was %.1f)",
                            len(y), len(y_train_full), imbalance_ratio)
            except Exception as e:
                logger.warning("SMOTE failed: %s — training without resampling", e)

        # ── Stratified K-Fold CV ──
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_f1s = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full), 1):
            X_tr, X_val = X_train_full[train_idx], X_train_full[val_idx]
            y_tr, y_val = y_train_full[train_idx], y_train_full[val_idx]

            clf = xgb.XGBClassifier(
                objective="multi:softprob",
                num_class=6,
                max_depth=6,
                n_estimators=300,
                learning_rate=0.1,
                eval_metric="mlogloss",
                early_stopping_rounds=20,
                verbosity=0,
                random_state=42,
            )
            clf.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            y_pred = clf.predict(X_val)
            f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
            fold_f1s.append(f1)
            logger.info("Fold %d/%d: F1-macro = %.4f", fold, n_folds, f1)

        mean_f1 = float(np.mean(fold_f1s))
        std_f1 = float(np.std(fold_f1s))
        logger.info("CV F1-macro: %.4f +/- %.4f", mean_f1, std_f1)

        # ── Train final model on full data ──
        self.model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=6,
            max_depth=6,
            n_estimators=300,
            learning_rate=0.1,
            eval_metric="mlogloss",
            verbosity=0,
            random_state=42,
        )
        self.model.fit(X_train_full, y_train_full, verbose=False)

        # Full-data metrics
        y_pred_full = self.model.predict(X.values.astype(np.float32))
        f1_full = f1_score(y, y_pred_full, average="macro", zero_division=0)
        cm = confusion_matrix(y, y_pred_full)
        report = classification_report(y, y_pred_full, target_names=CLASS_NAMES, zero_division=0)
        logger.info("Full-data F1-macro: %.4f", f1_full)
        logger.info("Classification report:\n%s", report)

        # Save model
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(self._models_dir / "confidence_xgb.json"))

        self._cv_results = {
            "cv_f1_macro_mean": round(mean_f1, 4),
            "cv_f1_macro_std": round(std_f1, 4),
            "full_f1_macro": round(f1_full, 4),
            "n_samples": len(y),
            "n_features": X.shape[1],
            "n_classes": 6,
            "imbalance_ratio": round(imbalance_ratio, 1),
            "smote_applied": use_smote and imbalance_ratio > 5,
        }

        self._set_idle()
        return self._cv_results

    # ──────────────────────────────────────────────────────
    # 4. Inference
    # ──────────────────────────────────────────────────────

    def classify(self, interaction: dict[str, Any]) -> ClassificationResult:
        """Classify a single interaction."""
        df = pd.DataFrame([interaction])
        results = self.classify_batch(interactions=df)
        return results[0] if results else ClassificationResult(
            confidence_class=3, class_name="CLEAR_GAP",
            skill_delta=-0.15, probabilities=[0]*6,
        )

    def classify_batch(
        self,
        user_id: str | None = None,
        interactions: pd.DataFrame | list | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Classify a batch of interactions.

        Returns dict with 'classes', 'class_names', 'skill_deltas',
        'mean_confidence', compatible with orchestrator pipelines.
        """
        if interactions is None:
            return {"classes": [], "class_names": [], "skill_deltas": [], "mean_confidence": 0}

        if isinstance(interactions, list):
            df = pd.DataFrame(interactions)
        else:
            df = interactions.copy()

        if len(df) == 0:
            return {"classes": [], "class_names": [], "skill_deltas": [], "mean_confidence": 0}

        # Ensure required columns
        for col, default in [("elapsed_time", self._global_median_elapsed),
                             ("changed_answer", False), ("correct", False)]:
            if col not in df.columns:
                df[col] = default

        if self.model is not None:
            X = self._build_features(df)
            preds = self.model.predict(X.values.astype(np.float32))
            probs = self.model.predict_proba(X.values.astype(np.float32))
        else:
            # Fallback: rule-based
            preds = self._assign_labels(df).values
            probs = np.eye(6)[preds]

        classes = [int(p) for p in preds]
        class_names = [CLASS_NAMES[c] for c in classes]
        deltas = [SKILL_DELTAS[ConfidenceClass(c)] for c in classes]

        return {
            "user_id": user_id,
            "classes": classes,
            "class_names": class_names,
            "skill_deltas": deltas,
            "mean_confidence": round(float(np.mean(classes)), 2),
            "n_classified": len(classes),
        }

    def classify_single(
        self,
        user_id: str | None = None,
        interaction: dict | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Classify a single interaction (continuous pipeline)."""
        if interaction is None:
            return {"class": 3, "class_name": "CLEAR_GAP", "skill_delta": -0.15, "rerank_needed": False}

        result = self.classify_batch(user_id=user_id, interactions=[interaction])
        cls = result["classes"][0] if result["classes"] else 3

        return {
            "class": cls,
            "class_name": CLASS_NAMES[cls],
            "skill_delta": SKILL_DELTAS[ConfidenceClass(cls)],
            "rerank_needed": cls in (ConfidenceClass.FALSE_CONFIDENCE, ConfidenceClass.CLEAR_GAP),
        }

    # ──────────────────────────────────────────────────────
    # 5. Skill delta
    # ──────────────────────────────────────────────────────

    @staticmethod
    def get_skill_delta(cls: int | ConfidenceClass) -> float:
        """Return the skill update delta for a confidence class."""
        return SKILL_DELTAS[ConfidenceClass(cls)]

    # ──────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────

    def get_feature_importance(self) -> pd.Series:
        """Return feature importance from the trained model."""
        if self.model is None:
            return pd.Series(dtype=float)
        imp = self.model.feature_importances_
        return pd.Series(imp, index=FEATURE_NAMES).sort_values(ascending=False)

    @property
    def cv_results(self) -> dict[str, float]:
        return self._cv_results
