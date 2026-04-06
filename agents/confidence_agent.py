"""
Confidence Agent for MARS.

Rule-based behavioural confidence classifier using response timing,
correctness, and answer-change patterns.  Deterministic and interpretable.

Supports multiple confidence granularities:
2-class, 3-class, 4-class, and the original 6-class MARS scheme.

Classification rules (6-class):
  SOLID              — correct + fast + no change  (confident mastery)
  UNSURE_CORRECT     — correct + slow + no change  (uncertain success)
  FALSE_CONFIDENCE   — incorrect + fast + no change (risky overconfidence)
  CLEAR_GAP          — incorrect + slow + no change (definite knowledge gap)
  DOUBT_CORRECT      — correct + changed answer    (self-correction success)
  DOUBT_INCORRECT    — incorrect + changed answer   (self-correction failure)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base_agent import BaseAgent

logger = logging.getLogger("mars.agent.confidence")


class ConfidenceClass(IntEnum):
    SOLID = 0
    UNSURE_CORRECT = 1
    FALSE_CONFIDENCE = 2
    CLEAR_GAP = 3
    DOUBT_CORRECT = 4
    DOUBT_INCORRECT = 5


CONFIDENCE_SCHEMAS: dict[int, dict[str, Any]] = {
    2: {
        "class_names": ["CORRECT", "INCORRECT"],
        "skill_deltas": {"CORRECT": 0.10, "INCORRECT": -0.10},
        "rerank_classes": {"INCORRECT"},
        "risk_classes": {"INCORRECT"},
        "description": "Binary correctness baseline.",
    },
    3: {
        "class_names": ["MASTERED", "LEARNING", "STRUGGLING"],
        "skill_deltas": {
            "MASTERED": 0.15,
            "LEARNING": 0.03,
            "STRUGGLING": -0.12,
        },
        "rerank_classes": {"STRUGGLING"},
        "risk_classes": {"STRUGGLING"},
        "description": "Three-class behavioural confidence scheme.",
    },
    4: {
        "class_names": [
            "CONFIDENT_CORRECT",
            "UNCERTAIN_CORRECT",
            "CONFIDENT_INCORRECT",
            "UNCERTAIN_INCORRECT",
        ],
        "skill_deltas": {
            "CONFIDENT_CORRECT": 0.15,
            "UNCERTAIN_CORRECT": 0.05,
            "CONFIDENT_INCORRECT": -0.10,
            "UNCERTAIN_INCORRECT": -0.15,
        },
        "rerank_classes": {"CONFIDENT_INCORRECT", "UNCERTAIN_INCORRECT"},
        "risk_classes": {"CONFIDENT_INCORRECT", "UNCERTAIN_INCORRECT"},
        "description": "2x2 correctness-by-certainty matrix.",
    },
    6: {
        "class_names": [
            "SOLID",
            "UNSURE_CORRECT",
            "FALSE_CONFIDENCE",
            "CLEAR_GAP",
            "DOUBT_CORRECT",
            "DOUBT_INCORRECT",
        ],
        "skill_deltas": {
            "SOLID": 0.15,
            "UNSURE_CORRECT": 0.05,
            "FALSE_CONFIDENCE": -0.10,
            "CLEAR_GAP": -0.15,
            "DOUBT_CORRECT": 0.03,
            "DOUBT_INCORRECT": -0.12,
        },
        "rerank_classes": {"FALSE_CONFIDENCE", "CLEAR_GAP"},
        "risk_classes": {"FALSE_CONFIDENCE", "CLEAR_GAP", "DOUBT_INCORRECT"},
        "description": "Original MARS six-class behavioural taxonomy.",
    },
}

DEFAULT_CONFIDENCE_N_CLASSES = 6


def get_confidence_schema(
    n_classes: int = DEFAULT_CONFIDENCE_N_CLASSES,
) -> dict[str, Any]:
    """Return metadata for the requested confidence scheme."""
    if n_classes not in CONFIDENCE_SCHEMAS:
        raise ValueError(
            f"Unsupported confidence schema {n_classes}. "
            f"Available: {sorted(CONFIDENCE_SCHEMAS)}"
        )

    base = CONFIDENCE_SCHEMAS[n_classes]
    class_names = list(base["class_names"])
    deltas_by_name = dict(base["skill_deltas"])
    return {
        "n_classes": n_classes,
        "class_names": class_names,
        "skill_deltas": deltas_by_name,
        "skill_deltas_by_id": {
            idx: deltas_by_name[name] for idx, name in enumerate(class_names)
        },
        "rerank_ids": {
            idx for idx, name in enumerate(class_names)
            if name in base.get("rerank_classes", set())
        },
        "risk_ids": {
            idx for idx, name in enumerate(class_names)
            if name in base.get("risk_classes", set())
        },
        "description": base.get("description", ""),
    }


def count_risk_confidence_events(
    class_names: list[str] | None,
    n_classes: int = DEFAULT_CONFIDENCE_N_CLASSES,
) -> int:
    """Count classes that should feed the recommendation risk feature."""
    if not class_names:
        return 0
    schema = get_confidence_schema(n_classes)
    risky_names = {schema["class_names"][idx] for idx in schema["risk_ids"]}
    return sum(1 for name in class_names if name in risky_names)


FEATURE_NAMES = [
    "elapsed_time_normalized",
    # "changed_answer" removed — direct component of label (DOUBT classes)
    "response_count",
    # "is_correct" removed — direct component of label (data leakage)
    "question_difficulty",
    "time_of_day_sin",
    "time_of_day_cos",
    "speed_vs_difficulty",
    "user_rolling_accuracy",
    "tag_familiarity",
    "part_1",
    "part_2",
    "part_3",
    "part_4",
    "part_5",
    "part_6",
    "part_7",
]


@dataclass
class ClassificationResult:
    confidence_class: int
    class_name: str
    skill_delta: float
    probabilities: list[float]


class ConfidenceAgent(BaseAgent):
    """
    Rule-based behavioural confidence classifier.

    Uses response timing (fast/slow relative to question median),
    correctness, and answer-change patterns to classify student
    confidence level. Fully deterministic and interpretable.
    """

    name = "confidence"
    REQUIRED_COLUMNS = {
        "train": [
            "user_id",
            "timestamp",
            "question_id",
            "correct",
            "elapsed_time",
            "changed_answer",
        ],
    }

    def __init__(self, n_classes: int | None = None) -> None:
        super().__init__()

        configured = int(self._config.get("n_classes", DEFAULT_CONFIDENCE_N_CLASSES))
        self.n_classes = int(n_classes if n_classes is not None else configured)
        self.class_schema = get_confidence_schema(self.n_classes)
        self.class_names = self.class_schema["class_names"]
        self.skill_deltas = self.class_schema["skill_deltas_by_id"]
        self._rerank_ids = self.class_schema["rerank_ids"]
        self._risk_ids = self.class_schema["risk_ids"]

        self._models_dir = Path("models")
        self._median_elapsed: dict[str, float] = {}
        self._global_median_elapsed: float = 15_000.0
        self._irt_difficulty: dict[str, float] = {}
        self._cv_results: dict[str, float] = {}
        self._trained = False

    @staticmethod
    def _to_label_ids(preds: np.ndarray) -> np.ndarray:
        """Convert model outputs to integer class ids."""
        preds = np.asarray(preds)
        if preds.ndim == 1:
            return preds.astype(int)
        if preds.shape[1] == 1:
            return (preds[:, 0] >= 0.5).astype(int)
        return np.argmax(preds, axis=1).astype(int)

    def initialize(self, **kwargs: Any) -> None:
        if "interactions_df" in kwargs:
            self.train(kwargs["interactions_df"], irt_params=kwargs.get("irt_params"))

    def receive_message(self, message):
        super().receive_message(message)
        action = message.data.get("action")
        if action == "classify":
            return self.classify(message.data["interaction"])
        return None

    def _assign_labels(self, df: pd.DataFrame) -> pd.Series:
        """Assign rule-based labels for the active schema."""
        med_elapsed = df.groupby("question_id")["elapsed_time"].transform("median")
        is_fast = df["elapsed_time"] < med_elapsed
        is_correct = df["correct"].astype(bool)
        changed = df["changed_answer"].astype(bool)

        labels = np.zeros(len(df), dtype=np.int64)

        if self.n_classes == 2:
            labels[~is_correct.values] = 1
            return pd.Series(labels, index=df.index, dtype=int)

        if self.n_classes == 3:
            labels[:] = 1
            labels[~is_correct.values] = 2
            labels[(is_correct & is_fast & ~changed).values] = 0
            return pd.Series(labels, index=df.index, dtype=int)

        if self.n_classes == 4:
            labels[:] = 3
            labels[(is_correct & is_fast & ~changed).values] = 0
            labels[(is_correct & (~is_fast | changed)).values] = 1
            labels[(~is_correct & is_fast & ~changed).values] = 2
            labels[(~is_correct & (~is_fast | changed)).values] = 3
            return pd.Series(labels, index=df.index, dtype=int)

        labels[:] = int(ConfidenceClass.CLEAR_GAP)
        labels[(changed & is_correct).values] = int(ConfidenceClass.DOUBT_CORRECT)
        labels[(changed & ~is_correct).values] = int(ConfidenceClass.DOUBT_INCORRECT)

        not_changed = ~changed
        labels[(not_changed & is_fast & is_correct).values] = int(ConfidenceClass.SOLID)
        labels[(not_changed & ~is_fast & is_correct).values] = int(ConfidenceClass.UNSURE_CORRECT)
        labels[(not_changed & is_fast & ~is_correct).values] = int(ConfidenceClass.FALSE_CONFIDENCE)
        labels[(not_changed & ~is_fast & ~is_correct).values] = int(ConfidenceClass.CLEAR_GAP)
        return pd.Series(labels, index=df.index, dtype=int)

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame(index=df.index)

        med = df.groupby("question_id")["elapsed_time"].transform("median")
        med = med.replace(0, self._global_median_elapsed)
        feats["elapsed_time_normalized"] = (
            (df["elapsed_time"] / med).clip(0, 10).fillna(1.0)
        )
        # changed_answer and is_correct removed: they are direct determinants of labels
        # (DOUBT classes use changed_answer; SOLID/GAP/FALSE_CONFIDENCE use is_correct)
        # Keeping them causes data leakage (cv_f1 → 1.0). Model now learns from
        # behavioral proxies only: timing, difficulty, past accuracy.
        feats["response_count"] = (
            df["response_count"].fillna(1).astype(int)
            if "response_count" in df.columns
            else 1
        )
        feats["question_difficulty"] = (
            df["question_id"].astype(str).map(self._irt_difficulty).fillna(0.0)
        )

        if "timestamp" in df.columns:
            seconds_in_day = 86400
            hour_frac = (df["timestamp"] / 1000 % seconds_in_day) / seconds_in_day
            feats["time_of_day_sin"] = np.sin(2 * np.pi * hour_frac)
            feats["time_of_day_cos"] = np.cos(2 * np.pi * hour_frac)
        else:
            feats["time_of_day_sin"] = 0.0
            feats["time_of_day_cos"] = 0.0

        diff = feats["question_difficulty"] + 4
        feats["speed_vs_difficulty"] = (
            feats["elapsed_time_normalized"] / diff.clip(0.1)
        ).clip(0, 10)

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

        if "tags" in df.columns and "user_id" in df.columns:
            feats["tag_familiarity"] = df.groupby("user_id").cumcount().clip(0, 200) / 200.0
        else:
            feats["tag_familiarity"] = 0.0

        if "part_id" in df.columns:
            for p in range(1, 8):
                feats[f"part_{p}"] = (df["part_id"] == p).astype(int)
        else:
            for p in range(1, 8):
                feats[f"part_{p}"] = 0

        return feats[FEATURE_NAMES]

    def _compute_sample_weights(self, y: np.ndarray) -> np.ndarray:
        classes, counts = np.unique(y, return_counts=True)
        n = len(y)
        weight_map = {c: n / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
        return np.array([weight_map[yi] for yi in y], dtype=np.float32)

    def train(
        self,
        interactions_df: pd.DataFrame,
        irt_params: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        Calibrate the rule-based classifier: compute per-question median
        elapsed times (used as fast/slow threshold) and store IRT difficulty.

        No ML model is trained — classification is purely rule-based.
        """
        self._set_processing()
        self._validate_dataframe(interactions_df, "train")

        df = interactions_df.copy()
        df = df.dropna(subset=["elapsed_time", "correct", "changed_answer"])

        # Calibrate timing thresholds
        self._global_median_elapsed = float(df["elapsed_time"].median())
        self._median_elapsed = df.groupby("question_id")["elapsed_time"].median().to_dict()

        if irt_params is not None:
            for qid, b_val in zip(irt_params.question_ids, irt_params.b):
                self._irt_difficulty[str(qid)] = float(b_val)

        # Validate: compute class distribution on training data
        labels = self._assign_labels(df)
        y = labels.values
        class_counts = pd.Series(y).value_counts().sort_index()

        logger.info(
            "Rule-based confidence calibrated: %d samples, %d classes",
            len(y), self.n_classes,
        )
        for cls_id, count in class_counts.items():
            name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"Class_{cls_id}"
            logger.info(
                "  %s (id=%d): %d samples (%.1f%%)",
                name, cls_id, count, 100 * count / len(y),
            )

        # Rule-based → F1=1.0 by definition (labels ARE the rules)
        self._cv_results = {
            "method": "rule_based",
            "cv_f1_macro_mean": 1.0,
            "cv_f1_macro_std": 0.0,
            "full_f1_macro": 1.0,
            "n_samples": len(y),
            "n_classes": self.n_classes,
            "class_distribution": {
                self.class_names[int(k)]: int(v)
                for k, v in class_counts.items()
                if int(k) < len(self.class_names)
            },
        }

        self._trained = True
        self._set_idle()
        return self._cv_results

    def classify(self, interaction: dict[str, Any]) -> ClassificationResult:
        df = pd.DataFrame([interaction])
        result = self.classify_batch(interactions=df)
        if result["classes"]:
            cls = result["classes"][0]
            return ClassificationResult(
                confidence_class=cls,
                class_name=result["class_names"][0],
                skill_delta=result["skill_deltas"][0],
                probabilities=[],
            )

        fallback_cls = max(self.n_classes - 1, 0)
        return ClassificationResult(
            confidence_class=fallback_cls,
            class_name=self.class_names[fallback_cls],
            skill_delta=self.skill_deltas[fallback_cls],
            probabilities=[0.0] * self.n_classes,
        )

    def classify_batch(
        self,
        user_id: str | None = None,
        interactions: pd.DataFrame | list | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        empty = {
            "user_id": user_id,
            "classes": [],
            "class_names": [],
            "skill_deltas": [],
            "mean_confidence": 0.0,
            "n_classified": 0,
            "n_classes": self.n_classes,
            "schema": self.class_schema["description"],
        }
        if interactions is None:
            return empty

        df = pd.DataFrame(interactions) if isinstance(interactions, list) else interactions.copy()
        if len(df) == 0:
            return empty

        for col, default in [
            ("elapsed_time", self._global_median_elapsed),
            ("changed_answer", False),
            ("correct", False),
        ]:
            if col not in df.columns:
                df[col] = default
        if "question_id" not in df.columns:
            df["question_id"] = "__unknown__"

        preds = self._assign_labels(df).values
        probs = np.eye(self.n_classes, dtype=np.float32)[preds]

        classes = [int(p) for p in preds]
        class_names = [self.class_names[c] for c in classes]
        deltas = [self.skill_deltas[c] for c in classes]

        return {
            "user_id": user_id,
            "classes": classes,
            "class_names": class_names,
            "skill_deltas": deltas,
            "mean_confidence": round(float(np.mean(np.max(probs, axis=1))), 4),
            "n_classified": len(classes),
            "n_classes": self.n_classes,
            "schema": self.class_schema["description"],
        }

    def classify_single(
        self,
        user_id: str | None = None,
        interaction: dict | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if interaction is None:
            fallback_cls = max(self.n_classes - 1, 0)
            return {
                "class": fallback_cls,
                "class_name": self.class_names[fallback_cls],
                "skill_delta": self.skill_deltas[fallback_cls],
                "rerank_needed": fallback_cls in self._rerank_ids,
                "n_classes": self.n_classes,
            }

        result = self.classify_batch(user_id=user_id, interactions=[interaction])
        cls = result["classes"][0] if result["classes"] else max(self.n_classes - 1, 0)
        return {
            "class": cls,
            "class_name": self.class_names[cls],
            "skill_delta": self.skill_deltas[cls],
            "rerank_needed": cls in self._rerank_ids,
            "n_classes": self.n_classes,
        }

    @staticmethod
    def get_skill_delta(
        cls: int | ConfidenceClass,
        n_classes: int = DEFAULT_CONFIDENCE_N_CLASSES,
    ) -> float:
        schema = get_confidence_schema(n_classes)
        return schema["skill_deltas_by_id"][int(cls)]

    def get_feature_importance(self) -> pd.Series:
        """Rule-based classifier — no feature importances. Returns empty."""
        return pd.Series(dtype=float)

    @property
    def cv_results(self) -> dict[str, float]:
        return self._cv_results
