"""
Confidence Agent for MARS.

Supports multiple confidence granularities:
2-class, 3-class, 4-class, and the original 6-class MARS scheme.
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, GroupKFold

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
    XGBoost-based confidence classifier with configurable class granularity.
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
        from .utils import set_global_seed

        set_global_seed(self._config.get("seed", 42))

        configured = int(self._config.get("n_classes", DEFAULT_CONFIDENCE_N_CLASSES))
        self.n_classes = int(n_classes if n_classes is not None else configured)
        self.class_schema = get_confidence_schema(self.n_classes)
        self.class_names = self.class_schema["class_names"]
        self.skill_deltas = self.class_schema["skill_deltas_by_id"]
        self._rerank_ids = self.class_schema["rerank_ids"]
        self._risk_ids = self.class_schema["risk_ids"]

        self._n_folds = self._config.get("cv_folds", 5)
        self._use_smote_default = self._config.get("use_smote", False)
        self._use_class_weight_default = self._config.get("use_class_weight", True)

        self.model: xgb.XGBClassifier | None = None
        self._models_dir = Path("models")
        self._median_elapsed: dict[str, float] = {}
        self._global_median_elapsed: float = 15_000.0
        self._irt_difficulty: dict[str, float] = {}
        self._cv_results: dict[str, float] = {}

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
        use_smote: bool | None = None,
        use_class_weight: bool | None = None,
        n_folds: int | None = None,
    ) -> dict[str, float]:
        if use_smote is None:
            use_smote = self._use_smote_default
        if use_class_weight is None:
            use_class_weight = self._use_class_weight_default
        if n_folds is None:
            n_folds = self._n_folds

        from .utils import set_global_seed

        set_global_seed(self._config.get("seed", 42))

        self._set_processing()
        self._validate_dataframe(interactions_df, "train")

        df = interactions_df.copy()
        df = df.dropna(subset=["elapsed_time", "correct", "changed_answer"])

        self._global_median_elapsed = float(df["elapsed_time"].median())
        self._median_elapsed = df.groupby("question_id")["elapsed_time"].median().to_dict()

        if irt_params is not None:
            for qid, b_val in zip(irt_params.question_ids, irt_params.b):
                self._irt_difficulty[str(qid)] = float(b_val)

        labels = self._assign_labels(df)
        X = self._build_features(df)
        y = labels.values

        class_counts = pd.Series(y).value_counts()
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / max(min_count, 1)

        X_train_full = X.values.astype(np.float32)
        y_train_full = y
        sample_weights_full = (
            self._compute_sample_weights(y_train_full) if use_class_weight else None
        )
        apply_smote = bool(use_smote and imbalance_ratio > 5)

        # Use GroupKFold by student to prevent data leakage across folds
        user_groups = df["user_id"].values if "user_id" in df.columns else np.arange(len(df))
        n_unique_users = len(np.unique(user_groups))

        fold_f1s = []
        if len(class_counts) > 1 and int(min_count) >= 2 and n_unique_users >= n_folds:
            effective_folds = min(int(n_folds), n_unique_users)
            # GroupKFold: no student appears in both train and val
            gkf = GroupKFold(n_splits=effective_folds)

            for fold, (train_idx, val_idx) in enumerate(
                gkf.split(X_train_full, y_train_full, groups=user_groups), 1
            ):
                X_tr, X_val = X_train_full[train_idx], X_train_full[val_idx]
                y_tr, y_val = y_train_full[train_idx], y_train_full[val_idx]
                sw_tr = sample_weights_full[train_idx] if sample_weights_full is not None else None

                if apply_smote:
                    try:
                        from imblearn.over_sampling import SMOTE

                        fold_min = int(pd.Series(y_tr).value_counts().min())
                        smote = SMOTE(
                            random_state=42,
                            k_neighbors=min(5, fold_min - 1) if fold_min > 1 else 1,
                        )
                        X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
                        sw_tr = None
                    except Exception as exc:
                        logger.warning("SMOTE failed on fold %d: %s", fold, exc)
                        sw_tr = self._compute_sample_weights(y_tr)

                xgb_cfg = self._config.get("xgboost", {})
                clf = xgb.XGBClassifier(
                    objective="multi:softprob",
                    num_class=self.n_classes,
                    max_depth=xgb_cfg.get("max_depth", 6),
                    n_estimators=xgb_cfg.get("n_estimators", 300),
                    learning_rate=xgb_cfg.get("learning_rate", 0.1),
                    eval_metric=xgb_cfg.get("eval_metric", "mlogloss"),
                    early_stopping_rounds=xgb_cfg.get("early_stopping_rounds", 20),
                    verbosity=0,
                    random_state=42,
                )
                clf.fit(
                    X_tr,
                    y_tr,
                    sample_weight=sw_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
                y_pred = self._to_label_ids(clf.predict(X_val))
                fold_f1s.append(f1_score(y_val, y_pred, average="macro", zero_division=0))
        else:
            logger.warning(
                "Skipping stratified CV for %d-class confidence due to rare labels.",
                self.n_classes,
            )

        mean_f1 = float(np.mean(fold_f1s)) if fold_f1s else float("nan")
        std_f1 = float(np.std(fold_f1s)) if fold_f1s else float("nan")

        X_final, y_final = X_train_full, y_train_full
        sw_final = sample_weights_full
        if apply_smote:
            try:
                from imblearn.over_sampling import SMOTE

                final_min = int(pd.Series(y_final).value_counts().min())
                smote_final = SMOTE(
                    random_state=42,
                    k_neighbors=min(5, final_min - 1) if final_min > 1 else 1,
                )
                X_final, y_final = smote_final.fit_resample(X_final, y_final)
                sw_final = None
            except Exception as exc:
                logger.warning("SMOTE failed for final model: %s", exc)
                sw_final = self._compute_sample_weights(y_final)

        xgb_cfg = self._config.get("xgboost", {})
        self.model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=self.n_classes,
            max_depth=xgb_cfg.get("max_depth", 6),
            n_estimators=xgb_cfg.get("n_estimators", 300),
            learning_rate=xgb_cfg.get("learning_rate", 0.1),
            eval_metric=xgb_cfg.get("eval_metric", "mlogloss"),
            verbosity=0,
            random_state=42,
        )
        self.model.fit(X_final, y_final, sample_weight=sw_final, verbose=False)

        y_pred_full = self._to_label_ids(self.model.predict(X.values.astype(np.float32)))
        f1_full = f1_score(y, y_pred_full, average="macro", zero_division=0)
        confusion_matrix(y, y_pred_full, labels=list(range(self.n_classes)))
        classification_report(
            y,
            y_pred_full,
            labels=list(range(self.n_classes)),
            target_names=self.class_names,
            zero_division=0,
        )

        self._models_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(self._models_dir / f"confidence_xgb_{self.n_classes}c.json"))

        balance_method = "smote" if apply_smote else ("class_weight" if use_class_weight else "none")
        self._cv_results = {
            "cv_f1_macro_mean": round(mean_f1, 4),
            "cv_f1_macro_std": round(std_f1, 4),
            "full_f1_macro": round(f1_full, 4),
            "n_samples": len(y),
            "n_features": X.shape[1],
            "n_classes": self.n_classes,
            "imbalance_ratio": round(imbalance_ratio, 1),
            "balance_method": balance_method,
        }

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

        if self.model is not None:
            X = self._build_features(df)
            preds = self._to_label_ids(self.model.predict(X.values.astype(np.float32)))
            probs = self.model.predict_proba(X.values.astype(np.float32))
        else:
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
        if self.model is None:
            return pd.Series(dtype=float)
        return pd.Series(self.model.feature_importances_, index=FEATURE_NAMES).sort_values(
            ascending=False
        )

    @property
    def cv_results(self) -> dict[str, float]:
        return self._cv_results
