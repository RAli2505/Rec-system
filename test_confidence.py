"""End-to-end test for ConfidenceAgent."""

import logging
import time

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

import numpy as np
import pandas as pd
from data.loader import EdNetLoader
from agents.confidence_agent import (
    ConfidenceAgent, ConfidenceClass, CLASS_NAMES, SKILL_DELTAS, FEATURE_NAMES,
)
from agents.diagnostic_agent import DiagnosticAgent
from agents.base_agent import BaseAgent
from agents.orchestrator import Orchestrator

t0 = time.time()
loader = EdNetLoader(data_dir="data/raw")
interactions = loader.load_interactions(sample_users=500)
print(f"Data loaded in {time.time()-t0:.1f}s")
print(f"  Interactions: {len(interactions):,}, Users: {interactions['user_id'].nunique()}")

# Calibrate IRT for difficulty features
diag = DiagnosticAgent()
irt_params = diag.calibrate_from_interactions(interactions, min_answers_per_q=5)

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 1: 6 classes and skill deltas defined")
print("=" * 60)
assert len(CLASS_NAMES) == 6
assert len(SKILL_DELTAS) == 6
assert ConfidenceClass.SOLID == 0
assert ConfidenceClass.DOUBT_INCORRECT == 5
assert SKILL_DELTAS[ConfidenceClass.SOLID] == +0.15
assert SKILL_DELTAS[ConfidenceClass.CLEAR_GAP] == -0.15
assert SKILL_DELTAS[ConfidenceClass.DOUBT_CORRECT] == +0.03
print(f"  Classes: {CLASS_NAMES}")
print(f"  Deltas: { {CLASS_NAMES[i]: SKILL_DELTAS[ConfidenceClass(i)] for i in range(6)} }")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 2: Auto-labelling")
print("=" * 60)
conf = ConfidenceAgent()
df = interactions.dropna(subset=["elapsed_time", "correct", "changed_answer"]).copy()
labels = conf._assign_labels(df)
assert len(labels) == len(df)
assert set(labels.unique()).issubset({0, 1, 2, 3, 4, 5})
dist = labels.value_counts().sort_index()
print("  Label distribution:")
for cls_id, count in dist.items():
    print(f"    {CLASS_NAMES[cls_id]:20s}: {count:>6,} ({count/len(df)*100:.1f}%)")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 3: Feature engineering (17 features)")
print("=" * 60)
feats = conf._build_features(df.head(100))
assert feats.shape[1] == len(FEATURE_NAMES), f"Expected {len(FEATURE_NAMES)} features, got {feats.shape[1]}"
assert list(feats.columns) == FEATURE_NAMES
assert not feats.isnull().any().any(), "NaN in features"
print(f"  Shape: {feats.shape}")
print(f"  Columns: {list(feats.columns)}")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 4: Train XGBoost (5-fold CV)")
print("=" * 60)
metrics = conf.train(interactions, irt_params=irt_params)
assert "cv_f1_macro_mean" in metrics
assert "full_f1_macro" in metrics
assert metrics["n_classes"] == 6
assert metrics["n_features"] == len(FEATURE_NAMES)
print(f"  CV F1-macro: {metrics['cv_f1_macro_mean']:.4f} +/- {metrics['cv_f1_macro_std']:.4f}")
print(f"  Full F1-macro: {metrics['full_f1_macro']:.4f}")
print(f"  SMOTE applied: {metrics['smote_applied']}")
print(f"  Imbalance ratio: {metrics['imbalance_ratio']}")
assert conf.model is not None
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 5: classify_batch")
print("=" * 60)
batch_result = conf.classify_batch(
    user_id="test_user",
    interactions=interactions.head(20),
)
assert "classes" in batch_result
assert "class_names" in batch_result
assert "skill_deltas" in batch_result
assert len(batch_result["classes"]) == 20
assert all(0 <= c <= 5 for c in batch_result["classes"])
print(f"  Classified {batch_result['n_classified']} interactions")
print(f"  Classes: {batch_result['classes'][:5]}...")
print(f"  Names: {batch_result['class_names'][:3]}...")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 6: classify_single")
print("=" * 60)
single = interactions.iloc[0].to_dict()
result = conf.classify_single(user_id="u1", interaction=single)
assert "class" in result
assert "class_name" in result
assert "skill_delta" in result
assert "rerank_needed" in result
assert 0 <= result["class"] <= 5
print(f"  Class: {result['class']} ({result['class_name']})")
print(f"  Delta: {result['skill_delta']}")
print(f"  Rerank: {result['rerank_needed']}")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 7: get_skill_delta")
print("=" * 60)
for i in range(6):
    delta = ConfidenceAgent.get_skill_delta(i)
    expected = SKILL_DELTAS[ConfidenceClass(i)]
    assert delta == expected, f"Class {i}: expected {expected}, got {delta}"
print("  All 6 deltas correct")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 8: Feature importance")
print("=" * 60)
imp = conf.get_feature_importance()
assert len(imp) == len(FEATURE_NAMES)
print(f"  Top-5 features:")
for feat, val in imp.head(5).items():
    print(f"    {feat:30s}: {val:.4f}")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 9: Model saved")
print("=" * 60)
from pathlib import Path
assert Path("models/confidence_xgb.json").exists()
print("  models/confidence_xgb.json exists")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 10: Orchestrator integration")
print("=" * 60)


class Stub(BaseAgent):
    def initialize(self, **kw):
        pass


class StubDiag(Stub):
    name = "diagnostic"
    def run_diagnostic(self, uid):
        return {
            "responses": [{"question_id": "q1", "correct": True}],
            "ability": 0.5, "se": 0.3,
        }


class StubKG(Stub):
    name = "knowledge_graph"
    def handle_cold_start(self, **kw):
        return {"mastered_tags": [], "gap_tags": [], "recommendations": [], "n_recommendations": 0}


class StubRec(Stub):
    name = "recommendation"
    def recommend(self, **kw):
        return {"items": ["q1"]}


class StubPers(Stub):
    name = "personalization"
    def assign_cluster(self, **kw):
        return {"cluster_id": 1}


orch = Orchestrator()
for a in [StubDiag(), conf, StubKG(), StubRec(), StubPers()]:
    orch.register_agent(a)

result = orch.cold_start_pipeline("conf_test_user")
assert "confidence" in result
conf_r = result["confidence"]
assert "classes" in conf_r
print(f"  Pipeline confidence: {conf_r['n_classified']} classified")
print("  PASS")

# ═══════════════════════════════════════════════════
elapsed = time.time() - t0
print()
print("=" * 60)
print(f"ALL 10 TESTS PASSED in {elapsed:.1f}s")
print("=" * 60)
