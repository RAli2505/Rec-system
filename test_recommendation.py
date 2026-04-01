"""End-to-end test for RecommendationAgent."""

import logging
import time

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

import numpy as np
import pandas as pd
from data.loader import EdNetLoader
from agents.kg_agent import KnowledgeGraphAgent
from agents.diagnostic_agent import DiagnosticAgent
from agents.recommendation_agent import (
    RecommendationAgent, Rec, DEFAULT_STRATEGY_PRIORS, LAMBDAMART_FEATURES,
)
from agents.base_agent import BaseAgent
from agents.orchestrator import Orchestrator

t0 = time.time()
loader = EdNetLoader(data_dir="data/raw")
questions = loader.load_questions()
lectures = loader.load_lectures()
interactions = loader.load_interactions(sample_users=300)
print(f"Data loaded in {time.time()-t0:.1f}s")

# Build supporting agents
kg = KnowledgeGraphAgent()
kg.build_graph(questions, lectures)
kg.update_difficulties(interactions)
kg.build_prerequisites(interactions)

diag = DiagnosticAgent()
diag.calibrate_from_interactions(interactions, min_answers_per_q=5)

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 1: Build content index (SBERT + FAISS)")
print("=" * 60)
rec = RecommendationAgent()
rec.build_content_index(lectures, questions)
assert rec._faiss_index is not None
assert rec._faiss_index.ntotal > 1000  # lectures + sample questions
assert rec._sbert_model is not None
print(f"  FAISS index: {rec._faiss_index.ntotal} items")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 2: Train collaborative (SVD/ALS)")
print("=" * 60)
rec.train_collaborative(interactions, n_factors=64)
assert rec._als_user_factors is not None
assert rec._als_item_factors is not None
print(f"  User factors: {rec._als_user_factors.shape}")
print(f"  Item factors: {rec._als_item_factors.shape}")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 3: Knowledge-based recommendations")
print("=" * 60)
uid = str(interactions["user_id"].iloc[0])
user_df = interactions[interactions["user_id"] == interactions["user_id"].iloc[0]]
diag_resp = [{"question_id": r["question_id"], "correct": bool(r["correct"])}
             for _, r in user_df.head(5).iterrows()]
kg_profile = kg.handle_cold_start(user_id=uid, diagnostic={"responses": diag_resp})

kb_recs = rec.get_knowledge_based(uid, kg_profile=kg_profile, n=10)
assert isinstance(kb_recs, list)
if kb_recs:
    assert isinstance(kb_recs[0], Rec)
    assert kb_recs[0].strategy == "knowledge_based"
print(f"  KB recommendations: {len(kb_recs)}")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 4: Content-based recommendations")
print("=" * 60)
gap_tags = kg_profile.get("gap_tags", [])
cb_recs = rec.get_content_based(uid, gap_tags=gap_tags, n=10)
assert isinstance(cb_recs, list)
assert len(cb_recs) > 0
assert cb_recs[0].strategy == "content_based"
print(f"  CB recommendations: {len(cb_recs)}, top score: {cb_recs[0].score:.4f}")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 5: Collaborative recommendations")
print("=" * 60)
cf_recs = rec.get_collaborative(uid, n=10)
assert isinstance(cf_recs, list)
if cf_recs:
    assert cf_recs[0].strategy == "collaborative"
print(f"  CF recommendations: {len(cf_recs)}")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 6: Thompson Sampling")
print("=" * 60)
# Initial priors
weights = rec.get_ts_weights("test_ts")
assert "knowledge_based" in weights
assert "content_based" in weights
assert "collaborative" in weights
assert weights["knowledge_based"] > weights["collaborative"]  # KB has higher prior

# Select strategy
strategy = rec.select_strategy("test_ts", n_interactions=50)
assert strategy in ["knowledge_based", "content_based", "collaborative"]

# Update reward
rec.update_reward("test_ts", "knowledge_based", 1.0)
w_after = rec.get_ts_weights("test_ts")
assert w_after["knowledge_based"] >= weights["knowledge_based"]

# CF disabled with few interactions
strategy_no_cf = rec.select_strategy("test_ts_nocf", n_interactions=5)
# Run 20 times — CF should never be picked
for _ in range(20):
    s = rec.select_strategy("test_ts_nocf", n_interactions=5)
    assert s != "collaborative", "CF should be disabled with <20 interactions"
print(f"  Strategy selected: {strategy}")
print(f"  Weights: {w_after}")
print(f"  CF correctly disabled for n<20")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 7: recommend() (full pipeline)")
print("=" * 60)
result = rec.recommend(uid, kg_profile=kg_profile, n=5)
assert "items" in result
assert "strategy_selected" in result
assert "strategy_weights" in result
assert len(result["items"]) > 0
for item in result["items"]:
    assert "item_id" in item
    assert "score" in item
    assert "strategy" in item
print(f"  Recommendations: {len(result['items'])}, strategy: {result['strategy_selected']}")
print(f"  Top item: {result['items'][0]['item_id']} ({result['items'][0]['strategy']})")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 8: LambdaMART ranking features")
print("=" * 60)
candidates = [
    Rec(item_id="l1", item_type="lecture", score=0.8, strategy="knowledge_based", related_tags=[5]),
    Rec(item_id="l2", item_type="lecture", score=0.6, strategy="content_based", related_tags=[12]),
    Rec(item_id="t3", item_type="tag", score=0.4, strategy="collaborative", related_tags=[88]),
]
feats = rec._build_ranking_features(uid, candidates)
assert feats.shape == (3, 12)
assert not np.isnan(feats).any()
print(f"  Ranking features: {feats.shape}")
print(f"  Feature names: {LAMBDAMART_FEATURES}")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 9: LambdaMART train + rank")
print("=" * 60)
# Create synthetic ranking data
np.random.seed(42)
n_groups = 20
X_rank_list = []
y_rank_list = []
groups = []
for _ in range(n_groups):
    n_items = np.random.randint(5, 15)
    X_rank_list.append(np.random.rand(n_items, 12).astype(np.float32))
    y_rank_list.append(np.random.randint(0, 3, n_items).astype(np.float32))
    groups.append(n_items)

X_rank = np.vstack(X_rank_list)
y_rank = np.concatenate(y_rank_list)

ranker = rec.train_ranker(X_rank, y_rank, groups)
assert ranker is not None
assert rec._ranker is not None

# Now rank_candidates should use the trained ranker
ranked = rec.rank_candidates(uid, candidates)
assert len(ranked) == 3
assert ranked[0].score >= ranked[1].score >= ranked[2].score
print(f"  Ranker trained, {len(groups)} groups")
print(f"  Ranked: {[r.item_id for r in ranked]}")
from pathlib import Path
assert Path("models/lambdamart.txt").exists()
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
        qs = questions.sample(5, random_state=42)
        return {
            "responses": [{"question_id": r["question_id"], "correct": i % 2 == 0}
                          for i, (_, r) in enumerate(qs.iterrows())],
            "ability": 0.5, "se": 0.3,
        }


class StubConf(Stub):
    name = "confidence"
    def classify_batch(self, **kw):
        return {"classes": [0], "class_names": ["SOLID"], "skill_deltas": [0.15],
                "mean_confidence": 0, "n_classified": 1}


class StubPers(Stub):
    name = "personalization"
    def assign_cluster(self, **kw):
        return {"cluster_id": 1}


orch = Orchestrator()
for a in [StubDiag(), StubConf(), kg, rec, StubPers()]:
    orch.register_agent(a)

pipeline = orch.cold_start_pipeline("orch_rec_user")
assert "recommendations" in pipeline
rec_result = pipeline["recommendations"]
assert "items" in rec_result
assert "strategy_selected" in rec_result
print(f"  Pipeline: {len(rec_result['items'])} items, strategy={rec_result['strategy_selected']}")
print("  PASS")

# ═══════════════════════════════════════════════════
elapsed = time.time() - t0
print()
print("=" * 60)
print(f"ALL 10 TESTS PASSED in {elapsed:.1f}s")
print("=" * 60)
