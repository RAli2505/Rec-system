"""End-to-end test for DiagnosticAgent."""

import logging
import time

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from data.loader import EdNetLoader
from agents.diagnostic_agent import DiagnosticAgent, IRTParams, DiagnosticResult
from agents.base_agent import BaseAgent
from agents.orchestrator import Orchestrator

t0 = time.time()
loader = EdNetLoader(data_dir="data/raw")
questions = loader.load_questions()
interactions = loader.load_interactions(sample_users=500)
print(f"Data loaded in {time.time()-t0:.1f}s")
print(f"  Interactions: {len(interactions):,}, Users: {interactions['user_id'].nunique()}")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 1: calibrate_from_interactions")
print("=" * 60)
diag = DiagnosticAgent()
params = diag.calibrate_from_interactions(interactions, min_answers_per_q=5, max_items=3000)

assert isinstance(params, IRTParams)
assert len(params) > 50, f"Expected >50 items, got {len(params)}"
assert params.b.shape == params.a.shape == params.c.shape
assert -3.0 <= params.b.min() and params.b.max() <= 3.0, "b out of range"
assert 0.2 <= params.a.min() and params.a.max() <= 3.0, "a out of range"
assert np.all(params.c == 0.25)
print(f"  Calibrated {len(params)} items")
print(f"  b: [{params.b.min():.2f}, {params.b.max():.2f}], mean={params.b.mean():.3f}")
print(f"  a: [{params.a.min():.2f}, {params.a.max():.2f}], mean={params.a.mean():.3f}")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 2: IRT probability (3PL)")
print("=" * 60)
# Easy item at high theta -> high probability
p_easy = diag._p_correct(3.0, a=1.0, b=-2.0, c=0.25)
p_hard = diag._p_correct(-2.0, a=1.0, b=2.0, c=0.25)
print(f"  P(correct | theta=3, b=-2) = {p_easy:.4f}")
print(f"  P(correct | theta=-2, b=2) = {p_hard:.4f}")
assert p_easy > 0.9, f"Easy item should be >0.9, got {p_easy}"
assert p_hard < 0.35, f"Hard item should be <0.35, got {p_hard}"
# Guessing floor
p_floor = diag._p_correct(-100, a=1.0, b=0.0, c=0.25)
assert abs(p_floor - 0.25) < 0.01, f"Guessing floor should be ~0.25, got {p_floor}"
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 3: Fisher Information")
print("=" * 60)
# For 3PL, information peaks slightly above b due to guessing floor
# Verify: info far below b (theta << b) should be near zero
info_near_b = diag._fisher_info(0.5, a=2.0, b=0.0, c=0.25)
info_far_below = diag._fisher_info(-4.0, a=2.0, b=0.0, c=0.25)
print(f"  Info near b (theta=0.5): {info_near_b:.4f}")
print(f"  Info far below (theta=-4): {info_far_below:.4f}")
assert info_near_b > info_far_below, "Info near b should exceed info far below"
# Also verify information is always non-negative
assert info_near_b >= 0 and info_far_below >= 0
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 4: update_theta (MLE)")
print("=" * 60)
# All-correct on easy items -> high theta
easy_items = np.argsort(params.b)[:5]
responses_correct = [(int(i), True) for i in easy_items]
theta_high, se_high = diag.update_theta(responses_correct)
print(f"  All correct (easy): theta={theta_high:.3f}, SE={se_high:.3f}")
assert theta_high > 0, "All correct on easy should give positive theta"

# All-wrong on hard items -> low theta
hard_items = np.argsort(params.b)[-5:]
responses_wrong = [(int(i), False) for i in hard_items]
theta_low, se_low = diag.update_theta(responses_wrong)
print(f"  All wrong (hard):   theta={theta_low:.3f}, SE={se_low:.3f}")
assert theta_low < theta_high, "Wrong responses should give lower theta"
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 5: select_next_question")
print("=" * 60)
idx = diag.select_next_question(theta=0.0, used_indices=set())
assert idx is not None
assert 0 <= idx < len(params)
print(f"  Selected item {idx}: b={params.b[idx]:.2f}, a={params.a[idx]:.2f}")

# With part filter
idx_part5 = diag.select_next_question(0.0, set(), target_parts=[5])
assert idx_part5 is not None
assert int(params.part_ids[idx_part5]) == 5
print(f"  Part 5 item {idx_part5}: b={params.b[idx_part5]:.2f}")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 6: run_diagnostic (15 items, all parts)")
print("=" * 60)
# Build simulated responses from a real user
uid = interactions["user_id"].iloc[0]
user_ints = interactions[interactions["user_id"] == uid]
sim_resp = {str(r["question_id"]): bool(r["correct"]) for _, r in user_ints.iterrows()}

result = diag.run_diagnostic(f"diag_{uid}", simulated_responses=sim_resp)

assert "ability" in result
assert "se" in result
assert "responses" in result
assert "parts_covered" in result
assert result["n_questions"] == 15
assert len(result["parts_covered"]) >= 5  # should cover most parts
print(f"  Theta: {result['ability']}, SE: {result['se']}")
print(f"  Parts covered: {result['parts_covered']}")
print(f"  Items: {result['n_questions']}")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 7: run_assessment (batch update)")
print("=" * 60)
assess = diag.run_assessment("assess_user", interactions=user_ints)
assert "ability" in assess
assert "se" in assess
assert "n_responses" in assess
assert assess["n_responses"] > 0
print(f"  Theta: {assess['ability']}, SE: {assess['se']}, N: {assess['n_responses']}")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 8: update_ability (incremental)")
print("=" * 60)
single_int = {"question_id": str(user_ints.iloc[0]["question_id"]), "correct": True}
update = diag.update_ability("assess_user", interaction=single_int)
assert "ability" in update
assert "se" in update
print(f"  Updated: theta={update['ability']}, SE={update['se']}")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 9: run_cat_session (adaptive)")
print("=" * 60)
cat = diag.run_cat_session(f"cat_{uid}", simulated_responses=sim_resp, max_items=20)
assert "ability" in cat
assert "n_items" in cat
assert "converged" in cat
assert "responses" in cat
assert cat["n_items"] >= 5
print(f"  CAT: theta={cat['ability']}, SE={cat['se']}, items={cat['n_items']}, converged={cat['converged']}")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 10: Theta vs Accuracy correlation")
print("=" * 60)
user_ids = interactions["user_id"].unique()[:100]
thetas = []
accs = []
for u in user_ids:
    u_df = interactions[interactions["user_id"] == u]
    r = diag.run_assessment(str(u), u_df)
    thetas.append(r["ability"])
    accs.append(u_df["correct"].mean())

r_val, p_val = pearsonr(thetas, accs)
print(f"  Pearson r = {r_val:.3f} (p = {p_val:.2e})")
assert r_val > 0.3, f"Expected positive correlation, got r={r_val:.3f}"
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 11: ICC and TIF curves")
print("=" * 60)
theta_range = np.linspace(-4, 4, 100)
icc = diag.icc_curve(0, theta_range)
assert icc.shape == (100,)
assert 0.25 <= icc.min()  # guessing floor
assert icc.max() <= 1.0

tif = diag.test_information_function(theta_range)
assert tif.shape == (100,)
assert tif.max() > 0
print(f"  ICC range: [{icc.min():.3f}, {icc.max():.3f}]")
print(f"  TIF peak: {tif.max():.1f} at theta={theta_range[np.argmax(tif)]:.2f}")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 12: Orchestrator integration")
print("=" * 60)


class StubAgent(BaseAgent):
    def initialize(self, **kw):
        pass


class StubConf(StubAgent):
    name = "confidence"
    def classify_batch(self, **kw):
        return {"classes": [2], "mean_confidence": 3}


class StubKG(StubAgent):
    name = "knowledge_graph"
    def handle_cold_start(self, **kw):
        return {"mastered_tags": [], "gap_tags": [], "recommendations": [], "n_recommendations": 0}


class StubRec(StubAgent):
    name = "recommendation"
    def recommend(self, **kw):
        return {"items": ["q1"]}


class StubPers(StubAgent):
    name = "personalization"
    def assign_cluster(self, **kw):
        return {"cluster_id": 1}


orch = Orchestrator()
for a in [diag, StubConf(), StubKG(), StubRec(), StubPers()]:
    orch.register_agent(a)

pipeline_result = orch.cold_start_pipeline("orch_user")
assert pipeline_result["pipeline"] == "cold_start"
assert "diagnostic" in pipeline_result
diag_r = pipeline_result["diagnostic"]
assert "ability" in diag_r
assert "responses" in diag_r
assert len(diag_r["responses"]) == 15
print(f"  Pipeline diagnostic: theta={diag_r['ability']}, {diag_r['n_questions']} items")
print("  PASS")

# ═══════════════════════════════════════════════════
elapsed = time.time() - t0
print()
print("=" * 60)
print(f"ALL 12 TESTS PASSED in {elapsed:.1f}s")
print("=" * 60)
