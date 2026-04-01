"""End-to-end test for KnowledgeGraphAgent."""

import logging
import time

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

import numpy as np
import pandas as pd
from pathlib import Path
from data.loader import EdNetLoader
from agents.kg_agent import KnowledgeGraphAgent, TagGap, Recommendation, PART_NAMES
from agents.base_agent import BaseAgent, Message
from agents.orchestrator import Orchestrator

t0 = time.time()
loader = EdNetLoader(data_dir="data/raw")
questions = loader.load_questions()
lectures = loader.load_lectures()
interactions = loader.load_interactions(sample_users=300)
print(f"Data loaded in {time.time()-t0:.1f}s")
print(f"  Questions: {len(questions)}, Lectures: {len(lectures)}, Interactions: {len(interactions)}")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 1: build_graph")
print("=" * 60)
kg = KnowledgeGraphAgent()
G = kg.build_graph(questions, lectures)
stats = kg.get_graph_stats()

assert stats["node_types"]["question"] == 13169
assert stats["node_types"]["lecture"] == 1021
assert stats["node_types"]["part"] == 7
assert stats["node_types"]["tag"] > 180
assert stats["edge_types"]["HAS_TAG"] > 20000
assert stats["edge_types"]["COVERS_TAG"] > 900

# Verify node attributes
q_node = G.nodes["q1"]
assert all(k in q_node for k in ["question_id", "bundle_id", "correct_answer", "part_id", "difficulty"])

tag_nodes_list = [n for n, d in G.nodes(data=True) if d.get("node_type") == "tag"]
td = G.nodes[tag_nodes_list[0]]
assert all(k in td for k in ["tag_id", "avg_difficulty", "question_count", "part_ids"])

# No unknown node types
for n, d in G.nodes(data=True):
    assert d.get("node_type") in ("question", "lecture", "tag", "part"), f"Unknown: {n}"

print(f"  Nodes: {stats['total_nodes']}, Edges: {stats['total_edges']}")
print(f"  Node types: {stats['node_types']}")
print(f"  Edge types: {stats['edge_types']}")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 2: update_difficulties")
print("=" * 60)
kg.update_difficulties(interactions)

# Check via a known question from interactions
sample_qid = str(interactions["question_id"].iloc[0])
for candidate in (sample_qid, f"q{sample_qid}"):
    if G.has_node(candidate):
        d = G.nodes[candidate]["difficulty"]
        print(f"  {candidate} difficulty = {d}")
        break

# Count questions that got updated (difficulty differs from default 0.5)
n_updated = sum(
    1 for _, d in G.nodes(data=True)
    if d.get("node_type") == "question" and abs(d.get("difficulty", 0.5) - 0.5) > 1e-6
)
print(f"  Questions with non-default difficulty: {n_updated}")
assert n_updated > 100, f"Expected >100, got {n_updated}"
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 3: build_prerequisites")
print("=" * 60)
kg.build_prerequisites(interactions)
stats2 = kg.get_graph_stats()
n_prereq = stats2["edge_types"].get("PREREQUISITE_OF", 0)
print(f"  PREREQUISITE_OF edges: {n_prereq}")
print(f"  Is DAG: {stats2['prerequisite_is_dag']}")
assert stats2["prerequisite_is_dag"] is True, "Prerequisite graph must be DAG!"
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 4: train_graphsage")
print("=" * 60)
embeddings = kg.train_graphsage(hidden=128, out_dim=64, epochs=100, lr=0.01)
print(f"  Embeddings shape: {embeddings.shape}")
assert embeddings.shape[1] == 64
assert embeddings.shape[0] == stats["node_types"]["tag"]
assert not np.isnan(embeddings).any(), "NaN in embeddings!"

assert Path("models/tag_embeddings.npy").exists()
assert Path("models/graphsage.pt").exists()
saved = np.load("models/tag_embeddings.npy")
assert np.array_equal(saved, embeddings)
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 5: handle_cold_start")
print("=" * 60)
sample_qs = questions.sample(15, random_state=42)
diagnostic = {
    "responses": [
        {"question_id": row["question_id"], "correct": i % 3 != 0}
        for i, (_, row) in enumerate(sample_qs.iterrows())
    ]
}
cs = kg.handle_cold_start(user_id="cold_user", diagnostic=diagnostic)

assert all(k in cs for k in ["mastered_tags", "gap_tags", "prerequisite_gaps", "recommendations", "n_recommendations"])
print(f"  Mastered: {cs['mastered_tags']}")
print(f"  Gaps: {cs['gap_tags']}")
print(f"  Prerequisite gaps: {len(cs['prerequisite_gaps'])} tags with unmet prereqs")
print(f"  Recommendations: {cs['n_recommendations']}")

for r in cs["recommendations"][:3]:
    assert all(k in r for k in ["item_id", "item_type", "reason", "priority"])
    assert r["item_type"] in ("lecture", "question")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 6: update_user_profile")
print("=" * 60)
first_user = interactions["user_id"].iloc[0]
prof = kg.update_user_profile(
    user_id="cold_user",
    interactions=interactions[interactions["user_id"] == first_user].head(30),
)
assert prof["updated"] is True
assert prof["n_tags_tracked"] > 0
print(f"  Tags tracked: {prof['n_tags_tracked']}, Weak: {len(prof['weak_tags'])}")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 7: get_user_gaps")
print("=" * 60)
gaps = kg.get_user_gaps("cold_user")
print(f"  Found {len(gaps)} gaps")
for g in gaps[:3]:
    assert isinstance(g, TagGap)
    assert 0 <= g.current_accuracy <= 1
    print(f"    Tag {g.tag_id}: acc={g.current_accuracy}, prereqs={g.prerequisite_tags[:3]}, lectures={len(g.recommended_lectures)}")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 8: get_prerequisite_chains")
print("=" * 60)
chains = kg.get_prerequisite_chains(max_chains=10)
print(f"  Found {len(chains)} chains")
for c in chains[:5]:
    print(f"    {' -> '.join(str(t) for t in c)}  (len={len(c)})")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 9: receive_message (orchestrator interface)")
print("=" * 60)
msg = Message(
    sender="orchestrator",
    target="knowledge_graph",
    data={"action": "get_gaps", "user_id": "cold_user"},
)
result = kg.receive_message(msg)
assert isinstance(result, list)
print(f"  Message returned {len(result)} TagGaps")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 10: Orchestrator integration (cold_start_pipeline)")
print("=" * 60)


class Stub(BaseAgent):
    def initialize(self, **kw):
        pass


class StubDiag(Stub):
    name = "diagnostic"

    def run_diagnostic(self, uid):
        qs = questions.sample(8, random_state=7)
        return {
            "responses": [
                {"question_id": r["question_id"], "correct": i % 2 == 0}
                for i, (_, r) in enumerate(qs.iterrows())
            ]
        }


class StubConf(Stub):
    name = "confidence"

    def classify_batch(self, **kw):
        return {"classes": [2], "mean_confidence": 3}


class StubRec(Stub):
    name = "recommendation"

    def recommend(self, **kw):
        return {"items": ["q100"]}


class StubPers(Stub):
    name = "personalization"

    def assign_cluster(self, **kw):
        return {"cluster_id": 1}


orch = Orchestrator()
for a in [StubDiag(), StubConf(), kg, StubRec(), StubPers()]:
    orch.register_agent(a)

result = orch.cold_start_pipeline("integration_user")
assert result["pipeline"] == "cold_start"
assert "kg_profile" in result
kg_prof = result["kg_profile"]
assert all(k in kg_prof for k in ["mastered_tags", "gap_tags", "recommendations"])
print(
    f"  Cold-start: mastered={len(kg_prof['mastered_tags'])}, "
    f"gaps={len(kg_prof['gap_tags'])}, recs={kg_prof['n_recommendations']}"
)
print("  PASS")

# ═══════════════════════════════════════════════════
elapsed = time.time() - t0
print()
print("=" * 60)
print(f"ALL 10 TESTS PASSED in {elapsed:.1f}s")
print("=" * 60)
