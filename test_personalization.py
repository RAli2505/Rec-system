"""End-to-end test for PersonalizationAgent."""

import logging
import time

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

import numpy as np
import pandas as pd
from data.loader import EdNetLoader
from agents.personalization_agent import (
    PersonalizationAgent,
    extract_user_features,
    ClusterProfile,
    FEATURE_NAMES,
    FEATURE_NAMES_SCALAR,
    FEATURE_NAMES_ONEHOT,
    K_RANGE,
)
from agents.base_agent import BaseAgent, Message

t0 = time.time()
loader = EdNetLoader(data_dir="data/raw")
interactions = loader.load_interactions(sample_users=500)
print(f"Data loaded in {time.time()-t0:.1f}s")
print(f"  Interactions: {len(interactions):,}, Users: {interactions['user_id'].nunique()}")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 1: Feature constants")
print("=" * 60)
assert len(FEATURE_NAMES_SCALAR) == 7
assert len(FEATURE_NAMES_ONEHOT) == 7
assert len(FEATURE_NAMES) == 14  # 7 scalar + 7 one-hot
print(f"  Scalar features (7): {FEATURE_NAMES_SCALAR}")
print(f"  One-hot features (7): {FEATURE_NAMES_ONEHOT}")
print(f"  Total dimensions: {len(FEATURE_NAMES)}")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 2: extract_user_features()")
print("=" * 60)
features = extract_user_features(interactions)
n_users = interactions["user_id"].nunique()
assert len(features) > 0
assert features.shape[1] == 14
assert set(FEATURE_NAMES).issubset(features.columns)
# Check value ranges
assert features["accuracy_rate"].between(0, 1).all()
assert features["changed_answer_rate"].between(0, 1).all()
assert features["false_confidence_rate"].between(0, 1).all()
assert (features["avg_elapsed_time"] >= 0).all()
assert (features["session_frequency"] >= 0).all()
# One-hot dominant_part: exactly one 1 per row
onehot_sum = features[FEATURE_NAMES_ONEHOT].sum(axis=1)
assert (onehot_sum == 1.0).all(), f"One-hot sum not 1: {onehot_sum.value_counts()}"
print(f"  Users: {len(features)}")
print(f"  Shape: {features.shape}")
print(f"  accuracy_rate: mean={features['accuracy_rate'].mean():.3f}")
print(f"  learning_speed: mean={features['learning_speed'].mean():.4f}")
print(f"  One-hot sum check: all 1.0")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 3: PersonalizationAgent inherits BaseAgent")
print("=" * 60)
agent = PersonalizationAgent()
assert isinstance(agent, BaseAgent)
assert agent.name == "personalization"
assert agent.status == "idle"
print(f"  Agent name: {agent.name}")
print(f"  repr: {repr(agent)}")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 4: train_clusters() — silhouette selection")
print("=" * 60)
t1 = time.time()
optimal_k = agent.train_clusters(features)
train_time = time.time() - t1

assert isinstance(optimal_k, int)
assert 3 <= optimal_k <= 8
assert agent.model is not None
assert agent.scaler is not None
assert agent.optimal_k == optimal_k
assert len(agent.silhouette_scores) > 0
assert len(agent.inertias) > 0
assert all(0 < s < 1 for s in agent.silhouette_scores.values())
assert len(agent._cluster_names) == optimal_k
print(f"  Training time: {train_time:.1f}s")
print(f"  Optimal K: {optimal_k}")
print(f"  Silhouette scores: {agent.silhouette_scores}")
print(f"  Cluster names: {agent._cluster_names}")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 5: assign_cluster() — existing user")
print("=" * 60)
test_user = str(features.index[0])
result = agent.assign_cluster(test_user)

assert isinstance(result, dict)
assert "cluster_id" in result
assert "cluster_name" in result
assert "user_type" in result
assert "feature_vector" in result
assert "centroid" in result
assert "distance_to_centroid" in result
assert 0 <= result["cluster_id"] < optimal_k
assert isinstance(result["cluster_name"], str) and len(result["cluster_name"]) > 0
assert result["distance_to_centroid"] >= 0
print(f"  User: {test_user}")
print(f"  Cluster: {result['cluster_id']} ({result['cluster_name']})")
print(f"  Distance: {result['distance_to_centroid']}")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 6: assign_cluster() — cold start (diagnostic + confidence)")
print("=" * 60)
cold_result = agent.assign_cluster(
    user_id="cold_start_user_999",
    diagnostic={
        "responses": [
            {"correct": True, "part_id": 1},
            {"correct": True, "part_id": 2},
            {"correct": False, "part_id": 3},
            {"correct": True, "part_id": 4},
            {"correct": False, "part_id": 5},
        ],
    },
    confidence={
        "class_names": ["SOLID", "UNSURE_CORRECT", "FALSE_CONFIDENCE",
                        "SOLID", "CLEAR_GAP"],
    },
)

assert isinstance(cold_result, dict)
assert 0 <= cold_result["cluster_id"] < optimal_k
assert len(cold_result["feature_vector"]) == 14
print(f"  Cold user cluster: {cold_result['cluster_id']} ({cold_result['cluster_name']})")
print(f"  Feature vector length: {len(cold_result['feature_vector'])}")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 7: get_user_type()")
print("=" * 60)
user_type = agent.get_user_type(test_user)
assert isinstance(user_type, str)
assert len(user_type) > 0
assert user_type != "Unknown"

unknown_type = agent.get_user_type("nonexistent_user_abc")
assert unknown_type == "Unknown"
print(f"  Known user type: '{user_type}'")
print(f"  Unknown user type: '{unknown_type}'")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 8: personalize() — assessment pipeline integration")
print("=" * 60)
pers_result = agent.personalize(
    user_id=test_user,
    diagnostic={"ability": 0.5, "se": 0.3},
    confidence={"class_names": ["SOLID", "SOLID", "UNSURE_CORRECT"]},
)

assert "cluster_name" in pers_result
assert "adjustments" in pers_result
adj = pers_result["adjustments"]
assert "difficulty_adjustment" in adj
assert "pacing" in adj
assert "content_mix" in adj
assert "review_frequency" in adj
print(f"  Cluster: {pers_result['cluster_name']}")
print(f"  Adjustments: {adj}")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 9: cluster_summary()")
print("=" * 60)
summary = agent.cluster_summary()
assert isinstance(summary, pd.DataFrame)
assert len(summary) == optimal_k
assert "cluster_name" in summary.columns
assert "size" in summary.columns
assert "mean_accuracy_rate" in summary.columns
total = summary["size"].sum()
print(f"  Clusters: {len(summary)}")
print(f"  Total users: {total}")
for _, row in summary.iterrows():
    print(f"    {row['cluster_name']:20s}: n={row['size']:4d} "
          f"({row['pct']:5.1f}%) acc={row['mean_accuracy_rate']:.3f}")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 10: get_centroids_df()")
print("=" * 60)
centroids = agent.get_centroids_df()
assert isinstance(centroids, pd.DataFrame)
assert len(centroids) == optimal_k
assert "cluster_name" in centroids.columns
for feat in FEATURE_NAMES:
    assert feat in centroids.columns, f"Missing feature: {feat}"
print(f"  Centroids shape: {centroids.shape}")
print(centroids[["cluster_name", "accuracy_rate", "avg_elapsed_time", "learning_speed"]].to_string())
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 11: Model save/load roundtrip")
print("=" * 60)
from pathlib import Path
model_path = Path("models/personalization_kmeans.pkl")
assert model_path.exists(), f"Model file not found: {model_path}"

agent2 = PersonalizationAgent()
loaded = agent2.load_model()
assert loaded
assert agent2.optimal_k == optimal_k
assert len(agent2._cluster_names) == optimal_k
assert agent2.model is not None

# Predict same user — should get same cluster
result2 = agent2.assign_cluster(
    user_id="cold_start_user_999",
    diagnostic={
        "responses": [
            {"correct": True, "part_id": 1},
            {"correct": True, "part_id": 2},
            {"correct": False, "part_id": 3},
            {"correct": True, "part_id": 4},
            {"correct": False, "part_id": 5},
        ],
    },
    confidence={
        "class_names": ["SOLID", "UNSURE_CORRECT", "FALSE_CONFIDENCE",
                        "SOLID", "CLEAR_GAP"],
    },
)
assert result2["cluster_id"] == cold_result["cluster_id"]
print(f"  Loaded model K={agent2.optimal_k}")
print(f"  Cluster match: {result2['cluster_id']} == {cold_result['cluster_id']}")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 12: Message handling (orchestrator integration)")
print("=" * 60)
msg = Message(
    sender="orchestrator",
    target="personalization",
    data={
        "action": "assign_cluster",
        "user_id": test_user,
    },
)
msg_result = agent.receive_message(msg)
assert msg_result is not None
assert "cluster_name" in msg_result
print(f"  Message result: cluster={msg_result['cluster_name']}")

msg2 = Message(
    sender="orchestrator",
    target="personalization",
    data={
        "action": "get_user_type",
        "user_id": test_user,
    },
)
msg2_result = agent.receive_message(msg2)
assert isinstance(msg2_result, str)
print(f"  get_user_type message: '{msg2_result}'")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
total_time = time.time() - t0
print(f"ALL 12 TESTS PASSED in {total_time:.1f}s")
print("=" * 60)
