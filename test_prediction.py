"""End-to-end test for PredictionAgent."""

import logging
import time

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

import numpy as np
import pandas as pd
import torch
from data.loader import EdNetLoader
from agents.prediction_agent import (
    PredictionAgent,
    GapPredictionLSTM,
    GapSequenceDataset,
    NUM_TAGS,
    SEQ_LEN,
    HORIZON,
    INPUT_DIM,
    DEVICE,
)
from agents.base_agent import BaseAgent

t0 = time.time()
loader = EdNetLoader(data_dir="data/raw")
interactions = loader.load_interactions(sample_users=300)
print(f"Data loaded in {time.time()-t0:.1f}s")
print(f"  Interactions: {len(interactions):,}, Users: {interactions['user_id'].nunique()}")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 1: Constants and dimensions")
print("=" * 60)
assert NUM_TAGS == 293
assert SEQ_LEN == 50
assert HORIZON == 10
assert INPUT_DIM == 51  # 32+1+1+1+8+8
print(f"  NUM_TAGS={NUM_TAGS}, SEQ_LEN={SEQ_LEN}, HORIZON={HORIZON}")
print(f"  INPUT_DIM={INPUT_DIM} (tag_emb=32 + 3 scalar + part_emb=8 + conf_emb=8)")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 2: GapPredictionLSTM forward pass")
print("=" * 60)
model = GapPredictionLSTM().to(DEVICE)
batch = torch.randn(4, SEQ_LEN, 6).to(DEVICE)
# Fix integer columns to valid ranges
batch[:, :, 0] = torch.randint(0, NUM_TAGS, (4, SEQ_LEN)).float()
batch[:, :, 4] = torch.randint(0, 7, (4, SEQ_LEN)).float()
batch[:, :, 5] = torch.randint(0, 6, (4, SEQ_LEN)).float()

with torch.no_grad():
    out = model(batch)

assert out.shape == (4, NUM_TAGS), f"Expected (4, {NUM_TAGS}), got {out.shape}"
assert (out >= 0).all() and (out <= 1).all(), "Output must be in [0, 1]"
print(f"  Input shape: {batch.shape}")
print(f"  Output shape: {out.shape}")
print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 3: GapSequenceDataset construction")
print("=" * 60)
# Use a subset of interactions
subset = interactions[interactions["user_id"].isin(interactions["user_id"].unique()[:50])]
dataset = GapSequenceDataset(subset)
print(f"  Users: {subset['user_id'].nunique()}")
print(f"  Sequences: {len(dataset)}")

if len(dataset) > 0:
    x, y = dataset[0]
    assert x.shape == (SEQ_LEN, 6), f"Expected ({SEQ_LEN}, 6), got {x.shape}"
    assert y.shape == (NUM_TAGS,), f"Expected ({NUM_TAGS},), got {y.shape}"
    assert y.dtype == torch.float32
    assert (y >= 0).all() and (y <= 1).all()
    print(f"  Sample x shape: {x.shape}")
    print(f"  Sample y shape: {y.shape}, active tags: {(y > 0).sum().item()}")
    print("  PASS")
else:
    print("  WARNING: No sequences built (users may not have enough interactions)")
    print("  SKIP")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 4: PredictionAgent inherits BaseAgent")
print("=" * 60)
agent = PredictionAgent()
assert isinstance(agent, BaseAgent)
assert agent.name == "prediction"
assert agent.status == "idle"
print(f"  Agent name: {agent.name}")
print(f"  Agent status: {agent.status}")
print(f"  repr: {repr(agent)}")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 5: PredictionAgent.train()")
print("=" * 60)
t1 = time.time()
metrics = agent.train(interactions, epochs=3, batch_size=128, patience=2)
train_time = time.time() - t1

print(f"  Training time: {train_time:.1f}s")
assert "train_loss" in metrics
assert "val_loss" in metrics
assert "val_auc" in metrics
assert "best_epoch" in metrics
assert "history" in metrics
assert agent.model is not None
print(f"  train_loss: {metrics['train_loss']}")
print(f"  val_loss: {metrics['val_loss']}")
print(f"  val_auc: {metrics['val_auc']}")
print(f"  best_epoch: {metrics['best_epoch']}")
print(f"  total_epochs: {metrics['total_epochs']}")
print(f"  n_train_sequences: {metrics['n_train_sequences']}")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 6: PredictionAgent.predict_gaps()")
print("=" * 60)
test_user = interactions["user_id"].unique()[0]
user_df = interactions[interactions["user_id"] == test_user].sort_values("timestamp")
recent = user_df.head(SEQ_LEN)

result = agent.predict_gaps(str(test_user), recent=recent, threshold=0.3)

assert "user_id" in result
assert "gaps" in result
assert "gap_probabilities" in result
assert "n_gaps" in result
assert len(result["gap_probabilities"]) == NUM_TAGS
assert result["user_id"] == str(test_user)
print(f"  User: {test_user}, recent interactions: {len(recent)}")
print(f"  Gaps found (thr=0.3): {result['n_gaps']}")
if result["gaps"]:
    print(f"  Top gap: tag_{result['gaps'][0]['tag_id']} "
          f"(p={result['gaps'][0]['probability']:.3f})")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 7: PredictionAgent.get_at_risk_tags()")
print("=" * 60)
at_risk = agent.get_at_risk_tags(str(test_user), recent=recent, threshold=0.5)
assert isinstance(at_risk, list)
for g in at_risk:
    assert "tag_id" in g
    assert "probability" in g
    assert g["probability"] >= 0.5
print(f"  At-risk tags (thr=0.5): {len(at_risk)}")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 8: PredictionAgent.update_state() (continuous pipeline)")
print("=" * 60)
# Fresh agent to test buffer
agent2 = PredictionAgent()
agent2.model = agent.model  # share trained model

# Feed interactions one by one
for i, (_, row) in enumerate(user_df.head(SEQ_LEN + 5).iterrows()):
    interaction = {
        "question_id": row.get("question_id"),
        "tags": row.get("tags", []),
        "correct": bool(row.get("correct", False)),
        "elapsed_time": float(row.get("elapsed_time", 15000)),
        "changed_answer": bool(row.get("changed_answer", False)),
        "part_id": int(row.get("part_id", 1)),
        "confidence_class": 0,
    }
    state = agent2.update_state(str(test_user), interaction)

assert state["buffer_size"] == SEQ_LEN  # truncated to SEQ_LEN
assert "n_at_risk" in state
assert "top_risks" in state
print(f"  Fed {SEQ_LEN + 5} interactions")
print(f"  Buffer size: {state['buffer_size']}")
print(f"  At-risk tags: {state['n_at_risk']}")
print(f"  Top risks: {state['top_risks'][:3]}")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 9: predict_gaps with no data (graceful fallback)")
print("=" * 60)
empty_result = agent.predict_gaps("unknown_user_999", recent=None, threshold=0.5)
assert empty_result["n_gaps"] == 0
assert empty_result["gaps"] == []
print(f"  Empty result: {empty_result}")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 10: Model save/load roundtrip")
print("=" * 60)
from pathlib import Path
model_path = Path("models/gap_lstm.pt")
assert model_path.exists(), f"Model file not found: {model_path}"

agent3 = PredictionAgent()
agent3._load_model(model_path)
assert agent3.model is not None

# Compare predictions
result_orig = agent.predict_gaps(str(test_user), recent=recent, threshold=0.3)
result_loaded = agent3.predict_gaps(str(test_user), recent=recent, threshold=0.3)
probs_orig = np.array(result_orig["gap_probabilities"])
probs_loaded = np.array(result_loaded["gap_probabilities"])
assert np.allclose(probs_orig, probs_loaded, atol=1e-5), "Loaded model differs from original"
print(f"  Saved to: {model_path}")
print(f"  Loaded and verified: max diff = {np.abs(probs_orig - probs_loaded).max():.2e}")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 11: Message handling (orchestrator integration)")
print("=" * 60)
from agents.base_agent import Message

msg = Message(
    sender="orchestrator",
    target="prediction",
    data={
        "action": "predict_gaps",
        "user_id": str(test_user),
        "recent": recent,
    },
)
msg_result = agent.receive_message(msg)
assert msg_result is not None
assert "gaps" in msg_result
print(f"  Message handling works: {msg_result['n_gaps']} gaps found")
print("  PASS")

# ═══════════════════════════════════════════════════
print()
print("=" * 60)
total_time = time.time() - t0
print(f"ALL 11 TESTS PASSED in {total_time:.1f}s")
print("=" * 60)
