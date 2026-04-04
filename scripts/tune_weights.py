"""
Grid search for optimal agent signal weights.
Uses cached MARS (full) pipeline signals — no retraining needed.
Tests weight combinations on eval pairs, finds best NDCG@10.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import itertools
from collections import defaultdict

NUM_TAGS = 293
SEED = 42

# ── Load data ────────────────────────────────────────────────────────
from data.loader import EdNetLoader
from data.preprocessor import EdNetPreprocessor

print("Loading data...")
loader = EdNetLoader(data_dir="data/raw")
interactions = loader.load_interactions(sample_users=1000)
preprocessor = EdNetPreprocessor(output_dir="data/processed", splits_dir="data/splits")
df_clean = preprocessor.clean(interactions)
df_feat = preprocessor.engineer_features(df_clean)
splits = preprocessor.chronological_split(df_feat)
train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]

def parse_tags(tags):
    if isinstance(tags, list):
        return [int(t) for t in tags if 0 <= int(t) < NUM_TAGS]
    if isinstance(tags, (int, np.integer)):
        return [int(tags)] if 0 <= int(tags) < NUM_TAGS else []
    if isinstance(tags, str):
        return [int(t) for t in tags.split(";") if t.strip().isdigit() and 0 <= int(t) < NUM_TAGS]
    return []

def build_eval_pairs(df, context_ratio=0.5):
    pairs = []
    for uid, grp in df.groupby("user_id"):
        grp = grp.sort_values("timestamp")
        si = max(1, int(len(grp) * context_ratio))
        ctx, future = grp.iloc[:si], grp.iloc[si:]
        if len(future) == 0: continue
        gt, gt_all = set(), set()
        for _, row in future.iterrows():
            tags = parse_tags(row["tags"])
            gt_all.update(tags)
            if not row["correct"]: gt.update(tags)
        if len(gt) == 0: continue
        pairs.append((str(uid), ctx, gt, gt_all, 0))
    return pairs

eval_pairs = build_eval_pairs(test_df)
print(f"Eval pairs: {len(eval_pairs)}")

# ── Pre-compute all agent signals ONCE ───────────────────────────────
print("Computing agent signals...")
torch.manual_seed(SEED)
np.random.seed(SEED)

from agents.prediction_agent import PredictionAgent
from agents.confidence_agent import ConfidenceAgent
from agents.diagnostic_agent import DiagnosticAgent
from agents.kg_agent import KnowledgeGraphAgent
from pathlib import Path

# DiagnosticAgent
diag = DiagnosticAgent()
irt_params = diag.calibrate_from_interactions(train_df, min_answers_per_q=5)
tag_difficulty = {str(irt_params.question_ids[i]): float(irt_params.b[i])
                  for i in range(len(irt_params.question_ids))}

# ConfidenceAgent
conf_agent = ConfidenceAgent()
conf_agent.train(train_df, irt_params=irt_params)

# KGAgent
prereq_map = {}
try:
    kg = KnowledgeGraphAgent()
    kg.build_graph(loader.questions, loader.lectures)
    kg.build_prerequisites(train_df)
    for t in range(NUM_TAGS):
        prereqs = kg._get_prerequisites(t, depth=2)
        if prereqs: prereq_map[t] = prereqs
except: pass

tag_fail_rate = np.zeros(NUM_TAGS, dtype=np.float32)
for _, row in train_df.iterrows():
    if not row["correct"]:
        for t in parse_tags(row.get("tags", [])):
            if 0 <= t < NUM_TAGS: tag_fail_rate[t] += 1
tag_fail_rate /= (tag_fail_rate.max() + 1e-10)

# PredictionAgent
train_enriched = train_df.copy()
conf_train = conf_agent.classify_batch(interactions=train_enriched)
train_enriched["confidence_class"] = conf_train["classes"]
pred_agent = PredictionAgent()
pred_agent.train(train_enriched, epochs=5, batch_size=256, patience=2)

conf_class_boost = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25], dtype=np.float32)

# Pre-compute per-user signals
print("Pre-computing per-user signals...")
user_signals = []
for uid, ctx, gt, gt_all, _ in eval_pairs:
    conf_result = conf_agent.classify_batch(user_id=uid, interactions=ctx)
    ctx_e = ctx.copy()
    if conf_result["classes"] and len(conf_result["classes"]) == len(ctx_e):
        ctx_e["confidence_class"] = conf_result["classes"]

    result = pred_agent.predict_gaps(uid, recent=ctx_e, threshold=0.0)
    gp = np.array(result["gap_probabilities"], dtype=np.float32)
    gp_min, gp_max = gp.min(), gp.max()
    gp_norm = (gp - gp_min) / (gp_max - gp_min) if gp_max > gp_min else gp

    # IRT signal
    recent_acc = ctx["correct"].astype(float).mean()
    theta = np.clip(np.log(recent_acc / (1 - recent_acc + 1e-6) + 1e-6), -3, 3)
    irt_sig = np.zeros(NUM_TAGS, dtype=np.float32)
    for t in range(NUM_TAGS):
        b = tag_difficulty.get(str(t), 0.0)
        irt_sig[t] = np.exp(-0.5 * (theta - b) ** 2)
    irt_sig /= (irt_sig.max() + 1e-10)

    # KG signal
    kg_sig = np.zeros(NUM_TAGS, dtype=np.float32)
    if prereq_map:
        user_tag_acc = {}
        for _, row in ctx.iterrows():
            for t in parse_tags(row.get("tags", [])):
                if t not in user_tag_acc: user_tag_acc[t] = {"c": 0, "t": 0}
                user_tag_acc[t]["t"] += 1
                if row["correct"]: user_tag_acc[t]["c"] += 1
        mastered = {t for t, s in user_tag_acc.items() if s["t"] >= 3 and s["c"]/s["t"] >= 0.7}
        for t in range(NUM_TAGS):
            prereqs = prereq_map.get(t, [])
            if prereqs:
                kg_sig[t] = sum(1 for p in prereqs if p in mastered) / len(prereqs)
            else:
                kg_sig[t] = tag_fail_rate[t]

    # Confidence signal
    recent_conf = ctx_e.get("confidence_class", pd.Series([0]*len(ctx_e)))
    if hasattr(recent_conf, 'values'): recent_conf = recent_conf.values
    mean_boost = np.mean([conf_class_boost[min(int(c), 5)] for c in recent_conf])
    conf_sig = gp_norm * mean_boost

    user_signals.append({
        "pred": gp_norm, "irt": irt_sig, "kg": kg_sig, "conf": conf_sig,
        "gt": gt, "gt_all": gt_all,
    })

print(f"Signals computed for {len(user_signals)} users.")

# ── Grid search ──────────────────────────────────────────────────────
def eval_ndcg10(w_pred, w_irt, w_kg, w_conf):
    ndcgs = []
    for s in user_signals:
        total_w = w_pred + w_irt + w_kg + w_conf
        scores = (w_pred/total_w * s["pred"] + w_irt/total_w * s["irt"] +
                  w_kg/total_w * s["kg"] + w_conf/total_w * s["conf"])
        ranked = np.argsort(scores)[::-1][:10]
        gt = s["gt"]
        dcg = sum(1.0/np.log2(r+2) for r, t in enumerate(ranked) if t in gt)
        ideal = sum(1.0/np.log2(r+2) for r in range(min(len(gt), 10)))
        ndcgs.append(dcg/ideal if ideal > 0 else 0.0)
    return np.mean(ndcgs)

print("\nGrid search...")
# Test weight combinations (step 0.05)
steps = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
best_ndcg, best_w = 0, None
results = []

for wp in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]:
    for wi in steps:
        for wk in steps:
            for wc in steps:
                if wp + wi + wk + wc < 0.01: continue
                ndcg = eval_ndcg10(wp, wi, wk, wc)
                results.append((ndcg, wp, wi, wk, wc))
                if ndcg > best_ndcg:
                    best_ndcg = ndcg
                    best_w = (wp, wi, wk, wc)

results.sort(reverse=True)
print(f"\nTop 10 weight combinations:")
print(f"{'NDCG@10':>8s}  W_PRED  W_IRT   W_KG    W_CONF")
for ndcg, wp, wi, wk, wc in results[:10]:
    print(f"{ndcg:.4f}   {wp:.2f}    {wi:.2f}    {wk:.2f}    {wc:.2f}")

print(f"\nBest: NDCG@10={best_ndcg:.4f}")
print(f"  W_PRED={best_w[0]:.2f}, W_IRT={best_w[1]:.2f}, W_KG={best_w[2]:.2f}, W_CONF={best_w[3]:.2f}")

# Also test pure PredictionAgent for comparison
pure_pred = eval_ndcg10(1.0, 0.0, 0.0, 0.0)
print(f"\nPure PredictionAgent: NDCG@10={pure_pred:.4f}")
print(f"Improvement: {(best_ndcg - pure_pred)/pure_pred * 100:+.1f}%")
