"""
Subgroup analysis: cold-start vs warm users.
Uses existing main cache — no retraining needed.
Shows where multi-agent pipeline adds value.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import json
from pathlib import Path
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

# ── Classify users into subgroups ────────────────────────────────────
# Count interactions per user in train
train_counts = train_df.groupby("user_id").size()

cold_threshold = 10   # < 10 train interactions
warm_threshold = 50   # >= 50 train interactions

subgroups = {}
for uid, ctx, gt, gt_all, _ in eval_pairs:
    uid_int = int(uid.replace("u", "")) if uid.startswith("u") else uid
    n_train = train_counts.get(uid_int, train_counts.get(uid, 0))
    ctx_len = len(ctx)

    if ctx_len < 5:
        subgroups.setdefault("cold-start (<5 ctx)", []).append((uid, ctx, gt, gt_all, 0))
    elif n_train < cold_threshold:
        subgroups.setdefault("few-shot (5-10 train)", []).append((uid, ctx, gt, gt_all, 0))
    elif n_train < warm_threshold:
        subgroups.setdefault("moderate (10-50 train)", []).append((uid, ctx, gt, gt_all, 0))
    else:
        subgroups.setdefault("warm (50+ train)", []).append((uid, ctx, gt, gt_all, 0))

print(f"\nSubgroups:")
for sg, pairs in sorted(subgroups.items()):
    print(f"  {sg}: {len(pairs)} users")

# ── Run methods on each subgroup ─────────────────────────────────────
import torch
from agents.prediction_agent import PredictionAgent
from agents.confidence_agent import ConfidenceAgent
from agents.diagnostic_agent import DiagnosticAgent
from agents.kg_agent import KnowledgeGraphAgent
from agents.personalization_agent import PersonalizationAgent, extract_user_features

torch.manual_seed(SEED)
np.random.seed(SEED)

# Train agents once
print("\nTraining agents...")
diag = DiagnosticAgent()
irt_params = diag.calibrate_from_interactions(train_df, min_answers_per_q=5)

tag_difficulty = {str(irt_params.question_ids[i]): float(irt_params.b[i])
                  for i in range(len(irt_params.question_ids))}

conf_agent = ConfidenceAgent()
conf_agent.train(train_df, irt_params=irt_params)

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

train_enriched = train_df.copy()
conf_train = conf_agent.classify_batch(interactions=train_enriched)
train_enriched["confidence_class"] = conf_train["classes"]

pred_agent = PredictionAgent()
pred_agent.train(train_enriched, epochs=5, batch_size=256, patience=2)

cluster_params = {}
try:
    pers = PersonalizationAgent()
    uf = extract_user_features(train_df)
    pers.fit(uf)
    cluster_params = {uid: pers.assign_cluster(uid, uf.loc[uid].to_dict())
                      for uid in uf.index[:200]}
except: pass

conf_class_boost = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25], dtype=np.float32)

def compute_ndcg_per_user(pairs, method="mars_full"):
    """Compute NDCG@5, NDCG@10, MRR for each user under a given method."""
    results = []
    for uid, ctx, gt, gt_all, _ in pairs:
        # PredictionAgent signal (always computed)
        conf_result = conf_agent.classify_batch(user_id=uid, interactions=ctx)
        ctx_e = ctx.copy()
        if conf_result["classes"] and len(conf_result["classes"]) == len(ctx_e):
            ctx_e["confidence_class"] = conf_result["classes"]
        result = pred_agent.predict_gaps(uid, recent=ctx_e, threshold=0.0)
        gp = np.array(result["gap_probabilities"], dtype=np.float32)
        gp_min, gp_max = gp.min(), gp.max()
        gp_norm = (gp - gp_min) / (gp_max - gp_min) if gp_max > gp_min else gp

        if method == "pred_only":
            scores = gp_norm
        elif method == "mars_full":
            # IRT signal
            irt_sig = np.zeros(NUM_TAGS, dtype=np.float32)
            recent_acc = ctx["correct"].astype(float).mean()
            theta = np.clip(np.log(recent_acc / (1 - recent_acc + 1e-6) + 1e-6), -3, 3)
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
                mastered = {t for t, s in user_tag_acc.items()
                            if s["t"] >= 3 and s["c"]/s["t"] >= 0.7}
                for t in range(NUM_TAGS):
                    prereqs = prereq_map.get(t, [])
                    if prereqs:
                        kg_sig[t] = sum(1 for p in prereqs if p in mastered) / len(prereqs)
                    else:
                        kg_sig[t] = tag_fail_rate[t]

            # Confidence signal
            rc = ctx_e.get("confidence_class", pd.Series([0]*len(ctx_e)))
            if hasattr(rc, 'values'): rc = rc.values
            mb = np.mean([conf_class_boost[min(int(c), 5)] for c in rc])
            conf_sig = gp_norm * mb

            # Cluster signal
            cl_sig = np.zeros(NUM_TAGS, dtype=np.float32)
            cp = cluster_params.get(uid, {})
            if isinstance(cp, dict):
                dm = cp.get("difficulty_multiplier", 1.0)
                if isinstance(dm, (int, float)):
                    cl_sig = gp_norm * (dm - 1.0) * 0.5

            # Weighted combination (tuned weights)
            w = {"pred": 0.70, "irt": 0.05, "kg": 0.05, "conf": 0.10, "clust": 0.10}
            tw = sum(w.values())
            scores = (w["pred"]/tw * gp_norm + w["irt"]/tw * irt_sig +
                      w["kg"]/tw * kg_sig + w["conf"]/tw * conf_sig +
                      w["clust"]/tw * cl_sig)
        elif method == "popular":
            scores = tag_fail_rate.copy()
        elif method == "random":
            scores = np.random.random(NUM_TAGS).astype(np.float32)
        elif method == "moving_avg":
            scores = np.full(NUM_TAGS, 0.5, dtype=np.float32)
            for _, row in ctx.tail(50).iterrows():
                for t in parse_tags(row.get("tags", [])):
                    if 0 <= t < NUM_TAGS:
                        scores[t] = 1.0 - float(row["correct"])
        else:
            scores = gp_norm

        ranked = np.argsort(scores)[::-1]

        # NDCG@5
        r5 = ranked[:5]
        dcg5 = sum(1.0/np.log2(r+2) for r, t in enumerate(r5) if t in gt)
        i5 = sum(1.0/np.log2(r+2) for r in range(min(len(gt), 5)))
        ndcg5 = dcg5/i5 if i5 > 0 else 0

        # NDCG@10
        r10 = ranked[:10]
        dcg10 = sum(1.0/np.log2(r+2) for r, t in enumerate(r10) if t in gt)
        i10 = sum(1.0/np.log2(r+2) for r in range(min(len(gt), 10)))
        ndcg10 = dcg10/i10 if i10 > 0 else 0

        # MRR
        mrr = 0
        for r, t in enumerate(r10):
            if t in gt:
                mrr = 1.0 / (r + 1)
                break

        # Coverage (unique tags in top-10)
        coverage = len(set(r10)) / NUM_TAGS

        results.append({"ndcg5": ndcg5, "ndcg10": ndcg10, "mrr": mrr, "coverage": coverage})
    return results

# ── Evaluate all methods on all subgroups ────────────────────────────
methods = ["random", "popular", "moving_avg", "pred_only", "mars_full"]

print("\n" + "=" * 90)
print(f"{'Subgroup':25s} {'Method':15s} {'NDCG@5':>8s} {'NDCG@10':>8s} {'MRR':>8s} {'Cov':>8s} {'N':>5s}")
print("=" * 90)

all_subgroup_results = {}
for sg_name in sorted(subgroups.keys()):
    pairs = subgroups[sg_name]
    for method in methods:
        res = compute_ndcg_per_user(pairs, method)
        ndcg5 = np.mean([r["ndcg5"] for r in res])
        ndcg10 = np.mean([r["ndcg10"] for r in res])
        mrr = np.mean([r["mrr"] for r in res])
        cov = np.mean([r["coverage"] for r in res])
        print(f"{sg_name:25s} {method:15s} {ndcg5:8.4f} {ndcg10:8.4f} {mrr:8.4f} {cov:8.4f} {len(pairs):5d}")
        all_subgroup_results[(sg_name, method)] = {
            "ndcg5": ndcg5, "ndcg10": ndcg10, "mrr": mrr, "coverage": cov, "n": len(pairs)
        }
    print("-" * 90)

# ── Summary: where does multi-agent beat pred_only? ──────────────────
print("\n\nDelta: mars_full vs pred_only (positive = multi-agent wins)")
print(f"{'Subgroup':25s} {'dNDCG@5':>10s} {'dNDCG@10':>10s} {'dMRR':>10s} {'dCov':>10s}")
print("-" * 70)
for sg_name in sorted(subgroups.keys()):
    full = all_subgroup_results[(sg_name, "mars_full")]
    pred = all_subgroup_results[(sg_name, "pred_only")]
    print(f"{sg_name:25s} {full['ndcg5']-pred['ndcg5']:+10.4f} "
          f"{full['ndcg10']-pred['ndcg10']:+10.4f} "
          f"{full['mrr']-pred['mrr']:+10.4f} "
          f"{full['coverage']-pred['coverage']:+10.4f}")

# Save results
results_path = Path("results/subgroup_analysis.json")
with open(results_path, "w") as f:
    json.dump({f"{k[0]}|{k[1]}": v for k, v in all_subgroup_results.items()}, f, indent=2)
print(f"\nSaved: {results_path}")
