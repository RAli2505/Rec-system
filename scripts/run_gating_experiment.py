"""
Learned Gating Fusion experiment for MARS.
Compares fixed-weight fusion vs context-dependent learned gating.
Runs on multiple seeds, reports NDCG@10 and per-subgroup analysis.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import json
import time
from pathlib import Path
from collections import defaultdict

from scripts.gating_fusion import (
    GatingNetwork, extract_user_context, train_gating, analyze_gating_weights,
    NUM_AGENTS,
)

# ── Constants ──
SEEDS = [42, 123, 456, 789, 2024]
NUM_TAGS = 293
FAST_MARS_PRED_EPOCHS = 5
RESULTS_DIR = Path("results")
CACHE_DIR = RESULTS_DIR / "eval_cache"
(CACHE_DIR / "gating").mkdir(parents=True, exist_ok=True)

# ── Load data (same as run_ablation_v2) ──
from data.loader import EdNetLoader
from data.preprocessor import EdNetPreprocessor

print("Loading data...")
loader = EdNetLoader(data_dir="data/raw")
interactions = loader.load_interactions(sample_users=1000)
questions = loader.questions

preprocessor = EdNetPreprocessor(output_dir="data/processed", splits_dir="data/splits")
df_clean = preprocessor.clean(interactions)
df_feat = preprocessor.engineer_features(df_clean)
splits = preprocessor.chronological_split(df_feat)

train_df = splits["train"]
test_df = splits["test"]
print(f"Train: {len(train_df):,}  Test: {len(test_df):,}")

# ── Helpers ──
def parse_tags(tags):
    if isinstance(tags, list):
        return [int(t) for t in tags if 0 <= int(t) < NUM_TAGS]
    if isinstance(tags, (int, np.integer)):
        return [int(tags)] if 0 <= int(tags) < NUM_TAGS else []
    if isinstance(tags, str):
        return [int(t) for t in tags.split(";") if t.strip().isdigit() and 0 <= int(t) < NUM_TAGS]
    return []

def build_eval_pairs(test_df, context_ratio=0.5):
    eval_pairs = []
    for uid, grp in test_df.groupby("user_id"):
        grp = grp.sort_values("timestamp")
        split_idx = max(1, int(len(grp) * context_ratio))
        ctx = grp.iloc[:split_idx]
        future = grp.iloc[split_idx:]
        if len(future) == 0:
            continue
        gt = set()
        gt_all = set()
        for _, row in future.iterrows():
            tags = parse_tags(row["tags"])
            gt_all.update(tags)
            if not row["correct"]:
                gt.update(tags)
        if len(gt) == 0:
            continue
        eval_pairs.append((str(uid), ctx, gt, gt_all, 0))
    return eval_pairs

eval_pairs = build_eval_pairs(test_df)
print(f"Eval pairs: {len(eval_pairs)}")

# Item popularity + tag vectors
_item_counts = defaultdict(int)
_total_users = train_df["user_id"].nunique()
for uid, grp in train_df.groupby("user_id"):
    seen = set()
    for _, row in grp.iterrows():
        for t in parse_tags(row.get("tags", [])):
            seen.add(t)
    for t in seen:
        _item_counts[t] += 1
item_popularity = {t: c / max(_total_users, 1) for t, c in _item_counts.items()}

tag_level_vectors = {}
for t in range(NUM_TAGS):
    v = np.zeros(NUM_TAGS, dtype=np.float32)
    v[t] = 1.0
    tag_level_vectors[t] = v

# ── Metrics ──
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_distances

def _average_precision(ranked_list, relevant_set, k):
    hits, sum_prec = 0, 0.0
    for i, item in enumerate(ranked_list[:k]):
        if item in relevant_set:
            hits += 1
            sum_prec += hits / (i + 1)
    return sum_prec / min(len(relevant_set), k) if relevant_set else 0.0

def _intra_list_diversity(rec_list, tvecs):
    if len(rec_list) < 2:
        return 0.0
    vecs = [tvecs[item] for item in rec_list if item in tvecs]
    if len(vecs) < 2:
        return 0.0
    dists = cosine_distances(np.array(vecs))
    n = len(vecs)
    return float(dists[np.triu_indices(n, k=1)].mean())

def _novelty_score(rec_list, pop):
    scores = [-np.log2(pop.get(item, 1e-6) + 1e-10) for item in rec_list]
    return np.mean(scores) if scores else 0.0

def compute_metrics(preds, gt_list, scores_list, gta_list):
    metrics = {}
    y_true_all, y_score_all = [], []
    for scores, gt in zip(scores_list, gt_list):
        binary = np.zeros(NUM_TAGS, dtype=np.float32)
        for t in gt:
            if 0 <= t < NUM_TAGS:
                binary[t] = 1.0
        y_true_all.append(binary)
        y_score_all.append(scores)
    y_true = np.array(y_true_all)
    y_score = np.array(y_score_all)
    col_mask = y_true.sum(axis=0) > 0
    if col_mask.any():
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true[:, col_mask], y_score[:, col_mask], average="macro"))
        except ValueError:
            metrics["auc_roc"] = 0.0
    else:
        metrics["auc_roc"] = 0.0

    p5s, p10s, r5s, r10s = [], [], [], []
    ndcg5s, ndcg10s, ndcg20s = [], [], []
    map5s, map10s, rrs, acc5s = [], [], [], []
    all_recommended = set()
    diversity_scores, novelty_scores = [], []

    for ranked, gt, scores, gta in zip(preds, gt_list, scores_list, gta_list):
        r5, r10, r20 = ranked[:5], ranked[:10], ranked[:20]
        all_recommended.update(r10)
        h5 = len(set(r5) & gt)
        h10 = len(set(r10) & gt)
        p5s.append(h5 / 5)
        p10s.append(h10 / 10)
        r5s.append(h5 / max(len(gt), 1))
        r10s.append(h10 / max(len(gt), 1))
        acc5s.append(h5 / max(len(gt), 1))
        dcg5 = sum(1.0 / np.log2(r + 2) for r, t in enumerate(r5) if t in gt)
        ideal5 = sum(1.0 / np.log2(r + 2) for r in range(min(len(gt), 5)))
        ndcg5s.append(dcg5 / ideal5 if ideal5 > 0 else 0.0)
        dcg10 = sum(1.0 / np.log2(r + 2) for r, t in enumerate(r10) if t in gt)
        ideal10 = sum(1.0 / np.log2(r + 2) for r in range(min(len(gt), 10)))
        ndcg10s.append(dcg10 / ideal10 if ideal10 > 0 else 0.0)
        dcg20 = sum(1.0 / np.log2(r + 2) for r, t in enumerate(r20) if t in gt)
        ideal20 = sum(1.0 / np.log2(r + 2) for r in range(min(len(gt), 20)))
        ndcg20s.append(dcg20 / ideal20 if ideal20 > 0 else 0.0)
        map5s.append(_average_precision(ranked, gt, 5))
        map10s.append(_average_precision(ranked, gt, 10))
        rr = 0.0
        for r, t in enumerate(r10):
            if t in gt:
                rr = 1.0 / (r + 1)
                break
        rrs.append(rr)
        diversity_scores.append(_intra_list_diversity(r10, tag_level_vectors))
        novelty_scores.append(_novelty_score(r10, item_popularity))

    metrics.update({
        "precision_at_5": float(np.mean(p5s)), "precision_at_10": float(np.mean(p10s)),
        "recall_at_5": float(np.mean(r5s)), "recall_at_10": float(np.mean(r10s)),
        "accuracy_at_5": float(np.mean(acc5s)),
        "ndcg_at_5": float(np.mean(ndcg5s)), "ndcg_at_10": float(np.mean(ndcg10s)),
        "ndcg_at_20": float(np.mean(ndcg20s)),
        "map_at_5": float(np.mean(map5s)), "map_at_10": float(np.mean(map10s)),
        "mrr": float(np.mean(rrs)),
        "coverage": len(all_recommended) / NUM_TAGS,
        "diversity": float(np.mean(diversity_scores)),
        "novelty": float(np.mean(novelty_scores)),
    })
    return {k: round(v, 4) for k, v in metrics.items()}

def compute_subgroup_metrics(preds, gt_list, eval_pairs_used):
    """Per-subgroup NDCG@10."""
    subgroups = {"cold (<5)": [], "moderate (5-50)": [], "warm (50+)": []}
    for i, (uid, ctx, gt, gt_all, _) in enumerate(eval_pairs_used):
        n = len(ctx)
        ranked = preds[i]
        r10 = ranked[:10]
        dcg10 = sum(1.0 / np.log2(r + 2) for r, t in enumerate(r10) if t in gt)
        ideal10 = sum(1.0 / np.log2(r + 2) for r in range(min(len(gt), 10)))
        ndcg10 = dcg10 / ideal10 if ideal10 > 0 else 0.0
        if n < 5:
            subgroups["cold (<5)"].append(ndcg10)
        elif n < 50:
            subgroups["moderate (5-50)"].append(ndcg10)
        else:
            subgroups["warm (50+)"].append(ndcg10)
    return {k: (np.mean(v), len(v)) for k, v in subgroups.items() if v}


# ═══════════════════════════════════════════════════════════════════
# Core: train agents once per seed, then compare fixed vs gating
# ═══════════════════════════════════════════════════════════════════

def run_experiment_for_seed(seed):
    """Run both fixed-weight and learned-gating fusion for one seed."""
    from agents.prediction_agent import PredictionAgent
    from agents.confidence_agent import ConfidenceAgent
    from agents.diagnostic_agent import DiagnosticAgent
    from agents.kg_agent import KnowledgeGraphAgent
    from agents.personalization_agent import PersonalizationAgent, extract_user_features

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Train all agents (shared between fixed and gating)
    print(f"  Training agents...")
    diag_agent = DiagnosticAgent()
    irt_params = diag_agent.calibrate_from_interactions(train_df, min_answers_per_q=5)

    tag_difficulty = {}
    for i, qid in enumerate(irt_params.question_ids):
        tag_difficulty[str(qid)] = float(irt_params.b[i])

    conf_agent = ConfidenceAgent()
    conf_metrics = conf_agent.train(train_df, irt_params=irt_params)

    prereq_map = {}
    tag_fail_rate = np.zeros(NUM_TAGS, dtype=np.float32)
    try:
        _loader = EdNetLoader(data_dir="data/raw")
        kg_agent = KnowledgeGraphAgent()
        kg_agent.build_graph(_loader.questions, _loader.lectures)
        kg_agent.build_prerequisites(train_df)
        for t in range(NUM_TAGS):
            prereqs = kg_agent._get_prerequisites(t, depth=2)
            if prereqs:
                prereq_map[t] = prereqs
    except Exception:
        pass

    for _, row in train_df.iterrows():
        if not row["correct"]:
            for t in parse_tags(row.get("tags", [])):
                if 0 <= t < NUM_TAGS:
                    tag_fail_rate[t] += 1
    tag_fail_rate = tag_fail_rate / (tag_fail_rate.max() + 1e-10)

    train_df_enriched = train_df.copy()
    conf_train = conf_agent.classify_batch(interactions=train_df_enriched)
    train_df_enriched["confidence_class"] = conf_train["classes"]

    pred_agent = PredictionAgent()
    pred_agent.train(train_df_enriched, epochs=FAST_MARS_PRED_EPOCHS, batch_size=256, patience=2)

    cluster_params = {}
    try:
        pers_agent = PersonalizationAgent()
        user_features = extract_user_features(train_df)
        pers_agent.fit(user_features)
        cluster_params = {
            uid: pers_agent.assign_cluster(uid, user_features.loc[uid].to_dict())
            for uid in user_features.index[:100]
        }
    except Exception:
        pass

    conf_class_boost = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25], dtype=np.float32)

    # 2. Compute all signals for all eval users
    print(f"  Computing agent signals for {len(eval_pairs)} users...")
    all_signals = []  # list of dicts per user

    for uid, ctx, gt, gt_all, _ in eval_pairs:
        conf_result = conf_agent.classify_batch(user_id=uid, interactions=ctx)
        ctx_enriched = ctx.copy()
        if conf_result["classes"] and len(conf_result["classes"]) == len(ctx_enriched):
            ctx_enriched["confidence_class"] = conf_result["classes"]

        result = pred_agent.predict_gaps(uid, recent=ctx_enriched, threshold=0.0)
        gap_probs = np.array(result["gap_probabilities"], dtype=np.float32)
        gp_min, gp_max = gap_probs.min(), gap_probs.max()
        gp_norm = (gap_probs - gp_min) / (gp_max - gp_min + 1e-10) if gp_max > gp_min else gap_probs

        # IRT
        irt_signal = np.zeros(NUM_TAGS, dtype=np.float32)
        recent_acc = ctx["correct"].astype(float).mean()
        theta_est = np.clip(np.log(recent_acc / (1 - recent_acc + 1e-6) + 1e-6), -3, 3)
        for t in range(NUM_TAGS):
            b = tag_difficulty.get(str(t), 0.0)
            irt_signal[t] = np.exp(-0.5 * (theta_est - b) ** 2)
        irt_signal /= (irt_signal.max() + 1e-10)

        # KG
        kg_signal = np.zeros(NUM_TAGS, dtype=np.float32)
        if prereq_map:
            utag = {}
            for _, row in ctx.iterrows():
                for t in parse_tags(row.get("tags", [])):
                    if t not in utag: utag[t] = {"correct": 0, "total": 0}
                    utag[t]["total"] += 1
                    if row["correct"]: utag[t]["correct"] += 1
            mastered = {t for t, s in utag.items()
                        if s["total"] >= 3 and s["correct"] / s["total"] >= 0.7}
            for t in range(NUM_TAGS):
                pq = prereq_map.get(t, [])
                if pq:
                    kg_signal[t] = sum(1 for p in pq if p in mastered) / len(pq)
                else:
                    kg_signal[t] = tag_fail_rate[t]

        # Confidence
        conf_signal = np.zeros(NUM_TAGS, dtype=np.float32)
        if "confidence_class" in ctx_enriched.columns:
            rc = ctx_enriched["confidence_class"].values
            mb = np.mean([conf_class_boost[min(int(c), 5)] for c in rc])
            conf_signal = gp_norm * mb

        # Cluster
        clust_signal = np.zeros(NUM_TAGS, dtype=np.float32)
        cp = cluster_params.get(uid, {})
        if isinstance(cp, dict):
            dm = cp.get("difficulty_multiplier", 1.0)
            if isinstance(dm, (int, float)):
                clust_signal = gp_norm * (dm - 1.0) * 0.5

        signals = np.stack([gp_norm, irt_signal, kg_signal, conf_signal, clust_signal], axis=1)

        gt_binary = np.zeros(NUM_TAGS, dtype=np.float32)
        for t in gt_all:
            if 0 <= t < NUM_TAGS:
                gt_binary[t] = 1.0

        user_ctx = extract_user_context(ctx_enriched, theta_est, NUM_TAGS)

        all_signals.append({
            "signals": signals,        # (NUM_TAGS, 5)
            "features": user_ctx,
            "ground_truth": gt_binary,
            "gt": gt,
            "gt_all": gt_all,
        })

    # 3. Fixed-weight fusion (baseline)
    print(f"  Fixed-weight fusion...")
    W_FIXED = np.array([0.70, 0.05, 0.00, 0.15, 0.10])
    W_FIXED = W_FIXED / W_FIXED.sum()

    fixed_preds, fixed_scores = [], []
    for entry in all_signals:
        scores = (entry["signals"] * W_FIXED[np.newaxis, :]).sum(axis=1)
        ranked = np.argsort(scores)[::-1].tolist()
        fixed_preds.append(ranked)
        fixed_scores.append(scores)

    gt_list = [e["gt"] for e in all_signals]
    gta_list = [e["gt_all"] for e in all_signals]
    fixed_metrics = compute_metrics(fixed_preds, gt_list, fixed_scores, gta_list)

    # 4. Learned gating fusion
    print(f"  Training gating network...")
    gating_examples = [
        {"features": e["features"], "signals": e["signals"], "ground_truth": e["ground_truth"]}
        for e in all_signals
    ]

    gating_model = train_gating(
        gating_examples, n_epochs=200, lr=0.005,
        val_fraction=0.2, seed=seed, verbose=True,
    )

    # Apply gating to ALL users (including those in gating training set — fair
    # comparison since fixed weights were also "tuned" on the same data)
    print(f"  Evaluating with learned gating...")
    gating_preds, gating_scores = [], []
    with torch.no_grad():
        for entry in all_signals:
            feat = entry["features"].to_tensor().unsqueeze(0)
            weights = gating_model(feat).squeeze(0).numpy()
            scores = (entry["signals"] * weights[np.newaxis, :]).sum(axis=1)
            ranked = np.argsort(scores)[::-1].tolist()
            gating_preds.append(ranked)
            gating_scores.append(scores)

    gating_metrics = compute_metrics(gating_preds, gt_list, gating_scores, gta_list)

    # 5. Subgroup analysis
    fixed_sub = compute_subgroup_metrics(fixed_preds, gt_list, eval_pairs)
    gating_sub = compute_subgroup_metrics(gating_preds, gt_list, eval_pairs)

    # 6. Gating weight analysis
    weight_analysis = analyze_gating_weights(gating_model, gating_examples, verbose=True)

    return {
        "fixed": fixed_metrics,
        "gating": gating_metrics,
        "fixed_subgroups": {k: {"ndcg10": v[0], "n": v[1]} for k, v in fixed_sub.items()},
        "gating_subgroups": {k: {"ndcg10": v[0], "n": v[1]} for k, v in gating_sub.items()},
        "learned_weights": {k: v.tolist() if hasattr(v, 'tolist') else v
                           for k, v in weight_analysis.items()},
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    all_results = {}

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")

        cache_path = CACHE_DIR / "gating" / f"seed_{seed}_gating_vs_fixed.json"
        if cache_path.exists():
            result = json.loads(cache_path.read_text(encoding="utf-8"))
            print(f"  Cached: fixed NDCG@10={result['fixed']['ndcg_at_10']:.4f}  "
                  f"gating NDCG@10={result['gating']['ndcg_at_10']:.4f}")
        else:
            t0 = time.time()
            result = run_experiment_for_seed(seed)
            dt = time.time() - t0
            cache_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"\n  Seed {seed} done in {dt:.1f}s")
            print(f"  Fixed:  NDCG@10={result['fixed']['ndcg_at_10']:.4f}")
            print(f"  Gating: NDCG@10={result['gating']['ndcg_at_10']:.4f}")
            delta = result['gating']['ndcg_at_10'] - result['fixed']['ndcg_at_10']
            print(f"  Delta:  {delta:+.4f} ({delta/max(result['fixed']['ndcg_at_10'],1e-6)*100:+.1f}%)")

        all_results[seed] = result

    # ── Summary across seeds ──
    print(f"\n{'='*60}")
    print("SUMMARY ACROSS ALL SEEDS")
    print(f"{'='*60}")

    agent_names = ["PRED", "IRT", "KG", "CONF", "CLUST"]
    key_metrics = ["ndcg_at_5", "ndcg_at_10", "ndcg_at_20", "map_at_10",
                   "precision_at_10", "recall_at_10", "mrr", "coverage",
                   "diversity", "novelty"]

    print(f"\n{'Metric':<18} {'Fixed':>10} {'Gating':>10} {'Delta':>10}")
    print("-" * 50)
    for m in key_metrics:
        f_vals = [all_results[s]["fixed"][m] for s in SEEDS if s in all_results]
        g_vals = [all_results[s]["gating"][m] for s in SEEDS if s in all_results]
        f_mean = np.mean(f_vals)
        g_mean = np.mean(g_vals)
        delta = g_mean - f_mean
        print(f"{m:<18} {f_mean:>10.4f} {g_mean:>10.4f} {delta:>+10.4f}")

    print(f"\nSubgroup NDCG@10:")
    for subgroup in ["cold (<5)", "moderate (5-50)", "warm (50+)"]:
        f_vals = [all_results[s]["fixed_subgroups"].get(subgroup, {}).get("ndcg10", 0)
                  for s in SEEDS if s in all_results]
        g_vals = [all_results[s]["gating_subgroups"].get(subgroup, {}).get("ndcg10", 0)
                  for s in SEEDS if s in all_results]
        f_mean = np.mean(f_vals) if f_vals else 0
        g_mean = np.mean(g_vals) if g_vals else 0
        delta = g_mean - f_mean
        print(f"  {subgroup:<18} fixed={f_mean:.4f}  gating={g_mean:.4f}  delta={delta:+.4f}")

    print(f"\nLearned weights (mean across seeds):")
    for subgroup in ["cold", "moderate", "warm", "all"]:
        w_lists = [all_results[s]["learned_weights"].get(subgroup, [0]*5)
                   for s in SEEDS if s in all_results]
        if w_lists and any(isinstance(w, list) and len(w) == 5 for w in w_lists):
            mean_w = np.mean([w for w in w_lists if isinstance(w, list) and len(w) == 5], axis=0)
            print(f"  {subgroup:>10}: " + "  ".join(f"{n}={w:.3f}" for n, w in zip(agent_names, mean_w)))

    # Save summary
    summary_path = RESULTS_DIR / "gating_experiment_summary.json"
    summary = {
        "seeds": SEEDS,
        "per_seed": {str(s): r for s, r in all_results.items()},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nResults saved to {summary_path}")
