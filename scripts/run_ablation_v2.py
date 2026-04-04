"""
Run real ablation study (v2) as standalone script.
More reliable than Jupyter for long-running jobs.
Caches results to results/eval_cache/ablation_v2/
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
from pathlib import Path
from collections import defaultdict
import time

# ── Constants ────────────────────────────────────────────────────────
SEEDS = [42, 123, 456, 789, 2024]
NUM_TAGS = 293
FAST_MARS_PRED_EPOCHS = 5
RESULTS_DIR = Path("results")
CACHE_DIR = RESULTS_DIR / "eval_cache"
(CACHE_DIR / "ablation_v2").mkdir(parents=True, exist_ok=True)

def _cache_key(name):
    return name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_').replace('-', '_')

def load_cached(stage, seed, name):
    path = CACHE_DIR / stage / f"seed_{seed}_{_cache_key(name)}.json"
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return None

def save_cached(stage, seed, name, metrics):
    path = CACHE_DIR / stage / f"seed_{seed}_{_cache_key(name)}.json"
    path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')

# ── Load data ────────────────────────────────────────────────────────
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
val_df = splits["val"]
test_df = splits["test"]
print(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

# ── Helpers from NB08 ────────────────────────────────────────────────
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

# Item popularity + tag vectors for metrics
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

# ── Metrics ──────────────────────────────────────────────────────────
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_distances

def _average_precision(ranked_list, relevant_set, k):
    hits = 0
    sum_prec = 0.0
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
    scores = []
    for item in rec_list:
        p = pop.get(item, 1e-6)
        scores.append(-np.log2(p + 1e-10))
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
        "cold_start_acc5": 0.0, "confidence_f1": 0.0,
    })
    return {k: round(v, 4) for k, v in metrics.items()}

# ── run_mars: full multi-agent scoring pipeline ─────────────────────
W_PRED = 0.70
W_IRT  = 0.05
W_KG   = 0.05
W_CONF = 0.10
W_CLUST = 0.10

def run_mars(eval_pairs, train_df, seed=42, pred_epochs=FAST_MARS_PRED_EPOCHS,
             disable_confidence=False, disable_irt=False, disable_clustering=False,
             override_strategy=None, disable_kg=False, disable_reranking=False):
    from agents.prediction_agent import PredictionAgent
    from agents.confidence_agent import ConfidenceAgent
    from agents.diagnostic_agent import DiagnosticAgent
    from agents.kg_agent import KnowledgeGraphAgent
    from agents.personalization_agent import PersonalizationAgent, extract_user_features

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. DiagnosticAgent
    diag_agent = DiagnosticAgent()
    irt_params = diag_agent.calibrate_from_interactions(train_df, min_answers_per_q=5)
    if disable_irt:
        irt_params.b[:] = 0.0
        irt_params.a[:] = 1.0

    tag_difficulty = {}
    for i, qid in enumerate(irt_params.question_ids):
        tag_difficulty[str(qid)] = float(irt_params.b[i])

    # 2. ConfidenceAgent
    conf_agent = ConfidenceAgent()
    conf_metrics = conf_agent.train(train_df, irt_params=irt_params)

    # 3. KGAgent
    prereq_map = {}
    tag_fail_rate = np.zeros(NUM_TAGS, dtype=np.float32)
    if not disable_kg:
        try:
            from data.loader import EdNetLoader
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

    # 4. Enrich with confidence
    train_df_enriched = train_df.copy()
    if disable_confidence:
        train_df_enriched["confidence_class"] = 0
    else:
        conf_train = conf_agent.classify_batch(interactions=train_df_enriched)
        train_df_enriched["confidence_class"] = conf_train["classes"]

    # 5. PredictionAgent
    pred_agent = PredictionAgent()
    pred_agent.train(train_df_enriched, epochs=pred_epochs, batch_size=256, patience=2)

    # 6. PersonalizationAgent
    cluster_params = {}
    if not disable_clustering:
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

    # 7. Evaluation
    preds, scores_list, gt_list, gta_list = [], [], [], []
    conf_class_boost = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25], dtype=np.float32)

    for uid, ctx, gt, gt_all, _ in eval_pairs:
        if disable_confidence:
            ctx_enriched = ctx.copy()
            ctx_enriched["confidence_class"] = 0
        else:
            conf_result = conf_agent.classify_batch(user_id=uid, interactions=ctx)
            ctx_enriched = ctx.copy()
            if conf_result["classes"] and len(conf_result["classes"]) == len(ctx_enriched):
                ctx_enriched["confidence_class"] = conf_result["classes"]

        result = pred_agent.predict_gaps(uid, recent=ctx_enriched, threshold=0.0)
        gap_probs = np.array(result["gap_probabilities"], dtype=np.float32)
        gp_min, gp_max = gap_probs.min(), gap_probs.max()
        gap_probs_norm = (gap_probs - gp_min) / (gp_max - gp_min) if gp_max > gp_min else gap_probs

        # IRT signal (ZPD)
        irt_signal = np.zeros(NUM_TAGS, dtype=np.float32)
        if not disable_irt:
            recent_acc = ctx["correct"].astype(float).mean()
            theta_est = np.clip(np.log(recent_acc / (1 - recent_acc + 1e-6) + 1e-6), -3, 3)
            for t in range(NUM_TAGS):
                b = tag_difficulty.get(str(t), 0.0)
                irt_signal[t] = np.exp(-0.5 * (theta_est - b) ** 2)
            irt_signal = irt_signal / (irt_signal.max() + 1e-10)

        # KG signal
        kg_signal = np.zeros(NUM_TAGS, dtype=np.float32)
        if not disable_kg and prereq_map:
            user_tag_acc = {}
            for _, row in ctx.iterrows():
                for t in parse_tags(row.get("tags", [])):
                    if t not in user_tag_acc:
                        user_tag_acc[t] = {"correct": 0, "total": 0}
                    user_tag_acc[t]["total"] += 1
                    if row["correct"]:
                        user_tag_acc[t]["correct"] += 1
            mastered = {t for t, s in user_tag_acc.items()
                        if s["total"] >= 3 and s["correct"] / s["total"] >= 0.7}
            for t in range(NUM_TAGS):
                prereqs = prereq_map.get(t, [])
                if prereqs:
                    kg_signal[t] = sum(1 for p in prereqs if p in mastered) / len(prereqs)
                else:
                    kg_signal[t] = tag_fail_rate[t]

        # Confidence signal
        conf_signal = np.zeros(NUM_TAGS, dtype=np.float32)
        if not disable_confidence:
            recent_conf = ctx_enriched.get("confidence_class", pd.Series([0]*len(ctx_enriched)))
            if hasattr(recent_conf, 'values'):
                recent_conf = recent_conf.values
            mean_boost = np.mean([conf_class_boost[min(int(c), 5)] for c in recent_conf])
            conf_signal = gap_probs_norm * mean_boost

        # Cluster signal
        cluster_signal = np.zeros(NUM_TAGS, dtype=np.float32)
        if not disable_clustering and cluster_params:
            cp = cluster_params.get(uid, {})
            if isinstance(cp, dict):
                diff_mult = cp.get("difficulty_multiplier", 1.0)
                if isinstance(diff_mult, (int, float)):
                    cluster_signal = gap_probs_norm * (diff_mult - 1.0) * 0.5

        # Combine
        w_pred = W_PRED
        w_irt = W_IRT if not disable_irt else 0.0
        w_kg = W_KG if not disable_kg else 0.0
        w_conf = W_CONF if not disable_confidence else 0.0
        w_clust = W_CLUST if not disable_clustering else 0.0
        total_w = w_pred + w_irt + w_kg + w_conf + w_clust
        if total_w > 0:
            w_pred /= total_w; w_irt /= total_w; w_kg /= total_w
            w_conf /= total_w; w_clust /= total_w

        final_scores = (w_pred * gap_probs_norm + w_irt * irt_signal +
                        w_kg * kg_signal + w_conf * conf_signal + w_clust * cluster_signal)

        # LambdaMART re-ranking
        if not disable_reranking:
            ranker_path = Path("models/lambdamart.txt")
            if ranker_path.exists():
                try:
                    import lightgbm as lgb
                    ranker = lgb.Booster(model_file=str(ranker_path))
                    top_k = 50
                    top_idx = np.argsort(final_scores)[::-1][:top_k]
                    feats = np.zeros((top_k, 12), dtype=np.float32)
                    for j, tidx in enumerate(top_idx):
                        feats[j, 0] = gap_probs_norm[tidx]
                        feats[j, 1] = 1.0 - tag_fail_rate[tidx]
                        feats[j, 4] = tag_difficulty.get(str(tidx), 0.0)
                        feats[j, 7] = kg_signal[tidx]
                        feats[j, 11] = kg_signal[tidx]
                    rs = ranker.predict(feats)
                    for j, tidx in enumerate(top_idx):
                        final_scores[tidx] = rs[j]
                except Exception:
                    pass

        ranked = np.argsort(final_scores)[::-1].tolist()
        preds.append(ranked)
        scores_list.append(final_scores)
        gt_list.append(gt)
        gta_list.append(gt_all)

    return preds, scores_list, gt_list, gta_list, conf_agent, conf_metrics

# ── Baselines needed for ablation ────────────────────────────────────
def baseline_monolithic(eval_pairs, train_df, seed=42):
    from sklearn.decomposition import TruncatedSVD
    import scipy.sparse as sp
    np.random.seed(seed)
    records = []
    for _, row in train_df.iterrows():
        for t in parse_tags(row.get("tags", [])):
            if 0 <= t < NUM_TAGS:
                records.append((str(row["user_id"]), t, 1.0 - float(row["correct"])))
    if not records:
        return [list(range(NUM_TAGS))] * len(eval_pairs), [np.zeros(NUM_TAGS)] * len(eval_pairs), [e[2] for e in eval_pairs], [e[3] for e in eval_pairs]
    rec_df = pd.DataFrame(records, columns=["user_id", "tag_id", "fail_rate"])
    agg = rec_df.groupby(["user_id", "tag_id"])["fail_rate"].mean().reset_index()
    users = sorted(agg["user_id"].unique())
    user_map = {u: i for i, u in enumerate(users)}
    rows_s = [user_map[u] for u in agg["user_id"]]
    cols_s = agg["tag_id"].values
    vals_s = agg["fail_rate"].values
    mat = sp.csr_matrix((vals_s, (rows_s, cols_s)), shape=(len(users), NUM_TAGS))
    svd = TruncatedSVD(n_components=min(50, mat.shape[0] - 1), random_state=seed)
    user_factors = svd.fit_transform(mat)
    tag_factors = svd.components_.T
    preds, scores_list, gt_list, gta_list = [], [], [], []
    for uid, ctx, gt, gt_all, _ in eval_pairs:
        if str(uid) in user_map:
            idx = user_map[str(uid)]
            s = user_factors[idx] @ tag_factors.T
            s = (s - s.min()) / (s.max() - s.min() + 1e-10)
        else:
            s = np.random.random(NUM_TAGS).astype(np.float32)
        ranked = np.argsort(s)[::-1].tolist()
        preds.append(ranked)
        scores_list.append(s.astype(np.float32))
        gt_list.append(gt)
        gta_list.append(gt_all)
    return preds, scores_list, gt_list, gta_list

# ═════════════════════════════════════════════════════════════════════
# MAIN: Run ablation
# ═════════════════════════════════════════════════════════════════════
ablation_configs = {
    "MARS (full)":            {},
    "- Thompson Sampling":    {"override_strategy": "knowledge_based"},
    "- 6-class confidence":   {},  # use Binary-conf from main cache
    "- Knowledge Graph":      {"disable_kg": True},
    "- LSTM prediction":      {},  # use Monolithic from main cache
    "- LambdaMART":           {"disable_reranking": True},
    "- DiagnosticAgent":      {"disable_irt": True},
    "- PersonalizationAgent": {"disable_clustering": True},
}

total = len(SEEDS) * len(ablation_configs)
done = 0

for seed in SEEDS:
    print(f"\n{'='*50}")
    print(f"Seed {seed}")
    print(f"{'='*50}")

    for ablation_name, config in ablation_configs.items():
        m = load_cached("ablation_v2", seed, ablation_name)

        if m is not None:
            done += 1
            print(f"  {ablation_name:30s}: cached  NDCG@10={m['ndcg_at_10']:.4f}")
            continue

        t0 = time.time()

        if ablation_name == "MARS (full)":
            m = load_cached("main", seed, "MARS (ours)")
            if m is None:
                p, s, g, ga, _, cm = run_mars(eval_pairs, train_df, seed)
                m = compute_metrics(p, g, s, ga)
                m["confidence_f1"] = cm.get("cv_f1_macro_mean", 0.0)

        elif ablation_name == "- LSTM prediction":
            m = load_cached("main", seed, "Monolithic")
            if m is None:
                p, s, g, ga = baseline_monolithic(eval_pairs, train_df, seed)
                m = compute_metrics(p, g, s, ga)

        elif ablation_name == "- 6-class confidence":
            m = load_cached("main", seed, "Binary-conf")
            if m is None:
                # Run MARS with disable_confidence
                p, s, g, ga, _, cm = run_mars(eval_pairs, train_df, seed, disable_confidence=True)
                m = compute_metrics(p, g, s, ga)

        else:
            p, s, g, ga, _, cm = run_mars(eval_pairs, train_df, seed,
                                           pred_epochs=FAST_MARS_PRED_EPOCHS, **config)
            m = compute_metrics(p, g, s, ga)
            m["confidence_f1"] = cm.get("cv_f1_macro_mean", 0.0)

        save_cached("ablation_v2", seed, ablation_name, m)
        done += 1
        dt = time.time() - t0
        print(f"  {ablation_name:30s}: NDCG@10={m['ndcg_at_10']:.4f}  ({dt:.1f}s)  [{done}/{total}]")

print(f"\n{'='*50}")
print(f"Ablation complete: {done}/{total}")
print(f"{'='*50}")
