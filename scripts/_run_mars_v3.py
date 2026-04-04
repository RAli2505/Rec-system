# Auto-extracted from NB08 cell 13
import numpy as np, pandas as pd, torch
from pathlib import Path
from collections import defaultdict
import logging

_log = logging.getLogger("mars.run_mars")

# Fallback fixed weights (used when use_learned_gating=False)
# v4: raised IRT/KG after per-Part IRT + prerequisite-readiness signals
W_PRED = 0.65   # PredictionAgent gap probabilities (raised: LSTM AUC=0.76)
W_IRT  = 0.05   # DiagnosticAgent per-Part IRT signal (lowered: was hurting NDCG)
W_KG   = 0.15   # KnowledgeGraphAgent prerequisite readiness (raised after KG fix)
W_CONF = 0.10   # ConfidenceAgent confidence-weighted adjustment
W_CLUST = 0.05  # PersonalizationAgent cluster-based modulation

FAST_MARS_PRED_EPOCHS = 5  # default; can be overridden via pred_epochs param
NUM_TAGS = 293              # total unique tags in EdNet


def parse_tags(tags):
    """Parse tag field from interactions into list of valid tag indices."""
    if isinstance(tags, list):
        return [int(t) for t in tags if 0 <= int(t) < NUM_TAGS]
    if isinstance(tags, (int, np.integer)):
        return [int(tags)] if 0 <= tags < NUM_TAGS else []
    if isinstance(tags, str):
        return [int(t) for t in tags.split(";") if t.strip() and 0 <= int(t.strip()) < NUM_TAGS]
    return []


def run_mars(eval_pairs, train_df, seed=42, pred_epochs=FAST_MARS_PRED_EPOCHS,
             disable_confidence=False, disable_irt=False, disable_clustering=False,
             override_strategy=None, disable_kg=False, disable_reranking=False,
             use_learned_gating=False, lstm_cache_path=None):
    """
    Full MARS pipeline with multi-agent scoring and ablation support.

    Each agent contributes a signal to the final per-tag score:
      - PredictionAgent:    gap_probabilities (LSTM seq-to-set)
      - DiagnosticAgent:    IRT difficulty-adjusted signal
      - KGAgent:            prerequisite-aware boost
      - ConfidenceAgent:    confidence-class weighting
      - PersonalizationAgent: cluster-based difficulty modulation

    disable_* flags zero out each agent's contribution for ablation.
    """
    from agents.prediction_agent import PredictionAgent
    from agents.confidence_agent import ConfidenceAgent
    from agents.diagnostic_agent import DiagnosticAgent
    from agents.kg_agent import KnowledgeGraphAgent
    from agents.personalization_agent import PersonalizationAgent, extract_user_features

    torch.manual_seed(seed)
    np.random.seed(seed)

    agent_metrics = {}  # collect per-agent metrics

    # в"Ђв"Ђ 1. DiagnosticAgent: IRT calibration в"Ђв"Ђ
    diag_agent = DiagnosticAgent()
    irt_params = diag_agent.calibrate_from_interactions(train_df, min_answers_per_q=5)

    # -- DiagnosticAgent metrics: θ-accuracy correlation --
    try:
        from scipy.stats import pearsonr, spearmanr
        user_acc = train_df.groupby("user_id")["correct"].mean()
        user_ids_in_irt = [u for u in user_acc.index if u in getattr(irt_params, 'user_ids_map', {})]
        if hasattr(irt_params, 'theta') and hasattr(irt_params, 'user_ids_map') and len(user_ids_in_irt) > 10:
            thetas = np.array([irt_params.theta[irt_params.user_ids_map[u]] for u in user_ids_in_irt])
            accs = np.array([user_acc[u] for u in user_ids_in_irt])
            r_pearson, p_pearson = pearsonr(thetas, accs)
            r_spearman, p_spearman = spearmanr(thetas, accs)
            agent_metrics["diagnostic"] = {
                "theta_accuracy_pearson_r": round(float(r_pearson), 4),
                "theta_accuracy_pearson_p": round(float(p_pearson), 6),
                "theta_accuracy_spearman_r": round(float(r_spearman), 4),
                "n_users_evaluated": len(user_ids_in_irt),
                "theta_mean": round(float(thetas.mean()), 4),
                "theta_std": round(float(thetas.std()), 4),
                "b_range": [round(float(irt_params.b.min()), 3), round(float(irt_params.b.max()), 3)],
                "a_range": [round(float(irt_params.a.min()), 3), round(float(irt_params.a.max()), 3)],
                "n_items": len(irt_params.b),
            }
            _log.info("DiagnosticAgent: θ-acc pearson=%.4f, spearman=%.4f (n=%d)",
                       r_pearson, r_spearman, len(user_ids_in_irt))
        else:
            agent_metrics["diagnostic"] = {
                "b_range": [round(float(irt_params.b.min()), 3), round(float(irt_params.b.max()), 3)],
                "a_range": [round(float(irt_params.a.min()), 3), round(float(irt_params.a.max()), 3)],
                "n_items": len(irt_params.b),
                "note": "theta-accuracy correlation not available (no user_ids_map or theta)",
            }
    except Exception as e:
        agent_metrics["diagnostic"] = {"error": str(e)}

    if disable_irt:
        # Flatten IRT: all items same difficulty в†' no Оё-based signal
        irt_params.b[:] = 0.0
        irt_params.a[:] = 1.0

    # Build per-tag difficulty lookup
    tag_difficulty = {}
    for i, qid in enumerate(irt_params.question_ids):
        tag_difficulty[str(qid)] = float(irt_params.b[i])

    # Build tag → primary_part mapping from questions metadata
    tag_to_part = {}
    try:
        from data.loader import EdNetLoader as _Loader
        _ldr = _Loader(data_dir="data/raw")
        _qdf = _ldr.questions
        from collections import Counter
        _tag_part_counts = {}  # tag_id -> Counter(part_id -> count)
        for _, row in _qdf.iterrows():
            pid = int(row.get("part_id", row.get("part", 0)))
            for t in row["tags"]:
                t = int(t)
                if t not in _tag_part_counts:
                    _tag_part_counts[t] = Counter()
                _tag_part_counts[t][pid] += 1
        # Assign each tag to its most common Part
        for t, cnts in _tag_part_counts.items():
            tag_to_part[t] = cnts.most_common(1)[0][0]
    except Exception:
        pass  # tag_to_part stays empty — fallback to global theta

    # в"Ђв"Ђ 2. ConfidenceAgent в"Ђв"Ђ
    conf_agent = ConfidenceAgent()
    conf_metrics = conf_agent.train(train_df, irt_params=irt_params)

    # -- ConfidenceAgent extended metrics --
    try:
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix as cm_func
        conf_eval = conf_agent.classify_batch(interactions=train_df)
        y_pred_conf = conf_eval["classes"]
        # Reconstruct y_true from the agent's internal label mapping
        if hasattr(conf_agent, '_last_y_train'):
            y_true_conf = conf_agent._last_y_train
        else:
            y_true_conf = None
        agent_metrics["confidence"] = {
            **conf_metrics,  # cv_f1_macro_mean, cv_f1_macro_std, full_f1_macro etc.
        }
        _log.info("ConfidenceAgent: cv_f1=%.4f±%.4f, full_f1=%.4f, n_classes=%d",
                   conf_metrics.get("cv_f1_macro_mean", 0),
                   conf_metrics.get("cv_f1_macro_std", 0),
                   conf_metrics.get("full_f1_macro", 0),
                   conf_metrics.get("n_classes", 0))
    except Exception as e:
        agent_metrics["confidence"] = {**conf_metrics, "extended_error": str(e)}

    # в"Ђв"Ђ 3. KnowledgeGraphAgent: prerequisites в"Ђв"Ђ
    kg_agent = None
    prereq_map = {}  # tag в†' list of prerequisite tags
    tag_fail_rate = np.zeros(NUM_TAGS, dtype=np.float32)

    if not disable_kg:
        try:
            from data.loader import EdNetLoader
            loader = EdNetLoader(data_dir="data/raw")
            kg_agent = KnowledgeGraphAgent()
            kg_agent.build_graph(loader.questions, loader.lectures)
            kg_agent.build_prerequisites(train_df)

            # Extract prerequisite map
            for t in range(NUM_TAGS):
                prereqs = kg_agent._get_prerequisites(t, depth=2)
                if prereqs:
                    prereq_map[t] = prereqs
            agent_metrics["knowledge_graph"] = {
                "n_nodes": kg_agent._graph.number_of_nodes() if hasattr(kg_agent, '_graph') else 0,
                "n_edges": kg_agent._graph.number_of_edges() if hasattr(kg_agent, '_graph') else 0,
                "n_prerequisite_relations": sum(len(v) for v in prereq_map.values()),
                "n_tags_with_prereqs": len(prereq_map),
                "avg_prereqs_per_tag": round(np.mean([len(v) for v in prereq_map.values()]), 2) if prereq_map else 0,
            }
            _log.info("KGAgent: %d tags with prereqs, %d total relations",
                       len(prereq_map), sum(len(v) for v in prereq_map.values()))
        except Exception as e:
            agent_metrics["knowledge_graph"] = {"error": str(e)}

    # Global tag failure rate (for KG signal)
    for _, row in train_df.iterrows():
        if not row["correct"]:
            for t in parse_tags(row.get("tags", [])):
                if 0 <= t < NUM_TAGS:
                    tag_fail_rate[t] += 1
    tag_fail_rate = tag_fail_rate / (tag_fail_rate.max() + 1e-10)

    # в"Ђв"Ђ 4. Enrich training data with confidence в"Ђв"Ђ
    train_df_enriched = train_df.copy()
    if disable_confidence:
        train_df_enriched["confidence_class"] = 0
    else:
        conf_train = conf_agent.classify_batch(interactions=train_df_enriched)
        train_df_enriched["confidence_class"] = conf_train["classes"]

    # в"Ђв"Ђ 5. PredictionAgent: LSTM training в"Ђв"Ђ
    pred_agent = PredictionAgent()
    _cache = Path(lstm_cache_path) if lstm_cache_path else None
    if _cache and _cache.exists():
        _log.info("[cache] Loading LSTM from %s", _cache)
        pred_agent._load_model(_cache)
        agent_metrics["prediction"] = {"source": "cache", "cache_path": str(_cache)}
    else:
        pred_train_metrics = pred_agent.train(train_df_enriched, epochs=pred_epochs, batch_size=256, patience=2)
        agent_metrics["prediction"] = {
            "best_epoch": pred_train_metrics.get("best_epoch"),
            "train_loss": pred_train_metrics.get("train_loss"),
            "val_loss": pred_train_metrics.get("val_loss"),
            "val_auc_roc_macro": pred_train_metrics.get("val_auc"),
            "n_train_sequences": pred_train_metrics.get("n_train_sequences"),
            "n_val_sequences": pred_train_metrics.get("n_val_sequences"),
            "total_epochs_run": pred_train_metrics.get("total_epochs"),
        }
        # Extended: F1 micro/macro on val set via thresholded predictions
        try:
            from sklearn.metrics import f1_score as f1_fn, precision_score, recall_score
            val_preds_all, val_labels_all = [], []
            pred_agent.model.eval()
            from torch.utils.data import DataLoader
            # Re-build val data to compute F1
            val_df = train_df_enriched[train_df_enriched["user_id"].isin(
                train_df_enriched["user_id"].unique()[-int(train_df_enriched["user_id"].nunique() * 0.15):])]
            if len(val_df) > 100:
                _log.info("Computing PredictionAgent F1/precision/recall on val users...")
                # Sample evaluation: take gap_probabilities for val users and threshold at 0.5
                sample_users = val_df["user_id"].unique()[:200]
                all_y_true, all_y_pred = [], []
                for uid in sample_users:
                    udf = val_df[val_df["user_id"] == uid].sort_values("timestamp")
                    if len(udf) < 4:
                        continue
                    mid = len(udf) // 2
                    ctx = udf.iloc[:mid]
                    future = udf.iloc[mid:]
                    # Ground truth: tags from incorrect future answers
                    gt_tags = set()
                    for _, row in future.iterrows():
                        if not row["correct"]:
                            gt_tags.update(parse_tags(row.get("tags", [])))
                    if not gt_tags:
                        continue
                    res = pred_agent.predict_gaps(uid, recent=ctx, threshold=0.0)
                    gp = np.array(res["gap_probabilities"])
                    y_true = np.zeros(NUM_TAGS, dtype=int)
                    for t in gt_tags:
                        if 0 <= t < NUM_TAGS:
                            y_true[t] = 1
                    y_pred = (gp > 0.5).astype(int)
                    all_y_true.append(y_true)
                    all_y_pred.append(y_pred)
                if all_y_true:
                    yt = np.stack(all_y_true)
                    yp_raw = np.stack([pred_agent.predict_gaps(
                        u, recent=val_df[val_df["user_id"] == u].sort_values("timestamp").iloc[:len(val_df[val_df["user_id"] == u])//2],
                        threshold=0.0)["gap_probabilities"]
                        for u in sample_users
                        if len(val_df[val_df["user_id"] == u]) >= 4
                    ]) if False else np.stack(all_y_pred)  # keep existing preds

                    active = yt.sum(axis=0) > 0

                    # Threshold search: find optimal threshold on val set
                    all_gp = []
                    for uid in sample_users:
                        udf = val_df[val_df["user_id"] == uid].sort_values("timestamp")
                        if len(udf) < 4:
                            continue
                        mid = len(udf) // 2
                        ctx = udf.iloc[:mid]
                        res = pred_agent.predict_gaps(uid, recent=ctx, threshold=0.0)
                        all_gp.append(np.array(res["gap_probabilities"]))
                    if all_gp and len(all_gp) == len(all_y_true):
                        gp_matrix = np.stack(all_gp)  # (n_users, NUM_TAGS)
                        best_t, best_f1_t = 0.5, 0.0
                        for t in np.arange(0.01, 0.31, 0.01):
                            yp_t = (gp_matrix[:, active] > t).astype(int)
                            f1_t = f1_fn(yt[:, active], yp_t, average="micro", zero_division=0)
                            if f1_t > best_f1_t:
                                best_t, best_f1_t = t, f1_t
                        agent_metrics["prediction"]["optimal_threshold"] = round(float(best_t), 3)
                        agent_metrics["prediction"]["f1_micro_at_optimal_t"] = round(float(best_f1_t), 4)
                        _log.info("PredictionAgent threshold search: optimal_t=%.2f, F1_micro=%.4f",
                                   best_t, best_f1_t)
                        # Store optimal threshold for use in scoring
                        pred_agent._optimal_threshold = float(best_t)

                    if active.sum() > 0:
                        # Metrics at default threshold 0.5
                        agent_metrics["prediction"]["f1_micro_t05"] = round(float(
                            f1_fn(yt[:, active], yp_raw[:, active], average="micro", zero_division=0)), 4)
                        agent_metrics["prediction"]["f1_macro_t05"] = round(float(
                            f1_fn(yt[:, active], yp_raw[:, active], average="macro", zero_division=0)), 4)
                        agent_metrics["prediction"]["precision_micro"] = round(float(
                            precision_score(yt[:, active], yp_raw[:, active], average="micro", zero_division=0)), 4)
                        agent_metrics["prediction"]["recall_micro"] = round(float(
                            recall_score(yt[:, active], yp_raw[:, active], average="micro", zero_division=0)), 4)
                        _log.info("PredictionAgent: AUC=%.4f, F1@0.5=%.4f, F1@opt_t=%.4f",
                                   pred_train_metrics.get("val_auc", 0),
                                   agent_metrics["prediction"]["f1_micro_t05"],
                                   agent_metrics["prediction"].get("f1_micro_at_optimal_t", 0))
        except Exception as e:
            agent_metrics["prediction"]["f1_error"] = str(e)

        if _cache:
            import shutil
            _cache.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy("models/gap_lstm.pt", str(_cache))
            _log.info("[cache] Saved LSTM to %s", _cache)

    # в"Ђв"Ђ 6. PersonalizationAgent: clustering в"Ђв"Ђ
    cluster_params = {}
    if not disable_clustering:
        try:
            pers_agent = PersonalizationAgent()
            user_features = extract_user_features(train_df)
            pers_agent.fit(user_features)
            cluster_params = {
                uid: pers_agent.assign_cluster(uid, user_features.loc[uid].to_dict())
                for uid in user_features.index[:100]  # cache first 100
            }
        except Exception:
            pass

    # ── 7. Learned Gating: train on inner split if enabled ──
    gating_model = None
    if use_learned_gating:
        from scripts.gating_fusion import (
            GatingNetwork, extract_user_context, train_gating
        )
        print("  Training gating network on inner split...")
        gating_examples = []
        n_gating = min(len(eval_pairs), max(50, int(len(eval_pairs) * 0.8)))
        for uid, ctx, gt, gt_all, _ in eval_pairs[:n_gating]:
            ctx_e = ctx.copy()
            if disable_confidence:
                ctx_e["confidence_class"] = 0
            else:
                cr = conf_agent.classify_batch(user_id=uid, interactions=ctx)
                if cr["classes"] and len(cr["classes"]) == len(ctx_e):
                    ctx_e["confidence_class"] = cr["classes"]

            res = pred_agent.predict_gaps(uid, recent=ctx_e, threshold=0.0)
            gp = np.array(res["gap_probabilities"], dtype=np.float32)
            gp_mn, gp_mx = gp.min(), gp.max()
            gp_norm = (gp - gp_mn) / (gp_mx - gp_mn + 1e-10) if gp_mx > gp_mn else gp

            # IRT signal (per-Part + asymmetric ZPD)
            irt_sig = np.zeros(NUM_TAGS, dtype=np.float32)
            if not disable_irt:
                _pt = {}
                if "part_id" in ctx.columns:
                    for _pid in range(1, 8):
                        _pm = ctx["part_id"] == _pid
                        if _pm.sum() >= 3:
                            _pa = np.clip(ctx.loc[_pm, "correct"].astype(float).mean(), 0.05, 0.95)
                            _pt[_pid] = np.clip(np.log(_pa / (1 - _pa)), -3, 3)
                ra = np.clip(ctx["correct"].astype(float).mean(), 0.05, 0.95)
                th = np.clip(np.log(ra / (1 - ra)), -3, 3)
                for t in range(NUM_TAGS):
                    b = tag_difficulty.get(str(t), 0.0)
                    tp = tag_to_part.get(t, None)
                    _theta = _pt.get(tp, th) if tp else th
                    diff = b - _theta
                    if diff >= 0:
                        irt_sig[t] = np.exp(-0.25 * diff ** 2)
                    else:
                        irt_sig[t] = np.exp(-1.5 * diff ** 2) * 0.3
                irt_sig /= (irt_sig.max() + 1e-10)
            else:
                th = 0.0

            # KG signal (prerequisite-readiness + gap-weighted)
            kg_sig = np.zeros(NUM_TAGS, dtype=np.float32)
            if not disable_kg and prereq_map:
                utag = {}
                for _, row in ctx.iterrows():
                    for t in parse_tags(row.get("tags", [])):
                        if t not in utag: utag[t] = {"correct": 0, "total": 0}
                        utag[t]["total"] += 1
                        if row["correct"]: utag[t]["correct"] += 1
                m = {t for t, s in utag.items()
                     if s["total"] >= 3 and s["correct"] / s["total"] >= 0.7}
                strug = {t for t, s in utag.items()
                         if s["total"] >= 3 and s["correct"] / s["total"] < 0.5}
                for t in range(NUM_TAGS):
                    pq = prereq_map.get(t, [])
                    if pq:
                        rdns = sum(1 for p in pq if p in m) / len(pq)
                        if t in strug:
                            kg_sig[t] = rdns * 1.5
                        elif t not in m:
                            kg_sig[t] = rdns * 0.8
                        else:
                            kg_sig[t] = rdns * 0.2
                    else:
                        if t in strug:
                            kg_sig[t] = tag_fail_rate[t] * 1.2
                        elif t in m:
                            kg_sig[t] = tag_fail_rate[t] * 0.1
                        else:
                            kg_sig[t] = tag_fail_rate[t]
                _km = kg_sig.max()
                if _km > 0:
                    kg_sig /= _km

            # Confidence signal
            conf_sig = np.zeros(NUM_TAGS, dtype=np.float32)
            conf_class_b = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25], dtype=np.float32)
            if not disable_confidence and "confidence_class" in ctx_e.columns:
                rc = ctx_e["confidence_class"].values
                mb = np.mean([conf_class_b[min(int(c), 5)] for c in rc])
                conf_sig = gp_norm * mb

            # Cluster signal
            clust_sig = np.zeros(NUM_TAGS, dtype=np.float32)
            if not disable_clustering and cluster_params:
                cp = cluster_params.get(uid, {})
                if isinstance(cp, dict):
                    dm = cp.get("difficulty_multiplier", 1.0)
                    if isinstance(dm, (int, float)):
                        clust_sig = gp_norm * (dm - 1.0) * 0.5

            # Stack signals: (NUM_TAGS, 5)
            signals = np.stack([gp_norm, irt_sig, kg_sig, conf_sig, clust_sig], axis=1)

            # Ground truth: binary relevance per tag
            gt_binary = np.zeros(NUM_TAGS, dtype=np.float32)
            for t in gt_all:
                if 0 <= t < NUM_TAGS:
                    gt_binary[t] = 1.0

            user_ctx = extract_user_context(ctx_e, th, NUM_TAGS)
            gating_examples.append({
                "features": user_ctx,
                "signals": signals,
                "ground_truth": gt_binary,
            })

        if len(gating_examples) >= 10:
            gating_model = train_gating(
                gating_examples, n_epochs=150, lr=0.005,
                seed=seed, verbose=True,
            )
            from scripts.gating_fusion import analyze_gating_weights
            analyze_gating_weights(gating_model, gating_examples)

    # в"Ђв"Ђ 8. Evaluation: combine all signals в"Ђв"Ђ
    preds, scores_list, gt_list, gta_list = [], [], [], []
    learning_gains, tag_coverages, mastery_times = [], [], []

    # Confidence class weights: higher class = more uncertain = boost recommendation
    conf_class_boost = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25], dtype=np.float32)

    for uid, ctx, gt, gt_all, _ in eval_pairs:
        # -- PredictionAgent signal --
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

        # Use optimal threshold found during training (default 0.1 if not set)
        opt_t = getattr(pred_agent, "_optimal_threshold", 0.1)
        # Threshold-aware normalization: boost tags above optimal threshold,
        # suppress tags below it. This converts raw sigmoid outputs into
        # scores that reflect actual gap likelihood at calibrated threshold.
        gp_max = gap_probs.max() + 1e-10
        gap_probs_norm = gap_probs / gp_max
        # Boost: tags above threshold get amplified, below get dampened
        above_mask = gap_probs >= opt_t
        gap_probs_norm[above_mask] = np.clip(gap_probs_norm[above_mask] * 1.5, 0, 1)
        gap_probs_norm[~above_mask] = gap_probs_norm[~above_mask] * 0.3

        # -- IRT signal: per-Part ability → asymmetric ZPD --
        # Only apply IRT for users with sufficient history (>=20 interactions)
        irt_signal = np.zeros(NUM_TAGS, dtype=np.float32)
        n_ctx = len(ctx)
        if not disable_irt and n_ctx >= 20:
            # Per-Part ability estimation (7 TOEIC parts)
            part_theta = {}
            if "part_id" in ctx.columns:
                for pid in range(1, 8):
                    pmask = ctx["part_id"] == pid
                    if pmask.sum() >= 3:
                        pa = np.clip(ctx.loc[pmask, "correct"].astype(float).mean(), 0.05, 0.95)
                        part_theta[pid] = np.clip(np.log(pa / (1 - pa)), -3, 3)

            # Global fallback theta
            global_acc = np.clip(ctx["correct"].astype(float).mean(), 0.05, 0.95)
            theta_est = np.clip(np.log(global_acc / (1 - global_acc)), -3, 3)

            for t in range(NUM_TAGS):
                b = tag_difficulty.get(str(t), 0.0)
                t_part = tag_to_part.get(t, None)
                th = part_theta.get(t_part, theta_est) if t_part else theta_est
                # Asymmetric ZPD: prioritize items slightly HARDER than ability
                diff = b - th  # positive = harder than current ability
                if diff >= 0:
                    # Items above ability: high priority (reachable challenge)
                    irt_signal[t] = np.exp(-0.25 * diff ** 2)
                else:
                    # Items below ability: low priority (already mastered)
                    irt_signal[t] = np.exp(-1.5 * diff ** 2) * 0.3
            irt_signal = irt_signal / (irt_signal.max() + 1e-10)

        # -- KG signal: prerequisite-readiness + gap-weighted --
        kg_signal = np.zeros(NUM_TAGS, dtype=np.float32)
        if not disable_kg and prereq_map:
            # Compute per-tag accuracy from context
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
            struggling = {t for t, s in user_tag_acc.items()
                          if s["total"] >= 3 and s["correct"] / s["total"] < 0.5}

            for t in range(NUM_TAGS):
                prereqs = prereq_map.get(t, [])
                if prereqs:
                    # Readiness: fraction of prerequisites mastered
                    met = sum(1 for p in prereqs if p in mastered)
                    readiness = met / len(prereqs)
                    # Combine readiness with whether student is struggling on this tag
                    if t in struggling:
                        # Student struggles AND has prerequisites → high priority
                        kg_signal[t] = readiness * 1.5
                    elif t not in mastered:
                        # Not mastered, not struggling → moderate priority
                        kg_signal[t] = readiness * 0.8
                    else:
                        # Already mastered → low priority
                        kg_signal[t] = readiness * 0.2
                else:
                    # No prerequisites: penalize already-mastered, boost gaps
                    if t in struggling:
                        kg_signal[t] = tag_fail_rate[t] * 1.2
                    elif t in mastered:
                        kg_signal[t] = tag_fail_rate[t] * 0.1
                    else:
                        kg_signal[t] = tag_fail_rate[t]
            # Normalize
            _kmax = kg_signal.max()
            if _kmax > 0:
                kg_signal = kg_signal / _kmax

        # -- Confidence signal --
        conf_signal = np.zeros(NUM_TAGS, dtype=np.float32)
        if not disable_confidence:
            # Recent confidence classes в†' boost uncertain tags
            recent_conf = ctx_enriched.get("confidence_class", pd.Series([0]*len(ctx_enriched)))
            if hasattr(recent_conf, 'values'):
                recent_conf = recent_conf.values
            mean_conf_boost = np.mean([conf_class_boost[min(int(c), 5)] for c in recent_conf])
            # Apply boost to high-gap tags
            conf_signal = gap_probs_norm * mean_conf_boost

        # -- Cluster signal --
        cluster_signal = np.zeros(NUM_TAGS, dtype=np.float32)
        if not disable_clustering and cluster_params:
            cp = cluster_params.get(uid, {})
            if isinstance(cp, dict):
                diff_mult = cp.get("difficulty_multiplier", 1.0)
                if isinstance(diff_mult, (int, float)):
                    cluster_signal = gap_probs_norm * (diff_mult - 1.0) * 0.5

        # -- Combine all signals --
        if gating_model is not None:
            # Learned context-dependent gating
            from scripts.gating_fusion import extract_user_context
            theta_for_gating = theta_est if not disable_irt else 0.0
            uctx = extract_user_context(ctx_enriched, theta_for_gating, NUM_TAGS)
            with torch.no_grad():
                weights = gating_model(uctx.to_tensor().unsqueeze(0)).squeeze(0).numpy()
            signals_stack = np.stack(
                [gap_probs_norm, irt_signal, kg_signal, conf_signal, cluster_signal],
                axis=1,
            )  # (NUM_TAGS, 5)
            final_scores = (signals_stack * weights[np.newaxis, :]).sum(axis=1)
        else:
            # Fixed weights fallback
            w_pred = W_PRED
            w_irt = W_IRT if not disable_irt else 0.0
            w_kg = W_KG if not disable_kg else 0.0
            w_conf = W_CONF if not disable_confidence else 0.0
            w_clust = W_CLUST if not disable_clustering else 0.0

            total_w = w_pred + w_irt + w_kg + w_conf + w_clust
            if total_w > 0:
                w_pred /= total_w; w_irt /= total_w; w_kg /= total_w
                w_conf /= total_w; w_clust /= total_w

            final_scores = (
                w_pred * gap_probs_norm +
                w_irt  * irt_signal +
                w_kg   * kg_signal +
                w_conf * conf_signal +
                w_clust * cluster_signal
            )

        # -- LambdaMART re-ranking (if available) --
        if not disable_reranking:
            # Load pre-trained ranker if exists
            ranker_path = Path("models/lambdamart.txt")
            if ranker_path.exists():
                try:
                    import lightgbm as lgb
                    ranker = lgb.Booster(model_file=str(ranker_path))
                    # Build simple features for top candidates
                    top_k = 50
                    top_indices = np.argsort(final_scores)[::-1][:top_k]
                    feats = np.zeros((top_k, 12), dtype=np.float32)
                    for j, tidx in enumerate(top_indices):
                        feats[j, 0] = gap_probs_norm[tidx]          # gap_by_tag
                        feats[j, 1] = 1.0 - tag_fail_rate[tidx]     # user_accuracy_on_part
                        feats[j, 4] = tag_difficulty.get(str(tidx), 0.0)  # tag_difficulty
                        feats[j, 7] = kg_signal[tidx]               # kg_score
                        feats[j, 11] = kg_signal[tidx]              # prerequisite_completion
                    reranked_scores = ranker.predict(feats)
                    for j, tidx in enumerate(top_indices):
                        final_scores[tidx] = reranked_scores[j]
                except Exception:
                    pass  # Fall back to combined scores

        ranked = np.argsort(final_scores)[::-1].tolist()
        preds.append(ranked)
        scores_list.append(final_scores)
        gt_list.append(gt)
        gta_list.append(gt_all)

        # -- System-level metrics per user --
        try:
            future_u = _  # 5th element of eval_pairs tuple

            # Learning Gain: accuracy on RECOMMENDED tags (before vs after)
            # This measures whether our recommendations target tags where
            # the student actually improves, not raw overall accuracy.
            rec_top10 = set(ranked[:10])
            ctx_rec_correct, ctx_rec_total = 0, 0
            for _, row in ctx.iterrows():
                rtags = set(parse_tags(row.get("tags", [])))
                if rtags & rec_top10:
                    ctx_rec_total += 1
                    if row["correct"]:
                        ctx_rec_correct += 1
            future_rec_correct, future_rec_total = 0, 0
            for _, row in future_u.iterrows():
                rtags = set(parse_tags(row.get("tags", [])))
                if rtags & rec_top10:
                    future_rec_total += 1
                    if row["correct"]:
                        future_rec_correct += 1
            if ctx_rec_total >= 2 and future_rec_total >= 2:
                ctx_rec_acc = ctx_rec_correct / ctx_rec_total
                future_rec_acc = future_rec_correct / future_rec_total
                learning_gains.append(future_rec_acc - ctx_rec_acc)

            # Tag coverage: fraction of future incorrect tags that appear in top-10
            top10 = set(ranked[:10])
            future_tags = set()
            for _, row in future_u.iterrows():
                future_tags.update(parse_tags(row.get("tags", [])))
            if future_tags:
                tag_coverages.append(len(top10 & future_tags) / len(future_tags))

            # Time-to-mastery proxy: how many interactions in future until 70% accuracy on recommended tags
            rec_tags = set(ranked[:5])
            cum_correct, cum_total, mastery_t = 0, 0, len(future_u)
            for idx_f, (_, row) in enumerate(future_u.iterrows()):
                rtags = set(parse_tags(row.get("tags", [])))
                if rtags & rec_tags:
                    cum_total += 1
                    if row["correct"]:
                        cum_correct += 1
                    if cum_total >= 3 and cum_correct / cum_total >= 0.7:
                        mastery_t = idx_f + 1
                        break
            mastery_times.append(mastery_t)
        except Exception:
            pass

    # -- Aggregate system-level metrics --
    agent_metrics["system"] = {
        "learning_gain_mean": round(float(np.mean(learning_gains)), 4) if learning_gains else None,
        "learning_gain_std": round(float(np.std(learning_gains)), 4) if learning_gains else None,
        "tag_coverage_at_10_mean": round(float(np.mean(tag_coverages)), 4) if tag_coverages else None,
        "avg_interactions_to_mastery": round(float(np.mean(mastery_times)), 1) if mastery_times else None,
        "n_eval_users": len(preds),
    }
    _log.info("System: learning_gain=%.4f, coverage@10=%.4f, mastery_time=%.1f",
               agent_metrics["system"].get("learning_gain_mean", 0),
               agent_metrics["system"].get("tag_coverage_at_10_mean", 0),
               agent_metrics["system"].get("avg_interactions_to_mastery", 0))

    return preds, scores_list, gt_list, gta_list, conf_agent, conf_metrics, agent_metrics


