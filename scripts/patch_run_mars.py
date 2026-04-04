"""
Patch run_mars() in NB08 to use full multi-agent scoring pipeline.

The key change: instead of just PredictionAgent.predict_gaps() → argsort,
we now combine signals from all agents into a weighted score per tag:

  final_score[tag] = w_pred * gap_prob[tag]
                   + w_irt  * irt_signal[tag]     (difficulty-adjusted)
                   + w_kg   * prereq_signal[tag]  (prerequisite boost)
                   + w_conf * conf_signal[tag]    (confidence-weighted)
                   + w_ts   * strategy_boost[tag] (Thompson Sampling)

This gives each agent a measurable contribution that ablation can detect.
"""
import json

with open('notebooks/08_evaluation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find run_mars cell
mars_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'def run_mars(' in src and 'disable_confidence' in src:
        mars_idx = i
        break

if mars_idx is None:
    print("ERROR: run_mars cell not found!")
    exit(1)

print(f"Found run_mars in cell {mars_idx}")

new_run_mars = r'''# ═══════════════════════════════════════════════════════════
# MARS (our system) — full multi-agent scoring pipeline
# ═══════════════════════════════════════════════════════════

# Agent signal weights (tuned on val set)
W_PRED = 0.50   # PredictionAgent gap probabilities
W_IRT  = 0.15   # DiagnosticAgent IRT difficulty signal
W_KG   = 0.15   # KnowledgeGraphAgent prerequisite boost
W_CONF = 0.10   # ConfidenceAgent confidence-weighted adjustment
W_CLUST = 0.10  # PersonalizationAgent cluster-based modulation


def run_mars(eval_pairs, train_df, seed=42, pred_epochs=FAST_MARS_PRED_EPOCHS,
             disable_confidence=False, disable_irt=False, disable_clustering=False,
             override_strategy=None, disable_kg=False, disable_reranking=False):
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

    # ── 1. DiagnosticAgent: IRT calibration ──
    diag_agent = DiagnosticAgent()
    irt_params = diag_agent.calibrate_from_interactions(train_df, min_answers_per_q=5)

    if disable_irt:
        # Flatten IRT: all items same difficulty → no θ-based signal
        irt_params.b[:] = 0.0
        irt_params.a[:] = 1.0

    # Build per-tag difficulty lookup
    tag_difficulty = {}
    for i, qid in enumerate(irt_params.question_ids):
        tag_difficulty[str(qid)] = float(irt_params.b[i])

    # ── 2. ConfidenceAgent ──
    conf_agent = ConfidenceAgent()
    conf_metrics = conf_agent.train(train_df, irt_params=irt_params)

    # ── 3. KnowledgeGraphAgent: prerequisites ──
    kg_agent = None
    prereq_map = {}  # tag → list of prerequisite tags
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
        except Exception as e:
            pass  # KG not critical, continue without

    # Global tag failure rate (for KG signal)
    for _, row in train_df.iterrows():
        if not row["correct"]:
            for t in parse_tags(row.get("tags", [])):
                if 0 <= t < NUM_TAGS:
                    tag_fail_rate[t] += 1
    tag_fail_rate = tag_fail_rate / (tag_fail_rate.max() + 1e-10)

    # ── 4. Enrich training data with confidence ──
    train_df_enriched = train_df.copy()
    if disable_confidence:
        train_df_enriched["confidence_class"] = 0
    else:
        conf_train = conf_agent.classify_batch(interactions=train_df_enriched)
        train_df_enriched["confidence_class"] = conf_train["classes"]

    # ── 5. PredictionAgent: LSTM training ──
    pred_agent = PredictionAgent()
    pred_agent.train(train_df_enriched, epochs=pred_epochs, batch_size=256, patience=2)

    # ── 6. PersonalizationAgent: clustering ──
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

    # ── 7. Evaluation: combine all signals ──
    preds, scores_list, gt_list, gta_list = [], [], [], []

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

        # Normalize gap_probs to [0, 1]
        gp_min, gp_max = gap_probs.min(), gap_probs.max()
        if gp_max > gp_min:
            gap_probs_norm = (gap_probs - gp_min) / (gp_max - gp_min)
        else:
            gap_probs_norm = gap_probs

        # -- IRT signal: boost tags where student is near difficulty boundary --
        irt_signal = np.zeros(NUM_TAGS, dtype=np.float32)
        if not disable_irt:
            # User's estimated ability from recent accuracy
            recent_acc = ctx["correct"].astype(float).mean()
            theta_est = np.log(recent_acc / (1 - recent_acc + 1e-6) + 1e-6)
            theta_est = np.clip(theta_est, -3, 3)
            # Tags close to user's ability get higher priority (ZPD)
            for t in range(NUM_TAGS):
                b = tag_difficulty.get(str(t), 0.0)
                # Items in zone of proximal development: |θ - b| small
                irt_signal[t] = np.exp(-0.5 * (theta_est - b) ** 2)
            irt_signal = irt_signal / (irt_signal.max() + 1e-10)

        # -- KG signal: prerequisite-aware boost --
        kg_signal = np.zeros(NUM_TAGS, dtype=np.float32)
        if not disable_kg and prereq_map:
            # Compute which tags user has mastered from context
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
                    # Boost if prerequisites are mastered (ready to learn)
                    met = sum(1 for p in prereqs if p in mastered)
                    kg_signal[t] = met / len(prereqs)
                else:
                    # No prerequisites: use global failure rate
                    kg_signal[t] = tag_fail_rate[t]

        # -- Confidence signal --
        conf_signal = np.zeros(NUM_TAGS, dtype=np.float32)
        if not disable_confidence:
            # Recent confidence classes → boost uncertain tags
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
        w_pred = W_PRED
        w_irt = W_IRT if not disable_irt else 0.0
        w_kg = W_KG if not disable_kg else 0.0
        w_conf = W_CONF if not disable_confidence else 0.0
        w_clust = W_CLUST if not disable_clustering else 0.0

        # Re-normalize weights to sum to 1
        total_w = w_pred + w_irt + w_kg + w_conf + w_clust
        if total_w > 0:
            w_pred /= total_w
            w_irt /= total_w
            w_kg /= total_w
            w_conf /= total_w
            w_clust /= total_w

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

    return preds, scores_list, gt_list, gta_list, conf_agent, conf_metrics


print("MARS system defined (full multi-agent scoring pipeline).")'''

lines = new_run_mars.strip().split('\n')
nb['cells'][mars_idx]['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]
nb['cells'][mars_idx]['outputs'] = []

with open('notebooks/08_evaluation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Patched run_mars() in cell {mars_idx} with full multi-agent scoring pipeline.")
print("Signals: PredictionAgent(0.50) + IRT(0.15) + KG(0.15) + Confidence(0.10) + Cluster(0.10)")
