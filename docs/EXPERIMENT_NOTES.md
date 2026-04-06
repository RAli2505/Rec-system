# MARS Experiment Notes: Before/After Comparison

## Purpose

Track metric deltas from code fixes applied in April 2026.
All "before" metrics are from seed_42 single run (pre-fix).
"After" metrics will come from multi-seed rerun (post-fix).

---

## Before (seed_42, pre-fix baseline)

Source: `results/seed_42/eval_metrics.json` (Apr 5 2026)

| Metric | Value |
|---|---:|
| lstm_auc | 0.6748 |
| lstm_auc_weighted | 0.6829 |
| lstm_f1_micro | 0.1387 |
| lstm_f1_macro | 0.0721 |
| ndcg@10 | 0.0834 |
| mrr | 0.1217 |
| precision@10 | 0.0780 |
| recall@10 | 0.0680 |
| coverage | 0.0428 |
| learning_gain | -0.0123 |
| learning_gain_trimmed | -0.0136 |
| n_users_evaluated | 14,577 |

Agent metrics (from `agent_metrics.json`):

| Agent | Key metric | Value |
|---|---|---:|
| Diagnostic | r_theta_accuracy | 0.8079 |
| Confidence | full_f1_macro | 1.0 (rule-based) |
| KG | n_nodes / n_edges | 14,491 / 45,896 |
| Prediction | val_auc | 0.7710 |
| Prediction | val_f1_at_threshold | 0.2342 |
| Personalization | n_levels | 5 |

---

## Fixes applied since baseline

1. **Seed propagation** — `BaseAgent` now exposes `global_seed`; PredictionAgent and KGAgent use it
2. **PredictionAgent online/offline alignment** — model now consumes all 14 features; `_build_sequence()` and `update_state()` match training layout
3. **RecommendationAgent signal integration** — orchestrator calls `update_user_profile()`; prediction gap probs injected into candidate scores; ranking rebalanced
4. **RecommendationAgent internal seed** — replaced hardcoded `42` with run seed
5. **Diagnostic logging** — candidate counts by strategy, dedup overlap, pre/post-MMR diversity
6. **Ablation switches** — config-driven toggles for prediction boost, MMR, ZPD, learner level

---

## After (multi-seed rerun — TO BE FILLED)

Seeds: [42, 123, 456, 789, 2024]

| Metric | Mean | Std | Delta vs Before |
|---|---:|---:|---:|
| lstm_auc | | | |
| lstm_f1_micro | | | |
| ndcg@10 | | | |
| mrr | | | |
| coverage | | | |
| learning_gain | | | |

### Variance check

| Metric | Std across seeds | Non-zero? |
|---|---:|---|
| lstm_auc | | |
| ndcg@10 | | |
| mrr | | |
| coverage | | |

---

## Expected outcomes

### Seed variance
- Before fix: seeds had near-zero variance (fallback to 42)
- After fix: expect non-zero std, especially for:
  - lstm_auc (model training randomness)
  - ndcg@10 (recommendation strategy sampling)
  - coverage (MMR + strategy selection)

### Prediction quality
- Fixes: temporal features now reach model; online/offline consistent
- Expectation: lstm_auc should improve (0.675 → ?)
  - Optimistic: 0.70-0.75
  - Conservative: 0.68-0.70
  - If no improvement: features may not carry enough signal

### Recommendation quality
- Fixes: prediction signal injected; profile updated before ranking
- Expectation: ndcg@10 should improve (0.083 → ?)
  - Depends on prediction quality improvement
  - Direct prediction boost may help even without better lstm_auc

### Coverage
- Now: 0.0428
- MMR with λ=0.80 should maintain or slightly improve
- If ablation shows MMR improves coverage without hurting ndcg, that's a valid paper finding

---

## Ablation experiment plan

Run after multi-seed baseline is established:

| Config | What's disabled | Expected impact |
|---|---|---|
| MARS (full) | Nothing | Baseline |
| No prediction boost | `use_prediction_boost: false` | NDCG may drop |
| No MMR | `use_mmr: false` | Coverage drops, NDCG may rise slightly |
| No ZPD | `use_zpd_bonus: false` | Small effect expected |
| No learner level | `use_learner_level: false` | Small effect expected |
| No KG | Disable KB strategy | Shows KG contribution |
| No confidence | Remove conf_class from LSTM | Shows confidence contribution |

---

## Statistical testing plan

- Paired t-test or Wilcoxon signed-rank (N=5 seeds)
- Bonferroni correction for multiple metrics
- Cohen's d for effect size
- Infrastructure already exists: `scripts/statistical_tests.py`
