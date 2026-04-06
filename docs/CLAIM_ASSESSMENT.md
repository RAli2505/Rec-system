# MARS: Honest Claim Assessment

Date: 2026-04-05
Status: Pre-rerun assessment (only seed_42 results available after fixes)

---

## 1. Knowledge Graph Agent

### What KG does

- Builds a heterogeneous graph: 14,491 nodes (questions, lectures, tags, parts), 45,896 edges
- Mines PREREQUISITE_OF edges from student mastery sequences (train-only, no leakage)
- Trains GraphSAGE embeddings via link prediction
- Provides Bayesian gap scoring per user (`get_user_gaps`)
- Generates cold-start recommendations via gap → lecture → prerequisite traversal

### How KG contributes downstream

| Integration point | Pipeline | Strength |
|---|---|---|
| `handle_cold_start()` → gap tags + lecture recs | Cold-start | **Primary**: directly generates initial recs |
| `update_user_profile()` → weak tag tracking | Assessment | **Moderate**: tracks per-tag accuracy |
| `kg_profile["gap_tags"]` → candidate generation | Recommendation | **Moderate**: feeds KB strategy + gap-based ranking |
| `kg_profile["theta"]` → ZPD bonus | Recommendation | **Weak**: only used if IRT params set |
| GraphSAGE embeddings | Prediction / Ranking | **Not used**: embeddings are trained and saved but not consumed by PredictionAgent or LambdaMART |

### Honest assessment

**Supportable claim**: KG provides structural knowledge (prerequisite mining, gap analysis, cold-start) that no other agent can deliver. It is the **only source of prerequisite relationships** in the system.

**Unsupportable claim**: "KG significantly improves recommendation accuracy." Current evidence:
- No ablation shows measurable NDCG/MRR delta from disabling KG
- GraphSAGE embeddings are trained but unused downstream
- KB strategy candidates often overlap with CB candidates (unknown dedup rate — diagnostics now added)

### Recommended paper positioning

Position KG as a **structural/interpretability module**, not a performance driver:
- "KG enables prerequisite-aware cold-start recommendations"
- "KG provides interpretable gap analysis grounded in mastery sequences"
- Run ablation (KG disabled) and report honestly: if delta is small, acknowledge it

### What would strengthen the claim

1. Feed GraphSAGE embeddings into PredictionAgent or ranking features
2. Show ablation delta: full MARS vs MARS-without-KG
3. Show cold-start quality improvement specifically attributed to KG prereqs

---

## 2. Confidence Agent

### What ConfidenceAgent does

Rule-based behavioural classifier with 6 classes:
- SOLID (correct + fast + no change)
- UNSURE_CORRECT (correct + slow + no change)
- FALSE_CONFIDENCE (incorrect + fast + no change)
- CLEAR_GAP (incorrect + slow + no change)
- DOUBT_CORRECT (correct + changed answer)
- DOUBT_INCORRECT (incorrect + changed answer)

### Why F1 = 1.0

**This is by design, not leakage.** The labels ARE the rules:
- Classification rules use `is_correct`, `is_fast`, `changed_answer`
- These same features deterministically produce the labels
- The code explicitly removes `is_correct` and `changed_answer` from ML features to avoid trivial leakage
- But since `method = "rule_based"`, the F1 on the rule-generated labels is tautologically 1.0

### Honest assessment

**Supportable claim**: "ConfidenceAgent provides an interpretable behavioural taxonomy that enriches learner state representation."
- The 6-class scheme is pedagogically grounded
- Class distribution is meaningful: 30% SOLID, 23% UNSURE_CORRECT, 12% FALSE_CONFIDENCE, etc.
- Confidence class feeds into PredictionAgent as `conf_class` feature (1 of 14 dimensions)
- Skill deltas per class drive learning gain computation

**Unsupportable claim**: "ConfidenceAgent achieves F1 = 1.0 as a predictive model."
- There is no prediction happening — it's deterministic rule application
- F1 = 1.0 will trigger reviewer suspicion if presented as ML result
- The "CV" evaluation is meaningless for a deterministic rule

### Recommended paper positioning

Position as **interpretable heuristic module**, not as a predictive classifier:
- "Rule-based behavioural confidence taxonomy inspired by [educational psychology refs]"
- "Deterministic classification ensures reproducibility and interpretability"
- Do NOT report F1 = 1.0 in a metrics table alongside ML results
- Instead, report class distribution and downstream impact (ablation: with/without conf_class in LSTM)

### What would strengthen the claim

1. Show ablation: PredictionAgent with vs without `conf_class` feature
2. Show learning gain breakdown by confidence class
3. Compare rule-based 6-class vs ML-based classification (XGBoost on non-leaking features)
   - If ML achieves ~0.55-0.65 macro-F1, that validates the feature space is learnable
   - If ML matches rules closely, rules are a good cheap proxy

---

## 3. PredictionAgent

### Current metrics

- `val_auc = 0.771` (training validation)
- `lstm_auc = 0.675` (evaluation protocol)
- `lstm_f1_micro = 0.139`

### Gap analysis

- val_auc vs eval_auc gap (0.771 → 0.675) suggests distribution shift between validation and evaluation protocol
- Engineering target: AUC >= 0.72-0.78
- Publication target: AUC >= 0.75-0.81 (SOTA on EdNet)
- Current gap to engineering target: ~0.05-0.10

### Honest assessment

PredictionAgent is the most important component for paper claims but currently underperforms targets. Fixes applied (online/offline alignment, temporal features) have not yet been validated via rerun.

---

## 4. Recommendation System

### Current metrics vs baselines

| Metric | MARS | Popularity | BPR | Multiple |
|---|---:|---:|---:|---:|
| NDCG@10 | 0.0834 | 0.0297 | 0.0180 | x2.8 |
| MRR | 0.1217 | 0.0495 | 0.0328 | x2.5 |
| Coverage | 0.0428 | 0.0009 | 0.0398 | ~1x BPR |

### Honest assessment

- MARS clearly outperforms simple baselines — this is a valid claim
- But baselines (Random, Popularity, BPR) are weak for Q1 paper
- Need DKT, SAINT, SASRec or similar as strong baselines
- Coverage is very low (4.28%) — limits "personalized" claim
- Learning gain is negative (-0.012) — cannot claim learning improvement yet

---

## Summary: What can be claimed now vs what needs work

| Claim | Status | Action needed |
|---|---|---|
| Multi-agent architecture | Supportable | Show ablation of 2+ components |
| Better than basic baselines | Supportable | Already demonstrated |
| KG enables cold-start | Supportable | Run cold-start specific evaluation |
| Confidence as interpretable taxonomy | Supportable | Reframe away from "predictive F1" |
| KG improves accuracy | NOT YET | Need ablation evidence |
| Confidence F1 = 1.0 | MISLEADING | Reframe as rule-based |
| Better than SOTA | NOT YET | Need stronger baselines |
| Positive learning gain | NOT YET | Currently negative |
| Statistical significance | NOT YET | Need multi-seed rerun |
