# Attention-based KT comparison (5-seed mean ± std)

All baselines (SAINT, AKT, SimpleKT, DTransformer) trained from scratch on XES3G5M with the same 14-dim per-step input as the MARS Prediction Agent, so the comparison isolates the attention mechanism and decoder choice from feature engineering. Metrics computed against the same multi-label failure-prediction task as MARS (Table 4 in the manuscript).

| Model | Params (M) | AUC | NDCG@10 | MRR | P@10 | Tag Coverage |
|---|---:|---:|---:|---:|---:|---:|
| NCF | 0.14 | 0.789±0.003 | 0.150±0.003 | 0.178±0.004 | 0.071±0.002 | 0.321±0.011 |
| SAINT | 4.45 | 0.956±0.003 | 0.545±0.002 | 0.598±0.003 | 0.227±0.001 | 0.747±0.040 |
| AKT | 3.44 | 0.963±0.001 | 0.546±0.010 | 0.600±0.008 | 0.227±0.003 | 0.779±0.028 |
| SimpleKT | 3.44 | 0.957±0.001 | 0.564±0.004 | 0.617±0.004 | 0.233±0.001 | 0.818±0.038 |
| DTransformer | 3.77 | 0.964±0.001 | 0.559±0.008 | 0.612±0.008 | 0.232±0.003 | 0.752±0.035 |
| SASRec | 0.60 | 0.961±0.003 | 0.543±0.010 | 0.602±0.008 | 0.226±0.004 | 0.623±0.019 |
| MARS (Full) | 2.84 | 0.923±0.005 | 0.683±0.023 | 0.893±0.029 | 0.640±0.020 | 0.736±0.024 |

Reading. Attention baselines lead on AUC by 3–4 percentage points, MARS dominates the ranking metrics (NDCG@10, MRR, P@10) by 12–41 percentage points. The dissociation supports the paper's claim that MARS's value comes from the multi-agent ranking pipeline, not from the choice of attention backbone.