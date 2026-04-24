# Per-tertile NDCG@10 / MRR for each ablation config (seed 42)

Ability proxy = mean correctness on the first 30% of each user's test sequence.
Tertile edges from the XES3G5M test set (equal-frequency, 33/66 percentiles):
- **low** ≤ 0.716
- **mid** ∈ (0.716, 0.846]
- **high** > 0.846

## NDCG@10 by ability tertile

| Config | Low (n) | Mid (n) | High (n) | Overall |
|---|---:|---:|---:|---:|
| Full MARS | 0.673±0.206 (n=300) | 0.659±0.219 (n=300) | 0.651±0.238 (n=300) | 0.661 |
| - Prediction | 0.618±0.171 (n=300) | 0.628±0.169 (n=300) | 0.631±0.192 (n=300) | 0.626 |
| - Knowledge Graph | 0.673±0.206 (n=300) | 0.659±0.219 (n=300) | 0.651±0.238 (n=300) | 0.661 |
| - Confidence | 0.665±0.215 (n=300) | 0.652±0.233 (n=300) | 0.642±0.247 (n=300) | 0.653 |
| - IRT (Diagnostic) | 0.730±0.222 (n=300) | 0.713±0.232 (n=300) | 0.718±0.240 (n=300) | 0.720 |

## MRR by ability tertile

| Config | Low | Mid | High | Overall |
|---|---:|---:|---:|---:|
| Full MARS | 0.885±0.240 | 0.865±0.256 | 0.840±0.275 | 0.863 |
| - Prediction | 0.801±0.284 | 0.807±0.272 | 0.781±0.284 | 0.796 |
| - Knowledge Graph | 0.885±0.240 | 0.865±0.256 | 0.840±0.275 | 0.863 |
| - Confidence | 0.883±0.251 | 0.850±0.268 | 0.831±0.285 | 0.855 |
| - IRT (Diagnostic) | 0.918±0.208 | 0.901±0.222 | 0.904±0.223 | 0.908 |

## Δ NDCG@10 within each tertile (Full MARS − ablated)

Positive Δ = ablated variant hurts vs Full. 
Negative Δ = ablated variant *improves* NDCG@10 in that tertile — the 
subgroup evidence of the IRT coverage/accuracy trade-off reported in §4.5.

| Ablation | Δ Low | Δ Mid | Δ High |
|---|---:|---:|---:|
| - Prediction | +0.055 | +0.031 | +0.020 |
| - Knowledge Graph | +0.000 | +0.000 | +0.000 |
| - Confidence | +0.008 | +0.007 | +0.008 |
| - IRT (Diagnostic) | -0.057 | -0.054 | -0.067 |