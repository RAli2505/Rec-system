# Ablation significance (Full MARS vs -Component, N=5 seeds)

Paired tests on 5 seed-level observations. BCa bootstrap 95% CI on mean delta, 10k resamples.

**Important caveat:** the Wilcoxon signed-rank test with N=5 paired observations has a hard minimum two-sided p of 2^-4 = 0.0625, so Wilcoxon can never reach p<0.05 here. Primary evidence is therefore (a) whether the 95% BCa CI on the mean delta excludes zero (column `CI*`), and (b) the paired t-test column for large-magnitude effects. This is also why the reviewer asked for a per-user test (N=899), which is left as follow-up work pending a re-run with per-user NDCG persisted.

Markers: `CI*` = BCa 95% CI excludes 0, `*`/`**`/`***` = parametric p<0.05 / 0.01 / 0.001, `ns` otherwise.

## ndcg@10

| Config | Full mean±std | Ablated mean±std | Δ=Full-Abl | 95% BCa CI | Wilcoxon p | t-test p | CI* | t-sig |
|---|---:|---:|---:|:---:|---:|---:|:---:|:---:|
| - Prediction | 0.6828 ± 0.0234 | 0.5847 ± 0.0659 | +0.0981 ± 0.0687 | [+0.0502, +0.1598] | 0.0625 | 0.0332 | CI* | * |
| - Knowledge Graph | 0.6828 ± 0.0234 | 0.6886 ± 0.0189 | -0.0059 ± 0.0061 | [-0.0100, -0.0002] | 0.1875 | 0.0999 | CI* | ns |
| - Confidence | 0.6828 ± 0.0234 | 0.6829 ± 0.0232 | -0.0001 ± 0.0004 | [-0.0004, +0.0003] | 0.8125 | 0.7040 | ns | ns |
| - IRT (Diagnostic) | 0.6828 ± 0.0234 | 0.7408 ± 0.0152 | -0.0580 ± 0.0122 | [-0.0670, -0.0476] | 0.0625 | 0.0004 | CI* | *** |

## mrr

| Config | Full mean±std | Ablated mean±std | Δ=Full-Abl | 95% BCa CI | Wilcoxon p | t-test p | CI* | t-sig |
|---|---:|---:|---:|:---:|---:|---:|:---:|:---:|
| - Prediction | 0.8929 ± 0.0291 | 0.7428 ± 0.1500 | +0.1501 ± 0.1502 | [+0.0749, +0.2920] | 0.0625 | 0.0892 | CI* | ns |
| - Knowledge Graph | 0.8929 ± 0.0291 | 0.8989 ± 0.0238 | -0.0060 ± 0.0077 | [-0.0105, +0.0023] | 0.1875 | 0.1568 | ns | ns |
| - Confidence | 0.8929 ± 0.0291 | 0.8929 ± 0.0291 | +0.0000 ± 0.0000 | [+0.0000, +0.0000] | 1.0000 | nan | ns | n/a |
| - IRT (Diagnostic) | 0.8929 ± 0.0291 | 0.9213 ± 0.0093 | -0.0284 ± 0.0201 | [-0.0456, -0.0141] | 0.0625 | 0.0342 | CI* | * |

## precision@10

| Config | Full mean±std | Ablated mean±std | Δ=Full-Abl | 95% BCa CI | Wilcoxon p | t-test p | CI* | t-sig |
|---|---:|---:|---:|:---:|---:|---:|:---:|:---:|
| - Prediction | 0.6399 ± 0.0198 | 0.5849 ± 0.0406 | +0.0550 ± 0.0443 | [+0.0196, +0.0877] | 0.1250 | 0.0502 | CI* | ns |
| - Knowledge Graph | 0.6399 ± 0.0198 | 0.6458 ± 0.0167 | -0.0059 ± 0.0052 | [-0.0107, -0.0021] | 0.0625 | 0.0639 | CI* | ns |
| - Confidence | 0.6399 ± 0.0198 | 0.6400 ± 0.0195 | -0.0001 ± 0.0007 | [-0.0006, +0.0005] | 0.7500 | 0.7267 | ns | ns |
| - IRT (Diagnostic) | 0.6399 ± 0.0198 | 0.7077 ± 0.0164 | -0.0678 ± 0.0133 | [-0.0822, -0.0602] | 0.0625 | 0.0003 | CI* | *** |

## recall@10

| Config | Full mean±std | Ablated mean±std | Δ=Full-Abl | 95% BCa CI | Wilcoxon p | t-test p | CI* | t-sig |
|---|---:|---:|---:|:---:|---:|---:|:---:|:---:|
| - Prediction | 0.0604 ± 0.0013 | 0.0783 ± 0.0023 | -0.0179 ± 0.0030 | [-0.0199, -0.0150] | 0.0625 | 0.0002 | CI* | *** |
| - Knowledge Graph | 0.0604 ± 0.0013 | 0.0616 ± 0.0013 | -0.0012 ± 0.0008 | [-0.0019, -0.0007] | 0.0625 | 0.0305 | CI* | * |
| - Confidence | 0.0604 ± 0.0013 | 0.0604 ± 0.0012 | +0.0000 ± 0.0001 | [-0.0001, +0.0001] | 1.0000 | 1.0000 | ns | ns |
| - IRT (Diagnostic) | 0.0604 ± 0.0013 | 0.0607 ± 0.0012 | -0.0003 ± 0.0009 | [-0.0011, +0.0003] | 0.8125 | 0.4414 | ns | ns |

## lstm_auc

| Config | Full mean±std | Ablated mean±std | Δ=Full-Abl | 95% BCa CI | Wilcoxon p | t-test p | CI* | t-sig |
|---|---:|---:|---:|:---:|---:|---:|:---:|:---:|
| - Prediction | 0.9235 ± 0.0052 | 0.4980 ± 0.0092 | +0.4254 ± 0.0132 | [+0.4143, +0.4350] | 0.0625 | 0.0000 | CI* | *** |
| - Knowledge Graph | 0.9235 ± 0.0052 | 0.9265 ± 0.0051 | -0.0031 ± 0.0012 | [-0.0044, -0.0023] | 0.0625 | 0.0053 | CI* | ** |
| - Confidence | 0.9235 ± 0.0052 | 0.9235 ± 0.0052 | +0.0000 ± 0.0000 | [+0.0000, +0.0000] | 1.0000 | nan | ns | n/a |
| - IRT (Diagnostic) | 0.9235 ± 0.0052 | 0.9265 ± 0.0051 | -0.0031 ± 0.0012 | [-0.0044, -0.0023] | 0.0625 | 0.0053 | CI* | ** |

## tag_coverage

| Config | Full mean±std | Ablated mean±std | Δ=Full-Abl | 95% BCa CI | Wilcoxon p | t-test p | CI* | t-sig |
|---|---:|---:|---:|:---:|---:|---:|:---:|:---:|
| - Prediction | 0.7357 ± 0.0244 | 0.0526 ± 0.0101 | +0.6831 ± 0.0335 | [+0.6626, +0.7151] | 0.0625 | 0.0000 | CI* | *** |
| - Knowledge Graph | 0.7357 ± 0.0244 | 0.7285 ± 0.0304 | +0.0073 ± 0.0119 | [-0.0013, +0.0158] | 0.3125 | 0.2453 | ns | ns |
| - Confidence | 0.7357 ± 0.0244 | 0.7335 ± 0.0249 | +0.0023 ± 0.0040 | [-0.0022, +0.0045] | 0.5000 | 0.2723 | ns | ns |
| - IRT (Diagnostic) | 0.7357 ± 0.0244 | 0.3339 ± 0.0336 | +0.4018 ± 0.0422 | [+0.3753, +0.4409] | 0.0625 | 0.0000 | CI* | *** |
