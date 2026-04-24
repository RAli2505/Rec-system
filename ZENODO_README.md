# MARS — Multi-Agent Recommender System for Personalised Learning

**Anonymised code release accompanying the manuscript "MARS: A Multi-
Agent Recommender System for Personalised Learning."**

This release contains the full implementation, training scripts,
evaluation pipeline, and per-seed result JSON files used to produce
all tables and figures in the manuscript.

## Contents

```
mars-recsys/
├── agents/                      # 6 agents + BaseAgent + Orchestrator
│   ├── base_agent.py
│   ├── diagnostic_agent.py      # 3PL IRT + CAT
│   ├── confidence_agent.py      # Rule-based behavioural confidence
│   ├── kg_agent.py              # Knowledge graph + GraphSAGE + prerequisites
│   ├── prediction_agent.py      # Transformer gap predictor
│   ├── recommendation_agent.py  # Thompson-sampled multi-strategy
│   ├── personalization_agent.py # K-Means ability clustering
│   ├── orchestrator.py          # Assessment / continuous pipelines
│   ├── baselines.py             # Random / Popularity / DKT / SAKT
│   └── utils.py                 # Seed propagation
├── scripts/                     # 50+ scripts (training, eval, figures, stats)
│   ├── run_multi_seed.py
│   ├── run_xes3g5m_full.py
│   ├── run_ednet_comparable.py
│   ├── run_ablation_inference_5seeds.py
│   ├── ablation_significance.py
│   ├── confidence_support_analysis.py
│   ├── posthoc_calibration.py
│   ├── tertile_ablation_s42.py
│   ├── sensitivity_kg_thresholds.py
│   ├── sensitivity_recommendation_weights.py
│   ├── sensitivity_context_window.py
│   ├── statistical_tests.py
│   └── generate_paper_alt_figures.py
├── configs/config.yaml          # Master configuration
├── data/                        # Loaders for XES3G5M and EdNet
├── notebooks/                   # 01-14 exploratory notebooks
├── results/                     # 5-seed main results, ablation, sensitivity
│   ├── aggregated/
│   ├── xes3g5m/
│   └── ednet_comparable/
├── tests/                       # Unit tests per agent
├── requirements.txt             # Exact pinned versions
└── ZENODO_README.md             # This file
```

## Reproduction

Installation (one-time):

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows PowerShell
pip install -r requirements.txt
```

Datasets (not included due to license):

- **XES3G5M.** Download from
  [https://github.com/ai4ed/XES3G5M](https://github.com/ai4ed/XES3G5M)
  and unpack to `data/xes3g5m/XES3G5M`.
- **EdNet KT2.** Download from
  [https://github.com/riiid/ednet](https://github.com/riiid/ednet)
  and unpack to `data/ednet/KT2`.

Main experiments (single command each):

```bash
# 5-seed XES3G5M main pipeline (≈ 4-5 h per seed on RTX 5050)
python scripts/run_multi_seed.py --seeds 42 123 456 789 2024

# 5-seed EdNet cross-dataset
for s in 42 123 456 789 2024; do
  python scripts/run_ednet_comparable.py --seed $s --n_students 6000
done

# 5-seed ablation in inference mode (≈ 75 min total)
python scripts/run_ablation_inference_5seeds.py

# Post-processing
python scripts/ablation_significance.py
python scripts/confidence_support_analysis.py
python scripts/posthoc_calibration.py
python scripts/tertile_ablation_s42.py

# Optional: sensitivity analyses (~6 h total)
python scripts/sensitivity_kg_thresholds.py
python scripts/sensitivity_recommendation_weights.py
python scripts/sensitivity_context_window.py
```

## Environment

- Python 3.12.10
- PyTorch 2.11.0+cu128 (CUDA 12.8)
- NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB VRAM (sufficient; smaller
  GPUs require `batch_size=64` in `configs/config.yaml`)
- 32 GB system RAM recommended

See `requirements.txt` for exact library versions.

## Seeds

Every stochastic component is seeded from a single global seed and
propagated through `agents.utils.set_global_seed(seed)`. The 5 seeds
used throughout the paper are `{42, 123, 456, 789, 2024}`.

## Result bundles

- `results/xes3g5m/xes3g5m_full_s{42,123,456,789,2024}_20260423_*/`
  — full main-pipeline runs (best.pt + metrics.json + history.json)
- `results/xes3g5m/ablation_inference_5seeds_20260424_010555/`
  — 5-seed inference-mode ablation
- `results/xes3g5m/ablation_significance.csv` — paired significance
  tests + BCa bootstrap
- `results/xes3g5m/confidence_support_{xes3g5m,ednet}.json` —
  per-class support for the 6-class confidence taxonomy
- `results/xes3g5m/posthoc_calibration_s42.json` — ECE, Brier,
  ability-tertile NDCG
- `results/xes3g5m/tertile_ablation_s42.json` — per-config tertile
  NDCG for ablation study
- `results/ednet_comparable/ednet_comparable_s*_n6000_min20_*/` —
  5-seed EdNet domain-adapted runs

## License

Code: MIT.
Result JSONs and figures: CC BY 4.0.
Models (best.pt checkpoints): CC BY 4.0, but the XES3G5M and EdNet
datasets themselves are subject to their original licenses
(see dataset pages).

## Citation

Will be filled in upon acceptance. Anonymised reviewers please cite
as `MARS: A Multi-Agent Recommender System for Personalised Learning
(under review, 2026)`.

## Contact

Corresponding author details omitted for double-blind review.
Upon acceptance, replace this section with ORCIDs and email addresses.
