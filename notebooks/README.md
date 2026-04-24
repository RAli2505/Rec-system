# Exploratory notebooks

These notebooks were used for early-stage analysis, EDA, and
per-agent prototyping (Apr 2026). **They are not the canonical
reproduction path for the paper.**

The canonical pipeline is in `../scripts/`, specifically:

| Purpose | Canonical script |
|---|---|
| Full 5-seed MARS training | `scripts/run_multi_seed.py` |
| XES3G5M single-seed full pipeline | `scripts/run_xes3g5m_full.py` |
| EdNet domain-adapted | `scripts/run_ednet_comparable.py` |
| 5-seed inference-mode ablation | `scripts/run_ablation_inference_5seeds.py` |
| Per-seed tertile analysis | `scripts/tertile_ablation_s42.py` |
| Paired significance + BCa | `scripts/ablation_significance.py` |
| Confidence support + confusion matrix | `scripts/confidence_support_analysis.py` |
| ECE + Brier + tertile calibration | `scripts/posthoc_calibration.py` |
| Subgroup analysis | `scripts/subgroup_xes3g5m.py` |
| Sensitivity (prereq / rec / context) | `scripts/sensitivity_*.py` |
| Paper figures | `scripts/generate_paper_alt_figures.py`, `scripts/generate_ieee_figures.py`, `scripts/generate_*_figures.py` |

## Contents of this directory (exploratory only)

| # | Notebook | Topic | Superseded by |
|---|---|---|---|
| 02 | `02_knowledge_graph.ipynb`          | Graph build, GraphSAGE | `agents/kg_agent.py` + `scripts/run_xes3g5m_full.py` |
| 03 | `03_irt_calibration.ipynb`         | 3PL IRT EM              | `agents/diagnostic_agent.py` |
| 04 | `04_confidence.ipynb`              | Rule-based 6-class       | `agents/confidence_agent.py` + `scripts/confidence_support_analysis.py` |
| 05 | `05_recommendations.ipynb`         | Multi-strategy Thompson  | `agents/recommendation_agent.py` |
| 06 | `06_lstm.ipynb`                    | Original LSTM prototype  | `agents/prediction_agent.py` (transformer) |
| 07 | `07_clustering.ipynb`              | K-Means ability levels   | `agents/personalization_agent.py` |
| 08 | `08_evaluation.ipynb`              | Offline eval walkthrough | `agents/orchestrator.py::batch_evaluation` |
| 09 | `09_figures.ipynb`                 | Prototype figures        | `scripts/generate_*.py` |
| 10 | `10_hyperparameter_search.ipynb`   | Manual hyperparam sweeps | `configs/config.yaml` + sensitivity scripts |
| 11 | `11_encoder_comparison.ipynb`      | LSTM / GRU / Transformer | `scripts/run_xes3g5m_baselines.py` |
| 12 | `12_confidence_ablation.ipynb`     | Binary / 4 / 6-class     | `scripts/run_xes3g5m_ablation.py` |
| 13 | `13_ablation_study.ipynb`          | Component ablations      | `scripts/run_ablation_inference_5seeds.py` |
| 14 | `14_baselines.ipynb`               | BPR / NeuMF / KG-RS      | `scripts/run_extra_baselines.py` |
| 15a| `15a_bandit_comparison.ipynb`      | Thompson vs UCB / ε-greedy | deferred |
| 15b| `15b_clustering_comparison.ipynb`  | K vs GMM vs spectral     | deferred |
| 15c| `15c_prerequisite_validation.ipynb`| Prerequisite-mining sanity | `scripts/sensitivity_kg_thresholds.py` |
| 16 | `16_case_studies.ipynb`            | Per-user recommendation traces | illustrative only |
| 17 | `17_computational_cost.ipynb`      | Timing / memory profiles | Appendix S5 (auto-generated) |

## Why they are kept

Notebooks show the exploratory path behind each design decision.
They were frozen in early April 2026 and not updated for the
5-seed / 900-user / 858-concept final pipeline. Numbers inside
may disagree with the paper — always trust `results/` JSON over
notebook outputs.
