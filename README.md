# MARS: Multi-Agent Recommender System for Personalized Learning

A multi-agent recommender system that combines IRT-based diagnostics, knowledge graph reasoning, transformer-based prediction, and adaptive recommendation strategies to deliver personalized learning paths.

## Datasets

- **EdNet KT2** -- TOEIC English proficiency. Raw corpus contains 784K students and 131M+ interactions; for the cross-dataset comparison in this paper the pipeline is retrained on a 6,000-student resample (min 20 interactions, user-level 70/15/15 split) so that the evaluation protocol matches XES3G5M one-to-one. Download: [github.com/riiid/ednet](https://github.com/riiid/ednet)
- **XES3G5M** -- Math knowledge tracing. 6,000-student resample, 858 concepts, 7,373 questions, same 70/15/15 user-level split. Download: [github.com/ai4ed/XES3G5M](https://github.com/ai4ed/XES3G5M?tab=readme-ov-file)

## Architecture

7 specialized ML agents + Orchestrator with 3 context-aware pipelines.

| Agent | Model | Purpose |
|-------|-------|---------|
| DiagnosticAgent | IRT 3PL + CAT | Ability estimation (theta) |
| ConfidenceAgent | Rule-based 6-class | Behavioral confidence classification |
| KGAgent | GraphSAGE + DAG | Knowledge graph embeddings + prerequisite mining |
| PredictionAgent | SAINT Transformer 4L/256d | Knowledge gap prediction (dataset-dependent gap vector — 858 concepts on XES3G5M, 293 on EdNet KT2) |
| RecommendationAgent | Thompson Sampling + Weighted Linear | Multi-strategy recommendation + MMR diversity |
| PersonalizationAgent | IRT rule-based 5-level | Learner stratification |
| Orchestrator | Rule-based | Pipeline coordination |

### Pipelines

| Pipeline | Trigger | Agents |
|----------|---------|--------|
| Cold-Start | New user | Diagnostic -> KG -> Recommend (explore mode) |
| Assessment | After responses | Diagnostic -> Confidence -> KG -> Prediction -> Personalization -> Recommend |
| Continuous | Ongoing | Prediction (update) -> Recommend (re-rank) |

## Project Structure

```
ednet-mars/
|-- agents/                     # Multi-agent system
|   |-- orchestrator.py         # Pipeline orchestrator (Cold-Start / Assessment / Continuous)
|   |-- diagnostic_agent.py     # IRT 3PL + CAT, theta ability estimation
|   |-- confidence_agent.py     # Rule-based 6-class behavioral confidence
|   |-- kg_agent.py             # GraphSAGE embeddings + prerequisite mining
|   |-- prediction_agent.py     # SAINT Transformer 4L/256d, dataset-dependent gap vector (858 on XES3G5M, 293 on EdNet)
|   |-- recommendation_agent.py # Thompson Sampling + weighted linear, MMR diversity
|   |-- personalization_agent.py# IRT rule-based 5-level learner stratification
|   |-- baselines.py            # Random, Popularity, DKT, GRU baselines
|   |-- base_agent.py           # Base agent class
|   `-- utils.py
|
|-- data/                       # Data loading and preprocessing
|   |-- loader.py               # EdNet data loader
|   |-- xes3g5m_loader.py       # XES3G5M data loader
|   `-- preprocessor.py         # Feature engineering + confidence synthesis
|
|-- models/                     # Trained model weights
|   |-- gap_transformer*.pt     # SAINT transformer checkpoints
|   |-- graphsage.pt            # GraphSAGE knowledge graph model
|   |-- irt_params.npz          # IRT parameters
|   |-- confidence_xgb*.json    # Confidence classifier configs
|   `-- tag_embeddings.npy      # Tag embeddings
|
|-- configs/
|   `-- config.yaml             # All hyperparameters
|
|-- scripts/                    # Experiment runners and utilities
|   |-- run_xes3g5m_full.py     # Full XES3G5M pipeline (with confidence synthesis)
|   |-- run_xes3g5m.py          # XES3G5M pipeline (without synthesis)
|   |-- run_xes3g5m_baselines.py# Baseline comparison
|   |-- run_xes3g5m_ablation.py # Ablation study
|   |-- run_multi_seed.py       # Multi-seed evaluation (5 seeds)
|   |-- generate_ieee_figures.py# IEEE-format figures
|   |-- generate_xes3g5m_figures.py # XES3G5M comparison figures
|   |-- generate_architecture_figs.py # Architecture diagrams
|   |-- statistical_tests.py    # Statistical significance tests
|   `-- subgroup_analysis.py    # Subgroup performance analysis
|
|-- results/xes3g5m/            # Experiment results
|   |-- figures/                # Generated figures (PNG + PDF)
|   |-- tables/                 # Result tables (CSV)
|   |-- xes3g5m_full_s*/        # Per-seed run results
|   |-- baselines_s*/           # Baseline results
|   `-- ablation_s*/            # Ablation results
|
|-- diagrams/                   # Mermaid architecture diagrams
|-- notebooks/                  # Jupyter notebooks
|-- docs/                       # Documentation
|-- test_*.py                   # Unit tests per agent
`-- utils/                      # Plotting utilities
```

## Setup

```bash
# Python 3.12.x recommended (tested on 3.12.2, Windows 11 + CUDA 12.8)
python -m pip install --upgrade pip

# PyTorch — install with the official CUDA wheel index BEFORE the rest:
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
            torchaudio==2.11.0+cu128 \
            --index-url https://download.pytorch.org/whl/cu128

# Then the rest of the pinned environment:
pip install -r requirements.txt
```

## Datasets

```bash
# XES3G5M (~360 MB tar.gz, contains kc_level/, question_level/, metadata/)
mkdir -p data/xes3g5m
# Download manually from https://github.com/ai4ed/XES3G5M and extract:
tar -xzf XES3G5M.tar.gz -C data/xes3g5m/
# Result: data/xes3g5m/XES3G5M/kc_level/train_valid_sequences.csv

# EdNet KT2 — download the raw KT2 tarball from github.com/riiid/ednet
# and extract to data/raw/KT2/. The loader
# (data/ednet_comparable_loader.py) then samples 6,000 students on
# the fly to match the XES3G5M protocol.
mkdir -p data/raw
# wget https://raw.githubusercontent.com/riiid/ednet/master/... # see
# EdNet README for the current download URL; the KT2 tarball is ~10 GB.
```

## Reproducing the paper results

### Full reproduction pipeline (~9 hours wall-clock on a single GPU)

```bash
# 1. Trains MARS on 5 random seeds, then runs baselines and the
#    component ablation. Logs go to logs/retrain_NUMTAGS858_<ts>/.
bash scripts/retrain_xes3g5m_pipeline.sh

# 2. Aggregates raw metrics into the 3 paper tables and regenerates
#    every downstream figure / LaTeX table. Run once the pipeline has
#    finished (the sentinel chain triggers it automatically — see
#    scripts/watch_and_postprocess.sh).
bash scripts/postprocess_xes3g5m.sh
```

### Individual components

```bash
# A single seed of the full multi-agent pipeline
python scripts/run_xes3g5m_full.py --seed 42

# 4 baselines (Random / Popularity / DKT-LSTM / GRU)
python scripts/run_xes3g5m_baselines.py --seed 42

# 3 extra baselines (BPR-MF / CF-only / Content-only)
python scripts/run_extra_baselines.py --seed 42

# Component ablation (4 configs after Full MARS reuse)
python scripts/run_xes3g5m_ablation.py --seed 42

# Subgroup analysis on saved best.pt (low/mid/high tertiles)
python scripts/subgroup_xes3g5m.py
```

### Reproducing every paper figure / table from the saved JSONs

```bash
# 1. Build the 3 source CSVs:
python scripts/aggregate_xes3g5m.py

# 2. Regenerate every downstream artefact:
python scripts/generate_paper_alt_figures.py   # Figs 3 (heatmap), 4 (ablation), CD diagram
python scripts/generate_radar.py               # Fig 8
python scripts/generate_seed_table.py          # Table 5
python scripts/generate_table3_no_r10.py       # Table 3 (LaTeX + Markdown)
python scripts/generate_cross_dataset.py       # Fig 9 (cross-dataset bars)
python scripts/generate_training_curves.py     # Figs 6 + 7 (loss + AUC)
python scripts/ablation_pareto_analysis.py     # Pareto-frontier figure for §4.5
python scripts/subgroup_xes3g5m.py             # Fig 5 (subgroup bars)
```

## Random seeds

All five random seeds used in the paper are baked into the orchestrator
script: **42, 123, 456, 789, 2024**. They control:

- the user-level 70/15/15 split inside `data/xes3g5m_loader.py`;
- the PyTorch / NumPy initialisation through `agents/utils.set_global_seed()`;
- the Thompson Sampling Beta-priors inside the Recommendation Agent;
- the negative-sampling order in the BPR baseline.

The same seeds are used on EdNet (`scripts/run_ednet_comparable.py`)
to keep the 5-seed stability tables comparable across datasets.

## Concept-space (NUM_TAGS) configuration

The original codebase hard-coded `NUM_TAGS = 293` (EdNet TOEIC) in
`agents/prediction_agent.py`. For XES3G5M (858 concepts) this silently
clipped concept IDs above 293 and degraded coverage. The current
release derives `NUM_TAGS` dynamically from the training split:

```python
train_max_id = max(int(t) for tags in train_df["tags"]
                   for t in tags if tags)
n_tags = train_max_id + 1
set_num_tags(n_tags)   # → 858 for XES3G5M, 293 for EdNet
```

This is performed automatically by `run_xes3g5m_*.py` and
`run_ednet_comparable.py`; manual invocation is not needed.

## Configuration

All hyperparameters are centralised in `configs/config.yaml`. Agents
load their section automatically via `BaseAgent._load_config()`. Any
override is picked up at agent construction time.

## Hardware used in the paper

- GPU: NVIDIA GeForce RTX 5050 Laptop (8 GB), driver 573.13, CUDA 12.8
- CPU: Intel-class laptop, 32 GB RAM
- OS: Windows 11 Pro 24H2, Git Bash 2.49 + PowerShell 5.1
- Python: 3.12.2 (Microsoft Store distribution)

Single-seed wall-clock: MARS pipeline ≈ 45–60 min; baselines ≈ 90 min
(includes DKT and GRU training); ablation ≈ 3 hours (4 configs after
reusing Full MARS metrics from the main pipeline run).

## Citation

```bibtex
@article{ali2026mars,
  title  = {A Multi-Agent Recommendation System for Personalized Learning
            with Behavioral Confidence Modeling and Data-Driven Prerequisite Mining},
  author = {Ali, Ramazan and Stelvaga, Oleg},
  year   = {2026},
  journal= {Submitted}
}
```
