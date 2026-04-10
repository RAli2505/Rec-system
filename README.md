# MARS: Multi-Agent Recommender System for Personalized Learning

A multi-agent recommender system that combines IRT-based diagnostics, knowledge graph reasoning, transformer-based prediction, and adaptive recommendation strategies to deliver personalized learning paths.

## Datasets

- **EdNet KT2** -- TOEIC English proficiency (20K users)
- **XES3G5M** -- Math knowledge tracing (6K users) -- [github.com/ai4ed/XES3G5M](https://github.com/ai4ed/XES3G5M?tab=readme-ov-file)

## Architecture

7 specialized ML agents + Orchestrator with 3 context-aware pipelines.

| Agent | Model | Purpose |
|-------|-------|---------|
| DiagnosticAgent | IRT 3PL + CAT | Ability estimation (theta) |
| ConfidenceAgent | Rule-based 6-class | Behavioral confidence classification |
| KGAgent | GraphSAGE + DAG | Knowledge graph embeddings + prerequisite mining |
| PredictionAgent | SAINT Transformer 4L/256d | Knowledge gap prediction (293-dim gap vector) |
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
|   |-- prediction_agent.py     # SAINT Transformer 4L/256d, 293-dim gap vector
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
pip install -r requirements.txt
```

## Running

```bash
# Full pipeline with confidence synthesis (XES3G5M)
python scripts/run_xes3g5m_full.py --seed 42

# Multi-seed evaluation (5 seeds)
python scripts/run_multi_seed.py

# Baselines
python scripts/run_xes3g5m_baselines.py

# Ablation study
python scripts/run_xes3g5m_ablation.py

# Generate figures
python scripts/generate_ieee_figures.py
python scripts/generate_xes3g5m_figures.py
python scripts/generate_architecture_figs.py
```

## Configuration

All hyperparameters are centralized in `configs/config.yaml`.
Agents load their section automatically via `BaseAgent._load_config()`.

## Citation

[will be added after publication]
