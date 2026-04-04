# MARS: Multi-Agent Recommender System for Personalized Learning

## Architecture
7 specialized ML agents + Orchestrator with 3 context-aware pipelines.

| Agent | Model | Purpose |
|-------|-------|---------|
| DiagnosticAgent | IRT 3PL + CAT | Ability estimation |
| ConfidenceAgent | XGBoost 6-class | Behavioral confidence classification |
| KGAgent | GraphSAGE + DAG | Knowledge graph + prerequisite mining |
| PredictionAgent | LSTM seq-to-set | Knowledge gap prediction (293 tags, horizon=10) |
| RecommendationAgent | TS + LambdaMART | Multi-strategy recommendation |
| PersonalizationAgent | K-Means | Learner clustering |
| Orchestrator | Rule-based | Pipeline coordination |

## Dataset
EdNet KT2 — TOEIC preparation, 784K students, 131M+ interactions.

## Setup
```bash
pip install -r requirements.txt
```

## Reproduce Results
```bash
# 1. Download EdNet KT2 data to data/raw/
# 2. Run notebooks in order:
jupyter notebook notebooks/01_eda.ipynb
# ... through 09_figures.ipynb

# Or run evaluation directly:
python scripts/run_multi_seed.py --config configs/config.yaml
```

## Project Structure
```
ednet-mars/
  agents/
    base_agent.py            -- Abstract base class
    utils.py                 -- set_global_seed(), load_config()
    diagnostic_agent.py      -- IRT 3PL + CAT
    confidence_agent.py      -- XGBoost 6-class
    kg_agent.py              -- GraphSAGE + prerequisites
    prediction_agent.py      -- LSTM seq-to-set
    recommendation_agent.py  -- TS + LambdaMART
    personalization_agent.py -- K-Means clustering
    orchestrator.py          -- 3 pipelines + batch evaluation
  configs/
    config.yaml              -- All hyperparameters
  data/
    loader.py                -- EdNet KT2 loader
    preprocessor.py          -- Feature engineering + splits
  models/                    -- Saved model weights
  notebooks/                 -- 01_eda through 09_figures
  test_*.py                  -- Unit tests per agent
  requirements.txt
```

## Configuration
All hyperparameters are centralized in `configs/config.yaml`.
Agents load their section automatically via `BaseAgent._load_config()`.

## Citation
[will be added after publication]
