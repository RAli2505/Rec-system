# ════════════════════════════════════════════════════════
# MARS — Multi-Agent Recommender System
# ════════════════════════════════════════════════════════

CONFIG = configs/config.yaml

.PHONY: data train evaluate all clean-results help

help:
	@echo "Usage:"
	@echo "  make data       - Load and preprocess EdNet KT2 data"
	@echo "  make train      - Train all agents (single seed)"
	@echo "  make evaluate   - Run multi-seed evaluation + aggregate"
	@echo "  make all        - Full pipeline: data -> train -> evaluate"
	@echo "  make clean-results - Remove results/ contents"

data:
	python -m data.loader --config $(CONFIG)
	python -m data.preprocessor --config $(CONFIG)

train:
	python scripts/run_multi_seed.py --config $(CONFIG) --seeds 42

evaluate:
	python scripts/run_multi_seed.py --config $(CONFIG)
	python scripts/aggregate_results.py
	python scripts/statistical_tests.py

all: data train evaluate

clean-results:
	rm -rf results/seed_*
	rm -rf results/aggregated
