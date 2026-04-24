"""
Run a SINGLE ablation config in isolation. Used to parallelise the ablation
loop in run_xes3g5m_ablation.py — pass a config name and it'll execute only
that one, writing to its own results dir.

Usage:
    python scripts/run_ablation_one.py --seed 42 --config "- IRT (Diagnostic)"

Available config names:
    "- Prediction"
    "- Knowledge Graph"
    "- Confidence"
    "- IRT (Diagnostic)"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd

from agents.prediction_agent import set_num_tags
from agents.utils import set_global_seed
from data.xes3g5m_loader import load_xes3g5m
from scripts.run_xes3g5m_full import (
    build_xes3g5m_questions_df, build_xes3g5m_lectures_df,
)
from scripts.run_xes3g5m_ablation import run_ablation_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("ablation_one")

CONFIG_KWARGS = {
    "- Prediction":       {"disable_prediction": True},
    "- Knowledge Graph":  {"disable_kg": True},
    "- Confidence":       {"disable_confidence": True},
    "- IRT (Diagnostic)": {"disable_irt": True},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, required=True,
                         choices=list(CONFIG_KWARGS.keys()))
    parser.add_argument("--n_students", type=int, default=6000)
    parser.add_argument("--min_interactions", type=int, default=20)
    args = parser.parse_args()

    set_global_seed(args.seed)
    logger.info("=" * 60)
    logger.info("ABLATION-ONE: %s  (seed=%d)", args.config, args.seed)
    logger.info("=" * 60)

    train_df, val_df, test_df = load_xes3g5m(
        n_students=args.n_students,
        min_interactions=args.min_interactions,
        seed=args.seed,
    )
    for df in [train_df, val_df, test_df]:
        df["confidence_class"] = 0

    train_max_id = 0
    for tags in train_df["tags"]:
        if isinstance(tags, list) and tags:
            train_max_id = max(train_max_id, max(int(t) for t in tags))
    n_tags = train_max_id + 1
    logger.info("NUM_TAGS = %d", n_tags)
    set_num_tags(n_tags)

    questions_df = build_xes3g5m_questions_df("data/xes3g5m/XES3G5M")
    if "bundle_id" not in questions_df.columns:
        questions_df["bundle_id"] = questions_df["question_id"]
    if "correct_answer" not in questions_df.columns:
        questions_df["correct_answer"] = "A"
    if "deployed_at" not in questions_df.columns:
        questions_df["deployed_at"] = 0
    lectures_df = pd.DataFrame({
        "lecture_id": [], "tags": [], "part_id": [],
        "type_of": [], "bundle_id": [],
    })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = args.config.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "minus")
    run_dir = ROOT / "results" / "xes3g5m" / f"ablation_one_s{args.seed}_{safe_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    kwargs = CONFIG_KWARGS[args.config]
    t0 = time.time()
    metrics = run_ablation_config(
        args.config, train_df, val_df, test_df, questions_df, lectures_df,
        args.seed, run_dir, **kwargs,
    )
    metrics["time_s"] = round(time.time() - t0, 1)

    out_file = run_dir / "ablation_one.json"
    with open(out_file, "w") as f:
        json.dump({args.config: metrics}, f, indent=2, default=str)
    logger.info("Saved to %s  (%.1fs)", out_file, metrics["time_s"])


if __name__ == "__main__":
    main()
