#!/usr/bin/env python3
"""
MARS v4: Improved pipeline with:
  1. 10x larger sample (10K users → ~4-5K after cleaning vs 390 before)
  2. Per-Part IRT signal (7 TOEIC parts → differential ability vector)
  3. Asymmetric ZPD (prioritise items slightly harder than ability)
  4. Prerequisite-readiness KG signal (combines mastery + gap status)
  5. Fixed config bug: prerequisites now use config thresholds (was hardcoded)
  6. Fixed min_cooccurrences filter (was not applied)

Usage:
  cd ednet-mars/
  python scripts/run_v4_improved.py

Expected runtime: ~30-60 min on CPU (10K users, 5 seeds)
"""

import sys
import os
import json
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mars.v4")

# ── Constants ────────────────────────────────────────────────────────
NUM_TAGS = 293
FAST_MARS_PRED_EPOCHS = 5
SEEDS = [42, 123, 456, 789, 2024]
SAMPLE_USERS = 10_000
RESULTS_DIR = Path("results/v4_improved")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_tags(tags):
    if isinstance(tags, list):
        return [int(t) for t in tags if 0 <= int(t) < NUM_TAGS]
    if isinstance(tags, (int, np.integer)):
        return [int(tags)] if 0 <= tags < NUM_TAGS else []
    if isinstance(tags, str):
        return [int(t) for t in tags.split(";") if t.strip() and 0 <= int(t.strip()) < NUM_TAGS]
    return []


# ── Step 1: Load & preprocess with larger sample ────────────────────
SPLITS_CACHE = {
    "train": Path("data/splits/train.parquet"),
    "val":   Path("data/splits/val.parquet"),
    "test":  Path("data/splits/test.parquet"),
}

def load_data():
    from data.loader import EdNetLoader
    from data.preprocessor import EdNetPreprocessor

    logger.info("=" * 60)
    logger.info("STEP 1: Loading data (sample_users=%d)", SAMPLE_USERS)
    logger.info("=" * 60)

    # Skip preprocessing if cached splits exist
    if all(p.exists() for p in SPLITS_CACHE.values()):
        logger.info("Found cached splits — loading from parquet (skipping preprocessing)")
        splits = {name: pd.read_parquet(path) for name, path in SPLITS_CACHE.items()}
        for name, df in splits.items():
            logger.info("  %s: %d rows, %d users", name, len(df), df["user_id"].nunique())
        return splits

    loader = EdNetLoader(data_dir="data/raw")
    interactions = loader.load_interactions(
        sample_users=SAMPLE_USERS,
        stratified_sampling=True,
        random_seed=42,
    )
    logger.info("Loaded %d interactions from %d users",
                len(interactions), interactions["user_id"].nunique())

    preprocessor = EdNetPreprocessor(
        output_dir="data/processed",
        splits_dir="data/splits",
        chunk_size=5000,
    )
    splits = preprocessor.run(interactions, chunked=True)

    for name, df in splits.items():
        logger.info("  %s: %d rows, %d users", name, len(df), df["user_id"].nunique())

    return splits


# ── Step 2: Build evaluation pairs ──────────────────────────────────
def build_eval_pairs(splits, context_ratio=0.5):
    """Build (user, context, ground_truth) pairs from test split."""
    train_df = splits["train"]
    test_df = splits["test"]

    # Parse tags if stored as string
    for df in [train_df, test_df]:
        if df["tags"].dtype == object and isinstance(df["tags"].iloc[0], str):
            df["tags"] = df["tags"].apply(
                lambda s: [int(t) for t in str(s).split(";") if t.strip()]
            )

    eval_pairs = []
    test_users = test_df["user_id"].unique()
    logger.info("Building eval pairs for %d test users", len(test_users))

    for uid in test_users:
        user_test = test_df[test_df["user_id"] == uid].sort_values("timestamp")
        if len(user_test) < 4:
            continue

        split_idx = int(len(user_test) * context_ratio)
        if split_idx < 2:
            split_idx = 2
        if split_idx >= len(user_test) - 1:
            continue

        context = user_test.iloc[:split_idx].copy()
        future = user_test.iloc[split_idx:].copy()

        # Ground truth:
        #   gt_all_tags = ALL future interaction tags (primary: predicts future engagement)
        #   gt_tags     = only future FAILED tags (secondary: gap-filling metric)
        gt_tags = set()
        gt_all_tags = set()
        for _, row in future.iterrows():
            tags = parse_tags(row.get("tags", []))
            gt_all_tags.update(tags)
            if not row["correct"]:
                gt_tags.update(tags)

        # Use gt_all_tags as primary ground truth (more items → stable NDCG)
        # Skip users with no future interactions at all
        if not gt_all_tags:
            continue

        eval_pairs.append((uid, context, list(gt_all_tags), list(gt_tags), future))

    logger.info("Built %d eval pairs (%.1f%% of test users)",
                len(eval_pairs), 100 * len(eval_pairs) / max(len(test_users), 1))
    return eval_pairs, train_df


# ── Step 3: Compute metrics ─────────────────────────────────────────
def compute_metrics(preds, gt_list, gta_list, k_values=(5, 10)):
    """Compute NDCG@K, MAP@K, MRR, P@K, R@K."""
    from math import log2

    results = {}

    for k in k_values:
        ndcg_scores = []
        map_scores = []
        precision_scores = []
        recall_scores = []

        for ranked, gt, gt_all in zip(preds, gt_list, gta_list):
            top_k = ranked[:k]
            gt_set = set(gt)
            gt_all_set = set(gt_all)

            # NDCG@K
            dcg = sum(1.0 / log2(i + 2) for i, t in enumerate(top_k) if t in gt_set)
            ideal = sorted([1.0 / log2(i + 2) for i in range(min(len(gt_set), k))], reverse=True)
            idcg = sum(ideal)
            ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

            # MAP@K
            hits = 0
            sum_prec = 0.0
            for i, t in enumerate(top_k):
                if t in gt_set:
                    hits += 1
                    sum_prec += hits / (i + 1)
            map_scores.append(sum_prec / min(len(gt_set), k) if gt_set else 0.0)

            # P@K, R@K
            relevant_in_k = len(set(top_k) & gt_set)
            precision_scores.append(relevant_in_k / k)
            recall_scores.append(relevant_in_k / max(len(gt_set), 1))

        results[f"ndcg@{k}"] = float(np.mean(ndcg_scores))
        results[f"map@{k}"] = float(np.mean(map_scores))
        results[f"precision@{k}"] = float(np.mean(precision_scores))
        results[f"recall@{k}"] = float(np.mean(recall_scores))

    # MRR
    mrr_scores = []
    for ranked, gt, _ in zip(preds, gt_list, gta_list):
        gt_set = set(gt)
        rr = 0.0
        for i, t in enumerate(ranked):
            if t in gt_set:
                rr = 1.0 / (i + 1)
                break
        mrr_scores.append(rr)
    results["mrr"] = float(np.mean(mrr_scores))

    return results


# ── Step 4: Run experiment across seeds ──────────────────────────────
def run_experiment():
    t_start = time.time()

    # Load data
    splits = load_data()
    eval_pairs, train_df = build_eval_pairs(splits)

    if len(eval_pairs) < 10:
        logger.error("Too few eval pairs (%d). Check data loading.", len(eval_pairs))
        return

    # Import run_mars (uses our v4 changes)
    from scripts._run_mars_v3 import run_mars

    # Load previously completed seeds
    partial_path = RESULTS_DIR / "v4_partial_results.json"
    all_results = {}
    if partial_path.exists():
        with open(partial_path) as f:
            partial = json.load(f)
        for s_str, res in partial.get("per_seed", {}).items():
            all_results[int(s_str)] = res
        if all_results:
            logger.info("Resuming: already completed seeds %s", list(all_results.keys()))

    for seed in SEEDS:
        if seed in all_results:
            logger.info("SEED %d — skipping (already done)", seed)
            continue
        logger.info("=" * 60)
        logger.info("SEED %d", seed)
        logger.info("=" * 60)

        # LSTM cache: train once per seed, reuse for ablation
        lstm_cache = RESULTS_DIR / f"lstm_cache_seed{seed}.pt"

        # Full MARS (v4: per-Part IRT + improved KG)
        preds, scores, gt, gta, _, _, agent_m = run_mars(
            eval_pairs, train_df, seed=seed,
            pred_epochs=FAST_MARS_PRED_EPOCHS,
            lstm_cache_path=str(lstm_cache),
        )
        metrics = compute_metrics(preds, gt, gta)
        logger.info("MARS v4: %s", {k: f"{v:.4f}" for k, v in metrics.items()})

        # Ablation: disable IRT (reuse cached LSTM)
        preds_no_irt, _, gt_ni, gta_ni, _, _, _ = run_mars(
            eval_pairs, train_df, seed=seed,
            pred_epochs=FAST_MARS_PRED_EPOCHS,
            disable_irt=True,
            lstm_cache_path=str(lstm_cache),
        )
        metrics_no_irt = compute_metrics(preds_no_irt, gt_ni, gta_ni)
        logger.info("  −IRT:  NDCG@10=%.4f (Δ=%.4f)",
                     metrics_no_irt["ndcg@10"],
                     metrics["ndcg@10"] - metrics_no_irt["ndcg@10"])

        # Ablation: disable KG (reuse cached LSTM)
        preds_no_kg, _, gt_nk, gta_nk, _, _, _ = run_mars(
            eval_pairs, train_df, seed=seed,
            pred_epochs=FAST_MARS_PRED_EPOCHS,
            disable_kg=True,
            lstm_cache_path=str(lstm_cache),
        )
        metrics_no_kg = compute_metrics(preds_no_kg, gt_nk, gta_nk)
        logger.info("  −KG:   NDCG@10=%.4f (Δ=%.4f)",
                     metrics_no_kg["ndcg@10"],
                     metrics["ndcg@10"] - metrics_no_kg["ndcg@10"])

        all_results[seed] = {
            "full": metrics,
            "no_irt": metrics_no_irt,
            "no_kg": metrics_no_kg,
            "agent_metrics": agent_m,
        }

        # Save intermediate results after each seed
        partial_path = RESULTS_DIR / "v4_partial_results.json"
        with open(partial_path, "w") as f:
            json.dump({
                "completed_seeds": list(all_results.keys()),
                "per_seed": {str(s): all_results[s] for s in all_results},
                "n_eval_pairs": len(eval_pairs),
                "runtime_minutes": round((time.time() - t_start) / 60, 1),
            }, f, indent=2)
        logger.info("Partial results saved to %s (%d/%d seeds done)",
                     partial_path, len(all_results), len(SEEDS))

    # ── Aggregate ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("AGGREGATE RESULTS (v4, %d seeds)", len(SEEDS))
    logger.info("=" * 60)

    for config_name in ["full", "no_irt", "no_kg"]:
        vals = [all_results[s][config_name] for s in SEEDS]
        agg = {}
        for key in vals[0]:
            arr = [v[key] for v in vals]
            agg[key] = {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
        logger.info("%s: NDCG@10=%.4f±%.4f, MAP@10=%.4f±%.4f, MRR=%.4f±%.4f",
                     config_name,
                     agg["ndcg@10"]["mean"], agg["ndcg@10"]["std"],
                     agg["map@10"]["mean"], agg["map@10"]["std"],
                     agg["mrr"]["mean"], agg["mrr"]["std"])

    # IRT contribution
    full_ndcg = np.mean([all_results[s]["full"]["ndcg@10"] for s in SEEDS])
    no_irt_ndcg = np.mean([all_results[s]["no_irt"]["ndcg@10"] for s in SEEDS])
    no_kg_ndcg = np.mean([all_results[s]["no_kg"]["ndcg@10"] for s in SEEDS])
    logger.info("IRT contribution: Δ NDCG@10 = %.4f", full_ndcg - no_irt_ndcg)
    logger.info("KG  contribution: Δ NDCG@10 = %.4f", full_ndcg - no_kg_ndcg)

    # Save results
    out_path = RESULTS_DIR / "v4_experiment_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "sample_users": SAMPLE_USERS,
                "seeds": SEEDS,
                "pred_epochs": FAST_MARS_PRED_EPOCHS,
                "changes": [
                    "per-Part IRT (7 parts) instead of global theta",
                    "asymmetric ZPD: prioritise harder items",
                    "prerequisite-readiness KG signal with gap weighting",
                    "fixed config bug: prereq thresholds now from config",
                    "fixed min_cooccurrences filter in build_prerequisites",
                    "increased sample to 10K users",
                    "adjusted weights: PRED=0.55, IRT=0.20, KG=0.10, CONF=0.10, CLUST=0.05",
                ],
            },
            "per_seed": {str(s): all_results[s] for s in SEEDS},
            "n_eval_pairs": len(eval_pairs),
            "n_train_users": int(train_df["user_id"].nunique()),
            "runtime_minutes": round((time.time() - t_start) / 60, 1),
        }, f, indent=2)
    logger.info("Results saved to %s", out_path)

    elapsed = time.time() - t_start
    logger.info("Total runtime: %.1f min", elapsed / 60)


if __name__ == "__main__":
    run_experiment()
