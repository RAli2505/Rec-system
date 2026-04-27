#!/bin/bash
# Resume chain after AKT/r_safe fix and skip-existing logic.
# Re-runs failed (model, seed) pairs only — already-OK pairs preserved.
#
# Sequence:
#   1. B: pykt SAINT/AKT (reuse pykt_baselines_20260427_003746, skip OK)
#   2. more_kt: SAINTPlus/DKVMN/IEKT/GKT (reuse more_kt_baselines_20260427_004623)
#   3. DTransformer (A-style, fresh dir)

set -u
mkdir -p logs
echo "[$(date)] chain_resume started (pid $$)"

# ── Step 1: B with skip-existing ──
echo "[$(date)] launching B (pykt SAINT, AKT) — skip-existing"
python scripts/run_pykt_baselines.py \
    --models SAINT AKT \
    --seeds 42 123 456 789 2024 \
    --epochs 20 --batch_size 64 --patience 5 --seq_len 100 \
    --skip-existing --reuse-dir pykt_baselines_20260427_003746 \
    > logs/pykt_baselines_v3.log 2>&1
RC1=$?
echo "[$(date)] B exit=$RC1"

# ── Step 2: more_kt with skip-existing ──
echo "[$(date)] launching more_kt (SAINTPlus DKVMN IEKT GKT) — skip-existing"
python scripts/run_more_kt_baselines.py \
    --models SAINTPlus DKVMN IEKT GKT \
    --seeds 42 123 456 789 2024 \
    --epochs 20 --batch_size 64 --patience 5 --seq_len 100 \
    --skip-existing --reuse-dir more_kt_baselines_20260427_004623 \
    > logs/more_kt_baselines_v2.log 2>&1
RC2=$?
echo "[$(date)] more_kt exit=$RC2"

# ── Step 3: DTransformer (A-style, fresh) ──
echo "[$(date)] launching DTransformer (A-style)"
python scripts/run_attention_kt_baselines.py \
    --models DTransformer \
    --seeds 42 123 456 789 2024 \
    --epochs 20 --batch_size 64 --patience 5 \
    > logs/dtransformer_baseline.log 2>&1
RC3=$?
echo "[$(date)] DTransformer exit=$RC3"

echo "[$(date)] chain_resume DONE (B=$RC1, more_kt=$RC2, DT=$RC3)"
