#!/bin/bash
# Chain: attention-KT baselines (A: matched-input self-contained → B: canonical pykt).
# Full standard: 5 seeds (42, 123, 456, 789, 2024), epochs 20, patience 5,
# batch_size 64. Aligned with modern KT-paper convention (pyKT NeurIPS 2022,
# SimpleKT ICLR 2023, DTransformer WWW 2023 all use 5 seeds).
# Authorised for daytime + overnight run (already-running jobs continue past 8:00 cutoff).

set -u
mkdir -p logs

echo "[$(date)] chain_AB started (pid $$)"

# ── Script A — matched-input SAINT/AKT/SimpleKT (~7.5h) ──
echo "[$(date)] launching run_attention_kt_baselines.py (A)"
python scripts/run_attention_kt_baselines.py \
    --models SAINT AKT SimpleKT \
    --seeds 42 123 456 789 2024 \
    --epochs 20 \
    --patience 5 \
    --batch_size 64 \
    > logs/attn_kt_baselines.log 2>&1
RC_A=$?
echo "[$(date)] attn_kt_baselines exit=$RC_A"

# ── Script B — canonical pykt SAINT/AKT (~4h) ──
echo "[$(date)] launching run_pykt_baselines.py (B)"
python scripts/run_pykt_baselines.py \
    --models SAINT AKT \
    --seeds 42 123 456 789 2024 \
    --epochs 20 \
    --patience 5 \
    --batch_size 64 \
    --seq_len 100 \
    > logs/pykt_baselines.log 2>&1
RC_B=$?
echo "[$(date)] pykt_baselines exit=$RC_B"

echo "[$(date)] chain_AB DONE (A=$RC_A, B=$RC_B)"
