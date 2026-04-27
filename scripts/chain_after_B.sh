#!/bin/bash
# After B (run_pykt_baselines.py) finishes, launch more_kt_baselines
# (SAINT+, DKVMN, IEKT, GKT — 4 models × 5 seeds, ~22-27h).
# Polls for B's DONE marker every 60s.

set -u
mkdir -p logs

echo "[$(date)] chain_after_B started (pid $$)"

B_LOG="logs/pykt_baselines_v2.log"
DONE_MARKER="DONE at"

WAIT_MAX=$((30 * 3600))   # 30h cap (B is ~5-7h)
WAITED=0
while ! grep -q "$DONE_MARKER" "$B_LOG" 2>/dev/null; do
  sleep 60
  WAITED=$((WAITED + 60))
  if [ $WAITED -ge $WAIT_MAX ]; then
    echo "[$(date)] gave up waiting for B; exiting"
    exit 2
  fi
done

echo "[$(date)] B finished, launching more_kt_baselines"

python scripts/run_more_kt_baselines.py \
    --models SAINTPlus DKVMN IEKT GKT \
    --seeds 42 123 456 789 2024 \
    --epochs 20 --batch_size 64 --patience 5 --seq_len 100 \
    > logs/more_kt_baselines.log 2>&1
RC1=$?
echo "[$(date)] more_kt_baselines exit=$RC1"

# After more_kt, also run DTransformer in A-style (matched 14-dim input)
echo "[$(date)] launching DTransformer (A-style, matched-input)"
python scripts/run_attention_kt_baselines.py \
    --models DTransformer \
    --seeds 42 123 456 789 2024 \
    --epochs 20 --batch_size 64 --patience 5 \
    > logs/dtransformer_baseline.log 2>&1
RC2=$?
echo "[$(date)] DTransformer exit=$RC2"

echo "[$(date)] chain_after_B DONE (more_kt=$RC1, DTransformer=$RC2)"
