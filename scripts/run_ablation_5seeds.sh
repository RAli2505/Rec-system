#!/usr/bin/env bash
# Ablation across all 5 seeds (Δ vs σ test for reviewer #1).
# Each seed runs full ablation (4 configs after Full MARS reuse) sequentially.
# Wall-clock estimate: 5 seeds × ~3 hours per seed = ~15 hours.
set -uo pipefail
cd "$(dirname "$0")/.."

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/ablation_5seeds_${TS}"
mkdir -p "${LOG_DIR}"
MASTER="${LOG_DIR}/MASTER.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${MASTER}"; }

log "=== ABLATION 5-SEEDS START (NUM_TAGS dynamic, log_dir=${LOG_DIR}) ==="

for SEED in 42 123 456 789 2024; do
    SEED_LOG="${LOG_DIR}/ablation_s${SEED}.log"
    log ">>> START seed_${SEED}  (cmd: scripts/run_xes3g5m_ablation.py --seed ${SEED})"
    t0=$(date +%s)
    if python scripts/run_xes3g5m_ablation.py --seed "${SEED}" > "${SEED_LOG}" 2>&1; then
        dt=$(( $(date +%s) - t0 ))
        log "<<< OK    seed_${SEED}  (${dt}s)"
    else
        rc=$?
        dt=$(( $(date +%s) - t0 ))
        log "!!! FAIL  seed_${SEED}  rc=${rc}  (${dt}s)"
    fi
done

log "=== ABLATION 5-SEEDS COMPLETE — see ${LOG_DIR}/ ==="
