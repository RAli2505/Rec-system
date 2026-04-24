#!/usr/bin/env bash
# Master driver: re-trains the full XES3G5M pipeline with the corrected
# dynamic NUM_TAGS (now 858 instead of hardcoded 293).
#
# Steps:
#   1. MARS full pipeline x 5 seeds  (Table 3 MARS row, Table 5)
#   2. Baselines (Random/Pop/DKT/GRU)  (Table 3 baseline rows)
#   3. Ablation (- Prediction, - KG, - Confidence, - IRT)  (Table 4)
#
# All stdout/stderr is teed into one log per step plus an overall log.
#
# Usage:
#   bash scripts/retrain_xes3g5m_pipeline.sh
#
# Wall-clock: ~9 hours on the seed_42 baseline GPU spec.

set -uo pipefail

cd "$(dirname "$0")/.."  # project root

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/retrain_NUMTAGS858_${TS}"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/MASTER.log"

log() {
    local msg="$*"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]  ${msg}" | tee -a "${MASTER_LOG}"
}

run_step() {
    local name="$1"; shift
    local step_log="${LOG_DIR}/${name}.log"
    local t0; t0=$(date +%s)
    log ">>> START  ${name}"
    log "    cmd : $*"
    log "    log : ${step_log}"
    if "$@" > "${step_log}" 2>&1; then
        local dt=$(( $(date +%s) - t0 ))
        log "<<< OK     ${name}  (${dt}s)"
    else
        local rc=$?
        local dt=$(( $(date +%s) - t0 ))
        log "!!! FAIL   ${name}  rc=${rc}  (${dt}s)"
        log "    See ${step_log} for details"
        return ${rc}
    fi
}

log "===================================================================="
log "  XES3G5M retraining pipeline (dynamic NUM_TAGS = 858)"
log "  Log directory: ${LOG_DIR}"
log "===================================================================="

# ---- 1. MARS full pipeline x 5 seeds ----
for SEED in 42 123 456 789 2024; do
    run_step "mars_full_s${SEED}" \
        python scripts/run_xes3g5m_full.py --seed "${SEED}" || true
done

# ---- 2. Baselines (single seed, deterministic for Random/Pop/DKT/GRU) ----
run_step "baselines_s42" \
    python scripts/run_xes3g5m_baselines.py --seed 42 || true

# ---- 3. Ablation ----
run_step "ablation_s42" \
    python scripts/run_xes3g5m_ablation.py --seed 42 || true

log "===================================================================="
log "  All training steps complete. Next: run aggregator + figure regen."
log "===================================================================="
