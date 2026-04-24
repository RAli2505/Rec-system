#!/usr/bin/env bash
# Sentinel: waits for the 5-seed inference ablation to finish, then
# chains four more analyses sequentially. Stops on any FAIL/Traceback.
# Total budget: ~8 hours. Marker `OVERNIGHT_CHAIN_COMPLETE` is written
# at the end.

set -uo pipefail
cd "$(dirname "$0")/.."

TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/overnight_chain_${TS}.log"
mkdir -p logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG}"; }

run_step() {
    local name="$1"; shift
    local script="$1"; shift
    local step_log="logs/overnight_${name}.log"
    log ">>> START  ${name}  (cmd: ${script} ${*-})"
    local t0; t0=$(date +%s)
    if python "${script}" "$@" > "${step_log}" 2>&1; then
        log "<<< OK     ${name}  ($(( $(date +%s) - t0 ))s)"
        return 0
    else
        log "!!! FAIL   ${name}  rc=$?  see ${step_log}"
        tail -10 "${step_log}" | tee -a "${LOG}"
        return 1
    fi
}

log "============================================================"
log "OVERNIGHT CHAIN START — log dir: logs/, master ${LOG}"
log "============================================================"

# Phase 1: wait for the running 5-seed ablation
log "Phase 0: waiting for 5-seed ablation to print '=== SUMMARY ==='"
while ! grep -q "=== SUMMARY ===" logs/ablation_inference_5seeds_master.log 2>/dev/null; do
    if grep -qE "^Traceback|Killed|MemoryError" logs/ablation_inference_5seeds_master.log 2>/dev/null; then
        log "5-seed ablation looks crashed; aborting chain"
        exit 1
    fi
    sleep 120
done
log "Phase 0 ✓: 5-seed ablation finished"

# Phase 2: KG threshold sensitivity (~2h)
run_step "kg_sensitivity"  scripts/sensitivity_kg_thresholds.py             || true

# Phase 3: Recommendation-weight sensitivity (~2h)
run_step "rec_weights"     scripts/sensitivity_recommendation_weights.py    || true

# Phase 4: Context window sweep (~30 min, inference only)
run_step "context_window"  scripts/sensitivity_context_window.py            || true

# Phase 5: Per-config ECE/Brier on every ablation seed (uses logits we
#   have just produced in phase 1) — ~1h
run_step "posthoc_calib"   scripts/posthoc_calibration.py                    || true

log "============================================================"
log "OVERNIGHT_CHAIN_COMPLETE at $(date '+%Y-%m-%d %H:%M:%S')"
log "============================================================"
echo "OVERNIGHT_CHAIN_COMPLETE" >> "${LOG}"
