#!/usr/bin/env bash
# Sentinel: waits until retrain_xes3g5m_pipeline.sh finishes, then runs
# postprocess_xes3g5m.sh, then writes a final ALL_DONE marker.
#
# Usage:
#   bash scripts/watch_and_postprocess.sh <RETRAIN_LOG_DIR>
#
# Designed for overnight run: keep polling forever, never exit on error,
# always log every state transition.

set -u
cd "$(dirname "$0")/.."  # project root

RETRAIN_LOG_DIR="${1:-}"
if [ -z "${RETRAIN_LOG_DIR}" ]; then
    echo "Usage: $0 <retrain_log_dir>"
    exit 2
fi

MASTER_LOG="${RETRAIN_LOG_DIR}/MASTER.log"
SENTINEL_LOG="${RETRAIN_LOG_DIR}/SENTINEL.log"
DONE_MARKER="${RETRAIN_LOG_DIR}/ALL_DONE"
FAIL_MARKER="${RETRAIN_LOG_DIR}/FAILED"

log() {
    echo "[SENTINEL $(date '+%Y-%m-%d %H:%M:%S')]  $*" | tee -a "${SENTINEL_LOG}"
}

log "Sentinel started. Watching ${MASTER_LOG}"
log "Will run postprocess once it sees 'All training steps complete'."

# ---- Phase 1: poll until master pipeline finishes ----
while true; do
    if [ -f "${MASTER_LOG}" ] && grep -q "All training steps complete" "${MASTER_LOG}" 2>/dev/null; then
        log "Detected pipeline completion."
        break
    fi
    sleep 60
done

# ---- Phase 2: run postprocess ----
log "Running scripts/postprocess_xes3g5m.sh ..."
POST_LOG="${RETRAIN_LOG_DIR}/postprocess.log"
if bash scripts/postprocess_xes3g5m.sh > "${POST_LOG}" 2>&1; then
    log "Postprocess OK. See ${POST_LOG}"
else
    rc=$?
    log "Postprocess FAILED (rc=${rc}). See ${POST_LOG}"
    echo "rc=${rc}" > "${FAIL_MARKER}"
fi

# ---- Phase 3: final summary ----
log "Generated paper-ready artifacts:"
{
    echo "Tables:"
    ls -la results/xes3g5m/tables/table_*.csv 2>/dev/null | awk '{print "  ",$0}'
    ls -la results/xes3g5m/tables/table3_main_results_no_r10.* 2>/dev/null | awk '{print "  ",$0}'
    ls -la results/xes3g5m/tables/table_seed_stability_full.* 2>/dev/null | awk '{print "  ",$0}'
    echo ""
    echo "Figures:"
    ls -la results/xes3g5m/figures/fig_methods_heatmap.* 2>/dev/null | awk '{print "  ",$0}'
    ls -la results/xes3g5m/figures/fig_ablation_heatmap.* 2>/dev/null | awk '{print "  ",$0}'
    ls -la results/xes3g5m/figures/fig_cd_diagram.* 2>/dev/null | awk '{print "  ",$0}'
    ls -la results/xes3g5m/figures/fig_radar_comparison.* 2>/dev/null | awk '{print "  ",$0}'
} | tee -a "${SENTINEL_LOG}"

date -Iseconds > "${DONE_MARKER}"
log "Wrote ALL_DONE marker. Sentinel exiting."
