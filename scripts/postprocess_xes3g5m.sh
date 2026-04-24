#!/usr/bin/env bash
# Run AFTER scripts/retrain_xes3g5m_pipeline.sh has finished.
# Aggregates raw metrics into 3 paper tables, then regenerates every
# downstream figure / table that depends on them.

set -uo pipefail
cd "$(dirname "$0")/.."  # project root

TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/postprocess_${TS}.log"

step() {
    local name="$1"; shift
    echo ""                           | tee -a "${LOG}"
    echo "=== ${name} ==="            | tee -a "${LOG}"
    echo "    cmd: $*"                | tee -a "${LOG}"
    if "$@" 2>&1 | tee -a "${LOG}"; then
        echo "  OK"                   | tee -a "${LOG}"
    else
        echo "  FAILED — continuing"  | tee -a "${LOG}"
    fi
}

RESTART_LOG="logs/retrain_NUMTAGS858_20260423_065312/mars_full_s2024_RESTART.log"
if [ -f "${RESTART_LOG}" ]; then
    echo ""                                          | tee -a "${LOG}"
    echo "=== 0a. Wait for parallel seed_2024 RESTART to finish ===" | tee -a "${LOG}"
    # Wait for either success line OR log inactivity > 10 min (process died)
    while ! grep -q "Saved to.*metrics.json" "${RESTART_LOG}" 2>/dev/null; do
        # mtime gap check (Git Bash compatible)
        last_mtime=$(stat -c %Y "${RESTART_LOG}" 2>/dev/null || echo 0)
        now=$(date +%s)
        gap=$((now - last_mtime))
        if [ "${gap}" -gt 600 ]; then
            echo "  seed_2024 RESTART log inactive for ${gap}s — assume dead, moving on" | tee -a "${LOG}"
            tail -5 "${RESTART_LOG}" | tee -a "${LOG}"
            break
        fi
        sleep 60
    done
    echo "  seed_2024 RESTART check complete"        | tee -a "${LOG}"
fi

step "0b. Backfill lstm_auc for seeds 42 and 123 (orchestrator was patched mid-run)" \
     python scripts/backfill_lstm_auc.py --seeds 42 123

step "1. Aggregate raw metrics into 3 CSV tables" \
     python scripts/aggregate_xes3g5m.py

step "2. Methods x metrics heatmap + Ablation heatmap + CD diagram" \
     python scripts/generate_paper_alt_figures.py

step "3. Per-seed stability table (LaTeX + Markdown)" \
     python scripts/generate_seed_table.py

step "4. Multi-metric radar (MARS / DKT / GRU)" \
     python scripts/generate_radar.py

step "5. Table 3 without R@10 (LaTeX + Markdown)" \
     python scripts/generate_table3_no_r10.py

echo ""                                | tee -a "${LOG}"
echo "Postprocess complete. Log: ${LOG}" | tee -a "${LOG}"
