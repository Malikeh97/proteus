#!/bin/bash
# Phases 3 and 4 -- metrics and plots.
# Usage: bash run_analysis.sh
# Note: runs directly on the login node (no GPU needed -- pure post-processing).

set -e

source setup/start_env.sh

ANALYSIS=$PROTEUS_OUTPUT_DIR/analysis
mkdir -p "$ANALYSIS"

# --- Phase 3: metrics ---
# python scripts/evaluate.py \
#     --results-dir "$PROTEUS_OUTPUT_DIR" \
#     --menu full \
#     --out "$ANALYSIS/metrics.csv" | tee "$ANALYSIS/summary.txt"

python scripts/evaluate.py --results-dir "$PROTEUS_OUTPUT_DIR" --menu mvp \
    --out "$ANALYSIS/metrics_mvp.csv" | tee "$ANALYSIS/summary_mvp.txt"

# --- Phase 4: plots ---
# python scripts/plot_results.py --metrics "$ANALYSIS/metrics.csv"

python scripts/plot_results.py --metrics "$ANALYSIS/metrics_mvp.csv" \
    --out-dir "$ANALYSIS/plots_mvp"

echo "Analysis written to $ANALYSIS"
