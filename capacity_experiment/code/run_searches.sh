#!/bin/bash
# Capacity experiment -- Stage A: per-budget architecture search.
# Runs every budget's NAS search concurrently and waits for all to finish.
# MUST be launched as a managed background task, otherwise the child search
# processes are reaped when the launching command returns.
#
# Paths resolve relative to this script, so the repository can live anywhere.
# Override the interpreter with KGD_PYTHON if `python` is not the right one.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO/unsupervised_nas" || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

PY="${KGD_PYTHON:-python}"
CFG="$REPO/capacity_experiment/configs"
LOGS="$REPO/capacity_experiment/logs"
mkdir -p "$LOGS"

BUDGETS="3 6 8 13 30"
echo "=== launch $(date '+%H:%M:%S') ==="

for B in $BUDGETS; do
    rm -rf "experiments/cap_budget_${B}M"
    "$PY" -u scripts/stage2_nas_search.py \
        --config "$CFG/budget_${B}M.yaml" \
        --output-dir "experiments/cap_budget_${B}M" \
        > "$LOGS/search_${B}M.log" 2>&1 &
    echo "  started ${B}M (pid $!)"
done

wait
echo "=== all searches finished $(date '+%H:%M:%S') ==="
for B in $BUDGETS; do
    if [ -f "experiments/cap_budget_${B}M/search_results.json" ]; then
        echo "  ${B}M: search_results.json OK"
    else
        echo "  ${B}M: MISSING search_results.json"
    fi
done
