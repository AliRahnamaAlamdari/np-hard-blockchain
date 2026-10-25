#!/usr/bin/env bash
#
# Reproduce every quantitative result in the paper, plus the ablations and
# sensitivity analyses the paper does not contain.
#
# Usage:
#   ./run_all.sh              # standard sweep (see README for runtimes)
#   ./run_all.sh --smoke      # toy sizes, finishes in a couple of minutes
#   ./run_all.sh --full       # includes the N=12 row at 50 repetitions (hours)
#   ./run_all.sh --skip-tests # skip the test suite
#
# Results land in results/<experiment>/, LaTeX in results/latex/, figures in
# results/figures/.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
fi

CONFIG_FLAG=()
RESULTS_DIR="results"
RUN_TESTS=1
PROBLEM_SIZE_CONFIG="configs/experiments/table4_problem_size.yaml"

for arg in "$@"; do
    case "$arg" in
        --smoke)
            CONFIG_FLAG=(--config configs/smoke.yaml)
            RESULTS_DIR="results/smoke"
            ;;
        --full)
            PROBLEM_SIZE_CONFIG="configs/experiments/table4_problem_size_full.yaml"
            ;;
        --skip-tests) RUN_TESTS=0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

EXPERIMENTS=(
    table3_time_allocation
    time_allocation_sweep
    table4_problem_size
    table5_fairness
    table6_validation_cost
    convergence
    ablation_vrf_grinding
    ablation_components
    sensitivity_repository_size
    sensitivity_hash_rate
    sensitivity_solver_advantage
)

banner() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

START=$(date +%s)

if [[ $RUN_TESTS -eq 1 ]]; then
    banner "Test suite"
    "$PYTHON" -m pytest -q
fi

for experiment in "${EXPERIMENTS[@]}"; do
    banner "$experiment"
    if [[ "$experiment" == "table4_problem_size" && ${#CONFIG_FLAG[@]} -eq 0 ]]; then
        "$PYTHON" scripts/run_experiment.py "$experiment" --config "$PROBLEM_SIZE_CONFIG"
    else
        "$PYTHON" scripts/run_experiment.py "$experiment" "${CONFIG_FLAG[@]}"
    fi
done

banner "LaTeX tables"
"$PYTHON" scripts/make_tables.py --results-dir "$RESULTS_DIR" --output-dir "$RESULTS_DIR/latex"

banner "Figures"
"$PYTHON" scripts/make_figures.py --results-dir "$RESULTS_DIR" --output-dir "$RESULTS_DIR/figures"

# The manuscript figures also need reseeding_bias from run_all_reviewer.sh.
if [[ ${#CONFIG_FLAG[@]} -eq 0 && -f "$RESULTS_DIR/reseeding_bias/reseeding_bias.json" ]]; then
    banner "Manuscript figures"
    "$PYTHON" scripts/make_paper_figures.py --results-dir "$RESULTS_DIR" --output-dir "$RESULTS_DIR/figures/paper"
fi

ELAPSED=$(( $(date +%s) - START ))
banner "Done in $((ELAPSED / 60))m $((ELAPSED % 60))s"
echo "Results:  $RESULTS_DIR/"
echo "Tables:   $RESULTS_DIR/latex/"
echo "Figures:  $RESULTS_DIR/figures/"
