#!/usr/bin/env bash
#
# Fast end-to-end check: the unit tests plus every experiment on toy parameters.
# Intended to run in a couple of minutes before committing to a full sweep.
#
# Numbers produced here are NOT paper results -- they only prove the code paths work.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
fi

echo "==> Unit tests (fast subset)"
"$PYTHON" -m pytest -q -m "not slow"

echo
echo "==> Full pipeline on toy data"
"$ROOT/run_all.sh" --smoke --skip-tests

echo
echo "==> Verifying expected artifacts"
MISSING=0
for experiment in table3_time_allocation table4_problem_size table5_fairness \
                  table6_validation_cost convergence ablation_vrf_grinding \
                  ablation_components sensitivity_repository_size \
                  sensitivity_hash_rate sensitivity_solver_advantage; do
    path="results/smoke/$experiment/$experiment.json"
    if [[ -s "$path" ]]; then
        echo "  ok   $path"
    else
        echo "  MISS $path"
        MISSING=1
    fi
done

if [[ $MISSING -ne 0 ]]; then
    echo
    echo "Smoke test FAILED: expected artifacts are missing." >&2
    exit 1
fi

echo
echo "Smoke test passed. Run ./run_all.sh for the real sweep."
