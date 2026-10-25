#!/usr/bin/env bash
#
# Regenerate every reviewer-response experiment, table, and figure.
#
#   ./run_all_reviewer.sh            # full runs (see per-experiment configs)
#   ./run_all_reviewer.sh --smoke    # quick pass on reduced parameters
#
# Mapping from reviewer comment / paper artifact -> regenerating command:
#
#   C9  problem_classes  -> Table tab:problemclasses, Fig problem_class_verify
#       python scripts/run_experiment.py problem_classes
#   C3  consensus        -> Tables tab:forkdelay, Figs consensus_fork_vs_delay/consensus_scale
#       python scripts/run_experiment.py consensus
#   C2  adversarial      -> Section sec:adversarial numbers, Fig adversarial_majority
#       python scripts/run_experiment.py adversarial
#   C5  baselines        -> Tables tab:comparison-measured / tab:comparison-full
#       python scripts/run_experiment.py baselines
#
# Reviewer 2 (second report):
#   R2-6/8 reseeding_bias       -> Tables tab:reseedrate, tab:reseedtarget; Fig reseeding_bias_vs_target
#       python scripts/run_experiment.py reseeding_bias
#   R2-3   precomputation_scale -> Tables tab:precompcost, tab:freshness; Fig precompute_freshness
#       python scripts/run_experiment.py precomputation_scale
#   R2-7/10 validation_scale    -> Tables tab:feasibilityscale, tab:commcost; Fig feasibility_vs_optimality
#       python scripts/run_experiment.py validation_scale
#   figures              -> python scripts/make_reviewer_figures.py
#
# The drafts in drafts/*.tex cite these outputs; numbers there trace to results/<exp>/.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
[[ -x "$ROOT/.venv/bin/python" ]] && PYTHON="$ROOT/.venv/bin/python"
export PYTHONPATH=src

SMOKE=""
for a in "$@"; do [[ "$a" == "--smoke" ]] && SMOKE=1; done

banner() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

EXPERIMENTS=(problem_classes baselines consensus adversarial reseeding_bias precomputation_scale validation_scale)

for exp in "${EXPERIMENTS[@]}"; do
    banner "$exp"
    if [[ -n "$SMOKE" ]]; then
        "$PYTHON" scripts/run_experiment.py "$exp" --seeds 0 1 --output-dir results/reviewer_smoke
    else
        "$PYTHON" scripts/run_experiment.py "$exp"
    fi
done

banner "Reviewer figures"
"$PYTHON" scripts/make_reviewer_figures.py

banner "Done"
echo "Results:  results/<experiment>/"
echo "Figures:  results/figures/"
echo "Drafts cite these: drafts/*.tex"
