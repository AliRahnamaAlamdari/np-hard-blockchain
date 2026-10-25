# Reproduction package — Two-Stage VRF-Based PoUW

A complete, runnable reimplementation of every quantitative result in *"A Two-Stage
VRF-Based Task Assignment Framework for Proof-of-Useful-Work Blockchains Using NP-hard
Computation"* (Rahnama Alamdari & Malekimajd), plus the ablations, sensitivity
analyses, and statistical tests the paper does not contain.

**Read [`PAPER_SPEC.md`](PAPER_SPEC.md) first.** It is the complete extraction of the
paper — every equation, algorithm, hyperparameter, and reported number — and §11 lists
every ambiguity and internal inconsistency found during extraction. This
implementation resolves none of them silently; each assumption is recorded in
`PAPER_SPEC.md` §11 and marked in the config files.

---

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

or with conda:

```bash
conda env create -f environment.yml
conda activate pouw-repro
```

Python ≥ 3.10 is required (PEP 604 unions, PEP 585 generics). The paper's prototype
used Python 3.9, which will not run this package. No compiled extension is needed —
the ECVRF is pure Python.

Verify the install in about two minutes:

```bash
./scripts/smoke_test.sh
```

That runs the fast unit tests, then every experiment on toy parameters, then checks
the expected artifacts exist. Numbers from the smoke config are **not** results.

## Reproducing everything

```bash
./run_all.sh                 # standard sweep
./run_all.sh --full          # includes the N=12 row at 50 repetitions (hours)
./run_all.sh --smoke         # toy sizes
./run_all.sh --skip-tests
```

Output layout:

```
results/<experiment>/<experiment>.json    # results + provenance manifest
results/<experiment>/*.csv                # flat tables
results/<experiment>/*.pdf|.png           # figures
results/<experiment>/<experiment>.log     # per-run log
results/latex/<experiment>.tex            # paste-ready LaTeX
results/figures/                          # regenerated figures
```

Every JSON file embeds the fully resolved config, the seed list, package versions, and
a machine fingerprint, so any number can be traced to the run that produced it.

## Paper artifact → command

| Paper artifact | Command | Runtime |
|---|---|---|
| Table 3 `tab:timeallocation` — time share by difficulty | `python scripts/run_experiment.py table3_time_allocation` | ~2 min |
| Table 4 `tab:problemsize` — solve time vs. N | `python scripts/run_experiment.py table4_problem_size` | ~1 min |
| Table 4 full sweep incl. N=12 | `python scripts/run_experiment.py table4_problem_size --config configs/experiments/table4_problem_size_full.yaml` | hours |
| Table 5 `tab:fairness` — block distribution | `python scripts/run_experiment.py table5_fairness` | ~4 min |
| Table 6 `tab:validationcost` — validation latency | `python scripts/run_experiment.py table6_validation_cost` | ~2 min |
| §"Convergence and Correctness" (L483) | `python scripts/run_experiment.py convergence` | ~1 min |
| Table 1 `tab:designrationale` (rhetorical → measured) | `python scripts/run_experiment.py ablation_components` | ~1 min |
| Grinding bound argued at L375 | `python scripts/run_experiment.py ablation_vrf_grinding` | ~1 min |
| **New** — repository size N of Equation (3) | `python scripts/run_experiment.py sensitivity_repository_size` | ~4 min |
| **New** — fairness beyond equal power | `python scripts/run_experiment.py sensitivity_hash_rate` | ~3 min |
| **New** — solver-quality mining advantage | `python scripts/run_experiment.py sensitivity_solver_advantage` | ~1 min |
| **New** — extended size sweep, all three solver backends | `python scripts/run_experiment.py table4_problem_size --config configs/experiments/table4_problem_size_extended.yaml` | ~15 min |
| **New** — threshold acceptance beyond exact-solver reach | `python scripts/run_experiment.py heuristic_acceptance` | ~2 min |
| All LaTeX tables | `python scripts/make_tables.py` | seconds |
| All figures | `python scripts/make_figures.py` | seconds |
| Useful-work share vs gate difficulty, $d$ = 6..18 | `python scripts/run_experiment.py time_allocation_sweep` | ~1 min |
| Manuscript figures (Fig1--Fig4, vector PDF at print width) | `python scripts/make_paper_figures.py` | seconds |

Runtimes are wall-clock on the reference machine below. `python scripts/run_experiment.py --list`
prints the registry.

Common overrides, applied on top of any config:

```bash
python scripts/run_experiment.py table5_fairness --seeds 0 1 2 3 4 --blocks 500
python scripts/run_experiment.py table6_validation_cost --vrf ecdsa_pseudo
python scripts/run_experiment.py table4_problem_size --repetitions 50
python scripts/run_experiment.py table4_problem_size --solver held_karp
```

## Solver backends

Five solvers sit behind one interface and are chosen per experiment by
`solver.name`. Block validity never depends on which one ran: a block is accepted on
its witness, its recomputed cost, and the repository's published quality bound.

| `solver.name` | Kind | Growth | Practical reach on the reference host |
|---|---|---|---|
| `brute_force` | exact | `(N-1)!` | N ≤ 12 (20 s/solve at N=12) |
| `held_karp` | exact | `O(2^N N^2)` | N ≤ 20 (11 s/solve at N=20) |
| `nearest_neighbor` | heuristic | `O(N^2)` | thousands |
| `nearest_neighbor_2opt` | heuristic | `O(N^2)` per pass | thousands |
| `or_opt` | heuristic | `O(N^2)` per pass | hundreds |

The two exact solvers return the same optimum on every shared instance, which the
test suite asserts. The heuristics are used with `repository.reference_solver`, which
makes the repository publish a per-instance reference cost inside the Merkle leaf;
validators then accept a tour within `(1 + solver.quality_threshold)` of that bound.
Setting `solver.quality_threshold` without `repository.reference_solver` is a config
error, because a bound taken from the block itself can never fail.

## Hardware assumptions

Reference machine for all reported runtimes:

- Apple Silicon, macOS 15 (Darwin 25.5.0), CPython 3.14
- Single-threaded throughout; no GPU, no multiprocessing
- Peak resident memory under 300 MB for every experiment except the N=12 row
- Measured baselines: ~240,000 header hashes/s, ~0.44 µs per TSP tour evaluated,
  ~9 ms per ECVRF proof, ~7 ms per ECVRF verification

The paper's machine (Intel Core i7, 16 GB, Python 3.9, L412) is slower per tour by
roughly 4×. **Absolute seconds are not expected to match the paper and are not the
reproducible claim.** What reproduces is every ratio, share, and scaling relationship:
useful-work share against difficulty, `(N-1)!` growth, the fairness statistics, and
the solve/verify asymmetry.

`run.repetitions`, `network.num_blocks`, and `sweep.instance_sizes` are the knobs for
trading runtime against precision. The N=12 row costs ~40 s per solve here; 50
repetitions × 3 seeds is about 100 minutes for that row alone.

## Package layout

```
src/pouw/
  config.py            typed YAML config with inheritance and validation
  seeding.py           deterministic per-component seed derivation
  logging_utils.py     timing and memory instrumentation
  checkpoint.py        checkpoint/resume, guarded by a config fingerprint
  stats_utils.py       aggregation that tolerates non-finite values
  crypto/
    ed25519.py         pure-Python Edwards25519 arithmetic
    vrf.py             three VRF backends (see below)
    merkle.py          domain-separated binary Merkle tree
    hashing.py         SHA-256 and the difficulty/target encoding
  tasks/
    tsp.py             instance generation, three solvers, V(I,w)
    repository.py      Merkle-committed repository, Equation (3)
  chain/
    block.py           Equation (4) block structure
    miner.py           Algorithm 1
    validator.py       Algorithm 2
    network.py         discrete-event race and simulation loop
  metrics/             fairness, timing, convergence — from PAPER_SPEC.md §9
  experiments/         one module per table/ablation
  reporting/           JSON/CSV, LaTeX, matplotlib
configs/               all hyperparameters; no magic numbers in code
tests/                 244 tests
```

### The three VRF backends

The paper says only "deterministic VRF-like signing and verification" (L405) over "an
ECDSA library" (L412) and never names a construction. `PAPER_SPEC.md` §11-I6 explains
why that gap is load-bearing, so all three readings are implemented and selectable via
`vrf.backend`:

| Backend | What it is | Unique output? |
|---|---|---|
| `ecvrf` | RFC 9381 ECVRF-EDWARDS25519-SHA512-TAI | **Yes** |
| `ecdsa_pseudo` | ECDSA "VRF-like signing", the paper-faithful reading | No — malleable nonce |
| `hash_stub` | keyless hash chain matching the paper's reported latency | No — verifies nothing |

`ablation_vrf_grinding` measures the consequence: with a real VRF, one paid PoW seed
yields exactly one task; with the malleable backend, one seed yields many, and every
forged proof passes validation.

## Data

No external dataset. All TSP instances are generated deterministically from the config
(`repository.seed`, `coordinate_min`/`coordinate_max`, `instance_size`), matching the
paper's description of JSON instances with float x/y coordinates on a 2D Euclidean
plane (L408). The paper does not state the coordinate range, distribution, or seed
(`PAPER_SPEC.md` §11-U2); uniform on [0, 100]² is the assumption here.

Nothing is downloaded, and there is no licensing obstacle to redistribution. A
repository can be materialised to disk with `ProblemRepository.save()` and reloaded
with `ProblemRepository.load()` if you want to pin exact instances.

## Tests

```bash
pytest                    # 244 tests, ~25 s
pytest -m "not slow"      # unit tests only, ~3 s
pytest tests/test_metrics.py -v
```

Coverage of note:

- **Metrics pinned to the paper's own numbers.** The paper's Table 5 is internally
  self-consistent, so its reported std dev (7.6420), max/min ratio (1.2198) and Jain
  index (0.9942) are used as ground truth. This also pins the `ddof=0` convention,
  which the paper does not state.
- **Adversarial validation tests.** Each tampers with exactly one block field and
  asserts the specific check that must reject it — forged VRF proof, proof from
  another miner's key, wrong instance index, substituted instance, infeasible tour,
  understated cost.
- **Cross-solver agreement.** Brute force and Held-Karp are both exact, so any
  disagreement is a bug in one of them; they are checked against each other and
  against hand-computed optima (unit square = 4.0, 3-4-5 triangle = 12.0).
- **Determinism.** Same seed ⇒ identical block hashes, task assignments and solution
  costs; different seeds ⇒ different chains (so a seed that was silently ignored would
  fail).
- **Sanity checks.** Useful-work share falls monotonically with difficulty, solve time
  grows with N, tours enumerated equal `(N-1)!` exactly, all nodes converge, solutions
  are verified optimal against an independent algorithm.

## Relationship to the manuscript

This package does not modify `sn-article.tex`. LaTeX tables are written to
`results/latex/` for the author to review and paste. They use the same `booktabs`
formatting, column specs, and `\label` names as the manuscript, so they are drop-in
replacements — but the numbers will differ from the printed ones wherever
`PAPER_SPEC.md` §11 records an inconsistency, and those differences are the point.
