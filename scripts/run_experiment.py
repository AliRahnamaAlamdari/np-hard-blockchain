#!/usr/bin/env python3
"""Run one experiment from its config.

Examples::

    python scripts/run_experiment.py table3_time_allocation
    python scripts/run_experiment.py table5_fairness --seeds 0 1 2
    python scripts/run_experiment.py ablation_vrf_grinding --config configs/smoke.yaml
    python scripts/run_experiment.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pouw.config import load_config  # noqa: E402
from pouw.experiments import PAPER_MAP, REGISTRY  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a single PoUW reproduction experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("experiment", nargs="?", help="experiment name (see --list)")
    parser.add_argument(
        "--config",
        help="config path; defaults to configs/experiments/<experiment>.yaml",
    )
    parser.add_argument("--seeds", nargs="+", type=int, help="override run.seeds")
    parser.add_argument("--blocks", type=int, help="override network.num_blocks")
    parser.add_argument("--miners", type=int, help="override network.num_miners")
    parser.add_argument("--difficulty-bits", type=int, help="override pow.difficulty_bits")
    parser.add_argument("--vrf", help="override vrf.backend")
    parser.add_argument("--solver", help="override solver.name")
    parser.add_argument("--repetitions", type=int, help="override run.repetitions")
    parser.add_argument("--output-dir", help="override run.output_dir")
    parser.add_argument("--log-level", default=None, help="override run.log_level")
    parser.add_argument("--list", action="store_true", help="list experiments and exit")
    return parser


def collect_overrides(args: argparse.Namespace) -> dict:
    overrides: dict = {}
    mapping = {
        "seeds": ("run", "seeds"),
        "blocks": ("network", "num_blocks"),
        "miners": ("network", "num_miners"),
        "difficulty_bits": ("pow", "difficulty_bits"),
        "vrf": ("vrf", "backend"),
        "solver": ("solver", "name"),
        "repetitions": ("run", "repetitions"),
        "output_dir": ("run", "output_dir"),
        "log_level": ("run", "log_level"),
    }
    for attribute, (section, key) in mapping.items():
        value = getattr(args, attribute, None)
        if value is not None:
            overrides.setdefault(section, {})[key] = value
    return overrides


def main() -> int:
    args = build_parser().parse_args()

    if args.list or not args.experiment:
        width = max(len(name) for name in REGISTRY)
        print("Available experiments:\n")
        for name in REGISTRY:
            print(f"  {name:<{width}}  {PAPER_MAP.get(name, '')}")
        print("\nRun with: python scripts/run_experiment.py <name>")
        return 0 if args.list else 1

    if args.experiment not in REGISTRY:
        print(f"unknown experiment {args.experiment!r}; use --list", file=sys.stderr)
        return 2

    config_path = args.config or f"configs/experiments/{args.experiment}.yaml"
    config = load_config(config_path, overrides=collect_overrides(args))
    experiment = REGISTRY[args.experiment](config)
    output = experiment.execute()

    print(f"\n{output.name} complete. Files written:")
    for path in output.files:
        print(f"  {path}")
    for figure in output.payload.get("figures", []):
        print(f"  {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
