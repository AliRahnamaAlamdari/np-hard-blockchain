"""Reproduction package for the two-stage VRF-based PoUW task assignment framework.

See ``PAPER_SPEC.md`` for the complete extraction of the paper this implements, and
``README.md`` for the mapping from each paper table to the command that regenerates it.
"""

from __future__ import annotations

__version__ = "1.0.0"

from .config import ExperimentConfig, load_config
from .seeding import SeedBundle, seed_everything

__all__ = [
    "ExperimentConfig",
    "SeedBundle",
    "__version__",
    "load_config",
    "seed_everything",
]
