"""Typed configuration loaded from YAML.  No experiment constant lives in code.

Every value that the paper states, and every value the paper leaves open, is a field
here.  Configs compose by shallow merge: a config may name a ``base`` file, and its
own keys override the base's.  The fully resolved dict is written into every result
file so any number can be traced back to the exact parameters that produced it.
"""

from __future__ import annotations

import copy
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


@dataclass
class PoWConfig:
    """Stage-1 lightweight proof-of-work gate."""

    difficulty_bits: int = 16
    max_attempts: int = 100_000_000


@dataclass
class VRFConfig:
    """Stage-2 task assignment primitive."""

    backend: str = "ecvrf"  # ecvrf | ecdsa_pseudo | hash_stub
    curve: str = "SECP256k1"  # only used by ecdsa_pseudo

    def build_kwargs(self) -> dict[str, Any]:
        return {"curve_name": self.curve} if self.backend == "ecdsa_pseudo" else {}


@dataclass
class RepositoryConfig:
    """The Merkle-committed problem repository."""

    num_instances: int = 256
    instance_size: int = 10
    seed: int = 20260722
    coordinate_min: float = 0.0
    coordinate_max: float = 100.0
    difficulty_class: str | None = None

    @property
    def coordinate_range(self) -> tuple[float, float]:
        return (self.coordinate_min, self.coordinate_max)


@dataclass
class SolverConfig:
    """Useful-work solver and the acceptance rule applied by validators."""

    name: str = "brute_force"  # brute_force | held_karp | nearest_neighbor_2opt
    quality_threshold: float | None = None  # None => exact optimum required
    cost_tolerance: float = 1e-9


@dataclass
class NetworkConfig:
    """Simulated peer-to-peer network."""

    num_miners: int = 5
    num_blocks: int = 500
    hash_rates: list[float] | None = None  # None => all miners equal
    propagation_delay_s: float = 0.0
    tie_break: str = "lowest_index"


@dataclass
class RunConfig:
    """Execution control: seeds, output, logging, checkpointing."""

    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    output_dir: str = "results"
    log_level: str = "INFO"
    log_every_blocks: int = 50
    checkpoint_every_blocks: int = 0  # 0 disables checkpointing
    track_memory: bool = True
    repetitions: int = 50  # paper L408: 50 runs per instance size


@dataclass
class ExperimentConfig:
    """Top-level config object handed to every experiment."""

    name: str = "unnamed"
    description: str = ""
    pow: PoWConfig = field(default_factory=PoWConfig)
    vrf: VRFConfig = field(default_factory=VRFConfig)
    repository: RepositoryConfig = field(default_factory=RepositoryConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    run: RunConfig = field(default_factory=RunConfig)
    sweep: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge; ``override`` wins at every leaf."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _nested_dataclass_types(cls: type) -> dict[str, type]:
    """Map field name -> nested dataclass type for the fields that have one.

    ``from __future__ import annotations`` turns every field's ``.type`` into a
    string, so the annotation has to be resolved before it can be tested.  Falling
    back to the field's default factory keeps this working even if resolution fails.
    """
    resolved: dict[str, Any] = {}
    try:
        resolved = get_type_hints(cls)
    except Exception:  # pragma: no cover - only on exotic forward references
        resolved = {}

    nested: dict[str, type] = {}
    for field_info in fields(cls):
        candidate = resolved.get(field_info.name)
        if isinstance(candidate, type) and is_dataclass(candidate):
            nested[field_info.name] = candidate
            continue
        factory = field_info.default_factory  # type: ignore[misc]
        if factory is not MISSING and isinstance(factory, type) and is_dataclass(factory):
            nested[field_info.name] = factory
    return nested


def _build(cls: type, payload: dict[str, Any]) -> Any:
    """Instantiate a (possibly nested) dataclass from a plain dict, checking keys."""
    if not is_dataclass(cls):
        return payload
    known = {f.name: f for f in fields(cls)}
    unknown = set(payload) - set(known)
    if unknown:
        raise ValueError(
            f"unknown config key(s) for {cls.__name__}: {sorted(unknown)}; "
            f"expected {sorted(known)}"
        )
    nested = _nested_dataclass_types(cls)
    kwargs: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict) and key in nested:
            kwargs[key] = _build(nested[key], value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    for root in (CONFIG_ROOT, CONFIG_ROOT / "experiments"):
        if (root / candidate).exists():
            return root / candidate
        if (root / f"{candidate}.yaml").exists():
            return root / f"{candidate}.yaml"
    raise FileNotFoundError(f"config not found: {path}")


def load_raw(path: str | Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Load a YAML config, resolving its ``base`` chain."""
    resolved = _resolve_path(path).resolve()
    seen = _seen or set()
    if resolved in seen:
        raise ValueError(f"circular config inheritance at {resolved}")
    seen.add(resolved)
    payload = yaml.safe_load(resolved.read_text()) or {}
    base_name = payload.pop("base", None)
    if base_name is None:
        return payload
    base_path = _resolve_path(base_name)
    if not base_path.is_absolute():
        base_path = (resolved.parent / base_name) if (resolved.parent / base_name).exists() else base_path
    return _merge(load_raw(base_path, seen), payload)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    """Load and validate a config, applying optional CLI overrides last."""
    payload = load_raw(path)
    if overrides:
        payload = _merge(payload, overrides)
    config = _build(ExperimentConfig, payload)
    validate(config)
    return config


def validate(config: ExperimentConfig) -> None:
    """Reject configurations that are internally impossible before any run starts."""
    from .crypto.hashing import DIGEST_BITS
    from .crypto.vrf import _BACKENDS
    from .tasks.tsp import SOLVERS

    if not 0 <= config.pow.difficulty_bits <= DIGEST_BITS:
        raise ValueError(f"pow.difficulty_bits out of range: {config.pow.difficulty_bits}")
    if config.vrf.backend not in _BACKENDS:
        raise ValueError(
            f"vrf.backend must be one of {sorted(_BACKENDS)}, got {config.vrf.backend!r}"
        )
    if config.solver.name not in SOLVERS:
        raise ValueError(
            f"solver.name must be one of {sorted(SOLVERS)}, got {config.solver.name!r}"
        )
    if config.repository.num_instances < 1:
        raise ValueError("repository.num_instances must be >= 1")
    if config.repository.instance_size < 3:
        raise ValueError("repository.instance_size must be >= 3")
    if config.network.num_miners < 1:
        raise ValueError("network.num_miners must be >= 1")
    if config.network.num_blocks < 1:
        raise ValueError("network.num_blocks must be >= 1")
    if config.network.hash_rates is not None:
        if len(config.network.hash_rates) != config.network.num_miners:
            raise ValueError(
                f"network.hash_rates has {len(config.network.hash_rates)} entries but "
                f"num_miners is {config.network.num_miners}"
            )
        if any(rate <= 0 for rate in config.network.hash_rates):
            raise ValueError("network.hash_rates entries must be positive")
    if not config.run.seeds:
        raise ValueError("run.seeds must contain at least one seed")
    if config.solver.quality_threshold is not None and config.solver.quality_threshold < 0:
        raise ValueError("solver.quality_threshold must be non-negative")
