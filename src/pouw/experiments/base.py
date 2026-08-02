"""Shared scaffolding for every experiment.

An experiment declares a name and a ``run`` method that returns a plain dict.  The
base class handles seeding, logging, timing, memory tracking, checkpoint paths, and
writing results with a provenance manifest, so the experiment bodies contain only
their measurement logic.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig
from ..logging_utils import configure_logging, current_memory_mb, get_logger, timed
from ..reporting.io import results_dir, write_csv, write_json
from ..seeding import SeedBundle, seed_everything
from ..tasks.repository import ProblemRepository, RepositorySpec


@dataclass
class ExperimentOutput:
    """What an experiment produces: a payload plus the files it wrote."""

    name: str
    payload: dict[str, Any]
    files: list[Path]

    def add_file(self, path: Path) -> None:
        self.files.append(path)


class Experiment(abc.ABC):
    """Base class for all experiments."""

    name: str = "experiment"
    description: str = ""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.logger = get_logger(f"pouw.{self.name}")
        self.output_dir = results_dir(config.run.output_dir, self.name)

    # -- helpers available to subclasses -------------------------------------

    def build_repository(
        self,
        seeds: SeedBundle,
        instance_size: int | None = None,
        num_instances: int | None = None,
    ) -> ProblemRepository:
        """Construct the Merkle-committed repository for one run."""
        repo_config = self.config.repository
        spec = RepositorySpec(
            num_instances=num_instances or repo_config.num_instances,
            instance_size=instance_size or repo_config.instance_size,
            seed=seeds.integer("repository") % (2**31),
            coordinate_range=repo_config.coordinate_range,
            difficulty_class=repo_config.difficulty_class,
        )
        repository = ProblemRepository.from_spec(spec)
        self.logger.debug(
            "repository: %d instances of size %d, root %s",
            len(repository),
            spec.instance_size,
            repository.root_hex[:16],
        )
        return repository

    def seed_bundles(self) -> list[SeedBundle]:
        return [seed_everything(seed) for seed in self.config.run.seeds]

    def checkpoint_file(self, seed: int) -> Path:
        return self.output_dir / f"{self.name}_seed{seed}.checkpoint.json"

    # -- lifecycle -----------------------------------------------------------

    @abc.abstractmethod
    def run(self) -> dict[str, Any]:
        """Execute the experiment and return its result payload."""

    def execute(self) -> ExperimentOutput:
        """Run with logging, timing and memory tracking, then serialise results."""
        configure_logging(
            level=self.config.run.log_level,
            log_file=self.output_dir / f"{self.name}.log",
        )
        self.logger.info("=== %s ===", self.name)
        if self.description:
            self.logger.info("%s", self.description)
        self.logger.info(
            "seeds=%s | vrf=%s | solver=%s | difficulty=%d bits",
            self.config.run.seeds,
            self.config.vrf.backend,
            self.config.solver.name,
            self.config.pow.difficulty_bits,
        )

        memory_before = current_memory_mb()
        with timed(self.name, self.logger) as elapsed:
            payload = self.run()

        payload.setdefault("meta", {})
        payload["meta"].update(
            {
                "experiment": self.name,
                "description": self.description,
                "wall_seconds": elapsed["seconds"],
                "memory_before_mb": memory_before,
                "memory_after_mb": current_memory_mb(),
                "memory_delta_mb": elapsed["memory_delta_mb"],
                "seeds": list(self.config.run.seeds),
            }
        )

        output = ExperimentOutput(name=self.name, payload=payload, files=[])
        json_path = write_json(
            self.output_dir / f"{self.name}.json", payload, config=self.config
        )
        output.add_file(json_path)
        for table_name, rows in payload.get("csv_tables", {}).items():
            if rows:
                output.add_file(
                    write_csv(self.output_dir / f"{table_name}.csv", rows)
                )
        self.logger.info(
            "wrote %d file(s) to %s in %.2f s",
            len(output.files),
            self.output_dir,
            elapsed["seconds"],
        )
        return output


from ..stats_utils import mean_std  # noqa: E402  (re-exported for experiments)

__all__ = ["Experiment", "ExperimentOutput", "mean_std"]
