"""Logging, wall-clock timing, and memory tracking.

Timing uses ``time.perf_counter`` (monotonic, highest available resolution).  Process
memory is sampled through ``psutil`` when available and falls back to ``resource``,
so the memory columns degrade gracefully rather than failing a run.
"""

from __future__ import annotations

import logging
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Install a console handler, plus a file handler when ``log_file`` is given."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=LOG_FORMAT,
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def current_memory_mb() -> float:
    """Resident set size in MiB, or 0.0 if it cannot be determined."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:  # pragma: no cover - platform dependent
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux reports KiB, macOS reports bytes.
            return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024
        except Exception:
            return 0.0


@dataclass
class Timer:
    """Accumulates named wall-clock durations across many calls.

    Every stage of mining and validation is timed through this object, which is what
    produces the time-allocation and validation-cost tables.
    """

    totals: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    samples: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    keep_samples: bool = True

    @contextmanager
    def measure(self, label: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(label, time.perf_counter() - start)

    def record(self, label: str, seconds: float) -> None:
        self.totals[label] += seconds
        self.counts[label] += 1
        if self.keep_samples:
            self.samples[label].append(seconds)

    def total(self, label: str) -> float:
        return self.totals.get(label, 0.0)

    def mean(self, label: str) -> float:
        count = self.counts.get(label, 0)
        return self.totals[label] / count if count else 0.0

    def merge(self, other: "Timer") -> None:
        for label, value in other.totals.items():
            self.totals[label] += value
            self.counts[label] += other.counts[label]
            if self.keep_samples:
                self.samples[label].extend(other.samples.get(label, []))

    def reset(self) -> None:
        self.totals.clear()
        self.counts.clear()
        self.samples.clear()

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            label: {
                "total_s": total,
                "count": self.counts[label],
                "mean_s": self.mean(label),
            }
            for label, total in sorted(self.totals.items())
        }


@contextmanager
def timed(label: str, logger: logging.Logger | None = None) -> Iterator[dict[str, float]]:
    """Time a block of work and report elapsed seconds and memory delta."""
    result = {"seconds": 0.0, "memory_delta_mb": 0.0}
    memory_before = current_memory_mb()
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start
        result["memory_delta_mb"] = current_memory_mb() - memory_before
        if logger is not None:
            logger.info(
                "%s finished in %.4f s (memory delta %+.1f MiB)",
                label,
                result["seconds"],
                result["memory_delta_mb"],
            )
