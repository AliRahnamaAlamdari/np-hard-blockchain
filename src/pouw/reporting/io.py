"""Result serialisation to JSON and CSV, with a provenance manifest on every file.

Each result file records the resolved config, the seeds, the git-less environment
fingerprint, and a timestamp, so any number in a regenerated table can be traced to
the run that produced it.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from dataclasses import is_dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def environment_fingerprint() -> dict[str, Any]:
    """Machine and interpreter details that materially affect timing results."""
    info: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "machine": platform.machine(),
    }
    try:
        import psutil

        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        info["total_memory_gb"] = round(psutil.virtual_memory().total / 1024**3, 2)
    except Exception:  # pragma: no cover - psutil is optional at runtime
        pass
    for module_name in ("numpy", "scipy", "matplotlib", "ecdsa", "yaml"):
        try:
            module = __import__(module_name)
            info[f"{module_name}_version"] = getattr(module, "__version__", "unknown")
        except ImportError:
            info[f"{module_name}_version"] = "absent"
    return info


def _jsonable(value: Any) -> Any:
    """Recursively convert dataclasses, bytes and tuples into JSON-safe values."""
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return str(value)
    return value


def write_json(path: Path, payload: Mapping[str, Any], config: Any = None) -> Path:
    """Write a result payload plus its provenance manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "manifest": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "environment": environment_fingerprint(),
            "config": _jsonable(config.to_dict() if hasattr(config, "to_dict") else config),
        },
        "results": _jsonable(payload),
    }
    path.write_text(json.dumps(document, indent=2))
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> Path:
    """Write rows to CSV.  Column order is taken from ``columns`` or the first row."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return path
    fieldnames = list(columns) if columns else list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key)) for key in fieldnames})
    return path


def results_dir(base: str | Path, experiment: str) -> Path:
    path = Path(base) / experiment
    path.mkdir(parents=True, exist_ok=True)
    return path
