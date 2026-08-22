"""Small helpers for writing reproducible experiment artefacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(value: Any) -> Any:
    """Convert NumPy scalars/arrays and Paths into JSON-serialisable values."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    """Write ``payload`` as JSON, creating parent directories as needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=indent, sort_keys=False)
    return out


def read_json(path: str | Path) -> Any:
    """Read a JSON file written by :func:`write_json`."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp for run provenance."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
