"""Configuration loading and seeding helpers.

Every script in this project takes ``--config config.yaml``; nothing reads a hardcoded
path, size or model parameter.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


class Config:
    """Thin dotted-path accessor over the parsed YAML configuration."""

    def __init__(self, raw: dict[str, Any], path: Path | None = None) -> None:
        self.raw = raw
        self.path = path

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Load a configuration file, defaulting to ``<repo>/config.yaml``."""
        cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        if not cfg_path.is_absolute():
            cfg_path = (Path.cwd() / cfg_path).resolve()
        if not cfg_path.exists():
            msg = f"Configuration file not found: {cfg_path}"
            raise FileNotFoundError(msg)
        with cfg_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls(raw=raw, path=cfg_path)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Return ``config['a']['b']`` for ``dotted_key='a.b'``."""
        node: Any = self.raw
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted_key: str) -> Any:
        """Like :meth:`get`, but raises when the key is absent."""
        sentinel = object()
        value = self.get(dotted_key, sentinel)
        if value is sentinel:
            msg = f"Missing required configuration key: {dotted_key}"
            raise KeyError(msg)
        return value

    def section(self, name: str) -> dict[str, Any]:
        """Return a whole top-level section as a plain dict."""
        value = self.get(name, {})
        return dict(value) if isinstance(value, dict) else {}

    def path_for(self, dotted_key: str) -> Path:
        """Resolve a configured relative path against the repository root."""
        value = Path(str(self.require(dotted_key)))
        return value if value.is_absolute() else (REPO_ROOT / value)

    @property
    def seed(self) -> int:
        return int(self.get("random_seed", 42))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Config(path={self.path}, sections={sorted(self.raw)})"


@dataclass(frozen=True)
class SeedBundle:
    """Seeds actually applied, recorded alongside every experiment result."""

    seed: int
    numpy_seed: int
    python_seed: int


def set_global_seed(seed: int) -> SeedBundle:
    """Seed ``random`` and NumPy's legacy global state.

    Components that need independent, reproducible streams take their own
    ``numpy.random.Generator`` built from this seed rather than relying on global
    state; this function exists so third-party libraries behave reproducibly too.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    return SeedBundle(seed=seed, numpy_seed=seed % (2**32), python_seed=seed)


def make_rng(seed: int, stream: int = 0) -> np.random.Generator:
    """Return an independent generator for a named stream of the same experiment."""
    return np.random.default_rng([seed, stream])
