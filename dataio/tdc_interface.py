"""Interface to Turing Synthetic Radar Dataset (TSRD) pulse trains.

The TSRD stores each pulse train as an HDF5 file holding

* ``data``     -- ``(seq_len, num_features)`` float32 PDW stream,
* ``labels``   -- ``(seq_len,)`` per-pulse-train-local emitter labels (optional),
* ``metadata`` -- a nested group of attributes/datasets.

If the official ``turing_deinterleaving_challenge`` package is installed we use its
``PulseTrain.load``. If it is not (it pulls in torch/jaxtyping, which the MVP does not
need), we read the identical file layout directly with :mod:`h5py`. Either way callers
see the same :class:`PulseTrainRecord`, so nothing downstream changes when the real
package or the real dataset becomes available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from common.logging_utils import get_logger

LOGGER = get_logger(__name__)

#: Canonical PDW field order used throughout the project.
CANONICAL_FIELDS: tuple[str, ...] = ("toa", "cf", "pw", "aoa", "amplitude")

#: Aliases seen in TSRD metadata / the challenge README mapped onto canonical names.
_FIELD_ALIASES: dict[str, str] = {
    "toa": "toa",
    "time_of_arrival": "toa",
    "time": "toa",
    "t": "toa",
    "cf": "cf",
    "centre_frequency": "cf",
    "center_frequency": "cf",
    "frequency": "cf",
    "freq": "cf",
    "f": "cf",
    "pw": "pw",
    "pulse_width": "pw",
    # The real TSRD writes this one as a single word ("PulseWidth"); the underscored
    # spelling below never appears in the released files.
    "pulsewidth": "pw",
    "width": "pw",
    "aoa": "aoa",
    "angle_of_arrival": "aoa",
    "angleofarrival": "aoa",
    "angle": "aoa",
    "bearing": "aoa",
    "amplitude": "amplitude",
    "amp": "amplitude",
    "power": "amplitude",
    "pulse_amplitude": "amplitude",
    "pulseamplitude": "amplitude",
    "timeofarrival": "toa",
    "centrefrequency": "cf",
    "centerfrequency": "cf",
}


def _decode(value: Any) -> Any:
    """Decode HDF5 byte strings (and arrays of them) into Python strings."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "S":
            return [v.decode("utf-8", errors="replace") for v in value.tolist()]
        if value.dtype.kind == "O":
            return [_decode(v) for v in value.tolist()]
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _read_group(group: h5py.Group) -> dict[str, Any]:
    """Recursively read an HDF5 group into a plain nested dictionary."""
    out: dict[str, Any] = {}
    for key, value in group.attrs.items():
        out[str(key)] = _decode(value)
    for key in group:
        item = group[key]
        if isinstance(item, h5py.Group):
            out[str(key)] = _read_group(item)
        else:
            out[str(key)] = _decode(item[()])
    return out


def normalise_feature_names(names: Iterable[Any] | None, n_features: int) -> list[str]:
    """Map raw TSRD feature names onto :data:`CANONICAL_FIELDS`.

    Falls back to the canonical order when the metadata does not name the columns,
    which is logged loudly because every downstream binning step depends on it.

    Args:
        names: raw feature names from the file metadata, or ``None``.
        n_features: width of the PDW data array.

    Returns:
        A list of column names aligned with the data array.
    """
    if names is None:
        LOGGER.warning(
            "Pulse train metadata carries no feature_names; assuming canonical order %s",
            CANONICAL_FIELDS[:n_features],
        )
        return list(CANONICAL_FIELDS[:n_features])

    raw = [str(_decode(n)).strip().lower().replace(" ", "_") for n in names]
    mapped = [_FIELD_ALIASES.get(name, name) for name in raw]
    if len(mapped) != n_features:
        LOGGER.warning(
            "feature_names length %d != data width %d; falling back to canonical order",
            len(mapped),
            n_features,
        )
        return list(CANONICAL_FIELDS[:n_features])
    unknown = [n for n in mapped if n not in CANONICAL_FIELDS]
    if unknown:
        LOGGER.warning("Unrecognised PDW field names kept verbatim: %s", unknown)
    return mapped


@dataclass
class PulseTrainRecord:
    """One TSRD pulse train, normalised for this project.

    Attributes:
        data: ``(n_pulses, n_features)`` PDW stream.
        labels: per-pulse emitter labels, local to this pulse train only, or ``None``.
        metadata: nested metadata dictionary as stored in the file.
        feature_names: canonical column names aligned with ``data``.
        source_path: file the record was loaded from (``None`` for in-memory records).
    """

    data: np.ndarray
    labels: np.ndarray | None
    metadata: dict[str, Any] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=lambda: list(CANONICAL_FIELDS))
    source_path: Path | None = None

    @property
    def n_pulses(self) -> int:
        """Number of pulses in this record."""
        return int(self.data.shape[0])

    @property
    def name(self) -> str:
        """Short identifier used in logs and manifests."""
        return self.source_path.stem if self.source_path is not None else "in-memory"

    def has(self, field_name: str) -> bool:
        """Whether a canonical PDW field is present in this record."""
        return field_name in self.feature_names

    def column(self, field_name: str) -> np.ndarray:
        """Return one PDW field as a 1-D float array.

        Raises:
            KeyError: if the field is not present in this pulse train.
        """
        if field_name not in self.feature_names:
            msg = (
                f"PDW field {field_name!r} not in pulse train {self.name!r} "
                f"(has {self.feature_names})"
            )
            raise KeyError(msg)
        index = self.feature_names.index(field_name)
        return np.asarray(self.data[:, index], dtype=np.float64)

    def sorted_by_toa(self) -> "PulseTrainRecord":
        """Return a record with pulses in non-decreasing ToA order."""
        toa = self.column("toa")
        if toa.size == 0 or bool(np.all(np.diff(toa) >= 0)):
            return self
        order = np.argsort(toa, kind="stable")
        return PulseTrainRecord(
            data=self.data[order],
            labels=None if self.labels is None else self.labels[order],
            metadata=self.metadata,
            feature_names=list(self.feature_names),
            source_path=self.source_path,
        )

    def contiguous_window(self, max_pulses: int) -> "PulseTrainRecord":
        """Take one contiguous ToA-ordered slice of at most ``max_pulses`` pulses.

        A contiguous slice preserves PRI and periodicity structure. A scattered
        sample of individual pulses would destroy exactly the structure the
        scheduler's periodicity features depend on.
        """
        ordered = self.sorted_by_toa()
        if max_pulses <= 0 or ordered.n_pulses <= max_pulses:
            return ordered
        return PulseTrainRecord(
            data=ordered.data[:max_pulses],
            labels=None if ordered.labels is None else ordered.labels[:max_pulses],
            metadata=ordered.metadata,
            feature_names=list(ordered.feature_names),
            source_path=ordered.source_path,
        )

    def save(self, path: str | Path) -> Path:
        """Write this record out in the TSRD HDF5 layout."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        metadata = {k: v for k, v in self.metadata.items() if k != "feature_names"}
        with h5py.File(out, "w") as handle:
            meta_group = handle.create_group("metadata")
            _write_metadata(meta_group, metadata)
            meta_group.create_dataset(
                "feature_names", data=np.array(self.feature_names, dtype="S")
            )
            handle.create_dataset(
                "data",
                data=self.data.astype(np.float32),
                compression="gzip",
                compression_opts=4,
            )
            if self.labels is not None:
                handle.create_dataset(
                    "labels",
                    data=np.asarray(self.labels).astype(np.int16),
                    compression="gzip",
                    compression_opts=4,
                )
        return out


def _write_metadata(group: h5py.Group, metadata: dict[str, Any]) -> None:
    """Write a nested metadata dictionary into an HDF5 group."""
    for key, value in metadata.items():
        name = str(key)
        if isinstance(value, dict):
            _write_metadata(group.create_group(name), value)
        elif isinstance(value, (list, tuple)):
            items = list(value)
            if items and isinstance(items[0], dict):
                sub = group.create_group(name)
                for index, item in enumerate(items):
                    _write_metadata(sub.create_group(f"{name}_{index}"), item)
            elif items and isinstance(items[0], str):
                group.create_dataset(name, data=np.array(items, dtype="S"))
            elif items:
                group.create_dataset(name, data=np.asarray(items))
        elif isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_)):
            group.attrs[name] = value
        elif isinstance(value, np.ndarray):
            group.create_dataset(name, data=value)
        elif value is None:
            continue
        else:
            group.attrs[name] = str(value)


def official_package_available() -> bool:
    """Whether the official ``turing_deinterleaving_challenge`` package is importable."""
    try:  # pragma: no cover - depends on the local environment
        import turing_deinterleaving_challenge  # noqa: F401
    except Exception:
        return False
    return True


def _load_via_official_package(path: Path) -> PulseTrainRecord | None:
    """Load with the official ``PulseTrain`` class, or return ``None`` if unavailable."""
    try:  # pragma: no cover - depends on the local environment
        from turing_deinterleaving_challenge import PulseTrain
    except Exception:
        return None
    try:  # pragma: no cover - depends on the local environment
        pulse_train = PulseTrain.load(path)
    except Exception as exc:
        LOGGER.warning(
            "Official PulseTrain.load failed for %s (%s); using the h5py reader", path, exc
        )
        return None

    metadata_obj = getattr(pulse_train, "metadata", None)
    metadata: dict[str, Any] = {}
    for attr in (
        "feature_names",
        "type",
        "receiver",
        "transmitters",
        "description",
        "collection_time_s",
        "num_pulses",
        "date_created",
    ):
        if hasattr(metadata_obj, attr):
            metadata[attr] = _decode(getattr(metadata_obj, attr))

    data = np.asarray(pulse_train.data, dtype=np.float64)
    labels = getattr(pulse_train, "labels", None)
    return PulseTrainRecord(
        data=data,
        labels=None if labels is None else np.asarray(labels).squeeze(),
        metadata=metadata,
        feature_names=normalise_feature_names(metadata.get("feature_names"), data.shape[1]),
        source_path=Path(path),
    )


def load_pulse_train(path: str | Path, *, prefer_official: bool = True) -> PulseTrainRecord:
    """Load one TSRD pulse train file.

    Args:
        path: path to a ``.h5`` pulse train.
        prefer_official: try the official package's loader first when it is installed.

    Returns:
        The pulse train as a :class:`PulseTrainRecord`.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file is not in the TSRD layout.
    """
    file_path = Path(path)
    if not file_path.exists():
        msg = f"Pulse train file not found: {file_path}"
        raise FileNotFoundError(msg)

    if prefer_official:
        record = _load_via_official_package(file_path)
        if record is not None:
            return record

    with h5py.File(file_path, "r") as handle:
        if "data" not in handle:
            msg = f"{file_path} is not a TSRD pulse train (no 'data' dataset)"
            raise ValueError(msg)
        data = np.asarray(handle["data"][:], dtype=np.float64)
        labels = np.asarray(handle["labels"][:]).squeeze() if "labels" in handle else None
        metadata = _read_group(handle["metadata"]) if "metadata" in handle else {}

    return PulseTrainRecord(
        data=data,
        labels=labels,
        metadata=metadata,
        feature_names=normalise_feature_names(metadata.get("feature_names"), data.shape[1]),
        source_path=file_path,
    )


def peek_pulse_train(path: str | Path) -> dict[str, Any]:
    """Read shape and metadata of a pulse train without loading the whole PDW stream.

    Used by Pass 1 of the sampler, which inspects many files cheaply.

    Args:
        path: path to a ``.h5`` pulse train.

    Returns:
        Dictionary with pulse/emitter counts, file size and raw metadata.
    """
    file_path = Path(path)
    with h5py.File(file_path, "r") as handle:
        shape = tuple(handle["data"].shape) if "data" in handle else (0, 0)
        metadata = _read_group(handle["metadata"]) if "metadata" in handle else {}
        n_emitters = None
        if "labels" in handle:
            labels = np.asarray(handle["labels"][:]).squeeze()
            n_emitters = int(np.unique(labels).size)
    return {
        "path": str(file_path),
        "name": file_path.stem,
        "n_pulses": int(shape[0]),
        "n_features": int(shape[1]) if len(shape) > 1 else 0,
        "n_emitters": n_emitters,
        "metadata": metadata,
        "file_bytes": int(file_path.stat().st_size),
    }
