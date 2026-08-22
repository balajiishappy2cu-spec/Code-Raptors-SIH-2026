"""TSRD-format synthetic PDW generator.

The real Turing Synthetic Radar Dataset is gated on Hugging Face and needs a token.
This module generates pulse trains in exactly the same HDF5 layout so that the whole
pipeline -- sampler, environment builder, features, model, schedulers, metrics -- runs
end to end without the real download, and runs unchanged once the real data lands.

Three emitter behaviours are modelled, matching the scenarios the problem statement
names:

* ``static``          -- fixed centre frequency, fixed bearing, continuous emission.
* ``spatial_scan``    -- rotating antenna; the fixed receiver only sees pulses while the
  emitter's mainbeam illuminates it, producing strongly periodic band activity.
* ``frequency_agile`` -- centre frequency hops between channels on a hop period.

This is a research simulation of an electromagnetic environment. Nothing here transmits,
jams or touches RF hardware.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from common.logging_utils import get_logger
from dataio.tdc_interface import CANONICAL_FIELDS, PulseTrainRecord

LOGGER = get_logger(__name__)

BEHAVIOURS: tuple[str, ...] = ("static", "spatial_scan", "frequency_agile")


@dataclass
class EmitterSpec:
    """Parameters of one simulated emitter.

    All fields are written into the pulse train metadata as a transmitter entry, so the
    sampler can stratify on real metadata exactly as it would with the real dataset.
    """

    emitter_id: int
    behaviour: str
    cf_mhz: float
    pri_us: float
    pw_us: float
    aoa_deg: float
    amplitude_db: float
    pri_jitter_frac: float
    duty_on: bool = True
    rotation_period_us: float = 0.0
    beamwidth_deg: float = 0.0
    aoa_rate_deg_per_s: float = 0.0
    hop_period_us: float = 0.0
    hop_channels_mhz: tuple[float, ...] = ()

    @property
    def is_spatially_scanning(self) -> bool:
        """Whether this emitter sweeps in bearing (problem statement scenario 1)."""
        return self.behaviour == "spatial_scan"

    @property
    def is_frequency_agile(self) -> bool:
        """Whether this emitter hops in frequency (problem statement scenario 2)."""
        return self.behaviour == "frequency_agile"

    def to_metadata(self) -> dict[str, Any]:
        """Return a metadata-writable dictionary for this emitter."""
        record = asdict(self)
        record["hop_channels_mhz"] = list(self.hop_channels_mhz)
        record["spatial_scan"] = bool(self.is_spatially_scanning)
        record["frequency_agile"] = bool(self.is_frequency_agile)
        return record


def _uniform(rng: np.random.Generator, bounds: Any) -> float:
    """Draw a uniform sample from a ``[low, high]`` pair."""
    low, high = float(bounds[0]), float(bounds[1])
    return float(rng.uniform(low, high))


def _choose_behaviour(rng: np.random.Generator, mix: dict[str, float]) -> str:
    """Sample an emitter behaviour from a (possibly unnormalised) probability mix."""
    weights = np.array([max(0.0, float(mix.get(name, 0.0))) for name in BEHAVIOURS])
    if weights.sum() <= 0:
        weights = np.ones(len(BEHAVIOURS))
    weights = weights / weights.sum()
    return str(rng.choice(np.array(BEHAVIOURS), p=weights))


def sample_emitters(
    rng: np.random.Generator,
    mock_cfg: dict[str, Any],
    n_emitters: int,
) -> list[EmitterSpec]:
    """Draw a set of emitter specifications for one pulse train.

    Args:
        rng: seeded generator for this pulse train.
        mock_cfg: the ``mock`` section of the configuration.
        n_emitters: how many emitters to place in the environment.

    Returns:
        A list of :class:`EmitterSpec`.
    """
    cf_range = mock_cfg.get("cf_range_mhz", [2000.0, 18000.0])
    pri_range = mock_cfg.get("pri_range_us", [200.0, 2500.0])
    pw_range = mock_cfg.get("pw_range_us", [0.5, 40.0])
    amp_range = mock_cfg.get("amplitude_range_db", [-95.0, -45.0])
    jitter = float(mock_cfg.get("pri_jitter_frac", 0.03))
    mix = dict(mock_cfg.get("behaviour_mix", {}))
    scan_cfg = dict(mock_cfg.get("spatial_scan", {}))
    agile_cfg = dict(mock_cfg.get("frequency_agile", {}))

    emitters: list[EmitterSpec] = []
    for emitter_id in range(n_emitters):
        behaviour = _choose_behaviour(rng, mix)
        cf = _uniform(rng, cf_range)
        spec = EmitterSpec(
            emitter_id=emitter_id,
            behaviour=behaviour,
            cf_mhz=cf,
            pri_us=_uniform(rng, pri_range),
            pw_us=_uniform(rng, pw_range),
            aoa_deg=float(rng.uniform(-180.0, 180.0)),
            amplitude_db=_uniform(rng, amp_range),
            pri_jitter_frac=jitter,
        )
        if behaviour == "spatial_scan":
            spec.rotation_period_us = _uniform(
                rng, scan_cfg.get("rotation_period_us_range", [1_000_000.0, 4_000_000.0])
            )
            spec.beamwidth_deg = _uniform(rng, scan_cfg.get("beamwidth_deg_range", [4.0, 20.0]))
            spec.aoa_rate_deg_per_s = 360.0 / (spec.rotation_period_us / 1e6)
        elif behaviour == "frequency_agile":
            spec.hop_period_us = _uniform(
                rng, agile_cfg.get("hop_period_us_range", [20_000.0, 400_000.0])
            )
            n_channels_range = agile_cfg.get("n_channels_range", [3, 9])
            n_channels = int(rng.integers(int(n_channels_range[0]), int(n_channels_range[1]) + 1))
            span = float(agile_cfg.get("hop_span_mhz", 4000.0))
            centres = cf + rng.uniform(-span / 2.0, span / 2.0, size=n_channels)
            centres = np.clip(centres, float(cf_range[0]), float(cf_range[1]))
            spec.hop_channels_mhz = tuple(float(c) for c in centres)
        emitters.append(spec)
    return emitters


def _emitter_pulses(
    spec: EmitterSpec,
    rng: np.random.Generator,
    collection_time_us: float,
) -> np.ndarray:
    """Generate the PDW rows emitted by one emitter over the collection window.

    Returns:
        ``(n_pulses, 5)`` array in :data:`CANONICAL_FIELDS` order, or an empty array.
    """
    pri = max(spec.pri_us, 1.0)
    n_nominal = int(collection_time_us / pri)
    if n_nominal <= 1:
        return np.empty((0, len(CANONICAL_FIELDS)), dtype=np.float64)

    index = np.arange(n_nominal, dtype=np.float64)
    jitter = rng.normal(0.0, spec.pri_jitter_frac * pri, size=n_nominal)
    start = float(rng.uniform(0.0, pri))
    toa = start + index * pri + jitter
    toa = toa[(toa >= 0.0) & (toa < collection_time_us)]
    if toa.size == 0:
        return np.empty((0, len(CANONICAL_FIELDS)), dtype=np.float64)

    cf = np.full(toa.shape, spec.cf_mhz, dtype=np.float64)
    aoa = np.full(toa.shape, spec.aoa_deg, dtype=np.float64)
    amplitude = spec.amplitude_db + rng.normal(0.0, 1.0, size=toa.size)

    if spec.is_spatially_scanning and spec.rotation_period_us > 0:
        # The receiver is illuminated only while the rotating mainbeam points at it.
        phase = np.mod(toa, spec.rotation_period_us) / spec.rotation_period_us
        bearing = phase * 360.0 - 180.0
        offset = np.abs(((bearing - spec.aoa_deg + 180.0) % 360.0) - 180.0)
        illuminated = offset <= (spec.beamwidth_deg / 2.0)
        toa, cf, amplitude = toa[illuminated], cf[illuminated], amplitude[illuminated]
        # Measured AoA tracks the emitter's instantaneous bearing while in the beam.
        aoa = bearing[illuminated]
        if toa.size == 0:
            return np.empty((0, len(CANONICAL_FIELDS)), dtype=np.float64)

    if spec.is_frequency_agile and spec.hop_channels_mhz and spec.hop_period_us > 0:
        hop_index = np.floor(toa / spec.hop_period_us).astype(np.int64)
        channels = np.asarray(spec.hop_channels_mhz, dtype=np.float64)
        # Deterministic pseudo-random hop order, reproducible from the emitter id.
        order = np.mod(hop_index * 7919 + spec.emitter_id * 104_729, channels.size)
        cf = channels[order]

    pw = np.full(toa.shape, spec.pw_us, dtype=np.float64) * (
        1.0 + rng.normal(0.0, 0.02, size=toa.size)
    )
    return np.column_stack([toa, cf, pw, aoa, amplitude])


def generate_pulse_train(
    *,
    seed: int,
    mock_cfg: dict[str, Any],
    train_id: int,
    n_emitters: int | None = None,
) -> PulseTrainRecord:
    """Generate one TSRD-format synthetic pulse train.

    Args:
        seed: seed for this pulse train (derived from the global experiment seed).
        mock_cfg: the ``mock`` section of the configuration.
        train_id: index used in the pulse train name and metadata.
        n_emitters: override for the number of emitters; sampled from config if ``None``.

    Returns:
        A :class:`PulseTrainRecord` with data, per-train-local labels and metadata.
    """
    rng = np.random.default_rng([seed, train_id])
    collection_time_us = float(mock_cfg.get("collection_time_us", 10_000_000.0))
    if n_emitters is None:
        lo, hi = mock_cfg.get("n_emitters_range", [12, 34])
        n_emitters = int(rng.integers(int(lo), int(hi) + 1))

    emitters = sample_emitters(rng, mock_cfg, n_emitters)
    blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for spec in emitters:
        pulses = _emitter_pulses(spec, rng, collection_time_us)
        if pulses.shape[0] == 0:
            continue
        blocks.append(pulses)
        labels.append(np.full(pulses.shape[0], spec.emitter_id, dtype=np.int16))

    if blocks:
        data = np.vstack(blocks)
        label_array = np.concatenate(labels)
    else:  # pragma: no cover - only reachable with a degenerate configuration
        data = np.empty((0, len(CANONICAL_FIELDS)), dtype=np.float64)
        label_array = np.empty((0,), dtype=np.int16)

    order = np.argsort(data[:, 0], kind="stable") if data.shape[0] else np.array([], dtype=int)
    data, label_array = data[order], label_array[order]

    # The TSRD stare receiver is described as an oracle "except randomly dropped pulses";
    # this reproduces that caveat so our ground truth is not artificially perfect.
    drop_rate = float(mock_cfg.get("drop_rate", 0.0))
    if 0.0 < drop_rate < 1.0 and data.shape[0]:
        keep = rng.random(data.shape[0]) >= drop_rate
        data, label_array = data[keep], label_array[keep]

    metadata: dict[str, Any] = {
        "type": "synthetic",
        "description": (
            "Synthetic TSRD-format pulse train generated by smart-ew-scan "
            "(mock source; not the real Turing Synthetic Radar Dataset)"
        ),
        "collection_time_s": collection_time_us / 1e6,
        "num_pulses": int(data.shape[0]),
        "date_created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "smart-ew-scan.dataio.mock_generator",
        "generator_seed": int(seed),
        "train_id": int(train_id),
        "receiver": {"mode": "stare", "note": "full-spectrum observation with random drops"},
        "transmitters": [spec.to_metadata() for spec in emitters],
    }

    return PulseTrainRecord(
        data=data,
        labels=label_array,
        metadata=metadata,
        feature_names=list(CANONICAL_FIELDS),
        source_path=None,
    )


def generate_mock_dataset(
    *,
    out_dir: str | Path,
    seed: int,
    mock_cfg: dict[str, Any],
    n_trains: int,
    prefix: str = "config",
    overwrite: bool = False,
) -> list[Path]:
    """Generate a pool of mock pulse train files on disk.

    Args:
        out_dir: directory to write ``<prefix>_<i>.h5`` files into.
        seed: global experiment seed.
        mock_cfg: the ``mock`` section of the configuration.
        n_trains: number of pulse trains to generate.
        prefix: file name prefix (matches the TSRD ``config_<i>.h5`` convention).
        overwrite: regenerate files that already exist.

    Returns:
        Paths of the generated (or already present) pulse train files.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for train_id in range(n_trains):
        path = directory / f"{prefix}_{train_id}.h5"
        if path.exists() and not overwrite:
            paths.append(path)
            continue
        record = generate_pulse_train(seed=seed, mock_cfg=mock_cfg, train_id=train_id)
        record.save(path)
        LOGGER.info(
            "Generated mock pulse train %s (%d pulses, %d emitters)",
            path.name,
            record.n_pulses,
            0 if record.labels is None else int(np.unique(record.labels).size),
        )
        paths.append(path)
    return paths
