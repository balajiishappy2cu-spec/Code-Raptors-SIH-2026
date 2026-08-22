"""Build the electromagnetic environment ground truth from raw PDWs.

The problem statement asks for exactly one thing from the environment:

    "The status of environment for each frequency band at each time step can be
     recorded as a transmission or a non-transmission."

That is obtained by binning each PDW's centre frequency into a band index and its time
of arrival into a timestep index. No deinterleaving, no clustering and no emitter labels
are involved -- which is what makes the scheduler buildable independently of the Turing
Deinterleaving Challenge's clustering task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from common.logging_utils import get_logger
from dataio.tdc_interface import PulseTrainRecord

LOGGER = get_logger(__name__)


@dataclass
class EnvironmentGrid:
    """Ground truth activity of the electromagnetic environment.

    Attributes:
        active: ``(n_timesteps, n_bands)`` boolean transmission / non-transmission grid.
        n_pulses: ``(n_timesteps, n_bands)`` pulse counts per cell.
        mean_aoa: ``(n_timesteps, n_bands)`` mean angle of arrival, ``NaN`` where idle.
        band_edges_mhz: ``(n_bands + 1,)`` band boundaries in MHz.
        timestep_us: duration of one timestep in microseconds.
        t0_us: time of arrival corresponding to timestep 0.
        name: identifier of the source pulse train.
        metadata: provenance and build parameters.
    """

    active: np.ndarray
    n_pulses: np.ndarray
    mean_aoa: np.ndarray
    band_edges_mhz: np.ndarray
    timestep_us: float
    t0_us: float
    name: str = "environment"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_timesteps(self) -> int:
        """Number of timesteps in the grid."""
        return int(self.active.shape[0])

    @property
    def n_bands(self) -> int:
        """Number of frequency bands in the grid."""
        return int(self.active.shape[1])

    @property
    def band_centres_mhz(self) -> np.ndarray:
        """Centre frequency of each band in MHz."""
        return 0.5 * (self.band_edges_mhz[:-1] + self.band_edges_mhz[1:])

    @property
    def band_width_mhz(self) -> float:
        """Width of a single band in MHz."""
        return float(self.band_edges_mhz[1] - self.band_edges_mhz[0])

    @property
    def occupancy(self) -> float:
        """Fraction of grid cells that carry a transmission."""
        total = self.active.size
        return float(self.active.sum() / total) if total else 0.0

    def is_active(self, timestep: int, band: int) -> bool:
        """Whether a band carries a transmission at a timestep."""
        return bool(self.active[timestep, band])

    def active_in_window(self, timestep: int, band: int, window: int) -> bool:
        """Whether a band transmits at any point in ``[timestep, timestep + window)``.

        This is the target definition used by the activity predictor.
        """
        end = min(timestep + window, self.n_timesteps)
        if timestep >= self.n_timesteps or end <= timestep:
            return False
        return bool(self.active[timestep:end, band].any())

    def active_window_matrix(self, window: int) -> np.ndarray:
        """Return ``(n_timesteps, n_bands)`` "active somewhere in the next window" flags.

        Cell ``(t, b)`` is ``True`` when band ``b`` transmits at any timestep in
        ``[t, t + window)``. This is the activity-model target, computed in one
        vectorised pass rather than per cell.

        Args:
            window: length of the prediction window in timesteps.

        Returns:
            Boolean matrix of the same shape as :attr:`active`.
        """
        span = max(1, int(window))
        counts = np.cumsum(self.active.astype(np.int64), axis=0)
        padded = np.vstack([np.zeros((1, self.n_bands), dtype=np.int64), counts])
        end = np.minimum(np.arange(self.n_timesteps) + span, self.n_timesteps)
        return (padded[end] - padded[: self.n_timesteps]) > 0

    def first_active_timestep(self) -> np.ndarray:
        """Return, per band, the first timestep at which it transmits.

        Bands that never transmit get ``-1``.
        """
        first = np.full(self.n_bands, -1, dtype=np.int64)
        for band in range(self.n_bands):
            hits = np.flatnonzero(self.active[:, band])
            if hits.size:
                first[band] = int(hits[0])
        return first

    def next_active_timestep(self, timestep: int, band: int) -> int:
        """First timestep at or after ``timestep`` where ``band`` transmits, else ``-1``."""
        if timestep >= self.n_timesteps:
            return -1
        column = self.active[timestep:, band]
        hits = np.flatnonzero(column)
        return int(timestep + hits[0]) if hits.size else -1

    def truncate(self, n_timesteps: int) -> "EnvironmentGrid":
        """Return a grid limited to the first ``n_timesteps`` timesteps."""
        if n_timesteps >= self.n_timesteps:
            return self
        return EnvironmentGrid(
            active=self.active[:n_timesteps],
            n_pulses=self.n_pulses[:n_timesteps],
            mean_aoa=self.mean_aoa[:n_timesteps],
            band_edges_mhz=self.band_edges_mhz,
            timestep_us=self.timestep_us,
            t0_us=self.t0_us,
            name=self.name,
            metadata={**self.metadata, "truncated_to": int(n_timesteps)},
        )

    def summary(self) -> dict[str, Any]:
        """Return a JSON-friendly summary used in run records and the dashboard."""
        per_band = self.active.sum(axis=0)
        return {
            "name": self.name,
            "n_timesteps": self.n_timesteps,
            "n_bands": self.n_bands,
            "timestep_us": self.timestep_us,
            "occupancy": self.occupancy,
            "total_pulses": int(self.n_pulses.sum()),
            "active_cells": int(self.active.sum()),
            "bands_ever_active": int((per_band > 0).sum()),
            "busiest_band": int(np.argmax(per_band)) if self.n_bands else -1,
            "band_edges_mhz": [float(self.band_edges_mhz[0]), float(self.band_edges_mhz[-1])],
        }


def build_environment(
    record: PulseTrainRecord,
    env_cfg: dict[str, Any],
    *,
    name: str | None = None,
    pulse_mask: np.ndarray | None = None,
) -> EnvironmentGrid:
    """Bin a PDW stream into an ``environment[timestep][band]`` truth grid.

    Args:
        record: the pulse train to bin.
        env_cfg: the ``environment`` section of the configuration.
        name: optional name override for the resulting grid.
        pulse_mask: optional boolean mask selecting a subset of pulses (used by the
            scenario builders to keep only one class of emitter).

    Returns:
        The :class:`EnvironmentGrid` ground truth.

    Raises:
        ValueError: if the pulse train lacks ToA or CF.
    """
    if not (record.has("toa") and record.has("cf")):
        msg = f"Pulse train {record.name!r} lacks ToA/CF; cannot build an environment"
        raise ValueError(msg)

    n_bands = int(env_cfg.get("n_bands", 32))
    cf_min = float(env_cfg.get("cf_min_mhz", 2000.0))
    cf_max = float(env_cfg.get("cf_max_mhz", 18000.0))
    timestep_us = float(env_cfg.get("timestep_us", 2000.0))
    max_timesteps = int(env_cfg.get("max_timesteps", 4000))
    min_pulses_for_active = int(env_cfg.get("min_pulses_for_active", 1))

    toa = record.column("toa")
    cf = record.column("cf")
    aoa = record.column("aoa") if record.has("aoa") else np.zeros_like(toa)

    if pulse_mask is not None:
        toa, cf, aoa = toa[pulse_mask], cf[pulse_mask], aoa[pulse_mask]

    grid_name = name or record.name
    if toa.size == 0:
        LOGGER.warning("Pulse train %s has no pulses after masking; empty environment", grid_name)
        empty_shape = (0, n_bands)
        return EnvironmentGrid(
            active=np.zeros(empty_shape, dtype=bool),
            n_pulses=np.zeros(empty_shape, dtype=np.int32),
            mean_aoa=np.full(empty_shape, np.nan),
            band_edges_mhz=np.linspace(cf_min, cf_max, n_bands + 1),
            timestep_us=timestep_us,
            t0_us=0.0,
            name=grid_name,
            metadata={"source": record.name, "n_pulses_binned": 0},
        )

    in_band = (cf >= cf_min) & (cf < cf_max)
    dropped = int((~in_band).sum())
    if dropped:
        LOGGER.info(
            "%s: %d/%d pulses outside [%.0f, %.0f) MHz dropped from the grid",
            grid_name,
            dropped,
            cf.size,
            cf_min,
            cf_max,
        )
    toa, cf, aoa = toa[in_band], cf[in_band], aoa[in_band]
    if toa.size == 0:
        msg = f"All pulses of {grid_name} fall outside the configured band range"
        raise ValueError(msg)

    band_edges = np.linspace(cf_min, cf_max, n_bands + 1)
    band_index = np.clip(
        ((cf - cf_min) / (cf_max - cf_min) * n_bands).astype(np.int64), 0, n_bands - 1
    )

    t0 = float(toa.min())
    step_index = ((toa - t0) / timestep_us).astype(np.int64)
    n_timesteps = int(step_index.max()) + 1
    if max_timesteps > 0 and n_timesteps > max_timesteps:
        keep = step_index < max_timesteps
        step_index, band_index, aoa = step_index[keep], band_index[keep], aoa[keep]
        n_timesteps = max_timesteps

    flat = step_index * n_bands + band_index
    counts = np.bincount(flat, minlength=n_timesteps * n_bands).reshape(n_timesteps, n_bands)
    aoa_sum = np.bincount(flat, weights=aoa, minlength=n_timesteps * n_bands).reshape(
        n_timesteps, n_bands
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_aoa = np.where(counts > 0, aoa_sum / np.maximum(counts, 1), np.nan)

    active = counts >= max(1, min_pulses_for_active)
    grid = EnvironmentGrid(
        active=active,
        n_pulses=counts.astype(np.int32),
        mean_aoa=mean_aoa,
        band_edges_mhz=band_edges,
        timestep_us=timestep_us,
        t0_us=t0,
        name=grid_name,
        metadata={
            "source": record.name,
            "source_path": str(record.source_path) if record.source_path else None,
            "n_pulses_binned": int(counts.sum()),
            "n_pulses_dropped_out_of_band": dropped,
            "cf_min_mhz": cf_min,
            "cf_max_mhz": cf_max,
            "min_pulses_for_active": min_pulses_for_active,
        },
    )
    LOGGER.info(
        "Built environment %s: %d timesteps x %d bands, occupancy %.3f, %d pulses",
        grid.name,
        grid.n_timesteps,
        grid.n_bands,
        grid.occupancy,
        int(grid.n_pulses.sum()),
    )
    return grid


def build_environments(
    records: Iterable[PulseTrainRecord],
    env_cfg: dict[str, Any],
) -> list[EnvironmentGrid]:
    """Build one environment grid per pulse train, skipping degenerate trains."""
    grids: list[EnvironmentGrid] = []
    for record in records:
        try:
            grids.append(build_environment(record, env_cfg))
        except ValueError as exc:
            LOGGER.warning("Skipping pulse train %s: %s", record.name, exc)
    return grids


def stack_summaries(grids: Sequence[EnvironmentGrid]) -> dict[str, Any]:
    """Aggregate summary statistics across a set of environments."""
    if not grids:
        return {"n_environments": 0}
    occupancies = [grid.occupancy for grid in grids]
    return {
        "n_environments": len(grids),
        "n_bands": grids[0].n_bands,
        "timestep_us": grids[0].timestep_us,
        "mean_occupancy": float(np.mean(occupancies)),
        "min_occupancy": float(np.min(occupancies)),
        "max_occupancy": float(np.max(occupancies)),
        "mean_timesteps": float(np.mean([grid.n_timesteps for grid in grids])),
        "total_pulses": int(sum(int(grid.n_pulses.sum()) for grid in grids)),
    }
