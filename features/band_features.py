"""Band-level features computed from observation history only.

Every feature here is derived from what the receiver has actually observed -- band
visits and their hit/miss outcomes -- never from the ground truth grid. No emitter
cluster identity is required, so the feature layer has no dependency on deinterleaving.

The PRI statistics are estimated from the intervals between *observed hits* in a band.
A scanning receiver does not see every pulse, so this is not the emitter's transmitted
pulse repetition interval; it is the interval at which that band is found active from
the receiver's point of view. That is the quantity interception scheduling actually
needs, and it is what feeds the scheduler's periodicity bonus.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Sequence

import numpy as np

#: Feature vector layout. Fixed and ordered so training and inference cannot drift apart.
FEATURE_NAMES: tuple[str, ...] = (
    "occupancy_rate",
    "recent_hit_rate",
    "global_hit_rate",
    "time_since_last_visit",
    "time_since_last_hit",
    "visit_count",
    "hit_count",
    "visit_share",
    "mean_signal_count",
    "pri_median",
    "pri_mad",
    "pri_cv",
    "n_pri_samples",
    "periodicity_score",
    "phase_score",
    "predicted_next_hit_delta",
    "band_position",
    "never_visited",
)

_UNSEEN_TIME = 1.0  # value used for "time since ..." features when it has never happened


@dataclass
class BandHistory:
    """Observation history for a single frequency band.

    Attributes:
        visits: total number of timesteps the band has been observed.
        hits: total number of observations that declared a detection.
        last_visit: timestep of the most recent observation, or ``-1``.
        last_hit: timestep of the most recent detection, or ``-1``.
        visit_log: recent ``(timestep, hit)`` pairs, bounded by the history window.
        hit_times: recent detection timesteps, bounded by ``max_pri_samples + 1``.
        signal_counts: recent reported pulse counts on detections.
    """

    visits: int = 0
    hits: int = 0
    last_visit: int = -1
    last_hit: int = -1
    visit_log: Deque[tuple[int, bool]] = field(default_factory=deque)
    hit_times: Deque[int] = field(default_factory=deque)
    signal_counts: Deque[int] = field(default_factory=deque)


class BandFeatureTracker:
    """Maintains per-band observation history and emits feature vectors.

    The tracker is updated with :class:`~simulation.receiver.Observation` results and
    queried for features before each scheduling decision, so no future information can
    leak into a feature vector.
    """

    def __init__(
        self,
        n_bands: int,
        features_cfg: dict[str, Any] | None = None,
        *,
        timestep_us: float = 1.0,
    ) -> None:
        """Create a tracker.

        Args:
            n_bands: number of frequency bands in the environment.
            features_cfg: the ``features`` section of the configuration.
            timestep_us: timestep duration, used to report PRI estimates in
                microseconds alongside the timestep-based values.
        """
        cfg = features_cfg or {}
        self.n_bands = int(n_bands)
        self.history_window = int(cfg.get("history_window", 50))
        self.max_pri_samples = int(cfg.get("max_pri_samples", 24))
        self.min_pri_samples = int(cfg.get("min_pri_samples", 3))
        self.timestep_us = float(timestep_us)
        self.total_visits = 0
        self.total_hits = 0
        self.histories: list[BandHistory] = [BandHistory() for _ in range(self.n_bands)]
        # PRI statistics are read several times per decision but only change when a
        # band records a new hit, so they are memoised per band and invalidated there.
        self._pri_cache: list[tuple[float, float, float, int] | None] = [None] * self.n_bands

    def reset(self) -> None:
        """Clear all history (used between independent simulation runs)."""
        self.total_visits = 0
        self.total_hits = 0
        self.histories = [BandHistory() for _ in range(self.n_bands)]
        self._pri_cache = [None] * self.n_bands

    def update(self, band: int, timestep: int, hit: bool, signal_count: int = 0) -> None:
        """Record the outcome of observing one band at one timestep.

        Args:
            band: observed band index.
            timestep: timestep of the observation.
            hit: whether the receiver declared a detection.
            signal_count: reported pulse count for the observation.
        """
        history = self.histories[band]
        history.visits += 1
        history.last_visit = timestep
        history.visit_log.append((timestep, bool(hit)))
        self.total_visits += 1

        if hit:
            self._pri_cache[band] = None
            history.hits += 1
            history.last_hit = timestep
            history.hit_times.append(timestep)
            history.signal_counts.append(int(signal_count))
            self.total_hits += 1
            while len(history.hit_times) > self.max_pri_samples + 1:
                history.hit_times.popleft()
            while len(history.signal_counts) > self.max_pri_samples + 1:
                history.signal_counts.popleft()

        cutoff = timestep - self.history_window
        while history.visit_log and history.visit_log[0][0] < cutoff:
            history.visit_log.popleft()

    def update_from_observation(self, observation: Any) -> None:
        """Update history from a :class:`~simulation.receiver.Observation`.

        Observations taken while the receiver was settling carry no information and
        are ignored.
        """
        if getattr(observation, "settling", False):
            return
        for band in observation.bands or (observation.selected_band,):
            self.update(
                band=band,
                timestep=observation.timestep,
                hit=bool(observation.detected),
                signal_count=int(observation.signal_count),
            )

    def pri_stats(self, band: int) -> tuple[float, float, float, int]:
        """Return robust statistics of the observed hit intervals for a band.

        Returns:
            ``(median, median_absolute_deviation, coefficient_of_variation, n_samples)``
            in timesteps. All zero when there are too few samples.
        """
        cached = self._pri_cache[band]
        if cached is not None:
            return cached
        stats = self._compute_pri_stats(band)
        self._pri_cache[band] = stats
        return stats

    def _compute_pri_stats(self, band: int) -> tuple[float, float, float, int]:
        """Compute (uncached) robust hit-interval statistics for a band."""
        history = self.histories[band]
        if len(history.hit_times) < self.min_pri_samples + 1:
            return 0.0, 0.0, 0.0, 0
        intervals = np.diff(np.asarray(history.hit_times, dtype=np.float64))
        intervals = intervals[intervals > 0]
        if intervals.size < self.min_pri_samples:
            return 0.0, 0.0, 0.0, int(intervals.size)
        median = float(np.median(intervals))
        mad = float(np.median(np.abs(intervals - median)))
        cv = float(mad / median) if median > 0 else 0.0
        return median, mad, cv, int(intervals.size)

    def periodicity_score(self, band: int) -> float:
        """Return a regularity score in ``[0, 1]`` for a band's observed hit intervals.

        A perfectly regular interval sequence scores 1; a highly irregular one tends to 0.
        Bands with too few samples score 0, which keeps the scheduler from acting on
        noise early in a run.
        """
        _, _, cv, n_samples = self.pri_stats(band)
        if n_samples < self.min_pri_samples:
            return 0.0
        confidence = min(1.0, n_samples / float(max(1, self.max_pri_samples)))
        return float(np.exp(-2.0 * cv) * confidence)

    def predicted_next_hit(self, band: int) -> float:
        """Predicted timestep of the next detection in a band.

        Returns ``-1.0`` when no PRI estimate is available yet.
        """
        median, _, _, n_samples = self.pri_stats(band)
        history = self.histories[band]
        if n_samples < self.min_pri_samples or median <= 0 or history.last_hit < 0:
            return -1.0
        return float(history.last_hit + median)

    def phase_score(self, band: int, timestep: int) -> float:
        """How close the current timestep is to a band's predicted next detection.

        Returns a value in ``[0, 1]``, scaled by the band's periodicity score so that
        an unreliable PRI estimate cannot dominate the scheduler.
        """
        predicted = self.predicted_next_hit(band)
        if predicted < 0:
            return 0.0
        median, mad, _, _ = self.pri_stats(band)
        tolerance = max(1.0, mad if mad > 0 else 0.05 * median)
        # Fold the error into the interval so that a missed prediction re-aligns.
        error = abs(timestep - predicted)
        if median > 0:
            error = min(error, abs(((timestep - predicted) % median)))
        return float(np.exp(-error / tolerance) * self.periodicity_score(band))

    def band_features(self, band: int, timestep: int) -> dict[str, float]:
        """Compute the feature dictionary for one band at one timestep."""
        history = self.histories[band]
        window = float(max(1, self.history_window))

        windowed = [hit for _, hit in history.visit_log]
        occupancy_rate = float(np.mean(windowed)) if windowed else 0.0
        recent = windowed[-10:]
        recent_hit_rate = float(np.mean(recent)) if recent else 0.0
        global_hit_rate = float(history.hits / history.visits) if history.visits else 0.0

        if history.last_visit < 0:
            time_since_visit = _UNSEEN_TIME
        else:
            time_since_visit = min(1.0, (timestep - history.last_visit) / window)
        if history.last_hit < 0:
            time_since_hit = _UNSEEN_TIME
        else:
            time_since_hit = min(1.0, (timestep - history.last_hit) / window)

        visit_share = (
            float(history.visits / self.total_visits) if self.total_visits else 0.0
        )
        mean_signal = float(np.mean(history.signal_counts)) if history.signal_counts else 0.0
        median, mad, cv, n_samples = self.pri_stats(band)
        predicted = self.predicted_next_hit(band)
        predicted_delta = (
            min(1.0, max(-1.0, (predicted - timestep) / window)) if predicted >= 0 else _UNSEEN_TIME
        )

        return {
            "occupancy_rate": occupancy_rate,
            "recent_hit_rate": recent_hit_rate,
            "global_hit_rate": global_hit_rate,
            "time_since_last_visit": time_since_visit,
            "time_since_last_hit": time_since_hit,
            "visit_count": float(min(1.0, history.visits / window)),
            "hit_count": float(min(1.0, history.hits / window)),
            "visit_share": visit_share,
            "mean_signal_count": float(np.log1p(mean_signal)),
            "pri_median": float(min(1.0, median / window)),
            "pri_mad": float(min(1.0, mad / window)),
            "pri_cv": float(min(2.0, cv)),
            "n_pri_samples": float(min(1.0, n_samples / float(max(1, self.max_pri_samples)))),
            "periodicity_score": self.periodicity_score(band),
            "phase_score": self.phase_score(band, timestep),
            "predicted_next_hit_delta": float(predicted_delta),
            "band_position": float(band / max(1, self.n_bands - 1)),
            "never_visited": 1.0 if history.visits == 0 else 0.0,
        }

    def feature_vector(self, band: int, timestep: int) -> np.ndarray:
        """Return one band's features as a vector ordered by :data:`FEATURE_NAMES`."""
        values = self.band_features(band, timestep)
        return np.asarray([values[name] for name in FEATURE_NAMES], dtype=np.float64)

    def snapshot(self, timestep: int, bands: Sequence[int] | None = None) -> np.ndarray:
        """Return an ``(n_bands, n_features)`` matrix for the given timestep.

        Args:
            timestep: current timestep.
            bands: optional subset of bands; defaults to all bands in order.

        Returns:
            Feature matrix with rows ordered as ``bands``.
        """
        indices = list(range(self.n_bands)) if bands is None else list(bands)
        return np.vstack([self.feature_vector(band, timestep) for band in indices])

    def pri_estimate_us(self, band: int) -> float:
        """Return the band's observed hit-interval estimate in microseconds."""
        median, _, _, _ = self.pri_stats(band)
        return float(median * self.timestep_us)


def feature_frame_columns() -> list[str]:
    """Return the feature column names, for building pandas frames."""
    return list(FEATURE_NAMES)
