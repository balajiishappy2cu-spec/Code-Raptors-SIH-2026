"""Tests for the band-level feature layer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.band_features import FEATURE_NAMES, BandFeatureTracker  # noqa: E402

FEATURES_CFG = {"history_window": 50, "max_pri_samples": 24, "min_pri_samples": 3}


def make_tracker(n_bands: int = 4) -> BandFeatureTracker:
    """Return a tracker with the test configuration."""
    return BandFeatureTracker(n_bands=n_bands, features_cfg=FEATURES_CFG, timestep_us=2000.0)


def test_feature_vector_matches_declared_layout() -> None:
    tracker = make_tracker()
    vector = tracker.feature_vector(band=0, timestep=0)
    assert vector.shape == (len(FEATURE_NAMES),)
    assert set(tracker.band_features(0, 0)) == set(FEATURE_NAMES)


def test_snapshot_has_one_row_per_band() -> None:
    tracker = make_tracker(n_bands=7)
    snapshot = tracker.snapshot(timestep=3)
    assert snapshot.shape == (7, len(FEATURE_NAMES))


def test_unvisited_band_is_flagged_and_becomes_visited() -> None:
    tracker = make_tracker()
    assert tracker.band_features(1, 0)["never_visited"] == 1.0
    tracker.update(band=1, timestep=0, hit=False)
    assert tracker.band_features(1, 1)["never_visited"] == 0.0


def test_hit_rates_reflect_observations() -> None:
    tracker = make_tracker()
    for timestep in range(10):
        tracker.update(band=2, timestep=timestep, hit=timestep % 2 == 0, signal_count=3)
    features = tracker.band_features(2, 10)
    assert features["global_hit_rate"] == pytest.approx(0.5)
    assert features["occupancy_rate"] == pytest.approx(0.5)
    assert 0.0 <= features["recent_hit_rate"] <= 1.0


def test_pri_estimated_from_regular_hits() -> None:
    tracker = make_tracker()
    for timestep in range(0, 100, 10):
        tracker.update(band=0, timestep=timestep, hit=True)
    median, mad, cv, n_samples = tracker.pri_stats(0)
    assert median == pytest.approx(10.0)
    assert mad == pytest.approx(0.0)
    assert cv == pytest.approx(0.0)
    assert n_samples == 9
    assert tracker.pri_estimate_us(0) == pytest.approx(20000.0)


def test_periodicity_score_high_for_regular_and_zero_when_unknown() -> None:
    tracker = make_tracker()
    assert tracker.periodicity_score(0) == 0.0
    for timestep in range(0, 240, 10):
        tracker.update(band=0, timestep=timestep, hit=True)
    regular = tracker.periodicity_score(0)

    irregular_tracker = make_tracker()
    for timestep in [0, 3, 31, 33, 90, 91, 150, 210, 211, 212]:
        irregular_tracker.update(band=0, timestep=timestep, hit=True)
    assert regular > irregular_tracker.periodicity_score(0)
    assert 0.0 <= regular <= 1.0


def test_predicted_next_hit_follows_the_interval() -> None:
    tracker = make_tracker()
    for timestep in range(0, 100, 10):
        tracker.update(band=0, timestep=timestep, hit=True)
    assert tracker.predicted_next_hit(0) == pytest.approx(100.0)


def test_phase_score_peaks_at_the_predicted_time() -> None:
    tracker = make_tracker()
    for timestep in range(0, 240, 10):
        tracker.update(band=0, timestep=timestep, hit=True)
    on_time = tracker.phase_score(0, timestep=240)
    off_time = tracker.phase_score(0, timestep=235)
    assert on_time > off_time
    assert 0.0 <= on_time <= 1.0


def test_features_do_not_leak_future_observations() -> None:
    """Features at time t must be identical whether or not later updates happen."""
    early = make_tracker()
    for timestep in range(5):
        early.update(band=0, timestep=timestep, hit=True)
    before = early.feature_vector(0, timestep=5)

    later = make_tracker()
    for timestep in range(5):
        later.update(band=0, timestep=timestep, hit=True)
    snapshot_at_five = later.feature_vector(0, timestep=5)
    for timestep in range(5, 20):
        later.update(band=0, timestep=timestep, hit=True)

    assert np.allclose(before, snapshot_at_five)


def test_pri_cache_is_invalidated_by_new_hits() -> None:
    tracker = make_tracker()
    for timestep in range(0, 50, 10):
        tracker.update(band=0, timestep=timestep, hit=True)
    first = tracker.pri_stats(0)
    for timestep in range(70, 200, 30):
        tracker.update(band=0, timestep=timestep, hit=True)
    assert tracker.pri_stats(0) != first


def test_reset_clears_history() -> None:
    tracker = make_tracker()
    tracker.update(band=0, timestep=0, hit=True)
    tracker.reset()
    assert tracker.total_visits == 0
    assert tracker.histories[0].visits == 0
    assert tracker.band_features(0, 0)["never_visited"] == 1.0


def test_update_from_observation_ignores_settling() -> None:
    class FakeObservation:
        timestep = 4
        selected_band = 1
        detected = True
        signal_count = 5
        bands = (1,)
        settling = True

    tracker = make_tracker()
    tracker.update_from_observation(FakeObservation())
    assert tracker.total_visits == 0


def test_update_from_observation_covers_every_band_in_the_bandwidth() -> None:
    class FakeObservation:
        timestep = 2
        selected_band = 0
        detected = True
        signal_count = 9
        bands = (0, 1)
        settling = False

    tracker = make_tracker()
    tracker.update_from_observation(FakeObservation())
    assert tracker.histories[0].hits == 1
    assert tracker.histories[1].hits == 1
    assert tracker.total_visits == 2


def test_all_features_are_finite() -> None:
    tracker = make_tracker()
    rng = np.random.default_rng(0)
    for timestep in range(200):
        band = int(rng.integers(0, tracker.n_bands))
        tracker.update(band=band, timestep=timestep, hit=bool(rng.random() < 0.4), signal_count=2)
    snapshot = tracker.snapshot(timestep=200)
    assert np.all(np.isfinite(snapshot))
