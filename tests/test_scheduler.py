"""Tests for the schedulers and their online adaptation."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.band_features import FEATURE_NAMES, BandFeatureTracker  # noqa: E402
from models.activity_predictor import HeuristicActivityPredictor  # noqa: E402
from models.scheduler import (  # noqa: E402
    RandomScheduler,
    SequentialSweepScheduler,
    SmartScanScheduler,
    ThompsonSampler,
    build_scheduler,
)
from simulation.receiver import Observation  # noqa: E402

N_BANDS = 6
FEATURES_CFG = {"history_window": 50, "max_pri_samples": 24, "min_pri_samples": 3}


@dataclass
class StubPredictor:
    """Predictor returning a fixed probability per band."""

    probabilities: np.ndarray
    name: str = "stub"

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return the fixed probabilities, ignoring the features."""
        return self.probabilities[: np.atleast_2d(features).shape[0]]


def make_tracker(n_bands: int = N_BANDS) -> BandFeatureTracker:
    """Return a tracker for the tests."""
    return BandFeatureTracker(n_bands=n_bands, features_cfg=FEATURES_CFG, timestep_us=1000.0)


def scheduler_cfg(**weights: float) -> dict:
    """Build a scheduler config with all weights zero except the ones given."""
    base = {
        "w1_predicted_probability": 0.0,
        "w2_exploration_bonus": 0.0,
        "w3_recency_bonus": 0.0,
        "w4_periodicity_bonus": 0.0,
        "w5_scan_cost": 0.0,
    }
    base.update(weights)
    return {
        "weights": base,
        "thompson": {"enabled": False, "prior_alpha": 1.0, "prior_beta": 1.0, "decay": 1.0},
        "exploration": {"staleness_saturation": 10},
    }


def observation(band: int, timestep: int, detected: bool) -> Observation:
    """Build an observation for one band."""
    return Observation(
        timestep=timestep,
        selected_band=band,
        detected=detected,
        signal_count=3 if detected else 0,
        bands=(band,),
        settling=False,
    )


def test_sequential_sweep_cycles_through_every_band() -> None:
    scheduler = SequentialSweepScheduler(n_bands=N_BANDS, rng=np.random.default_rng(0))
    tracker = make_tracker()
    chosen = [scheduler.select_band(t, tracker) for t in range(N_BANDS * 2)]
    assert chosen == list(range(N_BANDS)) * 2


def test_sequential_sweep_reset_restarts_at_zero() -> None:
    scheduler = SequentialSweepScheduler(n_bands=N_BANDS, rng=np.random.default_rng(0))
    tracker = make_tracker()
    scheduler.select_band(0, tracker)
    scheduler.select_band(1, tracker)
    scheduler.reset()
    assert scheduler.select_band(2, tracker) == 0


def test_random_scheduler_stays_in_range() -> None:
    scheduler = RandomScheduler(n_bands=N_BANDS, rng=np.random.default_rng(1))
    tracker = make_tracker()
    chosen = [scheduler.select_band(t, tracker) for t in range(200)]
    assert min(chosen) >= 0
    assert max(chosen) < N_BANDS
    assert len(set(chosen)) > 1


def test_thompson_posterior_moves_with_feedback() -> None:
    sampler = ThompsonSampler(n_bands=3, rng=np.random.default_rng(2))
    for _ in range(20):
        sampler.update(band=0, hit=True)
        sampler.update(band=1, hit=False)
    means = sampler.means()
    assert means[0] > means[2] > means[1]


def test_thompson_decay_forgets_old_evidence() -> None:
    fast = ThompsonSampler(n_bands=1, decay=0.5, rng=np.random.default_rng(3))
    slow = ThompsonSampler(n_bands=1, decay=1.0, rng=np.random.default_rng(3))
    for _ in range(30):
        fast.update(band=0, hit=True)
        slow.update(band=0, hit=True)
    for _ in range(5):
        fast.update(band=0, hit=False)
        slow.update(band=0, hit=False)
    # With discounting the recent misses move the posterior much further.
    assert fast.means()[0] < slow.means()[0]


def test_thompson_reset_restores_the_prior() -> None:
    sampler = ThompsonSampler(n_bands=2, rng=np.random.default_rng(4))
    sampler.update(band=0, hit=True)
    sampler.reset()
    assert np.allclose(sampler.alpha, 1.0)
    assert np.allclose(sampler.beta, 1.0)


def test_smart_scheduler_follows_the_predicted_probability() -> None:
    probabilities = np.array([0.1, 0.2, 0.95, 0.3, 0.1, 0.05])
    scheduler = SmartScanScheduler(
        n_bands=N_BANDS,
        rng=np.random.default_rng(5),
        predictor=StubPredictor(probabilities),
        scheduler_cfg=scheduler_cfg(w1_predicted_probability=1.0),
    )
    assert scheduler.select_band(0, make_tracker()) == 2


def test_smart_scheduler_explores_the_stalest_band() -> None:
    cfg = scheduler_cfg(w2_exploration_bonus=1.0)
    # Saturation must exceed the staleness gap, otherwise every band saturates and ties.
    cfg["exploration"]["staleness_saturation"] = 200
    scheduler = SmartScanScheduler(
        n_bands=N_BANDS,
        rng=np.random.default_rng(6),
        predictor=StubPredictor(np.full(N_BANDS, 0.5)),
        scheduler_cfg=cfg,
    )
    tracker = make_tracker()
    # Every band seen recently except band 4.
    for band in range(N_BANDS):
        tracker.update(band=band, timestep=90 if band != 4 else 10, hit=False)
    assert scheduler.select_band(100, tracker) == 4


def test_smart_scheduler_penalises_distant_retunes() -> None:
    scheduler = SmartScanScheduler(
        n_bands=N_BANDS,
        rng=np.random.default_rng(7),
        predictor=StubPredictor(np.full(N_BANDS, 0.5)),
        scheduler_cfg=scheduler_cfg(w5_scan_cost=1.0),
    )
    scheduler.current_band = 0
    assert scheduler.select_band(1, make_tracker()) == 0


def test_smart_scheduler_prefers_the_predicted_periodic_window() -> None:
    scheduler = SmartScanScheduler(
        n_bands=N_BANDS,
        rng=np.random.default_rng(8),
        predictor=StubPredictor(np.full(N_BANDS, 0.5)),
        scheduler_cfg=scheduler_cfg(w4_periodicity_bonus=1.0),
    )
    tracker = make_tracker()
    for timestep in range(0, 240, 10):
        tracker.update(band=3, timestep=timestep, hit=True)
    for band in (0, 1, 2, 4, 5):
        tracker.update(band=band, timestep=5, hit=False)
    assert scheduler.select_band(240, tracker) == 3


def test_smart_scheduler_records_its_score_components() -> None:
    scheduler = SmartScanScheduler(
        n_bands=N_BANDS,
        rng=np.random.default_rng(9),
        predictor=HeuristicActivityPredictor(),
        scheduler_cfg=scheduler_cfg(w1_predicted_probability=1.0),
    )
    scheduler.select_band(0, make_tracker())
    decision = scheduler.last_decision
    for key in (
        "band",
        "score",
        "predicted_probability",
        "exploration_bonus",
        "recency_bonus",
        "periodicity_bonus",
        "scan_cost",
    ):
        assert key in decision
    assert len(decision["all_scores"]) == N_BANDS


def test_smart_scheduler_adapts_from_hit_and_miss_feedback() -> None:
    cfg = scheduler_cfg(w1_predicted_probability=1.0)
    cfg["thompson"] = {"enabled": True, "prior_alpha": 1.0, "prior_beta": 1.0, "decay": 1.0}
    scheduler = SmartScanScheduler(
        n_bands=N_BANDS,
        rng=np.random.default_rng(10),
        predictor=StubPredictor(np.full(N_BANDS, 0.5)),
        scheduler_cfg=cfg,
    )
    for timestep in range(30):
        scheduler.update(observation(band=1, timestep=timestep, detected=True))
        scheduler.update(observation(band=2, timestep=timestep, detected=False))
    means = scheduler.thompson.means()
    assert means[1] > means[0] > means[2]


def test_settling_observations_do_not_update_the_posterior() -> None:
    cfg = scheduler_cfg(w1_predicted_probability=1.0)
    cfg["thompson"] = {"enabled": True, "prior_alpha": 1.0, "prior_beta": 1.0, "decay": 1.0}
    scheduler = SmartScanScheduler(
        n_bands=N_BANDS,
        rng=np.random.default_rng(11),
        predictor=StubPredictor(np.full(N_BANDS, 0.5)),
        scheduler_cfg=cfg,
    )
    scheduler.update(
        Observation(
            timestep=0, selected_band=1, detected=True, signal_count=0, bands=(1,), settling=True
        )
    )
    assert scheduler.thompson.alpha[1] == pytest.approx(1.0)


def test_smart_scheduler_reset_clears_posteriors_and_state() -> None:
    cfg = scheduler_cfg(w1_predicted_probability=1.0)
    cfg["thompson"] = {"enabled": True, "prior_alpha": 1.0, "prior_beta": 1.0, "decay": 1.0}
    scheduler = SmartScanScheduler(
        n_bands=N_BANDS,
        rng=np.random.default_rng(12),
        predictor=StubPredictor(np.full(N_BANDS, 0.5)),
        scheduler_cfg=cfg,
    )
    scheduler.update(observation(band=0, timestep=0, detected=True))
    scheduler.reset()
    assert scheduler.current_band == -1
    assert np.allclose(scheduler.thompson.alpha, 1.0)


def test_scheduler_only_needs_the_observation_tracker() -> None:
    """The scheduler must be able to decide with no access to ground truth."""
    scheduler = SmartScanScheduler(
        n_bands=N_BANDS,
        rng=np.random.default_rng(13),
        predictor=HeuristicActivityPredictor(),
        scheduler_cfg=scheduler_cfg(w1_predicted_probability=1.0, w2_exploration_bonus=1.0),
    )
    tracker = make_tracker()
    band = scheduler.select_band(0, tracker)
    assert 0 <= band < N_BANDS
    assert tracker.snapshot(0).shape == (N_BANDS, len(FEATURE_NAMES))


def test_build_scheduler_validates_its_arguments() -> None:
    rng = np.random.default_rng(14)
    assert isinstance(build_scheduler("sequential", N_BANDS, rng), SequentialSweepScheduler)
    assert isinstance(build_scheduler("random", N_BANDS, rng), RandomScheduler)
    with pytest.raises(ValueError):
        build_scheduler("smart", N_BANDS, rng)
    with pytest.raises(ValueError):
        build_scheduler("nonexistent", N_BANDS, rng)


def test_describe_reports_the_configuration() -> None:
    scheduler = SmartScanScheduler(
        n_bands=N_BANDS,
        rng=np.random.default_rng(15),
        predictor=HeuristicActivityPredictor(),
        scheduler_cfg=scheduler_cfg(w1_predicted_probability=1.0),
    )
    described = scheduler.describe()
    assert described["name"] == "smart"
    assert described["predictor"] == "heuristic"
    assert described["weights"]["w1_predicted_probability"] == 1.0


def test_revisit_interval_forces_an_overdue_band() -> None:
    """A band past its revisit deadline must pre-empt a higher-scoring fresh band."""
    cfg = scheduler_cfg(w1_predicted_probability=1.0)
    cfg["exploration"]["max_revisit_interval"] = 20
    probabilities = np.array([0.99, 0.1, 0.1, 0.1, 0.1, 0.1])
    scheduler = SmartScanScheduler(
        n_bands=N_BANDS,
        rng=np.random.default_rng(20),
        predictor=StubPredictor(probabilities),
        scheduler_cfg=cfg,
    )
    tracker = make_tracker()
    # Band 0 looks best and was just seen; band 3 is overdue and must win anyway.
    for band in range(N_BANDS):
        tracker.update(band=band, timestep=95 if band != 3 else 10, hit=False)
    assert scheduler.select_band(100, tracker) == 3
    assert scheduler.last_decision["forced_revisit"] is True


def test_revisit_interval_picks_the_best_among_overdue_bands() -> None:
    cfg = scheduler_cfg(w1_predicted_probability=1.0)
    cfg["exploration"]["max_revisit_interval"] = 20
    probabilities = np.array([0.1, 0.2, 0.9, 0.3, 0.1, 0.1])
    scheduler = SmartScanScheduler(
        n_bands=N_BANDS,
        rng=np.random.default_rng(21),
        predictor=StubPredictor(probabilities),
        scheduler_cfg=cfg,
    )
    tracker = make_tracker()
    # Bands 2 and 3 are both overdue; band 2 scores higher, so it should win.
    for band in range(N_BANDS):
        tracker.update(band=band, timestep=95 if band not in (2, 3) else 10, hit=False)
    assert scheduler.select_band(100, tracker) == 2


def test_no_forced_revisit_when_nothing_is_overdue() -> None:
    cfg = scheduler_cfg(w1_predicted_probability=1.0)
    cfg["exploration"]["max_revisit_interval"] = 500
    probabilities = np.array([0.1, 0.95, 0.1, 0.1, 0.1, 0.1])
    scheduler = SmartScanScheduler(
        n_bands=N_BANDS,
        rng=np.random.default_rng(22),
        predictor=StubPredictor(probabilities),
        scheduler_cfg=cfg,
    )
    tracker = make_tracker()
    for band in range(N_BANDS):
        tracker.update(band=band, timestep=99, hit=False)
    assert scheduler.select_band(100, tracker) == 1
    assert scheduler.last_decision["forced_revisit"] is False


def test_revisit_disabled_by_zero_leaves_scoring_untouched() -> None:
    cfg = scheduler_cfg(w1_predicted_probability=1.0)
    cfg["exploration"]["max_revisit_interval"] = 0
    probabilities = np.array([0.99, 0.1, 0.1, 0.1, 0.1, 0.1])
    scheduler = SmartScanScheduler(
        n_bands=N_BANDS,
        rng=np.random.default_rng(23),
        predictor=StubPredictor(probabilities),
        scheduler_cfg=cfg,
    )
    tracker = make_tracker()
    for band in range(N_BANDS):
        tracker.update(band=band, timestep=95 if band != 3 else 10, hit=False)
    assert scheduler.select_band(100, tracker) == 0


def test_never_visited_bands_count_as_overdue() -> None:
    cfg = scheduler_cfg(w1_predicted_probability=1.0)
    cfg["exploration"]["max_revisit_interval"] = 20
    probabilities = np.array([0.99, 0.1, 0.1, 0.1, 0.1, 0.1])
    scheduler = SmartScanScheduler(
        n_bands=N_BANDS,
        rng=np.random.default_rng(24),
        predictor=StubPredictor(probabilities),
        scheduler_cfg=cfg,
    )
    tracker = make_tracker()
    # Every band seen except band 4, which has never been visited at all.
    for band in range(N_BANDS):
        if band != 4:
            tracker.update(band=band, timestep=99, hit=False)
    assert scheduler.select_band(100, tracker) == 4
