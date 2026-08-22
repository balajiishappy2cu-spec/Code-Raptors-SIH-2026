"""Tests for the scorecard: the compressed 0-100 view of the figures of merit."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import HEADLINE_METRICS, METRIC_DESCRIPTIONS, describe_metric  # noqa: E402
from evaluation.scorecard import (  # noqa: E402
    activity_model_scorecard,
    oracle_ceiling,
    ratio_score,
    scheduler_scorecard,
)
from simulation.environment import EnvironmentGrid  # noqa: E402


def make_metrics(**overrides: float) -> dict[str, float]:
    """Return a plausible figures-of-merit dictionary with optional overrides."""
    base = {
        "average_intercept_rate": 0.30,
        "average_time_to_intercept_censored": 150.0,
        "active_band_coverage": 0.95,
        "average_intercept_time_error": 160.0,
        "percentage_of_correct_predictions": float("nan"),
    }
    base.update(overrides)
    return base


def make_environment(active: np.ndarray) -> EnvironmentGrid:
    """Build a minimal environment grid from an activity mask."""
    n_timesteps, n_bands = active.shape
    return EnvironmentGrid(
        active=active,
        n_pulses=active.astype(np.int32),
        mean_aoa=np.full(active.shape, np.nan),
        band_edges_mhz=np.linspace(2000.0, 18000.0, n_bands + 1),
        timestep_us=2000.0,
        t0_us=0.0,
        name="test",
    )


# --- ratio_score --------------------------------------------------------------------


def test_ratio_score_puts_parity_at_fifty() -> None:
    assert ratio_score(1.0) == pytest.approx(50.0)


def test_ratio_score_doubling_reaches_one_hundred() -> None:
    assert ratio_score(2.0) == pytest.approx(100.0)
    assert ratio_score(0.5) == pytest.approx(0.0)


def test_ratio_score_is_monotonic_and_clipped() -> None:
    values = [ratio_score(r) for r in (0.25, 0.5, 0.9, 1.0, 1.5, 2.0, 8.0)]
    assert values == sorted(values)
    assert min(values) >= 0.0
    assert max(values) <= 100.0


def test_ratio_score_rejects_undefined_ratios() -> None:
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        assert math.isnan(ratio_score(bad))


# --- scheduler scorecard ------------------------------------------------------------


def test_identical_strategies_score_parity() -> None:
    metrics = make_metrics(percentage_of_correct_predictions=50.0)
    card = scheduler_scorecard(metrics, dict(metrics))
    assert card.overall == pytest.approx(50.0, abs=1e-6)
    assert card.regressions == []


def test_a_better_strategy_scores_above_parity() -> None:
    baseline = make_metrics()
    candidate = make_metrics(
        average_intercept_rate=0.45,
        average_time_to_intercept_censored=110.0,
        active_band_coverage=0.98,
        average_intercept_time_error=100.0,
        percentage_of_correct_predictions=90.0,
    )
    card = scheduler_scorecard(baseline, candidate)
    assert card.overall > 50.0
    assert card.grade in {"A", "B", "C"}
    assert card.regressions == []


def test_a_worse_strategy_scores_below_parity() -> None:
    baseline = make_metrics()
    candidate = make_metrics(
        average_intercept_rate=0.18,
        average_time_to_intercept_censored=300.0,
        active_band_coverage=0.70,
        average_intercept_time_error=260.0,
        percentage_of_correct_predictions=40.0,
    )
    card = scheduler_scorecard(baseline, candidate)
    assert card.overall < 50.0
    assert set(card.regressions) == {"Interception", "Discovery", "Prediction"}


def test_regression_is_named_in_the_verdict_not_averaged_away() -> None:
    """A strategy that wins overall must still declare where it is worse."""
    baseline = make_metrics()
    candidate = make_metrics(
        average_intercept_rate=0.60,  # much better
        average_time_to_intercept_censored=200.0,  # worse
        active_band_coverage=0.95,
        average_intercept_time_error=100.0,
        percentage_of_correct_predictions=90.0,
    )
    card = scheduler_scorecard(baseline, candidate)
    assert card.overall > 50.0
    assert "Discovery" in card.regressions
    assert "worse on" in card.verdict
    assert "Discovery" in card.verdict


def test_components_carry_their_formula_and_inputs() -> None:
    card = scheduler_scorecard(make_metrics(), make_metrics())
    assert {c.key for c in card.components} == {"interception", "discovery", "prediction"}
    for component in card.components:
        assert component.formula
        assert component.detail


def test_missing_predictions_do_not_break_the_score() -> None:
    """The open-loop baseline makes no predictions; the card must still compute."""
    baseline = make_metrics()
    candidate = make_metrics(average_intercept_rate=0.40)
    card = scheduler_scorecard(baseline, candidate)
    assert np.isfinite(card.overall)


def test_scorecard_record_is_json_friendly() -> None:
    card = scheduler_scorecard(make_metrics(), make_metrics(average_intercept_rate=0.4))
    record = card.to_record()
    assert set(record) >= {"overall", "grade", "verdict", "components", "weights"}
    assert isinstance(record["components"], list)
    assert all(isinstance(component["score"], float) for component in record["components"])


# --- oracle ceiling -----------------------------------------------------------------


def test_oracle_ceiling_caps_at_one_band_per_timestep() -> None:
    # Every band active at every timestep: a one-band receiver can still only take one.
    active = np.ones((100, 8), dtype=bool)
    ceiling = oracle_ceiling(make_environment(active), detection_probability=1.0)
    assert ceiling["oracle_intercept_rate"] == pytest.approx(1.0)
    assert ceiling["oracle_probability_of_detection"] == pytest.approx(1.0 / 8.0)


def test_oracle_ceiling_respects_detection_probability() -> None:
    active = np.ones((50, 4), dtype=bool)
    ceiling = oracle_ceiling(make_environment(active), detection_probability=0.5)
    assert ceiling["oracle_intercept_rate"] == pytest.approx(0.5)


def test_oracle_ceiling_scales_with_instantaneous_bandwidth() -> None:
    active = np.ones((50, 8), dtype=bool)
    narrow = oracle_ceiling(make_environment(active), detection_probability=1.0)
    wide = oracle_ceiling(
        make_environment(active), detection_probability=1.0, instantaneous_bandwidth=4
    )
    assert wide["oracle_intercept_rate"] == pytest.approx(4.0 * narrow["oracle_intercept_rate"])


def test_oracle_ceiling_handles_a_silent_environment() -> None:
    ceiling = oracle_ceiling(
        make_environment(np.zeros((30, 5), dtype=bool)), detection_probability=0.9
    )
    assert ceiling["oracle_intercept_rate"] == pytest.approx(0.0)


def test_measured_rate_never_exceeds_the_ceiling() -> None:
    rng = np.random.default_rng(0)
    active = rng.random((200, 16)) < 0.3
    ceiling = oracle_ceiling(make_environment(active), detection_probability=0.95)
    # A real single-band receiver intercepts at most one active cell per timestep.
    assert ceiling["oracle_intercept_rate"] <= 0.95 + 1e-9


# --- activity model scorecard -------------------------------------------------------


def test_model_scorecard_grades_a_strong_model_highly() -> None:
    card = activity_model_scorecard(
        {"roc_auc": 0.95, "pr_auc": 0.94, "brier": 0.08, "expected_calibration_error": 0.01,
         "positive_rate": 0.42, "n_rows": 1000}
    )
    assert card.grade == "A"
    assert card.overall > 60.0


def test_model_scorecard_grades_a_chance_model_poorly() -> None:
    card = activity_model_scorecard(
        {"roc_auc": 0.51, "pr_auc": 0.42, "brier": 0.25, "expected_calibration_error": 0.20,
         "positive_rate": 0.42, "n_rows": 1000}
    )
    assert card.grade == "E"
    assert card.overall < 20.0


def test_model_grade_follows_roc_auc_not_calibration_alone() -> None:
    """A well-calibrated but non-discriminating model must not earn a top grade."""
    card = activity_model_scorecard(
        {"roc_auc": 0.55, "pr_auc": 0.45, "brier": 0.24, "expected_calibration_error": 0.0,
         "positive_rate": 0.42, "n_rows": 1000}
    )
    assert card.grade == "E"


# --- metric descriptions ------------------------------------------------------------


def test_every_headline_metric_has_a_description() -> None:
    for metric in HEADLINE_METRICS:
        assert metric in METRIC_DESCRIPTIONS
        assert len(describe_metric(metric)) > 20


def test_unknown_metric_gets_a_fallback_description() -> None:
    assert "No description" in describe_metric("not_a_real_metric")
