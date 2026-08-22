"""Tests for the pluggable activity-model registry and the model comparison helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.band_features import FEATURE_NAMES  # noqa: E402
from models.activity_predictor import (  # noqa: E402
    MODEL_TYPES,
    ActivityPredictor,
    HeuristicActivityPredictor,
    build_estimator,
)
from scripts.compare_models import paired_interval  # noqa: E402

N_FEATURES = len(FEATURE_NAMES)


def make_dataset(n_rows: int = 900, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return a small separable dataset shaped like the real feature matrix."""
    rng = np.random.default_rng(seed)
    x = rng.random((n_rows, N_FEATURES)).astype(np.float32)
    # Make the first feature genuinely predictive so every model can learn something.
    y = (x[:, 0] + 0.25 * rng.standard_normal(n_rows) > 0.5).astype(np.int32)
    return x, y


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_every_model_type_builds(model_type: str) -> None:
    estimator = build_estimator(model_type, {})
    assert hasattr(estimator, "fit")
    assert hasattr(estimator, "predict_proba")


def test_unknown_model_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown activity model type"):
        build_estimator("not_a_model", {})


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_every_model_trains_and_predicts_in_range(model_type: str) -> None:
    x, y = make_dataset()
    params = {"mlp": {"max_iter": 8, "hidden_layer_sizes": (8,)},
              "random_forest": {"n_estimators": 10},
              "xgboost": {"n_estimators": 10},
              "logistic": {"max_iter": 50}}[model_type]
    predictor = ActivityPredictor(model_type=model_type, params=params, calibrate=False)
    predictor.fit(x[:700], y[:700], x[700:], y[700:])
    probability = predictor.predict_proba(x[:50])
    assert probability.shape == (50,)
    assert np.all((probability >= 0.0) & (probability <= 1.0))


def test_model_type_survives_save_and_load(tmp_path: Path) -> None:
    x, y = make_dataset()
    predictor = ActivityPredictor(
        model_type="random_forest", params={"n_estimators": 10}, calibrate=False
    )
    predictor.fit(x[:700], y[:700], x[700:], y[700:])
    before = predictor.predict_proba(x[:20])

    path = predictor.save(tmp_path / "model.joblib")
    reloaded = ActivityPredictor.load(path)
    assert reloaded.model_type == "random_forest"
    assert np.allclose(before, reloaded.predict_proba(x[:20]))


def test_predictors_satisfy_the_scheduler_protocol() -> None:
    """Anything the scheduler accepts needs only ``name`` and ``predict_proba``."""
    x, y = make_dataset()
    trained = ActivityPredictor(
        model_type="logistic", params={"max_iter": 50}, calibrate=False
    )
    trained.fit(x[:700], y[:700], x[700:], y[700:])
    for predictor in (HeuristicActivityPredictor(), trained):
        assert isinstance(getattr(predictor, "name"), str)
        assert predictor.predict_proba(x[:5]).shape == (5,)


def test_linear_model_reports_coefficient_importances() -> None:
    x, y = make_dataset()
    predictor = ActivityPredictor(
        model_type="logistic", params={"max_iter": 50}, calibrate=False
    )
    predictor.fit(x[:700], y[:700], x[700:], y[700:])
    importances = predictor.feature_importances()
    assert set(importances) == set(FEATURE_NAMES)
    assert sum(importances.values()) == pytest.approx(1.0, abs=1e-6)


def test_model_without_importances_returns_empty() -> None:
    x, y = make_dataset()
    predictor = ActivityPredictor(
        model_type="mlp",
        params={"max_iter": 8, "hidden_layer_sizes": (8,)},
        calibrate=False,
    )
    predictor.fit(x[:700], y[:700], x[700:], y[700:])
    assert predictor.feature_importances() == {}


def test_single_class_target_is_rejected_with_a_useful_message() -> None:
    x, _ = make_dataset()
    predictor = ActivityPredictor(model_type="logistic", params={}, calibrate=False)
    with pytest.raises(ValueError, match="single class"):
        predictor.fit(x, np.zeros(x.shape[0], dtype=np.int32))


# --- paired interval ----------------------------------------------------------------


def test_paired_interval_detects_a_real_difference() -> None:
    rng = np.random.default_rng(3)
    base = rng.random(40)
    mean, low, high = paired_interval(base + 0.05, base)
    assert mean == pytest.approx(0.05, abs=1e-9)
    assert low > 0  # a constant shift is unambiguous


def test_paired_interval_spans_zero_for_noise() -> None:
    rng = np.random.default_rng(4)
    mean, low, high = paired_interval(rng.random(40), rng.random(40))
    assert low < 0 < high


def test_paired_interval_needs_at_least_two_samples() -> None:
    mean, low, high = paired_interval(np.array([1.0]), np.array([0.5]))
    assert np.isnan(mean) and np.isnan(low) and np.isnan(high)


def test_paired_interval_ignores_non_finite_pairs() -> None:
    a = np.array([1.0, 2.0, np.nan, 4.0])
    b = np.array([0.5, 1.5, 2.0, 3.5])
    mean, low, high = paired_interval(a, b)
    assert mean == pytest.approx(0.5, abs=1e-9)
