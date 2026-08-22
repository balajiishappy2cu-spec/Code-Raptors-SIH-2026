"""Band activity prediction: P(band transmits during the next prediction window).

The scheduler consumes these probabilities directly, so calibration matters as much as
ranking quality -- a model with a good ROC-AUC but badly scaled probabilities makes the
weighted score in :mod:`models.scheduler` meaningless. Isotonic calibration is fitted on
the validation split and reported alongside the usual classification metrics.

Prediction and decision are deliberately separate: this module only predicts, it never
chooses a band.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from common.logging_utils import get_logger
from features.band_features import FEATURE_NAMES

LOGGER = get_logger(__name__)


def cuda_available() -> bool:
    """Whether this XGBoost build can actually train on a CUDA device.

    Probed once with a tiny fit rather than trusted from configuration, because an
    ``xgboost`` wheel without GPU support and a machine without a driver both fail only
    at fit time.
    """
    global _CUDA_AVAILABLE
    if _CUDA_AVAILABLE is None:
        try:  # pragma: no cover - depends on the local machine
            import numpy as _np
            from xgboost import XGBClassifier

            probe = XGBClassifier(n_estimators=1, max_depth=1, tree_method="hist", device="cuda")
            probe.fit(_np.zeros((8, 2), dtype=_np.float32), _np.array([0, 1] * 4), verbose=False)
            _CUDA_AVAILABLE = True
        except Exception:  # pragma: no cover - depends on the local machine
            _CUDA_AVAILABLE = False
    return bool(_CUDA_AVAILABLE)


#: Cached result of :func:`cuda_available`.
_CUDA_AVAILABLE: bool | None = None



#: Classifiers the activity model can be built from. Every builder returns an
#: sklearn-compatible estimator exposing ``fit`` and ``predict_proba``, so the rest of the
#: pipeline -- calibration, saving, the scheduler protocol -- is unchanged by the choice.
#:
#: The linear and neural models are wrapped in a StandardScaler pipeline. The tree models
#: do not need it: they split on thresholds and are invariant to feature scaling.
MODEL_TYPES: tuple[str, ...] = ("xgboost", "logistic", "random_forest", "mlp")


def build_estimator(model_type: str, params: dict[str, Any]) -> Any:
    """Build one classifier by name.

    Args:
        model_type: one of :data:`MODEL_TYPES`.
        params: hyper-parameters passed to the underlying estimator.

    Returns:
        An unfitted sklearn-compatible classifier.

    Raises:
        ValueError: for an unknown model type.
    """
    kind = str(model_type).strip().lower()
    settings = dict(params)

    if kind == "xgboost":
        from xgboost import XGBClassifier

        settings.setdefault("objective", "binary:logistic")
        settings.setdefault("eval_metric", "logloss")
        device = str(settings.get("device", "cpu")).lower()
        if device.startswith("cuda") and not cuda_available():
            LOGGER.warning("device=%s requested but no usable CUDA device; using CPU", device)
            settings["device"] = "cpu"
        return XGBClassifier(**settings)

    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        settings.setdefault("max_iter", 300)
        return make_pipeline(StandardScaler(), LogisticRegression(**settings))

    if kind == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        settings.setdefault("n_jobs", -1)
        return RandomForestClassifier(**settings)

    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        settings.setdefault("hidden_layer_sizes", (64, 32))
        settings.setdefault("max_iter", 60)
        settings.setdefault("early_stopping", True)
        if isinstance(settings.get("hidden_layer_sizes"), list):
            settings["hidden_layer_sizes"] = tuple(settings["hidden_layer_sizes"])
        return make_pipeline(StandardScaler(), MLPClassifier(**settings))

    msg = f"Unknown activity model type {model_type!r}; expected one of {MODEL_TYPES}"
    raise ValueError(msg)


class ActivityPredictorProtocol(Protocol):
    """Minimal interface the scheduler needs from an activity model."""

    name: str

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return P(active in the next window) for each row of ``features``."""
        ...


@dataclass
class HeuristicActivityPredictor:
    """Fallback predictor used before an XGBoost model has been trained.

    Blends a band's windowed occupancy rate with its recent hit rate and gives unvisited
    bands an optimistic prior, so the Smart scheduler is runnable end to end from the
    very first milestone (and so the ablation has an "ML-free" arm).

    Attributes:
        unvisited_prior: probability assigned to a band that has never been observed.
        feature_names: feature layout this predictor expects.
    """

    unvisited_prior: float = 0.5
    feature_names: tuple[str, ...] = FEATURE_NAMES
    name: str = "heuristic"

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return heuristic activity probabilities for a feature matrix."""
        matrix = np.atleast_2d(np.asarray(features, dtype=np.float64))
        index = {name: i for i, name in enumerate(self.feature_names)}
        occupancy = matrix[:, index["occupancy_rate"]]
        recent = matrix[:, index["recent_hit_rate"]]
        never = matrix[:, index["never_visited"]] > 0.5
        probability = 0.5 * occupancy + 0.5 * recent
        probability = np.where(never, self.unvisited_prior, probability)
        return np.clip(probability, 0.0, 1.0)


@dataclass
class ActivityPredictor:
    """XGBoost classifier for next-window band activity.

    Attributes:
        params: XGBoost hyper-parameters from ``config.yaml``.
        prediction_window: how many timesteps ahead the target looks.
        calibrate: whether to fit isotonic calibration on the validation split.
        feature_names: feature layout, checked at predict time.
    """

    model_type: str = "xgboost"
    params: dict[str, Any] = field(default_factory=dict)
    prediction_window: int = 5
    calibrate: bool = True
    inference_device: str = "cpu"
    feature_names: tuple[str, ...] = FEATURE_NAMES
    name: str = "activity_model"
    model: Any = None
    calibrator: IsotonicRegression | None = None
    training_report: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls, model_cfg: dict[str, Any], model_type: str | None = None
    ) -> "ActivityPredictor":
        """Build an untrained predictor from the ``activity_model`` config section.

        Args:
            model_cfg: the ``activity_model`` configuration section.
            model_type: override the configured model type, used by the model
                comparison script to build several predictors from one config.
        """
        kind = str(model_type or model_cfg.get("model_type", "xgboost")).lower()
        if kind == "xgboost":
            params = dict(model_cfg.get("xgb_params", {}))
        else:
            params = dict(dict(model_cfg.get("model_params", {})).get(kind, {}))
        return cls(
            model_type=kind,
            params=params,
            prediction_window=int(model_cfg.get("prediction_window", 5)),
            calibrate=bool(model_cfg.get("calibrate", True)),
            inference_device=str(model_cfg.get("inference_device", "cpu")),
            name=kind,
        )

    def _build_model(self, scale_pos_weight: float | None = None) -> Any:
        """Instantiate the underlying classifier for :attr:`model_type`.

        ``scale_pos_weight`` is only applied to estimators that accept it; the others
        receive ``class_weight="balanced"`` where they support it instead.
        """
        params = dict(self.params)
        if scale_pos_weight is not None:
            if self.model_type == "xgboost":
                params.setdefault("scale_pos_weight", float(scale_pos_weight))
            elif self.model_type in {"logistic", "random_forest"}:
                params.setdefault("class_weight", "balanced")
        return build_estimator(self.model_type, params)

    def _move_to_inference_device(self) -> None:
        """Put the trained booster on the device used for prediction.

        Training is one large batch, which a GPU does well. Inference here is the
        opposite shape: the scheduler asks for ``n_bands`` rows once per decision,
        thousands of times per run, and a per-call host-to-device transfer costs far more
        than the kernel saves. So the booster is moved back to the CPU after training
        unless ``inference_device`` says otherwise.
        """
        if self.model is None or self.model_type != "xgboost":
            return
        target = str(self.inference_device).lower()
        try:
            self.model.get_booster().set_param({"device": target})
            self.model.set_params(device=target)
        except Exception as exc:  # pragma: no cover - depends on the xgboost build
            LOGGER.warning("Could not move the booster to %s (%s)", target, exc)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        *,
        balance_classes: bool = False,
    ) -> dict[str, Any]:
        """Train the classifier and, if configured, its probability calibrator.

        Args:
            x_train: ``(n_rows, n_features)`` training features.
            y_train: binary training targets.
            x_val: optional validation features (used for calibration and metrics).
            y_val: optional validation targets.
            balance_classes: pass a ``scale_pos_weight`` derived from the training split.

        Returns:
            A report dictionary with class balance and validation metrics.
        """
        x_train = np.asarray(x_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.int32)
        positives = int(y_train.sum())
        negatives = int(y_train.size - positives)
        if positives == 0 or negatives == 0:
            msg = (
                "Training targets contain a single class "
                f"(positives={positives}, negatives={negatives}); "
                "adjust activity_model.prediction_window or the environment configuration"
            )
            raise ValueError(msg)

        scale = (negatives / positives) if balance_classes else None
        self.model = self._build_model(scale_pos_weight=scale)
        train_device = str(self.model.get_params().get("device", "cpu"))
        if self.model_type != "xgboost":
            train_device = "cpu"
        start = time.perf_counter()
        # ``verbose`` is an XGBoost-specific fit argument; sklearn Pipelines reject
        # unknown fit parameters, so it is only passed where it is understood.
        if self.model_type == "xgboost":
            self.model.fit(x_train, y_train, verbose=False)
        else:
            self.model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - start
        self._move_to_inference_device()
        LOGGER.info(
            "Trained %s on device=%s in %.2fs", self.model_type, train_device, fit_seconds
        )

        report: dict[str, Any] = {
            "model_type": self.model_type,
            "train_device": train_device,
            "inference_device": self.inference_device,
            "fit_seconds": fit_seconds,
            "n_train_rows": int(y_train.size),
            "train_positive_rate": float(positives / y_train.size),
            "scale_pos_weight": scale,
            "prediction_window": self.prediction_window,
            "n_features": int(x_train.shape[1]),
        }

        if x_val is not None and y_val is not None and len(y_val) > 0:
            x_val = np.asarray(x_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.int32)
            raw = self.model.predict_proba(x_val)[:, 1]
            report["val_positive_rate"] = float(y_val.mean())
            report["uncalibrated"] = classification_report(y_val, raw)
            if self.calibrate and len(np.unique(y_val)) > 1:
                self.calibrator = IsotonicRegression(
                    y_min=0.0, y_max=1.0, out_of_bounds="clip"
                ).fit(raw, y_val)
                calibrated = self.calibrator.predict(raw)
                report["calibrated"] = classification_report(y_val, calibrated)
            else:
                self.calibrator = None

        self.training_report = report
        return report

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return calibrated P(active in the next window) for each row.

        Raises:
            RuntimeError: if the model has not been trained or loaded.
            ValueError: if the feature matrix width does not match training.
        """
        if self.model is None:
            msg = "ActivityPredictor.predict_proba called before fit/load"
            raise RuntimeError(msg)
        matrix = np.atleast_2d(np.asarray(features, dtype=np.float32))
        if matrix.shape[1] != len(self.feature_names):
            msg = (
                f"Expected {len(self.feature_names)} features, got {matrix.shape[1]}; "
                "the feature layout changed since training"
            )
            raise ValueError(msg)
        raw = self.model.predict_proba(matrix)[:, 1]
        if self.calibrator is not None:
            raw = self.calibrator.predict(raw)
        return np.clip(np.asarray(raw, dtype=np.float64), 0.0, 1.0)

    def feature_importances(self) -> dict[str, float]:
        """Return feature importances keyed by feature name."""
        if self.model is None:
            return {}
        estimator = self.model
        if hasattr(estimator, "steps"):  # a Pipeline -- take the final estimator
            estimator = estimator.steps[-1][1]
        if hasattr(estimator, "feature_importances_"):
            values = np.asarray(estimator.feature_importances_, dtype=np.float64)
        elif hasattr(estimator, "coef_"):
            # For a linear model, absolute standardised coefficients are the closest
            # analogue; they are normalised so the scale matches tree importances.
            values = np.abs(np.asarray(estimator.coef_, dtype=np.float64)).ravel()
            total = values.sum()
            values = values / total if total > 0 else values
        else:
            return {}
        if values.size != len(self.feature_names):
            return {}
        return {name: float(value) for name, value in zip(self.feature_names, values)}

    def save(self, path: str | Path) -> Path:
        """Persist the model, calibrator, feature layout and training report."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "calibrator": self.calibrator,
                "feature_names": list(self.feature_names),
                "model_type": self.model_type,
                "prediction_window": self.prediction_window,
                "inference_device": self.inference_device,
                "params": self.params,
                "training_report": self.training_report,
            },
            out,
        )
        LOGGER.info("Saved activity predictor to %s", out)
        return out

    @classmethod
    def load(cls, path: str | Path) -> "ActivityPredictor":
        """Load a predictor saved by :meth:`save`.

        Raises:
            FileNotFoundError: if the artifact does not exist.
        """
        artifact_path = Path(path)
        if not artifact_path.exists():
            msg = f"Activity predictor artifact not found: {artifact_path}"
            raise FileNotFoundError(msg)
        payload = joblib.load(artifact_path)
        predictor = cls(
            model_type=str(payload.get("model_type", "xgboost")),
            params=dict(payload.get("params", {})),
            prediction_window=int(payload.get("prediction_window", 5)),
            calibrate=payload.get("calibrator") is not None,
            inference_device=str(payload.get("inference_device", "cpu")),
            feature_names=tuple(payload.get("feature_names", FEATURE_NAMES)),
            name=str(payload.get("model_type", "xgboost")),
        )
        predictor.model = payload["model"]
        predictor._move_to_inference_device()
        predictor.calibrator = payload.get("calibrator")
        predictor.training_report = dict(payload.get("training_report", {}))
        return predictor


def classification_report(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Compute the metrics the MVP reports for the activity model.

    Args:
        y_true: binary targets.
        probability: predicted probabilities of the positive class.

    Returns:
        Dictionary with ROC-AUC, PR-AUC, precision, recall, F1, Brier score, log loss
        and a coarse calibration error.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1 - 1e-6)
    predictions = (probability >= 0.5).astype(np.int32)
    single_class = len(np.unique(y_true)) < 2
    return {
        "roc_auc": float("nan") if single_class else float(roc_auc_score(y_true, probability)),
        "pr_auc": float("nan")
        if single_class
        else float(average_precision_score(y_true, probability)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "brier": float(brier_score_loss(y_true, probability)),
        "log_loss": float("nan") if single_class else float(log_loss(y_true, probability)),
        "expected_calibration_error": expected_calibration_error(y_true, probability),
        "positive_rate": float(y_true.mean()),
        "predicted_positive_rate": float(predictions.mean()),
        "n_rows": int(y_true.size),
    }


def expected_calibration_error(
    y_true: np.ndarray, probability: np.ndarray, n_bins: int = 10
) -> float:
    """Return the bin-weighted absolute gap between confidence and accuracy."""
    y_true = np.asarray(y_true, dtype=np.float64)
    probability = np.asarray(probability, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.digitize(probability, edges[1:-1], right=False), 0, n_bins - 1)
    error = 0.0
    for b in range(n_bins):
        mask = bin_index == b
        if not mask.any():
            continue
        weight = float(mask.mean())
        error += weight * abs(float(probability[mask].mean()) - float(y_true[mask].mean()))
    return float(error)


def calibration_curve_points(
    y_true: np.ndarray, probability: np.ndarray, n_bins: int = 10
) -> dict[str, list[float]]:
    """Return mean predicted probability and observed frequency per bin, for plotting."""
    y_true = np.asarray(y_true, dtype=np.float64)
    probability = np.asarray(probability, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.digitize(probability, edges[1:-1], right=False), 0, n_bins - 1)
    predicted: list[float] = []
    observed: list[float] = []
    counts: list[float] = []
    for b in range(n_bins):
        mask = bin_index == b
        if not mask.any():
            continue
        predicted.append(float(probability[mask].mean()))
        observed.append(float(y_true[mask].mean()))
        counts.append(float(mask.sum()))
    return {"predicted": predicted, "observed": observed, "count": counts}
