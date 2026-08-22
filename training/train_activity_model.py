"""Train the XGBoost band-activity model.

The model answers one question: does this band transmit at any point in the next
``prediction_window`` timesteps, given only what the receiver has observed so far?

Class balance is checked and reported before training, because with a badly chosen
prediction window the target collapses towards "inactive" and a ROC-AUC can look strong
for a model that is barely predicting anything. Probabilities are isotonically calibrated
on the validation split, since the scheduler consumes them directly.

Usage::

    python training/train_activity_model.py --config config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import Config, set_global_seed  # noqa: E402
from common.io_utils import utc_timestamp, write_json  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402
from features.band_features import FEATURE_NAMES  # noqa: E402
from models.activity_predictor import (  # noqa: E402
    ActivityPredictor,
    calibration_curve_points,
    classification_report,
)

LOGGER = get_logger("training.train_activity_model")


def load_split(features_dir: Path, split: str) -> dict[str, np.ndarray]:
    """Load one prepared split.

    Raises:
        FileNotFoundError: if the split has not been prepared yet.
    """
    path = features_dir / f"{split}.npz"
    if not path.exists():
        msg = (
            f"Prepared split not found: {path}. "
            "Run 'python training/prepare_dataset.py --config config.yaml' first."
        )
        raise FileNotFoundError(msg)
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument(
        "--balance-classes",
        action="store_true",
        help="apply scale_pos_weight derived from the training split",
    )
    parser.add_argument("--out", default=None, help="override the model artifact path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Train and persist the activity predictor."""
    args = parse_args(argv)
    config = Config.load(args.config)
    set_global_seed(config.seed)

    features_dir = config.path_for("paths.processed_dir") / "features"
    train = load_split(features_dir, "train")
    validation = load_split(features_dir, "validation")

    train_rate = float(train["y"].mean())
    val_rate = float(validation["y"].mean())
    LOGGER.info(
        "Training rows %d (positive rate %.3f) | validation rows %d (positive rate %.3f)",
        train["y"].size,
        train_rate,
        validation["y"].size,
        val_rate,
    )
    if train_rate < 0.02 or train_rate > 0.98:
        LOGGER.warning(
            "Training target is heavily imbalanced (positive rate %.3f); headline ranking "
            "metrics will be optimistic for a model that predicts one class.",
            train_rate,
        )

    predictor = ActivityPredictor.from_config(config.section("activity_model"))
    report = predictor.fit(
        train["x"],
        train["y"],
        validation["x"],
        validation["y"],
        balance_classes=args.balance_classes,
    )

    artifact_path = Path(args.out) if args.out else config.path_for("paths.model_artifact")
    predictor.save(artifact_path)

    validation_probability = predictor.predict_proba(validation["x"])
    report["validation_final"] = classification_report(validation["y"], validation_probability)
    report["calibration_curve"] = calibration_curve_points(
        validation["y"], validation_probability
    )
    report["feature_importances"] = predictor.feature_importances()
    report["feature_names"] = list(FEATURE_NAMES)
    report["artifact"] = str(artifact_path)
    report["created"] = utc_timestamp()
    report["seed"] = config.seed

    results_dir = config.path_for("paths.results_dir")
    write_json(results_dir / "activity_model_training.json", report)

    final: dict[str, Any] = report["validation_final"]
    LOGGER.info(
        "Validation | ROC-AUC %.4f | PR-AUC %.4f | F1 %.4f | Brier %.4f | ECE %.4f",
        final["roc_auc"],
        final["pr_auc"],
        final["f1"],
        final["brier"],
        final["expected_calibration_error"],
    )
    top = sorted(report["feature_importances"].items(), key=lambda kv: kv[1], reverse=True)[:6]
    LOGGER.info("Top features: %s", ", ".join(f"{name} {value:.3f}" for name, value in top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
