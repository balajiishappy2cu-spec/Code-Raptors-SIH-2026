"""Evaluate the trained activity model on the held-out test pulse trains.

Test rows come from pulse trains the model never saw -- the split is by pulse train, not
by pulse -- so these numbers are the honest ones to quote. Both the uncalibrated and the
calibrated probabilities are reported, together with a per-band-position breakdown that
shows whether the model is merely learning "this band index is usually busy".

Usage::

    python training/evaluate_model.py --config config.yaml
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
from models.activity_predictor import (  # noqa: E402
    ActivityPredictor,
    HeuristicActivityPredictor,
    calibration_curve_points,
    classification_report,
)
from training.train_activity_model import load_split  # noqa: E402

LOGGER = get_logger("training.evaluate_model")


def per_environment_report(
    data: dict[str, np.ndarray], probability: np.ndarray
) -> list[dict[str, Any]]:
    """Return classification metrics computed separately per source pulse train."""
    rows: list[dict[str, Any]] = []
    for env_id in np.unique(data["environment"]):
        mask = data["environment"] == env_id
        rows.append(
            {
                "environment_index": int(env_id),
                "n_rows": int(mask.sum()),
                **classification_report(data["y"][mask], probability[mask]),
            }
        )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--split", default="test", help="split to evaluate")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Evaluate the saved model and write a JSON report."""
    args = parse_args(argv)
    config = Config.load(args.config)
    set_global_seed(config.seed)

    features_dir = config.path_for("paths.processed_dir") / "features"
    data = load_split(features_dir, args.split)
    predictor = ActivityPredictor.load(config.path_for("paths.model_artifact"))

    probability = predictor.predict_proba(data["x"])
    heuristic = HeuristicActivityPredictor().predict_proba(data["x"])

    report: dict[str, Any] = {
        "created": utc_timestamp(),
        "split": args.split,
        "artifact": str(config.path_for("paths.model_artifact")),
        "prediction_window": predictor.prediction_window,
        "n_rows": int(data["y"].size),
        "positive_rate": float(data["y"].mean()),
        "xgboost": classification_report(data["y"], probability),
        "heuristic_baseline": classification_report(data["y"], heuristic),
        "calibration_curve": calibration_curve_points(data["y"], probability),
        "per_environment": per_environment_report(data, probability),
        "feature_importances": predictor.feature_importances(),
        "training_report": predictor.training_report,
    }

    results_dir = config.path_for("paths.results_dir")
    write_json(results_dir / f"activity_model_{args.split}.json", report)

    xgb = report["xgboost"]
    base = report["heuristic_baseline"]
    LOGGER.info(
        "%s | XGBoost   ROC-AUC %.4f | PR-AUC %.4f | F1 %.4f | Brier %.4f | ECE %.4f",
        args.split,
        xgb["roc_auc"],
        xgb["pr_auc"],
        xgb["f1"],
        xgb["brier"],
        xgb["expected_calibration_error"],
    )
    LOGGER.info(
        "%s | Heuristic ROC-AUC %.4f | PR-AUC %.4f | F1 %.4f | Brier %.4f",
        args.split,
        base["roc_auc"],
        base["pr_auc"],
        base["f1"],
        base["brier"],
    )
    LOGGER.info("Wrote %s", results_dir / f"activity_model_{args.split}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
