"""Randomised hyper-parameter search for the activity model.

The XGBoost settings in ``config.yaml`` were hand-picked at the start of the project and
never tuned, which is the one part of the model that had unexplored headroom. This
searches them properly.

Discipline, since it is easy to fool yourself here:

* candidates are **selected on the validation split**, never on test;
* the test split is scored **once**, for the selected configuration and the incumbent, so
  the reported number is not the maximum of many test evaluations;
* the incumbent is always evaluated alongside, so "improvement" is measured rather than
  assumed, and a search that finds nothing says so.

Selection uses validation PR-AUC rather than ROC-AUC: positives are the minority class
here, and PR-AUC is the more sensitive measure of ranking them correctly.

Usage::

    python scripts/tune_model.py --config config.yaml
    python scripts/tune_model.py --config config.yaml --trials 60 --model-type xgboost
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import Config, make_rng, set_global_seed  # noqa: E402
from common.io_utils import utc_timestamp, write_json  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402
from models.activity_predictor import ActivityPredictor, classification_report  # noqa: E402
from training.train_activity_model import load_split  # noqa: E402

LOGGER = get_logger("scripts.tune_model")

#: Search space. Ranges bracket the hand-picked incumbent on both sides so the search can
#: move in either direction rather than only away from it.
SEARCH_SPACE: dict[str, list[Any]] = {
    "n_estimators": [200, 300, 500, 800],
    "max_depth": [4, 5, 6, 8, 10],
    "learning_rate": [0.03, 0.05, 0.08, 0.12, 0.2],
    "min_child_weight": [1, 3, 5, 10],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.8, 0.9, 1.0],
    "reg_lambda": [0.5, 1.0, 3.0, 10.0],
}


def sample_params(rng: np.random.Generator, base: dict[str, Any]) -> dict[str, Any]:
    """Draw one candidate configuration, keeping the incumbent's fixed settings."""
    params = {
        key: value
        for key, value in base.items()
        if key not in SEARCH_SPACE
    }
    for key, choices in SEARCH_SPACE.items():
        params[key] = choices[int(rng.integers(0, len(choices)))]
    return params


def evaluate(
    params: dict[str, Any],
    splits: dict[str, dict[str, np.ndarray]],
    model_cfg: dict[str, Any],
    model_type: str,
) -> tuple[ActivityPredictor, dict[str, float], float]:
    """Train one configuration and score it on the validation split."""
    predictor = ActivityPredictor(
        model_type=model_type,
        params=params,
        prediction_window=int(model_cfg.get("prediction_window", 5)),
        calibrate=bool(model_cfg.get("calibrate", True)),
        inference_device=str(model_cfg.get("inference_device", "cpu")),
    )
    started = time.perf_counter()
    predictor.fit(
        splits["train"]["x"],
        splits["train"]["y"],
        splits["validation"]["x"],
        splits["validation"]["y"],
    )
    elapsed = time.perf_counter() - started
    probability = predictor.predict_proba(splits["validation"]["x"])
    return predictor, classification_report(splits["validation"]["y"], probability), elapsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--trials", type=int, default=40, help="random configurations to try")
    parser.add_argument("--model-type", default="xgboost", help="model family to tune")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="save the selected model over the shipped artifact if it beats the incumbent",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the search and report validation selection plus a single test scoring."""
    args = parse_args(argv)
    config = Config.load(args.config)
    set_global_seed(config.seed)

    features_dir = config.path_for("paths.processed_dir") / "features"
    splits = {name: load_split(features_dir, name) for name in ("train", "validation", "test")}
    model_cfg = config.section("activity_model")
    base_params = dict(model_cfg.get("xgb_params", {}))

    LOGGER.info(
        "train %d rows | validation %d rows | test %d rows",
        splits["train"]["y"].size,
        splits["validation"]["y"].size,
        splits["test"]["y"].size,
    )

    # --- incumbent -------------------------------------------------------------------
    incumbent, incumbent_val, incumbent_seconds = evaluate(
        base_params, splits, model_cfg, args.model_type
    )
    LOGGER.info(
        "incumbent      validation PR-AUC %.4f | ROC-AUC %.4f | fit %.1fs",
        incumbent_val["pr_auc"],
        incumbent_val["roc_auc"],
        incumbent_seconds,
    )

    # --- search ----------------------------------------------------------------------
    rng = make_rng(config.seed, stream=4242)
    trials: list[dict[str, Any]] = []
    best_params: dict[str, Any] | None = None
    best_predictor: ActivityPredictor | None = None
    best_score = float(incumbent_val["pr_auc"])

    for trial in range(int(args.trials)):
        params = sample_params(rng, base_params)
        try:
            predictor, report, elapsed = evaluate(params, splits, model_cfg, args.model_type)
        except Exception as exc:  # noqa: BLE001 - a bad draw must not end the search
            LOGGER.warning("trial %d failed (%s)", trial, exc)
            continue
        trials.append({"trial": trial, "params": params, "validation": report, "seconds": elapsed})
        if report["pr_auc"] > best_score:
            best_score = float(report["pr_auc"])
            best_params = params
            best_predictor = predictor
            LOGGER.info(
                "trial %2d  validation PR-AUC %.4f  NEW BEST  %s",
                trial,
                report["pr_auc"],
                {k: params[k] for k in SEARCH_SPACE},
            )
        elif trial % 10 == 0:
            LOGGER.info("trial %2d  validation PR-AUC %.4f", trial, report["pr_auc"])

    # --- single test scoring ----------------------------------------------------------
    incumbent_test = classification_report(
        splits["test"]["y"], incumbent.predict_proba(splits["test"]["x"])
    )
    record: dict[str, Any] = {
        "created": utc_timestamp(),
        "seed": config.seed,
        "model_type": args.model_type,
        "trials": len(trials),
        "search_space": {k: list(map(str, v)) for k, v in SEARCH_SPACE.items()},
        "selection_metric": "validation pr_auc",
        "note": (
            "Candidates selected on validation only. Test scored once for the incumbent "
            "and once for the selection, so the reported test figure is not the maximum "
            "of many test evaluations."
        ),
        "incumbent": {"params": base_params, "validation": incumbent_val, "test": incumbent_test},
        "all_trials": trials,
    }

    LOGGER.info("")
    if best_params is None:
        LOGGER.info("=== No configuration beat the incumbent on validation PR-AUC ===")
        LOGGER.info(
            "  incumbent test  ROC-AUC %.4f | PR-AUC %.4f | Brier %.4f",
            incumbent_test["roc_auc"],
            incumbent_test["pr_auc"],
            incumbent_test["brier"],
        )
        LOGGER.info("  The hand-picked settings are at or near the ceiling for this feature set.")
        record["selected"] = None
    else:
        assert best_predictor is not None
        selected_test = classification_report(
            splits["test"]["y"], best_predictor.predict_proba(splits["test"]["x"])
        )
        record["selected"] = {
            "params": best_params,
            "validation_pr_auc": best_score,
            "test": selected_test,
        }
        LOGGER.info("=== Selected on validation ===")
        LOGGER.info("  %s", {k: best_params[k] for k in SEARCH_SPACE})
        LOGGER.info(
            "  validation PR-AUC %.4f against incumbent %.4f",
            best_score,
            incumbent_val["pr_auc"],
        )
        LOGGER.info("=== Test, scored once ===")
        header = f"  {'metric':<28}{'incumbent':>12}{'selected':>12}{'change':>12}"
        LOGGER.info("%s", header)
        for key in ("roc_auc", "pr_auc", "f1", "brier", "expected_calibration_error"):
            before, after = incumbent_test[key], selected_test[key]
            LOGGER.info(
                "  %-28s%12.4f%12.4f%12.4f", key, before, after, after - before
            )

        if args.apply:
            artifact = config.path_for("paths.model_artifact")
            best_predictor.save(artifact)
            LOGGER.info("Applied: saved the selected model over %s", artifact.name)
        else:
            LOGGER.info("Not applied. Re-run with --apply to save it over the shipped artifact.")

    write_json(config.path_for("paths.results_dir") / "model_tuning.json", record)
    LOGGER.info("Wrote %s", config.path_for("paths.results_dir") / "model_tuning.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
