"""Compare activity models on identical splits, then through the identical scheduler.

A classification bake-off on its own would answer the wrong question. What matters here is
not which model separates active bands best, but whether a better classifier produces a
better *scheduler* -- and those are not the same thing. Earlier measurements showed a large
classification gap (ROC-AUC 0.880 to 0.980, heuristic against XGBoost) translating into a
small scheduling gap (+0.0050 intercept rate). This script measures both ends for every
model so the relationship can be read directly.

Controls:

* every model trains on the **same** train split and is scored on the **same** test split,
  both pinned by the sampler's manifest and split by pulse train, not by pulse;
* every model then drives the **same** scheduler over the **same** test environments with
  the **same** per-environment receiver seeds, so the only thing that differs is the
  predicted probability;
* differences in intercept rate are tested **paired per environment**, because environments
  vary far more than models do and an unpaired comparison would drown the effect.

Usage::

    python scripts/compare_models.py --config config.yaml
    python scripts/compare_models.py --config config.yaml --models xgboost,logistic
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import Config, make_rng, set_global_seed  # noqa: E402
from common.io_utils import utc_timestamp, write_json  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402
from dataio.manifest import DatasetManifest  # noqa: E402
from evaluation.metrics import RewardModel  # noqa: E402
from models.activity_predictor import (  # noqa: E402
    MODEL_TYPES,
    ActivityPredictor,
    HeuristicActivityPredictor,
    classification_report,
)
from models.scheduler import build_scheduler  # noqa: E402
from simulation.runner import run_simulation  # noqa: E402
from simulation.scenarios import build_scenarios_for_split  # noqa: E402
from training.train_activity_model import load_split  # noqa: E402

LOGGER = get_logger("scripts.compare_models")

#: The no-ML floor is included as a model so the table shows what the learning buys.
HEURISTIC_KEY = "heuristic"


def paired_interval(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Return the mean paired difference ``a - b`` and its 95% confidence interval."""
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    if diff.size < 2:
        return float("nan"), float("nan"), float("nan")
    mean = float(diff.mean())
    stderr = float(diff.std(ddof=1) / np.sqrt(diff.size))
    return mean, mean - 1.96 * stderr, mean + 1.96 * stderr


def train_and_score(
    config: Config,
    model_type: str,
    splits: dict[str, dict[str, np.ndarray]],
) -> tuple[Any, dict[str, Any]]:
    """Train one model on the train split and score it on the test split.

    Returns:
        ``(predictor, report)``; the heuristic is returned untrained.
    """
    if model_type == HEURISTIC_KEY:
        predictor = HeuristicActivityPredictor()
        probability = predictor.predict_proba(splits["test"]["x"])
        return predictor, {
            "model_type": HEURISTIC_KEY,
            "fit_seconds": 0.0,
            "classification": classification_report(splits["test"]["y"], probability),
            "feature_importances": {},
        }

    predictor = ActivityPredictor.from_config(
        config.section("activity_model"), model_type=model_type
    )
    started = time.perf_counter()
    predictor.fit(
        splits["train"]["x"],
        splits["train"]["y"],
        splits["validation"]["x"],
        splits["validation"]["y"],
    )
    fit_seconds = time.perf_counter() - started
    probability = predictor.predict_proba(splits["test"]["x"])

    artifact = config.path_for("paths.model_artifact").with_name(
        f"activity_predictor_{model_type}.joblib"
    )
    predictor.save(artifact)
    return predictor, {
        "model_type": model_type,
        "fit_seconds": fit_seconds,
        "artifact": str(artifact),
        "classification": classification_report(splits["test"]["y"], probability),
        "feature_importances": predictor.feature_importances(),
    }


def run_scheduler_with(
    config: Config,
    predictor: Any,
    environments: list[Any],
    horizon: int,
) -> tuple[dict[str, float], np.ndarray]:
    """Drive the Smart scheduler with one predictor over every test environment.

    Returns:
        ``(mean metrics, per-environment intercept rate)``.
    """
    reward_model = RewardModel.from_config(config.section("reward"))
    per_env: list[dict[str, float]] = []
    rates: list[float] = []

    for index, environment in enumerate(environments):
        scheduler = build_scheduler(
            "smart",
            n_bands=environment.n_bands,
            rng=make_rng(config.seed, stream=5000 + index),
            predictor=predictor,
            scheduler_cfg=config.section("scheduler"),
        )
        run = run_simulation(
            environment=environment,
            scheduler=scheduler,
            receiver_cfg=config.section("receiver"),
            features_cfg=config.section("features"),
            reward_model=reward_model,
            rng=make_rng(config.seed, stream=9000 + index),
            n_timesteps=horizon,
            prediction_window=int(config.get("activity_model.prediction_window", 5)),
            record_decisions=False,
        )
        per_env.append(run.metrics)
        rates.append(float(run.metrics["average_intercept_rate"]))

    keys = per_env[0].keys() if per_env else []
    mean = {}
    for key in keys:
        values = np.array([m[key] for m in per_env], dtype=np.float64)
        finite = values[np.isfinite(values)]
        mean[key] = float(finite.mean()) if finite.size else float("nan")
    return mean, np.asarray(rates, dtype=np.float64)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--split", default="test", help="split to evaluate the scheduler on")
    parser.add_argument(
        "--models",
        default=f"{HEURISTIC_KEY}," + ",".join(MODEL_TYPES),
        help="comma-separated model types to compare",
    )
    parser.add_argument("--timesteps", type=int, default=None, help="timesteps per run")
    parser.add_argument(
        "--reference",
        default="xgboost",
        help="model the paired confidence intervals are measured against",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Train every model, run each through the scheduler, and write the comparison."""
    args = parse_args(argv)
    config = Config.load(args.config)
    set_global_seed(config.seed)

    features_dir = config.path_for("paths.processed_dir") / "features"
    splits = {name: load_split(features_dir, name) for name in ("train", "validation", "test")}
    LOGGER.info(
        "Identical splits for every model: train %d rows, validation %d, test %d",
        splits["train"]["y"].size,
        splits["validation"]["y"].size,
        splits["test"]["y"].size,
    )

    manifest = DatasetManifest.load(config.path_for("paths.manifest"))
    horizon = int(args.timesteps or config.get("simulation.n_timesteps", 4000))
    scenarios = build_scenarios_for_split(
        manifest,
        args.split,
        config.section("environment"),
        list(config.get("simulation.scenarios", ["spatial_scan", "frequency_agile"])),
        max_pulses=int(config.get("data.max_pulses_per_train", 400_000)),
        max_timesteps=horizon,
    )
    environments = [
        scenario.environment
        for items in scenarios.values()
        for scenario in items
        if scenario.environment.n_timesteps > 0
    ]
    LOGGER.info("Identical %d test environments for every model", len(environments))

    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    rows: list[dict[str, Any]] = []
    rates: dict[str, np.ndarray] = {}

    for model_type in wanted:
        LOGGER.info("--- %s ---", model_type)
        predictor, report = train_and_score(config, model_type, splits)
        metrics, per_env_rate = run_scheduler_with(config, predictor, environments, horizon)
        rates[model_type] = per_env_rate
        classification = report["classification"]
        rows.append(
            {
                "model": model_type,
                "roc_auc": classification["roc_auc"],
                "pr_auc": classification["pr_auc"],
                "f1": classification["f1"],
                "brier": classification["brier"],
                "ece": classification["expected_calibration_error"],
                "fit_seconds": report["fit_seconds"],
                "intercept_rate": metrics["average_intercept_rate"],
                "average_reward": metrics["average_reward"],
                "intercept_time_error": metrics["average_intercept_time_error"],
                "tti_censored": metrics["average_time_to_intercept_censored"],
                "active_band_coverage": metrics["active_band_coverage"],
            }
        )
        LOGGER.info(
            "%-14s ROC-AUC %.4f | PR-AUC %.4f | intercept rate %.4f",
            model_type,
            classification["roc_auc"],
            classification["pr_auc"],
            metrics["average_intercept_rate"],
        )

    frame = pd.DataFrame(rows)
    reference = args.reference if args.reference in rates else wanted[-1]
    paired: list[dict[str, Any]] = []
    for model_type in wanted:
        if model_type == reference:
            continue
        mean, low, high = paired_interval(rates[model_type], rates[reference])
        paired.append(
            {
                "model": model_type,
                "vs": reference,
                "mean_difference": mean,
                "ci_low": low,
                "ci_high": high,
                "distinguishable": bool(np.isfinite(low) and (low > 0 or high < 0)),
            }
        )

    results_dir = config.path_for("paths.results_dir")
    frame.to_csv(results_dir / "model_comparison.csv", index=False)
    write_json(
        results_dir / "model_comparison.json",
        {
            "created": utc_timestamp(),
            "seed": config.seed,
            "split": args.split,
            "timesteps": horizon,
            "n_environments": len(environments),
            "n_rows": {name: int(splits[name]["y"].size) for name in splits},
            "reference_model": reference,
            "note": (
                "Every model trained on the identical train split and scored on the "
                "identical test split, then run through the identical scheduler over the "
                "identical environments with matched receiver seeds."
            ),
            "rows": rows,
            "paired_vs_reference": paired,
        },
    )

    LOGGER.info("")
    LOGGER.info("=== Classification vs scheduling ===")
    header = (
        f"{'model':<15}{'ROC-AUC':>9}{'PR-AUC':>9}{'Brier':>9}"
        f"{'rate':>9}{'reward':>9}{'err':>8}{'tti':>9}"
    )
    LOGGER.info("%s", header)
    LOGGER.info("%s", "-" * len(header))
    for row in rows:
        LOGGER.info(
            "%s",
            f"{row['model']:<15}{row['roc_auc']:>9.4f}{row['pr_auc']:>9.4f}{row['brier']:>9.4f}"
            f"{row['intercept_rate']:>9.4f}{row['average_reward']:>9.4f}"
            f"{row['intercept_time_error']:>8.1f}{row['tti_censored']:>9.1f}",
        )

    LOGGER.info("")
    LOGGER.info("=== Paired per-environment intercept rate, vs %s ===", reference)
    for row in paired:
        verdict = "distinguishable" if row["distinguishable"] else "indistinguishable"
        LOGGER.info(
            "  %-15s %+.4f  CI [%+.4f, %+.4f]  %s",
            row["model"],
            row["mean_difference"],
            row["ci_low"],
            row["ci_high"],
            verdict,
        )

    LOGGER.info("")
    LOGGER.info("Wrote %s", results_dir / "model_comparison.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
