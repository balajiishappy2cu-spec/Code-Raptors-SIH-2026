"""Run the full Smart Scan experiment and write results and figures.

This is the controlled experiment: every strategy is run on identical environments with
identical seeds, over both scenarios the problem statement names, and scored with its
figures of merit. Nothing here tunes anything -- the scheduler weights and the model come
from ``config.yaml`` and the trained artifact.

Usage::

    python scripts/run_mvp.py --config config.yaml
    python scripts/run_mvp.py --config config.yaml --split test --timesteps 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import Config, set_global_seed  # noqa: E402
from common.io_utils import utc_timestamp, write_json  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402
from dataio.manifest import DatasetManifest  # noqa: E402
from evaluation.compare_strategies import (  # noqa: E402
    BASELINE_KEY,
    CANDIDATE_KEY,
    DEFAULT_STRATEGIES,
    ablation_table,
    format_comparison_table,
    run_strategy_matrix,
)
from evaluation.metrics import HEADLINE_METRICS  # noqa: E402
from evaluation.scorecard import (  # noqa: E402
    activity_model_scorecard,
    oracle_ceiling,
    scheduler_scorecard,
)
from simulation.scenarios import SCENARIO_LABELS, build_scenarios_for_split  # noqa: E402
from visualization import plots  # noqa: E402

LOGGER = get_logger("scripts.run_mvp")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--split", default="test", help="manifest split to evaluate on")
    parser.add_argument("--timesteps", type=int, default=None, help="timesteps per run")
    parser.add_argument(
        "--max-trains", type=int, default=None, help="limit pulse trains from the split"
    )
    parser.add_argument(
        "--scenarios", default=None, help="comma-separated scenarios to run"
    )
    parser.add_argument(
        "--heatmap-window", type=int, default=600, help="timesteps shown in the heatmap"
    )
    parser.add_argument("--no-figures", action="store_true", help="skip figure generation")
    return parser.parse_args(argv)


def make_figures(
    config: Config,
    result: Any,
    scenarios_by_name: dict[str, list],
    *,
    heatmap_window: int,
) -> list[str]:
    """Generate every figure for the experiment and return their paths."""
    figures_dir = config.path_for("paths.figures_dir")
    written: list[str] = []

    for scenario_name, scenario_list in scenarios_by_name.items():
        if not scenario_list:
            continue
        environment = scenario_list[0].environment
        runs = [
            result.runs[key]
            for key in (f"{scenario_name}:{BASELINE_KEY}", f"{scenario_name}:{CANDIDATE_KEY}")
            if key in result.runs
        ]
        if not runs:
            continue
        written.append(
            str(
                plots.plot_frequency_time_heatmap(
                    environment,
                    runs,
                    figures_dir / f"heatmap_{scenario_name}.png",
                    time_window=(0, heatmap_window),
                    title=(
                        f"{SCENARIO_LABELS.get(scenario_name, scenario_name)} - "
                        f"{environment.name}"
                    ),
                )
            )
        )
        written.append(
            str(
                plots.plot_learning_curves(
                    runs,
                    figures_dir / f"learning_curve_{scenario_name}.png",
                    title=f"Reward and hit rate - {SCENARIO_LABELS.get(scenario_name, scenario_name)}",
                )
            )
        )
        written.append(
            str(
                plots.plot_band_occupancy(
                    environment,
                    runs,
                    figures_dir / f"dwell_allocation_{scenario_name}.png",
                    title=f"Dwell time allocation - {scenario_name}",
                )
            )
        )
        smart_run = result.runs.get(f"{scenario_name}:{CANDIDATE_KEY}")
        if smart_run is not None and smart_run.decisions:
            written.append(
                str(
                    plots.plot_decision_timeline(
                        smart_run, figures_dir / f"decision_timeline_{scenario_name}.png"
                    )
                )
            )
        if scenario_name in result.aggregated:
            written.append(
                str(
                    plots.plot_metric_comparison(
                        {
                            key: result.aggregated[scenario_name][key]
                            for key in (BASELINE_KEY, CANDIDATE_KEY)
                            if key in result.aggregated[scenario_name]
                        },
                        figures_dir / f"metrics_{scenario_name}.png",
                        title=f"Figures of merit - {scenario_name}",
                    )
                )
            )

    if result.overall:
        written.append(
            str(
                plots.plot_metric_comparison(
                    result.overall,
                    figures_dir / "metrics_all_strategies.png",
                    title="Figures of merit - all strategies, both scenarios",
                )
            )
        )
    if result.aggregated:
        for metric in ("average_intercept_rate", "probability_of_detection", "average_reward"):
            written.append(
                str(
                    plots.plot_scenario_summary(
                        result.aggregated,
                        config.path_for("paths.figures_dir") / f"scenario_{metric}.png",
                        metric=metric,
                    )
                )
            )

    training_report_path = config.path_for("paths.results_dir") / "activity_model_training.json"
    if training_report_path.exists():
        from common.io_utils import read_json

        report = read_json(training_report_path)
        curve = report.get("calibration_curve")
        if curve:
            written.append(
                str(plots.plot_calibration(curve, figures_dir / "activity_model_calibration.png"))
            )
    return written


def build_scorecards(
    config: Config,
    result: Any,
    scenarios_by_name: dict[str, list],
    horizon: int,
) -> dict[str, Any]:
    """Grade the scheduler and the activity model, with the interception ceiling.

    Args:
        config: loaded configuration.
        result: the strategy matrix result.
        scenarios_by_name: the scenario environments the experiment ran on.
        horizon: run length in timesteps.

    Returns:
        A record with the scheduler scorecard, the activity model scorecard and the
        oracle ceiling averaged across environments.
    """
    receiver_cfg = config.section("receiver")
    ceilings = [
        oracle_ceiling(
            scenario.environment,
            detection_probability=float(receiver_cfg.get("detection_probability", 0.95)),
            instantaneous_bandwidth=int(receiver_cfg.get("instantaneous_bandwidth", 1)),
            horizon=horizon,
        )
        for items in scenarios_by_name.values()
        for scenario in items
        if scenario.environment.n_timesteps > 0
    ]
    oracle = (
        {key: float(np.nanmean([c[key] for c in ceilings])) for key in ceilings[0]}
        if ceilings
        else {}
    )

    record: dict[str, Any] = {"oracle_ceiling": oracle}
    if BASELINE_KEY in result.overall and CANDIDATE_KEY in result.overall:
        record["scheduler"] = scheduler_scorecard(
            result.overall[BASELINE_KEY], result.overall[CANDIDATE_KEY], oracle=oracle
        ).to_record()

    model_report = config.path_for("paths.results_dir") / "activity_model_test.json"
    if model_report.exists():
        from common.io_utils import read_json

        report = read_json(model_report)
        record["activity_model"] = activity_model_scorecard(
            {
                **report.get("xgboost", {}),
                "positive_rate": report.get("positive_rate", float("nan")),
                "n_rows": report.get("n_rows", float("nan")),
            }
        ).to_record()
    return record


def main(argv: list[str] | None = None) -> int:
    """Run the experiment end to end."""
    args = parse_args(argv)
    config = Config.load(args.config)
    set_global_seed(config.seed)

    manifest = DatasetManifest.load(config.path_for("paths.manifest"))
    scenario_names = (
        [s.strip() for s in args.scenarios.split(",") if s.strip()]
        if args.scenarios
        else list(config.get("simulation.scenarios", ["spatial_scan", "frequency_agile"]))
    )
    horizon = int(args.timesteps or config.get("simulation.n_timesteps", 2000))

    scenarios_by_name = build_scenarios_for_split(
        manifest,
        args.split,
        config.section("environment"),
        scenario_names,
        max_pulses=int(config.get("data.max_pulses_per_train", 100_000)),
        max_timesteps=horizon,
        max_trains=args.max_trains,
    )
    counts = {name: len(items) for name, items in scenarios_by_name.items()}
    LOGGER.info("Built scenario environments from split %s: %s", args.split, counts)
    if not any(counts.values()):
        LOGGER.error("No scenario environments were built; run scripts/sample_dataset.py first")
        return 1

    result = run_strategy_matrix(
        config, scenarios_by_name, strategies=DEFAULT_STRATEGIES, n_timesteps=horizon
    )

    results_dir = config.path_for("paths.results_dir")
    frame = pd.DataFrame(result.rows)
    frame.to_csv(results_dir / "experiment_rows.csv", index=False)

    record: dict[str, Any] = {
        "created": utc_timestamp(),
        "seed": config.seed,
        "split": args.split,
        "timesteps": horizon,
        "scenarios": scenario_names,
        "environments_per_scenario": counts,
        "data_source": manifest.source,
        "manifest": str(config.path_for("paths.manifest")),
        "strategies": [
            {"key": spec.key, "description": spec.description} for spec in DEFAULT_STRATEGIES
        ],
        "config": {
            "environment": config.section("environment"),
            "receiver": config.section("receiver"),
            "features": config.section("features"),
            "scheduler": config.section("scheduler"),
            "reward": config.section("reward"),
            "activity_model": {
                key: value
                for key, value in config.section("activity_model").items()
                if key != "xgb_params"
            },
        },
        "ablation": ablation_table(result.overall),
        **result.to_record(),
    }

    record["scorecards"] = build_scorecards(config, result, scenarios_by_name, horizon)

    figures: list[str] = []
    if not args.no_figures:
        figures = make_figures(
            config, result, scenarios_by_name, heatmap_window=args.heatmap_window
        )
    record["figures"] = figures
    write_json(results_dir / "experiment_results.json", record)

    for scenario_name, comparison in result.comparison.items():
        LOGGER.info("")
        LOGGER.info("=== %s: %s vs %s ===", scenario_name, BASELINE_KEY, CANDIDATE_KEY)
        for line in format_comparison_table(comparison).splitlines():
            LOGGER.info("%s", line)

    LOGGER.info("")
    LOGGER.info("=== Ablation (mean over both scenarios) ===")
    header = f"{'strategy':<18}" + "".join(
        f"{metric[:16]:>18}" for metric in HEADLINE_METRICS[:5]
    )
    LOGGER.info("%s", header)
    for row in record["ablation"]:
        LOGGER.info(
            "%s",
            f"{row['strategy']:<18}"
            + "".join(f"{float(row[metric]):>18.4f}" for metric in HEADLINE_METRICS[:5]),
        )

    scorecards = record["scorecards"]
    if "scheduler" in scorecards:
        card = scorecards["scheduler"]
        LOGGER.info("")
        LOGGER.info("=== Scorecard ===")
        LOGGER.info(
            "Scheduler: grade %s, %.1f/100  (%s)",
            card["grade"],
            card["overall"],
            card["context"]["scale"],
        )
        for component in card["components"]:
            LOGGER.info(
                "  %-14s %5.1f  %s", component["label"], component["score"], component["detail"]
            )
        if card["regressions"]:
            LOGGER.info("  Worse than the baseline on: %s", ", ".join(card["regressions"]))
        ceiling = card["context"].get("fraction_of_ceiling")
        if ceiling is not None and np.isfinite(ceiling):
            LOGGER.info(
                "  Reaches %.1f%% of the single-band interception ceiling "
                "(baseline reaches %.1f%%)",
                100.0 * ceiling,
                100.0 * card["context"].get("baseline_fraction_of_ceiling", float("nan")),
            )
    if "activity_model" in scorecards:
        model = scorecards["activity_model"]
        LOGGER.info("Activity model: grade %s, %.1f/100", model["grade"], model["overall"])

    LOGGER.info("")
    LOGGER.info("Wrote %s", results_dir / "experiment_results.json")
    LOGGER.info("Wrote %s", results_dir / "experiment_rows.csv")
    if figures:
        LOGGER.info("Wrote %d figures to %s", len(figures), config.path_for("paths.figures_dir"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
