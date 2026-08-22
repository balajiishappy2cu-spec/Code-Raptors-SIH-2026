"""Run the open-loop Sequential Sweep baseline on its own.

This is the strategy the problem statement's background describes and criticises: sweep
the whole band as fast as possible, in fixed order, giving every band the same dwell
regardless of what is found there. Running it alone is useful as a sanity check and as
the reference every Smart Scan number is quoted against.

It needs no trained model, so it also works immediately after sampling the dataset.

Usage::

    python scripts/run_baseline.py --config config.yaml
    python scripts/run_baseline.py --config config.yaml --split test --timesteps 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import Config, make_rng, set_global_seed  # noqa: E402
from common.io_utils import utc_timestamp, write_json  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402
from dataio.manifest import DatasetManifest  # noqa: E402
from evaluation.metrics import HEADLINE_METRICS, RewardModel  # noqa: E402
from models.scheduler import build_scheduler  # noqa: E402
from simulation.runner import run_simulation  # noqa: E402
from simulation.scenarios import build_scenarios_for_split  # noqa: E402
from visualization import plots  # noqa: E402

LOGGER = get_logger("scripts.run_baseline")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--split", default="test", help="manifest split to run on")
    parser.add_argument("--timesteps", type=int, default=None, help="timesteps per run")
    parser.add_argument("--scenarios", default=None, help="comma-separated scenarios")
    parser.add_argument("--no-figures", action="store_true", help="skip figure generation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the baseline sweep and write its metrics."""
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

    scenarios = build_scenarios_for_split(
        manifest,
        args.split,
        config.section("environment"),
        scenario_names,
        max_pulses=int(config.get("data.max_pulses_per_train", 100_000)),
        max_timesteps=horizon,
    )
    reward_model = RewardModel.from_config(config.section("reward"))

    rows: list[dict[str, Any]] = []
    first_runs: dict[str, Any] = {}
    for scenario_name, items in scenarios.items():
        for index, scenario in enumerate(items):
            environment = scenario.environment
            if environment.n_timesteps == 0:
                continue
            scheduler = build_scheduler(
                "sequential",
                n_bands=environment.n_bands,
                rng=make_rng(config.seed, stream=5000 + index),
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
                record_decisions=(index == 0),
            )
            rows.append(
                {
                    "scenario": scenario_name,
                    "environment": environment.name,
                    "strategy": "sequential",
                    **run.metrics,
                }
            )
            if index == 0:
                first_runs[scenario_name] = (environment, run)

    if not rows:
        LOGGER.error("No environments were available; run scripts/sample_dataset.py first")
        return 1

    summary: dict[str, dict[str, float]] = {}
    for scenario_name in scenarios:
        subset = [row for row in rows if row["scenario"] == scenario_name]
        if not subset:
            continue
        summary[scenario_name] = {
            metric: float(np.nanmean([row.get(metric, np.nan) for row in subset]))
            for metric in HEADLINE_METRICS
        }

    figures: list[str] = []
    if not args.no_figures:
        for scenario_name, (environment, run) in first_runs.items():
            figures.append(
                str(
                    plots.plot_frequency_time_heatmap(
                        environment,
                        [run],
                        config.path_for("paths.figures_dir") / f"baseline_{scenario_name}.png",
                        time_window=(0, min(600, environment.n_timesteps)),
                        title=f"Sequential sweep baseline - {scenario_name}",
                    )
                )
            )

    write_json(
        config.path_for("paths.results_dir") / "baseline_results.json",
        {
            "created": utc_timestamp(),
            "seed": config.seed,
            "split": args.split,
            "timesteps": horizon,
            "data_source": manifest.source,
            "rows": rows,
            "summary": summary,
            "figures": figures,
        },
    )

    for scenario_name, metrics in summary.items():
        LOGGER.info("=== Sequential sweep | %s ===", scenario_name)
        for metric in HEADLINE_METRICS:
            LOGGER.info("  %-38s %10.4f", metric, metrics[metric])
    LOGGER.info("Wrote %s", config.path_for("paths.results_dir") / "baseline_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
