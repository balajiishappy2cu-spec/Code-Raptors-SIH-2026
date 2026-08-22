"""Small grid search over the Smart Scan score weights.

The scheduler's weights are experimental parameters, and a single hand-picked set is easy
to argue with. This script makes the choice explicit and reproducible: it sweeps a grid on
the **validation** split only, scores each configuration against the Sequential baseline
on the same environments and seeds, and prints the frontier.

The selection rule used for the shipped defaults is stated rather than buried: take the
highest average reward among configurations that do not regress *discovery* -- censored
time-to-intercept at least ``DISCOVERY_MARGIN`` better than the baseline, and active-band
coverage at least as good. Discovery is what the problem statement's "absence of prior
reliable intelligence" framing is about, so a configuration that racks up hits on a handful
of known-busy bands while losing track of the rest is not an acceptable trade.

The margin is deliberate. A first version of this script accepted any configuration that
merely matched the baseline, and the borderline point it selected did not transfer: it met
the constraint on validation and then regressed censored time-to-intercept by 81% on the
test split. Requiring headroom on the split you tune on is what makes the choice hold up
on the split you report.

Test-split results are never used here.

Usage::

    python scripts/tune_weights.py --config config.yaml
    python scripts/tune_weights.py --config config.yaml --timesteps 1500 --top 20
"""

from __future__ import annotations

import argparse
import copy
import itertools
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import Config, make_rng, set_global_seed  # noqa: E402
from common.io_utils import utc_timestamp, write_json  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402
from dataio.manifest import DatasetManifest  # noqa: E402
from evaluation.metrics import RewardModel  # noqa: E402
from models.activity_predictor import ActivityPredictor, HeuristicActivityPredictor  # noqa: E402
from models.scheduler import build_scheduler  # noqa: E402
from simulation.runner import run_simulation  # noqa: E402
from simulation.scenarios import build_scenarios_for_split  # noqa: E402

LOGGER = get_logger("scripts.tune_weights")

#: Required headroom on censored time-to-intercept against the baseline, on the tuning
#: split. A configuration that only just meets the constraint here does not transfer.
DISCOVERY_MARGIN = 0.90

#: The grid searched. Kept small on purpose: this is an MVP, not an HPO study, and at 80
#: validation environments each configuration costs real wall-clock time.
#:
#: The axes are trimmed on measured sensitivity, not taste. A previous 48-configuration
#: sweep on real data gave, as mean average reward over feasible configurations:
#:
#: * ``staleness_saturation`` dominated (0.073 at 32 against -0.015 at 64), so only the
#:   useful range is kept;
#: * ``w2_exploration_bonus`` was monotone decreasing above 2.5, so 5.0 and 7.0 are dropped
#:   and 1.75 brackets the lower edge;
#: * ``w4_periodicity_bonus`` and ``w5_scan_cost`` moved reward by under 0.007 and are
#:   fixed at their better levels.
#:
#: ``max_revisit_interval`` is the new axis and the reason this search is worth rerunning.
#: The previous grid found **no configuration** meeting the discovery constraint, because a
#: weighted exploration bonus can always be outvoted by a band that looks productive -- so
#: no weighting could bound how long a band goes unwatched. A hard revisit deadline can. An
#: open-loop sweep covers 32 bands every 64 timesteps, so 64 is sweep-equivalent, lower is
#: stricter, higher is looser, and 0 keeps the old unconstrained behaviour so the search can
#: still reject the mechanism outright rather than being forced to adopt it.
GRID: dict[str, list[float]] = {
    "w2_exploration_bonus": [1.75, 2.5, 3.5],
    "w4_periodicity_bonus": [1.5],
    "w5_scan_cost": [0.1],
    "staleness_saturation": [32.0, 48.0],
    # A hard revisit deadline is the lever the weighted score could not provide. An
    # open-loop sweep covers 32 bands every 64 timesteps, so values at and below that are
    # stricter than the baseline and values above it are looser; 0 keeps the previous
    # unconstrained behaviour so the search can still reject the mechanism entirely.
    "max_revisit_interval": [0.0, 64.0, 128.0, 256.0],
}


def evaluate_config(
    config: Config,
    environments: list[Any],
    scheduler_cfg: dict[str, Any],
    predictor: Any,
    *,
    kind: str,
    horizon: int,
) -> dict[str, float]:
    """Run one scheduler configuration over every environment and average its metrics."""
    receiver_cfg = config.section("receiver")
    features_cfg = config.section("features")
    reward_model = RewardModel.from_config(config.section("reward"))
    prediction_window = int(config.get("activity_model.prediction_window", 5))

    per_env: list[dict[str, float]] = []
    for index, environment in enumerate(environments):
        scheduler = build_scheduler(
            kind,
            n_bands=environment.n_bands,
            rng=make_rng(config.seed, stream=5000 + index),
            predictor=predictor,
            scheduler_cfg=scheduler_cfg,
        )
        run = run_simulation(
            environment=environment,
            scheduler=scheduler,
            receiver_cfg=receiver_cfg,
            features_cfg=features_cfg,
            reward_model=reward_model,
            rng=make_rng(config.seed, stream=9000 + index),
            n_timesteps=horizon,
            prediction_window=prediction_window,
            record_decisions=False,
        )
        per_env.append(run.metrics)

    keys = per_env[0].keys() if per_env else []
    out: dict[str, float] = {}
    for key in keys:
        values = np.array([m[key] for m in per_env], dtype=np.float64)
        finite = values[np.isfinite(values)]
        out[key] = float(finite.mean()) if finite.size else float("nan")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--split", default="validation", help="split to tune on")
    parser.add_argument("--timesteps", type=int, default=1500, help="timesteps per run")
    parser.add_argument("--top", type=int, default=15, help="rows of the frontier to print")
    parser.add_argument(
        "--predictor",
        default="xgboost",
        choices=["xgboost", "heuristic"],
        help="activity model used during tuning",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the sweep and write the frontier to ``results/``."""
    args = parse_args(argv)
    config = Config.load(args.config)
    set_global_seed(config.seed)
    if args.split == "test":
        LOGGER.error("Refusing to tune on the test split; use validation")
        return 1

    manifest = DatasetManifest.load(config.path_for("paths.manifest"))
    scenario_names = list(config.get("simulation.scenarios", ["spatial_scan", "frequency_agile"]))
    scenarios = build_scenarios_for_split(
        manifest,
        args.split,
        config.section("environment"),
        scenario_names,
        max_pulses=int(config.get("data.max_pulses_per_train", 100_000)),
        max_timesteps=args.timesteps,
    )
    environments = [
        scenario.environment
        for items in scenarios.values()
        for scenario in items
        if scenario.environment.n_timesteps > 0
    ]
    if not environments:
        LOGGER.error("No environments built for split %s", args.split)
        return 1
    LOGGER.info("Tuning on %d environments from split %s", len(environments), args.split)

    predictor = (
        ActivityPredictor.load(config.path_for("paths.model_artifact"))
        if args.predictor == "xgboost"
        else HeuristicActivityPredictor()
    )
    baseline = evaluate_config(
        config, environments, config.section("scheduler"), None, kind="sequential", horizon=args.timesteps
    )
    LOGGER.info(
        "Baseline (sequential): reward %.4f | Pd %.4f | tti_censored %.1f | band coverage %.3f",
        baseline["average_reward"],
        baseline["probability_of_detection"],
        baseline["average_time_to_intercept_censored"],
        baseline["active_band_coverage"],
    )

    keys = list(GRID)
    rows: list[dict[str, Any]] = []
    combos = list(itertools.product(*(GRID[key] for key in keys)))
    LOGGER.info("Evaluating %d configurations", len(combos))
    for combo in combos:
        settings = dict(zip(keys, combo))
        scheduler_cfg = copy.deepcopy(config.section("scheduler"))
        for key, value in settings.items():
            if key in {"staleness_saturation", "max_revisit_interval"}:
                scheduler_cfg.setdefault("exploration", {})[key] = value
            else:
                scheduler_cfg.setdefault("weights", {})[key] = value
        metrics = evaluate_config(
            config, environments, scheduler_cfg, predictor, kind="smart", horizon=args.timesteps
        )
        feasible = bool(
            metrics["average_time_to_intercept_censored"]
            <= DISCOVERY_MARGIN * baseline["average_time_to_intercept_censored"]
            and metrics["active_band_coverage"] >= baseline["active_band_coverage"] - 1e-9
        )
        rows.append({**settings, "feasible": feasible, **metrics})

    rows.sort(key=lambda row: (row["feasible"], row["average_reward"]), reverse=True)
    header = (
        f"{'ok':<3}{'w2':>6}{'w5':>6}{'sat':>6}{'revisit':>9}"
        f"{'reward':>9}{'Pd':>9}{'rate':>8}{'tti_cens':>10}{'bandcov':>9}"
    )
    LOGGER.info("%s", header)
    LOGGER.info("%s", "-" * len(header))
    for row in rows[: args.top]:
        LOGGER.info(
            "%s",
            f"{'OK' if row['feasible'] else '':<3}"
            f"{row['w2_exploration_bonus']:>6}{row['w5_scan_cost']:>6}"
            f"{row['staleness_saturation']:>6}{row['max_revisit_interval']:>9}"
            f"{row['average_reward']:>9.4f}{row['probability_of_detection']:>9.4f}"
            f"{row['average_intercept_rate']:>8.3f}"
            f"{row['average_time_to_intercept_censored']:>10.1f}"
            f"{row['active_band_coverage']:>9.3f}",
        )

    best = next((row for row in rows if row["feasible"]), None)
    if best is None:
        LOGGER.warning(
            "No configuration met the discovery constraint on this split. "
            "The frontier above is the honest trade-off; report it rather than picking "
            "a point that only looks good on intercept rate."
        )
    else:
        LOGGER.info("")
        LOGGER.info(
            "Selected: w2=%s w5=%s staleness_saturation=%s max_revisit_interval=%s",
            best["w2_exploration_bonus"],
            best["w5_scan_cost"],
            best["staleness_saturation"],
            best["max_revisit_interval"],
        )

    write_json(
        config.path_for("paths.results_dir") / "weight_tuning.json",
        {
            "created": utc_timestamp(),
            "split": args.split,
            "timesteps": args.timesteps,
            "predictor": args.predictor,
            "n_environments": len(environments),
            "grid": GRID,
            "selection_rule": (
                "highest average_reward among configurations whose censored "
                f"time-to-intercept is at most {DISCOVERY_MARGIN:.2f} x the sequential "
                "baseline and whose active-band coverage does not regress"
            ),
            "discovery_margin": DISCOVERY_MARGIN,
            "baseline": baseline,
            "selected": best,
            "frontier": rows,
        },
    )
    LOGGER.info("Wrote %s", config.path_for("paths.results_dir") / "weight_tuning.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
