"""The controlled experiment: every strategy on identical environments and seeds.

For a comparison to mean anything, the only thing allowed to differ between strategies is
the scheduling decision. So each ``(environment, strategy)`` pair gets a receiver seeded
from the environment index alone -- the detection and false-alarm draws are therefore the
same sequence for every strategy on that environment -- and the scheduler's own generator
is seeded from the environment index too, so a rerun reproduces the run exactly.

Strategies compared:

* ``sequential``      -- the open-loop baseline sweep.
* ``random``          -- a control that shows the baseline is not trivially beaten.
* ``smart_heuristic`` -- Smart Scan with the no-ML occupancy predictor + Thompson Sampling.
* ``smart_ml_only``   -- Smart Scan with XGBoost but no Thompson Sampling.
* ``smart``           -- the full strategy: XGBoost + Thompson Sampling.

The last three form the ablation: they isolate what the learned model adds and what the
online hit/miss adaptation adds.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from common.config import Config, make_rng
from common.logging_utils import get_logger
from evaluation.metrics import (
    HEADLINE_METRICS,
    RewardModel,
    compare_metric_tables,
    relative_improvement,
)
from models.activity_predictor import (
    ActivityPredictor,
    ActivityPredictorProtocol,
    HeuristicActivityPredictor,
)
from models.scheduler import Scheduler, build_scheduler
from simulation.runner import SimulationRun, run_simulation
from simulation.scenarios import Scenario

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class StrategySpec:
    """How to build one strategy under test.

    Attributes:
        key: name used in tables and figures.
        kind: underlying scheduler kind (``sequential``, ``random`` or ``smart``).
        predictor: ``xgboost``, ``heuristic`` or ``none``.
        thompson: whether Thompson Sampling is enabled (``smart`` only). ``None`` means
            "use whatever ``config.yaml`` says", which is what the headline ``smart``
            strategy does so that it always reflects the shipped default. The ablation
            arms pin it explicitly, because an ablation must control the thing it varies.
        description: one-line description for the results record.
    """

    key: str
    kind: str
    predictor: str = "none"
    thompson: bool | None = True
    description: str = ""


#: The strategies compared in the headline experiment and the ablation.
DEFAULT_STRATEGIES: tuple[StrategySpec, ...] = (
    StrategySpec(
        key="sequential",
        kind="sequential",
        description="Open-loop sequential sweep (baseline)",
    ),
    StrategySpec(key="random", kind="random", description="Uniform random band selection"),
    StrategySpec(
        key="smart_heuristic",
        kind="smart",
        predictor="heuristic",
        thompson=True,
        description="Smart Scan without the learned model (occupancy heuristic + Thompson)",
    ),
    StrategySpec(
        key="smart_ml_only",
        kind="smart",
        predictor="xgboost",
        thompson=False,
        description="Smart Scan with XGBoost but no online Thompson Sampling",
    ),
    StrategySpec(
        key="smart",
        kind="smart",
        predictor="xgboost",
        thompson=None,  # follows config.yaml -- currently disabled, see scheduler.thompson
        description="Smart Scan as shipped: XGBoost, Thompson Sampling per config.yaml",
    ),
)

#: The two strategies quoted as the headline comparison.
BASELINE_KEY = "sequential"
CANDIDATE_KEY = "smart"


@dataclass
class ExperimentResult:
    """Outcome of a strategy matrix run.

    Attributes:
        rows: one record per ``(scenario, environment, strategy)``.
        aggregated: ``{scenario: {strategy: mean metrics}}``.
        overall: ``{strategy: mean metrics}`` across every scenario.
        comparison: baseline vs candidate table per scenario.
        runs: the retained :class:`SimulationRun` objects, keyed for plotting.
        scenario_summaries: provenance for each scenario instance used.
    """

    rows: list[dict[str, Any]] = field(default_factory=list)
    aggregated: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    overall: dict[str, dict[str, float]] = field(default_factory=dict)
    comparison: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    runs: dict[str, SimulationRun] = field(default_factory=dict)
    scenario_summaries: list[dict[str, Any]] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-friendly record (without the heavy run objects)."""
        return {
            "rows": self.rows,
            "aggregated": self.aggregated,
            "overall": self.overall,
            "comparison": self.comparison,
            "scenarios": self.scenario_summaries,
        }


def load_predictor(config: Config, kind: str) -> ActivityPredictorProtocol | None:
    """Load the predictor a strategy needs.

    Args:
        config: loaded configuration.
        kind: ``xgboost``, ``heuristic`` or ``none``.

    Returns:
        The predictor, or ``None`` when the strategy needs none.

    Raises:
        FileNotFoundError: if the XGBoost artifact has not been trained yet.
    """
    if kind == "xgboost":
        return ActivityPredictor.load(config.path_for("paths.model_artifact"))
    if kind == "heuristic":
        return HeuristicActivityPredictor()
    return None


def build_strategy_scheduler(
    spec: StrategySpec,
    n_bands: int,
    rng: np.random.Generator,
    scheduler_cfg: dict[str, Any],
    predictor: ActivityPredictorProtocol | None,
) -> Scheduler:
    """Instantiate the scheduler described by a :class:`StrategySpec`."""
    cfg = copy.deepcopy(scheduler_cfg)
    if spec.kind == "smart" and spec.thompson is not None:
        cfg.setdefault("thompson", {})
        cfg["thompson"] = {**cfg.get("thompson", {}), "enabled": bool(spec.thompson)}
    return build_scheduler(spec.kind, n_bands=n_bands, rng=rng, predictor=predictor, scheduler_cfg=cfg)


def run_strategy_matrix(
    config: Config,
    scenarios_by_name: dict[str, list[Scenario]],
    *,
    strategies: Sequence[StrategySpec] = DEFAULT_STRATEGIES,
    n_timesteps: int | None = None,
    keep_runs_for: Iterable[str] | None = None,
) -> ExperimentResult:
    """Run every strategy on every scenario environment with matched seeds.

    Args:
        config: loaded configuration.
        scenarios_by_name: ``{scenario name: [Scenario, ...]}``.
        strategies: strategies to compare.
        n_timesteps: run length; defaults to ``simulation.n_timesteps``.
        keep_runs_for: scenario names whose runs should be retained for plotting
            (the first environment of each is kept).

    Returns:
        The assembled :class:`ExperimentResult`.
    """
    receiver_cfg = config.section("receiver")
    features_cfg = config.section("features")
    scheduler_cfg = config.section("scheduler")
    reward_model = RewardModel.from_config(config.section("reward"))
    prediction_window = int(config.get("activity_model.prediction_window", 5))
    horizon = int(n_timesteps or config.get("simulation.n_timesteps", 2000))
    keep = set(keep_runs_for or scenarios_by_name)

    predictors = {
        kind: load_predictor(config, kind)
        for kind in {spec.predictor for spec in strategies}
        if kind != "none"
    }

    result = ExperimentResult()
    for scenario_name, scenario_list in scenarios_by_name.items():
        for env_index, scenario in enumerate(scenario_list):
            environment = scenario.environment
            if environment.n_timesteps == 0:
                LOGGER.warning("Skipping empty environment %s", environment.name)
                continue
            result.scenario_summaries.append(scenario.summary())

            for spec in strategies:
                scheduler = build_strategy_scheduler(
                    spec,
                    n_bands=environment.n_bands,
                    rng=make_rng(config.seed, stream=5000 + env_index),
                    scheduler_cfg=scheduler_cfg,
                    predictor=predictors.get(spec.predictor),
                )
                # Identical receiver seed for every strategy on this environment.
                run = run_simulation(
                    environment=environment,
                    scheduler=scheduler,
                    receiver_cfg=receiver_cfg,
                    features_cfg=features_cfg,
                    reward_model=reward_model,
                    rng=make_rng(config.seed, stream=9000 + env_index),
                    n_timesteps=horizon,
                    prediction_window=prediction_window,
                    record_decisions=(env_index == 0 and scenario_name in keep),
                )
                run.strategy = spec.key
                result.rows.append(
                    {
                        "scenario": scenario_name,
                        "environment": environment.name,
                        "environment_index": env_index,
                        "strategy": spec.key,
                        "description": spec.description,
                        **run.metrics,
                    }
                )
                if env_index == 0 and scenario_name in keep:
                    result.runs[f"{scenario_name}:{spec.key}"] = run

    result.aggregated = _aggregate(result.rows, group_key="scenario")
    result.overall = _aggregate_overall(result.rows)
    for scenario_name, per_strategy in result.aggregated.items():
        if BASELINE_KEY in per_strategy and CANDIDATE_KEY in per_strategy:
            result.comparison[scenario_name] = compare_metric_tables(
                per_strategy[BASELINE_KEY], per_strategy[CANDIDATE_KEY]
            )
    if BASELINE_KEY in result.overall and CANDIDATE_KEY in result.overall:
        result.comparison["overall"] = compare_metric_tables(
            result.overall[BASELINE_KEY], result.overall[CANDIDATE_KEY]
        )
    return result


def _metric_keys(rows: list[dict[str, Any]]) -> list[str]:
    """Return the numeric metric keys present in the result rows."""
    skip = {"scenario", "environment", "environment_index", "strategy", "description"}
    keys: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in skip and isinstance(value, (int, float)) and key not in keys:
                keys.append(key)
    return keys


def _mean_metrics(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, float]:
    """Average metrics across rows, ignoring undefined values."""
    out: dict[str, float] = {}
    for key in keys:
        values = np.array([float(row.get(key, np.nan)) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        out[key] = float(finite.mean()) if finite.size else float("nan")
    return out


def _aggregate(
    rows: list[dict[str, Any]], group_key: str
) -> dict[str, dict[str, dict[str, float]]]:
    """Average metrics per ``(group, strategy)``."""
    keys = _metric_keys(rows)
    groups = sorted({str(row[group_key]) for row in rows})
    out: dict[str, dict[str, dict[str, float]]] = {}
    for group in groups:
        out[group] = {}
        strategies = sorted({str(row["strategy"]) for row in rows if str(row[group_key]) == group})
        for strategy in strategies:
            subset = [
                row
                for row in rows
                if str(row[group_key]) == group and str(row["strategy"]) == strategy
            ]
            out[group][strategy] = _mean_metrics(subset, keys)
    return out


def _aggregate_overall(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Average metrics per strategy across every scenario."""
    keys = _metric_keys(rows)
    return {
        strategy: _mean_metrics([row for row in rows if row["strategy"] == strategy], keys)
        for strategy in sorted({str(row["strategy"]) for row in rows})
    }


def format_comparison_table(comparison: list[dict[str, Any]]) -> str:
    """Render a baseline-vs-candidate comparison as an aligned text table."""
    header = f"{'metric':<38} {'sequential':>12} {'smart':>12} {'improvement':>12}"
    lines = [header, "-" * len(header)]
    for row in comparison:
        improvement = row["improvement"]
        arrow = "" if not np.isfinite(improvement) else ("+" if improvement >= 0 else "")
        lines.append(
            f"{row['metric']:<38} {row['baseline']:>12.4f} {row['candidate']:>12.4f} "
            f"{arrow}{improvement:>11.1%}"
            if np.isfinite(improvement)
            else f"{row['metric']:<38} {row['baseline']:>12.4f} {row['candidate']:>12.4f} "
            f"{'n/a':>12}"
        )
    return "\n".join(lines)


def ablation_table(overall: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    """Build the ablation table: each strategy against the sequential baseline."""
    if BASELINE_KEY not in overall:
        return []
    baseline = overall[BASELINE_KEY]
    rows: list[dict[str, Any]] = []
    for strategy, metrics in overall.items():
        row: dict[str, Any] = {"strategy": strategy}
        for metric in HEADLINE_METRICS:
            row[metric] = float(metrics.get(metric, np.nan))
            row[f"{metric}_vs_baseline"] = relative_improvement(
                float(baseline.get(metric, np.nan)), float(metrics.get(metric, np.nan)), metric
            )
        rows.append(row)
    return rows
