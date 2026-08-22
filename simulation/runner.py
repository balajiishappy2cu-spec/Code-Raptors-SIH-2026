"""Simulation loop: run one scheduler against one environment.

The loop is deliberately thin. It owns the boundary between what the receiver may see
(its own observations) and what only the evaluator may see (the ground truth grid, used
for metrics and rewards after the fact). Every strategy runs through this same loop, so
a comparison between strategies differs only in the scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from common.logging_utils import get_logger
from evaluation.metrics import InterceptCounters, RewardModel, compute_figures_of_merit
from features.band_features import BandFeatureTracker
from models.scheduler import Scheduler
from simulation.environment import EnvironmentGrid
from simulation.receiver import Observation, make_receiver

LOGGER = get_logger(__name__)

#: Values used in :attr:`SimulationRun.visit_grid`.
VISIT_NONE, VISIT_MISS, VISIT_HIT = 0, 1, 2


@dataclass
class SimulationRun:
    """Everything recorded during one scheduler/environment run.

    Attributes:
        strategy: scheduler name.
        environment_name: name of the environment the run used.
        selected_band: band observed at each timestep.
        detected: whether a detection was declared at each timestep.
        truth_active: whether the observed band was truly transmitting.
        signal_count: reported pulse count at each timestep.
        predicted_probability: pre-scan probability for the chosen band (``NaN`` for
            schedulers that make no prediction).
        event_reward: per-timestep discrete reward.
        continuous_reward: per-timestep continuous reward.
        visit_grid: ``(n_timesteps, n_bands)`` map of visits and hits, for the heatmap.
        decisions: one record per scheduling decision, for the timeline table.
        metrics: figures of merit computed at the end of the run.
        meta: run provenance (seed, configuration snapshots, receiver statistics).
    """

    strategy: str
    environment_name: str
    selected_band: np.ndarray
    detected: np.ndarray
    truth_active: np.ndarray
    signal_count: np.ndarray
    predicted_probability: np.ndarray
    event_reward: np.ndarray
    continuous_reward: np.ndarray
    visit_grid: np.ndarray
    decisions: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_timesteps(self) -> int:
        """Length of the run in timesteps."""
        return int(self.selected_band.size)

    @property
    def hits(self) -> np.ndarray:
        """Boolean array of true intercepts (detection on a genuinely active band)."""
        return self.detected & self.truth_active

    def to_record(self, *, include_series: bool = False) -> dict[str, Any]:
        """Return a JSON-friendly record of the run.

        Args:
            include_series: also include the full per-timestep series (large).
        """
        record: dict[str, Any] = {
            "strategy": self.strategy,
            "environment": self.environment_name,
            "metrics": self.metrics,
            "meta": self.meta,
        }
        if include_series:
            record["series"] = {
                "selected_band": self.selected_band.tolist(),
                "detected": self.detected.astype(bool).tolist(),
                "truth_active": self.truth_active.astype(bool).tolist(),
                "event_reward": self.event_reward.tolist(),
            }
        return record


def run_simulation(
    *,
    environment: EnvironmentGrid,
    scheduler: Scheduler,
    receiver_cfg: dict[str, Any],
    features_cfg: dict[str, Any],
    reward_model: RewardModel,
    rng: np.random.Generator,
    n_timesteps: int | None = None,
    prediction_window: int = 5,
    record_decisions: bool = True,
) -> SimulationRun:
    """Run one scheduler against one environment.

    Args:
        environment: ground truth grid to scan.
        scheduler: strategy under test (reset before the run).
        receiver_cfg: the ``receiver`` configuration section.
        features_cfg: the ``features`` configuration section.
        reward_model: reward weights used for evaluation only.
        rng: seeded generator for the receiver's detection draws.
        n_timesteps: run length; defaults to the environment length.
        prediction_window: window used when scoring pre-scan predictions.
        record_decisions: keep the per-decision explainability records.

    Returns:
        The completed :class:`SimulationRun`.

    Raises:
        ValueError: if the environment has no timesteps.
    """
    if environment.n_timesteps == 0:
        msg = f"Environment {environment.name!r} has no timesteps"
        raise ValueError(msg)

    horizon = min(int(n_timesteps or environment.n_timesteps), environment.n_timesteps)
    n_bands = environment.n_bands

    tracker = BandFeatureTracker(
        n_bands=n_bands, features_cfg=features_cfg, timestep_us=environment.timestep_us
    )
    receiver = make_receiver(receiver_cfg, rng)
    scheduler.reset()

    selected_band = np.zeros(horizon, dtype=np.int32)
    detected = np.zeros(horizon, dtype=bool)
    truth_active = np.zeros(horizon, dtype=bool)
    signal_count = np.zeros(horizon, dtype=np.int32)
    predicted_probability = np.full(horizon, np.nan, dtype=np.float64)
    event_reward = np.zeros(horizon, dtype=np.float64)
    continuous_reward = np.zeros(horizon, dtype=np.float64)
    visit_grid = np.zeros((horizon, n_bands), dtype=np.uint8)

    counters = InterceptCounters()
    counters.emission_opportunities = int(environment.active[:horizon].sum())

    first_active = environment.truncate(horizon).first_active_timestep()
    first_intercept = np.full(n_bands, -1, dtype=np.int64)
    last_fruitless_visit = np.full(n_bands, -(10**9), dtype=np.int64)
    intercepted_mask = np.zeros(n_bands, dtype=bool)

    decisions: list[dict[str, Any]] = []
    dwell_lengths: list[int] = []
    intercept_time_errors: list[float] = []
    pending_prediction: dict[str, Any] | None = None
    current_dwell = 0
    current_probability = float("nan")

    for timestep in range(horizon):
        if receiver.needs_decision:
            if current_dwell:
                dwell_lengths.append(current_dwell)
                current_dwell = 0

            band = scheduler.select_band(timestep, tracker)
            receiver.tune(band)
            decision = scheduler.last_decision

            # Pre-scan prediction, scored later against what the dwell actually found.
            probability = decision.get(
                "combined_probability", decision.get("predicted_probability")
            )
            current_probability = (
                float(probability) if probability is not None else float("nan")
            )
            if pending_prediction is not None:
                _score_prediction(pending_prediction, environment, counters, prediction_window)
            pending_prediction = (
                {"band": band, "timestep": timestep, "probability": float(probability)}
                if probability is not None and np.isfinite(float(probability))
                else None
            )

            # Intercept time error: PRI-based prediction of the next activity in this band
            # versus when the band is genuinely next active.
            predicted_next = tracker.predicted_next_hit(band)
            if predicted_next >= 0:
                actual_next = environment.next_active_timestep(timestep, band)
                if actual_next >= 0:
                    intercept_time_errors.append(abs(float(predicted_next) - float(actual_next)))

            if record_decisions:
                decisions.append(
                    {
                        "timestep": timestep,
                        "band": int(band),
                        "band_centre_mhz": float(environment.band_centres_mhz[band]),
                        "predicted_probability": float(
                            decision.get("predicted_probability", float("nan"))
                        ),
                        "combined_probability": float(
                            decision.get("combined_probability", float("nan"))
                        ),
                        "thompson_sample": float(decision.get("thompson_sample", float("nan"))),
                        "exploration_bonus": float(
                            decision.get("exploration_bonus", float("nan"))
                        ),
                        "recency_bonus": float(decision.get("recency_bonus", float("nan"))),
                        "periodicity_bonus": float(
                            decision.get("periodicity_bonus", float("nan"))
                        ),
                        "scan_cost": float(decision.get("scan_cost", float("nan"))),
                        "score": float(decision.get("score", float("nan"))),
                        "reason": decision.get("reason", "weighted score"),
                        "pri_estimate_us": tracker.pri_estimate_us(band),
                        "periodicity_score": tracker.periodicity_score(band),
                    }
                )

        observation: Observation = receiver.observe(environment, timestep)
        current_dwell += 1
        band = observation.selected_band
        bands = list(observation.bands or (band,))

        active_flags = environment.active[timestep, bands]
        cell_active = bool(active_flags.any())
        n_active_cells = int(active_flags.sum())
        n_idle_cells = len(bands) - n_active_cells

        selected_band[timestep] = band
        predicted_probability[timestep] = current_probability
        detected[timestep] = bool(observation.detected)
        truth_active[timestep] = cell_active
        signal_count[timestep] = int(observation.signal_count)
        if pending_prediction is not None and pending_prediction["band"] == band:
            pending_prediction["observed_active"] = (
                pending_prediction.get("observed_active", False) or cell_active
            )

        if observation.settling:
            counters.settling_timesteps += 1
        else:
            counters.visits += 1
            counters.observed_active_cells += n_active_cells
            counters.observed_idle_cells += n_idle_cells
            if observation.detected and n_active_cells:
                counters.intercepts += n_active_cells
            if observation.detected and n_active_cells == 0:
                counters.false_alarms += n_idle_cells

        is_hit = bool(observation.detected and cell_active)
        for covered in bands:
            visit_grid[timestep, covered] = VISIT_HIT if is_hit else VISIT_MISS

        first_detection = False
        if is_hit:
            for covered in bands:
                if environment.active[timestep, covered] and first_intercept[covered] < 0:
                    first_intercept[covered] = timestep
                    intercepted_mask[covered] = True
                    first_detection = True

        unnecessary_repeat = (not is_hit) and (
            timestep - last_fruitless_visit[band] <= reward_model.repeat_scan_window
        )
        if not is_hit:
            last_fruitless_visit[band] = timestep

        waiting = int((environment.active[timestep] & ~intercepted_mask).sum())
        event_reward[timestep] = reward_model.event_reward(
            detected=bool(observation.detected),
            truth_active=cell_active,
            first_detection=first_detection,
            unnecessary_repeat=bool(unnecessary_repeat),
        )
        continuous_reward[timestep] = reward_model.continuous_reward(
            detected=bool(observation.detected),
            truth_active=cell_active,
            waiting_active_bands=waiting,
        )

        tracker.update_from_observation(observation)
        scheduler.update(observation)

    if pending_prediction is not None:
        _score_prediction(pending_prediction, environment, counters, prediction_window)
    if current_dwell:
        dwell_lengths.append(current_dwell)

    intercepted = (first_intercept >= 0) & (first_active >= 0)
    time_to_intercept = (first_intercept[intercepted] - first_active[intercepted]).astype(
        np.float64
    )
    # Censored variant: an active band never intercepted is charged the remaining horizon,
    # so a strategy cannot look fast simply by ignoring the bands that are hard to find.
    censored: list[float] = []
    for band in range(n_bands):
        if first_active[band] < 0:
            continue
        if first_intercept[band] >= 0:
            censored.append(float(first_intercept[band] - first_active[band]))
        else:
            censored.append(float(horizon - first_active[band]))

    metrics = compute_figures_of_merit(
        counters=counters,
        n_timesteps=horizon,
        n_bands=n_bands,
        bands_visited=selected_band,
        dwell_lengths=dwell_lengths,
        event_rewards=event_reward,
        continuous_rewards=continuous_reward,
        time_to_intercept=time_to_intercept,
        time_to_intercept_censored=censored,
        intercept_time_errors=intercept_time_errors,
        bands_ever_active=int((first_active >= 0).sum()),
        bands_intercepted=int(intercepted.sum()),
        timestep_us=environment.timestep_us,
    )

    run = SimulationRun(
        strategy=scheduler.name,
        environment_name=environment.name,
        selected_band=selected_band,
        detected=detected,
        truth_active=truth_active,
        signal_count=signal_count,
        predicted_probability=predicted_probability,
        event_reward=event_reward,
        continuous_reward=continuous_reward,
        visit_grid=visit_grid,
        decisions=decisions,
        metrics=metrics,
        meta={
            "scheduler": scheduler.describe(),
            "receiver": receiver.stats(),
            "reward_weights": reward_model.describe(),
            "environment": environment.summary(),
            "prediction_window": prediction_window,
            "horizon": horizon,
        },
    )
    LOGGER.info(
        "%s on %s: Pd=%.4f intercept_rate=%.3f scan_efficiency=%.3f reward=%.3f",
        run.strategy,
        run.environment_name,
        metrics["probability_of_detection"],
        metrics["average_intercept_rate"],
        metrics["scan_efficiency"],
        metrics["average_reward"],
    )
    return run


def _score_prediction(
    pending: dict[str, Any],
    environment: EnvironmentGrid,
    counters: InterceptCounters,
    prediction_window: int,
) -> None:
    """Score one pre-scan prediction against ground truth.

    A prediction is counted correct when ``probability >= 0.5`` matches whether the band
    genuinely transmitted anywhere in the prediction window that followed the decision.
    """
    band = int(pending["band"])
    timestep = int(pending["timestep"])
    predicted_positive = float(pending["probability"]) >= 0.5
    actually_active = environment.active_in_window(timestep, band, prediction_window)
    counters.predictions += 1
    if predicted_positive == actually_active:
        counters.correct_predictions += 1
