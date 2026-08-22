"""Figures of merit for interception performance.

The problem statement names the figures of merit it wants:

    "...probability of detection, probability of false alarm, sensitivity, average
     intercept rate, average reward/cost function, percentage of correct predictions,
     and average intercept time error."

Those exact terms are used verbatim as metric keys here, in the plots and in the README.
Each one is defined explicitly below, because several of them admit more than one
reasonable reading and a silent choice would make the numbers uncomparable.

Everything in this module is computed offline, after a run, against the ground truth
grid. None of it is visible to the receiver or the scheduler during the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass
class RewardModel:
    """Reward and cost accounting for a scanning receiver.

    The weights are experimental parameters chosen for this MVP. They are not
    optimised or validated constants, and results should be read with that in mind.

    Attributes:
        hit: reward for observing a band that is transmitting and detecting it.
        miss: penalty for dwelling on a band with nothing to find.
        first_detection: extra reward the first time a band is intercepted in a run.
        repeat_scan_penalty: penalty for re-visiting a band that was just found empty.
        repeat_scan_window: how recently a fruitless visit counts as "just".
        false_alarm_penalty: penalty applied to detections on idle bands.
        scan_cost: fixed cost charged for every timestep of scanning.
        intercept_time_penalty_per_timestep: penalty proportional to how long an active
            band has been waiting to be intercepted.
    """

    hit: float = 1.0
    miss: float = -0.1
    first_detection: float = 2.0
    repeat_scan_penalty: float = -0.05
    repeat_scan_window: int = 10
    false_alarm_penalty: float = 0.5
    scan_cost: float = 0.01
    intercept_time_penalty_per_timestep: float = 0.001

    @classmethod
    def from_config(cls, reward_cfg: dict[str, Any]) -> "RewardModel":
        """Build a reward model from the ``reward`` configuration section."""
        return cls(
            hit=float(reward_cfg.get("hit", 1.0)),
            miss=float(reward_cfg.get("miss", -0.1)),
            first_detection=float(reward_cfg.get("first_detection", 2.0)),
            repeat_scan_penalty=float(reward_cfg.get("repeat_scan_penalty", -0.05)),
            repeat_scan_window=int(reward_cfg.get("repeat_scan_window", 10)),
            false_alarm_penalty=float(reward_cfg.get("false_alarm_penalty", 0.5)),
            scan_cost=float(reward_cfg.get("scan_cost", 0.01)),
            intercept_time_penalty_per_timestep=float(
                reward_cfg.get("intercept_time_penalty_per_timestep", 0.001)
            ),
        )

    def event_reward(
        self,
        *,
        detected: bool,
        truth_active: bool,
        first_detection: bool,
        unnecessary_repeat: bool,
    ) -> float:
        """Discrete event reward for one observed timestep (Section 15 of the brief)."""
        reward = 0.0
        if detected and truth_active:
            reward += self.hit
            if first_detection:
                reward += self.first_detection
        else:
            reward += self.miss
        if unnecessary_repeat:
            reward += self.repeat_scan_penalty
        return reward

    def continuous_reward(
        self,
        *,
        detected: bool,
        truth_active: bool,
        waiting_active_bands: int,
    ) -> float:
        """Normalised continuous reward for one observed timestep.

        ``reward = detection_reward - false_alarm_penalty - scan_cost
                   - intercept_time_penalty``
        """
        detection_reward = self.hit if (detected and truth_active) else 0.0
        false_alarm = self.false_alarm_penalty if (detected and not truth_active) else 0.0
        intercept_time = self.intercept_time_penalty_per_timestep * float(waiting_active_bands)
        return detection_reward - false_alarm - self.scan_cost - intercept_time

    def describe(self) -> dict[str, float]:
        """Return the reward weights for the run record."""
        return {
            "hit": self.hit,
            "miss": self.miss,
            "first_detection": self.first_detection,
            "repeat_scan_penalty": self.repeat_scan_penalty,
            "repeat_scan_window": float(self.repeat_scan_window),
            "false_alarm_penalty": self.false_alarm_penalty,
            "scan_cost": self.scan_cost,
            "intercept_time_penalty_per_timestep": self.intercept_time_penalty_per_timestep,
        }


@dataclass
class InterceptCounters:
    """Raw counts accumulated during a run, from which the figures of merit derive.

    Attributes:
        emission_opportunities: active (timestep, band) cells in the simulated horizon.
        observed_active_cells: active cells the receiver actually looked at.
        intercepts: observed active cells the receiver declared a detection on.
        observed_idle_cells: idle cells the receiver looked at.
        false_alarms: idle cells the receiver wrongly declared a detection on.
        visits: observation timesteps (excluding settling timesteps).
        settling_timesteps: timesteps lost to retuning.
        correct_predictions: pre-scan predictions that matched the observed truth.
        predictions: pre-scan predictions made.
    """

    emission_opportunities: int = 0
    observed_active_cells: int = 0
    intercepts: int = 0
    observed_idle_cells: int = 0
    false_alarms: int = 0
    visits: int = 0
    settling_timesteps: int = 0
    correct_predictions: int = 0
    predictions: int = 0


def _safe_divide(numerator: float, denominator: float, default: float = float("nan")) -> float:
    """Divide, returning ``default`` when the denominator is zero."""
    return float(numerator / denominator) if denominator else default


def compute_figures_of_merit(
    *,
    counters: InterceptCounters,
    n_timesteps: int,
    n_bands: int,
    bands_visited: Sequence[int] | np.ndarray,
    dwell_lengths: Sequence[int] | np.ndarray,
    event_rewards: Sequence[float] | np.ndarray,
    continuous_rewards: Sequence[float] | np.ndarray,
    time_to_intercept: Sequence[float] | np.ndarray,
    time_to_intercept_censored: Sequence[float] | np.ndarray,
    intercept_time_errors: Sequence[float] | np.ndarray,
    bands_ever_active: int,
    bands_intercepted: int,
    timestep_us: float = 1.0,
) -> dict[str, float]:
    """Compute the problem statement's figures of merit for one run.

    Definitions used here, stated explicitly:

    * **Probability of Detection (Pd)** -- intercepts divided by *all* emission
      opportunities in the horizon, i.e. the fraction of the environment's transmission
      cells that this receiver actually caught. With one instantaneous band out of
      ``n_bands``, an ideal receiver is bounded well below 1; that bound is identical
      for every strategy compared, which is what makes the comparison fair.
    * **Sensitivity** -- detections divided by active cells *the receiver looked at*.
      This isolates the detector from the scheduler: it measures whether energy present
      in the tuned band was declared, not whether the receiver was tuned to the right band.
    * **Probability of False Alarm (Pfa)** -- false detections divided by idle cells the
      receiver looked at. A band that was never observed can neither be detected nor
      false-alarmed, so unvisited cells are excluded from this denominator.
    * **Average Intercept Rate** -- intercepts per timestep.
    * **Percentage of Correct Predictions** -- share of pre-scan predictions
      (``predicted probability >= 0.5``) that matched what the tuned band was truly doing.
    * **Average Intercept Time Error** -- mean absolute error, in timesteps, between the
      scheduler's PRI-based prediction of when a band would next be active and when it
      was actually next active.
    * **Average Time To Intercept** -- mean delay, in timesteps, between a band first
      transmitting and the receiver first intercepting it. Reported twice: over the
      bands that were intercepted (comparable to the usual reading, but conditional,
      so a strategy that intercepts only easy bands can look good on it), and
      *censored*, where every active band that was never intercepted is charged the
      full remaining horizon. The censored figure is the one to compare strategies on.

    Args:
        counters: raw counts accumulated during the run.
        n_timesteps: length of the run.
        n_bands: number of bands in the environment.
        bands_visited: band index observed at each timestep.
        dwell_lengths: length of every completed dwell, in timesteps.
        event_rewards: per-timestep discrete rewards.
        continuous_rewards: per-timestep continuous rewards.
        time_to_intercept: per-band interception delays (bands that were intercepted).
        time_to_intercept_censored: as above, but including never-intercepted active
            bands charged the remaining horizon.
        intercept_time_errors: absolute PRI-prediction errors, in timesteps.
        bands_ever_active: number of bands that transmit at least once.
        bands_intercepted: number of distinct bands intercepted at least once.
        timestep_us: timestep duration, used to report time metrics in microseconds too.

    Returns:
        Dictionary of figures of merit plus the supporting counts.
    """
    visited = np.asarray(bands_visited, dtype=np.int64)
    unique_visited = int(np.unique(visited).size) if visited.size else 0
    dwells = np.asarray(dwell_lengths, dtype=np.float64)
    events = np.asarray(event_rewards, dtype=np.float64)
    continuous = np.asarray(continuous_rewards, dtype=np.float64)
    tti = np.asarray(time_to_intercept, dtype=np.float64)
    tti_censored = np.asarray(time_to_intercept_censored, dtype=np.float64)
    errors = np.asarray(intercept_time_errors, dtype=np.float64)

    probability_of_detection = _safe_divide(
        counters.intercepts, counters.emission_opportunities, 0.0
    )
    sensitivity = _safe_divide(counters.intercepts, counters.observed_active_cells, float("nan"))
    probability_of_false_alarm = _safe_divide(
        counters.false_alarms, counters.observed_idle_cells, float("nan")
    )

    return {
        # --- Problem statement figures of merit (exact terms) ---
        "probability_of_detection": probability_of_detection,
        "probability_of_false_alarm": probability_of_false_alarm,
        "sensitivity": sensitivity,
        "average_intercept_rate": _safe_divide(counters.intercepts, n_timesteps, 0.0),
        "average_reward": float(events.mean()) if events.size else 0.0,
        "average_cost": float(-continuous.mean()) if continuous.size else 0.0,
        "average_continuous_reward": float(continuous.mean()) if continuous.size else 0.0,
        "percentage_of_correct_predictions": 100.0
        * _safe_divide(counters.correct_predictions, counters.predictions, float("nan")),
        "average_intercept_time_error": float(errors.mean()) if errors.size else float("nan"),
        "average_intercept_time_error_us": float(errors.mean() * timestep_us)
        if errors.size
        else float("nan"),
        # --- Supporting operational metrics ---
        "average_time_to_intercept": float(tti.mean()) if tti.size else float("nan"),
        "average_time_to_intercept_censored": float(tti_censored.mean())
        if tti_censored.size
        else float("nan"),
        "average_time_to_intercept_us": float(tti.mean() * timestep_us)
        if tti.size
        else float("nan"),
        "scan_efficiency": _safe_divide(counters.intercepts, counters.visits, 0.0),
        "coverage": _safe_divide(unique_visited, n_bands, 0.0),
        "active_band_coverage": _safe_divide(bands_intercepted, bands_ever_active, float("nan")),
        "average_dwell_time": float(dwells.mean()) if dwells.size else float("nan"),
        "total_reward": float(events.sum()) if events.size else 0.0,
        # --- Raw counts ---
        "n_timesteps": float(n_timesteps),
        "n_bands": float(n_bands),
        "emission_opportunities": float(counters.emission_opportunities),
        "observed_active_cells": float(counters.observed_active_cells),
        "intercepts": float(counters.intercepts),
        "observed_idle_cells": float(counters.observed_idle_cells),
        "false_alarms": float(counters.false_alarms),
        "visits": float(counters.visits),
        "settling_timesteps": float(counters.settling_timesteps),
        "predictions": float(counters.predictions),
        "correct_predictions": float(counters.correct_predictions),
        "bands_ever_active": float(bands_ever_active),
        "bands_intercepted": float(bands_intercepted),
        "n_intercept_time_predictions": float(errors.size),
    }


#: Metrics where a lower value is better, used when reporting relative improvement.
LOWER_IS_BETTER: frozenset[str] = frozenset(
    {
        "probability_of_false_alarm",
        "average_intercept_time_error",
        "average_intercept_time_error_us",
        "average_time_to_intercept",
        "average_time_to_intercept_censored",
        "average_time_to_intercept_us",
        "average_cost",
    }
)

#: The metrics shown in the headline comparison table and bar chart.
HEADLINE_METRICS: tuple[str, ...] = (
    "probability_of_detection",
    "probability_of_false_alarm",
    "sensitivity",
    "average_intercept_rate",
    "average_reward",
    "percentage_of_correct_predictions",
    "average_intercept_time_error",
    "average_time_to_intercept",
    "average_time_to_intercept_censored",
    "scan_efficiency",
    "coverage",
    "active_band_coverage",
    "average_dwell_time",
)


def relative_improvement(baseline: float, candidate: float, metric: str) -> float:
    """Return the fractional improvement of ``candidate`` over ``baseline``.

    For metrics where lower is better the brief's formula ``(baseline - smart) /
    baseline`` is used directly; for the rest the sign is flipped so that a positive
    number always means "the candidate is better".

    Args:
        baseline: baseline metric value.
        candidate: candidate metric value.
        metric: metric name, used to decide the direction.

    Returns:
        Fractional improvement, or ``nan`` when it is undefined.
    """
    if not np.isfinite(baseline) or not np.isfinite(candidate) or baseline == 0:
        return float("nan")
    if metric in LOWER_IS_BETTER:
        return float((baseline - candidate) / abs(baseline))
    return float((candidate - baseline) / abs(baseline))


def compare_metric_tables(
    baseline: dict[str, float],
    candidate: dict[str, float],
    metrics: Sequence[str] = HEADLINE_METRICS,
) -> list[dict[str, Any]]:
    """Build a row-per-metric comparison of two runs.

    Args:
        baseline: figures of merit for the baseline strategy.
        candidate: figures of merit for the candidate strategy.
        metrics: metric names to include.

    Returns:
        A list of dictionaries suitable for a pandas frame or a JSON record.
    """
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        base_value = float(baseline.get(metric, float("nan")))
        cand_value = float(candidate.get(metric, float("nan")))
        rows.append(
            {
                "metric": metric,
                "baseline": base_value,
                "candidate": cand_value,
                "improvement": relative_improvement(base_value, cand_value, metric),
                "lower_is_better": metric in LOWER_IS_BETTER,
            }
        )
    return rows


@dataclass
class LearningCurve:
    """Cumulative reward over time, for the learning-curve plot.

    Attributes:
        timesteps: timestep index of each sample.
        cumulative_reward: cumulative discrete reward at that timestep.
        rolling_hit_rate: hit rate over a trailing window.
    """

    timesteps: list[int] = field(default_factory=list)
    cumulative_reward: list[float] = field(default_factory=list)
    rolling_hit_rate: list[float] = field(default_factory=list)


def build_learning_curve(
    event_rewards: Sequence[float],
    hits: Sequence[bool],
    window: int = 100,
) -> LearningCurve:
    """Build cumulative-reward and rolling-hit-rate series from a run.

    Args:
        event_rewards: per-timestep discrete rewards.
        hits: per-timestep intercept flags.
        window: trailing window for the rolling hit rate.

    Returns:
        The assembled :class:`LearningCurve`.
    """
    rewards = np.asarray(event_rewards, dtype=np.float64)
    hit_array = np.asarray(hits, dtype=np.float64)
    cumulative = np.cumsum(rewards)
    rolling = np.empty_like(hit_array)
    for index in range(hit_array.size):
        start = max(0, index - window + 1)
        rolling[index] = float(hit_array[start : index + 1].mean())
    return LearningCurve(
        timesteps=list(range(rewards.size)),
        cumulative_reward=cumulative.tolist(),
        rolling_hit_rate=rolling.tolist(),
    )


#: Plain-English descriptions of every metric, single-sourced so the dashboard, the
#: figures and the README cannot drift apart on what a number means.
METRIC_DESCRIPTIONS: dict[str, str] = {
    "probability_of_detection": (
        "Of every transmission that happened anywhere in the spectrum, the share this "
        "receiver actually caught. Looks small by nature: a receiver watching one band "
        "out of many cannot be everywhere at once."
    ),
    "probability_of_false_alarm": (
        "How often the receiver cried wolf - declared a detection on a band that was "
        "silent. Counted only over bands it actually looked at."
    ),
    "sensitivity": (
        "When the receiver was tuned to a band that really was transmitting, how often "
        "it noticed. This grades the detector, not the scheduler."
    ),
    "average_intercept_rate": (
        "Transmissions intercepted per timestep. The headline 'how much is it catching' "
        "number."
    ),
    "average_reward": (
        "Mean per-timestep score under the reward model: hits earn, misses and wasted "
        "re-scans cost, and finding a band for the first time earns a bonus."
    ),
    "average_cost": "The negative of the continuous reward, so lower is better.",
    "percentage_of_correct_predictions": (
        "Before each dwell the scheduler predicts whether the band will be busy. This is "
        "how often that call was right. The open-loop sweep makes no predictions, so it "
        "has no value here."
    ),
    "average_intercept_time_error": (
        "How far off, in timesteps, the scheduler's prediction of when a band would next "
        "go active turned out to be. Lower is better."
    ),
    "average_time_to_intercept": (
        "Average delay between a band starting to transmit and the receiver first "
        "catching it - but only counting bands it eventually caught, so it flatters a "
        "strategy that ignores hard bands. Lower is better."
    ),
    "average_time_to_intercept_censored": (
        "The same delay, but bands never caught at all are charged the rest of the run. "
        "This is the fair one to compare strategies on. Lower is better."
    ),
    "scan_efficiency": (
        "Share of dwells that found something. High means little time wasted on empty "
        "bands."
    ),
    "coverage": "Share of all frequency bands the receiver visited at least once.",
    "active_band_coverage": (
        "Share of the bands that genuinely had emitters on them which the receiver found "
        "at least once. The 'did it miss anyone' number."
    ),
    "average_dwell_time": "Average number of timesteps spent on a band before retuning.",
    "emission_opportunities": "Total band-timesteps in which some emitter was transmitting.",
    "intercepts": "Transmitting band-timesteps the receiver both looked at and detected.",
    "false_alarms": "Silent band-timesteps the receiver wrongly declared as active.",
    "visits": "Timesteps the receiver spent observing (excluding retune settling).",
}


def describe_metric(metric: str) -> str:
    """Return the plain-English description of a metric, or a fallback."""
    return METRIC_DESCRIPTIONS.get(metric, "No description available for this metric.")
