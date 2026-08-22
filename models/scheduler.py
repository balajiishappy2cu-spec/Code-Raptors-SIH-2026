"""Receiver schedulers: the open-loop baseline and the Smart Scan strategy.

Prediction and decision are kept strictly separate. :mod:`models.activity_predictor`
answers "how likely is this band to transmit next?"; the schedulers here answer "which
band should the receiver tune to now?", combining that probability with exploration,
recency, periodicity and retune cost.

Adaptation from hit/miss feedback is handled by per-band Thompson Sampling, which
updates after every observation without retraining the gradient-boosted model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from common.logging_utils import get_logger
from features.band_features import FEATURE_NAMES, BandFeatureTracker
from models.activity_predictor import ActivityPredictorProtocol
from simulation.receiver import Observation

LOGGER = get_logger(__name__)


class Scheduler(ABC):
    """Base class for receiver schedulers.

    A scheduler sees only the feature tracker (built from the receiver's own
    observations) and its own internal state. It never reads the ground truth grid.
    """

    name: str = "scheduler"

    def __init__(self, n_bands: int, rng: np.random.Generator) -> None:
        """Initialise a scheduler for an environment with ``n_bands`` bands."""
        self.n_bands = int(n_bands)
        self.rng = rng
        self.current_band = -1
        self._last_decision: dict[str, Any] = {}

    @abstractmethod
    def select_band(self, timestep: int, tracker: BandFeatureTracker) -> int:
        """Choose the band to tune to at ``timestep``."""

    def update(self, observation: Observation) -> None:
        """Consume the outcome of an observation (hit/miss feedback)."""
        self.current_band = observation.selected_band

    def reset(self) -> None:
        """Clear per-run state so the scheduler can be reused on another environment."""
        self.current_band = -1
        self._last_decision = {}

    @property
    def last_decision(self) -> dict[str, Any]:
        """Score components behind the most recent decision (explainability panel)."""
        return dict(self._last_decision)

    def describe(self) -> dict[str, Any]:
        """Return a JSON-friendly description of the scheduler configuration."""
        return {"name": self.name, "n_bands": self.n_bands}


class SequentialSweepScheduler(Scheduler):
    """Open-loop baseline: sweep bands ``0, 1, ..., N-1, 0, ...`` in order.

    This is the pre-mission, prior-data strategy the problem statement criticises for
    spending equal time on every band regardless of what is found there.
    """

    name = "sequential"

    def __init__(self, n_bands: int, rng: np.random.Generator, *, step: int = 1) -> None:
        """Create a sequential sweep scheduler with a configurable band step."""
        super().__init__(n_bands=n_bands, rng=rng)
        self.step = max(1, int(step))
        self._next_band = 0

    def select_band(self, timestep: int, tracker: BandFeatureTracker) -> int:
        """Return the next band in the fixed sweep order."""
        band = self._next_band % self.n_bands
        self._next_band = (band + self.step) % self.n_bands
        self._last_decision = {"timestep": timestep, "band": band, "reason": "fixed sweep order"}
        return band

    def reset(self) -> None:
        """Restart the sweep at band 0."""
        super().reset()
        self._next_band = 0

    def describe(self) -> dict[str, Any]:
        """Return the scheduler configuration."""
        return {**super().describe(), "step": self.step}


class RandomScheduler(Scheduler):
    """Uniformly random band selection.

    Used to diversify the exploration policies that collect activity-model training
    rows, and as a control arm in the ablation.
    """

    name = "random"

    def select_band(self, timestep: int, tracker: BandFeatureTracker) -> int:
        """Return a uniformly random band."""
        band = int(self.rng.integers(0, self.n_bands))
        self._last_decision = {"timestep": timestep, "band": band, "reason": "uniform random"}
        return band


@dataclass
class ThompsonSampler:
    """Discounted Beta-Bernoulli Thompson Sampling over bands.

    Each band keeps ``Beta(alpha, beta)`` posterior over "this band yields a hit".
    A hit increments ``alpha``, a miss increments ``beta``; both are decayed on every
    update so the scheduler keeps adapting to emitters that change behaviour instead of
    freezing on early evidence.

    Attributes:
        n_bands: number of bands tracked.
        prior_alpha: prior successes.
        prior_beta: prior failures.
        decay: multiplicative discount applied to the posterior on every update.
    """

    n_bands: int
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    decay: float = 1.0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    alpha: np.ndarray = field(init=False)
    beta: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Initialise the posteriors to the configured prior."""
        self.reset()

    def reset(self) -> None:
        """Reset every band's posterior to the prior."""
        self.alpha = np.full(self.n_bands, float(self.prior_alpha), dtype=np.float64)
        self.beta = np.full(self.n_bands, float(self.prior_beta), dtype=np.float64)

    def sample(self) -> np.ndarray:
        """Draw one posterior sample of the hit probability for every band."""
        return self.rng.beta(self.alpha, self.beta)

    def means(self) -> np.ndarray:
        """Return the posterior mean hit probability for every band."""
        return self.alpha / (self.alpha + self.beta)

    def update(self, band: int, hit: bool) -> None:
        """Apply hit/miss feedback for one band, with discounting."""
        if self.decay < 1.0:
            self.alpha[band] = self.prior_alpha + (self.alpha[band] - self.prior_alpha) * self.decay
            self.beta[band] = self.prior_beta + (self.beta[band] - self.prior_beta) * self.decay
        if hit:
            self.alpha[band] += 1.0
        else:
            self.beta[band] += 1.0

    def state(self) -> dict[str, Any]:
        """Return the posterior state for logging and the dashboard."""
        return {
            "alpha": self.alpha.tolist(),
            "beta": self.beta.tolist(),
            "posterior_mean": self.means().tolist(),
        }


class SmartScanScheduler(Scheduler):
    """Closed-loop Smart Scan strategy.

    At each decision the scheduler scores every band with::

        score = w1 * predicted_probability
              + w2 * exploration_bonus
              + w3 * recency_bonus
              + w4 * periodicity_bonus
              - w5 * scan_cost

    where ``predicted_probability`` blends the activity model's calibrated probability
    with a Thompson Sampling draw. All weights come from ``config.yaml`` and are
    experimental values chosen by hand for this MVP, not optimised constants.
    """

    name = "smart"

    def __init__(
        self,
        n_bands: int,
        rng: np.random.Generator,
        *,
        predictor: ActivityPredictorProtocol,
        scheduler_cfg: dict[str, Any] | None = None,
    ) -> None:
        """Create a Smart Scan scheduler.

        Args:
            n_bands: number of bands in the environment.
            rng: seeded generator for Thompson draws and tie-breaking.
            predictor: activity model providing per-band probabilities.
            scheduler_cfg: the ``scheduler`` section of the configuration.
        """
        super().__init__(n_bands=n_bands, rng=rng)
        cfg = scheduler_cfg or {}
        weights = dict(cfg.get("weights", {}))
        self.w1 = float(weights.get("w1_predicted_probability", 1.0))
        self.w2 = float(weights.get("w2_exploration_bonus", 0.35))
        self.w3 = float(weights.get("w3_recency_bonus", 0.2))
        self.w4 = float(weights.get("w4_periodicity_bonus", 0.6))
        self.w5 = float(weights.get("w5_scan_cost", 0.1))

        thompson_cfg = dict(cfg.get("thompson", {}))
        self.thompson_enabled = bool(thompson_cfg.get("enabled", True))
        self.thompson_blend = float(thompson_cfg.get("blend", 0.4))
        self.thompson = ThompsonSampler(
            n_bands=n_bands,
            prior_alpha=float(thompson_cfg.get("prior_alpha", 1.0)),
            prior_beta=float(thompson_cfg.get("prior_beta", 1.0)),
            decay=float(thompson_cfg.get("decay", 1.0)),
            rng=rng,
        )

        exploration_cfg = dict(cfg.get("exploration", {}))
        self.staleness_saturation = max(
            1.0, float(exploration_cfg.get("staleness_saturation", 100))
        )
        # Hard revisit guarantee. A weighted bonus can always be outvoted by a band that
        # looks productive, so exploration alone cannot bound how long a band goes
        # unwatched -- which is exactly why an open-loop sweep beats this scheduler on
        # discovery. Real ES receivers solve it the same way: a mandatory revisit
        # interval per band, with priority scheduling in between. 0 disables it.
        self.max_revisit_interval = float(exploration_cfg.get("max_revisit_interval", 0) or 0)
        self.predictor = predictor
        self._feature_index = {name: i for i, name in enumerate(FEATURE_NAMES)}
        self._last_scores: np.ndarray = np.zeros(n_bands, dtype=np.float64)
        self._last_probabilities: np.ndarray = np.zeros(n_bands, dtype=np.float64)

    def _exploration_bonus(self, timestep: int, tracker: BandFeatureTracker) -> np.ndarray:
        """Bonus that grows with the time since a band was last observed."""
        staleness = np.empty(self.n_bands, dtype=np.float64)
        for band in range(self.n_bands):
            last_visit = tracker.histories[band].last_visit
            staleness[band] = (
                self.staleness_saturation if last_visit < 0 else float(timestep - last_visit)
            )
        return np.clip(staleness / self.staleness_saturation, 0.0, 1.0)

    def _recency_bonus(self, timestep: int, tracker: BandFeatureTracker) -> np.ndarray:
        """Bonus for bands that produced a detection recently."""
        bonus = np.zeros(self.n_bands, dtype=np.float64)
        scale = float(max(1, tracker.history_window))
        for band in range(self.n_bands):
            last_hit = tracker.histories[band].last_hit
            if last_hit >= 0:
                bonus[band] = float(np.exp(-(timestep - last_hit) / scale))
        return bonus

    def _scan_cost(self) -> np.ndarray:
        """Normalised cost of retuning from the current band to each candidate band."""
        if self.current_band < 0 or self.n_bands < 2:
            return np.zeros(self.n_bands, dtype=np.float64)
        distance = np.abs(np.arange(self.n_bands) - self.current_band)
        return distance / float(self.n_bands - 1)

    def select_band(self, timestep: int, tracker: BandFeatureTracker) -> int:
        """Score every band and return the highest-scoring one."""
        features = tracker.snapshot(timestep)
        model_probability = np.asarray(self.predictor.predict_proba(features), dtype=np.float64)

        if self.thompson_enabled:
            thompson_sample = self.thompson.sample()
            probability = (
                self.thompson_blend * thompson_sample
                + (1.0 - self.thompson_blend) * model_probability
            )
        else:
            thompson_sample = np.zeros(self.n_bands, dtype=np.float64)
            probability = model_probability

        exploration = self._exploration_bonus(timestep, tracker)
        recency = self._recency_bonus(timestep, tracker)
        periodicity = features[:, self._feature_index["phase_score"]]
        scan_cost = self._scan_cost()

        scores = (
            self.w1 * probability
            + self.w2 * exploration
            + self.w3 * recency
            + self.w4 * periodicity
            - self.w5 * scan_cost
        )
        # Any band overdue for a revisit pre-empts the score: the choice is restricted to
        # overdue bands, and the best-scoring one among them wins. This bounds the time any
        # band can go unobserved while still letting the model allocate every other dwell.
        overdue = np.zeros(self.n_bands, dtype=bool)
        if self.max_revisit_interval > 0:
            for candidate_band in range(self.n_bands):
                last_visit = tracker.histories[candidate_band].last_visit
                staleness = (
                    float("inf") if last_visit < 0 else float(timestep - last_visit)
                )
                overdue[candidate_band] = staleness >= self.max_revisit_interval

        effective = np.where(overdue, scores, -np.inf) if overdue.any() else scores
        best = float(effective.max())
        candidates = np.flatnonzero(effective >= best - 1e-12)
        band = int(self.rng.choice(candidates))

        self._last_scores = scores
        self._last_probabilities = probability
        self._last_decision = {
            "timestep": timestep,
            "band": band,
            "score": float(scores[band]),
            "predicted_probability": float(model_probability[band]),
            "thompson_sample": float(thompson_sample[band]),
            "combined_probability": float(probability[band]),
            "exploration_bonus": float(exploration[band]),
            "recency_bonus": float(recency[band]),
            "periodicity_bonus": float(periodicity[band]),
            "scan_cost": float(scan_cost[band]),
            "weights": {
                "w1_predicted_probability": self.w1,
                "w2_exploration_bonus": self.w2,
                "w3_recency_bonus": self.w3,
                "w4_periodicity_bonus": self.w4,
                "w5_scan_cost": self.w5,
            },
            "forced_revisit": bool(overdue[band]),
            "n_overdue_bands": int(overdue.sum()),
            "runner_up_band": int(np.argsort(scores)[-2]) if self.n_bands > 1 else band,
            "all_scores": scores.tolist(),
            "all_probabilities": probability.tolist(),
        }
        return band

    def update(self, observation: Observation) -> None:
        """Apply hit/miss feedback to the Thompson posteriors."""
        super().update(observation)
        if observation.settling:
            return
        for band in observation.bands or (observation.selected_band,):
            self.thompson.update(band=band, hit=bool(observation.detected))

    def reset(self) -> None:
        """Reset decision state and Thompson posteriors."""
        super().reset()
        self.thompson.reset()
        self._last_scores = np.zeros(self.n_bands, dtype=np.float64)
        self._last_probabilities = np.zeros(self.n_bands, dtype=np.float64)

    def predicted_probabilities(self) -> np.ndarray:
        """Return the probabilities used in the most recent decision."""
        return self._last_probabilities.copy()

    def describe(self) -> dict[str, Any]:
        """Return the scheduler configuration for the run record."""
        return {
            **super().describe(),
            "predictor": getattr(self.predictor, "name", type(self.predictor).__name__),
            "weights": {
                "w1_predicted_probability": self.w1,
                "w2_exploration_bonus": self.w2,
                "w3_recency_bonus": self.w3,
                "w4_periodicity_bonus": self.w4,
                "w5_scan_cost": self.w5,
            },
            "thompson": {
                "enabled": self.thompson_enabled,
                "blend": self.thompson_blend,
                "decay": self.thompson.decay,
                "prior_alpha": self.thompson.prior_alpha,
                "prior_beta": self.thompson.prior_beta,
            },
            "staleness_saturation": self.staleness_saturation,
            "max_revisit_interval": self.max_revisit_interval,
        }


def build_scheduler(
    kind: str,
    n_bands: int,
    rng: np.random.Generator,
    *,
    predictor: ActivityPredictorProtocol | None = None,
    scheduler_cfg: dict[str, Any] | None = None,
) -> Scheduler:
    """Construct a scheduler by name.

    Args:
        kind: ``sequential``, ``random`` or ``smart``.
        n_bands: number of bands in the environment.
        rng: seeded generator.
        predictor: activity model, required for ``smart``.
        scheduler_cfg: the ``scheduler`` configuration section.

    Returns:
        The constructed scheduler.

    Raises:
        ValueError: for an unknown scheduler name, or ``smart`` without a predictor.
    """
    key = kind.strip().lower()
    if key == "sequential":
        return SequentialSweepScheduler(n_bands=n_bands, rng=rng)
    if key == "random":
        return RandomScheduler(n_bands=n_bands, rng=rng)
    if key == "smart":
        if predictor is None:
            msg = "The smart scheduler requires an activity predictor"
            raise ValueError(msg)
        return SmartScanScheduler(
            n_bands=n_bands, rng=rng, predictor=predictor, scheduler_cfg=scheduler_cfg
        )
    msg = f"Unknown scheduler kind: {kind!r}"
    raise ValueError(msg)
