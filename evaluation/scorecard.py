"""Turn the figures of merit into a small number of interpretable scores.

A table of a dozen metrics answers "what happened" but not "is this any good". This
module answers the second question, and it does so with formulas that are printed
alongside the numbers rather than hidden behind a weighting nobody can inspect.

Two different things get graded, because "how good is the model" has two readings:

* **The activity model** -- a classifier, graded on ranking and calibration quality
  (ROC-AUC, PR-AUC, Brier, calibration error). This is an absolute judgement.
* **The scheduler** -- a decision policy, graded *relative to the open-loop sequential
  sweep it is supposed to beat*, because the absolute figures of merit are bounded by
  the receiver's instantaneous bandwidth and are not interpretable on their own.

Every scheduler sub-score uses the same mapping, so one convention covers all of them::

    score = 50 * (1 + log2(ratio))      clipped to [0, 100]

which puts **50 = parity with the baseline**, 100 = twice as good, 0 = half as good.
A ratio is always oriented so that larger is better before the mapping is applied.

The scores compress; they do not add information. Any regression visible in the metric
table is still a regression, and :func:`scheduler_scorecard` reports the individual
component scores and a list of regressions precisely so a good headline cannot bury one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from simulation.environment import EnvironmentGrid

#: Weights of the three scheduler sub-scores in the overall figure.
COMPONENT_WEIGHTS: dict[str, float] = {
    "interception": 0.40,
    "discovery": 0.30,
    "prediction": 0.30,
}

#: Letter grades by score. Ordered from best to worst.
GRADE_BANDS: tuple[tuple[float, str, str], ...] = (
    (80.0, "A", "clearly better than the open-loop sweep"),
    (65.0, "B", "better than the open-loop sweep"),
    (55.0, "C", "modestly better than the open-loop sweep"),
    (45.0, "D", "roughly level with the open-loop sweep"),
    (0.0, "E", "worse than the open-loop sweep"),
)

#: Absolute grade bands for the activity model's ranking quality (ROC-AUC).
MODEL_GRADE_BANDS: tuple[tuple[float, str, str], ...] = (
    (0.95, "A", "excellent separation"),
    (0.90, "B", "strong separation"),
    (0.80, "C", "fair separation"),
    (0.70, "D", "weak separation"),
    (0.0, "E", "little better than chance"),
)


def ratio_score(ratio: float) -> float:
    """Map a larger-is-better ratio onto ``[0, 100]`` with 50 at parity.

    Args:
        ratio: candidate / baseline, oriented so that larger is better.

    Returns:
        ``50 * (1 + log2(ratio))``, clipped to ``[0, 100]``; ``nan`` if undefined.
    """
    if not np.isfinite(ratio) or ratio <= 0:
        return float("nan")
    return float(np.clip(50.0 * (1.0 + math.log2(ratio)), 0.0, 100.0))


def _grade(score: float, bands: tuple[tuple[float, str, str], ...]) -> tuple[str, str]:
    """Return the ``(letter, description)`` for a score."""
    if not np.isfinite(score):
        return "-", "not measured"
    for threshold, letter, description in bands:
        if score >= threshold:
            return letter, description
    return bands[-1][1], bands[-1][2]


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Ratio guarded against zero and non-finite inputs."""
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def oracle_ceiling(
    environment: EnvironmentGrid,
    *,
    detection_probability: float,
    instantaneous_bandwidth: int = 1,
    horizon: int | None = None,
) -> dict[str, float]:
    """Best interception any receiver with this bandwidth could achieve here.

    The absolute Probability of Detection looks alarmingly small (a few percent) until
    you notice that a receiver seeing one band out of 32 physically cannot intercept
    more than one band's worth of transmission per timestep. This computes that bound
    so the measured numbers can be read as a fraction of what was ever available.

    The bound is deliberately generous: it assumes the receiver is always tuned to an
    active band whenever one exists, with no dwell, retune or knowledge constraints. No
    real scheduler can reach it.

    Args:
        environment: ground truth grid.
        detection_probability: receiver's probability of declaring an active band.
        instantaneous_bandwidth: bands observable per timestep.
        horizon: run length; defaults to the whole environment.

    Returns:
        Ceiling values for intercept rate and probability of detection.
    """
    end = min(int(horizon or environment.n_timesteps), environment.n_timesteps)
    if end <= 0:
        return {"oracle_intercept_rate": float("nan"), "oracle_probability_of_detection": float("nan")}

    active = environment.active[:end]
    per_timestep = np.minimum(active.sum(axis=1), max(1, int(instantaneous_bandwidth)))
    oracle_intercepts = float(per_timestep.sum()) * float(detection_probability)
    opportunities = float(active.sum())
    return {
        "oracle_intercepts": oracle_intercepts,
        "oracle_intercept_rate": oracle_intercepts / end,
        "oracle_probability_of_detection": _safe_ratio(oracle_intercepts, opportunities),
        "emission_opportunities": opportunities,
        "timesteps_with_any_activity": float((active.sum(axis=1) > 0).sum()),
    }


@dataclass
class ScoreComponent:
    """One graded component of the scorecard.

    Attributes:
        key: machine name.
        label: short human-readable name.
        score: 0-100 score, 50 = parity with the baseline.
        detail: plain-English sentence explaining what the number means.
        formula: the ratio that produced the score, written out.
        baseline: the baseline value the ratio used.
        candidate: the candidate value the ratio used.
        regression: whether the candidate is worse than the baseline here.
    """

    key: str
    label: str
    score: float
    detail: str
    formula: str = ""
    baseline: float = float("nan")
    candidate: float = float("nan")
    regression: bool = False

    @property
    def grade(self) -> str:
        """Letter grade for this component."""
        return _grade(self.score, GRADE_BANDS)[0]


@dataclass
class Scorecard:
    """The graded summary of one strategy against the baseline.

    Attributes:
        overall: weighted 0-100 score.
        grade: letter grade.
        verdict: one-sentence plain-English summary.
        components: the individual sub-scores.
        regressions: labels of the components where the candidate is worse.
        context: supporting numbers (oracle ceiling, achieved fraction, caveats).
    """

    overall: float
    grade: str
    verdict: str
    components: list[ScoreComponent] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-friendly record of the scorecard."""
        return {
            "overall": self.overall,
            "grade": self.grade,
            "verdict": self.verdict,
            "weights": COMPONENT_WEIGHTS,
            "components": [
                {
                    "key": c.key,
                    "label": c.label,
                    "score": c.score,
                    "grade": c.grade,
                    "detail": c.detail,
                    "formula": c.formula,
                    "baseline": c.baseline,
                    "candidate": c.candidate,
                    "regression": c.regression,
                }
                for c in self.components
            ],
            "regressions": self.regressions,
            "context": self.context,
        }


def scheduler_scorecard(
    baseline: dict[str, float],
    candidate: dict[str, float],
    *,
    oracle: dict[str, float] | None = None,
    baseline_name: str = "sequential sweep",
    candidate_name: str = "Smart Scan",
) -> Scorecard:
    """Grade a scheduler against the open-loop baseline.

    Args:
        baseline: figures of merit for the baseline strategy.
        candidate: figures of merit for the strategy being graded.
        oracle: optional ceiling from :func:`oracle_ceiling`, used for context.
        baseline_name: display name of the baseline.
        candidate_name: display name of the candidate.

    Returns:
        The assembled :class:`Scorecard`.
    """
    components: list[ScoreComponent] = []

    # --- Interception: is it catching more transmissions? -------------------------
    rate_ratio = _safe_ratio(
        candidate.get("average_intercept_rate", float("nan")),
        baseline.get("average_intercept_rate", float("nan")),
    )
    components.append(
        ScoreComponent(
            key="interception",
            label="Interception",
            score=ratio_score(rate_ratio),
            detail=(
                f"{candidate_name} intercepts {rate_ratio:.2f}x as many transmissions per "
                f"timestep as the {baseline_name}."
                if np.isfinite(rate_ratio)
                else "Not measured."
            ),
            formula="intercept rate (candidate) / intercept rate (baseline)",
            baseline=baseline.get("average_intercept_rate", float("nan")),
            candidate=candidate.get("average_intercept_rate", float("nan")),
            regression=bool(np.isfinite(rate_ratio) and rate_ratio < 1.0),
        )
    )

    # --- Discovery: is it still finding emitters it has not seen? -----------------
    tti_ratio = _safe_ratio(
        baseline.get("average_time_to_intercept_censored", float("nan")),
        candidate.get("average_time_to_intercept_censored", float("nan")),
    )
    coverage_ratio = _safe_ratio(
        candidate.get("active_band_coverage", float("nan")),
        baseline.get("active_band_coverage", float("nan")),
    )
    tti_score = ratio_score(tti_ratio)
    coverage_score = ratio_score(coverage_ratio)
    discovery_score = float(np.nanmean([tti_score, coverage_score]))
    components.append(
        ScoreComponent(
            key="discovery",
            label="Discovery",
            score=discovery_score,
            detail=(
                f"Time to first intercept is {1 / tti_ratio:.2f}x the baseline's and "
                f"{candidate.get('active_band_coverage', float('nan')):.1%} of genuinely "
                "active bands were found at least once."
                if np.isfinite(tti_ratio)
                else "Not measured."
            ),
            formula=(
                "mean of: baseline censored time-to-intercept / candidate's, "
                "and candidate active-band coverage / baseline's"
            ),
            baseline=baseline.get("average_time_to_intercept_censored", float("nan")),
            candidate=candidate.get("average_time_to_intercept_censored", float("nan")),
            regression=bool(np.isfinite(tti_ratio) and tti_ratio < 1.0),
        )
    )

    # --- Prediction: does it know when a band will be busy? -----------------------
    error_ratio = _safe_ratio(
        baseline.get("average_intercept_time_error", float("nan")),
        candidate.get("average_intercept_time_error", float("nan")),
    )
    correct = candidate.get("percentage_of_correct_predictions", float("nan"))
    error_score = ratio_score(error_ratio)
    prediction_score = float(np.nanmean([error_score, correct]))
    components.append(
        ScoreComponent(
            key="prediction",
            label="Prediction",
            score=prediction_score,
            detail=(
                f"Predicts when a band will next be active with {1 / error_ratio:.2f}x the "
                f"baseline's error, and {correct:.1f}% of its pre-scan calls were right."
                if np.isfinite(error_ratio) and np.isfinite(correct)
                else "Not measured (the open-loop baseline makes no predictions)."
            ),
            formula=(
                "mean of: baseline intercept-time error / candidate's, "
                "and percentage of correct predictions"
            ),
            baseline=baseline.get("average_intercept_time_error", float("nan")),
            candidate=candidate.get("average_intercept_time_error", float("nan")),
            regression=bool(np.isfinite(error_ratio) and error_ratio < 1.0),
        )
    )

    scores = np.array([c.score for c in components], dtype=np.float64)
    weights = np.array([COMPONENT_WEIGHTS[c.key] for c in components], dtype=np.float64)
    usable = np.isfinite(scores)
    overall = (
        float(np.sum(scores[usable] * weights[usable]) / np.sum(weights[usable]))
        if usable.any()
        else float("nan")
    )
    letter, description = _grade(overall, GRADE_BANDS)
    regressions = [c.label for c in components if c.regression]

    context: dict[str, Any] = {
        "baseline_name": baseline_name,
        "candidate_name": candidate_name,
        "scale": "50 = parity with the baseline; 100 = twice as good; 0 = half as good",
    }
    if oracle:
        ceiling = oracle.get("oracle_intercept_rate", float("nan"))
        achieved = _safe_ratio(candidate.get("average_intercept_rate", float("nan")), ceiling)
        baseline_achieved = _safe_ratio(
            baseline.get("average_intercept_rate", float("nan")), ceiling
        )
        context.update(
            {
                "oracle_intercept_rate": ceiling,
                "fraction_of_ceiling": achieved,
                "baseline_fraction_of_ceiling": baseline_achieved,
                "ceiling_note": (
                    "Ceiling assumes a receiver always tuned to an active band, with no "
                    "dwell, retune or knowledge constraints. It is not reachable."
                ),
            }
        )

    verdict = f"{candidate_name} scores {overall:.0f}/100 versus the {baseline_name} - {description}."
    if regressions:
        verdict += f" It is worse on: {', '.join(regressions)}."

    return Scorecard(
        overall=overall,
        grade=letter,
        verdict=verdict,
        components=components,
        regressions=regressions,
        context=context,
    )


def activity_model_scorecard(metrics: dict[str, float]) -> Scorecard:
    """Grade the activity classifier on its own terms.

    Ranking quality and calibration are graded separately because the scheduler needs
    both: a model that ranks bands correctly but reports badly scaled probabilities
    corrupts the weighted score it feeds into.

    Args:
        metrics: a classification report from :mod:`models.activity_predictor`.

    Returns:
        The assembled :class:`Scorecard`.
    """
    roc = float(metrics.get("roc_auc", float("nan")))
    pr = float(metrics.get("pr_auc", float("nan")))
    brier = float(metrics.get("brier", float("nan")))
    ece = float(metrics.get("expected_calibration_error", float("nan")))
    positive_rate = float(metrics.get("positive_rate", float("nan")))

    # Ranking: rescale ROC-AUC so 0.5 (chance) maps to 0 and 1.0 maps to 100.
    ranking_score = float(np.clip((roc - 0.5) * 200.0, 0.0, 100.0)) if np.isfinite(roc) else float("nan")
    # Calibration: an expected calibration error of 0 is perfect, 0.10 is poor.
    calibration_score = (
        float(np.clip(100.0 * (1.0 - ece / 0.10), 0.0, 100.0)) if np.isfinite(ece) else float("nan")
    )
    # Skill over always predicting the base rate, via the Brier skill score.
    reference_brier = positive_rate * (1.0 - positive_rate) if np.isfinite(positive_rate) else float("nan")
    skill = (
        float(np.clip(100.0 * (1.0 - brier / reference_brier), 0.0, 100.0))
        if np.isfinite(brier) and np.isfinite(reference_brier) and reference_brier > 0
        else float("nan")
    )

    components = [
        ScoreComponent(
            key="ranking",
            label="Ranking quality",
            score=ranking_score,
            detail=(
                f"ROC-AUC {roc:.3f}, PR-AUC {pr:.3f}. Given one band that will transmit and "
                f"one that will not, the model ranks them correctly {roc:.1%} of the time."
            ),
            formula="(ROC-AUC - 0.5) x 200",
            candidate=roc,
        ),
        ScoreComponent(
            key="calibration",
            label="Calibration",
            score=calibration_score,
            detail=(
                f"Expected calibration error {ece:.4f}. When the model says 70%, the band "
                "genuinely transmits about that often - which matters because the "
                "scheduler multiplies this probability by a weight."
            ),
            formula="100 x (1 - expected calibration error / 0.10)",
            candidate=ece,
        ),
        ScoreComponent(
            key="skill",
            label="Skill over base rate",
            score=skill,
            detail=(
                f"Brier score {brier:.4f} against {reference_brier:.4f} for always "
                f"predicting the {positive_rate:.1%} base rate."
            ),
            formula="Brier skill score against the base-rate forecast",
            baseline=reference_brier,
            candidate=brier,
        ),
    ]

    scores = np.array([c.score for c in components], dtype=np.float64)
    usable = np.isfinite(scores)
    overall = float(scores[usable].mean()) if usable.any() else float("nan")
    letter, description = _grade_model(roc)

    return Scorecard(
        overall=overall,
        grade=letter,
        verdict=(
            f"The activity model scores {overall:.0f}/100 - {description}. "
            f"It predicts whether a band will transmit in the next window from the "
            f"receiver's own observation history alone."
        ),
        components=components,
        regressions=[],
        context={
            "scale": "0 = no better than chance or badly calibrated; 100 = perfect",
            "positive_rate": positive_rate,
            "n_rows": metrics.get("n_rows", float("nan")),
        },
    )


def _grade_model(roc_auc: float) -> tuple[str, str]:
    """Grade the classifier by ROC-AUC, which is the interpretable anchor.

    The letter follows ROC-AUC rather than the blended score so that the grade keeps a
    fixed, checkable meaning: a model cannot earn an A on calibration alone.
    """
    if not np.isfinite(roc_auc):
        return "-", "not measured"
    for threshold, letter, description in MODEL_GRADE_BANDS:
        if roc_auc >= threshold:
            return letter, description
    return MODEL_GRADE_BANDS[-1][1], MODEL_GRADE_BANDS[-1][2]
