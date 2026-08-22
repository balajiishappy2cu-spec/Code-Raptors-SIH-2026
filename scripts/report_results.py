"""Emit the README's results tables straight from the recorded experiment output.

Every results table in the README has been hand-edited after each rerun, and the numbers
have moved often enough that transcription is a real risk. This renders them from
``results/experiment_results.json`` and ``results/experiment_rows.csv`` instead, so what
the README claims and what the experiment recorded cannot drift apart.

It prints markdown to stdout rather than rewriting the README, because the prose around
each table needs a human decision about what the numbers mean -- only the figures are
mechanical.

Usage::

    python scripts/report_results.py --config config.yaml
    python scripts/report_results.py --config config.yaml --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import Config  # noqa: E402
from common.io_utils import read_json  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402
from evaluation.metrics import LOWER_IS_BETTER  # noqa: E402

LOGGER = get_logger("scripts.report_results")

#: Metrics quoted in the headline table, in the order the README shows them.
HEADLINE = (
    "probability_of_detection",
    "probability_of_false_alarm",
    "sensitivity",
    "average_intercept_rate",
    "average_reward",
    "percentage_of_correct_predictions",
    "average_intercept_time_error",
    "average_time_to_intercept_censored",
    "scan_efficiency",
    "coverage",
    "active_band_coverage",
)

#: Display names, so the table reads as prose rather than as field names.
LABELS = {
    "probability_of_detection": "Probability of Detection",
    "probability_of_false_alarm": "Probability of False Alarm",
    "sensitivity": "Sensitivity",
    "average_intercept_rate": "Average Intercept Rate",
    "average_reward": "Average Reward",
    "percentage_of_correct_predictions": "Percentage of Correct Predictions",
    "average_intercept_time_error": "Average Intercept Time Error",
    "average_time_to_intercept_censored": "Average Time To Intercept (censored)",
    "scan_efficiency": "Scan Efficiency",
    "coverage": "Coverage",
    "active_band_coverage": "Active-band Coverage",
}


def _fmt(value: float) -> str:
    """Format a metric for the README, matching the precision already used there."""
    if not np.isfinite(value):
        return "n/a"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.4f}"


def headline_table(overall: dict[str, Any], comparison: list[dict[str, Any]]) -> str:
    """Render the sequential-vs-smart headline table."""
    improvements = {row["metric"]: row["improvement"] for row in comparison}
    lines = [
        "| Figure of merit | Sequential | Smart Scan | Improvement |",
        "|---|---|---|---|",
    ]
    for metric in HEADLINE:
        base = float(overall["sequential"].get(metric, float("nan")))
        cand = float(overall["smart"].get(metric, float("nan")))
        change = improvements.get(metric, float("nan"))
        if metric == "percentage_of_correct_predictions":
            lines.append(
                f"| {LABELS[metric]} | n/a (open loop) | {cand:.2f}% | — |"
            )
            continue
        better = np.isfinite(change) and change > 0
        base_cell = f"**{_fmt(base)}**" if not better else _fmt(base)
        cand_cell = f"**{_fmt(cand)}**" if better else _fmt(cand)
        change_cell = "n/a" if not np.isfinite(change) else f"{change * 100:+.1f}%"
        lines.append(f"| {LABELS[metric]} | {base_cell} | {cand_cell} | {change_cell} |")
    return "\n".join(lines)


def ablation_table(overall: dict[str, Any]) -> str:
    """Render the ablation table over every strategy."""
    metrics = [
        ("average_intercept_rate", "Intercept rate"),
        ("average_reward", "Avg reward"),
        ("average_intercept_time_error", "Intercept time error"),
        ("average_time_to_intercept_censored", "TTI (censored)"),
        ("active_band_coverage", "Active-band coverage"),
    ]
    order = ["random", "sequential", "smart_heuristic", "smart_ml_only", "smart"]
    lines = [
        "| Strategy | " + " | ".join(label for _, label in metrics) + " |",
        "|---" * (len(metrics) + 1) + "|",
    ]
    for key in order:
        if key not in overall:
            continue
        cells = [_fmt(float(overall[key].get(metric, float("nan")))) for metric, _ in metrics]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def power_table(rows_csv: Path, candidate: str = "smart", baseline: str = "sequential") -> str:
    """Render the per-environment confidence-interval table."""
    frame = pd.read_csv(rows_csv)
    metrics = [
        ("average_intercept_rate", "Average Intercept Rate"),
        ("average_reward", "Average Reward"),
        ("average_intercept_time_error", "Average Intercept Time Error"),
        ("average_time_to_intercept_censored", "Time To Intercept (censored)"),
        ("active_band_coverage", "Active-band Coverage"),
    ]
    lines = [
        "| Metric | Mean per-env improvement | 95% CI | Solid? |",
        "|---|---|---|---|",
    ]
    for metric, label in metrics:
        base = frame[frame.strategy == baseline][metric].to_numpy(dtype=np.float64)
        cand = frame[frame.strategy == candidate][metric].to_numpy(dtype=np.float64)
        if base.size != cand.size or base.size < 2:
            continue
        lower_better = metric in LOWER_IS_BETTER
        with np.errstate(divide="ignore", invalid="ignore"):
            denominator = np.where(base == 0, np.nan, np.abs(base))
            change = (base - cand) / denominator if lower_better else (cand - base) / denominator
        change = change[np.isfinite(change)]
        mean = float(change.mean())
        stderr = float(change.std(ddof=1) / np.sqrt(change.size))
        low, high = (mean - 1.96 * stderr) * 100, (mean + 1.96 * stderr) * 100
        solid = "**yes**" if low > 0 else "no — spans zero"
        lines.append(
            f"| {label} | {mean * 100:+.1f}% | [{low:+.1f}%, {high:+.1f}%] | {solid} |"
        )
    return "\n".join(lines)


def scorecard_block(scorecards: dict[str, Any]) -> str:
    """Render the scorecard grade table."""
    scheduler = scorecards.get("scheduler")
    model = scorecards.get("activity_model")
    if not scheduler:
        return "_no scorecard recorded_"
    lines = ["| | Grade | Score | |", "|---|---|---|---|"]
    lines.append(
        f"| **Smart Scan scheduler** | **{scheduler['grade']}** | "
        f"**{scheduler['overall']:.1f} / 100** | vs the sequential sweep, "
        f"{'no regressions flagged' if not scheduler['regressions'] else 'regressions: ' + ', '.join(scheduler['regressions'])} |"
    )
    for component in scheduler["components"]:
        lines.append(
            f"| {component['label']} | {component['grade']} | {component['score']:.1f} | "
            f"{component['detail']} |"
        )
    if model:
        lines.append(
            f"| **Activity model** | **{model['grade']}** | **{model['overall']:.1f} / 100** | "
            + ", ".join(f"{c['label'].lower()} {c['score']:.1f}" for c in model["components"])
            + " |"
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify every rendered number already appears in the README",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Render the results tables, or check the README against them."""
    args = parse_args(argv)
    config = Config.load(args.config)
    results_dir = config.path_for("paths.results_dir")

    record = read_json(results_dir / "experiment_results.json")
    overall = record["overall"]
    comparison = record.get("comparison", {}).get("overall", [])

    blocks = {
        "Headline": headline_table(overall, comparison),
        "Statistical power": power_table(results_dir / "experiment_rows.csv"),
        "Ablation": ablation_table(overall),
        "Scorecard": scorecard_block(record.get("scorecards", {})),
    }

    environments = record.get("environments_per_scenario", {})
    print(f"source: {record.get('data_source')} | timesteps: {record.get('timesteps')}")
    print(f"environments: {environments} (total {sum(environments.values())})\n")
    for title, block in blocks.items():
        print(f"### {title}\n\n{block}\n")

    if args.check:
        readme = (config.path_for("paths.results_dir").parent / "README.md").read_text(
            encoding="utf-8"
        )
        # The README uses a typographic minus (U+2212); normalise so the check compares
        # numbers rather than punctuation.
        readme = readme.replace("−", "-")
        missing = []
        for metric in HEADLINE:
            for strategy in ("sequential", "smart"):
                value = float(overall[strategy].get(metric, float("nan")))
                if np.isfinite(value) and _fmt(value) not in readme:
                    missing.append(f"{strategy}.{metric}={_fmt(value)}")
        if missing:
            LOGGER.warning("README is missing %d recorded values:", len(missing))
            for item in missing:
                LOGGER.warning("  %s", item)
            return 1
        LOGGER.info("README contains every recorded headline value")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
