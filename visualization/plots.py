"""Figures for the Smart Scan experiments.

The centrepiece is the frequency-time heatmap: time on the x-axis, frequency band on the
y-axis, ground truth emitter activity as the background, and each receiver's actual scan
path drawn over it with hits and misses marked. Everything else -- the metric comparison,
the learning curves, the calibration curve, the decision timeline -- supports it.

All figures are written with the non-interactive Agg backend so the scripts run headless.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from common.logging_utils import get_logger  # noqa: E402
from evaluation.metrics import HEADLINE_METRICS, LOWER_IS_BETTER, build_learning_curve  # noqa: E402
from simulation.environment import EnvironmentGrid  # noqa: E402
from simulation.runner import SimulationRun  # noqa: E402

LOGGER = get_logger(__name__)

_TRUTH_CMAP = ListedColormap(["#f4f4f6", "#b9c6dd"])
_HIT_COLOUR = "#1a7f37"
_MISS_COLOUR = "#d1242f"


def _save(fig: plt.Figure, path: str | Path, *, dpi: int = 140) -> Path:
    """Save and close a figure, creating parent directories."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote figure %s", out)
    return out


def plot_frequency_time_heatmap(
    environment: EnvironmentGrid,
    runs: Sequence[SimulationRun],
    path: str | Path,
    *,
    time_window: tuple[int, int] | None = None,
    title: str = "",
) -> Path:
    """Plot ground truth activity with each strategy's scan path drawn over it.

    Args:
        environment: ground truth grid the runs were executed against.
        runs: runs to draw, one panel each.
        path: output file path.
        time_window: optional ``(start, end)`` timestep range to zoom into.
        title: figure title.

    Returns:
        The written figure path.
    """
    start, end = time_window or (0, min(environment.n_timesteps, 600))
    end = min(end, environment.n_timesteps)
    truth = environment.active[start:end].T  # (bands, time)

    n_panels = max(1, len(runs))
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(13, 3.1 * n_panels + 0.8), sharex=True, squeeze=False
    )
    axes_flat = axes[:, 0]

    for axis, run in zip(axes_flat, runs):
        axis.imshow(
            truth,
            aspect="auto",
            origin="lower",
            cmap=_TRUTH_CMAP,
            vmin=0,
            vmax=1,
            interpolation="nearest",
            extent=(start, end, -0.5, environment.n_bands - 0.5),
        )
        window = slice(start, min(end, run.n_timesteps))
        timesteps = np.arange(start, min(end, run.n_timesteps))
        bands = run.selected_band[window]
        hits = run.hits[window]

        axis.plot(timesteps, bands, color="#444444", linewidth=0.7, alpha=0.55, zorder=2)
        axis.scatter(
            timesteps[~hits], bands[~hits], s=7, c=_MISS_COLOUR, zorder=3, label="miss"
        )
        axis.scatter(timesteps[hits], bands[hits], s=11, c=_HIT_COLOUR, zorder=4, label="hit")
        axis.set_ylabel("Frequency band")
        axis.set_title(
            f"{run.strategy}  |  Pd={run.metrics.get('probability_of_detection', float('nan')):.3f}"
            f"  intercept rate={run.metrics.get('average_intercept_rate', float('nan')):.3f}"
            f"  coverage={run.metrics.get('coverage', float('nan')):.2f}",
            fontsize=10,
            loc="left",
        )
        axis.set_ylim(-0.5, environment.n_bands - 0.5)

    axes_flat[-1].set_xlabel("Timestep")
    handles = [
        Line2D([], [], marker="s", linestyle="", color=_TRUTH_CMAP(1.0), label="emitter active"),
        Line2D([], [], marker="o", linestyle="", color=_HIT_COLOUR, label="intercept (hit)"),
        Line2D([], [], marker="o", linestyle="", color=_MISS_COLOUR, label="miss"),
        Line2D([], [], color="#444444", label="receiver scan path"),
    ]
    axes_flat[0].legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle(
        title or f"Frequency-time interception map - {environment.name}", fontsize=12, y=0.995
    )
    fig.tight_layout()
    return _save(fig, path)


def plot_metric_comparison(
    metrics_by_strategy: dict[str, dict[str, float]],
    path: str | Path,
    *,
    metrics: Sequence[str] = HEADLINE_METRICS,
    title: str = "Sequential sweep vs Smart Scan",
) -> Path:
    """Plot a grouped bar chart of the figures of merit for each strategy.

    Metrics are min-max normalised per metric so that quantities on different scales can
    share an axis; the raw value is printed on each bar. Metrics where lower is better
    are marked in the tick label.
    """
    names = list(metrics_by_strategy)
    usable = [
        metric
        for metric in metrics
        if any(np.isfinite(metrics_by_strategy[name].get(metric, np.nan)) for name in names)
    ]
    values = np.array(
        [[metrics_by_strategy[name].get(metric, np.nan) for metric in usable] for name in names],
        dtype=np.float64,
    )
    spans = np.nanmax(np.abs(values), axis=0)
    spans[spans == 0] = 1.0
    normalised = values / spans

    x = np.arange(len(usable))
    width = 0.8 / max(1, len(names))
    fig, axis = plt.subplots(figsize=(1.15 * len(usable) + 3.0, 5.0))
    for index, name in enumerate(names):
        offset = (index - (len(names) - 1) / 2) * width
        bars = axis.bar(x + offset, normalised[index], width=width, label=name)
        for bar, raw in zip(bars, values[index]):
            if not np.isfinite(raw):
                continue
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{raw:.3g}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    labels = [
        f"{metric}\n(lower better)" if metric in LOWER_IS_BETTER else metric for metric in usable
    ]
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    axis.set_ylabel("value, normalised per metric")
    axis.set_title(title)
    axis.axhline(0.0, color="#888888", linewidth=0.8)
    axis.legend()
    axis.margins(y=0.18)
    fig.tight_layout()
    return _save(fig, path)


def plot_learning_curves(
    runs: Sequence[SimulationRun],
    path: str | Path,
    *,
    window: int = 100,
    title: str = "Reward and hit rate over time",
) -> Path:
    """Plot cumulative reward and rolling hit rate for each strategy."""
    fig, (reward_axis, rate_axis) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
    for run in runs:
        curve = build_learning_curve(run.event_reward, run.hits, window=window)
        reward_axis.plot(curve.timesteps, curve.cumulative_reward, label=run.strategy)
        rate_axis.plot(curve.timesteps, curve.rolling_hit_rate, label=run.strategy)
    reward_axis.set_ylabel("cumulative reward")
    reward_axis.set_title(title, loc="left")
    reward_axis.legend()
    reward_axis.grid(alpha=0.25)
    rate_axis.set_ylabel(f"hit rate (trailing {window})")
    rate_axis.set_xlabel("Timestep")
    rate_axis.grid(alpha=0.25)
    fig.tight_layout()
    return _save(fig, path)


def plot_calibration(
    curve: dict[str, list[float]],
    path: str | Path,
    *,
    title: str = "Activity model calibration (validation)",
) -> Path:
    """Plot predicted probability against observed frequency."""
    fig, axis = plt.subplots(figsize=(5.2, 5.0))
    axis.plot([0, 1], [0, 1], linestyle="--", color="#888888", label="perfect calibration")
    predicted = curve.get("predicted", [])
    observed = curve.get("observed", [])
    counts = np.asarray(curve.get("count", []), dtype=np.float64)
    sizes = 20.0 + 180.0 * (counts / counts.max()) if counts.size and counts.max() > 0 else 40.0
    axis.plot(predicted, observed, color="#1f4e9c", linewidth=1.4)
    axis.scatter(predicted, observed, s=sizes, color="#1f4e9c", zorder=3, label="bins")
    axis.set_xlabel("predicted P(active in next window)")
    axis.set_ylabel("observed frequency")
    axis.set_title(title, fontsize=10)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def plot_band_occupancy(
    environment: EnvironmentGrid,
    runs: Sequence[SimulationRun],
    path: str | Path,
    *,
    title: str = "Where the time went",
) -> Path:
    """Plot true band occupancy against how each strategy allocated its dwell time."""
    occupancy = environment.active.mean(axis=0)
    bands = np.arange(environment.n_bands)
    fig, axis = plt.subplots(figsize=(11, 4.4))
    axis.bar(bands, occupancy, color="#b9c6dd", label="true band occupancy")
    for run in runs:
        share = np.bincount(run.selected_band, minlength=environment.n_bands) / max(
            1, run.n_timesteps
        )
        axis.plot(bands, share, marker="o", markersize=3.5, linewidth=1.2, label=f"{run.strategy} dwell share")
    axis.set_xlabel("Frequency band")
    axis.set_ylabel("fraction")
    axis.set_title(title, loc="left")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    return _save(fig, path)


def decision_timeline_table(run: SimulationRun, n_rows: int = 25) -> list[dict[str, Any]]:
    """Return the first ``n_rows`` scheduling decisions as plain dictionaries.

    Columns follow the brief's "time / band / prediction / action / result" shape.
    """
    rows: list[dict[str, Any]] = []
    hits = run.hits
    for decision in run.decisions[:n_rows]:
        timestep = int(decision["timestep"])
        result = "hit" if (timestep < hits.size and hits[timestep]) else "miss"
        rows.append(
            {
                "timestep": timestep,
                "band": int(decision["band"]),
                "band_centre_mhz": round(float(decision["band_centre_mhz"]), 1),
                "predicted_probability": round(float(decision["combined_probability"]), 3),
                "periodicity": round(float(decision["periodicity_bonus"]), 3),
                "exploration": round(float(decision["exploration_bonus"]), 3),
                "score": round(float(decision["score"]), 3),
                "action": f"tune band {int(decision['band'])}",
                "result": result,
            }
        )
    return rows


def plot_decision_timeline(
    run: SimulationRun,
    path: str | Path,
    *,
    n_rows: int = 18,
    title: str = "Scheduler decision timeline",
) -> Path:
    """Render the decision timeline as a table figure."""
    rows = decision_timeline_table(run, n_rows=n_rows)
    if not rows:
        fig, axis = plt.subplots(figsize=(8, 2))
        axis.text(0.5, 0.5, "no recorded decisions", ha="center", va="center")
        axis.axis("off")
        return _save(fig, path)

    columns = list(rows[0])
    cell_text = [[str(row[column]) for column in columns] for row in rows]
    fig, axis = plt.subplots(figsize=(min(16, 1.5 * len(columns) + 2), 0.36 * len(rows) + 1.2))
    axis.axis("off")
    table = axis.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.25)
    for index, row in enumerate(rows, start=1):
        colour = "#e8f5ea" if row["result"] == "hit" else "#fdecec"
        for column_index in range(len(columns)):
            table[(index, column_index)].set_facecolor(colour)
    axis.set_title(f"{title} - {run.strategy}", fontsize=10, loc="left")
    fig.tight_layout()
    return _save(fig, path)


def plot_scenario_summary(
    scenario_metrics: dict[str, dict[str, dict[str, float]]],
    path: str | Path,
    *,
    metric: str = "average_intercept_rate",
    title: str = "",
) -> Path:
    """Plot one metric across scenarios and strategies.

    Args:
        scenario_metrics: ``{scenario: {strategy: metrics}}``.
        path: output path.
        metric: metric key to plot.
        title: figure title.
    """
    scenarios = list(scenario_metrics)
    strategies = sorted({name for values in scenario_metrics.values() for name in values})
    x = np.arange(len(scenarios))
    width = 0.8 / max(1, len(strategies))

    fig, axis = plt.subplots(figsize=(2.6 * len(scenarios) + 3.0, 4.6))
    for index, strategy in enumerate(strategies):
        offset = (index - (len(strategies) - 1) / 2) * width
        values = [
            float(scenario_metrics[scenario].get(strategy, {}).get(metric, np.nan))
            for scenario in scenarios
        ]
        bars = axis.bar(x + offset, values, width=width, label=strategy)
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.3g}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )
    axis.set_xticks(x)
    axis.set_xticklabels(scenarios, fontsize=9)
    axis.set_ylabel(metric)
    axis.set_title(title or f"{metric} by scenario", loc="left")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25, axis="y")
    axis.margins(y=0.15)
    fig.tight_layout()
    return _save(fig, path)
