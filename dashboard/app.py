"""Tactical console dashboard for the Smart Scan Electronic Support scheduler.

The visual language follows the Code-Raptors SIH 2026 front end: a dotted-grid ground,
monospace type, neon-green readouts, cyan accents, banner header and footer. Everything
visual is defined in :mod:`dashboard.theme` so this module stays about the data.

Unlike the reference, which renders a single pre-baked ``mock_run.json``, this console
runs the real simulation live against the sampled Turing dataset and re-runs whenever a
control changes. That brings in things the reference had no data for -- a graded
scorecard, the interception ceiling, per-metric explanations and the ablation arms -- and
those are added in the same idiom rather than bolted on.

Run with::

    streamlit run dashboard/app.py

The dashboard **loads** the saved model artifact and never retrains. It visualises a
research simulation: nothing here transmits, receives or controls RF hardware; it is
Electronic Support (passive) only.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import Config, make_rng  # noqa: E402
from common.io_utils import read_json  # noqa: E402
from dashboard import theme  # noqa: E402
from dataio.manifest import DatasetManifest  # noqa: E402
from evaluation.compare_strategies import (  # noqa: E402
    BASELINE_KEY,
    CANDIDATE_KEY,
    DEFAULT_STRATEGIES,
    build_strategy_scheduler,
    load_predictor,
)
from evaluation.metrics import (  # noqa: E402
    HEADLINE_METRICS,
    LOWER_IS_BETTER,
    RewardModel,
    describe_metric,
    relative_improvement,
)
from evaluation.scorecard import (  # noqa: E402
    activity_model_scorecard,
    oracle_ceiling,
    scheduler_scorecard,
)
from simulation.runner import SimulationRun, run_simulation  # noqa: E402
from simulation.scenarios import SCENARIO_LABELS, build_scenario  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

#: Plain-English meaning of each term in the scheduler's weighted score.
SCORE_TERM_HELP = {
    "Predicted Activity": (
        "The activity model's calibrated estimate that this band transmits in the next "
        "few cycles, from the receiver's own observation history."
    ),
    "Exploration Bonus": (
        "Grows the longer a band has gone unobserved. Stops the scheduler camping on a "
        "few known-busy bands and losing track of the rest of the spectrum."
    ),
    "Recency Bonus": "Rewards a band that produced a detection recently.",
    "Periodicity Score": (
        "Peaks when now is close to the moment this band is predicted to go active "
        "again, from the interval between its past detections. This is what catches "
        "rotating radars as the beam sweeps back round."
    ),
    "Scan Cost": (
        "Penalty for retuning far across the spectrum, standing in for the settling time "
        "a real receiver loses."
    ),
}


# --- cached loaders -------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_config() -> Config:
    """Load the project configuration once per session."""
    return Config.load(CONFIG_PATH)


@st.cache_resource(show_spinner=False)
def get_manifest(path: str) -> DatasetManifest:
    """Load the dataset manifest, falling back to the bundled demo subset.

    A public deployment ships only ``data/demo`` -- the full processed cache is hundreds of
    megabytes and the source dataset is gated, so neither can travel with the app. Rather
    than maintain a second configuration file that has to be kept in step, the console
    prefers the configured manifest when it exists and otherwise picks up the bundle. One
    config therefore works in both places.
    """
    configured = Path(path)
    if configured.exists():
        return DatasetManifest.load(configured)
    demo = Path(__file__).resolve().parent.parent / "data" / "demo" / "manifest.json"
    if demo.exists():
        return DatasetManifest.load(demo)
    return DatasetManifest.load(configured)  # raises with the actionable message


@st.cache_resource(show_spinner=False)
def get_predictor(kind: str) -> Any:
    """Load a predictor. The artifact is loaded, never retrained."""
    return load_predictor(get_config(), kind)


@st.cache_data(show_spinner=False)
def get_report(path: str) -> dict[str, Any] | None:
    """Load a saved JSON report if it exists."""
    report_path = Path(path)
    return read_json(report_path) if report_path.exists() else None


@st.cache_data(show_spinner="BUILDING ELECTROMAGNETIC ENVIRONMENT...")
def build_environment_cached(
    entry_path: str, entry_name: str, scenario: str, horizon: int, max_pulses: int
) -> dict[str, Any]:
    """Build one scenario environment and return it in a cache-friendly form."""
    config = get_config()
    from dataio.tdc_interface import load_pulse_train

    record = load_pulse_train(entry_path).contiguous_window(max_pulses)
    record.source_path = Path(entry_path)
    built = build_scenario(record, scenario, config.section("environment"), max_timesteps=horizon)
    return {
        "active": built.environment.active,
        "n_pulses": built.environment.n_pulses,
        "mean_aoa": built.environment.mean_aoa,
        "band_edges_mhz": built.environment.band_edges_mhz,
        "timestep_us": built.environment.timestep_us,
        "t0_us": built.environment.t0_us,
        "name": f"{entry_name}:{scenario}",
        "metadata": built.environment.metadata,
        "summary": built.summary(),
    }


def rehydrate(payload: dict[str, Any]):
    """Rebuild an :class:`EnvironmentGrid` from cached arrays."""
    from simulation.environment import EnvironmentGrid

    return EnvironmentGrid(
        active=payload["active"],
        n_pulses=payload["n_pulses"],
        mean_aoa=payload["mean_aoa"],
        band_edges_mhz=payload["band_edges_mhz"],
        timestep_us=payload["timestep_us"],
        t0_us=payload["t0_us"],
        name=payload["name"],
        metadata=payload["metadata"],
    )


def run_strategy(environment, strategy_key: str, horizon: int, seed_stream: int) -> SimulationRun:
    """Run one strategy against an environment with the standard seeding."""
    config = get_config()
    spec = next(s for s in DEFAULT_STRATEGIES if s.key == strategy_key)
    predictor = get_predictor(spec.predictor) if spec.predictor != "none" else None
    scheduler = build_strategy_scheduler(
        spec,
        n_bands=environment.n_bands,
        rng=make_rng(config.seed, stream=5000 + seed_stream),
        scheduler_cfg=config.section("scheduler"),
        predictor=predictor,
    )
    run = run_simulation(
        environment=environment,
        scheduler=scheduler,
        receiver_cfg=config.section("receiver"),
        features_cfg=config.section("features"),
        reward_model=RewardModel.from_config(config.section("reward")),
        rng=make_rng(config.seed, stream=9000 + seed_stream),
        n_timesteps=horizon,
        prediction_window=int(config.get("activity_model.prediction_window", 5)),
        record_decisions=True,
    )
    run.strategy = spec.key
    return run


# --- charts ---------------------------------------------------------------------------


def spectrum_figure(environment, run: SimulationRun, start: int, end: int) -> go.Figure:
    """Frequency-time console map, built so the receiver's story reads at a glance.

    Three deliberate choices keep it legible where the earlier version was busy:

    * **No connecting line.** Joining consecutive dwells drew long strokes leaping across
      the spectrum, which added no information -- the receiver does not pass through the
      bands between two tunings -- and buried everything underneath.
    * **Misses recede, hits advance.** Misses outnumber hits several times over, so they are
      small and dim; hits are bright and larger. The eye should land on interceptions.
    * **Truth is a field, not a foreground.** Ground truth is drawn small and semi-transparent
      so it reads as terrain, rather than competing with the receiver's own marks.
    """
    end = int(min(end, environment.n_timesteps, run.n_timesteps))
    start = int(max(0, start))
    figure = go.Figure()

    steps, bands = np.nonzero(environment.active[start:end])
    figure.add_trace(
        go.Scatter(
            x=steps + start,
            y=bands,
            mode="markers",
            marker={"size": 6, "symbol": "square", "color": theme.TRUTH, "opacity": 0.45},
            name="Emitter Active",
            hovertemplate="T %{x} | Band %{y} | ACTIVE<extra></extra>",
        )
    )

    window = slice(start, end)
    timesteps = np.arange(start, end)
    selected = run.selected_band[window]
    hits = run.hits[window]

    figure.add_trace(
        go.Scatter(
            x=timesteps[~hits],
            y=selected[~hits],
            mode="markers",
            marker={"size": 5, "symbol": "circle", "color": theme.ACCENT_MISS, "opacity": 0.55},
            name="Scan Failed (MISS)",
            hovertemplate="T %{x} | Band %{y} | MISS<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=timesteps[hits],
            y=selected[hits],
            mode="markers",
            marker={
                "size": 11,
                "symbol": "circle",
                "color": theme.ACCENT_HIT,
                "line": {"width": 1.2, "color": "#FFFFFF"},
            },
            name="Target Intercept (HIT)",
            hovertemplate="T %{x} | Band %{y} | HIT<extra></extra>",
        )
    )

    figure.update_layout(
        **theme.plotly_layout(
            height=430,
            xaxis=theme.axis("TIME (CYCLES)", range=[start - 0.5, end + 0.5], showgrid=False),
            yaxis=theme.axis(
                "FREQUENCY BAND", range=[-0.5, environment.n_bands - 0.5], dtick=4
            ),
            hovermode="closest",
        )
    )
    return figure


def reward_figure(runs: dict[str, SimulationRun], keys: list[str]) -> go.Figure:
    """Cumulative reward over the run, drawn in left-to-right when scrolled into view.

    A single averaged number hides *when* a strategy earns: this shows the open-loop sweep
    drifting steadily negative on empty dwells while an adaptive scheduler climbs away from
    it. Animating the draw makes that divergence the thing the eye follows.
    """
    palette = {BASELINE_KEY: theme.ACCENT_NEUTRAL, CANDIDATE_KEY: theme.ACCENT_HIT}
    series: list[tuple[str, np.ndarray, np.ndarray]] = []
    for key in keys:
        cumulative = np.cumsum(runs[key].event_reward)
        stride = max(1, cumulative.size // 600)
        series.append((key, np.arange(0, cumulative.size, stride), cumulative[::stride]))

    def traces(fraction: float) -> list[go.Scatter]:
        drawn = []
        for key, x, y in series:
            cut = max(2, int(len(x) * fraction))
            drawn.append(
                go.Scatter(
                    x=x[:cut],
                    y=y[:cut],
                    mode="lines",
                    name=key,
                    line={"color": palette.get(key, theme.ACCENT_INFO), "width": 2},
                    hovertemplate="T %{x} | cumulative reward %{y:.1f}<extra></extra>",
                )
            )
        return drawn

    figure = go.Figure(data=traces(1.0))
    figure.frames = [go.Frame(data=traces(step / 24), name=str(step)) for step in range(1, 25)]

    limit = max(float(np.max(np.abs(y))) for _, _, y in series) if series else 1.0
    span_end = max(float(x[-1]) for _, x, _ in series) if series else 1.0
    # Horizontal zero reference kept: it marks where a strategy stops losing reward and
    # starts gaining, which is the point of the chart.
    figure.add_hline(y=0, line={"color": theme.GRID, "width": 1, "dash": "dot"})
    figure.update_layout(
        **theme.plotly_layout(
            height=320,
            xaxis=theme.axis("TIME (CYCLES)", range=[0, span_end], showgrid=False),
            yaxis=theme.axis("CUMULATIVE REWARD", range=[-limit * 0.25, limit * 1.15]),
        )
    )
    return figure


def diagnostics_figure(
    labels: list[str], baseline: list[float], candidate: list[float]
) -> go.Figure:
    """Grouped bars of the figures of merit, in the reference style.

    Bars share one axis and carry their raw value as a label, exactly as the reference
    console does. Frames are attached for the scroll-triggered growth animation; the
    figure's initial state is the finished chart, so it still reads correctly if the
    animation never fires.
    """
    ceiling = max(max(baseline), max(candidate), 1e-9)

    def bars(factor: float) -> list[go.Bar]:
        return [
            go.Bar(
                x=labels,
                y=[v * factor for v in baseline],
                name="Sequential Scan",
                marker_color=theme.ACCENT_NEUTRAL,
                text=[f"{v:.3g}" for v in baseline],
                textposition="auto",
            ),
            go.Bar(
                x=labels,
                y=[v * factor for v in candidate],
                name="Smart Scan",
                marker_color=theme.ACCENT_HIT,
                text=[f"{v:.3g}" for v in candidate],
                textposition="auto",
            ),
        ]

    figure = go.Figure(data=bars(1.0))
    figure.frames = [go.Frame(data=bars(step / 20), name=str(step)) for step in range(1, 21)]
    figure.update_layout(
        **theme.plotly_layout(
            barmode="group",
            bargap=0.35,
            height=400,
            xaxis=theme.axis("", showgrid=False),
            yaxis=theme.axis("", range=[0, ceiling * 1.2]),
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.15,
                "xanchor": "center",
                "x": 0.5,
                "font": {"color": theme.TEXT_BRIGHT},
            },
        )
    )
    return figure


def render_scroll_animated(figure: go.Figure, height: int = 430) -> None:
    """Render a figure that plays its frames the first time it scrolls into view.

    Streamlit has no scroll hook, so the chart is rendered as a self-contained component
    and an ``IntersectionObserver`` inside it starts the animation when at least a third
    of the chart becomes visible. It fires once, then disconnects -- an animation that
    replays on every scroll past would be a distraction rather than a flourish.

    The chart is drawn at its final values first and only reset to zero once the observer
    is armed, so a browser that never fires the callback still shows a correct chart
    rather than an empty one.
    """
    import streamlit.components.v1 as components

    payload = figure.to_json()
    components.html(
        f"""
        <div id="chart"></div>
        <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
        <script>
          const figure = {payload};
          const target = document.getElementById("chart");
          // Frames must ride on the figure object. The (div, data, layout, config)
          // signature silently drops them, which leaves Plotly.animate with nothing to
          // play and the chart stuck at whatever it was last set to.
          Plotly.newPlot(target, {{
            data: figure.data,
            layout: figure.layout,
            frames: figure.frames,
            config: {{displayModeBar: false, responsive: true}}
          }});

          const zeroed = figure.frames[0].data;
          let played = false;
          const observer = new IntersectionObserver((entries) => {{
            entries.forEach((entry) => {{
              if (entry.isIntersecting && !played) {{
                played = true;
                observer.disconnect();
                Plotly.animate(target, {{data: zeroed}}, {{
                  transition: {{duration: 0}}, frame: {{duration: 0, redraw: false}}
                }}).then(() => Plotly.animate(target, null, {{
                  transition: {{duration: 0}},
                  frame: {{duration: 40, redraw: false}},
                  mode: "immediate"
                }}));
              }}
            }});
          }}, {{threshold: 0.33}});
          observer.observe(target);
        </script>
        """,
        height=height,
    )


# --- panels ---------------------------------------------------------------------------


def render_verdict(runs: dict[str, SimulationRun], strategy_key: str) -> None:
    """State the headline outcome in one plain sentence, before any chart.

    Someone opening this console should learn what happened without first decoding five
    panels of instrumentation.
    """
    baseline = runs[BASELINE_KEY].metrics
    candidate = runs[strategy_key].metrics
    ratio = (
        candidate["average_intercept_rate"] / baseline["average_intercept_rate"]
        if baseline["average_intercept_rate"] > 0
        else float("nan")
    )
    if not np.isfinite(ratio):
        return
    if ratio >= 1.05:
        colour, verb = theme.ACCENT_HIT, f"intercepted {ratio:.2f}x as many transmissions as"
    elif ratio <= 0.95:
        colour, verb = theme.ACCENT_MISS, f"intercepted only {ratio:.2f}x as many transmissions as"
    else:
        colour, verb = theme.ACCENT_WARN, "intercepted about as many transmissions as"
    theme.status_line(
        f"RESULT :: on this environment <b>{strategy_key}</b> {verb} the open-loop sweep, "
        f"catching {candidate['average_intercept_rate'] * 100:.1f} per 100 cycles against "
        f"{baseline['average_intercept_rate'] * 100:.1f}.",
        colour,
    )


def render_briefing() -> None:
    """Render the collapsible orientation panel."""
    with st.expander("MISSION BRIEFING :: WHAT THIS CONSOLE SHOWS", expanded=False):
        st.markdown(
            """
**The problem.** A radar-warning receiver must watch a wide frequency range but can only
listen to **one band at a time**. Every cycle spent on one band is a cycle deaf to the
other 31. Tune to the wrong band at the wrong moment and the transmission is missed.

**The two algorithms.**

- **SEQUENTIAL** - the open-loop baseline. Marches through bands in fixed order, giving
  every band identical time whether or not anything is there. This is what fielded
  open-loop systems do.
- **SMART** - decides where to point next from a machine-learning prediction of which
  bands are about to be busy, plus what it has learned during the run.

**Reading the spectrum map.** Teal squares are transmissions that genuinely happened
(ground truth, which the receiver never sees). The receiver's own path is drawn over it:
green where it was tuned to the right band at the right moment, red where it looked and
found nothing.

**Why the percentages look small.** With one band observable out of 32, no receiver can
catch more than a fraction of everything transmitted. The scorecard therefore reports what
fraction of the *achievable* maximum each strategy reached.

**Scope.** Electronic Support - passive listening and scheduling. Nothing transmits, jams
or touches radio hardware.
            """
        )


def render_scorecard(
    environment, runs: dict[str, SimulationRun], config: Config, strategy_key: str
) -> None:
    """Render the graded assessment for the selected strategy and the activity model."""
    st.markdown("#### MISSION SCORECARD")
    receiver_cfg = config.section("receiver")
    ceiling = oracle_ceiling(
        environment,
        detection_probability=float(receiver_cfg.get("detection_probability", 0.95)),
        instantaneous_bandwidth=int(receiver_cfg.get("instantaneous_bandwidth", 1)),
        horizon=runs[strategy_key].n_timesteps,
    )
    card = scheduler_scorecard(
        runs[BASELINE_KEY].metrics,
        runs[strategy_key].metrics,
        oracle=ceiling,
        candidate_name=strategy_key,
    )

    model_report = get_report(
        str(config.path_for("paths.results_dir") / "activity_model_test.json")
    )
    model_card = (
        activity_model_scorecard(
            {
                **model_report.get("xgboost", {}),
                "positive_rate": model_report.get("positive_rate", float("nan")),
                "n_rows": model_report.get("n_rows", float("nan")),
            }
        )
        if model_report
        else None
    )

    left, right = st.columns(2)
    with left:
        theme.grade_badge(
            f"{strategy_key} scheduler :: this environment",
            card.grade,
            card.overall,
            "50 = level with the sequential sweep. 100 = twice as good.",
        )
    with right:
        if model_card:
            theme.grade_badge(
                "activity model :: held-out test set",
                model_card.grade,
                model_card.overall,
                "0 = no better than guessing. 100 = perfect.",
            )
        else:
            theme.status_line(
                "NO MODEL REPORT :: run training/evaluate_model.py", theme.ACCENT_WARN
            )

    columns = st.columns(3)
    for column, component in zip(columns, card.components):
        with column:
            state = "DEGRADED" if component.regression else "IMPROVED"
            colour = theme.ACCENT_MISS if component.regression else theme.ACCENT_HIT
            theme.callout(f"{component.label} :: {state}", f"{component.score:.0f}", colour, 110)
            st.progress(min(1.0, max(0.0, component.score / 100.0)))
            st.caption(component.detail)

    if card.regressions:
        theme.status_line(
            f"ADVISORY :: worse than the plain sweep on {', '.join(card.regressions)}. "
            "An adaptive scheduler concentrates time on productive bands, so it cannot "
            "also guarantee the fixed-period coverage a sweep gives for free.",
            theme.ACCENT_WARN,
        )

    fraction = card.context.get("fraction_of_ceiling", float("nan"))
    baseline_fraction = card.context.get("baseline_fraction_of_ceiling", float("nan"))
    if np.isfinite(fraction):
        columns = st.columns(3)
        columns[0].metric(
            "Theoretical Ceiling",
            f"{ceiling['oracle_intercept_rate']:.3f}",
            help=(
                "Best any receiver of this bandwidth could do: always already tuned to a "
                "transmitting band, with no dwell, retune or knowledge limits. Unreachable "
                "by construction - it exists to make the measured numbers interpretable."
            ),
        )
        columns[1].metric(
            f"{strategy_key} achieves",
            f"{fraction:.1%}",
            f"{(fraction - baseline_fraction) * 100:+.1f} pts vs sweep",
        )
        columns[2].metric("Sequential achieves", f"{baseline_fraction:.1%}")

    experiment = get_report(
        str(config.path_for("paths.results_dir") / "experiment_results.json")
    )
    overall_card = (experiment or {}).get("scorecards", {}).get("scheduler")
    if overall_card:
        environments = sum((experiment or {}).get("environments_per_scenario", {}).values())
        if strategy_key == CANDIDATE_KEY:
            note = (
                f"across the whole test split ({environments} environments) the last run "
                f"scored {CANDIDATE_KEY} {overall_card['grade']} {overall_card['overall']:.0f}/100"
            )
        else:
            note = (
                f"the recorded split-wide figure covers {CANDIDATE_KEY} "
                f"({overall_card['grade']} {overall_card['overall']:.0f}/100), not {strategy_key}"
            )
        theme.status_line(f"SINGLE ENVIRONMENT READOUT :: {note}.")


def render_glossary() -> None:
    """Render the glossary of terms and the metric reference."""
    with st.expander("GLOSSARY :: EVERY TERM ON THIS CONSOLE", expanded=False):
        st.markdown(
            """
**BAND** - one slice of the frequency range. The spectrum is split into 32; the receiver
hears one at a time.

**CYCLE / TIMESTEP** - one tick of the simulation clock (2 ms). One observation per cycle.

**DWELL** - consecutive cycles spent on a band before re-deciding.

**PDW (Pulse Descriptor Word)** - the record of one detected pulse: arrival time,
frequency, width, bearing, amplitude.

**PRI (Pulse Repetition Interval)** - the gap between a radar's pulses. Estimated here
from the gaps between *detections of a band*, which is what tells the scheduler when that
band is likely to be active again.

**HIT / INTERCEPT** - the receiver was tuned to a genuinely transmitting band and detected
it.

**OCCUPANCY** - fraction of all band-cycles carrying a transmission. Higher means a busier,
more crowded spectrum.

**GROUND TRUTH** - what genuinely happened. The receiver never sees it; it is used only to
score performance after the fact.

**SPATIALLY SCANNING EMITTER** - a rotating antenna. A fixed receiver hears it only while
the beam sweeps past, so it appears in short, regular bursts.

**FREQUENCY-AGILE EMITTER** - hops between frequencies, so activity migrates across bands
and past behaviour is a weaker guide to the future.
            """
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "METRIC": metric,
                        "DIRECTION": "lower is better"
                        if metric in LOWER_IS_BETTER
                        else "higher is better",
                        "MEANING": describe_metric(metric),
                    }
                    for metric in HEADLINE_METRICS
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )


# --- main -----------------------------------------------------------------------------


def main() -> None:
    """Render the console."""
    st.set_page_config(layout="wide", page_title="Smart Scan EW Dashboard")
    theme.inject_css()
    config = get_config()

    theme.header(
        "EW RECEIVER SCHEDULER :: SMART SCAN",
        "Tactical Interception &amp; Spectrum Management Console",
    )

    try:
        manifest = get_manifest(str(config.path_for("paths.manifest")))
    except FileNotFoundError as exc:
        theme.status_line(f"SYSTEM ALERT :: {exc}", theme.ACCENT_MISS)
        st.stop()

    with st.sidebar:
        st.markdown("#### MISSION CONFIGURATION")
        split = st.selectbox("DATA SPLIT", ["test", "validation", "train"], index=0)
        entries = manifest.for_split(split)
        if not entries:
            theme.status_line(f"NO PULSE TRAINS IN SPLIT {split.upper()}", theme.ACCENT_MISS)
            st.stop()

        emitter_counts = [int(e.characterisation.get("n_emitters") or 0) for e in entries]
        default_index = (
            min(range(len(entries)), key=lambda i: abs(emitter_counts[i] - 15))
            if any(emitter_counts)
            else 0
        )
        entry_index = st.selectbox(
            "PULSE TRAIN",
            range(len(entries)),
            index=default_index,
            format_func=lambda i: f"{entries[i].name}  [{entries[i].stratum}]",
            help=(
                "One recorded electromagnetic environment. The default is a train busy "
                "enough to make the spectrum map readable, not the first in the list."
            ),
        )
        entry = entries[entry_index]
        scenario = st.selectbox(
            "SCENARIO PROFILE",
            ["spatial_scan", "frequency_agile", "mixed"],
            format_func=lambda key: SCENARIO_LABELS.get(key, key).upper(),
        )
        horizon = st.slider(
            "DURATION (CYCLES)",
            200,
            int(config.get("environment.max_timesteps", 5000)),
            int(config.get("simulation.n_timesteps", 4000)),
            step=100,
        )
        st.divider()
        st.caption(f"SOURCE :: {manifest.source.upper()}")
        if manifest.source == "mock":
            st.caption(":orange[SYNTHETIC GENERATOR - REAL DATASET GATED]")
        st.caption(f"SEED :: {config.seed}")
        st.caption(f"ARTIFACT :: {config.path_for('paths.model_artifact').name}")

    render_briefing()
    # Reserved now, filled once the strategy control below has been read. A container
    # holds its place in the layout, so the takeaway can lead the page even though it
    # depends on a control rendered further down.
    verdict_container = st.container()

    payload = build_environment_cached(
        entry.path,
        entry.name,
        scenario,
        horizon,
        int(config.get("data.max_pulses_per_train", 400_000)),
    )
    environment = rehydrate(payload)
    if environment.n_timesteps == 0:
        theme.status_line(
            "SYSTEM ALERT :: environment empty after scenario filtering", theme.ACCENT_MISS
        )
        st.stop()

    vis_left, vis_right = st.columns([3.5, 1])
    with vis_left:
        heatmap_container = st.container()
        controls_container = st.container()
    with vis_right:
        telemetry_container = st.container()

    # Controls execute first so the scrubber value is available to the map rendered above.
    with controls_container:
        st.markdown("#### TACTICAL DISPLAY CONTROLS")
        control_a, control_b, control_c = st.columns([1.4, 2.4, 1.2])
        with control_a:
            strategy_key = st.radio(
                "ACTIVE ALGORITHM",
                [spec.key for spec in DEFAULT_STRATEGIES],
                index=[spec.key for spec in DEFAULT_STRATEGIES].index(CANDIDATE_KEY),
                format_func=lambda x: f"[{x.upper()}]",
            )

        with st.spinner("EXECUTING SCHEDULERS..."):
            runs = {
                key: run_strategy(environment, key, horizon, seed_stream=entry_index)
                for key in {BASELINE_KEY, CANDIDATE_KEY, strategy_key}
            }
        run = runs[strategy_key]

        with control_b:
            cursor = st.slider(
                "TIMELINE SCRUBBER (CYCLE)",
                0,
                run.n_timesteps - 1,
                min(400, run.n_timesteps - 1),
                help="Moves the readout and the marker line to a moment in the run.",
            )
        with control_c:
            span = st.slider(
                "WINDOW (CYCLES)",
                100,
                min(1200, run.n_timesteps),
                min(400, run.n_timesteps),
                step=50,
                help="How much of the run the map shows around the scrubber position.",
            )

    window_start = max(0, min(cursor - span + 1, run.n_timesteps - span))

    with verdict_container:
        render_verdict(runs, strategy_key)

    with heatmap_container:
        st.markdown("#### FREQUENCY-TIME SPECTRUM HEATMAP")
        st.plotly_chart(
            spectrum_figure(environment, run, window_start, cursor + 1),
            use_container_width=True,
        )
        visible = run.hits[window_start : cursor + 1]
        st.caption(
            f"Showing cycles {window_start}-{cursor} of {run.n_timesteps} "
            f"({(cursor + 1 - window_start) * environment.timestep_us / 1000:.0f} ms of "
            f"simulated time) — {int(visible.sum())} intercepts in view. "
            "Teal squares are real transmissions; the receiver only ever sees one band per "
            "cycle."
        )

    with telemetry_container:
        st.markdown("#### RECEIVER TELEMETRY")
        band = int(run.selected_band[cursor])
        st.metric("Current Target Band", f"BAND {band}")
        st.metric("Centre Frequency", f"{environment.band_centres_mhz[band]:,.0f} MHz")
        probability = run.predicted_probability[cursor]
        st.metric(
            "Predicted Activity", "N/A" if not np.isfinite(probability) else f"{probability:.3f}"
        )
        st.divider()
        error_change = relative_improvement(
            runs[BASELINE_KEY].metrics["average_intercept_time_error"],
            run.metrics["average_intercept_time_error"],
            "average_intercept_time_error",
        )
        st.metric(
            "Avg Time Error",
            f"{run.metrics['average_intercept_time_error']:.1f} cyc",
            "n/a" if not np.isfinite(error_change) else f"{error_change * 100:+.1f}% vs Seq",
        )
        correct = run.metrics.get("percentage_of_correct_predictions", float("nan"))
        st.metric(
            "Correct Predictions",
            "N/A" if not np.isfinite(correct) else f"{correct:.1f}%",
            "open loop makes none" if not np.isfinite(correct) else "AI model",
        )

    st.divider()

    summary = payload["summary"]
    columns = st.columns([1, 1, 1.8, 1, 1])
    columns[0].metric("Spectrum Bands", environment.n_bands)
    columns[1].metric("Active Emitters", f"{summary['emitters_kept']}")
    columns[2].metric("Scenario Profile", scenario.replace("_", " ").upper())
    columns[3].metric("Duration (Cycles)", environment.n_timesteps)
    columns[4].metric("Occupancy", f"{environment.occupancy:.3f}")

    st.divider()
    render_scorecard(environment, runs, config, strategy_key)
    st.divider()

    st.markdown("#### ALGORITHM PERFORMANCE DIAGNOSTICS")

    labels = [
        "Detection (Pd)",
        "False Alarm (Pfa)",
        "Sensitivity",
        "Intercept Rate",
        "Avg Reward",
    ]
    keys = [
        "probability_of_detection",
        "probability_of_false_alarm",
        "sensitivity",
        "average_intercept_rate",
        "average_reward",
    ]
    baseline_values = [float(runs[BASELINE_KEY].metrics.get(k, 0.0)) for k in keys]
    candidate_values = [float(run.metrics.get(k, 0.0)) for k in keys]

    render_scroll_animated(diagnostics_figure(labels, baseline_values, candidate_values), height=430)
    st.caption(
        "Bars grow in when the panel scrolls into view. Raw values are printed on each "
        "bar: the metrics span two orders of magnitude, so the smaller ones are short by "
        "nature rather than by omission."
    )

    st.markdown("#### CUMULATIVE REWARD")
    render_scroll_animated(
        reward_figure(
            runs, [BASELINE_KEY] + ([strategy_key] if strategy_key != BASELINE_KEY else [])
        ),
        height=350,
    )
    st.caption(
        "The sweep loses ground steadily on empty dwells; an adaptive scheduler crosses "
        "into positive reward and keeps climbing."
    )

    st.divider()

    left, right = st.columns(2)
    with left:
        st.markdown("#### AI DECISION EXPLAINABILITY")
        decisions = [d for d in run.decisions if d["timestep"] <= cursor]
        if not decisions:
            theme.status_line(
                "EXPLAINABILITY UNAVAILABLE :: the open-loop sweep does not choose, it "
                "advances in fixed order - the behaviour the problem statement criticises.",
                theme.ACCENT_WARN,
            )
        else:
            decision = decisions[-1]
            weights = config.get("scheduler.weights", {})
            rows = [
                (
                    "Predicted Activity",
                    decision["combined_probability"],
                    weights.get("w1_predicted_probability"),
                ),
                ("Exploration Bonus", decision["exploration_bonus"], weights.get("w2_exploration_bonus")),
                ("Recency Bonus", decision["recency_bonus"], weights.get("w3_recency_bonus")),
                ("Periodicity Score", decision["periodicity_bonus"], weights.get("w4_periodicity_bonus")),
                ("Scan Cost", -decision["scan_cost"], weights.get("w5_scan_cost")),
            ]
            theme.status_line(
                f"CYCLE {decision['timestep']} ACTION :: scanner directed to BAND {decision['band']}"
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "VECTOR": name,
                            "VALUE": f"{value:.3f}",
                            "WEIGHT": f"{float(weight):.2f}",
                            "CONTRIBUTION": f"{value * float(weight):.3f}",
                        }
                        for name, value, weight in rows
                    ]
                ),
                use_container_width=True,
                hide_index=True,
                height=215,
            )
            winner = max(((n, v * float(w)) for n, v, w in rows), key=lambda item: item[1])
            theme.callout("Final Calculated Score", f"{decision['score']:.2f}")
            st.caption(f"**{winner[0]}** dominated this decision. {SCORE_TERM_HELP[winner[0]]}")
            metric_columns = st.columns(2)
            metric_columns[0].metric("PRI Estimate", f"{decision['pri_estimate_us'] / 1000:.1f} ms")
            metric_columns[1].metric("Periodicity Score", f"{decision['periodicity_score']:.3f}")

    with right:
        st.markdown("#### SCHEDULER DECISION LOG")
        end = min(cursor + 1, run.n_timesteps)
        st.dataframe(
            pd.DataFrame(
                {
                    "CYCLE": np.arange(window_start, end),
                    "BAND": run.selected_band[window_start:end],
                    "PROBABILITY": np.round(run.predicted_probability[window_start:end], 3),
                    "PULSES": run.signal_count[window_start:end],
                    "OUTCOME": np.where(run.hits[window_start:end], "HIT", "MISS"),
                }
            ).sort_values("CYCLE", ascending=False),
            use_container_width=True,
            hide_index=True,
            height=425,
            column_config={
                "PROBABILITY": st.column_config.NumberColumn(format="%.3f"),
                "OUTCOME": st.column_config.TextColumn(
                    help="HIT means the receiver was on a transmitting band and detected it."
                ),
            },
        )

    st.divider()

    st.markdown("#### FIGURES OF MERIT :: SEQUENTIAL VS SMART")
    merit = pd.DataFrame(
        [
            {
                "METRIC": metric.replace("_", " ").title(),
                "BETTER WHEN": "lower" if metric in LOWER_IS_BETTER else "higher",
                "SEQUENTIAL": runs[BASELINE_KEY].metrics.get(metric, float("nan")),
                "SMART": runs[CANDIDATE_KEY].metrics.get(metric, float("nan")),
                "IMPROVEMENT": relative_improvement(
                    runs[BASELINE_KEY].metrics.get(metric, float("nan")),
                    runs[CANDIDATE_KEY].metrics.get(metric, float("nan")),
                    metric,
                )
                * 100.0,
                "MEANING": describe_metric(metric),
            }
            for metric in HEADLINE_METRICS
        ]
    )
    st.dataframe(
        merit,
        use_container_width=True,
        hide_index=True,
        column_config={
            "SEQUENTIAL": st.column_config.NumberColumn(format="%.4f"),
            "SMART": st.column_config.NumberColumn(format="%.4f"),
            "IMPROVEMENT": st.column_config.NumberColumn(
                help="Positive always means Smart Scan is better.", format="%+.1f%%"
            ),
            "MEANING": st.column_config.TextColumn(width="large"),
        },
    )
    st.download_button(
        "DOWNLOAD THIS RUN (CSV)",
        merit.to_csv(index=False).encode("utf-8"),
        file_name=f"{environment.name.replace(':', '_')}_figures_of_merit.csv",
        mime="text/csv",
    )
    st.caption(
        "Both strategies ran on this identical environment with identical receiver seeds, "
        "so the only difference is where each chose to point. A positive improvement always "
        "means Smart Scan is better, whichever direction the raw metric runs in."
    )

    render_glossary()
    theme.footer(
        [
            "Smart Scan Strategy for Electronic Warfare :: Electronic Support (passive) research simulation",
            "No RF transmission, jamming or hardware control :: &copy; 2026",
        ]
    )


if __name__ == "__main__":
    main()
