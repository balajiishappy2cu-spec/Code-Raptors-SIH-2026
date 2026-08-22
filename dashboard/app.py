"""Streamlit dashboard for the Smart Scan Electronic Support scheduler.

The page is built to be readable by someone who has not seen the code. Every number
carries a plain-English explanation, the scores state the formula that produced them, and
the scheduler's own decision is broken into the weighted terms that caused it.

Sections: Scorecard, Environment, Current receiver, Spectrum, Performance, Comparison,
Explainability, Glossary.

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
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import Config, make_rng  # noqa: E402
from common.io_utils import read_json  # noqa: E402
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
from visualization.plots import decision_timeline_table  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

#: Colour used for a grade badge.
GRADE_COLOURS = {"A": "#1a7f37", "B": "#2f7d32", "C": "#b26a00", "D": "#c25b00", "E": "#d1242f"}

#: Plain-English meaning of each term in the scheduler's weighted score.
SCORE_TERM_HELP = {
    "predicted probability": (
        "The activity model's calibrated estimate that this band will transmit in the "
        "next few timesteps, blended with a Thompson Sampling draw from what the receiver "
        "has learned about this band during the run."
    ),
    "exploration bonus": (
        "Grows the longer a band has gone unobserved. This is what stops the scheduler "
        "camping on a few known-busy bands and losing track of the rest of the spectrum."
    ),
    "recency bonus": (
        "Rewards a band that produced a detection recently, on the assumption that an "
        "emitter just seen is probably still there."
    ),
    "periodicity bonus": (
        "Peaks when now is close to the moment this band is predicted to go active again, "
        "based on the interval between its past detections. This is what catches rotating "
        "radars as their beam sweeps back round."
    ),
    "scan cost": (
        "A penalty for retuning far across the spectrum, standing in for the time a real "
        "receiver loses settling on a new frequency."
    ),
}


@st.cache_resource(show_spinner=False)
def get_config() -> Config:
    """Load the project configuration once per session."""
    return Config.load(CONFIG_PATH)


@st.cache_resource(show_spinner=False)
def get_manifest(path: str) -> DatasetManifest:
    """Load the dataset manifest."""
    return DatasetManifest.load(path)


@st.cache_resource(show_spinner=False)
def get_predictor(kind: str) -> Any:
    """Load a predictor (the XGBoost artifact is loaded, never retrained)."""
    return load_predictor(get_config(), kind)


@st.cache_data(show_spinner=False)
def get_model_report(path: str) -> dict[str, Any] | None:
    """Load the saved activity-model test report, if the model has been evaluated."""
    report_path = Path(path)
    return read_json(report_path) if report_path.exists() else None


@st.cache_data(show_spinner="Building environment...")
def build_environment_cached(
    entry_path: str, entry_name: str, scenario: str, horizon: int, max_pulses: int
) -> dict[str, Any]:
    """Build one scenario environment and return it in a cache-friendly form."""
    config = get_config()
    from dataio.tdc_interface import load_pulse_train

    record = load_pulse_train(entry_path).contiguous_window(max_pulses)
    record.source_path = Path(entry_path)
    built = build_scenario(
        record, scenario, config.section("environment"), max_timesteps=horizon
    )
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


def spectrum_frame(environment, run: SimulationRun, window: tuple[int, int]) -> pd.DataFrame:
    """Build a long-form frame of truth and receiver path for the spectrum chart."""
    start, end = window
    end = min(end, environment.n_timesteps, run.n_timesteps)
    rows: list[dict[str, Any]] = []
    truth = environment.active[start:end]
    for offset in range(truth.shape[0]):
        timestep = start + offset
        for band in np.flatnonzero(truth[offset]):
            rows.append({"timestep": timestep, "band": int(band), "kind": "emitter active"})
    for timestep in range(start, end):
        rows.append(
            {
                "timestep": timestep,
                "band": int(run.selected_band[timestep]),
                "kind": "receiver hit" if run.hits[timestep] else "receiver miss",
            }
        )
    return pd.DataFrame(rows)


def grade_badge(label: str, grade: str, score: float, caption: str) -> None:
    """Render a coloured grade badge with its score."""
    colour = GRADE_COLOURS.get(grade, "#555555")
    st.markdown(
        f"""
        <div style="border:1px solid #d0d7de;border-radius:10px;padding:14px 18px;">
          <div style="font-size:0.82rem;text-transform:uppercase;letter-spacing:.05em;
                      opacity:.75;">{label}</div>
          <div style="display:flex;align-items:baseline;gap:12px;margin-top:4px;">
            <span style="font-size:2.6rem;font-weight:700;color:{colour};">{grade}</span>
            <span style="font-size:1.5rem;font-weight:600;">{score:.0f}<span
                  style="font-size:0.9rem;opacity:.6;">/100</span></span>
          </div>
          <div style="font-size:0.85rem;opacity:.8;margin-top:6px;">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_how_to_read() -> None:
    """Render the top-level explainer."""
    with st.expander("New here? What this page is showing", expanded=False):
        st.markdown(
            """
**The problem.** A radar-warning receiver has to watch a very wide frequency range, but
it can only listen to one narrow **band** at a time. So it has to keep re-tuning, and
every moment it spends on one band is a moment it is deaf to all the others. If it tunes
to the wrong band at the wrong moment, it misses the transmission entirely.

**The two strategies being compared.**

- **Sequential sweep** (the baseline) marches through the bands in fixed order,
  `0, 1, 2, ... , 31, 0, ...`, giving every band the same time whether or not anything is
  there. Simple, predictable, and guaranteed to look at everything eventually. This is
  what current open-loop systems do.
- **Smart Scan** decides where to point next, using a machine-learning model that predicts
  which bands are about to be busy plus what it has learned about each band during the run.

**How to read the picture.** In the Spectrum chart, light points are transmissions that
genuinely happened (ground truth). The receiver's own path is drawn on top: green where it
was tuned to the right band at the right moment, red where it looked and found nothing.

**Why the percentages look small.** With one band observable out of 32, a receiver can
never catch more than a small fraction of everything transmitted. The Scorecard therefore
shows what fraction of the *achievable* maximum each strategy reached, which is the
meaningful comparison.

**Important caveat.** These runs use a synthetic electromagnetic environment, not real
recorded signals. This is Electronic Support - passive listening and scheduling. Nothing
here transmits, jams, or touches radio hardware.
            """
        )


def render_scorecard(
    environment,
    runs: dict[str, SimulationRun],
    config: Config,
    strategy_key: str = CANDIDATE_KEY,
) -> None:
    """Render the headline grades for the selected strategy and the activity model.

    The scheduler grade follows the strategy chosen in the sidebar, so switching to an
    ablation arm (``smart_ml_only``, ``smart_heuristic``) re-grades that arm rather than
    silently continuing to show the full system.
    """
    st.subheader("Scorecard")
    st.caption(
        "Two different things get graded. The **scheduler** is graded against the "
        "sequential sweep it is meant to beat, because the raw numbers are capped by the "
        "receiver's hardware and mean little on their own. The **activity model** is the "
        "machine-learning component inside it, graded on its own terms as a classifier."
    )

    receiver_cfg = config.section("receiver")
    ceiling = oracle_ceiling(
        environment,
        detection_probability=float(receiver_cfg.get("detection_probability", 0.95)),
        instantaneous_bandwidth=int(receiver_cfg.get("instantaneous_bandwidth", 1)),
        horizon=runs[strategy_key].n_timesteps,
    )
    if strategy_key == BASELINE_KEY:
        st.info(
            "The sequential sweep **is** the baseline, so grading it against itself is "
            "parity by definition. Pick another strategy to see a meaningful grade."
        )
    card = scheduler_scorecard(
        runs[BASELINE_KEY].metrics,
        runs[strategy_key].metrics,
        oracle=ceiling,
        candidate_name=strategy_key,
    )

    model_report = get_model_report(
        str(config.path_for("paths.results_dir") / "activity_model_test.json")
    )
    model_card = None
    if model_report:
        model_card = activity_model_scorecard(
            {
                **model_report.get("xgboost", {}),
                "positive_rate": model_report.get("positive_rate", float("nan")),
                "n_rows": model_report.get("n_rows", float("nan")),
            }
        )

    left, right = st.columns(2)
    with left:
        grade_badge(
            f"{strategy_key} scheduler (this environment)",
            card.grade,
            card.overall,
            "50 = level with the sequential sweep. 100 = twice as good.",
        )
    with right:
        if model_card:
            grade_badge(
                "Activity model (held-out test set)",
                model_card.grade,
                model_card.overall,
                "0 = no better than guessing. 100 = perfect.",
            )
        else:
            st.info(
                "No model report yet. Run `python training/evaluate_model.py --config "
                "config.yaml` to grade the activity model."
            )

    st.markdown("**What makes up the scheduler grade**")
    for component in card.components:
        columns = st.columns([2, 1, 5])
        columns[0].markdown(f"**{component.label}**")
        arrow = "▼ worse" if component.regression else "▲ better"
        colour = "#d1242f" if component.regression else "#1a7f37"
        columns[1].markdown(
            f"<span style='color:{colour};font-weight:600;'>{component.score:.0f}/100 "
            f"{arrow}</span>",
            unsafe_allow_html=True,
        )
        columns[2].caption(component.detail)
        st.progress(min(1.0, max(0.0, component.score / 100.0)))
        with st.expander(f"How {component.label} is calculated", expanded=False):
            st.markdown(
                f"- **Formula:** {component.formula}\n"
                f"- **Sequential sweep:** {component.baseline:.4g}\n"
                f"- **Smart Scan:** {component.candidate:.4g}\n"
                f"- Mapped with `50 x (1 + log2(ratio))`, so parity scores 50, twice as "
                f"good scores 100, half as good scores 0.\n"
                f"- **Weight in the overall grade:** "
                f"{card.to_record()['weights'][component.key]:.0%}"
            )

    if card.regressions:
        st.warning(
            f"**Where Smart Scan is worse than the plain sweep: "
            f"{', '.join(card.regressions)}.** This is a real trade, not a rounding error. "
            "An adaptive scheduler spends more time on bands it expects to be productive, "
            "so it cannot also guarantee the fixed-period full-spectrum coverage a "
            "sequential sweep gives for free."
        )

    fraction = card.context.get("fraction_of_ceiling", float("nan"))
    baseline_fraction = card.context.get("baseline_fraction_of_ceiling", float("nan"))
    if np.isfinite(fraction):
        columns = st.columns(3)
        columns[0].metric(
            "Best possible intercept rate",
            f"{ceiling['oracle_intercept_rate']:.3f}",
            help=(
                "The ceiling for any receiver of this bandwidth: it assumes the receiver "
                "is always already tuned to a transmitting band, with no dwell, retune or "
                "knowledge limits. No real scheduler can reach it - it exists to make the "
                "measured numbers interpretable."
            ),
        )
        columns[1].metric(
            f"{strategy_key} reaches",
            f"{fraction:.1%}",
            f"{(fraction - baseline_fraction) * 100:+.1f} pts vs sweep",
            help="Share of that ceiling the Smart Scan strategy actually achieved.",
        )
        columns[2].metric(
            "Sequential sweep reaches",
            f"{baseline_fraction:.1%}",
            help="Share of the same ceiling the open-loop baseline achieved.",
        )

    experiment = get_model_report(
        str(config.path_for("paths.results_dir") / "experiment_results.json")
    )
    overall_card = (experiment or {}).get("scorecards", {}).get("scheduler")
    if overall_card:
        same = strategy_key == CANDIDATE_KEY
        st.caption(
            "The grade above is for the single environment selected in the sidebar. "
            + (
                f"Across the whole test split, the last `run_mvp.py` scored `{CANDIDATE_KEY}` "
                f"**{overall_card['grade']}, {overall_card['overall']:.0f}/100**."
                if same
                else f"The whole-test-split figure recorded by `run_mvp.py` covers "
                f"`{CANDIDATE_KEY}` (**{overall_card['grade']}, "
                f"{overall_card['overall']:.0f}/100**), not `{strategy_key}`; compare "
                "ablation arms in `results/experiment_rows.csv`."
            )
        )


def render_glossary() -> None:
    """Render the glossary of terms and metrics."""
    with st.expander("Glossary: every term on this page", expanded=False):
        st.markdown(
            """
**Band** - one slice of the frequency range. The spectrum here is split into 32 bands,
and the receiver can listen to one at a time.

**Timestep** - one tick of the simulation clock (2 ms here). The receiver makes one
observation per timestep.

**Dwell** - how many consecutive timesteps the receiver stays on a band before deciding
where to go next.

**PDW (Pulse Descriptor Word)** - the record of one detected radar pulse: when it arrived,
its frequency, how long it lasted, what direction it came from, and how strong it was.

**PRI (Pulse Repetition Interval)** - the gap between a radar's pulses. Here it is
estimated from the gaps between *detections of a band*, which is what tells the scheduler
when that band is likely to be active again.

**Hit / intercept** - the receiver was tuned to a band that was genuinely transmitting,
and detected it.

**Miss** - the receiver listened and found nothing there.

**Occupancy** - the fraction of all band-timesteps in which some emitter was transmitting.
Higher means a busier, more crowded spectrum.

**Ground truth** - what genuinely happened in the environment. The receiver never sees
this; it is used only to score performance after the fact.

**Spatially scanning emitter** - a radar with a rotating antenna. A fixed receiver only
hears it while its beam sweeps past, so it appears in short, regular bursts.

**Frequency-agile emitter** - a radar that hops between frequencies, so its activity moves
between bands and its past behaviour is a weaker guide to its future.
            """
        )
        st.markdown("**The performance metrics**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "metric": metric,
                        "direction": "lower is better"
                        if metric in LOWER_IS_BETTER
                        else "higher is better",
                        "what it means": describe_metric(metric),
                    }
                    for metric in HEADLINE_METRICS
                ]
            ),
            width="stretch",
            hide_index=True,
        )


def main() -> None:
    """Render the dashboard."""
    st.set_page_config(page_title="Smart Scan EW Scheduler", layout="wide")
    config = get_config()

    st.title("Smart Scan Strategy for Electronic Warfare")
    st.caption(
        "Electronic Support (passive detection and scheduling) research simulation. "
        "No RF transmission, jamming or hardware control."
    )
    render_how_to_read()

    try:
        manifest = get_manifest(str(config.path_for("paths.manifest")))
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    with st.sidebar:
        st.header("Simulation controls")
        st.caption("Change anything here and the simulation re-runs from scratch.")
        split = st.selectbox(
            "Split",
            ["test", "validation", "train"],
            index=0,
            help=(
                "Which group of pulse trains to use. **test** is the held-out set the "
                "model never saw during training, so it gives the honest numbers."
            ),
        )
        entries = manifest.for_split(split)
        if not entries:
            st.error(f"No pulse trains in split {split}")
            st.stop()
        entry_names = [f"{entry.name}  ({entry.stratum})" for entry in entries]
        entry_index = st.selectbox(
            "Pulse train",
            range(len(entries)),
            format_func=lambda i: entry_names[i],
            help=(
                "One recorded electromagnetic environment: a stream of radar pulses from "
                "a set of emitters. The label in brackets is the emitter behaviour this "
                "train was chosen to represent."
            ),
        )
        entry = entries[entry_index]
        scenario = st.selectbox(
            "Scenario",
            ["spatial_scan", "frequency_agile", "mixed"],
            format_func=lambda key: SCENARIO_LABELS.get(key, key),
            help=(
                "Which emitters to keep. The two named scenarios are the ones the problem "
                "statement calls for; **mixed** keeps every emitter in the train."
            ),
        )
        horizon = st.slider(
            "Timesteps",
            200,
            int(config.get("environment.max_timesteps", 5000)),
            int(config.get("simulation.n_timesteps", 4000)),
            step=100,
            help="How long to run the simulation. Each timestep is one receiver observation.",
        )
        strategy_key = st.selectbox(
            "Strategy on display",
            [spec.key for spec in DEFAULT_STRATEGIES],
            index=[spec.key for spec in DEFAULT_STRATEGIES].index(CANDIDATE_KEY),
            help=(
                "**sequential** is the open-loop baseline sweep. **smart** is the full "
                "strategy. The `smart_heuristic` and `smart_ml_only` variants switch off "
                "the learned model and the online adaptation respectively, which is how "
                "their contribution is measured."
            ),
        )
        st.divider()
        st.caption(f"Data source: **{manifest.source}**")
        if manifest.source == "mock":
            st.caption(
                ":orange[Synthetic generator - the real Turing dataset is gated and needs "
                "an access token.]"
            )
        st.caption(f"Seed: **{config.seed}** (every run reproduces exactly from it)")
        st.caption(f"Model artifact: `{config.path_for('paths.model_artifact').name}`")

    payload = build_environment_cached(
        entry.path,
        entry.name,
        scenario,
        horizon,
        int(config.get("data.max_pulses_per_train", 400_000)),
    )
    environment = rehydrate(payload)
    if environment.n_timesteps == 0:
        st.error("This pulse train produced an empty environment.")
        st.stop()

    with st.spinner("Running schedulers..."):
        runs = {
            key: run_strategy(environment, key, horizon, seed_stream=entry_index)
            for key in {BASELINE_KEY, CANDIDATE_KEY, strategy_key}
        }
    run = runs[strategy_key]

    render_scorecard(environment, runs, config, strategy_key=strategy_key)
    st.divider()

    # --- Environment -------------------------------------------------------------
    st.subheader("Environment")
    st.caption(
        "The electromagnetic environment the receiver is searching. This is the ground "
        "truth - what the emitters genuinely did. The receiver never gets to see it."
    )
    summary = payload["summary"]
    columns = st.columns(5)
    columns[0].metric(
        "Bands", environment.n_bands, help="Frequency slices. The receiver hears one at a time."
    )
    columns[1].metric(
        "Timesteps", environment.n_timesteps, help="Length of the run in receiver observations."
    )
    columns[2].metric(
        "Occupancy",
        f"{environment.occupancy:.3f}",
        help=(
            "Fraction of all band-timesteps carrying a transmission. 0.28 means roughly "
            "28% of the spectrum-time is busy - a crowded environment."
        ),
    )
    columns[3].metric(
        "Emitters kept",
        f"{summary['emitters_kept']}/{summary['emitters_total']}",
        help=(
            "How many of the train's emitters this scenario keeps. Scenarios filter to one "
            "behaviour class plus static background clutter."
        ),
    )
    columns[4].metric(
        "Timestep",
        f"{environment.timestep_us / 1000:.1f} ms",
        help="Real time represented by one simulation timestep.",
    )
    if summary.get("fallback"):
        st.info(f"Scenario note: {summary['fallback']}")

    # --- Current receiver --------------------------------------------------------
    st.subheader("Current receiver")
    st.caption(
        "Drag the cursor to step through the run and watch what the receiver did at that "
        "moment."
    )
    timestep = st.slider(
        "Timestep cursor",
        0,
        run.n_timesteps - 1,
        min(200, run.n_timesteps - 1),
        help="Which moment of the run to inspect. The panels below follow this cursor.",
    )
    band = int(run.selected_band[timestep])
    columns = st.columns(5)
    columns[0].metric("Tuned band", band, help="The band the receiver was listening to.")
    columns[1].metric(
        "Centre frequency",
        f"{environment.band_centres_mhz[band]:,.0f} MHz",
        help="The middle of that band in real frequency units.",
    )
    columns[2].metric(
        "Predicted probability",
        "n/a"
        if not np.isfinite(run.predicted_probability[timestep])
        else f"{run.predicted_probability[timestep]:.3f}",
        help=(
            "How likely the scheduler thought this band was to be transmitting, before it "
            "listened. The open-loop sweep makes no prediction, so it shows n/a."
        ),
    )
    columns[3].metric(
        "Result",
        "HIT" if run.hits[timestep] else "miss",
        help="HIT means it was tuned to a genuinely transmitting band and detected it.",
    )
    columns[4].metric(
        "Pulses reported",
        int(run.signal_count[timestep]),
        help="How many radar pulses the receiver reported in that observation.",
    )

    # --- Spectrum ----------------------------------------------------------------
    st.subheader("Spectrum")
    st.caption(
        "Time runs left to right, frequency band bottom to top. **emitter active** points "
        "are transmissions that genuinely happened; **receiver hit** and **receiver miss** "
        "trace where the receiver actually pointed. A good strategy puts its path on top "
        "of the activity."
    )
    span = st.slider(
        "Window shown",
        100,
        min(1500, run.n_timesteps),
        min(400, run.n_timesteps),
        help="How many timesteps of the run to display around the cursor.",
    )
    start = max(0, min(timestep - span // 2, run.n_timesteps - span))
    frame = spectrum_frame(environment, run, (start, start + span))
    st.scatter_chart(frame, x="timestep", y="band", color="kind", height=430, size=18)
    st.caption(
        "Static, higher-resolution versions of this chart are written to `figures/` by "
        "`scripts/run_mvp.py`."
    )

    # --- Performance -------------------------------------------------------------
    st.subheader("Performance")
    st.caption(
        "The figures of merit the problem statement asks for, measured on this run. The "
        "third column explains what each one is actually telling you."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "metric": metric,
                    "value": run.metrics.get(metric, float("nan")),
                    "direction": "lower is better"
                    if metric in LOWER_IS_BETTER
                    else "higher is better",
                    "what it means": describe_metric(metric),
                }
                for metric in HEADLINE_METRICS
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    # --- Comparison --------------------------------------------------------------
    st.subheader("Comparison - Sequential sweep vs Smart Scan")
    st.caption(
        "Both strategies ran on this identical environment with identical receiver seeds, "
        "so the only thing that differs is where each chose to point. A positive "
        "improvement always means Smart Scan is better, whichever direction the raw metric "
        "runs in."
    )
    baseline_metrics = runs[BASELINE_KEY].metrics
    candidate_metrics = runs[CANDIDATE_KEY].metrics
    comparison = pd.DataFrame(
        [
            {
                "metric": metric,
                "sequential sweep": baseline_metrics.get(metric, float("nan")),
                "smart scan": candidate_metrics.get(metric, float("nan")),
                "improvement": relative_improvement(
                    baseline_metrics.get(metric, float("nan")),
                    candidate_metrics.get(metric, float("nan")),
                    metric,
                ),
                "verdict": "",
            }
            for metric in HEADLINE_METRICS
        ]
    )
    comparison["verdict"] = [
        "-"
        if not np.isfinite(value)
        else ("better" if value > 0.005 else ("worse" if value < -0.005 else "level"))
        for value in comparison["improvement"]
    ]
    st.dataframe(
        comparison,
        width="stretch",
        hide_index=True,
        column_config={
            "improvement": st.column_config.NumberColumn("improvement", format="%.1f%%")
        },
    )

    # --- Explainability ----------------------------------------------------------
    st.subheader("Explainability - why did it choose that band?")
    st.caption(
        "The scheduler scores every band, then tunes to the highest. Each term below is a "
        "weighted piece of that score; the weights come from `config.yaml`."
    )
    decisions = [d for d in run.decisions if d["timestep"] <= timestep]
    if not decisions:
        st.info(
            "This strategy records no score breakdown. The sequential sweep does not "
            "choose - it simply advances to the next band in fixed order, which is exactly "
            "the open-loop behaviour the problem statement criticises."
        )
    else:
        decision = decisions[-1]
        weights = config.get("scheduler.weights", {})
        breakdown = pd.DataFrame(
            [
                {
                    "component": "predicted probability",
                    "value": decision["combined_probability"],
                    "weight": weights.get("w1_predicted_probability"),
                },
                {
                    "component": "exploration bonus",
                    "value": decision["exploration_bonus"],
                    "weight": weights.get("w2_exploration_bonus"),
                },
                {
                    "component": "recency bonus",
                    "value": decision["recency_bonus"],
                    "weight": weights.get("w3_recency_bonus"),
                },
                {
                    "component": "periodicity bonus",
                    "value": decision["periodicity_bonus"],
                    "weight": weights.get("w4_periodicity_bonus"),
                },
                {
                    "component": "scan cost",
                    "value": -decision["scan_cost"],
                    "weight": weights.get("w5_scan_cost"),
                },
            ]
        )
        breakdown["contribution"] = breakdown["value"] * breakdown["weight"].astype(float)
        breakdown["what it means"] = [
            SCORE_TERM_HELP[name] for name in breakdown["component"]
        ]

        left, right = st.columns([3, 1])
        left.dataframe(breakdown, width="stretch", hide_index=True)
        right.metric("Chosen band", int(decision["band"]))
        right.metric(
            "Total score",
            f"{decision['score']:.3f}",
            help="Sum of the weighted contributions. The highest-scoring band wins.",
        )
        right.metric(
            "PRI estimate",
            f"{decision['pri_estimate_us'] / 1000:.1f} ms",
            help=(
                "How often this band has been found active, measured from the gaps between "
                "past detections. 0 means not enough detections yet to estimate."
            ),
        )
        right.metric(
            "Periodicity score",
            f"{decision['periodicity_score']:.3f}",
            help=(
                "How regular those gaps are, from 0 (irregular or unknown) to 1 (clockwork). "
                "A rotating radar scores high; a frequency-hopping emitter scores low."
            ),
        )

        biggest = breakdown.loc[breakdown["contribution"].idxmax()]
        st.info(
            f"**In plain terms:** band {int(decision['band'])} won mainly on "
            f"**{biggest['component']}** ({biggest['contribution']:.3f} of the "
            f"{decision['score']:.3f} total). {SCORE_TERM_HELP[biggest['component']]}"
        )

        st.markdown("**Decision timeline** - the first 25 choices of this run")
        st.dataframe(
            pd.DataFrame(decision_timeline_table(run, n_rows=25)),
            width="stretch",
            hide_index=True,
        )

    st.divider()
    render_glossary()
    source_note = (
        "the **real Turing Synthetic Radar Dataset**"
        if manifest.source == "huggingface"
        else "a **synthetic generator** (the real dataset was not reachable)"
    )
    st.caption(
        "Scheduler weights are experimental parameters. The pre-registered selection rule "
        "in `scripts/tune_weights.py` selected no configuration on the validation split, "
        "so the shipped weights were chosen post hoc under a weaker no-regression "
        f"criterion — see the README. Data source: {source_note}."
    )


if __name__ == "__main__":
    main()
