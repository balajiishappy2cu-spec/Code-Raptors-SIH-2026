"""The two scenarios the problem statement names.

* **Scenario 1 -- spatially scanning emitters.** Bearing/position varies over time. A
  fixed receiver sees such an emitter only while its mainbeam sweeps past, so the band
  goes active in short, strongly periodic bursts. This is the case the scheduler's PRI /
  periodicity machinery exists for.
* **Scenario 2 -- frequency-agile emitters.** Centre frequency hops between channels, so
  activity migrates across bands and a band's past occupancy is a weaker guide to its
  future. This is the case that stresses exploration and uncertainty handling.

A scenario environment keeps the emitters of its class and, by default, the static
emitters as background clutter -- a scanning emitter in an otherwise empty spectrum would
be an unrealistically easy target, and the resulting grid would be nearly empty. When a
pulse train cannot be classified per emitter (no labels), the whole train is used and the
fallback is recorded in the scenario metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from common.logging_utils import get_logger
from dataio.characterise import STRATA, emitter_behaviours
from dataio.manifest import DatasetManifest
from dataio.tdc_interface import PulseTrainRecord
from simulation.environment import EnvironmentGrid, build_environment

LOGGER = get_logger(__name__)

SCENARIOS: tuple[str, ...] = STRATA

#: Human-readable scenario names used in plots, the dashboard and the README.
SCENARIO_LABELS: dict[str, str] = {
    "spatial_scan": "Scenario 1 - spatially scanning emitters",
    "frequency_agile": "Scenario 2 - frequency-agile emitters",
    "mixed": "Mixed environment (all emitters)",
}


@dataclass
class Scenario:
    """One scenario instance: an environment plus how it was constructed.

    Attributes:
        name: scenario key (``spatial_scan``, ``frequency_agile`` or ``mixed``).
        environment: the ground truth grid.
        source_train: pulse train the environment came from.
        emitters_kept: number of emitters retained.
        emitters_total: number of emitters in the source train.
        fallback: reason the emitter filter could not be applied, if any.
        details: extra provenance for the run record.
    """

    name: str
    environment: EnvironmentGrid
    source_train: str
    emitters_kept: int
    emitters_total: int
    fallback: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Human-readable scenario label."""
        return SCENARIO_LABELS.get(self.name, self.name)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-friendly summary of the scenario."""
        return {
            "scenario": self.name,
            "label": self.label,
            "source_train": self.source_train,
            "emitters_kept": self.emitters_kept,
            "emitters_total": self.emitters_total,
            "fallback": self.fallback,
            "environment": self.environment.summary(),
            **self.details,
        }


def build_scenario(
    record: PulseTrainRecord,
    scenario: str,
    env_cfg: dict[str, Any],
    *,
    include_static: bool = True,
    max_timesteps: int | None = None,
) -> Scenario:
    """Build one scenario environment from a pulse train.

    Args:
        record: source pulse train.
        scenario: ``spatial_scan``, ``frequency_agile`` or ``mixed``.
        env_cfg: the ``environment`` configuration section.
        include_static: keep static emitters as background clutter.
        max_timesteps: optional truncation of the resulting grid.

    Returns:
        The constructed :class:`Scenario`.

    Raises:
        ValueError: for an unknown scenario name.
    """
    if scenario not in {*SCENARIOS, "mixed"}:
        msg = f"Unknown scenario {scenario!r}; expected one of {[*SCENARIOS, 'mixed']}"
        raise ValueError(msg)

    behaviours, source = emitter_behaviours(record)
    emitters_total = (
        int(np.unique(record.labels).size) if record.labels is not None else len(behaviours)
    )

    mask: np.ndarray | None = None
    fallback = ""
    if scenario == "mixed":
        emitters_kept = emitters_total
    elif record.labels is None or record.labels.size != record.n_pulses or not behaviours:
        fallback = (
            f"emitters could not be classified ({source}); using every emitter in the train"
        )
        emitters_kept = emitters_total
    else:
        wanted = {scenario} | ({"static"} if include_static else set())
        keep_labels = sorted({label for label, name in behaviours.items() if name in wanted})
        if not keep_labels:
            fallback = f"no {scenario} emitters in this train; using every emitter"
            emitters_kept = emitters_total
        else:
            mask = np.isin(np.asarray(record.labels), np.array(keep_labels))
            emitters_kept = len(keep_labels)

    name = f"{record.name}:{scenario}"
    environment = build_environment(record, env_cfg, name=name, pulse_mask=mask)
    if max_timesteps:
        environment = environment.truncate(max_timesteps)

    scenario_emitters = sum(1 for value in behaviours.values() if value == scenario)
    if fallback:
        LOGGER.info("Scenario %s on %s: %s", scenario, record.name, fallback)

    return Scenario(
        name=scenario,
        environment=environment,
        source_train=record.name,
        emitters_kept=emitters_kept,
        emitters_total=emitters_total,
        fallback=fallback,
        details={
            "behaviour_source": source,
            "scenario_emitters": scenario_emitters,
            "include_static_background": include_static,
        },
    )


def build_scenarios_for_split(
    manifest: DatasetManifest,
    split: str,
    env_cfg: dict[str, Any],
    scenarios: Iterable[str],
    *,
    max_pulses: int = 0,
    max_timesteps: int | None = None,
    include_static: bool = True,
    max_trains: int | None = None,
) -> dict[str, list[Scenario]]:
    """Build every scenario environment for one manifest split.

    Args:
        manifest: dataset manifest produced by the sampler.
        split: split name.
        env_cfg: the ``environment`` configuration section.
        scenarios: scenario names to build.
        max_pulses: contiguous-window cap per pulse train.
        max_timesteps: optional truncation of each grid.
        include_static: keep static emitters as background clutter.
        max_trains: limit on how many pulse trains to use from the split.

    Returns:
        Mapping from scenario name to the list of built scenarios.
    """
    wanted = list(scenarios)
    out: dict[str, list[Scenario]] = {name: [] for name in wanted}
    entries = manifest.for_split(split)
    if max_trains is not None:
        entries = entries[:max_trains]

    for entry in entries:
        try:
            record = entry.load(max_pulses=max_pulses)
        except (FileNotFoundError, ValueError) as exc:
            LOGGER.warning("Skipping %s: %s", entry.path, exc)
            continue
        for name in wanted:
            try:
                out[name].append(
                    build_scenario(
                        record,
                        name,
                        env_cfg,
                        include_static=include_static,
                        max_timesteps=max_timesteps,
                    )
                )
            except ValueError as exc:
                LOGGER.warning("Scenario %s failed on %s: %s", name, entry.name, exc)
    return out


def pick_demo_scenario(scenarios: list[Scenario]) -> tuple[int, Scenario] | tuple[None, None]:
    """Pick the most illustrative scenario instance for a plot or the dashboard.

    Not the first one, and not the median either. A frequency-time map only tells a story
    when several bands are genuinely active: with one active band both strategies draw a
    single stripe and the figure says nothing, while with almost every band busy the
    receiver cannot help hitting something and the comparison looks trivial.

    So this scores each environment on how close it is to a readable number of active bands
    and prefers mid-range occupancy as a tie-break.

    Args:
        scenarios: candidate scenario instances.

    Returns:
        ``(index, scenario)`` of the best demonstration, or ``(None, None)`` if there are
        no usable environments.
    """
    usable = [(i, s) for i, s in enumerate(scenarios) if s.environment.n_timesteps > 0]
    if not usable:
        return None, None

    #: Active-band count that reads best on a 32-band frequency-time map.
    target_active_bands = 8.0

    def score(item: tuple[int, Scenario]) -> float:
        environment = item[1].environment
        active_bands = float((environment.active.sum(axis=0) > 0).sum())
        occupancy = environment.occupancy
        # Distance from a readable band count, plus a mild penalty for extreme occupancy.
        band_penalty = abs(active_bands - target_active_bands) / max(1.0, target_active_bands)
        occupancy_penalty = abs(occupancy - 0.25)
        return band_penalty + occupancy_penalty

    best = min(usable, key=score)
    return best[0], best[1]
