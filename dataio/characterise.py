"""Characterise pulse trains so the sampler can stratify on real behaviour.

Pass 1 of the sampler inspects a broad random sample of pulse trains; this module turns
each one into the small set of numbers Pass 2 stratifies on. Two paths are supported:

* **Metadata path** -- when the file's ``metadata.transmitters`` entries describe the
  emitters (as the TSRD metadata does, and as the mock generator reproduces), the
  emitter mix is read straight from them.
* **Signal path** -- otherwise the same properties are derived from the PDWs themselves:
  an emitter whose centre frequency takes several distinct values is frequency agile, and
  an emitter whose bearing sweeps while its emission is bursty is spatially scanning.

The signal path never needs metadata and works on any TSRD file. Emitter labels are used
here only when present, and only to characterise the *environment* offline; the receiver
and scheduler never see them.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from common.logging_utils import get_logger
from dataio.tdc_interface import PulseTrainRecord

LOGGER = get_logger(__name__)

#: Strata the sampler can select on, named after the problem statement's scenarios.
STRATA: tuple[str, ...] = ("spatial_scan", "frequency_agile")

# Signal-path thresholds. Deliberately loose: they only steer stratification.
_AGILE_CF_SPREAD_MHZ = 50.0
_AGILE_MIN_DISTINCT_CHANNELS = 2
_SCAN_AOA_SPREAD_DEG = 5.0
_SCAN_MAX_DUTY_CYCLE = 0.5


def _transmitter_entries(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Return transmitter metadata entries, whatever container shape they arrive in.

    The official saver writes a list of nested dataclasses as a group of
    ``transmitters_<i>`` subgroups, so a round-tripped file yields a dict of dicts while
    an in-memory record yields a list of dicts. Both are accepted.

    The numeric suffix of each ``transmitters_<i>`` key is the emitter's label, so it is
    parsed out and stored as ``emitter_id``. Position in the container cannot be used:
    HDF5 key order is alphabetical, which puts ``transmitters_10`` before
    ``transmitters_2``, and the real dataset also contains transmitters that emit no
    captured pulses, so the labels present are a subset of the transmitter indices.
    """
    raw = metadata.get("transmitters")
    if raw is None:
        return []
    if isinstance(raw, list):
        entries = []
        for index, entry in enumerate(raw):
            if isinstance(entry, dict):
                entries.append({**entry, "emitter_id": entry.get("emitter_id", index)})
        return entries
    if isinstance(raw, dict):
        entries: list[tuple[int, dict[str, Any]]] = []
        for key, value in raw.items():
            if not isinstance(value, dict):
                continue
            match = re.search(r"(\d+)\s*$", str(key))
            if match is None:
                continue
            emitter_id = int(match.group(1))
            entries.append((emitter_id, {**value, "emitter_id": emitter_id}))
        if entries:
            entries.sort(key=lambda item: item[0])
            return [entry for _, entry in entries]
        return [raw]
    return []


#: ``freq_mode`` values in the real TSRD that mean the emitter changes frequency over
#: time. Observed values are FixedSingle, FixedMultiSimultaneous, HoppingLinear,
#: HoppingSawtooth, RandomRange and RandomFixed; the Fixed* ones are not time-agile
#: (FixedMultiSimultaneous transmits several frequencies at once but does not hop).
_AGILE_FREQ_MODE_PREFIXES = ("hopping", "random", "agile")

#: ``scan_type`` values that mean the antenna sweeps. Observed values are Circular and
#: Omni; an omnidirectional emitter illuminates continuously and does not scan.
_NON_SCANNING_TYPES = {"omni", "none", "steady", "fixed", "stare"}


def _nested(entry: dict[str, Any], group: str) -> dict[str, Any]:
    """Return a nested config group from a transmitter entry, or an empty dict."""
    value = entry.get(group)
    return value if isinstance(value, dict) else {}


def _behaviour_from_metadata(entry: dict[str, Any]) -> str | None:
    """Infer one emitter's behaviour class from its transmitter metadata entry.

    Handles both the flat form written by this project's generator and the nested
    ``frequency_config`` / ``scan_config`` / ``position_config`` groups the real TSRD
    uses. Frequency agility is checked first: an emitter that both hops and scans is
    more usefully treated as the harder frequency-agile case, since a band-level history
    tracks a rotating beam far better than it tracks a frequency hop.
    """
    behaviour = entry.get("behaviour") or entry.get("behavior")
    if isinstance(behaviour, str) and behaviour in {"static", *STRATA}:
        return behaviour
    if bool(entry.get("frequency_agile", False)):
        return "frequency_agile"
    if bool(entry.get("spatial_scan", False)):
        return "spatial_scan"

    # --- Real TSRD nested configuration -------------------------------------------
    frequency = _nested(entry, "frequency_config")
    freq_mode = str(frequency.get("freq_mode", "")).strip().lower()
    if freq_mode.startswith(_AGILE_FREQ_MODE_PREFIXES):
        return "frequency_agile"

    scan = _nested(entry, "scan_config")
    scan_type = str(scan.get("scan_type", "")).strip().lower()
    scan_rate = float(scan.get("scan_rate_rpm", 0.0) or 0.0)
    if scan_type and scan_type not in _NON_SCANNING_TYPES and scan_rate > 0.0:
        return "spatial_scan"

    # --- Flat fields this project's generator writes -------------------------------
    if float(entry.get("hop_period_us", 0.0) or 0.0) > 0.0:
        return "frequency_agile"
    for key in ("rotation_period_us", "aoa_rate_deg_per_s", "scan_rate_deg_per_s"):
        if float(entry.get(key, 0.0) or 0.0) > 0.0:
            return "spatial_scan"

    if freq_mode or scan_type:
        # The configuration was readable and says the emitter neither hops nor scans.
        return "static"
    return None


def emitter_behaviours_from_pdws(record: PulseTrainRecord) -> dict[int, str]:
    """Classify each labelled emitter from its own PDWs.

    Args:
        record: pulse train with per-train-local labels.

    Returns:
        Mapping from emitter label to ``static``, ``spatial_scan`` or ``frequency_agile``.
        Empty when the record has no labels.
    """
    if record.labels is None or record.labels.size != record.n_pulses:
        return {}

    toa = record.column("toa")
    cf = record.column("cf")
    aoa = record.column("aoa") if record.has("aoa") else np.zeros_like(toa)
    labels = np.asarray(record.labels)
    span = float(toa.max() - toa.min()) if toa.size else 0.0

    behaviours: dict[int, str] = {}
    for label in np.unique(labels):
        mask = labels == label
        if int(mask.sum()) < 8:
            continue
        emitter_cf = cf[mask]
        emitter_aoa = aoa[mask]
        emitter_toa = np.sort(toa[mask])

        # Frequency agility: several distinct centre-frequency channels.
        channels = np.unique(np.round(emitter_cf / _AGILE_CF_SPREAD_MHZ))
        is_agile = channels.size >= _AGILE_MIN_DISTINCT_CHANNELS

        # Spatial scan: bearing sweeps and emission is bursty rather than continuous.
        aoa_spread = float(emitter_aoa.max() - emitter_aoa.min())
        duty = 1.0
        if span > 0 and emitter_toa.size > 2:
            gaps = np.diff(emitter_toa)
            median_gap = float(np.median(gaps))
            if median_gap > 0:
                # Fraction of the window covered by intervals close to the median gap.
                duty = float(np.sum(gaps[gaps <= 3.0 * median_gap]) / span)
        is_scanning = aoa_spread > _SCAN_AOA_SPREAD_DEG and duty < _SCAN_MAX_DUTY_CYCLE

        if is_agile:
            behaviours[int(label)] = "frequency_agile"
        elif is_scanning:
            behaviours[int(label)] = "spatial_scan"
        else:
            behaviours[int(label)] = "static"
    return behaviours


def emitter_behaviours(record: PulseTrainRecord) -> tuple[dict[int, str], str]:
    """Return per-emitter behaviours and the path used to derive them.

    Args:
        record: the pulse train to classify.

    Returns:
        ``(behaviours, source)`` where ``source`` is ``"metadata"``, ``"pdw"`` or
        ``"none"``.
    """
    entries = _transmitter_entries(record.metadata)
    if entries:
        behaviours: dict[int, str] = {}
        for index, entry in enumerate(entries):
            behaviour = _behaviour_from_metadata(entry)
            if behaviour is None:
                continue
            emitter_id = entry.get("emitter_id", entry.get("id", index))
            try:
                behaviours[int(emitter_id)] = behaviour
            except (TypeError, ValueError):  # pragma: no cover - defensive
                behaviours[index] = behaviour
        if behaviours:
            return behaviours, "metadata"

    behaviours = emitter_behaviours_from_pdws(record)
    return (behaviours, "pdw") if behaviours else ({}, "none")


def characterise(record: PulseTrainRecord, *, max_pulses: int = 200_000) -> dict[str, Any]:
    """Summarise one pulse train for stratification.

    Args:
        record: the pulse train to characterise.
        max_pulses: cap on the pulses inspected, keeping Pass 1 cheap on large files.

    Returns:
        Dictionary of summary statistics, per-behaviour emitter counts, the dominant
        stratum and the classification path used.
    """
    sample = record.contiguous_window(max_pulses)
    toa = sample.column("toa")
    cf = sample.column("cf")
    aoa = sample.column("aoa") if sample.has("aoa") else np.zeros_like(toa)
    behaviours, source = emitter_behaviours(sample)

    counts = {name: 0 for name in ("static", *STRATA)}
    for behaviour in behaviours.values():
        counts[behaviour] = counts.get(behaviour, 0) + 1
    n_classified = sum(counts.values())

    scenario_counts = {name: counts.get(name, 0) for name in STRATA}
    dominant = max(scenario_counts, key=lambda key: scenario_counts[key]) if n_classified else ""
    if dominant and scenario_counts[dominant] == 0:
        dominant = ""

    n_emitters = (
        int(np.unique(sample.labels).size) if sample.labels is not None else len(behaviours)
    )
    return {
        "name": record.name,
        "path": str(record.source_path) if record.source_path else None,
        "n_pulses": int(record.n_pulses),
        "n_pulses_inspected": int(sample.n_pulses),
        "n_emitters": n_emitters,
        "toa_span_us": float(toa.max() - toa.min()) if toa.size else 0.0,
        "cf_min_mhz": float(cf.min()) if cf.size else 0.0,
        "cf_max_mhz": float(cf.max()) if cf.size else 0.0,
        "aoa_span_deg": float(aoa.max() - aoa.min()) if aoa.size else 0.0,
        "mean_pw_us": float(sample.column("pw").mean()) if sample.has("pw") else float("nan"),
        "behaviour_counts": counts,
        "n_spatial_scan": scenario_counts.get("spatial_scan", 0),
        "n_frequency_agile": scenario_counts.get("frequency_agile", 0),
        "frac_spatial_scan": float(scenario_counts.get("spatial_scan", 0) / n_classified)
        if n_classified
        else 0.0,
        "frac_frequency_agile": float(scenario_counts.get("frequency_agile", 0) / n_classified)
        if n_classified
        else 0.0,
        "dominant_stratum": dominant,
        "behaviour_source": source,
        "description": str(record.metadata.get("description", "")),
    }


def pulse_mask_for_behaviour(
    record: PulseTrainRecord,
    behaviour: str,
    *,
    include_static: bool = False,
) -> np.ndarray | None:
    """Return a pulse mask selecting emitters of one behaviour class.

    Args:
        record: the pulse train to filter.
        behaviour: ``spatial_scan`` or ``frequency_agile``.
        include_static: also keep static emitters as background clutter.

    Returns:
        A boolean mask over the record's pulses, or ``None`` when the emitters cannot be
        classified (no labels), in which case the caller should use the whole train.
    """
    if record.labels is None or record.labels.size != record.n_pulses:
        return None
    behaviours, _ = emitter_behaviours(record)
    if not behaviours:
        return None
    wanted = {behaviour} | ({"static"} if include_static else set())
    keep_labels = {label for label, name in behaviours.items() if name in wanted}
    if not keep_labels:
        return None
    labels = np.asarray(record.labels)
    return np.isin(labels, np.array(sorted(keep_labels)))
