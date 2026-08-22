"""Electronic Support receiver model.

A passive, narrow-instantaneous-bandwidth receiver sweeping a wide surveillance band:
at each timestep it observes only the band(s) it has tuned to, and it learns about the
environment solely through its own observations. It never reads ground truth -- that
separation is what makes the "absence of prior reliable intelligence" framing of the
problem statement meaningful.

This is a simulation of a receiver. It performs no RF transmission or reception.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from common.logging_utils import get_logger
from simulation.environment import EnvironmentGrid

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class Observation:
    """What the receiver reports after dwelling on a band for one timestep.

    Attributes:
        timestep: simulation timestep of the observation.
        selected_band: lowest band index in the receiver's instantaneous bandwidth.
        detected: whether the receiver declared a detection.
        signal_count: number of pulses reported (0 when nothing was declared).
        bands: every band index covered by the instantaneous bandwidth.
        settling: ``True`` when the receiver was retuning and could not observe.
    """

    timestep: int
    selected_band: int
    detected: bool
    signal_count: int
    bands: tuple[int, ...] = ()
    settling: bool = False


@dataclass
class ReceiverConfig:
    """Receiver parameters, all sourced from ``config.yaml``.

    Attributes:
        instantaneous_bandwidth: number of contiguous bands observable per timestep.
        dwell_timesteps: timesteps spent on a band before the scheduler re-decides.
        detection_probability: probability of declaring a detection on an active band.
        false_alarm_probability: probability of declaring a detection on an idle band.
        retune_cost_timesteps: blind settling timesteps incurred when changing band.
    """

    instantaneous_bandwidth: int = 1
    dwell_timesteps: int = 1
    detection_probability: float = 0.95
    false_alarm_probability: float = 0.01
    retune_cost_timesteps: int = 0

    @classmethod
    def from_config(cls, receiver_cfg: dict[str, Any]) -> "ReceiverConfig":
        """Build a receiver configuration from the ``receiver`` config section."""
        return cls(
            instantaneous_bandwidth=int(receiver_cfg.get("instantaneous_bandwidth", 1)),
            dwell_timesteps=max(1, int(receiver_cfg.get("dwell_timesteps", 1))),
            detection_probability=float(receiver_cfg.get("detection_probability", 0.95)),
            false_alarm_probability=float(receiver_cfg.get("false_alarm_probability", 0.01)),
            retune_cost_timesteps=int(receiver_cfg.get("retune_cost_timesteps", 0)),
        )


@dataclass
class Receiver:
    """Simulated Electronic Support receiver with a narrow instantaneous bandwidth.

    Attributes:
        config: receiver parameters.
        rng: seeded generator driving detection and false-alarm draws.
        current_band: band the receiver is currently tuned to (``-1`` before first tune).
        dwell_remaining: timesteps left in the current dwell.
        settling_remaining: timesteps left before the receiver is usable after retuning.
    """

    config: ReceiverConfig
    rng: np.random.Generator
    current_band: int = -1
    dwell_remaining: int = 0
    settling_remaining: int = 0
    _visits: int = field(default=0, init=False, repr=False)
    _retunes: int = field(default=0, init=False, repr=False)

    @property
    def needs_decision(self) -> bool:
        """Whether the scheduler must choose a band before the next observation."""
        return self.dwell_remaining <= 0

    @property
    def scan_time_timesteps(self) -> int:
        """Timesteps a full open-loop sweep of ``n_bands`` bands would take.

        Depends only on the receiver, so callers pass the band count in.
        """
        return self.config.dwell_timesteps

    def bands_for(self, band: int, n_bands: int) -> tuple[int, ...]:
        """Return every band covered when tuned to ``band``."""
        width = max(1, self.config.instantaneous_bandwidth)
        return tuple(b for b in range(band, min(band + width, n_bands)))

    def tune(self, band: int) -> None:
        """Tune to a band, starting a new dwell and paying any settling cost."""
        if band != self.current_band:
            self._retunes += 1
            self.settling_remaining = max(0, self.config.retune_cost_timesteps)
        self.current_band = int(band)
        self.dwell_remaining = self.config.dwell_timesteps

    def observe(self, environment: EnvironmentGrid, timestep: int) -> Observation:
        """Observe the currently tuned band for one timestep.

        Args:
            environment: ground truth grid; only the tuned cells are consulted, which
                is the simulation's stand-in for physically receiving energy.
            timestep: current simulation timestep.

        Returns:
            The receiver's :class:`Observation`.

        Raises:
            RuntimeError: if called before the receiver has been tuned.
        """
        if self.current_band < 0:
            msg = "Receiver.observe called before Receiver.tune"
            raise RuntimeError(msg)

        bands = self.bands_for(self.current_band, environment.n_bands)
        self.dwell_remaining -= 1

        if self.settling_remaining > 0:
            self.settling_remaining -= 1
            return Observation(
                timestep=timestep,
                selected_band=self.current_band,
                detected=False,
                signal_count=0,
                bands=bands,
                settling=True,
            )

        self._visits += 1
        truth_active = bool(environment.active[timestep, list(bands)].any())
        pulse_count = int(environment.n_pulses[timestep, list(bands)].sum())

        if truth_active:
            detected = bool(self.rng.random() < self.config.detection_probability)
            signal_count = pulse_count if detected else 0
        else:
            detected = bool(self.rng.random() < self.config.false_alarm_probability)
            # A false alarm reports a small spurious pulse count.
            signal_count = int(self.rng.integers(1, 3)) if detected else 0

        return Observation(
            timestep=timestep,
            selected_band=self.current_band,
            detected=detected,
            signal_count=signal_count,
            bands=bands,
            settling=False,
        )

    def stats(self) -> dict[str, Any]:
        """Return receiver-level counters for the run record."""
        return {
            "visits": self._visits,
            "retunes": self._retunes,
            "instantaneous_bandwidth": self.config.instantaneous_bandwidth,
            "dwell_timesteps": self.config.dwell_timesteps,
            "detection_probability": self.config.detection_probability,
            "false_alarm_probability": self.config.false_alarm_probability,
        }


def make_receiver(receiver_cfg: dict[str, Any], rng: np.random.Generator) -> Receiver:
    """Construct a :class:`Receiver` from the ``receiver`` configuration section."""
    return Receiver(config=ReceiverConfig.from_config(receiver_cfg), rng=rng)
