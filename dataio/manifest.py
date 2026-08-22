"""Dataset manifest: the reproducible record of which pulse trains were sampled.

The sampler writes one manifest; every later stage (feature preparation, training,
experiments, dashboard) reads it instead of touching the raw data directory. That is what
makes an experiment reproducible from a single seeded configuration: the manifest pins
the exact files, splits and strata that were used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from common.config import REPO_ROOT
from common.io_utils import read_json, utc_timestamp, write_json
from common.logging_utils import get_logger
from dataio.tdc_interface import PulseTrainRecord, load_pulse_train

LOGGER = get_logger(__name__)

SPLITS: tuple[str, ...] = ("train", "validation", "test")


@dataclass
class ManifestEntry:
    """One selected pulse train.

    Attributes:
        name: pulse train identifier (file stem).
        path: local path of the cached pulse train.
        split: ``train``, ``validation`` or ``test``.
        stratum: stratum the train was selected to represent.
        characterisation: the Pass 1 summary for this train.
    """

    name: str
    path: str
    split: str
    stratum: str
    characterisation: dict[str, Any] = field(default_factory=dict)

    def resolved_path(self) -> Path:
        """Return the pulse train path, resolving relative entries against the repo.

        Manifests written by the sampler store absolute paths, which is fine on the
        machine that produced them and useless anywhere else. A deployed copy stores paths
        relative to the repository root instead, so the same manifest works on a checkout
        with a different layout.
        """
        candidate = Path(self.path)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        relative = REPO_ROOT / self.path
        if relative.exists():
            return relative
        return candidate

    def load(self, max_pulses: int = 0) -> PulseTrainRecord:
        """Load this pulse train, optionally limited to a contiguous window."""
        record = load_pulse_train(self.resolved_path())
        return record.contiguous_window(max_pulses) if max_pulses else record


@dataclass
class DatasetManifest:
    """The full sampling record.

    Attributes:
        seed: seed used for the sampling draw.
        mode: receiver mode of the source data (``stare`` for this MVP).
        source: ``huggingface`` or ``mock``.
        created: UTC timestamp of the sampling run.
        entries: selected pulse trains.
        pass1: the Pass 1 characterisation table.
        notes: free-form provenance notes, including why a source was chosen.
    """

    seed: int
    mode: str
    source: str
    created: str = field(default_factory=utc_timestamp)
    entries: list[ManifestEntry] = field(default_factory=list)
    pass1: list[dict[str, Any]] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    def for_split(self, split: str) -> list[ManifestEntry]:
        """Return the entries belonging to one split."""
        return [entry for entry in self.entries if entry.split == split]

    def strata(self) -> dict[str, int]:
        """Return the number of selected trains per stratum."""
        counts: dict[str, int] = {}
        for entry in self.entries:
            counts[entry.stratum] = counts.get(entry.stratum, 0) + 1
        return counts

    def load_split(self, split: str, max_pulses: int = 0) -> Iterator[PulseTrainRecord]:
        """Yield the pulse trains of one split.

        Args:
            split: split name.
            max_pulses: contiguous-window cap per train (0 means no cap).
        """
        for entry in self.for_split(split):
            try:
                yield entry.load(max_pulses=max_pulses)
            except (FileNotFoundError, ValueError) as exc:
                LOGGER.warning("Skipping %s: %s", entry.path, exc)

    def save(self, path: str | Path) -> Path:
        """Write the manifest to JSON."""
        payload = {
            "seed": self.seed,
            "mode": self.mode,
            "source": self.source,
            "created": self.created,
            "notes": self.notes,
            "strata": self.strata(),
            "counts": {split: len(self.for_split(split)) for split in SPLITS},
            "entries": [asdict(entry) for entry in self.entries],
            "pass1": self.pass1,
        }
        out = write_json(path, payload)
        LOGGER.info(
            "Wrote manifest %s (%s source, %d trains: %s)",
            out,
            self.source,
            len(self.entries),
            {split: len(self.for_split(split)) for split in SPLITS},
        )
        return out

    @classmethod
    def load(cls, path: str | Path) -> "DatasetManifest":
        """Read a manifest written by :meth:`save`.

        Raises:
            FileNotFoundError: if the manifest does not exist. Run
                ``scripts/sample_dataset.py`` first.
        """
        manifest_path = Path(path)
        if not manifest_path.exists():
            msg = (
                f"Manifest not found: {manifest_path}. "
                "Run 'python scripts/sample_dataset.py --config config.yaml' first."
            )
            raise FileNotFoundError(msg)
        payload = read_json(manifest_path)
        return cls(
            seed=int(payload.get("seed", 42)),
            mode=str(payload.get("mode", "stare")),
            source=str(payload.get("source", "mock")),
            created=str(payload.get("created", "")),
            entries=[ManifestEntry(**entry) for entry in payload.get("entries", [])],
            pass1=list(payload.get("pass1", [])),
            notes=dict(payload.get("notes", {})),
        )
