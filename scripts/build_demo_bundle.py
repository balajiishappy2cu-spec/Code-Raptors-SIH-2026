"""Build the small, self-contained data bundle a public deployment can ship.

Streamlit Community Cloud clones the repository and runs it on about 1 GB of RAM, so the
working dataset cannot travel with the app: the raw downloads are ~4.6 GB, the processed
cache ~863 MB, and the TSRD is gated anyway, so the deployed app cannot fetch it at run
time either.

This writes a demo bundle instead -- a handful of pulse trains, trimmed to a short
contiguous window each, plus a manifest whose paths are **relative to the repository
root** so the same files resolve on any checkout. Everything else the console needs
(the model artifact and the recorded results) is already small enough to commit.

The bundle is a demonstration subset, not an experiment: the numbers the README quotes
come from the full 55-train run and are read from ``results/``, which ships as recorded.

Usage::

    python scripts/build_demo_bundle.py --config config.yaml
    python scripts/build_demo_bundle.py --trains-per-split 3 --max-pulses 60000
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import REPO_ROOT, Config  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402
from dataio.manifest import SPLITS, DatasetManifest, ManifestEntry  # noqa: E402

LOGGER = get_logger("scripts.build_demo_bundle")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument(
        "--trains-per-split", type=int, default=4, help="pulse trains to ship per split"
    )
    parser.add_argument(
        "--max-pulses", type=int, default=60_000, help="contiguous window kept per train"
    )
    parser.add_argument("--out", default="data/demo", help="bundle directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Write the demo bundle and its portable manifest."""
    args = parse_args(argv)
    config = Config.load(args.config)
    manifest = DatasetManifest.load(config.path_for("paths.manifest"))

    out_dir = REPO_ROOT / args.out
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: list[ManifestEntry] = []
    total_bytes = 0

    for split in SPLITS:
        candidates = manifest.for_split(split)
        if not candidates:
            continue
        # Prefer the busiest trains: an almost-empty environment makes a poor demo, which
        # is the same reason the figure picker does not simply take the first one.
        ranked = sorted(
            candidates,
            key=lambda entry: int(entry.characterisation.get("n_emitters") or 0),
            reverse=True,
        )
        chosen = ranked[: max(1, args.trains_per_split)]
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for entry in chosen:
            try:
                record = entry.load(max_pulses=args.max_pulses)
            except (FileNotFoundError, ValueError) as exc:
                LOGGER.warning("Skipping %s: %s", entry.name, exc)
                continue
            destination = split_dir / f"{entry.name}.h5"
            record.save(destination)
            size = destination.stat().st_size
            total_bytes += size
            entries.append(
                ManifestEntry(
                    name=entry.name,
                    # Stored relative to the repository root so the bundle is portable.
                    path=str(destination.relative_to(REPO_ROOT)).replace("\\", "/"),
                    split=split,
                    stratum=entry.stratum,
                    characterisation=entry.characterisation,
                )
            )
            LOGGER.info(
                "%-11s %-12s %7d pulses  %5.2f MB",
                split,
                entry.name,
                record.n_pulses,
                size / 1e6,
            )

    if not entries:
        LOGGER.error("No pulse trains could be bundled")
        return 1

    bundle = DatasetManifest(
        seed=manifest.seed,
        mode=manifest.mode,
        source=manifest.source,
        entries=entries,
        pass1=[],
        notes={
            **manifest.notes,
            "bundle": (
                "Demonstration subset for public deployment: a few pulse trains trimmed to "
                f"{args.max_pulses} pulses each, with repository-relative paths. The "
                "published metrics come from the full run recorded in results/, not from "
                "this subset."
            ),
            "trains_per_split": args.trains_per_split,
            "max_pulses_per_train": args.max_pulses,
        },
    )
    bundle.save(out_dir / "manifest.json")
    LOGGER.info(
        "Bundle complete: %d pulse trains, %.1f MB total", len(entries), total_bytes / 1e6
    )
    LOGGER.info("Point a deployment at it with paths.manifest: %s/manifest.json", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
