"""Two-pass sampler for the Turing Synthetic Radar Dataset.

Pass 1 -- cheap, broad, random. A random sample of pulse train files is fetched and each
one's *real* metadata, centre-frequency range, emitter count and motion pattern is
inspected. Random selection is right here: there is no way to know a file's content
before fetching it, and it avoids whatever ordering bias the file listing has.

Pass 2 -- deliberate, stratified. From Pass 1's measured properties, whole intact pulse
trains are selected to cover the two scenarios the problem statement names: spatially
scanning emitters and frequency-agile emitters. Trains are kept whole and in ToA order;
nothing is flattened or shuffled across trains, because every downstream feature (PRI,
periodicity, the environment timeline itself) needs an intact temporal sequence.

Usage::

    python scripts/sample_dataset.py --config config.yaml
    python scripts/sample_dataset.py --mode stare --train-trains 15 \
        --val-trains 5 --test-trains 5 --stratify-on spatial_scan,frequency_agile --seed 42
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import Config, make_rng  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402
from dataio import hf_source  # noqa: E402
from dataio.characterise import STRATA, characterise  # noqa: E402
from dataio.manifest import SPLITS, DatasetManifest, ManifestEntry  # noqa: E402
from dataio.mock_generator import generate_mock_dataset  # noqa: E402
from dataio.tdc_interface import load_pulse_train  # noqa: E402

LOGGER = get_logger("scripts.sample_dataset")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--mode", default=None, choices=["stare", "scan"], help="receiver mode")
    parser.add_argument("--train-trains", type=int, default=None, help="pulse trains for train")
    parser.add_argument("--val-trains", type=int, default=None, help="pulse trains for validation")
    parser.add_argument("--test-trains", type=int, default=None, help="pulse trains for test")
    parser.add_argument(
        "--stratify-on",
        default=None,
        help="comma-separated strata, e.g. spatial_scan,frequency_agile",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--source",
        default=None,
        choices=["auto", "huggingface", "mock"],
        help="where pulse trains come from",
    )
    parser.add_argument(
        "--pass1-files", type=int, default=None, help="files inspected in Pass 1"
    )
    parser.add_argument("--force", action="store_true", help="re-cache trains that exist")
    return parser.parse_args(argv)


def resolve_source(config: Config, requested: str | None) -> tuple[str, str]:
    """Decide whether to use the real dataset or the mock generator.

    Args:
        config: loaded configuration.
        requested: source override from the command line.

    Returns:
        ``(source, reason)`` where source is ``huggingface`` or ``mock``.
    """
    source = (requested or config.get("data.source", "auto")).lower()
    repo_id = str(config.get("data.hf_repo_id"))
    mode = str(config.get("data.mode", "stare"))

    if source == "mock":
        return "mock", "mock source requested"
    if source == "huggingface":
        available, reason = hf_source.probe_availability(repo_id, mode)
        if not available:
            msg = f"Hugging Face source requested but unavailable: {reason}"
            raise SystemExit(msg)
        return "huggingface", reason

    available, reason = hf_source.probe_availability(repo_id, mode)
    if available:
        return "huggingface", reason
    LOGGER.warning("Real TSRD unavailable (%s)", reason)
    LOGGER.warning("Falling back to the TSRD-format mock generator; see README Limitations")
    return "mock", reason


def pass1_allocation(
    counts: dict[str, int], budget: int, headroom: float
) -> dict[str, int]:
    """Decide how many files Pass 1 inspects per split.

    Each split must end up with at least as many candidates as trains it needs, plus
    headroom so Pass 2 has something to stratify over. Splitting a flat budget evenly
    would starve the train split, which usually needs the most pulse trains.

    Args:
        counts: pulse trains wanted per split.
        budget: total Pass 1 file budget; raised if it cannot cover the requirements.
        headroom: multiplier applied to each split's requirement.

    Returns:
        Files to inspect per split.
    """
    floors = {split: int(counts.get(split, 0)) for split in SPLITS}
    wanted = {
        split: max(floors[split], int(math.ceil(floors[split] * max(1.0, headroom))))
        for split in SPLITS
    }
    required = sum(wanted.values())
    if budget >= required:
        # Spend the surplus in proportion to each split's requirement.
        surplus = budget - required
        total_floor = max(1, sum(floors.values()))
        for split in SPLITS:
            wanted[split] += int(surplus * floors[split] / total_floor)
    else:
        LOGGER.warning(
            "data.hf_pass1_files=%d is below the %d files needed to fill the requested "
            "splits with %.1fx headroom; raising it to %d",
            budget,
            required,
            headroom,
            required,
        )
    return wanted


def pass1_huggingface(
    config: Config,
    rng: np.random.Generator,
    raw_dir: Path,
    n_files: int,
    counts: dict[str, int],
) -> list[dict[str, Any]]:
    """Download a broad random sample of real pulse trains and characterise them.

    Files are drawn separately from each of the dataset's own ``train_``/``val_``/``test_``
    directories, so our splits inherit the dataset's split boundaries rather than
    cutting across them.
    """
    repo_id = str(config.get("data.hf_repo_id"))
    mode = str(config.get("data.mode", "stare"))
    max_bytes = int(config.get("data.hf_max_file_bytes", 40_000_000))
    headroom = float(config.get("data.hf_pass1_headroom", 2.0))
    token = hf_source.read_hf_token()

    allocation = pass1_allocation(counts, n_files, headroom)
    LOGGER.info("Pass 1 file allocation per split: %s", allocation)

    table: list[dict[str, Any]] = []
    downloaded_bytes = 0
    for split in SPLITS:
        per_split = allocation.get(split, 0)
        if per_split <= 0:
            continue
        directory = hf_source.split_dir(mode, split)
        listing = hf_source.list_files(repo_id, directory, token=token)
        candidates = [entry for entry in listing if entry["size"] <= max_bytes]
        if not candidates:
            LOGGER.warning("No files under %d bytes in %s", max_bytes, directory)
            continue
        chosen = rng.choice(len(candidates), size=min(per_split, len(candidates)), replace=False)
        LOGGER.info(
            "Pass 1: inspecting %d/%d files from %s (~%.0f MB)",
            len(chosen),
            len(candidates),
            directory,
            sum(candidates[int(i)]["size"] for i in np.atleast_1d(chosen)) / 1e6,
        )
        for index in np.atleast_1d(chosen):
            entry = candidates[int(index)]
            destination = raw_dir / entry["path"]
            try:
                hf_source.download_file(
                    repo_id, entry["path"], destination, token=token, overwrite=False
                )
                record = load_pulse_train(destination)
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop Pass 1
                LOGGER.warning("Pass 1 skipped %s (%s)", entry["path"], exc)
                continue
            summary = characterise(record)
            summary["split_hint"] = split
            summary["file_bytes"] = entry["size"]
            downloaded_bytes += int(entry["size"])
            table.append(summary)
    LOGGER.info(
        "Pass 1 downloaded %d files, %.2f GB total", len(table), downloaded_bytes / 1e9
    )
    return table


def pass1_mock(config: Config, raw_dir: Path, n_files: int) -> list[dict[str, Any]]:
    """Generate a pool of mock pulse trains and characterise them."""
    mock_cfg = config.section("mock")
    mode = str(config.get("data.mode", "stare"))
    pool = max(int(mock_cfg.get("n_trains_pool", 40)), n_files)
    out_dir = raw_dir / "mock" / mode
    paths = generate_mock_dataset(
        out_dir=out_dir,
        seed=config.seed,
        mock_cfg=mock_cfg,
        n_trains=pool,
    )
    LOGGER.info("Pass 1: characterising %d mock pulse trains", len(paths))
    table: list[dict[str, Any]] = []
    for path in paths:
        record = load_pulse_train(path)
        summary = characterise(record)
        summary["file_bytes"] = int(Path(path).stat().st_size)
        table.append(summary)
    return table


def stratified_selection(
    table: list[dict[str, Any]],
    strata: list[str],
    counts: dict[str, int],
    rng: np.random.Generator,
) -> list[tuple[dict[str, Any], str, str]]:
    """Select whole pulse trains covering the requested strata, disjointly per split.

    Args:
        table: Pass 1 characterisation table.
        strata: strata to cover, in priority order.
        counts: number of trains wanted per split.
        rng: seeded generator for the draw.

    Returns:
        List of ``(characterisation, split, stratum)`` tuples.
    """
    for stratum in strata:
        members = [row for row in table if row.get(f"n_{stratum}", 0) > 0]
        LOGGER.info("Stratum %-16s: %d candidate pulse trains", stratum, len(members))

    used: set[str] = set()
    selection: list[tuple[dict[str, Any], str, str]] = []

    for split in SPLITS:
        wanted = int(counts.get(split, 0))
        if wanted <= 0:
            continue

        # Real TSRD files carry the dataset's own split directory as split_hint, and our
        # splits must inherit those boundaries rather than cutting across them. Generated
        # data has no official splits, so the whole table is the candidate pool.
        hinted = [row for row in table if row.get("split_hint") == split]
        candidates = hinted if hinted else list(table)
        if hinted:
            LOGGER.info(
                "Split %-10s: choosing from %d files of the dataset's own %s directory",
                split,
                len(hinted),
                split,
            )

        pools: dict[str, list[dict[str, Any]]] = {}
        for stratum in strata:
            members = [row for row in candidates if row.get(f"n_{stratum}", 0) > 0]
            members.sort(key=lambda row: row.get(f"frac_{stratum}", 0.0), reverse=True)
            pools[stratum] = members

        leftover_pool = list(candidates)
        rng.shuffle(leftover_pool)

        taken = 0
        # Round-robin across strata so every split covers both scenarios.
        while taken < wanted:
            progressed = False
            for stratum in strata:
                if taken >= wanted:
                    break
                for row in pools[stratum]:
                    key = str(row.get("path") or row.get("name"))
                    if key in used:
                        continue
                    used.add(key)
                    selection.append((row, split, stratum))
                    taken += 1
                    progressed = True
                    break
            if not progressed:
                break
        # Top up from whatever is left if the strata could not fill the split.
        while taken < wanted:
            leftover = [
                row for row in leftover_pool if str(row.get("path") or row.get("name")) not in used
            ]
            if not leftover:
                LOGGER.warning(
                    "Only %d/%d trains available for split %s. Increase "
                    "data.hf_pass1_files (or mock.n_trains_pool) and re-run.",
                    taken,
                    wanted,
                    split,
                )
                break
            row = leftover[0]
            used.add(str(row.get("path") or row.get("name")))
            selection.append((row, split, row.get("dominant_stratum") or "unstratified"))
            taken += 1
    return selection


def cache_selection(
    selection: list[tuple[dict[str, Any], str, str]],
    processed_dir: Path,
    max_pulses: int,
    *,
    force: bool,
) -> list[ManifestEntry]:
    """Cache the selected pulse trains as whole, ToA-ordered contiguous windows."""
    entries: list[ManifestEntry] = []
    for row, split, stratum in selection:
        source_path = row.get("path")
        if not source_path:
            LOGGER.warning("Selection row without a path: %s", row.get("name"))
            continue
        destination = processed_dir / split / f"{Path(source_path).stem}.h5"
        if force or not destination.exists():
            record = load_pulse_train(source_path).contiguous_window(max_pulses)
            record.save(destination)
        entries.append(
            ManifestEntry(
                name=Path(source_path).stem,
                path=str(destination),
                split=split,
                stratum=stratum,
                characterisation=row,
            )
        )
    return entries


def main(argv: list[str] | None = None) -> int:
    """Run the two-pass sampler."""
    args = parse_args(argv)
    config = Config.load(args.config)

    seed = args.seed if args.seed is not None else config.seed
    mode = args.mode or str(config.get("data.mode", "stare"))
    strata = (
        [s.strip() for s in args.stratify_on.split(",") if s.strip()]
        if args.stratify_on
        else list(config.get("data.stratify_on", list(STRATA)))
    )
    counts = {
        "train": args.train_trains
        if args.train_trains is not None
        else int(config.get("data.max_train_trains", 15)),
        "validation": args.val_trains
        if args.val_trains is not None
        else int(config.get("data.max_validation_trains", 5)),
        "test": args.test_trains
        if args.test_trains is not None
        else int(config.get("data.max_test_trains", 5)),
    }
    n_pass1 = (
        args.pass1_files
        if args.pass1_files is not None
        else int(config.get("data.hf_pass1_files", 40))
    )
    max_pulses = int(config.get("data.max_pulses_per_train", 100_000))

    raw_dir = config.path_for("paths.raw_dir")
    processed_dir = config.path_for("paths.processed_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    source, reason = resolve_source(config, args.source)
    LOGGER.info("Data source: %s (%s)", source, reason)

    rng = make_rng(seed, stream=1)
    if source == "huggingface":
        table = pass1_huggingface(config, rng, raw_dir, n_pass1, counts)
    else:
        table = pass1_mock(config, raw_dir, n_pass1)

    if not table:
        LOGGER.error("Pass 1 produced no usable pulse trains")
        return 1
    LOGGER.info(
        "Pass 1 complete: %d trains, %d with spatially scanning emitters, "
        "%d with frequency-agile emitters",
        len(table),
        sum(1 for row in table if row.get("n_spatial_scan", 0) > 0),
        sum(1 for row in table if row.get("n_frequency_agile", 0) > 0),
    )

    selection = stratified_selection(table, strata, counts, make_rng(seed, stream=2))
    entries = cache_selection(selection, processed_dir, max_pulses, force=args.force)

    manifest = DatasetManifest(
        seed=seed,
        mode=mode,
        source=source,
        entries=entries,
        pass1=table,
        notes={
            "source_reason": reason,
            "strata": strata,
            "counts_requested": counts,
            "max_pulses_per_train": max_pulses,
            "window_mode": str(config.get("data.window_mode", "contiguous")),
            "ground_truth_note": (
                "stare mode is used as ground truth: scan-mode files already have pulses "
                "missing because of the dataset's own sweeping receiver, which would be "
                "conflated with the environment if used as truth here"
            ),
        },
    )
    manifest.save(config.path_for("paths.manifest"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
