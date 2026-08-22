"""Build the activity-model training set from observation histories.

Rows are collected by running *exploration* policies (a sequential sweep and a uniform
random sweep) over the sampled environments. At a regular stride, the feature vectors of
every band are recorded together with the ground truth target "does this band transmit at
any point in the next ``prediction_window`` timesteps?".

Two properties matter and are enforced here:

* **No future leakage.** Features come only from the receiver's own observation history
  up to that timestep. Ground truth is used solely to write the target column.
* **Splitting by pulse train, not by pulse.** Train, validation and test rows come from
  disjoint pulse trains, as fixed by the sampler's manifest, so a model cannot memorise
  one environment and be scored on it.

Usage::

    python training/prepare_dataset.py --config config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import Config, make_rng  # noqa: E402
from common.io_utils import utc_timestamp, write_json  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402
from dataio.manifest import SPLITS, DatasetManifest  # noqa: E402
from features.band_features import FEATURE_NAMES, BandFeatureTracker  # noqa: E402
from models.scheduler import build_scheduler  # noqa: E402
from simulation.environment import EnvironmentGrid, build_environment  # noqa: E402
from simulation.receiver import make_receiver  # noqa: E402

LOGGER = get_logger("training.prepare_dataset")


def collect_rows(
    environment: EnvironmentGrid,
    *,
    policy: str,
    receiver_cfg: dict[str, Any],
    features_cfg: dict[str, Any],
    prediction_window: int,
    stride: int,
    rng: np.random.Generator,
    n_timesteps: int | None = None,
) -> dict[str, np.ndarray]:
    """Run one exploration policy over one environment and collect feature rows.

    Args:
        environment: ground truth grid to explore.
        policy: ``sequential`` or ``random``.
        receiver_cfg: the ``receiver`` configuration section.
        features_cfg: the ``features`` configuration section.
        prediction_window: how far ahead the target looks, in timesteps.
        stride: record every band's features once every ``stride`` timesteps.
        rng: seeded generator.
        n_timesteps: optional cap on the run length.

    Returns:
        Dictionary of arrays: ``x``, ``y``, ``band``, ``timestep``.
    """
    horizon = min(int(n_timesteps or environment.n_timesteps), environment.n_timesteps)
    n_bands = environment.n_bands
    targets = environment.active_window_matrix(prediction_window)

    tracker = BandFeatureTracker(
        n_bands=n_bands, features_cfg=features_cfg, timestep_us=environment.timestep_us
    )
    receiver = make_receiver(receiver_cfg, rng)
    scheduler = build_scheduler(policy, n_bands=n_bands, rng=rng)

    feature_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    band_blocks: list[np.ndarray] = []
    timestep_blocks: list[np.ndarray] = []

    for timestep in range(horizon):
        if receiver.needs_decision:
            receiver.tune(scheduler.select_band(timestep, tracker))

        if timestep % max(1, stride) == 0:
            feature_blocks.append(tracker.snapshot(timestep))
            target_blocks.append(targets[timestep].astype(np.int8))
            band_blocks.append(np.arange(n_bands, dtype=np.int16))
            timestep_blocks.append(np.full(n_bands, timestep, dtype=np.int32))

        observation = receiver.observe(environment, timestep)
        tracker.update_from_observation(observation)
        scheduler.update(observation)

    if not feature_blocks:
        empty = np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        return {
            "x": empty,
            "y": np.empty(0, dtype=np.int8),
            "band": np.empty(0, dtype=np.int16),
            "timestep": np.empty(0, dtype=np.int32),
        }

    return {
        "x": np.vstack(feature_blocks).astype(np.float32),
        "y": np.concatenate(target_blocks),
        "band": np.concatenate(band_blocks),
        "timestep": np.concatenate(timestep_blocks),
    }


def build_split(
    manifest: DatasetManifest,
    split: str,
    config: Config,
) -> dict[str, np.ndarray]:
    """Collect all rows for one split, across pulse trains and exploration policies."""
    env_cfg = config.section("environment")
    receiver_cfg = config.section("receiver")
    features_cfg = config.section("features")
    model_cfg = config.section("activity_model")
    prediction_window = int(model_cfg.get("prediction_window", 5))
    stride = int(model_cfg.get("collection_stride", 5))
    policies = list(model_cfg.get("collection_policies", ["sequential", "random"]))
    max_pulses = int(config.get("data.max_pulses_per_train", 100_000))
    horizon = int(
        model_cfg.get("collection_timesteps", config.get("simulation.n_timesteps", 2000))
    )

    blocks: list[dict[str, np.ndarray]] = []
    environment_ids: list[np.ndarray] = []
    policy_ids: list[np.ndarray] = []
    names: list[str] = []

    for env_index, entry in enumerate(manifest.for_split(split)):
        try:
            record = entry.load(max_pulses=max_pulses)
            environment = build_environment(record, env_cfg, name=entry.name)
        except (FileNotFoundError, ValueError) as exc:
            LOGGER.warning("Skipping %s: %s", entry.path, exc)
            continue
        names.append(entry.name)
        for policy_index, policy in enumerate(policies):
            rows = collect_rows(
                environment,
                policy=policy,
                receiver_cfg=receiver_cfg,
                features_cfg=features_cfg,
                prediction_window=prediction_window,
                stride=stride,
                rng=make_rng(config.seed, stream=1000 + env_index * 10 + policy_index),
                n_timesteps=horizon,
            )
            if rows["y"].size == 0:
                continue
            blocks.append(rows)
            environment_ids.append(np.full(rows["y"].size, env_index, dtype=np.int16))
            policy_ids.append(np.full(rows["y"].size, policy_index, dtype=np.int8))
        LOGGER.info(
            "Split %-10s | %-12s | %d timesteps | occupancy %.3f",
            split,
            entry.name,
            environment.n_timesteps,
            environment.occupancy,
        )

    if not blocks:
        msg = f"No rows collected for split {split!r}"
        raise RuntimeError(msg)

    return {
        "x": np.vstack([block["x"] for block in blocks]),
        "y": np.concatenate([block["y"] for block in blocks]),
        "band": np.concatenate([block["band"] for block in blocks]),
        "timestep": np.concatenate([block["timestep"] for block in blocks]),
        "environment": np.concatenate(environment_ids),
        "policy": np.concatenate(policy_ids),
        "_names": np.array(names, dtype=object),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument(
        "--splits", default=",".join(SPLITS), help="comma-separated splits to build"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build and save the activity-model datasets."""
    args = parse_args(argv)
    config = Config.load(args.config)
    manifest = DatasetManifest.load(config.path_for("paths.manifest"))
    out_dir = config.path_for("paths.processed_dir") / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    prediction_window = int(config.get("activity_model.prediction_window", 5))
    summary: dict[str, Any] = {
        "created": utc_timestamp(),
        "seed": config.seed,
        "manifest": str(config.path_for("paths.manifest")),
        "prediction_window": prediction_window,
        "collection_stride": int(config.get("activity_model.collection_stride", 5)),
        "collection_policies": list(
            config.get("activity_model.collection_policies", ["sequential", "random"])
        ),
        "feature_names": list(FEATURE_NAMES),
        "splits": {},
    }

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        data = build_split(manifest, split, config)
        names = data.pop("_names")
        path = out_dir / f"{split}.npz"
        np.savez_compressed(path, **data)
        positive_rate = float(data["y"].mean())
        summary["splits"][split] = {
            "path": str(path),
            "n_rows": int(data["y"].size),
            "positive_rate": positive_rate,
            "n_environments": int(np.unique(data["environment"]).size),
            "pulse_trains": names.tolist(),
        }
        LOGGER.info(
            "Split %-10s: %d rows, positive rate %.3f -> %s",
            split,
            data["y"].size,
            positive_rate,
            path.name,
        )
        if positive_rate < 0.02 or positive_rate > 0.98:
            LOGGER.warning(
                "Split %s is heavily imbalanced (positive rate %.3f). A ROC-AUC on this "
                "split will look good for a model that predicts almost nothing; consider "
                "changing activity_model.prediction_window or environment.timestep_us.",
                split,
                positive_rate,
            )

    write_json(out_dir / "dataset_summary.json", summary)
    LOGGER.info("Wrote %s", out_dir / "dataset_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
