"""Optional P2 bonus: stage-1 deinterleaving with HDBSCAN.

**This is not required by the problem statement.** The scheduler in this repository works
directly on ``environment[timestep][band]`` transmission truth and never needs emitter
cluster identities. This module exists only to engage with the Turing Deinterleaving
Challenge itself, and its metrics are kept entirely separate from the scheduler's figures
of merit -- they are two different stages and must never be mixed in one table.

The clustering metrics follow the challenge's own ``evaluate_labels``: homogeneity,
completeness, V-measure, ARI, AMI, and the cluster-wise MCC/F1, all scaled by the ratio of
pulses the model was willing to label. When the official
``turing_deinterleaving_challenge`` package is installed its implementation is used
directly; otherwise the same computation is reproduced here with scikit-learn.

Note on scaling: ToA, CF, PW, AoA and amplitude live on wildly different scales, and ToA
grows monotonically across the whole collection window. Scaling them together without
thinking would let ToA dominate every distance. The default here standardises each
feature and then down-weights ToA, which is a crude but explicit choice.

Usage::

    python optional/deinterleaver.py --config config.yaml --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    f1_score,
    homogeneity_score,
    matthews_corrcoef,
    v_measure_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import Config, set_global_seed  # noqa: E402
from common.io_utils import utc_timestamp, write_json  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402
from dataio.manifest import DatasetManifest  # noqa: E402
from dataio.tdc_interface import PulseTrainRecord  # noqa: E402

LOGGER = get_logger("optional.deinterleaver")

#: Exponent applied to the labelled-pulse ratio, matching the challenge's PENALTY_ALPHA.
PENALTY_ALPHA = 1.0

#: Weight applied to the standardised ToA column. ToA spans the whole collection window,
#: so left at 1.0 it dominates every pairwise distance and the clustering degenerates into
#: time slicing.
DEFAULT_TOA_WEIGHT = 0.15


def scale_pdws(
    record: PulseTrainRecord,
    fields: list[str],
    *,
    toa_weight: float = DEFAULT_TOA_WEIGHT,
) -> np.ndarray:
    """Standardise the selected PDW fields, down-weighting ToA.

    Args:
        record: pulse train to featurise.
        fields: canonical PDW field names to use.
        toa_weight: multiplier applied to the standardised ToA column.

    Returns:
        ``(n_pulses, len(fields))`` scaled feature matrix.
    """
    columns: list[np.ndarray] = []
    for name in fields:
        if not record.has(name):
            LOGGER.warning("Pulse train %s has no %s column; skipping it", record.name, name)
            continue
        values = record.column(name).astype(np.float64)
        spread = float(values.std())
        scaled = (values - values.mean()) / spread if spread > 0 else np.zeros_like(values)
        if name == "toa":
            scaled = scaled * float(toa_weight)
        columns.append(scaled)
    return np.column_stack(columns) if columns else np.empty((record.n_pulses, 0))


def cluster_wise_score(labels_true: np.ndarray, labels_pred: np.ndarray, score: str) -> float:
    """Reproduce the challenge's cluster-wise MCC / F1 score.

    For every true emitter, the best-matching predicted cluster is found and scored as a
    binary problem; the reported figure is the worst such score over all emitters.
    """
    score_fun = {"mcc": matthews_corrcoef, "f1": f1_score}[score]
    per_cluster: list[float] = []
    for target in np.unique(labels_true):
        target_mask = (labels_true == target).astype(int)
        best = 0.0
        for cluster in np.unique(labels_pred):
            cluster_mask = (labels_pred == cluster).astype(int)
            best = max(best, float(score_fun(cluster_mask, target_mask)))
        per_cluster.append(best)
    return float(np.min(per_cluster)) if per_cluster else 0.0


def evaluate_labels(
    labels_pred: np.ndarray,
    labels_true: np.ndarray,
    predict_ratio: float = 1.0,
) -> dict[str, float]:
    """Score predicted clusters with the challenge's metric set.

    Uses ``turing_deinterleaving_challenge.models.evaluate.evaluate_labels`` when the
    package is importable, and an equivalent scikit-learn implementation otherwise.

    Args:
        labels_pred: predicted cluster ids.
        labels_true: ground truth emitter labels (local to this pulse train).
        predict_ratio: fraction of pulses the model assigned to a cluster.

    Returns:
        Dictionary of clustering metrics, each scaled by ``predict_ratio``.
    """
    try:  # pragma: no cover - depends on the local environment
        from turing_deinterleaving_challenge.models.evaluate import (
            evaluate_labels as official_evaluate,
        )

        return dict(official_evaluate(labels_pred, labels_true, predict_ratio))
    except Exception:
        pass

    penalty = float(predict_ratio**PENALTY_ALPHA)
    return {
        "Homogeneity": penalty * float(homogeneity_score(labels_true, labels_pred)),
        "Completeness": penalty * float(completeness_score(labels_true, labels_pred)),
        "V-measure": penalty * float(v_measure_score(labels_true, labels_pred)),
        "Adjusted Rand Index": penalty * float(adjusted_rand_score(labels_true, labels_pred)),
        "Adjusted Mutual Information": penalty
        * float(adjusted_mutual_info_score(labels_true, labels_pred)),
        "MCC": penalty * cluster_wise_score(labels_true, labels_pred, "mcc"),
        "F1": penalty * cluster_wise_score(labels_true, labels_pred, "f1"),
        "discount": penalty,
    }


def deinterleave(
    record: PulseTrainRecord,
    deinterleaver_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Cluster one pulse train's PDWs and score the result.

    Args:
        record: pulse train with ground truth labels.
        deinterleaver_cfg: the ``deinterleaver`` configuration section.

    Returns:
        Metrics plus cluster counts, or a record explaining why it was skipped.
    """
    from sklearn.cluster import HDBSCAN

    max_pulses = int(deinterleaver_cfg.get("max_pulses", 20_000))
    fields = list(deinterleaver_cfg.get("features", ["toa", "cf", "pw", "aoa", "amplitude"]))
    min_cluster_size = int(deinterleaver_cfg.get("min_cluster_size", 25))
    toa_weight = float(deinterleaver_cfg.get("toa_weight", DEFAULT_TOA_WEIGHT))

    window = record.contiguous_window(max_pulses)
    if window.labels is None or window.labels.size != window.n_pulses:
        return {"pulse_train": record.name, "skipped": "no ground truth labels"}

    features = scale_pdws(window, fields, toa_weight=toa_weight)
    if features.shape[1] == 0 or features.shape[0] < min_cluster_size:
        return {"pulse_train": record.name, "skipped": "too few pulses or no usable features"}

    clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
    predicted = clusterer.fit_predict(features)
    labelled = predicted >= 0
    predict_ratio = float(labelled.mean())
    if predict_ratio == 0.0:
        return {"pulse_train": record.name, "skipped": "every pulse was labelled as noise"}

    labels_true = np.asarray(window.labels)[labelled]
    labels_pred = predicted[labelled]
    metrics = evaluate_labels(labels_pred, labels_true, predict_ratio=predict_ratio)
    return {
        "pulse_train": record.name,
        "n_pulses": int(window.n_pulses),
        "n_true_emitters": int(np.unique(window.labels).size),
        "n_clusters_found": int(np.unique(labels_pred).size),
        "predict_ratio": predict_ratio,
        "metrics": metrics,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--split", default="test", help="manifest split to cluster")
    parser.add_argument("--max-trains", type=int, default=3, help="pulse trains to cluster")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the optional deinterleaving stage over a few pulse trains."""
    args = parse_args(argv)
    config = Config.load(args.config)
    set_global_seed(config.seed)

    manifest = DatasetManifest.load(config.path_for("paths.manifest"))
    deinterleaver_cfg = config.section("deinterleaver")
    entries = manifest.for_split(args.split)[: args.max_trains]
    if not entries:
        LOGGER.error("No pulse trains in split %s", args.split)
        return 1

    rows: list[dict[str, Any]] = []
    for entry in entries:
        record = entry.load(max_pulses=int(deinterleaver_cfg.get("max_pulses", 20_000)))
        LOGGER.info("Clustering %s (%d pulses)", entry.name, record.n_pulses)
        rows.append(deinterleave(record, deinterleaver_cfg))

    scored = [row for row in rows if "metrics" in row]
    summary: dict[str, float] = {}
    if scored:
        for key in scored[0]["metrics"]:
            summary[key] = float(np.mean([row["metrics"][key] for row in scored]))

    write_json(
        config.path_for("paths.results_dir") / "deinterleaving_results.json",
        {
            "created": utc_timestamp(),
            "split": args.split,
            "data_source": manifest.source,
            "note": (
                "Stage-1 deinterleaving. Not required by the problem statement and kept "
                "entirely separate from the scheduler figures of merit."
            ),
            "config": deinterleaver_cfg,
            "rows": rows,
            "summary": summary,
        },
    )

    LOGGER.info("=== HDBSCAN deinterleaving (%s, %d pulse trains) ===", args.split, len(scored))
    for key, value in summary.items():
        LOGGER.info("  %-30s %8.4f", key, value)
    LOGGER.info(
        "Wrote %s", config.path_for("paths.results_dir") / "deinterleaving_results.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
