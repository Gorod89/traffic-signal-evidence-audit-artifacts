"""Shared loading and inference machinery for PACE Phase 2.

Phase 2 answers one question: does a selective intervention rule, fitted only on
data the model was allowed to see, produce a positive realised gain under a
bounded harm rate?  Three data blocks are used and they are never mixed:

    training     representation_train + development   960 rows / 160 clusters
    calibration  calibration                          720 rows / 120 clusters
    evaluation   phase0/sealed/raw/*.json            1800 rows / 300 clusters

The sealed block was generated on seeds 9801-9900 after the estimators in
`phase0` were frozen; nothing in this module or its callers is permitted to read
it while fitting a model or choosing a threshold.  The cluster key is
(network, seed), which is the coarsest available unit: `route_sha256` is nested
strictly inside it (verified: zero route hashes span two (network, seed) pairs),
so clustering at (network, seed) is conservative relative to clustering at the
route seed used by `v9/src/v9evidence/statistics.py`.

Two interval families are provided, and callers must pick deliberately:

  * `cluster_bootstrap_ci` -- nonparametric percentile interval over resampled
    clusters.  Used for unbounded quantities (RMSE, mean gain) and reported as a
    diagnostic for bounded ones.
  * `empirical_bernstein` -- the finite-sample interval for a mean of bounded
    cluster statistics, transcribed from
    `v9/src/v9evidence/statistics.py:_empirical_bernstein_interval`.  Used for
    every gate-facing harm and coverage bound, because the percentile bootstrap
    degenerates to a point when every observed cluster sits on a boundary, and
    because V4 reported its 56.15% harm ceiling with this estimator -- keeping it
    makes the historical comparison like-for-like rather than method-for-method.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PAIRED = REPO / "v6" / "results" / "cfra" / "paired_branches.csv"
SEALED_RAW = REPO / "v10_proposal" / "phase0" / "sealed" / "raw"
RESULTS = Path(__file__).resolve().parent / "results"

EXPECTED_PAIRED_SHA = "24231bba4eadff8051da0af7705ff429d86c93ec10c554ede9247836aea55d8e"

TARGET = "gain_h4"
TRAIN_SPLITS = ("representation_train", "development")
CALIB_SPLIT = "calibration"
SEALED_SEED_BLOCK = (9801, 9900)

SEED = 20260811
BOOTSTRAP_REPLICATES = 4000
CONFIDENCE = 0.95


@dataclass(frozen=True)
class Block:
    """One analysis block: features, target, clusters, provenance."""

    name: str
    obs: np.ndarray  # context_observation, 10 columns
    con: np.ndarray  # feature_vector, 81 columns
    y: np.ndarray
    clusters: np.ndarray  # "<network>|<seed>"
    network: np.ndarray
    condition: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(self.y.size)

    @property
    def n_clusters(self) -> int:
        return int(np.unique(self.clusters).size)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _as_matrix(values) -> np.ndarray:
    """`feature_vector` is a native list in the sealed JSON and JSON text in the CSV."""
    rows = []
    for v in values:
        if isinstance(v, str):
            v = json.loads(v)
        rows.append(np.asarray(v, dtype=np.float64).ravel())
    return np.vstack(rows)


def load_csv_blocks() -> tuple[Block, Block, str]:
    """Training and calibration blocks from the V6.1 paired archive."""
    actual = sha256_file(PAIRED)
    if actual != EXPECTED_PAIRED_SHA:
        raise SystemExit(
            f"paired_branches.csv hash changed: expected {EXPECTED_PAIRED_SHA}, found {actual}"
        )
    df = pd.read_csv(PAIRED)
    if not (df.status == "complete").all():
        raise SystemExit("paired archive contains incomplete rows")
    df["cluster"] = df.network + "|" + df.seed.astype(str)

    def build(name: str, frame: pd.DataFrame) -> Block:
        frame = frame.reset_index(drop=True)
        return Block(
            name=name,
            obs=_as_matrix(frame["context_observation"]),
            con=_as_matrix(frame["feature_vector"]),
            y=frame[TARGET].to_numpy(dtype=np.float64),
            clusters=frame["cluster"].to_numpy().astype(str),
            network=frame["network"].to_numpy().astype(str),
            condition=frame["condition"].to_numpy().astype(str),
        )

    train = build("training", df[df.split.isin(TRAIN_SPLITS)])
    calib = build("calibration", df[df.split == CALIB_SPLIT])

    overlap = set(train.clusters) & set(calib.clusters)
    if overlap:
        raise SystemExit(f"training and calibration share {len(overlap)} clusters")
    return train, calib, actual


def load_sealed_block() -> tuple[Block, dict]:
    """The sealed evaluation block. Read only after every fit and threshold is fixed."""
    files = sorted(glob.glob(str(SEALED_RAW / "*.json")))
    records, dropped = [], 0
    for path in files:
        rec = json.loads(Path(path).read_text(encoding="utf-8"))
        if rec.get("status") != "complete" or rec.get(TARGET) is None:
            dropped += 1
            continue
        records.append(rec)

    seeds = {int(r["seed"]) for r in records}
    lo, hi = SEALED_SEED_BLOCK
    if seeds - set(range(lo, hi + 1)):
        raise SystemExit("sealed block contains seeds outside the preregistered window")

    block = Block(
        name="sealed",
        obs=_as_matrix([r["context_observation"] for r in records]),
        con=_as_matrix([r["feature_vector"] for r in records]),
        y=np.asarray([float(r[TARGET]) for r in records]),
        clusters=np.asarray([f"{r['network']}|{r['seed']}" for r in records]),
        network=np.asarray([r["network"] for r in records]),
        condition=np.asarray([r["condition"] for r in records]),
    )
    provenance = {
        "source": "v10_proposal/phase0/sealed/raw/*.json",
        "scheduled": 1800,
        "analysable": block.n_rows,
        "dropped_incomplete": dropped,
        "attrition": 1.0 - block.n_rows / 1800,
        "clusters": block.n_clusters,
        "seed_block": list(SEALED_SEED_BLOCK),
        "manifest_sha256": hashlib.sha256(
            "".join(Path(p).name for p in files).encode("utf-8")
        ).hexdigest(),
    }
    return block, provenance


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------


def cluster_index(clusters: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    uniq = np.unique(clusters)
    return uniq, {c: np.flatnonzero(clusters == c) for c in uniq}


def cluster_bootstrap_ci(
    fn,
    clusters: np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = SEED,
    alpha: float = 1.0 - CONFIDENCE,
) -> tuple[float | None, float | None, int]:
    """Percentile interval over clusters resampled with replacement.

    `fn` receives row indices and may raise ValueError to reject a degenerate
    replicate (an empty selected set, a single-class AUROC).  Rejections are
    counted and reported rather than silently absorbed: a metric that is
    undefined in a large share of replicates is not a metric with an interval.
    """
    rng = np.random.default_rng(seed)
    uniq, index_of = cluster_index(clusters)
    draws: list[float] = []
    rejected = 0
    for _ in range(replicates):
        picked = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([index_of[c] for c in picked])
        try:
            draws.append(float(fn(idx)))
        except ValueError:
            rejected += 1
    if len(draws) < replicates // 2:
        return None, None, rejected
    return (
        float(np.percentile(draws, 100 * alpha / 2)),
        float(np.percentile(draws, 100 * (1 - alpha / 2))),
        rejected,
    )


def empirical_bernstein(values, *, confidence_level: float = CONFIDENCE):
    """Finite-sample two-sided interval for a mean of values in [0, 1].

    Transcribed unchanged from v9/src/v9evidence/statistics.py so that the harm
    ceiling reported here and the 56.15% ceiling reported by V4 are produced by
    the same estimator.
    """
    values = [float(v) for v in values]
    if len(values) < 2:
        raise ValueError("at least two independent clusters are required for a bound")
    if any(not math.isfinite(v) or not 0.0 <= v <= 1.0 for v in values):
        raise ValueError("empirical Bernstein inputs must be finite and in [0, 1]")
    n = len(values)
    mean = sum(values) / n
    sample_variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    delta = 1.0 - confidence_level
    log_term = math.log(2.0 / delta)
    radius = math.sqrt(2.0 * sample_variance * log_term / n) + 7.0 * log_term / (3.0 * (n - 1))
    return mean, max(0.0, mean - radius), min(1.0, mean + radius)


def cluster_means(values: np.ndarray, clusters: np.ndarray, mask: np.ndarray) -> list[float]:
    """Within-cluster means of `values` over rows where `mask` holds.

    Clusters with no unmasked row are omitted; the caller decides what an empty
    result means.  Equal weight per cluster is the V9 estimand
    (`equal_weight_mean_route_seed_*`) and is used here for every bounded
    quantity so the numbers stay comparable to the V4/V5 gate.
    """
    out = []
    for c in np.unique(clusters):
        sel = mask & (clusters == c)
        if sel.any():
            out.append(float(values[sel].mean()))
    return out


def one_sided_lower(
    fn, clusters: np.ndarray, *, replicates: int = BOOTSTRAP_REPLICATES, seed: int = SEED,
    level: float = CONFIDENCE,
) -> tuple[float | None, int]:
    """One-sided cluster-bootstrap lower confidence bound at `level`."""
    rng = np.random.default_rng(seed)
    uniq, index_of = cluster_index(clusters)
    draws, rejected = [], 0
    for _ in range(replicates):
        picked = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([index_of[c] for c in picked])
        try:
            draws.append(float(fn(idx)))
        except ValueError:
            rejected += 1
    if len(draws) < replicates // 2:
        return None, rejected
    return float(np.percentile(draws, 100 * (1.0 - level))), rejected


def bootstrap_active_cluster_mean(
    per_cluster: np.ndarray,
    active: np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = SEED,
    alpha: float = 1.0 - CONFIDENCE,
) -> dict:
    """Cluster bootstrap for an equal-weight mean taken over active clusters only.

    `per_cluster[c]` is the within-cluster statistic (its value is ignored where
    `active[c]` is False, e.g. a cluster in which the rule selected nothing).
    Clusters are resampled with replacement; each replicate averages the statistic
    over whichever resampled clusters happen to be active.

    This is algebraically the same estimator as resampling clusters, concatenating
    their rows, and recomputing the equal-weight mean -- the within-cluster
    statistic does not depend on which other clusters were drawn -- but it costs
    O(replicates * n_clusters) instead of O(replicates * n_rows * n_clusters).
    Replicates in which no drawn cluster is active are counted and excluded, since
    the estimand is undefined there.
    """
    per_cluster = np.asarray(per_cluster, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    n = per_cluster.size
    if n == 0 or not active.any():
        return {"ci95": [None, None], "one_sided_lower": None, "degenerate_replicates": replicates}
    rng = np.random.default_rng(seed)
    draw_idx = rng.integers(0, n, size=(replicates, n))
    values = np.where(active[draw_idx], per_cluster[draw_idx], 0.0)
    counts = active[draw_idx].sum(axis=1)
    ok = counts > 0
    if ok.sum() < replicates // 2:
        return {
            "ci95": [None, None],
            "one_sided_lower": None,
            "degenerate_replicates": int((~ok).sum()),
        }
    means = values[ok].sum(axis=1) / counts[ok]
    return {
        "ci95": [
            float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))),
        ],
        "one_sided_lower": float(np.percentile(means, 100 * alpha)),
        "degenerate_replicates": int((~ok).sum()),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")
