"""Score the frozen estimator on the sealed block against the prediction lock.

This is the confirmatory step. `preregister_sealed.py` wrote a prediction and a
frozen model before the sealed seeds existed; `collect_sealed.py` then generated
them. Nothing here fits a model, tunes a threshold, or chooses an endpoint --
all of that was fixed in the lock, and this script reads it rather than
restating it.

The primary prediction is that the frozen estimator attains an RMSE gain of at
least 10% over the training-set mean on data it has never seen, drawn from seeds
that did not exist when it was fitted. A point estimate below that threshold
falsifies the sample-size account of the V4 and V5 failures, and is reported as
such.
"""

from __future__ import annotations

import glob
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from integrity import write_json_atomic

HERE = Path(__file__).resolve().parent
SEALED = HERE / "sealed"
LOCK_PATH = SEALED / "PREDICTION_LOCK.json"
MODEL_PATH = SEALED / "frozen_estimator.pkl"
OUT = HERE / "results" / "sealed_confirmation.json"


def cluster_bootstrap(fn, clusters: np.ndarray, seed: int, replicates: int):
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    index_of = {c: np.flatnonzero(clusters == c) for c in uniq}
    draws = []
    for _ in range(replicates):
        picked = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([index_of[c] for c in picked])
        try:
            draws.append(fn(idx))
        except ValueError:
            continue
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))

    # The lock names the estimator by hash; refuse to score a different object.
    actual_model_sha = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    if actual_model_sha != lock["frozen_estimator"]["sha256"]:
        raise SystemExit(
            "frozen estimator does not match the prediction lock; "
            f"expected {lock['frozen_estimator']['sha256'][:16]}..., "
            f"found {actual_model_sha[:16]}..."
        )
    model = pickle.loads(MODEL_PATH.read_bytes())

    files = sorted(glob.glob(str(SEALED / "raw" / "*.json")))
    rows, failed = [], 0
    for path in files:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if record.get("status") != "complete" or record.get("gain_h4") is None:
            failed += 1
            continue
        rows.append(record)

    scheduled = 1800
    attrition = 1.0 - len(rows) / scheduled
    def as_vector(value) -> np.ndarray:
        """Raw branch records store the vector natively; the CSV stores it as JSON text."""
        if isinstance(value, str):
            value = json.loads(value)
        return np.asarray(value, dtype=np.float64).ravel()

    x = np.vstack([as_vector(r["feature_vector"]) for r in rows])
    y = np.asarray([float(r["gain_h4"]) for r in rows])
    clusters = np.asarray([f"{r['network']}|{r['seed']}" for r in rows])
    pred = model.predict(x)

    # The reference is the training-set mean, exactly as the Stage-B gate defined
    # it. It is recomputed from the lock's training description, not re-derived
    # from the sealed data, which would leak.
    import pandas as pd

    train_src = Path(__file__).resolve().parents[2] / lock["training"]["source"]
    train_df = pd.read_csv(train_src)
    train = train_df[train_df.split.isin(lock["training"]["splits"])]
    reference = float(train[lock["training"]["target"]].mean())

    def rmse_gain(idx: np.ndarray) -> float:
        m = float(np.sqrt(np.mean((pred[idx] - y[idx]) ** 2)))
        d = float(np.sqrt(np.mean((reference - y[idx]) ** 2)))
        return 1.0 - m / d

    analysis = lock["fixed_analysis"]
    all_idx = np.arange(len(y))
    point = rmse_gain(all_idx)
    lo, hi = cluster_bootstrap(
        rmse_gain, clusters, analysis["bootstrap_seed"], analysis["bootstrap_replicates"]
    )

    from sklearn.metrics import roc_auc_score

    def auroc(idx: np.ndarray) -> float:
        labels = (y[idx] > 0).astype(int)
        if labels.min() == labels.max():
            raise ValueError("degenerate")
        return float(roc_auc_score(labels, pred[idx]))

    au = auroc(all_idx)
    au_lo, au_hi = cluster_bootstrap(
        auroc, clusters, analysis["bootstrap_seed"], analysis["bootstrap_replicates"]
    )

    threshold = lock["predictions"]["primary"]["threshold"]
    interval = lock["predictions"]["secondary_interval"]["interval"]
    max_attrition = 0.10

    primary_met = bool(point >= threshold and lo > 0.0)
    secondary_met = bool(interval[0] <= point <= interval[1])
    inconclusive = bool(attrition > max_attrition)

    per_network = {}
    for net in sorted({r["network"] for r in rows}):
        mask = np.asarray([r["network"] == net for r in rows])
        per_network[net] = {
            "rows": int(mask.sum()),
            "clusters": int(np.unique(clusters[mask]).size),
            "rmse_gain": rmse_gain(np.flatnonzero(mask)),
        }

    report = {
        "evidence_class": "confirmatory" if not inconclusive else "inconclusive_by_attrition",
        "prediction_lock_written_utc": lock["written_utc"],
        "frozen_estimator_sha256": actual_model_sha,
        "estimator_refitted": False,
        "sealed_data": {
            "scheduled": scheduled,
            "analysable": len(rows),
            "failed_or_incomplete": failed,
            "attrition": attrition,
            "attrition_limit": max_attrition,
            "clusters": int(np.unique(clusters).size),
            "seed_block": lock["evaluation_population"]["seed_block"],
        },
        "reference_predictor_value": reference,
        "primary": {
            "statement": lock["predictions"]["primary"]["statement"],
            "threshold": threshold,
            "observed_rmse_gain": point,
            "ci95": [lo, hi],
            "met": primary_met,
        },
        "secondary": {
            "statement": lock["predictions"]["secondary_interval"]["statement"],
            "predicted_interval": interval,
            "observed": point,
            "met": secondary_met,
        },
        "benefit_auroc": {"observed": au, "ci95": [au_lo, au_hi]},
        "per_network": per_network,
    }
    write_json_atomic(OUT, report)

    print("=" * 66)
    print("CONFIRMATORY TEST AGAINST A PREREGISTERED PREDICTION")
    print("=" * 66)
    print(f"prediction written : {lock['written_utc']}")
    print(f"estimator sha256   : {actual_model_sha[:16]}...  (not refitted)")
    print(f"sealed seeds       : {lock['evaluation_population']['seed_block']}")
    print(f"analysable pairs   : {len(rows)} / {scheduled}   attrition {100*attrition:.2f}%")
    print(f"independent clusters: {np.unique(clusters).size}")
    print()
    print(f"PRIMARY   predicted >= {100*threshold:.0f}%")
    print(f"          observed    {100*point:.2f}%   95% CI [{100*lo:.2f}%, {100*hi:.2f}%]")
    print(f"          -> {'MET' if primary_met else 'NOT MET'}")
    print()
    print(f"SECONDARY predicted in [{100*interval[0]:.2f}%, {100*interval[1]:.2f}%]")
    print(f"          observed    {100*point:.2f}%  -> {'MET' if secondary_met else 'NOT MET'}")
    print()
    print(f"benefit AUROC       {au:.3f}   95% CI [{au_lo:.3f}, {au_hi:.3f}]")
    print()
    for net, r in per_network.items():
        print(f"  {net:<14} {r['rows']:>4} rows / {r['clusters']:>3} clusters   gain {100*r['rmse_gain']:6.2f}%")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
