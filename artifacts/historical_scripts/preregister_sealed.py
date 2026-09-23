"""Commit the prediction before the sealed data exist.

The reanalysis in `verify_fact2.py` and `sample_size_curve.py` is retrospective:
it used a validation split the V6 study had already consumed at Gate B. That is
why the paper labels it diagnostic. The only way to upgrade it is to state a
falsifiable prediction, then generate data that did not exist when the
prediction was made.

This script does the first half. It fits the estimator once on
representation_train + development, freezes it to disk with a hash, records the
prediction it implies for a fresh sample, and fixes the analysis that will score
it. `collect_sealed.py` refuses to run until this file exists.

Nothing here reads the sealed seed block; it does not exist yet.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

REPO = Path(__file__).resolve().parents[2]
PAIRED = REPO / "v6" / "results" / "cfra" / "paired_branches.csv"
EXPECTED_SHA = "24231bba4eadff8051da0af7705ff429d86c93ec10c554ede9247836aea55d8e"
OUTDIR = Path(__file__).resolve().parent / "sealed"
MODEL_PATH = OUTDIR / "frozen_estimator.pkl"
LOCK_PATH = OUTDIR / "PREDICTION_LOCK.json"

TARGET = "gain_h4"
SEED = 20260805


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_vector_column(series: pd.Series) -> np.ndarray:
    return np.vstack([np.asarray(json.loads(v), dtype=np.float64).ravel() for v in series])


def main() -> None:
    if LOCK_PATH.exists():
        raise SystemExit(
            f"{LOCK_PATH} already exists. A prediction lock is written once; "
            "regenerating it after seeing outcomes would void the whole exercise."
        )

    actual = sha256_file(PAIRED)
    if actual != EXPECTED_SHA:
        raise SystemExit(f"training input hash changed: {actual}")

    df = pd.read_csv(PAIRED)
    train = df[df.split.isin(("representation_train", "development"))].reset_index(drop=True)
    x = parse_vector_column(train.feature_vector)
    y = train[TARGET].to_numpy(dtype=np.float64)

    model = ExtraTreesRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
    model.fit(x, y)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_bytes(pickle.dumps(model))

    lock = {
        "schema_version": 1,
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Out-of-sample confirmation of the sample-size account of the V4/V5 "
            "Stage-B failures. Registered before the sealed data were generated."
        ),
        "training": {
            "source": "v6/results/cfra/paired_branches.csv",
            "source_sha256": actual,
            "splits": ["representation_train", "development"],
            "rows": int(len(train)),
            "clusters": int(train.groupby(["network", "seed"]).ngroups),
            "target": TARGET,
        },
        "frozen_estimator": {
            "path": "sealed/frozen_estimator.pkl",
            "sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
            "spec": "ExtraTreesRegressor(n_estimators=300, random_state=%d)" % SEED,
            "features": "feature_vector only; no history, no graph, no JEPA",
            "note": "This exact object will be applied to the sealed data. It is not refitted.",
        },
        "evaluation_population": {
            "seed_block": "untouched_test, seeds 9801-9900",
            "networks": ["moscow", "cologne8", "ingolstadt21"],
            "conditions": [
                "normal", "packet_loss", "sensor_delay_noise",
                "lane_closure", "emergency_vehicle", "demand_shock",
            ],
            "status_when_locked": "not generated; no paired branch exists for these seeds",
            "protocol_note": (
                "V6 reserved this block for a confirmatory traffic grid that was "
                "never opened and defined no paired-branch configuration for it. "
                "This is a new operation for a new question, mirroring the "
                "validation split's configuration."
            ),
        },
        "predictions": {
            "primary": {
                "statement": "RMSE gain of the frozen estimator over the training-set mean, on the sealed block, is at least 10%.",
                "threshold": 0.10,
                "rationale": "10% is the Stage-B gate that terminated V4 (observed 5.34%) and V5 (observed 8.99%).",
                "decision_rule": "confirmed if the point estimate >= 0.10 AND the 95% cluster-bootstrap lower bound > 0",
            },
            "secondary_interval": {
                "statement": "The point estimate falls inside the validation-split 95% interval.",
                "interval": [0.1146, 0.1719],
                "source": "verify_fact2.py on the consumed validation split",
            },
            "falsification": (
                "A point estimate below 10% on the sealed block would show that the "
                "validation-split result did not generalise, and would weaken the "
                "claim that V4 and V5 failed for sample-size reasons rather than "
                "model-class reasons. It would not be reinterpreted post hoc."
            ),
        },
        "fixed_analysis": {
            "reference_predictor": "mean of the training target, as in the V4/V5 Stage-B gate",
            "statistic": "1 - RMSE_model / RMSE_mean",
            "cluster_unit": "(network, seed)",
            "bootstrap_replicates": 2000,
            "bootstrap_seed": SEED,
            "confidence_level": 0.95,
            "secondary_endpoints": ["benefit AUROC", "per-network breakdown"],
            "no_other_model_will_be_fitted": True,
        },
        "attrition_policy": (
            "Failed or incomplete branch pairs are reported as attrition and are "
            "not silently dropped. If more than 10% of scheduled pairs fail, the "
            "result is reported as inconclusive."
        ),
    }
    LOCK_PATH.write_text(json.dumps(lock, indent=2), encoding="utf-8")

    print(f"frozen estimator: {MODEL_PATH}")
    print(f"  sha256 {lock['frozen_estimator']['sha256'][:16]}...")
    print(f"  fitted on {len(train)} rows / {lock['training']['clusters']} clusters")
    print(f"prediction lock: {LOCK_PATH}")
    print(f"  primary: RMSE gain >= 10% on seeds 9801-9900")
    print(f"  secondary: point estimate inside [11.46%, 17.19%]")


if __name__ == "__main__":
    main()
