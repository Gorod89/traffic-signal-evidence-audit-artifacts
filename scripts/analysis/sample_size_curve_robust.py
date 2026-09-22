"""Re-run the sample-size curve with enough resamples to report quantiles.

The original curve used twelve resamples per budget. At that count the share of
resamples clearing the gate moves in steps of 8.3 percentage points and cannot
be read as a probability, and a one-SD ribbon on twelve draws is not a stable
summary. A reviewer is right to object. This repeats the identical analysis with
200 resamples and reports the median and the 5th-95th percentile band instead.

Nothing else changes: same model class, same data, same fold protocol, same
cluster unit. Output goes to a separate file so the original artifact -- which
the claim audit checks against the published numbers -- stays intact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from integrity import write_json_atomic  # noqa: E402
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

REPO = Path(__file__).resolve().parents[2]
PAIRED = REPO / "artifacts" / "derived" / "v6" / "paired_branches.csv"
EXPECTED_SHA = "24231bba4eadff8051da0af7705ff429d86c93ec10c554ede9247836aea55d8e"
OUT = REPO / "artifacts" / "derived" / "phase0" / "sample_size_curve_robust.json"

TARGET = "gain_h4"
GATE = 0.10
SEED = 20260813
REPEATS = 200
TRAIN_GRID = [10, 20, 30, 40, 60, 80, 120, 160]


def parse(series: pd.Series) -> np.ndarray:
    return np.vstack([np.asarray(json.loads(v), dtype=np.float64).ravel() for v in series])


def main() -> None:
    actual = hashlib.sha256(PAIRED.read_bytes()).hexdigest()
    if actual != EXPECTED_SHA:
        raise SystemExit(f"input hash changed: {actual}")

    df = pd.read_csv(PAIRED)
    df["cluster"] = df.network + "|" + df.seed.astype(str)
    pool = df[df.split.isin(("representation_train", "development"))].reset_index(drop=True)
    evalu = df[df.split == "validation"].reset_index(drop=True)

    x_pool = parse(pool.feature_vector)
    y_pool = pool[TARGET].to_numpy(dtype=np.float64)
    clusters_pool = pool.cluster.to_numpy()
    x_eval = parse(evalu.feature_vector)
    y_eval = evalu[TARGET].to_numpy(dtype=np.float64)

    unique = np.unique(clusters_pool)
    rng = np.random.default_rng(SEED)
    curve = []

    for budget in TRAIN_GRID:
        if budget > unique.size:
            continue
        gains = []
        for rep in range(REPEATS):
            picked = rng.choice(unique, size=budget, replace=False)
            mask = np.isin(clusters_pool, picked)
            model = ExtraTreesRegressor(
                n_estimators=300, random_state=SEED + rep, n_jobs=-1
            )
            model.fit(x_pool[mask], y_pool[mask])
            pred = model.predict(x_eval)
            reference = float(np.mean(y_pool[mask]))
            model_rmse = float(np.sqrt(np.mean((pred - y_eval) ** 2)))
            mean_rmse = float(np.sqrt(np.mean((reference - y_eval) ** 2)))
            gains.append(1.0 - model_rmse / mean_rmse)
        g = np.asarray(gains)
        row = {
            "train_clusters": int(budget),
            "repeats": REPEATS,
            "median": float(np.median(g)),
            "mean": float(g.mean()),
            "q05": float(np.percentile(g, 5)),
            "q95": float(np.percentile(g, 95)),
            "sd": float(g.std(ddof=1)),
            "share_clearing_gate": float(np.mean(g >= GATE)),
        }
        curve.append(row)
        print(
            "clusters %3d: median %6.2f%%  [q05 %6.2f%%, q95 %6.2f%%]  "
            "clears gate in %5.1f%% of %d resamples"
            % (budget, 100 * row["median"], 100 * row["q05"],
               100 * row["q95"], 100 * row["share_clearing_gate"], REPEATS)
        )

    report = {
        "input_sha256": actual,
        "target": TARGET,
        "gate": GATE,
        "repeats": REPEATS,
        "supersedes": "results/sample_size_curve.json (12 resamples)",
        "note": "Identical analysis at 200 resamples; quantile band replaces the "
                "one-SD ribbon and the clearance share is now resolvable to 0.5%.",
        "training_curve": curve,
    }
    write_json_atomic(OUT, report)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
