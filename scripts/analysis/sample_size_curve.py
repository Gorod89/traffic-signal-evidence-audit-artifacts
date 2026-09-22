"""Phase 0 / Analysis B: does training-set size explain the V4 and V5 STOP decisions?

V4 and V5 both terminated at the Stage-B verifier gate, which required a >=10%
RMSE improvement over a mean predictor.  V4 observed 5.34%, V5 observed 8.99%.
Both were interpreted as method failures and answered by building a new
architecture.

This script tests a competing explanation: that the binding constraint was the
number of independent paired route-seed clusters available for fitting, not the
model class.  It holds the model class fixed (an ordinary ExtraTrees regressor,
no history, no graph, no JEPA) and varies only the number of training clusters,
evaluating every fit on the same held-out validation population.

Two curves are produced:
  1. RMSE gain vs number of TRAINING clusters -- how much data the estimator
     needs before it clears the gate at all;
  2. CI half-width vs number of EVALUATION clusters -- how much data the
     decision needs before the gate can be cleared with confidence.

Read-only with respect to the repository.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from integrity import write_json_atomic
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

REPO = Path(__file__).resolve().parents[2]
PAIRED = REPO / "artifacts" / "derived" / "v6" / "paired_branches.csv"
EXPECTED_SHA = "24231bba4eadff8051da0af7705ff429d86c93ec10c554ede9247836aea55d8e"
OUT = REPO / "artifacts" / "derived" / "phase0" / "sample_size_curve.json"

TARGET = "gain_h4"
RMSE_GAIN_THRESHOLD = 0.10
SEED = 20260805
REPEATS = 12

# Training-cluster budgets. 30 reproduces the V4/V5 scale; 160 is all of V6.1.
TRAIN_CLUSTER_GRID = [10, 20, 30, 40, 60, 80, 120, 160]
# Evaluation-cluster budgets for the confidence-width curve.
EVAL_CLUSTER_GRID = [15, 30, 60, 100, 150, 200, 300]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_vector_column(series: pd.Series) -> np.ndarray:
    rows = [np.asarray(json.loads(v), dtype=np.float64).ravel() for v in series]
    return np.vstack(rows)


def rmse_gain(pred: np.ndarray, truth: np.ndarray, reference: float) -> float:
    model = float(np.sqrt(np.mean((pred - truth) ** 2)))
    mean = float(np.sqrt(np.mean((reference - truth) ** 2)))
    return 1.0 - model / mean


def main() -> None:
    actual = sha256_file(PAIRED)
    if actual != EXPECTED_SHA:
        raise SystemExit(f"input hash changed: {actual}")

    df = pd.read_csv(PAIRED)
    df["cluster"] = df.network + "|" + df.seed.astype(str)

    train_pool = df[df.split.isin(("representation_train", "development"))].reset_index(drop=True)
    evalu = df[df.split == "validation"].reset_index(drop=True)

    x_train_all = parse_vector_column(train_pool.feature_vector)
    y_train_all = train_pool[TARGET].to_numpy(dtype=np.float64)
    train_clusters = train_pool.cluster.to_numpy()

    x_eval = parse_vector_column(evalu.feature_vector)
    y_eval = evalu[TARGET].to_numpy(dtype=np.float64)
    eval_clusters = evalu.cluster.to_numpy()

    unique_train = np.unique(train_clusters)
    unique_eval = np.unique(eval_clusters)
    rng = np.random.default_rng(SEED)

    # ---- Curve 1: RMSE gain as a function of training clusters -------------
    training_curve = []
    for budget in TRAIN_CLUSTER_GRID:
        if budget > unique_train.size:
            continue
        gains = []
        rows_used = []
        for rep in range(REPEATS):
            picked = rng.choice(unique_train, size=budget, replace=False)
            mask = np.isin(train_clusters, picked)
            x_sub, y_sub = x_train_all[mask], y_train_all[mask]
            model = ExtraTreesRegressor(
                n_estimators=300, random_state=SEED + rep, n_jobs=-1
            )
            model.fit(x_sub, y_sub)
            pred = model.predict(x_eval)
            gains.append(rmse_gain(pred, y_eval, float(np.mean(y_sub))))
            rows_used.append(int(mask.sum()))
        gains_arr = np.asarray(gains)
        training_curve.append(
            {
                "train_clusters": int(budget),
                "train_rows_mean": float(np.mean(rows_used)),
                "rmse_gain_mean": float(gains_arr.mean()),
                "rmse_gain_sd": float(gains_arr.std(ddof=1)),
                "rmse_gain_min": float(gains_arr.min()),
                "rmse_gain_max": float(gains_arr.max()),
                "repeats": REPEATS,
                "fraction_of_repeats_clearing_gate": float(
                    np.mean(gains_arr >= RMSE_GAIN_THRESHOLD)
                ),
            }
        )
        print(
            "train clusters %3d (%4.0f rows): rmse_gain %6.2f%% +/- %4.2f  "
            "[%.2f%%, %.2f%%]  cleared %3.0f%% of repeats"
            % (
                budget,
                np.mean(rows_used),
                100 * gains_arr.mean(),
                100 * gains_arr.std(ddof=1),
                100 * gains_arr.min(),
                100 * gains_arr.max(),
                100 * np.mean(gains_arr >= RMSE_GAIN_THRESHOLD),
            )
        )

    # ---- Curve 2: CI width as a function of evaluation clusters ------------
    full_model = ExtraTreesRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
    full_model.fit(x_train_all, y_train_all)
    full_pred = full_model.predict(x_eval)
    reference = float(np.mean(y_train_all))
    index_by_cluster = {c: np.flatnonzero(eval_clusters == c) for c in unique_eval}

    evaluation_curve = []
    for budget in EVAL_CLUSTER_GRID:
        if budget > unique_eval.size:
            continue
        lowers, uppers, points = [], [], []
        for rep in range(REPEATS):
            chosen = rng.choice(unique_eval, size=budget, replace=False)
            idx = np.concatenate([index_by_cluster[c] for c in chosen])
            points.append(rmse_gain(full_pred[idx], y_eval[idx], reference))
            draws = []
            for _ in range(600):
                boot = rng.choice(chosen, size=budget, replace=True)
                bidx = np.concatenate([index_by_cluster[c] for c in boot])
                draws.append(rmse_gain(full_pred[bidx], y_eval[bidx], reference))
            lowers.append(float(np.percentile(draws, 2.5)))
            uppers.append(float(np.percentile(draws, 97.5)))
        evaluation_curve.append(
            {
                "eval_clusters": int(budget),
                "rmse_gain_mean": float(np.mean(points)),
                "ci_lower_mean": float(np.mean(lowers)),
                "ci_upper_mean": float(np.mean(uppers)),
                "ci_halfwidth_mean": float((np.mean(uppers) - np.mean(lowers)) / 2),
                "fraction_with_lower_bound_above_gate": float(
                    np.mean(np.asarray(lowers) >= RMSE_GAIN_THRESHOLD)
                ),
                "repeats": REPEATS,
            }
        )
        print(
            "eval clusters %3d: gain %6.2f%%  CI [%5.2f%%, %5.2f%%]  "
            "halfwidth %4.2f pp  LB clears gate in %3.0f%% of repeats"
            % (
                budget,
                100 * np.mean(points),
                100 * np.mean(lowers),
                100 * np.mean(uppers),
                100 * (np.mean(uppers) - np.mean(lowers)) / 2,
                100
                * np.mean(np.asarray(lowers) >= RMSE_GAIN_THRESHOLD),
            )
        )

    report = {
        "input": "artifacts/derived/v6/paired_branches.csv",
        "input_sha256": actual,
        "target": TARGET,
        "model": "ExtraTreesRegressor(n_estimators=300) on feature_vector; no history, no graph, no JEPA",
        "gate_threshold_rmse_gain": RMSE_GAIN_THRESHOLD,
        "gate_source": "V4/V5 stage_b_verifier.json",
        "training_curve": training_curve,
        "evaluation_curve": evaluation_curve,
        "historical_anchors": {
            "v4": {"observed_rmse_gain": 0.0534, "paired_rows": 360, "decision": "STOP"},
            "v5": {"observed_rmse_gain": 0.0899, "paired_rows": 420, "decision": "STOP"},
        },
        "evidence_class": "retrospective_diagnostic_not_confirmatory",
    }
    write_json_atomic(OUT, report)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
