"""Phase 0 / Analyses A and E: the baseline ladder and the history/context decomposition.

Across V5, V6.1, V7.2 and V8 the same pattern recurs: a model looks strong
against a naive persistence or zero-effect reference and nearly identical to a
capacity-matched control that removes the mechanism being claimed.  Those
comparisons, however, were made in four different codebases on four different
splits, so a sceptical reader can attribute the pattern to protocol drift.

This script puts every rung of the ladder on ONE dataset, ONE split, ONE fold
protocol and ONE metric, so the comparison is internal and checkable:

    rung 0  zero effect                      (predict 0 for every pair)
    rung 1  training mean                    (the Stage-B reference predictor)
    rung 2  ridge on the current observation (10 features)
    rung 3  ridge on the full action contrast (81 features)
    rung 4  trees on the current observation
    rung 5  trees on the full action contrast
    rung 6  trees on contrast + 8-step history + action history

Rung 6 minus rung 5 is the incremental value of temporal history, measured
against a capacity-comparable control rather than against persistence.

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
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

REPO = Path(__file__).resolve().parents[2]
PAIRED = REPO / "artifacts" / "derived" / "v6" / "paired_branches.csv"
EXPECTED_SHA = "24231bba4eadff8051da0af7705ff429d86c93ec10c554ede9247836aea55d8e"
OUT = REPO / "artifacts" / "derived" / "phase0" / "baseline_ladder.json"

TARGET = "gain_h4"
SEED = 20260805
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_vector_column(series: pd.Series) -> np.ndarray:
    rows = [np.asarray(json.loads(v), dtype=np.float64).ravel() for v in series]
    return np.vstack(rows)


def cluster_bootstrap_ci(fn, clusters: np.ndarray, *, replicates: int = 2000, seed: int = SEED):
    rng = np.random.default_rng(seed)
    unique = np.unique(clusters)
    index_by_cluster = {c: np.flatnonzero(clusters == c) for c in unique}
    draws = []
    for _ in range(replicates):
        picked = rng.choice(unique, size=unique.size, replace=True)
        idx = np.concatenate([index_by_cluster[c] for c in picked])
        try:
            draws.append(fn(idx))
        except ValueError:
            continue
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    actual = sha256_file(PAIRED)
    if actual != EXPECTED_SHA:
        raise SystemExit(f"input hash changed: {actual}")

    df = pd.read_csv(PAIRED)
    df["cluster"] = df.network + "|" + df.seed.astype(str)
    train = df[df.split.isin(("representation_train", "development"))].reset_index(drop=True)
    evalu = df[df.split == "validation"].reset_index(drop=True)

    y_tr = train[TARGET].to_numpy(dtype=np.float64)
    y_ev = evalu[TARGET].to_numpy(dtype=np.float64)
    clusters = evalu.cluster.to_numpy()

    def block(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
        return np.hstack([parse_vector_column(frame[c]) for c in columns])

    obs_tr = block(train, ("context_observation",))
    obs_ev = block(evalu, ("context_observation",))
    con_tr = block(train, ("feature_vector",))
    con_ev = block(evalu, ("feature_vector",))
    hist_tr = block(train, ("feature_vector", "context_history", "context_action_history"))
    hist_ev = block(evalu, ("feature_vector", "context_history", "context_action_history"))

    mean_ref = float(np.mean(y_tr))
    denom_rmse = float(np.sqrt(np.mean((mean_ref - y_ev) ** 2)))

    def evaluate(name: str, pred: np.ndarray, width: int | None) -> dict:
        def gain(idx: np.ndarray) -> float:
            m = float(np.sqrt(np.mean((pred[idx] - y_ev[idx]) ** 2)))
            d = float(np.sqrt(np.mean((mean_ref - y_ev[idx]) ** 2)))
            return 1.0 - m / d

        def auroc(idx: np.ndarray) -> float:
            labels = (y_ev[idx] > 0).astype(int)
            if labels.min() == labels.max():
                raise ValueError("degenerate")
            if np.allclose(pred[idx], pred[idx][0]):
                return 0.5
            return float(roc_auc_score(labels, pred[idx]))

        all_idx = np.arange(len(y_ev))
        g = gain(all_idx)
        lo, hi = cluster_bootstrap_ci(gain, clusters)
        try:
            a = auroc(all_idx)
            alo, ahi = cluster_bootstrap_ci(auroc, clusters)
        except ValueError:
            a, alo, ahi = 0.5, 0.5, 0.5
        row = {
            "rung": name,
            "n_features": width,
            "rmse": float(np.sqrt(np.mean((pred - y_ev) ** 2))),
            "rmse_gain": g,
            "rmse_gain_ci95": [lo, hi],
            "auroc": a,
            "auroc_ci95": [alo, ahi],
        }
        print(
            "%-46s %4s  rmse=%7.2f  gain=%6.2f%% CI[%6.2f,%6.2f]  auroc=%.3f"
            % (name, width if width is not None else "-", row["rmse"], 100 * g, 100 * lo, 100 * hi, a)
        )
        return row

    print(f"train {len(train)} rows / {train.cluster.nunique()} clusters -> "
          f"eval {len(evalu)} rows / {evalu.cluster.nunique()} clusters")
    print(f"mean-predictor RMSE on eval: {denom_rmse:.2f}\n")

    ladder = []
    ladder.append(evaluate("0. zero effect", np.zeros_like(y_ev), None))
    ladder.append(evaluate("1. training mean (Stage-B reference)", np.full_like(y_ev, mean_ref), None))

    for name, xt, xe in (
        ("2. ridge, current observation", obs_tr, obs_ev),
        ("3. ridge, action contrast", con_tr, con_ev),
    ):
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
        model.fit(xt, y_tr)
        ladder.append(evaluate(name, model.predict(xe), xt.shape[1]))

    for name, xt, xe in (
        ("4. trees, current observation", obs_tr, obs_ev),
        ("5. trees, action contrast", con_tr, con_ev),
        ("6. trees, contrast + 8-step history", hist_tr, hist_ev),
    ):
        model = ExtraTreesRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
        model.fit(xt, y_tr)
        ladder.append(evaluate(name, model.predict(xe), xt.shape[1]))

    contrast = ladder[5]["rmse_gain"]
    with_history = ladder[6]["rmse_gain"]
    naive = ladder[0]["rmse_gain"]

    print()
    print("incremental value of 8-step history over a matched no-history control: "
          f"{100 * (with_history - contrast):+.3f} pp")
    print("apparent value of the same model against the zero-effect reference:    "
          f"{100 * (contrast - naive):+.3f} pp")

    report = {
        "input": "artifacts/derived/v6/paired_branches.csv",
        "input_sha256": actual,
        "target": TARGET,
        "reference_predictor": "training mean, as defined by the V4/V5 Stage-B gate",
        "train_rows": int(len(train)),
        "train_clusters": int(train.cluster.nunique()),
        "eval_rows": int(len(evalu)),
        "eval_clusters": int(evalu.cluster.nunique()),
        "mean_predictor_rmse": denom_rmse,
        "ladder": ladder,
        "history_increment_pp": float(100 * (with_history - contrast)),
        "zero_reference_increment_pp": float(100 * (contrast - naive)),
        "evidence_class": "retrospective_diagnostic_not_confirmatory",
    }
    write_json_atomic(OUT, report)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
