"""PACE Phase 2, task 1: the effect-model ladder (hypothesis P3).

Component 4 of the method is deliberately low-capacity: the Phase 0 diagnosis
found that history, topology and latent prediction add under 2 percentage points
over a matched base, so a deep model buys overfitting at the same accuracy
ceiling.  This script tests that claim where it matters -- on 1800 sealed pairs
from seeds that did not exist when any of these models could have been fitted.

Five rungs, all fitted on the same 960 training rows (160 clusters) and nothing
else:

    0  zero effect                  predict 0
    1  training mean                the V4/V5 Stage-B reference predictor
    2  ridge, context_observation   10 features
    3  ridge, feature_vector        81 features
    4  ExtraTrees, feature_vector   81 features

Reported on the sealed block: benefit AUROC (P3: does the model rank beneficial
interventions above harmful ones better than chance?) and RMSE, both with
cluster bootstrap intervals over (network, seed).  The same metrics are reported
on the calibration split, because model selection for the policy in
`selective_policy.py` is made there and must be visibly separate from the sealed
numbers.

Ridge penalties come from RidgeCV over a fixed grid, resolved by internal
generalised cross-validation on the training rows only.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from phase2_common import (
    BOOTSTRAP_REPLICATES,
    CONFIDENCE,
    RESULTS,
    SEED,
    cluster_bootstrap_ci,
    load_csv_blocks,
    load_sealed_block,
    write_json,
)

ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
OUT = RESULTS / "model_ladder.json"


def build_models(train):
    """Fit every rung on the training block. Returns name -> predict callable."""
    mean_ref = float(train.y.mean())

    ridge_obs = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    ridge_obs.fit(train.obs, train.y)

    ridge_con = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    ridge_con.fit(train.con, train.y)

    trees_con = ExtraTreesRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
    trees_con.fit(train.con, train.y)

    return [
        ("0. zero effect", None, lambda b: np.zeros(b.n_rows)),
        ("1. training mean", None, lambda b: np.full(b.n_rows, mean_ref)),
        ("2. ridge, context_observation", 10, lambda b: ridge_obs.predict(b.obs)),
        ("3. ridge, feature_vector", 81, lambda b: ridge_con.predict(b.con)),
        ("4. extratrees, feature_vector", 81, lambda b: trees_con.predict(b.con)),
    ], {
        "training_mean": mean_ref,
        "ridge_observation_alpha": float(ridge_obs[-1].alpha_),
        "ridge_contrast_alpha": float(ridge_con[-1].alpha_),
        "extratrees": {"n_estimators": 300, "random_state": SEED},
    }


def score(pred: np.ndarray, block) -> dict:
    y, clusters = block.y, block.clusters

    def rmse(idx: np.ndarray) -> float:
        return float(np.sqrt(np.mean((pred[idx] - y[idx]) ** 2)))

    def auroc(idx: np.ndarray) -> float:
        labels = (y[idx] > 0).astype(int)
        if labels.min() == labels.max():
            raise ValueError("single-class replicate")
        if np.allclose(pred[idx], pred[idx][0]):
            return 0.5  # a constant predictor ranks at chance by definition
        return float(roc_auc_score(labels, pred[idx]))

    all_idx = np.arange(y.size)
    constant = bool(np.allclose(pred, pred[0]))
    r = rmse(all_idx)
    r_lo, r_hi, _ = cluster_bootstrap_ci(rmse, clusters)
    a = auroc(all_idx)
    if constant:
        a_lo, a_hi, rej = 0.5, 0.5, 0
    else:
        a_lo, a_hi, rej = cluster_bootstrap_ci(auroc, clusters)
    return {
        "rmse": r,
        "rmse_ci95": [r_lo, r_hi],
        "auroc": a,
        "auroc_ci95": [a_lo, a_hi],
        "auroc_beats_chance": bool((not constant) and a_lo is not None and a_lo > 0.5),
        "degenerate_replicates": rej,
        "constant_predictor": constant,
    }


def main() -> None:
    train, calib, paired_sha = load_csv_blocks()
    sealed, sealed_prov = load_sealed_block()

    print(
        f"train {train.n_rows} rows / {train.n_clusters} clusters   "
        f"calib {calib.n_rows} / {calib.n_clusters}   "
        f"sealed {sealed.n_rows} / {sealed.n_clusters}"
    )
    print(f"sealed target: mean {sealed.y.mean():.2f}  sd {sealed.y.std(ddof=1):.2f}  "
          f"beneficial {(sealed.y > 0).mean():.3f}  harmful {(sealed.y < 0).mean():.3f}\n")

    models, hyper = build_models(train)

    ladder = []
    for name, width, predict in models:
        on_sealed = score(predict(sealed), sealed)
        on_calib = score(predict(calib), calib)
        ladder.append(
            {"rung": name, "n_features": width, "sealed": on_sealed, "calibration": on_calib}
        )
        print(
            f"{name:<32} sealed rmse={on_sealed['rmse']:8.2f} "
            f"auroc={on_sealed['auroc']:.3f} CI[{on_sealed['auroc_ci95'][0]:.3f},"
            f"{on_sealed['auroc_ci95'][1]:.3f}] "
            f"| calib rmse={on_calib['rmse']:8.2f} auroc={on_calib['auroc']:.3f}"
        )

    # P3 is a claim about ranking, so it is judged on AUROC, not RMSE.
    p3 = [r for r in ladder if r["n_features"] is not None and r["sealed"]["auroc_beats_chance"]]

    report = {
        "task": "PACE Phase 2 / task 1 -- effect-model ladder (P3)",
        "evidence_class": "held_out_evaluation_on_sealed_seeds_no_refit_no_tuning",
        "inputs": {
            "paired_archive": "v6/results/cfra/paired_branches.csv",
            "paired_archive_sha256": paired_sha,
            "sealed": sealed_prov,
        },
        "training": {
            "splits": ["representation_train", "development"],
            "rows": train.n_rows,
            "clusters": train.n_clusters,
            "target": "gain_h4",
        },
        "calibration": {"rows": calib.n_rows, "clusters": calib.n_clusters},
        "cluster_key": "network|seed",
        "interval": {
            "method": "cluster_percentile_bootstrap",
            "confidence_level": CONFIDENCE,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": SEED,
        },
        "hyperparameters": hyper,
        "sealed_target_summary": {
            "mean": float(sealed.y.mean()),
            "sd": float(sealed.y.std(ddof=1)),
            "beneficial_fraction": float((sealed.y > 0).mean()),
            "harmful_fraction": float((sealed.y < 0).mean()),
            "exact_zero_fraction": float((sealed.y == 0).mean()),
        },
        "ladder": ladder,
        "p3_verdict": {
            "claim": "a low-capacity model trained on paired data ranks effects better than chance",
            "rungs_with_auroc_lower_bound_above_half": [r["rung"] for r in p3],
            "supported": bool(p3),
        },
    }
    write_json(OUT, report)


if __name__ == "__main__":
    main()
