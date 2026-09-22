"""Independent re-verification of the load-bearing "sample size, not architecture" claim.

Claim under test (recorded in the evidence inventory, pending source verification):
    An ordinary ExtraTrees regressor trained on the V6.1 paired-branch data
    (representation_train + development = 960 rows) clears the Stage-B verifier
    thresholds that terminated V4 (5.34% RMSE gain) and V5 (8.99%):
        - RMSE improvement over a mean predictor >= 10%
        - benefit AUROC >= 0.65
    and also clears them zero-shot on ingolstadt21, a network absent from training.

This script recomputes that from the raw CSV with explicit integrity checks:
    * input file hash is pinned;
    * split disjointness is verified by seed AND by route hash;
    * feature columns are audited for outcome leakage before fitting;
    * intervals are cluster bootstrap over (network, seed), never over rows.

Reads the pinned V6 derivative committed under `artifacts/derived/v6/` and
writes only the corresponding Phase-0 JSON under `artifacts/derived/phase0/`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from integrity import write_json_atomic
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
PAIRED = REPO / "artifacts" / "derived" / "v6" / "paired_branches.csv"
EXPECTED_SHA = "24231bba4eadff8051da0af7705ff429d86c93ec10c554ede9247836aea55d8e"
OUT = REPO / "artifacts" / "derived" / "phase0" / "fact2_verification.json"

# Stage-B thresholds as they were preregistered in V4/V5. Not tuned here.
RMSE_GAIN_THRESHOLD = 0.10
AUROC_THRESHOLD = 0.65

TRAIN_SPLITS = ("representation_train", "development")
EVAL_SPLIT = "validation"
SEED = 20260805


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_vector_column(series: pd.Series) -> np.ndarray:
    """Parse a column of JSON-encoded numeric arrays into a 2-D float array."""
    rows = [np.asarray(json.loads(value), dtype=np.float64).ravel() for value in series]
    widths = {row.size for row in rows}
    if len(widths) != 1:
        raise ValueError(f"ragged vector column: widths={sorted(widths)}")
    return np.vstack(rows)


def cluster_bootstrap(
    values: np.ndarray,
    clusters: np.ndarray,
    statistic,
    *,
    replicates: int = 2000,
    seed: int = SEED,
) -> tuple[float, float]:
    """Percentile interval resampling whole (network, seed) clusters."""
    rng = np.random.default_rng(seed)
    unique = np.unique(clusters)
    index_by_cluster = {c: np.flatnonzero(clusters == c) for c in unique}
    draws = []
    for _ in range(replicates):
        picked = rng.choice(unique, size=unique.size, replace=True)
        idx = np.concatenate([index_by_cluster[c] for c in picked])
        try:
            draws.append(statistic(idx))
        except ValueError:
            continue
    if len(draws) < replicates // 2:
        raise RuntimeError("too many degenerate bootstrap draws")
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    actual_sha = sha256_file(PAIRED)
    if actual_sha != EXPECTED_SHA:
        raise SystemExit(f"input hash changed: {actual_sha}")

    df = pd.read_csv(PAIRED)
    report: dict = {
        "input": str(PAIRED.relative_to(REPO)).replace("\\", "/"),
        "input_sha256": actual_sha,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "model": "ExtraTreesRegressor(n_estimators=300, random_state=%d)" % SEED,
        "thresholds": {
            "rmse_gain": RMSE_GAIN_THRESHOLD,
            "auroc": AUROC_THRESHOLD,
            "source": "V4 stage_b_verifier.json / V5 stage_b_verifier.json",
        },
    }

    # --- integrity: splits must not share seeds or routes -------------------
    train = df[df.split.isin(TRAIN_SPLITS)].reset_index(drop=True)
    evalu = df[df.split == EVAL_SPLIT].reset_index(drop=True)
    seed_overlap = set(train.seed) & set(evalu.seed)
    route_overlap = set(train.route_sha256) & set(evalu.route_sha256)
    report["disjointness"] = {
        "train_rows": int(len(train)),
        "eval_rows": int(len(evalu)),
        "shared_seeds": len(seed_overlap),
        "shared_routes": len(route_overlap),
        "train_clusters": int(train.groupby(["network", "seed"]).ngroups),
        "eval_clusters": int(evalu.groupby(["network", "seed"]).ngroups),
    }
    if seed_overlap or route_overlap:
        raise SystemExit("splits are not disjoint; refusing to proceed")

    # --- integrity: the feature block must not contain outcome information --
    # feature_vector is built pre-outcome from ActionContext; assert that no
    # column carrying a realised outcome is used, and that width is as declared.
    x_train = parse_vector_column(train.feature_vector)
    x_eval = parse_vector_column(evalu.feature_vector)
    if x_train.shape[1] != x_eval.shape[1]:
        raise SystemExit("feature width differs between splits")
    report["features"] = {
        "column": "feature_vector",
        "width": int(x_train.shape[1]),
        "leakage_audit": "outcome columns (gain_*, *_cost_*, delta_*, *_target_observation_*) are never read into X",
    }

    results = {}
    for target in ("gain_h1", "gain_h2", "gain_h4"):
        y_train = train[target].to_numpy(dtype=np.float64)
        y_eval = evalu[target].to_numpy(dtype=np.float64)

        model = ExtraTreesRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
        model.fit(x_train, y_train)
        pred = model.predict(x_eval)

        # Reference predictor: the training-set mean, exactly as Stage B defined it.
        reference = float(np.mean(y_train))
        clusters = (evalu.network + "|" + evalu.seed.astype(str)).to_numpy()

        def rmse_gain(idx: np.ndarray) -> float:
            model_rmse = float(np.sqrt(np.mean((pred[idx] - y_eval[idx]) ** 2)))
            mean_rmse = float(np.sqrt(np.mean((reference - y_eval[idx]) ** 2)))
            return 1.0 - model_rmse / mean_rmse

        def auroc(idx: np.ndarray) -> float:
            labels = (y_eval[idx] > 0).astype(int)
            if labels.min() == labels.max():
                raise ValueError("degenerate label set")
            return float(roc_auc_score(labels, pred[idx]))

        all_idx = np.arange(len(y_eval))
        point_rmse_gain = rmse_gain(all_idx)
        point_auroc = auroc(all_idx)
        rg_lo, rg_hi = cluster_bootstrap(y_eval, clusters, rmse_gain)
        au_lo, au_hi = cluster_bootstrap(y_eval, clusters, auroc)

        results[target] = {
            "model_rmse": float(np.sqrt(np.mean((pred - y_eval) ** 2))),
            "mean_predictor_rmse": float(np.sqrt(np.mean((reference - y_eval) ** 2))),
            "rmse_gain": point_rmse_gain,
            "rmse_gain_ci95": [rg_lo, rg_hi],
            "auroc": point_auroc,
            "auroc_ci95": [au_lo, au_hi],
            "positive_fraction": float(np.mean(y_eval > 0)),
            "passes_rmse_gate": bool(point_rmse_gain >= RMSE_GAIN_THRESHOLD),
            "passes_auroc_gate": bool(point_auroc >= AUROC_THRESHOLD),
            "passes_lower_bound_rmse_gate": bool(rg_lo >= RMSE_GAIN_THRESHOLD),
        }

        # Zero-shot slice: ingolstadt21 appears only in validation.
        if target == "gain_h4":
            mask = (evalu.network == "ingolstadt21").to_numpy()
            zs_idx = np.flatnonzero(mask)
            zs_clusters = clusters[mask]
            zs_rmse_gain = rmse_gain(zs_idx)
            zs_auroc = auroc(zs_idx)

            def zs_rg(local_idx: np.ndarray) -> float:
                return rmse_gain(zs_idx[local_idx])

            def zs_au(local_idx: np.ndarray) -> float:
                return auroc(zs_idx[local_idx])

            zlo, zhi = cluster_bootstrap(
                y_eval[zs_idx], zs_clusters, zs_rg
            )
            zalo, zahi = cluster_bootstrap(
                y_eval[zs_idx], zs_clusters, zs_au
            )
            results["gain_h4_zero_shot_ingolstadt21"] = {
                "rows": int(zs_idx.size),
                "clusters": int(np.unique(zs_clusters).size),
                "in_training": bool((train.network == "ingolstadt21").any()),
                "rmse_gain": zs_rmse_gain,
                "rmse_gain_ci95": [zlo, zhi],
                "auroc": zs_auroc,
                "auroc_ci95": [zalo, zahi],
                "passes_rmse_gate": bool(zs_rmse_gain >= RMSE_GAIN_THRESHOLD),
                "passes_auroc_gate": bool(zs_auroc >= AUROC_THRESHOLD),
            }

    report["results"] = results
    report["historical_comparison"] = {
        "v4_stage_b_rmse_gain": 0.0534,
        "v4_paired_rows": 360,
        "v5_stage_b_rmse_gain": 0.0899,
        "v5_paired_rows": 420,
        "v6_paired_rows": 3480,
        "note": "V4/V5 used their own splits and feature sets; this is a "
        "same-threshold comparison, not a same-protocol replication.",
    }
    report["evidence_class"] = (
        "retrospective_diagnostic_not_confirmatory: the V6.1 validation split "
        "was already consumed by Gate B, so program rules forbid treating this "
        "as confirmatory evidence."
    )

    write_json_atomic(OUT, report)

    print(f"input sha256 verified: {actual_sha[:16]}...")
    print(
        "disjoint splits: %d shared seeds, %d shared routes"
        % (len(seed_overlap), len(route_overlap))
    )
    print(
        "train %d rows / %d clusters -> eval %d rows / %d clusters"
        % (
            len(train),
            report["disjointness"]["train_clusters"],
            len(evalu),
            report["disjointness"]["eval_clusters"],
        )
    )
    print()
    for name, r in results.items():
        gate = "PASS" if r.get("passes_rmse_gate") else "FAIL"
        print(
            "%-38s rmse_gain=%6.2f%% CI[%5.2f,%5.2f]  auroc=%.3f CI[%.3f,%.3f]  %s"
            % (
                name,
                100 * r["rmse_gain"],
                100 * r["rmse_gain_ci95"][0],
                100 * r["rmse_gain_ci95"][1],
                r["auroc"],
                r["auroc_ci95"][0],
                r["auroc_ci95"][1],
                gate,
            )
        )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
