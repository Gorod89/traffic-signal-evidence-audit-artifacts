"""D+ step 1: read all 94 primary_search_v2 trial JSONs into tidy tables.

Read-only with respect to v8/.  Every output goes to this directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
V8 = Path(
    os.environ.get("V8_SOURCE_ROOT", HERE.parents[1] / "external" / "v8")
).resolve()
TRIALS = V8 / "results" / "primary_search_v2" / "trials"

CORRUPTIONS = ("mcar", "burst", "spatial", "mnar")


def load_all() -> list[dict]:
    records = []
    for path in sorted(TRIALS.glob("*.json")):
        with path.open(encoding="utf-8") as stream:
            records.append((path.name, json.load(stream)))
    return records


def main() -> None:
    raw = load_all()
    print(f"trial files: {len(raw)}")

    # ---- schema census -------------------------------------------------
    top_keys: dict[str, int] = {}
    for _, doc in raw:
        for key in doc:
            top_keys[key] = top_keys.get(key, 0) + 1
    print("\nTOP-LEVEL KEYS (count / 94):")
    for key, count in sorted(top_keys.items()):
        print(f"  {key:34s} {count}")

    metric_keys: dict[str, int] = {}
    for _, doc in raw:
        for key in doc.get("metrics", {}):
            metric_keys[key] = metric_keys.get(key, 0) + 1
    print("\nmetrics.* KEYS:")
    for key, count in sorted(metric_keys.items()):
        print(f"  {key:34s} {count}")

    # ---- candidate-level table -----------------------------------------
    rows = []
    for name, doc in raw:
        met = doc.get("metrics", {}) or {}
        cfg = doc.get("config", {}) or {}
        corr = doc.get("corruption_h4_improvement", {}) or {}
        struct = doc.get("structural", {}) or {}
        cond = met.get("condition_improvements", {}) or {}
        net = met.get("network_improvements", {}) or {}
        row = {
            "file": name,
            "rung": doc.get("rung"),
            "candidate_id": doc.get("candidate_id"),
            "recipe_id": doc.get("recipe_id"),
            "status": doc.get("status"),
            "error": doc.get("error"),
            "model_seed": doc.get("model_seed"),
            "parameter_count": doc.get("parameter_count"),
            "encoder": cfg.get("encoder"),
            "paired_head": cfg.get("paired_head"),
            "width": cfg.get("width"),
            "depth": cfg.get("depth"),
            "dropout": cfg.get("dropout"),
            "activation": cfg.get("activation"),
            "action_dim": cfg.get("action_dim"),
            "max_actions": cfg.get("max_actions"),
            "overall_improvement": met.get("overall_improvement"),
            "h4_improvement": met.get("h4_improvement"),
            "overall_nrmse": met.get("overall_nrmse"),
            "h4_nrmse": met.get("h4_nrmse"),
            "worst_condition_improvement": met.get("worst_condition_improvement"),
            "worst_network_improvement": met.get("worst_network_improvement"),
            "groupcv_folds": met.get("groupcv_folds"),
            "action_permutation_degradation": doc.get("action_permutation_degradation"),
            "effective_rank_fraction": doc.get("effective_rank_fraction"),
            "equal_action_max_abs": struct.get("equal_action_max_abs"),
            "swap_antisymmetry_max_abs": struct.get("swap_antisymmetry_max_abs"),
            "peak_vram_mib": doc.get("peak_vram_mib"),
            "p95_batch1_ms": doc.get("p95_batch1_ms"),
            "wall_seconds": doc.get("wall_seconds"),
            "train_rows": doc.get("train_rows"),
            "tune_rows": doc.get("tune_rows"),
            "n_train_clusters": len(doc.get("train_clusters", []) or []),
            "n_folds_recorded": len(doc.get("fold_results", []) or []),
            "model_sha256": doc.get("model_sha256"),
        }
        for mode in CORRUPTIONS:
            row[f"corr_{mode}"] = corr.get(mode)
        vals = [corr.get(m) for m in CORRUPTIONS if corr.get(m) is not None]
        row["corr_min"] = min(vals) if vals else np.nan
        row["corr_mean"] = float(np.mean(vals)) if vals else np.nan
        for key, value in cond.items():
            row[f"cond_{key}"] = value
        for key, value in net.items():
            row[f"net_{key}"] = value
        rows.append(row)
    candidates = pd.DataFrame(rows)
    candidates.to_csv(HERE / "trials_candidate_level.csv", index=False, encoding="utf-8")

    # ---- fold-level table ----------------------------------------------
    fold_rows = []
    for name, doc in raw:
        cfg = doc.get("config", {}) or {}
        for fold in doc.get("fold_results", []) or []:
            met = fold.get("metrics", {}) or {}
            corr = fold.get("corruption_h4_improvement", {}) or {}
            struct = fold.get("structural", {}) or {}
            frow = {
                "rung": doc.get("rung"),
                "candidate_id": doc.get("candidate_id"),
                "encoder": cfg.get("encoder"),
                "paired_head": cfg.get("paired_head"),
                "width": cfg.get("width"),
                "depth": cfg.get("depth"),
                "dropout": cfg.get("dropout"),
                "activation": cfg.get("activation"),
                "fold": fold.get("fold"),
                "status": fold.get("status"),
                "overall_improvement": met.get("overall_improvement"),
                "h4_improvement": met.get("h4_improvement"),
                "overall_nrmse": met.get("overall_nrmse"),
                "h4_nrmse": met.get("h4_nrmse"),
                "worst_condition_improvement": met.get("worst_condition_improvement"),
                "worst_network_improvement": met.get("worst_network_improvement"),
                "action_permutation_degradation": fold.get("action_permutation_degradation"),
                "effective_rank_fraction": fold.get("effective_rank_fraction"),
                "equal_action_max_abs": struct.get("equal_action_max_abs"),
                "swap_antisymmetry_max_abs": struct.get("swap_antisymmetry_max_abs"),
                "peak_vram_mib": fold.get("peak_vram_mib"),
                "p95_batch1_ms": fold.get("p95_batch1_ms"),
                "n_epochs": len(fold.get("training_curve", []) or []),
                "final_train_loss": (
                    (fold.get("training_curve") or [{}])[-1].get("mean_train_loss")
                    if fold.get("training_curve")
                    else None
                ),
            }
            for mode in CORRUPTIONS:
                frow[f"corr_{mode}"] = corr.get(mode)
            for key, value in (met.get("condition_improvements") or {}).items():
                frow[f"cond_{key}"] = value
            for key, value in (met.get("network_improvements") or {}).items():
                frow[f"net_{key}"] = value
            fold_rows.append(frow)
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(HERE / "trials_fold_level.csv", index=False, encoding="utf-8")

    print(f"\ncandidate rows: {len(candidates)}  fold rows: {len(folds)}")
    print("\nrung / status:")
    print(candidates.groupby(["rung", "status"]).size().to_string())
    print("\nepochs per fold by rung:")
    print(folds.groupby("rung")["n_epochs"].agg(["min", "max", "mean"]).to_string())

    r1 = set(candidates.loc[candidates.rung == "R1", "candidate_id"])
    r2 = set(candidates.loc[candidates.rung == "R2", "candidate_id"])
    promo = json.loads(
        (V8 / "results" / "primary_search_v2" / "PROMOTION_R1.json").read_text(encoding="utf-8")
    )
    promoted = list(promo["promoted"])
    print(f"\nR1 trials: {len(r1)}  R2 trials: {len(r2)}  promoted at R1: {len(promoted)}")
    print(f"promoted but MISSING an R2 trial: {[c for c in promoted if c not in r2]}")
    print(f"R2 trials not in promoted list: {sorted(r2 - set(promoted))}")

    print("\nTOP 12 R2 by overall_improvement:")
    cols = [
        "candidate_id", "encoder", "paired_head", "width", "depth",
        "overall_improvement", "h4_improvement", "worst_condition_improvement", "corr_min",
        "action_permutation_degradation",
    ]
    print(
        candidates[candidates.rung == "R2"]
        .sort_values("overall_improvement", ascending=False)[cols]
        .to_string(index=False)
    )
    print("\nTOP 10 R1 by overall_improvement:")
    print(
        candidates[candidates.rung == "R1"]
        .sort_values("overall_improvement", ascending=False)
        .head(10)[cols]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
