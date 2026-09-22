"""D+ step 2: verify comparability of primary_search_v2 vs the baseline panel.

Everything here is read-only against v8/.  Outputs land in this directory only.

Three things are established:
  (a) the fold partition recorded in SEARCH_LOCK.json is byte-identical to the
      partition grouped_cv_predictions() gives the baseline panel;
  (b) the published pooled baseline panel reproduces exactly;
  (c) the same baseline predictions re-aggregated with the SEARCH's convention
      (mean over folds for overall/h4, min-over-folds-of-min-over-groups for
      worst_condition / worst_network), which is what the trial JSONs contain.

Only (c) is legitimately comparable to the trial numbers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
V8 = Path(
    os.environ.get("V8_SOURCE_ROOT", HERE.parents[1] / "external" / "v8")
).resolve()
sys.path.insert(0, str(V8 / "src"))

from v8search.baselines import grouped_cv_predictions  # noqa: E402
from v8search.data import load_legacy_physical_data  # noqa: E402
from v8search.metrics import (  # noqa: E402
    improvement_bootstrap_interval,
    mse_improvement_over_zero,
    summarize_physical_metrics,
)
from v8search.search import TrainingRecipe, grouped_fold_ids  # noqa: E402


def main() -> None:
    arrays, manifest = load_legacy_physical_data(
        V8 / "data" / "representation_train_physical.csv", purpose="search"
    )
    print(f"rows={arrays.n_rows} clusters={len(np.unique(arrays.cluster))} "
          f"roles={sorted(set(arrays.role.tolist()))}")
    print(f"target_delta shape={arrays.target_delta.shape}")
    print(f"loaded_source_splits={manifest['loaded_source_splits']}")
    print(f"source_sha256={manifest['source_sha256']}")

    search_pool = arrays.subset("search_pool")
    assert search_pool.n_rows == arrays.n_rows, "search_pool != full frame"

    # ---- (a) fold partition identity ------------------------------------
    lock = json.loads(
        (V8 / "results" / "primary_search_v2" / "SEARCH_LOCK.json").read_text(encoding="utf-8")
    )
    recorded = {str(k): int(v) for k, v in lock["fold_assignment"].items()}
    fold_ids_search = grouped_fold_ids(search_pool.cluster, n_folds=5)  # default salt
    derived = {}
    for cluster, fold in zip(search_pool.cluster.tolist(), fold_ids_search.tolist()):
        derived[str(cluster)] = int(fold)
    print("\n(a) SEARCH_LOCK.fold_assignment == grouped_fold_ids(salt='v8-groupcv-v1'):",
          derived == recorded)

    fold_ids_baseline = grouped_fold_ids(arrays.cluster, n_folds=5, salt="v8-groupcv-v1")
    print("    baseline-panel fold ids identical to search fold ids:",
          bool(np.array_equal(fold_ids_baseline, fold_ids_search)))
    print("    rows per fold:", np.bincount(fold_ids_search).tolist())
    print("    clusters per fold:",
          [int(len(np.unique(arrays.cluster[fold_ids_search == f]))) for f in range(5)])

    # ---- predictions -----------------------------------------------------
    predictions = grouped_cv_predictions(arrays)
    recipe = TrainingRecipe()
    multiplier = np.sqrt(np.asarray(recipe.channel_weights))[None, None, :]
    weighted_target = arrays.target_delta * multiplier

    published = json.loads(
        (V8 / "results" / "baselines" / "SEARCH_POOL_BASELINES.json").read_text(encoding="utf-8")
    )
    published_by_name = {r["name"]: r for r in published["records"]}

    pooled_rows, foldwise_rows, perfold_rows = [], [], []
    for name, prediction in predictions.items():
        weighted_prediction = prediction * multiplier

        # (b) pooled -- the convention the published panel uses
        pooled = summarize_physical_metrics(
            weighted_prediction, weighted_target,
            condition=arrays.condition, network=arrays.network,
        ).as_dict()
        ref = published_by_name[name]["metrics"]
        agrees = all(
            np.isclose(pooled[k], ref[k], atol=1e-10)
            for k in ("overall_improvement", "h4_improvement",
                      "worst_condition_improvement", "worst_network_improvement")
        )
        ov = improvement_bootstrap_interval(
            weighted_prediction, weighted_target, arrays.cluster,
            replicates=2000, alpha=0.05, seed=8081,
        )
        h4 = improvement_bootstrap_interval(
            weighted_prediction, weighted_target, arrays.cluster,
            horizon_index=-1, replicates=2000, alpha=0.05, seed=8082,
        )
        pooled_rows.append({
            "name": name, "reproduces_published": agrees,
            "overall_improvement": pooled["overall_improvement"],
            "overall_lo95": ov[1], "overall_hi95": ov[2],
            "h4_improvement": pooled["h4_improvement"],
            "h4_lo95": h4[1], "h4_hi95": h4[2],
            "worst_condition_improvement": pooled["worst_condition_improvement"],
            "worst_network_improvement": pooled["worst_network_improvement"],
        })

        # (c) fold-wise -- the convention every trial JSON uses
        per_fold = []
        for fold in range(5):
            test = np.flatnonzero(fold_ids_search == fold)
            summary = summarize_physical_metrics(
                weighted_prediction[test], weighted_target[test],
                condition=arrays.condition[test], network=arrays.network[test],
            ).as_dict()
            per_fold.append(summary)
            perfold_rows.append({
                "name": name, "fold": fold,
                "overall_improvement": summary["overall_improvement"],
                "h4_improvement": summary["h4_improvement"],
                "worst_condition_improvement": summary["worst_condition_improvement"],
                "worst_network_improvement": summary["worst_network_improvement"],
            })
        foldwise_rows.append({
            "name": name,
            "overall_improvement": float(np.mean([s["overall_improvement"] for s in per_fold])),
            "h4_improvement": float(np.mean([s["h4_improvement"] for s in per_fold])),
            "overall_nrmse": float(np.mean([s["overall_nrmse"] for s in per_fold])),
            "h4_nrmse": float(np.mean([s["h4_nrmse"] for s in per_fold])),
            "worst_condition_improvement": float(
                min(s["worst_condition_improvement"] for s in per_fold)
            ),
            "worst_network_improvement": float(
                min(s["worst_network_improvement"] for s in per_fold)
            ),
        })

    pooled_df = pd.DataFrame(pooled_rows).sort_values("overall_improvement", ascending=False)
    fold_df = pd.DataFrame(foldwise_rows).sort_values("overall_improvement", ascending=False)
    per_fold_df = pd.DataFrame(perfold_rows)
    pooled_df.to_csv(HERE / "baselines_pooled.csv", index=False, encoding="utf-8")
    fold_df.to_csv(HERE / "baselines_foldwise.csv", index=False, encoding="utf-8")
    per_fold_df.to_csv(HERE / "baselines_per_fold.csv", index=False, encoding="utf-8")

    print(f"\n(b) all {len(pooled_df)} published baseline records reproduce exactly:",
          bool(pooled_df.reproduces_published.all()))

    pd.set_option("display.width", 200)
    print("\n(b) POOLED convention (as published in SEARCH_POOL_BASELINES.json):")
    print(pooled_df[[
        "name", "overall_improvement", "overall_lo95", "h4_improvement", "h4_lo95",
        "worst_condition_improvement",
    ]].to_string(index=False, float_format=lambda v: f"{v:9.5f}"))

    print("\n(c) FOLD-WISE convention (identical to what the trial JSONs report):")
    print(fold_df[[
        "name", "overall_improvement", "h4_improvement",
        "worst_condition_improvement", "worst_network_improvement",
    ]].to_string(index=False, float_format=lambda v: f"{v:9.5f}"))

    delta = fold_df.merge(pooled_df, on="name", suffixes=("_fold", "_pooled"))
    delta["d_overall"] = delta.overall_improvement_fold - delta.overall_improvement_pooled
    delta["d_worst_cond"] = (
        delta.worst_condition_improvement_fold - delta.worst_condition_improvement_pooled
    )
    print("\n(c-vs-b) convention shift, fold-wise minus pooled:")
    print(delta[["name", "d_overall", "d_worst_cond"]].to_string(
        index=False, float_format=lambda v: f"{v:9.5f}"))


if __name__ == "__main__":
    main()
