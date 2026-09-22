"""D+ addendum: a worst-condition statistic that is actually matched.

The trial JSONs report worst_condition_improvement = min over 5 folds of
(min over 6 conditions inside that fold), i.e. the worst of 30 cells of ~20 rows.
The published baseline panel reports min over 6 conditions computed on all 600
pooled rows.  Those are different estimators and cannot be compared.

Both sides do, however, expose one common estimator:
    min over the 6 conditions of the fold-averaged per-condition improvement.
For the neural side this is min(metrics.condition_improvements), which
train_cv_trial builds by averaging each condition across the 5 folds.
This script computes the same quantity for every baseline.
"""

from __future__ import annotations

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
from v8search.metrics import summarize_physical_metrics  # noqa: E402
from v8search.search import TrainingRecipe, grouped_fold_ids  # noqa: E402


def main() -> None:
    arrays, _ = load_legacy_physical_data(
        V8 / "data" / "representation_train_physical.csv", purpose="search"
    )
    fold_ids = grouped_fold_ids(arrays.cluster, n_folds=5)
    predictions = grouped_cv_predictions(arrays)
    recipe = TrainingRecipe()
    multiplier = np.sqrt(np.asarray(recipe.channel_weights))[None, None, :]
    weighted_target = arrays.target_delta * multiplier

    rows = []
    for name, prediction in predictions.items():
        weighted_prediction = prediction * multiplier
        per_condition: dict[str, list[float]] = {}
        per_network: dict[str, list[float]] = {}
        for fold in range(5):
            test = np.flatnonzero(fold_ids == fold)
            summary = summarize_physical_metrics(
                weighted_prediction[test], weighted_target[test],
                condition=arrays.condition[test], network=arrays.network[test],
            ).as_dict()
            for key, value in summary["condition_improvements"].items():
                per_condition.setdefault(key, []).append(value)
            for key, value in summary["network_improvements"].items():
                per_network.setdefault(key, []).append(value)
        cond_mean = {k: float(np.mean(v)) for k, v in per_condition.items()}
        net_mean = {k: float(np.mean(v)) for k, v in per_network.items()}
        row = {
            "system": name,
            "kind": "baseline",
            "min_mean_condition": min(cond_mean.values()),
            "min_mean_network": min(net_mean.values()),
        }
        row.update({f"cond_{k}": v for k, v in cond_mean.items()})
        rows.append(row)

    cand = pd.read_csv(HERE / "trials_candidate_level.csv")
    cond_cols = [c for c in cand.columns if c.startswith("cond_")]
    net_cols = [c for c in cand.columns if c.startswith("net_")]
    for _, c in cand[cand.rung == "R2"].sort_values(
        "overall_improvement", ascending=False
    ).head(5).iterrows():
        row = {
            "system": f"NEURAL {c.candidate_id} ({c.encoder}/{c.paired_head})",
            "kind": "neural",
            "min_mean_condition": float(c[cond_cols].min()),
            "min_mean_network": float(c[net_cols].min()),
        }
        row.update({col: float(c[col]) for col in cond_cols})
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("min_mean_condition", ascending=False)
    table.to_csv(HERE / "matched_worst_condition.csv", index=False, encoding="utf-8")
    pd.set_option("display.width", 220)
    print("MATCHED worst-condition estimator: min over conditions of fold-averaged improvement")
    print("(higher is better; this is the only worst-case statistic both sides can produce)\n")
    print(table[["system", "kind", "min_mean_condition", "min_mean_network"]].to_string(
        index=False, float_format=lambda v: f"{v:9.5f}"))

    print("\nper-condition fold-averaged improvement, best neural vs top baselines:")
    show = table[table.system.isin([
        "history_ridge_alpha=100", "current_ridge_alpha=100", "global_canonical_mean",
    ]) | table.kind.eq("neural")]
    print(show[["system"] + sorted(c for c in table.columns if c.startswith("cond_"))].to_string(
        index=False, float_format=lambda v: f"{v:9.5f}"))

    # how much of the published -0.129 is fold noise rather than condition weakness?
    folds = pd.read_csv(HERE / "trials_fold_level.csv")
    best = cand[cand.rung == "R2"].sort_values("overall_improvement", ascending=False).iloc[0]
    bf = folds[(folds.candidate_id == best.candidate_id) & (folds.rung == "R2")]
    print(f"\nbest neural {best.candidate_id}: per-fold worst_condition_improvement = "
          f"{[round(v, 5) for v in bf.sort_values('fold').worst_condition_improvement]}")
    print(f"  min over folds (what the trial JSON reports) = "
          f"{best.worst_condition_improvement:.5f}")
    print(f"  min over fold-averaged conditions (matched)  = "
          f"{float(best[cond_cols].min()):.5f}")


if __name__ == "__main__":
    main()
