"""D+ final consolidation: corruption control re-read, summary JSON, figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
MODES = ("mcar", "burst", "spatial", "mnar")


def main() -> None:
    cand = pd.read_csv(HERE / "trials_candidate_level.csv")
    folds = pd.read_csv(HERE / "trials_fold_level.csv")
    base_fold = pd.read_csv(HERE / "baselines_foldwise.csv")
    matched = pd.read_csv(HERE / "matched_worst_condition.csv")
    r2 = cand[cand.rung == "R2"]

    # ---- corruption control, worst-of-20-cells vs fold-averaged ----------
    print("CORRUPTION CONTROL (history masked at rate 0.30; h4 improvement vs zero-effect)")
    print("The trial JSON stores min-over-folds per mode; corr_min then mins over modes,")
    print("so the headline is the worst of 20 cells.  Fold-averaged view for contrast:\n")
    rows = []
    for cid in r2.sort_values("overall_improvement", ascending=False).head(5).candidate_id:
        ff = folds[(folds.candidate_id == cid) & (folds.rung == "R2")]
        c = r2[r2.candidate_id == cid].iloc[0]
        mode_means = {m: float(ff[f"corr_{m}"].mean()) for m in MODES}
        rows.append({
            "candidate": cid,
            "corr_min_reported": c.corr_min,
            "worst_mode_foldavg": min(mode_means.values()),
            "mean_all_modes_folds": float(np.mean(list(mode_means.values()))),
            "folds_x_modes_positive": int(sum(
                int(v > 0) for m in MODES for v in ff[f"corr_{m}"]
            )),
            "clean_h4": c.h4_improvement,
            **{f"foldavg_{m}": v for m, v in mode_means.items()},
        })
    corr = pd.DataFrame(rows)
    corr.to_csv(HERE / "corruption_control.csv", index=False, encoding="utf-8")
    print(corr.to_string(index=False, float_format=lambda v: f"{v:9.5f}"))
    print("\n(20 = folds x modes cells per candidate)")
    print("No baseline in v8/results/baselines/ is evaluated under corruption at all, so")
    print("F5's 'matched mask-aware baseline' comparator does not exist in this run.")

    allf = folds[folds.rung == "R2"]
    cells = [v for m in MODES for v in allf[f"corr_{m}"]]
    print(f"\nacross all 22 R2 candidates: {sum(int(v > 0) for v in cells)}/{len(cells)} "
          f"fold x mode cells positive; mean = {np.mean(cells):.5f}")

    # ---- summary payload -------------------------------------------------
    best = r2.sort_values("overall_improvement", ascending=False).iloc[0]
    best_ridge = base_fold.sort_values("overall_improvement", ascending=False).iloc[0]
    cur_ridge = base_fold[base_fold.name == "current_ridge_alpha=100"].iloc[0]
    summary = {
        "run": "v8/results/primary_search_v2",
        "trials": {"total": 94, "R1": 72, "R2": 22, "R3": 0,
                   "promoted_at_R1": 24, "R2_missing": 2,
                   "all_complete": bool((cand.status == "complete").all())},
        "comparability": {
            "same_source_csv": True,
            "same_source_sha256": "4052e1c15063dd9a5b2b267c83085cac7e6bf03df1ebb0ddaf232cc658dfa558",
            "same_target_indices": [2, 3, 4, 5, 8, 9],
            "same_channel_weights": [0.35, 0.2, 0.15, 0.15, 0.1, 0.05],
            "same_fold_salt": "v8-groupcv-v1",
            "fold_partition_identical": True,
            "same_metric_function": "summarize_physical_metrics on sqrt(channel_weight)-scaled arrays",
            "aggregation_differs": True,
            "overall_and_h4_comparable": True,
            "worst_case_metrics_comparable_as_published": False,
            "corruption_comparable": False,
        },
        "headline": {
            "best_neural_overall": {
                "candidate_id": best.candidate_id,
                "encoder": best.encoder, "paired_head": best.paired_head,
                "overall_improvement": float(best.overall_improvement),
                "h4_improvement": float(best.h4_improvement),
                "worst_condition_reported": float(best.worst_condition_improvement),
                "worst_condition_matched": float(
                    matched[matched.system.str.contains(best.candidate_id)]
                    .min_mean_condition.iloc[0]
                ),
                "corr_min": float(best.corr_min),
            },
            "best_baseline_overall": {
                "name": best_ridge["name"],
                "overall_improvement": float(best_ridge.overall_improvement),
                "h4_improvement": float(best_ridge.h4_improvement),
                "worst_condition_matched": float(
                    matched[matched.system == best_ridge["name"]].min_mean_condition.iloc[0]
                ),
            },
            "current_ridge_alpha_100": {
                "overall_improvement": float(cur_ridge.overall_improvement),
                "h4_improvement": float(cur_ridge.h4_improvement),
                "worst_condition_matched": float(
                    matched[matched.system == "current_ridge_alpha=100"].min_mean_condition.iloc[0]
                ),
            },
            "neural_beating_best_baseline_overall": 0,
            "neural_h4_above_0.05": 0,
        },
    }
    (HERE / "dplus_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # ---- figure ----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    order = base_fold.sort_values("overall_improvement", ascending=False)

    ax = axes[0]
    ax.barh(range(len(order)), order.overall_improvement, color="#9aa5b1",
            label="baseline panel")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order.name, fontsize=7)
    ax.invert_yaxis()
    top5 = r2.sort_values("overall_improvement", ascending=False).head(5)
    for i, (_, c) in enumerate(top5.iterrows()):
        ax.axvline(c.overall_improvement, color="#c0392b", lw=1.2,
                   alpha=0.9 if i == 0 else 0.35,
                   label="top-5 neural (R2)" if i == 0 else None)
    ax.set_xlabel("overall_improvement (fold-wise convention)")
    ax.set_title("A. Overall: no neural candidate\nreaches the best ridge baseline", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1]
    m = matched.sort_values("min_mean_condition", ascending=False)
    colors = ["#c0392b" if k == "neural" else "#9aa5b1" for k in m.kind]
    ax.barh(range(len(m)), m.min_mean_condition, color=colors)
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels([s[:44] for s in m.system], fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("min over conditions of fold-averaged improvement")
    ax.set_title("B. Worst condition, matched estimator:\nneural leads the whole panel",
                 fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    ax = axes[2]
    ax.scatter(cand[cand.rung == "R1"].overall_improvement,
               cand[cand.rung == "R1"].h4_improvement, s=22, alpha=0.65,
               c="#7f8c8d", label="R1 (72)")
    ax.scatter(r2.overall_improvement, r2.h4_improvement, s=42, alpha=0.9,
               c="#c0392b", label="R2 (22)")
    ax.axhline(0.05, color="#c0392b", ls="--", lw=1.2, label="F1 threshold h4 >= 0.05")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.axvline(0.10, color="#2980b9", ls=":", lw=1.2, label="F2 threshold overall >= 0.10")
    ax.set_xlabel("overall_improvement")
    ax.set_ylabel("h4_improvement")
    ax.set_title("C. F1 is the binding gate: max h4 = 0.0037,\n"
                 "14x below the 0.05 threshold", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "V8 primary_search_v2 (94 trials, representation_train only) vs the grouped-CV baseline panel",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(HERE / "dplus_summary.png", dpi=160)
    print(f"\nwrote {HERE / 'dplus_summary.png'}")
    print(f"wrote {HERE / 'dplus_summary.json'}")


if __name__ == "__main__":
    main()
