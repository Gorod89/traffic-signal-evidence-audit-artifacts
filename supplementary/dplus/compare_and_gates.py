"""D+ step 3-6: neural-vs-baseline comparison, F1-F7 gate census, family effects.

Read-only against v8/.  All outputs land in this directory.
"""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
V8 = Path(
    os.environ.get("V8_SOURCE_ROOT", HERE.parents[1] / "external" / "v8")
).resolve()

VRAM_LIMIT = 6656.0
LATENCY_LIMIT = 100.0
STRUCT_TOL = 1e-5


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cand = pd.read_csv(HERE / "trials_candidate_level.csv")
    folds = pd.read_csv(HERE / "trials_fold_level.csv")
    base_fold = pd.read_csv(HERE / "baselines_foldwise.csv")
    base_pooled = pd.read_csv(HERE / "baselines_pooled.csv")
    base_per_fold = pd.read_csv(HERE / "baselines_per_fold.csv")
    return cand, folds, base_fold, base_pooled, base_per_fold


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def paired_fold_test(neural: np.ndarray, base: np.ndarray) -> dict[str, float]:
    """Paired comparison across the 5 shared GroupCV folds."""
    diff = neural - base
    n = len(diff)
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1))
    se = sd / np.sqrt(n) if sd > 0 else 0.0
    t_crit = stats.t.ppf(0.95, n - 1)
    wins = int((diff > 0).sum())
    # exact one-sided sign test, H1: neural > baseline
    p_sign = float(stats.binomtest(wins, n, 0.5, alternative="greater").pvalue)
    return {
        "mean_diff": mean,
        "sd_diff": sd,
        "one_sided_95_lower": mean - t_crit * se if se > 0 else mean,
        "folds_won": wins,
        "p_sign_one_sided": p_sign,
    }


def main() -> None:
    cand, folds, base_fold, base_pooled, base_per_fold = load()
    r1 = cand[cand.rung == "R1"].copy()
    r2 = cand[cand.rung == "R2"].copy()
    pd.set_option("display.width", 220)

    # =================================================================
    section("1. RUN INVENTORY")
    print(f"trials total          : {len(cand)}  (R1={len(r1)}, R2={len(r2)})")
    print(f"all statuses complete : {bool((cand.status == 'complete').all())}")
    print(f"errors recorded       : {int(cand.error.notna().sum())}")
    print(f"folds per trial       : {sorted(cand.n_folds_recorded.unique().tolist())}")
    print(f"epochs R1 / R2        : "
          f"{sorted(folds[folds.rung=='R1'].n_epochs.unique())} / "
          f"{sorted(folds[folds.rung=='R2'].n_epochs.unique())}")
    print(f"train_rows / tune_rows: {sorted(cand.train_rows.unique())} / "
          f"{sorted(cand.tune_rows.unique())}  (summed over 5 folds)")
    print(f"wall_seconds total    : {cand.wall_seconds.sum():.1f} s over all 94 trials")

    # =================================================================
    section("2. NEURAL vs BASELINE PANEL (matched fold-wise convention)")
    best_ov = r2.sort_values("overall_improvement", ascending=False).iloc[0]
    best_h4 = r2.sort_values("h4_improvement", ascending=False).iloc[0]
    best_worst = r2.sort_values("worst_condition_improvement", ascending=False).iloc[0]
    print(f"best R2 by overall : {best_ov.candidate_id} {best_ov.encoder}/{best_ov.paired_head} "
          f"overall={best_ov.overall_improvement:.5f} h4={best_ov.h4_improvement:.5f} "
          f"worst_cond={best_ov.worst_condition_improvement:.5f} corr_min={best_ov.corr_min:.5f}")
    print(f"best R2 by h4      : {best_h4.candidate_id} {best_h4.encoder}/{best_h4.paired_head} "
          f"overall={best_h4.overall_improvement:.5f} h4={best_h4.h4_improvement:.5f}")
    print(f"best R2 by worst   : {best_worst.candidate_id} {best_worst.encoder}/"
          f"{best_worst.paired_head} worst_cond={best_worst.worst_condition_improvement:.5f}")

    rows = []
    for _, b in base_fold.iterrows():
        rows.append({
            "system": f"BASELINE {b['name']}",
            "kind": "baseline",
            "overall": b.overall_improvement,
            "h4": b.h4_improvement,
            "worst_cond": b.worst_condition_improvement,
            "worst_net": b.worst_network_improvement,
            "corr_min": np.nan,
        })
    for _, c in r2.sort_values("overall_improvement", ascending=False).head(5).iterrows():
        rows.append({
            "system": f"NEURAL R2 {c.candidate_id} ({c.encoder}/{c.paired_head})",
            "kind": "neural",
            "overall": c.overall_improvement,
            "h4": c.h4_improvement,
            "worst_cond": c.worst_condition_improvement,
            "worst_net": c.worst_network_improvement,
            "corr_min": c.corr_min,
        })
    table = pd.DataFrame(rows).sort_values("overall", ascending=False)
    table.to_csv(HERE / "headline_comparison.csv", index=False, encoding="utf-8")
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:9.5f}"))

    print(f"\nneural candidates (of 94) beating best visible baseline overall "
          f"({base_fold.overall_improvement.max():.5f}): "
          f"{int((cand.overall_improvement > base_fold.overall_improvement.max()).sum())}")
    ridge_current = base_fold.loc[base_fold.name == "current_ridge_alpha=100",
                                  "overall_improvement"].iloc[0]
    print(f"neural candidates beating current_ridge_alpha=100 ({ridge_current:.5f}): "
          f"{int((cand.overall_improvement > ridge_current).sum())}")
    print(f"neural candidates with h4_improvement > 0: "
          f"{int((cand.h4_improvement > 0).sum())} / 94   "
          f"max h4 = {cand.h4_improvement.max():.5f}")
    print(f"baselines with h4_improvement > 0: "
          f"{int((base_fold.h4_improvement > 0).sum())} / 16   "
          f"max h4 = {base_fold.h4_improvement.max():.5f}")

    # =================================================================
    section("3. PAIRED-BY-FOLD TESTS (5 shared GroupCV folds; route-seed bootstrap NOT possible)")
    print("Per-row neural predictions were never persisted, so a 50-cluster route-seed")
    print("bootstrap of the NEURAL side cannot be reconstructed from the artifacts.")
    print("The finest shared unit available is the fold (n=5).  Underpowered by design:")
    print("the best attainable one-sided exact sign-test p at n=5 is 0.03125 (5/5 wins).\n")

    top5 = r2.sort_values("overall_improvement", ascending=False).head(5).candidate_id.tolist()
    contenders = ["history_ridge_alpha=100", "current_ridge_alpha=100",
                  "global_canonical_mean", "zero_effect"]
    test_rows = []
    for cid in top5:
        nf = folds[(folds.candidate_id == cid) & (folds.rung == "R2")].sort_values("fold")
        for bname in contenders:
            bf = base_per_fold[base_per_fold.name == bname].sort_values("fold")
            for metric in ("overall_improvement", "h4_improvement"):
                res = paired_fold_test(nf[metric].to_numpy(), bf[metric].to_numpy())
                test_rows.append({"candidate": cid, "baseline": bname, "metric": metric, **res})
    tests = pd.DataFrame(test_rows)
    tests.to_csv(HERE / "paired_fold_tests.csv", index=False, encoding="utf-8")
    for metric in ("overall_improvement", "h4_improvement"):
        print(f"--- {metric} : neural minus baseline, paired over 5 folds")
        print(tests[tests.metric == metric][
            ["candidate", "baseline", "mean_diff", "one_sided_95_lower",
             "folds_won", "p_sign_one_sided"]
        ].to_string(index=False, float_format=lambda v: f"{v:9.5f}"))
        print()

    # =================================================================
    section("4. GATE CENSUS F1-F7 (fast_falsification.json, schema_version 2)")
    g = cand.copy()
    g["F0_structural"] = (
        (g.paired_head != "direct")
        & (g.equal_action_max_abs <= STRUCT_TOL)
        & (g.swap_antisymmetry_max_abs <= STRUCT_TOL)
    )
    g["F1_h4_ge_0.05"] = g.h4_improvement >= 0.05
    g["F2_overall_ge_0.10"] = g.overall_improvement >= 0.10
    g["F3_permutation_ge_0.05"] = g.action_permutation_degradation >= 0.05
    g["F5proxy_corruption_pos"] = g.corr_min > 0.0
    g["F7_resources"] = (g.peak_vram_mib <= VRAM_LIMIT) & (g.p95_batch1_ms <= LATENCY_LIMIT)
    gate_cols = ["F0_structural", "F1_h4_ge_0.05", "F2_overall_ge_0.10",
                 "F3_permutation_ge_0.05", "F5proxy_corruption_pos", "F7_resources"]
    g.to_csv(HERE / "gate_census.csv", index=False, encoding="utf-8")

    print("pass counts (n / cohort size):")
    print(f"{'gate':28s} {'ALL 94':>10s} {'R1 (72)':>10s} {'R2 (22)':>10s}")
    for col in gate_cols:
        print(f"{col:28s} {int(g[col].sum()):>10d} "
              f"{int(g[g.rung=='R1'][col].sum()):>10d} {int(g[g.rung=='R2'][col].sum()):>10d}")
    print(f"{'F4 (SCM graph shuffle)':28s} {'n/a':>10s} {'n/a':>10s} {'n/a':>10s}"
          "   <- never executed in this run")
    print(f"{'F6 (effective rank)':28s} {'diag':>10s} {'diag':>10s} {'diag':>10s}"
          "   <- diagnostic only, not a kill gate")

    g["all_pass"] = g[gate_cols].all(axis=1)
    g["pass_wo_F5"] = g[[c for c in gate_cols if c != "F5proxy_corruption_pos"]].all(axis=1)
    print(f"\nALL of F0,F1,F2,F3,F5proxy,F7 : {int(g.all_pass.sum())} / 94")
    print(f"same set excluding F5 proxy    : {int(g.pass_wo_F5.sum())} / 94")
    print(f"F0+F2+F3+F7 only (drop h4+corr): "
          f"{int(g[['F0_structural','F2_overall_ge_0.10','F3_permutation_ge_0.05','F7_resources']].all(axis=1).sum())} / 94")

    print("\nbinding constraint -- distance to threshold:")
    print(f"  h4_improvement    : max={g.h4_improvement.max():.5f} vs threshold 0.05  "
          f"(gap {0.05 - g.h4_improvement.max():.5f}); median={g.h4_improvement.median():.5f}")
    print(f"  overall_improvement: max={g.overall_improvement.max():.5f} vs threshold 0.10; "
          f"median={g.overall_improvement.median():.5f}")
    print(f"  perm degradation  : min={g.action_permutation_degradation.min():.5f} vs 0.05")
    print(f"  corr_min          : max={g.corr_min.max():.5f} vs threshold 0.0  "
          f"(gap {0.0 - g.corr_min.max():.5f}); median={g.corr_min.median():.5f}")
    print(f"  peak_vram_mib     : max={g.peak_vram_mib.max():.1f} vs {VRAM_LIMIT}")
    print(f"  p95_batch1_ms     : max={g.p95_batch1_ms.max():.2f} vs {LATENCY_LIMIT}")
    print(f"  effective_rank_frac (F6 diag): min={g.effective_rank_fraction.min():.4f} "
          f"median={g.effective_rank_fraction.median():.4f} max={g.effective_rank_fraction.max():.4f}")
    print(f"\nF0 failures are exactly the 'direct' heads: "
          f"{int((g.paired_head=='direct').sum())} candidates; "
          f"non-direct with structural violation: "
          f"{int(((g.paired_head!='direct') & ~g.F0_structural).sum())}")

    # =================================================================
    section("5. ENCODER FAMILY AND PAIRED-HEAD EFFECTS (R1 cohort = unbiased, n=72)")
    print("R2 is a promotion-selected subset, so family aggregates use R1.\n")
    metrics = ["overall_improvement", "h4_improvement", "worst_condition_improvement",
               "corr_min", "action_permutation_degradation", "parameter_count"]

    for key in ("encoder", "paired_head"):
        print(f"--- by {key} (R1, n=72)")
        agg = r1.groupby(key)[metrics].agg(["count", "mean", "std", "median"])
        summary = pd.DataFrame({
            "n": agg[("overall_improvement", "count")],
            "overall_mean": agg[("overall_improvement", "mean")],
            "overall_sd": agg[("overall_improvement", "std")],
            "h4_mean": agg[("h4_improvement", "mean")],
            "worst_cond_mean": agg[("worst_condition_improvement", "mean")],
            "corr_min_mean": agg[("corr_min", "mean")],
            "perm_mean": agg[("action_permutation_degradation", "mean")],
        })
        print(summary.to_string(float_format=lambda v: f"{v:9.5f}"))
        for metric in ("overall_improvement", "h4_improvement", "corr_min"):
            groups = [grp[metric].to_numpy() for _, grp in r1.groupby(key)]
            h, p = stats.kruskal(*groups)
            f, pf = stats.f_oneway(*groups)
            print(f"    {metric:28s} Kruskal-Wallis H={h:6.3f} p={p:.4f} | "
                  f"ANOVA F={f:6.3f} p={pf:.4f}")
        print()

    print("--- two-way cell means, overall_improvement (R1)")
    print(r1.pivot_table(index="encoder", columns="paired_head",
                         values="overall_improvement", aggfunc="mean")
          .to_string(float_format=lambda v: f"{v:9.5f}"))
    print("\n--- cell counts")
    print(r1.pivot_table(index="encoder", columns="paired_head",
                         values="overall_improvement", aggfunc="count").to_string())
    print("\n--- two-way cell means, h4_improvement (R1)")
    print(r1.pivot_table(index="encoder", columns="paired_head",
                         values="h4_improvement", aggfunc="mean")
          .to_string(float_format=lambda v: f"{v:9.5f}"))
    print("\n--- two-way cell means, corr_min (R1)")
    print(r1.pivot_table(index="encoder", columns="paired_head",
                         values="corr_min", aggfunc="mean")
          .to_string(float_format=lambda v: f"{v:9.5f}"))

    print("\n--- other design factors (R1, mean overall_improvement)")
    for key in ("width", "depth", "dropout", "activation"):
        sub = r1.groupby(key)[["overall_improvement", "h4_improvement"]].agg(["count", "mean"])
        print(f"  {key}:")
        print("    " + sub.to_string().replace("\n", "\n    "))

    # R1 vs R2 rank agreement on the 22 that ran both
    both = r1.merge(r2, on="candidate_id", suffixes=("_r1", "_r2"))
    rho, prho = stats.spearmanr(both.overall_improvement_r1, both.overall_improvement_r2)
    rho4, prho4 = stats.spearmanr(both.h4_improvement_r1, both.h4_improvement_r2)
    print(f"\n--- R1->R2 rank agreement on the {len(both)} candidates run at both rungs")
    print(f"    overall_improvement Spearman rho={rho:.3f} p={prho:.4f}")
    print(f"    h4_improvement      Spearman rho={rho4:.3f} p={prho4:.4f}")
    print(f"    mean overall gain 3->10 epochs: "
          f"{(both.overall_improvement_r2 - both.overall_improvement_r1).mean():+.5f}")
    print(f"    mean h4 gain 3->10 epochs     : "
          f"{(both.h4_improvement_r2 - both.h4_improvement_r1).mean():+.5f}")

    # =================================================================
    section("6. PER-CONDITION DETAIL, best neural vs best baselines (fold-wise)")
    cond_cols = [c for c in cand.columns if c.startswith("cond_")]
    best = r2.sort_values("overall_improvement", ascending=False).iloc[0]
    print(f"best neural {best.candidate_id}:")
    for c in cond_cols:
        print(f"    {c[5:]:22s} {best[c]:9.5f}")
    print(f"    {'worst (min over folds)':22s} {best.worst_condition_improvement:9.5f}")


if __name__ == "__main__":
    main()
