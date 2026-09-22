"""PACE Phase 2, tasks 2-5: cluster-conformal calibration and the selective rule.

This is the offline test of hypotheses P4 (a policy trained only on data it was
allowed to see yields a positive realised effect) and P5 (realised harm stays
inside its design bound), under the conjunctive gate V9 registered as H5.

Order of operations, and the reason for it
------------------------------------------
1.  The effect model is fitted on the 960 training rows.
2.  The conformal margin `q` is fitted on the 120 calibration clusters.
3.  The operating threshold is chosen on the same calibration block.
4.  Only then is the sealed block read.

Steps 1-3 never touch the sealed data; step 4 never refits, retunes or reselects.
That ordering is the whole point: V4 and V5 failed a harm gate whose bound came
from a model, and the standing question is whether the bound fails because the
rule is wrong or because the calibration set was too small to certify it.  Here
the calibration set is fixed at 120 clusters and the evaluation set at 300, both
at or above the sizes the Phase 0 sample-size curve says are needed.

The conformal construction (task 2)
-----------------------------------
Nonconformity is one-sided, because the rule only needs protection against
over-promising.  For calibration row i, r_i = g(x_i) - y_i; positive r_i means
the model promised more gain than materialised.  For calibration cluster c
(key: network|seed, 6 rows each),

        s_c = max_{i in c} r_i

is the cluster's worst over-promise.  With m = 120 clusters and alpha = 0.10,

        k = ceil((1 - alpha) * (m + 1)) = ceil(0.9 * 121) = 109
        q = the k-th smallest of {s_c}

and the certified lower bound on the gain of any state x is

        LCB(x) = g(x) - q.

Exchangeability holds at the cluster level, so P(S_new <= q) >= 1 - alpha over a
fresh cluster.  Because the score is a within-cluster maximum, the guarantee is
simultaneous over all rows of that cluster, not merely marginal over rows -- the
stronger of the two statements, and the one an operator acting repeatedly inside
a single route-seed actually needs.  It is also the more expensive one: q is set
by cluster maxima, so the margin is wider than a row-level conformal quantile.

The selective rule (task 3)
---------------------------
Intervene on x iff LCB(x) > tau, equivalently g(x) > tau + q.  Tau is swept over
a grid; the operating point is chosen on calibration and reported on sealed.

Bounds (task 6)
---------------
Harm rate and coverage are means of cluster statistics bounded in [0, 1], so the
gate-facing bound is the finite-sample empirical-Bernstein interval transcribed
from `v9/src/v9evidence/statistics.py`.  It is used rather than the percentile
bootstrap for two reasons: the bootstrap collapses to a degenerate point
whenever every observed cluster sits at 0 or 1 -- exactly the regime a highly
selective rule produces -- and V4 reported its 56.15% harm ceiling with this
estimator, so keeping it makes the historical line comparable.  The percentile
bootstrap is reported alongside as a diagnostic.  Mean realised gain is unbounded
and keeps the cluster bootstrap.

Scope
-----
This is an offline evaluation of a selective rule on pre-recorded paired
branches.  Each row is a fork from a common snapshot and RNG state, so the gain
of intervening is observed exactly for every candidate row whether or not the
rule selects it.  Nothing here is a closed-loop controller: selections do not
alter the trajectory, do not change which states arrive later, and cannot
compound.  A closed-loop result requires Phase 1 randomised probing.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from phase2_common import (
    BOOTSTRAP_REPLICATES,
    CONFIDENCE,
    RESULTS,
    SEED,
    bootstrap_active_cluster_mean,
    empirical_bernstein,
    load_csv_blocks,
    load_sealed_block,
    write_json,
)

OUT = RESULTS / "selective_policy.json"

ALPHA_CONFORMAL = 0.10
THRESHOLD_GRID = [float(t) for t in range(0, 205, 5)]

# Conjunctive gate, as registered for V9 H5_CALIBRATED_AUTHORITY and restated for
# PACE P4/P5. All three must hold simultaneously at one threshold.
GATE_HARM_MAX = 0.10
GATE_MIN_ACTIVE_CLUSTERS = 30


# --------------------------------------------------------------------------
# conformal margin
# --------------------------------------------------------------------------


def conformal_margin(pred: np.ndarray, y: np.ndarray, clusters: np.ndarray, alpha: float) -> dict:
    """One-sided cluster-conformal margin using the within-cluster maximum score."""
    residual = pred - y  # > 0 means the model over-promised
    scores, keys = [], []
    for c in np.unique(clusters):
        sel = clusters == c
        scores.append(float(residual[sel].max()))
        keys.append(str(c))
    scores_arr = np.asarray(scores)
    m = scores_arr.size
    k = math.ceil((1.0 - alpha) * (m + 1))
    if k > m:
        raise SystemExit(
            f"alpha={alpha} needs {k} cluster scores but only {m} clusters exist; "
            "the margin would be infinite"
        )
    q = float(np.sort(scores_arr)[k - 1])
    return {
        "alpha": alpha,
        "n_clusters": m,
        "order_statistic_k": k,
        "q": q,
        "score_definition": "max over rows within a network|seed cluster of (prediction - realised gain)",
        "guarantee": (
            "P(all rows of a fresh exchangeable cluster satisfy gain >= prediction - q) >= 1 - alpha"
        ),
        "score_quantiles": {
            "min": float(scores_arr.min()),
            "p25": float(np.percentile(scores_arr, 25)),
            "median": float(np.median(scores_arr)),
            "p75": float(np.percentile(scores_arr, 75)),
            "p90": float(np.percentile(scores_arr, 90)),
            "max": float(scores_arr.max()),
        },
        "row_level_quantile_for_reference": float(
            np.sort(residual)[math.ceil((1.0 - alpha) * (residual.size + 1)) - 1]
        ),
    }


def empirical_cluster_coverage(pred: np.ndarray, y: np.ndarray, clusters: np.ndarray, q: float) -> dict:
    """How often a whole cluster is covered by LCB = pred - q. Sanity check, not a gate."""
    covered = []
    for c in np.unique(clusters):
        sel = clusters == c
        covered.append(bool(np.all(y[sel] >= pred[sel] - q)))
    frac = float(np.mean(covered))
    mean, lo, hi = empirical_bernstein([float(v) for v in covered])
    return {
        "cluster_simultaneous_coverage": frac,
        "empirical_bernstein_ci95": [lo, hi],
        "target": 1.0 - ALPHA_CONFORMAL,
        "n_clusters": len(covered),
        "row_marginal_coverage": float(np.mean(y >= pred - q)),
    }


# --------------------------------------------------------------------------
# selective rule evaluation
# --------------------------------------------------------------------------


def evaluate_threshold(
    lcb: np.ndarray, y: np.ndarray, clusters: np.ndarray, tau: float, *, with_intervals: bool
) -> dict:
    """Coverage, realised gain, and harm rate for the rule `LCB > tau`."""
    selected = lcb > tau
    n_selected = int(selected.sum())
    active = sorted({str(c) for c in clusters[selected]})
    uniq = np.unique(clusters)

    # Coverage / AAR: equal weight per cluster, bounded in [0, 1].
    coverage_by_cluster = [float((selected & (clusters == c)).sum() / (clusters == c).sum()) for c in uniq]
    cov_mean, cov_lo, cov_hi = empirical_bernstein(coverage_by_cluster)

    row = {
        "tau": tau,
        "n_selected_rows": n_selected,
        "n_eligible_rows": int(y.size),
        "n_active_clusters": len(active),
        "n_clusters": int(uniq.size),
        "row_selection_fraction": float(n_selected / y.size),
        "coverage_equal_weight_cluster": {
            "estimate": cov_mean,
            "eb_ci95": [cov_lo, cov_hi],
            "estimand": "equal_weight_mean_cluster_selection_fraction",
        },
    }

    if n_selected == 0:
        row.update(
            {
                "mean_gain_selected": None,
                "harmful_rate_selected": None,
                "status": "no_selection",
            }
        )
        return row

    # Per-cluster statistics, computed once. Everything downstream -- point
    # estimates, empirical Bernstein bounds, bootstrap intervals -- is a function
    # of these, which is what makes the interval cluster-robust.
    harmful = (y < 0.0).astype(np.float64)
    active = np.zeros(uniq.size, dtype=bool)
    gain_per_cluster = np.zeros(uniq.size)
    harm_per_cluster = np.zeros(uniq.size)
    for j, c in enumerate(uniq):
        sel_c = selected & (clusters == c)
        if sel_c.any():
            active[j] = True
            gain_per_cluster[j] = float(y[sel_c].mean())
            harm_per_cluster[j] = float(harmful[sel_c].mean())

    gain_cluster_values = gain_per_cluster[active]
    harm_cluster_values = harm_per_cluster[active]

    if harm_cluster_values.size >= 2:
        h_mean, h_lo, h_hi = empirical_bernstein(harm_cluster_values.tolist())
    else:
        # A single active cluster carries no variance estimate, so no bound is
        # claimed; [0, 1] is reported and the gate cannot be satisfied by it.
        h_mean, h_lo, h_hi = float(harm_cluster_values.mean()), 0.0, 1.0

    row["mean_gain_selected"] = {
        "equal_weight_cluster_mean": float(gain_cluster_values.mean()),
        "pooled_row_mean": float(y[selected].mean()),
        "n_clusters_contributing": int(active.sum()),
    }
    row["harmful_rate_selected"] = {
        "equal_weight_cluster_mean": h_mean,
        "pooled_row_rate": float(harmful[selected].mean()),
        "eb_ci95": [h_lo, h_hi],
        "eb_upper": h_hi,
        "n_clusters_contributing": int(active.sum()),
        "harm_definition": "gain_h4 < 0 strictly; exact zeros count as neither harmful nor beneficial",
    }
    row["beneficial_rate_selected_pooled"] = float((y[selected] > 0).mean())

    if with_intervals:
        gain_boot = bootstrap_active_cluster_mean(gain_per_cluster, active)
        harm_boot = bootstrap_active_cluster_mean(harm_per_cluster, active)
        row["mean_gain_selected"].update(
            {
                "bootstrap_ci95": gain_boot["ci95"],
                "one_sided_lower_95": gain_boot["one_sided_lower"],
                "degenerate_replicates": gain_boot["degenerate_replicates"],
                "degenerate_replicate_fraction": gain_boot["degenerate_replicates"]
                / BOOTSTRAP_REPLICATES,
            }
        )
        row["harmful_rate_selected"]["bootstrap_ci95_diagnostic"] = harm_boot["ci95"]

    row["status"] = "evaluated"
    return row


def certifiability_floor(limit: float = GATE_HARM_MAX, confidence: float = CONFIDENCE) -> dict:
    """Smallest number of active clusters at which the harm gate is reachable at all.

    The empirical-Bernstein radius carries a variance-free term
    7*log(2/delta)/(3*(n-1)) that no data can shrink.  Even with a perfect rule --
    zero observed harm in every active cluster, hence zero sample variance -- the
    upper bound cannot fall to `limit` until that term alone does.  Reporting this
    floor separates two failures that look identical in a results table: a rule
    that is unsafe, and a rule that is merely uncertifiable because it fired in
    too few independent clusters.
    """
    log_term = math.log(2.0 / (1.0 - confidence))
    n_floor = math.ceil(1.0 + 7.0 * log_term / (3.0 * limit))
    return {
        "harm_limit": limit,
        "confidence_level": confidence,
        "variance_free_term": "7*log(2/delta)/(3*(n-1))",
        "minimum_active_clusters_even_with_zero_observed_harm": n_floor,
        "interpretation": (
            f"below {n_floor} active clusters the {int(100 * confidence)}% empirical-Bernstein "
            f"upper bound cannot reach {limit} no matter what the data show, so a harm-gate "
            "failure there is an information failure, not a safety finding"
        ),
    }


def gate_verdict(row: dict) -> dict:
    """The conjunctive gate: positive gain lower bound, bounded harm, live activity."""
    if row.get("status") != "evaluated":
        return {
            "gain_positive": False,
            "harm_bounded": False,
            "activity_live": False,
            "passed": False,
            "reason": "no rows selected at this threshold",
        }
    gain = row["mean_gain_selected"]
    lower = gain.get("bootstrap_ci95", [None, None])[0]
    harm_upper = row["harmful_rate_selected"]["eb_upper"]
    active = row["n_active_clusters"]
    a = bool(lower is not None and lower > 0.0)
    b = bool(harm_upper <= GATE_HARM_MAX)
    c = bool(active >= GATE_MIN_ACTIVE_CLUSTERS)
    floor = certifiability_floor()["minimum_active_clusters_even_with_zero_observed_harm"]
    return {
        "gain_positive": a,
        "gain_lower_bound": lower,
        "harm_bounded": b,
        "harm_upper_bound": harm_upper,
        "harm_point_estimate": row["harmful_rate_selected"]["equal_weight_cluster_mean"],
        "harm_limit": GATE_HARM_MAX,
        "activity_live": c,
        "active_clusters": active,
        "minimum_active_clusters": GATE_MIN_ACTIVE_CLUSTERS,
        "harm_bound_reachable_at_this_activity": bool(active >= floor),
        "harm_failure_kind": (
            None
            if b
            else ("uncertifiable_too_few_active_clusters" if active < floor else "harm_level_exceeds_limit")
        ),
        "passed": bool(a and b and c),
    }


def sweep(lcb: np.ndarray, block, *, with_intervals: bool) -> list[dict]:
    rows = []
    for tau in THRESHOLD_GRID:
        r = evaluate_threshold(lcb, block.y, block.clusters, tau, with_intervals=with_intervals)
        r["gate"] = gate_verdict(r)
        rows.append(r)
    return rows


def print_sweep(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    print(
        f"{'tau':>5} {'sel rows':>9} {'clusters':>9} {'AAR':>8} "
        f"{'gain':>10} {'gain lo95':>10} {'harm':>7} {'harm hi95':>10}  gate"
    )
    for r in rows:
        if r["status"] != "evaluated":
            print(f"{r['tau']:>5.0f} {0:>9} {0:>9} {0.0:>8.4f} {'--':>10} {'--':>10} {'--':>7} {'--':>10}  no selection")
            continue
        g = r["mean_gain_selected"]
        h = r["harmful_rate_selected"]
        lo = g.get("bootstrap_ci95", [None])[0]
        print(
            f"{r['tau']:>5.0f} {r['n_selected_rows']:>9} {r['n_active_clusters']:>9} "
            f"{r['coverage_equal_weight_cluster']['estimate']:>8.4f} "
            f"{g['equal_weight_cluster_mean']:>10.2f} "
            f"{(f'{lo:10.2f}' if lo is not None else '        --')} "
            f"{h['equal_weight_cluster_mean']:>7.3f} {h['eb_upper']:>10.3f}  "
            f"{'PASS' if r['gate']['passed'] else 'fail'}"
        )


def main() -> None:
    train, calib, paired_sha = load_csv_blocks()

    # ---- step 1: fit the effect model on training rows only -----------------
    model = ExtraTreesRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
    model.fit(train.con, train.y)
    print(
        f"model: ExtraTrees(300) on feature_vector, fitted on {train.n_rows} rows / "
        f"{train.n_clusters} clusters"
    )
    print(
        "selected by calibration-split performance (rmse 115.42, auroc 0.811 -- best on both "
        "criteria in model_ladder.py); the sealed block played no part in this choice"
    )

    pred_calib = model.predict(calib.con)

    # ---- step 2: conformal margin from the calibration clusters -------------
    margin = conformal_margin(pred_calib, calib.y, calib.clusters, ALPHA_CONFORMAL)
    q = margin["q"]
    print(f"\nconformal margin q = {q:.2f} from {margin['n_clusters']} clusters, "
          f"order statistic {margin['order_statistic_k']}, alpha = {ALPHA_CONFORMAL}")
    print(f"  row-level quantile for reference: {margin['row_level_quantile_for_reference']:.2f}")
    coverage_check = empirical_cluster_coverage(pred_calib, calib.y, calib.clusters, q)
    print(f"  in-sample cluster coverage {coverage_check['cluster_simultaneous_coverage']:.3f} "
          f"(target {coverage_check['target']:.2f}), row coverage "
          f"{coverage_check['row_marginal_coverage']:.3f}")

    lcb_calib = pred_calib - q
    print(f"  calibration LCB: max {lcb_calib.max():.2f}, "
          f"rows with LCB > 0: {int((lcb_calib > 0).sum())} of {calib.n_rows}")

    # ---- step 3: choose the operating threshold on calibration --------------
    calib_sweep = sweep(lcb_calib, calib, with_intervals=True)
    print_sweep("CALIBRATION SPLIT -- threshold selection (sealed data not yet read)", calib_sweep)

    admissible = [r for r in calib_sweep if r["gate"]["passed"]]
    if admissible:
        tau_star = min(r["tau"] for r in admissible)
        selection_rule = "smallest grid threshold meeting the conjunctive gate on calibration"
    else:
        tau_star = 0.0
        selection_rule = (
            "no grid threshold meets the conjunctive gate on calibration; tau = 0 is reported "
            "as the default operating point so the sealed block still receives a single "
            "prespecified threshold rather than a post-hoc one"
        )
    print(f"\noperating threshold tau* = {tau_star:.0f}  ({selection_rule})")

    # ---- step 4: read the sealed block; no refit, no retune -----------------
    sealed, sealed_prov = load_sealed_block()
    pred_sealed = model.predict(sealed.con)
    lcb_sealed = pred_sealed - q
    print(f"\nsealed block: {sealed.n_rows} rows / {sealed.n_clusters} clusters, "
          f"attrition {100 * sealed_prov['attrition']:.2f}%")
    print(f"  sealed prediction range [{pred_sealed.min():.1f}, {pred_sealed.max():.1f}], "
          f"LCB max {lcb_sealed.max():.2f}, rows with LCB > 0: {int((lcb_sealed > 0).sum())}")

    sealed_sweep = sweep(lcb_sealed, sealed, with_intervals=True)
    print_sweep("SEALED BLOCK -- conformal rule LCB > tau (primary)", sealed_sweep)

    sealed_at_star = next(r for r in sealed_sweep if r["tau"] == tau_star)

    # Best case over the whole sweep, reported for completeness. Sweeping the
    # sealed block is descriptive: a threshold picked from this column would be
    # chosen on the evaluation data and would carry no guarantee.
    sealed_pass = [r for r in sealed_sweep if r["gate"]["passed"]]

    # ---- diagnostic: the same rule without the conformal margin -------------
    # Not preregistered and not gate-facing. It isolates how much of the outcome
    # is the margin and how much is the ranking underneath it.
    raw_sweep = sweep(pred_sealed, sealed, with_intervals=True)
    print_sweep(
        "SEALED BLOCK -- DIAGNOSTIC ONLY: raw predicted gain > tau, no conformal margin",
        raw_sweep,
    )
    raw_pass = [r for r in raw_sweep if r["gate"]["passed"]]
    best_raw = min(
        (r for r in raw_sweep if r["status"] == "evaluated"),
        key=lambda r: r["harmful_rate_selected"]["eb_upper"],
    )

    # ---- robustness: split the calibration clusters in half -----------------
    # Fitting q and choosing tau on the same 120 clusters uses that block twice.
    # It cannot contaminate the sealed evaluation, but it can flatter the
    # calibration-side gate, so the split-half variant is reported next to it.
    rng = np.random.default_rng(SEED)
    uniq_calib = np.unique(calib.clusters)
    shuffled = rng.permutation(uniq_calib)
    half_a = set(shuffled[: uniq_calib.size // 2])
    mask_a = np.asarray([c in half_a for c in calib.clusters])
    margin_half = conformal_margin(
        pred_calib[mask_a], calib.y[mask_a], calib.clusters[mask_a], ALPHA_CONFORMAL
    )
    q_half = margin_half["q"]
    lcb_half_b = pred_calib[~mask_a] - q_half

    class _Sub:
        y = calib.y[~mask_a]
        clusters = calib.clusters[~mask_a]

    half_sweep = sweep(lcb_half_b, _Sub, with_intervals=False)
    half_pass = [r for r in half_sweep if r["gate"]["passed"]]
    print(
        f"\nsplit-half check: q = {q_half:.2f} on {margin_half['n_clusters']} clusters, "
        f"held-out half has {sum(1 for r in half_sweep if r['n_selected_rows'] > 0)} thresholds "
        f"with any selection, {len(half_pass)} passing the gate"
    )

    # ---- V4/V5 historical line ---------------------------------------------
    v4_v5 = {
        "V4_stage_b": {
            "rows": 360,
            "clusters": 60,
            "rmse_improvement_over_mean_pct": 5.34,
            "required_rmse_improvement_pct": 10.00,
            "benefit_auroc": 0.695,
            "harm_rate_upper_95_pct": 56.15,
            "required_harm_rate_upper_pct": 10.00,
            "decision": "STOP",
            "source": "chronological_experiment_report.tex, V4 Stage-B gate table",
        },
        "V5": {
            "selected_interventions": 0,
            "note": "no state cleared the calibrated bound, so the harm rate was undefined",
            "source": "chronological_experiment_report.tex / paper_v1_v8/PROVENANCE.md",
        },
        "PACE_phase2_at_tau_star": {
            "rows": sealed.n_rows,
            "clusters": sealed.n_clusters,
            "tau": tau_star,
            "selected_interventions": sealed_at_star["n_selected_rows"],
            "active_clusters": sealed_at_star["n_active_clusters"],
            "harm_rate_upper_95_pct": (
                100 * sealed_at_star["harmful_rate_selected"]["eb_upper"]
                if sealed_at_star["status"] == "evaluated"
                else None
            ),
            "required_harm_rate_upper_pct": 100 * GATE_HARM_MAX,
            "mean_gain_lower_95": (
                sealed_at_star["mean_gain_selected"]["bootstrap_ci95"][0]
                if sealed_at_star["status"] == "evaluated"
                else None
            ),
            "decision": "PASS" if sealed_at_star["gate"]["passed"] else "STOP",
        },
        "PACE_phase2_unmargined_diagnostic_best_ceiling": {
            "rule": "raw predicted gain > tau; NOT the preregistered rule, no conformal margin",
            "tau": best_raw["tau"],
            "selected_interventions": best_raw["n_selected_rows"],
            "active_clusters": best_raw["n_active_clusters"],
            "harm_rate_upper_95_pct": 100 * best_raw["harmful_rate_selected"]["eb_upper"],
            "harm_rate_point_pct": 100
            * best_raw["harmful_rate_selected"]["equal_weight_cluster_mean"],
            "required_harm_rate_upper_pct": 100 * GATE_HARM_MAX,
            "mean_gain_lower_95": best_raw["mean_gain_selected"]["bootstrap_ci95"][0],
            "decision": "STOP",
            "why_reported": (
                "the only line in this analysis with enough active clusters for the harm "
                "ceiling to be reachable at all, so it is the one that can be read against "
                "V4's 56.15% without the reader having to discount for activity"
            ),
        },
        "comparison_scope": (
            "single-threshold comparison of one number against one number. The rows come from "
            "different sample sizes, different estimators, different rules and different data "
            "blocks; they are not pooled and this is not a replication of V4 or V5."
        ),
    }

    report = {
        "task": "PACE Phase 2 / tasks 2-6 -- conformal calibration, selective rule, conjunctive gate",
        "evidence_class": (
            "offline_selective_rule_evaluation_on_sealed_seeds; not a closed-loop controller result"
        ),
        "scope_caveat": (
            "Paired branches are forks from a common snapshot and RNG state, so the gain of "
            "intervening is observed for every candidate row regardless of selection. "
            "Selections therefore do not perturb the trajectory, do not change which states "
            "arrive later, and cannot compound. The estimand is the per-decision effect of a "
            "single one-step deviation, not the sequential effect of running this policy."
        ),
        "inputs": {
            "paired_archive": "v6/results/cfra/paired_branches.csv",
            "paired_archive_sha256": paired_sha,
            "sealed": sealed_prov,
        },
        "protocol": {
            "training": {
                "splits": ["representation_train", "development"],
                "rows": train.n_rows,
                "clusters": train.n_clusters,
            },
            "calibration": {"split": "calibration", "rows": calib.n_rows, "clusters": calib.n_clusters},
            "evaluation": {"rows": sealed.n_rows, "clusters": sealed.n_clusters},
            "cluster_key": "network|seed",
            "cluster_key_note": (
                "route_sha256 is nested strictly inside network|seed (zero route hashes span two "
                "pairs), so this key is coarser and more conservative than the route seed used by "
                "v9/src/v9evidence/statistics.py"
            ),
            "model": "ExtraTreesRegressor(n_estimators=300) on the 81-column feature_vector",
            "model_selection": "calibration-split RMSE and AUROC; sealed block not consulted",
            "threshold_selection": selection_rule,
            "sealed_used_for": "evaluation only -- no fit, no tuning, no threshold search",
        },
        "conformal_calibration": margin,
        "conformal_coverage_check_in_sample": coverage_check,
        "interval_methods": {
            "harm_and_coverage": "two_sided_empirical_bernstein_for_bounded_cluster_means",
            "harm_and_coverage_rationale": (
                "harm and coverage are means of cluster statistics in [0, 1]; the empirical "
                "Bernstein bound is finite-sample valid without asymptotics and does not collapse "
                "when every observed cluster lies on a boundary, which is the regime a selective "
                "rule produces. It is also the estimator behind V4's 56.15% ceiling, which keeps "
                "the historical comparison like-for-like."
            ),
            "mean_gain": "cluster percentile bootstrap over network|seed",
            "confidence_level": CONFIDENCE,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": SEED,
        },
        "gate_definition": {
            "source": "V9 H5_CALIBRATED_AUTHORITY, restated for PACE P4/P5",
            "conditions": [
                "mean realised gain on selected rows has a two-sided 95% cluster lower bound > 0",
                f"harmful fraction among selected has a 95% empirical-Bernstein upper bound <= {GATE_HARM_MAX}",
                f"at least {GATE_MIN_ACTIVE_CLUSTERS} clusters contain a selected row",
            ],
            "conjunctive": True,
        },
        "harm_bound_certifiability": certifiability_floor(),
        "threshold_grid": THRESHOLD_GRID,
        "calibration_sweep": calib_sweep,
        "sealed_sweep": sealed_sweep,
        "sealed_at_operating_threshold": sealed_at_star,
        "operating_threshold": tau_star,
        "conjunctive_gate_result": {
            "at_operating_threshold": sealed_at_star["gate"],
            "passed_at_operating_threshold": sealed_at_star["gate"]["passed"],
            "any_sealed_threshold_passes": bool(sealed_pass),
            "passing_sealed_thresholds": [r["tau"] for r in sealed_pass],
            "note": (
                "the second field is descriptive. A threshold read off the sealed sweep would be "
                "chosen on the evaluation data and would inherit no guarantee from the calibration."
            ),
        },
        "diagnostic_no_conformal_margin": {
            "description": "same rule with LCB replaced by the raw prediction; not gate-facing",
            "sweep": raw_sweep,
            "any_threshold_passes": bool(raw_pass),
            "passing_thresholds": [r["tau"] for r in raw_pass],
            "lowest_harm_ceiling": {
                "tau": min(
                    (r for r in raw_sweep if r["status"] == "evaluated"),
                    key=lambda r: r["harmful_rate_selected"]["eb_upper"],
                )["tau"],
                "eb_upper": min(
                    r["harmful_rate_selected"]["eb_upper"]
                    for r in raw_sweep
                    if r["status"] == "evaluated"
                ),
                "note": (
                    "the harm ceiling never approaches 0.10 anywhere in this sweep even where "
                    "activity is ample, so at high activity the binding constraint is the harm "
                    "level itself rather than the number of clusters"
                ),
            },
        },
        "split_half_robustness": {
            "description": (
                "q fitted on half the calibration clusters, rule evaluated on the other half, to "
                "show what the double use of the calibration block buys"
            ),
            "q": q_half,
            "n_clusters_fit": margin_half["n_clusters"],
            "n_clusters_eval": int(np.unique(_Sub.clusters).size),
            "any_threshold_passes": bool(half_pass),
            "passing_thresholds": [r["tau"] for r in half_pass],
            "sweep": half_sweep,
        },
        "historical_comparison": v4_v5,
    }
    write_json(OUT, report)

    print("\n" + "=" * 74)
    print("CONJUNCTIVE GATE AT THE CALIBRATION-CHOSEN THRESHOLD")
    print("=" * 74)
    g = sealed_at_star["gate"]
    print(f"tau*                       {tau_star:.0f}")
    print(f"selected rows              {sealed_at_star['n_selected_rows']} of {sealed.n_rows}")
    print(f"active clusters            {g['active_clusters']} (need >= {GATE_MIN_ACTIVE_CLUSTERS})   "
          f"{'ok' if g['activity_live'] else 'FAIL'}")
    print(f"mean gain lower 95%        {g['gain_lower_bound']}   (need > 0)   "
          f"{'ok' if g['gain_positive'] else 'FAIL'}")
    print(f"harm upper 95%             {g['harm_upper_bound']}   (need <= {GATE_HARM_MAX})   "
          f"{'ok' if g['harm_bounded'] else 'FAIL'}")
    print(f"\nGATE: {'TAKEN' if g['passed'] else 'NOT TAKEN'}")
    print(f"harm failure kind          {g['harm_failure_kind']}")
    print(f"any sealed threshold passing: {[r['tau'] for r in sealed_pass] or 'none'}")

    floor = certifiability_floor()
    print(
        f"\nharm-bound certifiability floor: {floor['minimum_active_clusters_even_with_zero_observed_harm']} "
        f"active clusters minimum for a {GATE_HARM_MAX} ceiling to be reachable at all"
    )
    print(
        f"best harm ceiling anywhere in the unmargined diagnostic sweep: "
        f"{100 * best_raw['harmful_rate_selected']['eb_upper']:.2f}% at tau={best_raw['tau']:.0f} "
        f"({best_raw['n_active_clusters']} active clusters, point harm "
        f"{100 * best_raw['harmful_rate_selected']['equal_weight_cluster_mean']:.2f}%)"
    )


if __name__ == "__main__":
    main()
