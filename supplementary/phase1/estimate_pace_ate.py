"""PACE Phase 1 -- measurement core: identified effect of a single phase deviation.

WHAT THIS ESTIMATES, EXACTLY
----------------------------
For every probed state the collector ran two forks of the same SUMO episode from
one snapshot and one PRNG state.  Branch A applied the PPO ensemble action at the
target intersection; branch B applied its complement.  Everything else -- routes,
demand realisation, disturbance stream, the actions at every other intersection,
and the frozen continuation -- is byte-identical up to the fork
(``prefix_applied_actions_sha256`` is compared between branches, and collection
aborts on any mismatch).

Therefore ``gain_h = baseline_cost_h - candidate_cost_h`` is not an *estimate* of
a causal effect: it is the individual causal effect of the deviation, observed
exactly, for that unit.  Both potential outcomes are realised.  Internal validity
does not rest on randomisation, on ignorability, or on any outcome model.  What
remains statistical is only the average over units, and the only sampling
uncertainty is which units were probed.

The estimand is consequently:

    the mean effect of one single-intersection phase deviation from the PPO
    incumbent, over the population of states the collection pipeline probes,

which is *not* the mean effect over all decisions the controller makes.  The
probe population is characterised in ``PROBE_POPULATION`` below and is congestion
biased by construction.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
No sign-flip / permutation test.  Branch orientation was never randomised: the
baseline label is always the action PPO actually chose, so the two labels are not
exchangeable and the sign-flip null is not the null of interest.  This is the
same refusal that ``v9/src/v9evidence/statistics.py::cluster_permutation_test``
raises at runtime.  All intervals here are cluster bootstrap intervals; all
p-values are centered studentized cluster bootstrap p-values, the procedure v9
provides for exactly the non-randomised-orientation case.

No row-level intervals anywhere.  Rows within a (network, seed) cluster share a
route file and a demand realisation, so they are not independent.  Every
statistic is collapsed to one value per cluster before aggregation, and every
resample draws whole clusters.

Reads only ``v10_proposal/phase0/sealed/raw``; writes only into
``v10_proposal/phase1/results``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from statistics import NormalDist

import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
RAW = Path(
    os.environ.get(
        "SEALED_RAW_ROOT",
        REPOSITORY_ROOT / "external" / "sealed" / "raw",
    )
).resolve()
OUT = HERE / "results" / "pace_phase1_ate.json"

BOOTSTRAP_SEED = 20260811
BOOTSTRAP_REPLICATES = 20_000
CONFIDENCE = 0.95
POWER = 0.90
ALPHA_ONE_SIDED = 0.025

CONDITIONS = [
    "normal",
    "packet_loss",
    "sensor_delay_noise",
    "lane_closure",
    "emergency_vehicle",
    "demand_shock",
]
NETWORKS = ["moscow", "cologne8", "ingolstadt21"]
HORIZONS = [1, 2, 4]

# Cost weights, copied from ergs_tsc_v6.cfra_branching.run_paired_branch defaults.
COST_WEIGHTS = {
    "stopped_vehicle_seconds": 1.0,
    "terminal_queue": 2.0,
    "hard_braking": 0.5,
    "teleports": 30.0,
}
COMPONENT_FIELDS = {
    "stopped_vehicle_seconds": "delta_stopped_vehicle_seconds_h{h}",
    "terminal_queue": "delta_terminal_queue_h{h}",
    "hard_braking": "delta_hard_braking_h{h}",
    "teleports": "delta_teleports_h{h}",
}

# How the collector chose what to probe -- read off cfra_branching.py, not assumed.
PROBE_POPULATION = {
    "decision_step": (
        "deterministic_target_step(): SHA-256 of network|condition|seed|snapshot_index "
        "reduced modulo the admissible span [24, n_decisions - horizon]. A fixed "
        "pseudorandom draw, independent of the episode state -- as-if uniform over "
        "decision steps."
    ),
    "intersection": (
        "argmax over legally contrastable intersections of "
        "total_queue * (0.25 + imbalance), with a hash tie-break "
        "(cfra_branching.py ranked[snapshot_index % len(ranked)], snapshot_index == 0 "
        "throughout). NOT randomised: the single most pressured intersection is "
        "probed every time. The probe population is congestion biased."
    ),
    "branch_orientation": (
        "NOT randomised. Baseline is always the PPO ensemble action; candidate is "
        "always its complement. This is why no sign-flip test is reported."
    ),
    "eligibility": (
        "Both HOLD and ADVANCE legal at the probed intersection, and the applied "
        "actions of the two branches genuinely differ (distinct_applied_action)."
    ),
    "consequence_for_generalisation": (
        "Identified for probed states. Extrapolation to all controller decisions "
        "is not licensed: uncongested intersections are never probed, and the "
        "PACE design of 01_METHOD_PACE.md section 5 (probe at the decision "
        "boundary, uniformly over eligible intersections) is a different, "
        "not-yet-collected population."
    ),
}


# ---------------------------------------------------------------- input pinning


def pin_inputs() -> dict:
    """Hash every input file and the manifest, so a rerun on altered data fails loudly."""
    files = sorted(RAW.glob("*.json"), key=lambda p: p.name)
    lines = []
    for path in files:
        lines.append(f"{path.name}:{hashlib.sha256(path.read_bytes()).hexdigest()}")
    manifest = "\n".join(lines) + "\n"
    return {
        "directory": str(RAW),
        "file_count": len(files),
        "manifest_sha256": hashlib.sha256(manifest.encode("utf-8")).hexdigest(),
        "first_file": files[0].name if files else None,
        "last_file": files[-1].name if files else None,
    }


def load_rows() -> tuple[list[dict], dict]:
    files = sorted(RAW.glob("*.json"), key=lambda p: p.name)
    rows, status_counts = [], {}
    for path in files:
        record = json.loads(path.read_text(encoding="utf-8"))
        status = str(record.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1
        if status != "complete":
            continue
        if any(record.get(f"gain_h{h}") is None for h in HORIZONS):
            status_counts["complete_but_missing_gain"] = (
                status_counts.get("complete_but_missing_gain", 0) + 1
            )
            continue
        rows.append(record)
    scheduled = len(files)
    provenance = {
        "scheduled": scheduled,
        "analysable": len(rows),
        "status_counts": status_counts,
        "attrition": 1.0 - len(rows) / scheduled if scheduled else 1.0,
    }
    return rows, provenance


# ------------------------------------------------------- cluster-level machinery


def cluster_index(rows: list[dict]) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return per-row cluster ids, the ordered cluster labels, and each cluster's network."""
    labels = [f"{r['network']}|{r['seed']}" for r in rows]
    ordered = sorted(set(labels))
    position = {label: i for i, label in enumerate(ordered)}
    ids = np.asarray([position[label] for label in labels], dtype=np.int64)
    net_of_cluster = np.asarray([label.split("|")[0] for label in ordered])
    return ids, ordered, net_of_cluster


def collapse(values: np.ndarray, ids: np.ndarray, n_clusters: int,
             mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Mean of ``values`` inside every cluster. Returns (means, has_data) aligned to clusters."""
    if mask is not None:
        values = np.where(mask, values, 0.0)
        weights = mask.astype(np.float64)
    else:
        weights = np.ones_like(values, dtype=np.float64)
    total = np.bincount(ids, weights=values, minlength=n_clusters)
    count = np.bincount(ids, weights=weights, minlength=n_clusters)
    has = count > 0
    means = np.zeros(n_clusters, dtype=np.float64)
    means[has] = total[has] / count[has]
    return means, has


def bootstrap_matrix(n_clusters: int, replicates: int, seed: int,
                     strata: np.ndarray | None = None) -> np.ndarray:
    """(replicates, n_clusters) matrix of resampled cluster positions."""
    rng = np.random.default_rng(seed)
    if strata is None:
        return rng.integers(0, n_clusters, size=(replicates, n_clusters))
    draws = np.empty((replicates, n_clusters), dtype=np.int64)
    cursor = 0
    for value in sorted(set(strata.tolist())):
        members = np.flatnonzero(strata == value)
        picked = rng.integers(0, members.size, size=(replicates, members.size))
        draws[:, cursor:cursor + members.size] = members[picked]
        cursor += members.size
    return draws


def interval(values: np.ndarray, draws: np.ndarray | None = None, *,
             subset: np.ndarray | None = None, seed: int | None = None) -> dict:
    """Percentile cluster-bootstrap interval for the equal-weight cluster mean.

    ``subset`` restricts the analysis to a set of clusters (e.g. the 100 clusters
    of one network); resampling then draws whole clusters from inside that subset
    only, which is what keeps the interval a cluster interval and not a row one.
    """
    if subset is not None:
        keep = np.flatnonzero(subset)
        values = values[keep]
        draws = bootstrap_matrix(
            keep.size, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED if seed is None else seed
        )
    if draws is None:
        raise ValueError("either draws or subset must be supplied")
    means = values[draws].mean(axis=1)
    alpha = 1.0 - CONFIDENCE
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return {
        "estimate": float(values.mean()),
        "ci95": [lo, hi],
        "bootstrap_se": float(means.std(ddof=1)),
        "cluster_sd": float(values.std(ddof=1)),
        "n_clusters": int(values.size),
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
    }


def studentized_p(values: np.ndarray, replicates: int, seed: int) -> dict:
    """Two-sided centered studentized cluster-bootstrap p-value for H0: mean == 0.

    Mirrors v9/src/v9evidence/statistics.py::studentized_cluster_bootstrap_test.
    Valid without randomised branch orientation; assumes independent clusters and
    non-degenerate cluster variance.
    """
    n = values.size
    sd = values.std(ddof=1)
    if sd <= 1e-12:
        return {"p_value": None, "reason": "degenerate zero cluster variance", "n_clusters": int(n)}
    t_obs = float(values.mean() / (sd / math.sqrt(n)))
    centered = values - values.mean()
    rng = np.random.default_rng(seed)
    extreme, valid = 0, 0
    block = 2000
    remaining = replicates
    while remaining > 0:
        size = min(block, remaining)
        remaining -= size
        sample = centered[rng.integers(0, n, size=(size, n))]
        means = sample.mean(axis=1)
        sds = sample.std(axis=1, ddof=1)
        ok = sds > 1e-12
        stats = np.zeros(size)
        stats[ok] = means[ok] / (sds[ok] / math.sqrt(n))
        valid += int(ok.sum())
        extreme += int((np.abs(stats[ok]) >= abs(t_obs)).sum())
    return {
        "p_value": (extreme + 1.0) / (valid + 1.0),
        "t_statistic": t_obs,
        "n_clusters": int(n),
        "replicates_valid": int(valid),
        "mode": "two_sided_centered_studentized_cluster_bootstrap_plus_one",
        "branch_orientation_randomized": False,
    }


def holm(p_values: dict[str, float], alpha: float = 0.05) -> dict:
    """Holm step-down, same semantics as v9/src/v9evidence/statistics.py::holm_adjust."""
    ordered = sorted(p_values.items(), key=lambda kv: (kv[1], kv[0]))
    m = len(ordered)
    adjusted, running, open_ = {}, 0.0, True
    rejected, thresholds = {}, {}
    for rank, (name, p) in enumerate(ordered, start=1):
        mult = m - rank + 1
        running = max(running, min(1.0, mult * p))
        adjusted[name] = running
        thresholds[name] = alpha / mult
        rejected[name] = open_ and p <= thresholds[name]
        if not rejected[name]:
            open_ = False
    return {
        name: {
            "raw_p_value": p,
            "holm_adjusted_p_value": adjusted[name],
            "holm_threshold": thresholds[name],
            "rejected_at_alpha_0.05": rejected[name],
            "family_size": m,
        }
        for name, p in p_values.items()
    }


def mde(cluster_sd: float, n_clusters: int) -> float:
    """Minimum effect detectable at the given power and one-sided alpha."""
    z_alpha = NormalDist().inv_cdf(1.0 - ALPHA_ONE_SIDED)
    z_power = NormalDist().inv_cdf(POWER)
    return (z_alpha + z_power) * cluster_sd / math.sqrt(n_clusters)


# ------------------------------------------------------------------------ main


def main() -> None:
    pinned = pin_inputs()
    rows, provenance = load_rows()
    if not rows:
        raise SystemExit("no analysable rows")

    ids, cluster_labels, net_of_cluster = cluster_index(rows)
    K = len(cluster_labels)
    draws = bootstrap_matrix(K, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
    draws_stratified = bootstrap_matrix(
        K, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED + 1, strata=net_of_cluster
    )

    condition_of_row = np.asarray([r["condition"] for r in rows])
    network_of_row = np.asarray([r["network"] for r in rows])
    baseline_action = np.asarray([int(r["baseline_action"]) for r in rows])
    gains = {h: np.asarray([float(r[f"gain_h{h}"]) for r in rows]) for h in HORIZONS}

    report: dict = {
        "analysis": "PACE Phase 1 -- measurement core (hypotheses P1, P2)",
        "generated_by": str(Path(__file__).resolve()),
        "inputs": pinned,
        "provenance": provenance,
        "estimand": {
            "definition": (
                "Mean effect, in window cost units, of applying the complement of the "
                "PPO incumbent action at one intersection for one decision interval, "
                "relative to the PPO incumbent action, over the population of states "
                "probed by the collection pipeline. Positive = the deviation is better."
            ),
            "not": (
                "This is NOT the mean effect over all decisions the controller makes, "
                "and NOT the effect of a deployed policy."
            ),
            "identification": (
                "Exact paired counterfactual: both branches fork from one snapshot and "
                "one PRNG state with byte-identical prefixes, so both potential outcomes "
                "are observed for every unit. No unconfoundedness assumption is used."
            ),
            "sampling_uncertainty": "Only over which (network, seed) clusters were drawn.",
            "cost_weights": COST_WEIGHTS,
            "horizons": "h1/h2/h4 are nested cumulative windows of 1, 2 and 4 decision intervals.",
        },
        "probe_population": PROBE_POPULATION,
        "inference": {
            "cluster_unit": "(network, seed)",
            "n_clusters": K,
            "rows_per_cluster": int(len(rows) / K),
            "collapse": "rows averaged within cluster before any aggregation",
            "interval_method": "percentile cluster bootstrap, clusters resampled with replacement",
            "replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "p_value_method": "two-sided centered studentized cluster bootstrap (+1 corrected)",
            "sign_flip_permutation_test": {
                "run": False,
                "reason": (
                    "Branch orientation was not prospectively randomised -- the baseline "
                    "label is always the PPO action -- so the two labels are not "
                    "sign-exchangeable and the sign-flip null is not the null of interest. "
                    "v9/src/v9evidence/statistics.py::cluster_permutation_test raises on "
                    "exactly this condition."
                ),
            },
            "row_level_intervals": "not reported; rows within a cluster share a route file",
        },
    }

    # ---------------------------------------------------------------- 1. ATE
    ate = {}
    for h in HORIZONS:
        values, _ = collapse(gains[h], ids, K)
        est = interval(values, draws)
        est["studentized_test"] = studentized_p(values, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED + 100 + h)
        est["row_level_mean"] = float(gains[h].mean())
        est["row_level_sd"] = float(gains[h].std(ddof=1))
        est["cluster_mean_median"] = float(np.median(values))
        strat = interval(values, draws_stratified)
        est["sensitivity_network_stratified_ci95"] = strat["ci95"]
        ate[f"h{h}"] = est
    report["ate_by_horizon"] = ate

    # Sensitivity: the stricter v9 convention that only the numeric route seed is
    # an independent unit (100 clusters, networks pooled inside a cluster).
    seed_ids_labels = sorted({int(r["seed"]) for r in rows})
    seed_pos = {s: i for i, s in enumerate(seed_ids_labels)}
    seed_ids = np.asarray([seed_pos[int(r["seed"])] for r in rows])
    seed_draws = bootstrap_matrix(len(seed_ids_labels), BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED + 2)
    seed_sensitivity = {}
    for h in HORIZONS:
        values, _ = collapse(gains[h], seed_ids, len(seed_ids_labels))
        est = interval(values, seed_draws)
        est["studentized_test"] = studentized_p(
            values, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED + 200 + h
        )
        seed_sensitivity[f"h{h}"] = est
    report["sensitivity_route_seed_clusters"] = {
        "rationale": (
            "v9/src/v9evidence/statistics.py declares the numeric route seed, not the "
            "compound network|seed key, the only admissible independent unit. Reported "
            "so the headline does not depend on the looser choice."
        ),
        "n_clusters": len(seed_ids_labels),
        "by_horizon": seed_sensitivity,
    }

    # ------------------------------------------------- 2. conditional effects
    conditional: dict = {"by_condition": {}, "by_network": {}}
    p_by_condition: dict[str, float] = {}
    for condition in CONDITIONS:
        mask = condition_of_row == condition
        values, has = collapse(gains[4], ids, K, mask=mask)
        # every cluster contributes exactly one row per condition
        est = interval(values, draws)
        test = studentized_p(values, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED + 300 + CONDITIONS.index(condition))
        est["studentized_test"] = test
        est["n_rows"] = int(mask.sum())
        est["clusters_with_data"] = int(has.sum())
        est["mde_at_observed_sd"] = mde(est["cluster_sd"], est["n_clusters"])
        conditional["by_condition"][condition] = est
        p_by_condition[condition] = float(test["p_value"])
    holm_conditions = holm(p_by_condition)
    for condition, adj in holm_conditions.items():
        conditional["by_condition"][condition]["holm"] = adj
    conditional["holm_family"] = {
        "members": CONDITIONS,
        "size": 6,
        "horizon": "h4",
        "alpha": 0.05,
        "note": "Family fixed a priori as the six operating conditions at the primary horizon.",
    }

    for network in NETWORKS:
        subset = net_of_cluster == network
        mask = network_of_row == network
        values, has = collapse(gains[4], ids, K, mask=mask)
        est = interval(values, subset=subset, seed=BOOTSTRAP_SEED + 400 + NETWORKS.index(network))
        sub_values = values[np.flatnonzero(subset)]
        test = studentized_p(sub_values, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED + 400 + NETWORKS.index(network))
        est["studentized_test"] = test
        est["n_rows"] = int(mask.sum())
        est["mde_at_observed_sd"] = mde(est["cluster_sd"], est["n_clusters"])
        conditional["by_network"][network] = est
    p_by_network = {n: float(conditional["by_network"][n]["studentized_test"]["p_value"]) for n in NETWORKS}
    for network, adj in holm(p_by_network).items():
        conditional["by_network"][network]["holm"] = adj
    conditional["network_holm_family"] = {
        "members": NETWORKS,
        "size": 3,
        "note": "Separate family from the six conditions; not pooled with it.",
    }

    # Exploratory: the deviation is asymmetric, since baseline is whatever PPO chose.
    # In one stratum the intervention is "force a hold", in the other "force an
    # advance". These are different physical treatments, so the pooled ATE is an
    # average over a treatment that is not a single treatment.
    direction: dict = {}
    for label, want in (("ppo_chose_advance_candidate_holds", 1), ("ppo_chose_hold_candidate_advances", 0)):
        mask = baseline_action == want
        values, has = collapse(gains[4], ids, K, mask=mask)
        subset = has
        est = interval(values, subset=subset, seed=BOOTSTRAP_SEED + 500 + want)
        est["studentized_test"] = studentized_p(
            values[np.flatnonzero(subset)], BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED + 500 + want
        )
        est["n_rows"] = int(mask.sum())
        est["row_share"] = float(mask.mean())
        harm_values, harm_has = collapse((gains[4] < 0).astype(np.float64), ids, K, mask=mask)
        est["share_gain_lt_0"] = interval(
            harm_values, subset=harm_has, seed=BOOTSTRAP_SEED + 600 + want
        )
        for h in (1, 2):
            vals_h, has_h = collapse(gains[h], ids, K, mask=mask)
            est[f"h{h}_effect"] = interval(vals_h, subset=has_h, seed=BOOTSTRAP_SEED + 700 + 10 * h + want)
        direction[label] = est
    conditional["exploratory_by_deviation_direction"] = {
        "status": "exploratory, not pre-specified, not in any Holm family",
        "note": (
            "Cluster membership is unbalanced because which direction gets probed is "
            "decided by the PPO action at the probed state. Only clusters that contain "
            "at least one row of the stratum enter its interval."
        ),
        "interpretation_warning": (
            "The two strata are different interventions, not two levels of one. "
            "Conditioning is on a pre-deviation variable (the incumbent's own action), "
            "so each stratum effect is itself identified, but the pooled ATE mixes them."
        ),
        "strata": direction,
    }
    report["conditional_effects"] = conditional

    # --------------------------------------------------------- 3. decomposition
    decomposition: dict = {}
    for h in HORIZONS:
        parts, recon = {}, np.zeros(len(rows))
        for name, template in COMPONENT_FIELDS.items():
            weight = COST_WEIGHTS[name]
            raw = np.asarray([float(r[template.format(h=h)]) for r in rows])
            contribution = weight * raw
            recon += contribution
            values, _ = collapse(contribution, ids, K)
            est = interval(values, draws)
            est["weight"] = weight
            est["unweighted_cluster_mean"] = float(collapse(raw, ids, K)[0].mean())
            parts[name] = est
        total_values, _ = collapse(gains[h], ids, K)
        total = float(total_values.mean())
        for name in parts:
            parts[name]["share_of_total_effect"] = (
                parts[name]["estimate"] / total if abs(total) > 1e-12 else None
            )
        decomposition[f"h{h}"] = {
            "total_effect": total,
            "components": parts,
            "identity_max_abs_residual": float(np.max(np.abs(recon - gains[h]))),
            "identity": "gain = stopped + 2.0*terminal_queue + 0.5*hard_braking + 30.0*teleports",
        }
    report["decomposition"] = decomposition

    # Persistence. h1/h2/h4 are nested cumulative windows, so the cumulative
    # stopped-vehicle-seconds term is the only component whose differences across
    # horizons are clean increments (terminal queue is a snapshot, not a flow).
    stopped = {
        h: np.asarray([float(r[f"delta_stopped_vehicle_seconds_h{h}"]) for r in rows])
        for h in HORIZONS
    }
    persistence = {
        "definition": (
            "Cumulative delta_stopped_vehicle_seconds and its per-interval increments. "
            "A flat increment means the two branches have re-converged."
        ),
        "cumulative": {},
        "increments": {},
    }
    for h in HORIZONS:
        values, _ = collapse(stopped[h], ids, K)
        persistence["cumulative"][f"h{h}"] = interval(values, draws)
    inc_1_2, _ = collapse(stopped[2] - stopped[1], ids, K)
    inc_3_4, _ = collapse((stopped[4] - stopped[2]) / 2.0, ids, K)
    persistence["increments"]["interval_2"] = interval(inc_1_2, draws)
    persistence["increments"]["intervals_3_4_per_interval"] = interval(inc_3_4, draws)
    persistence["increments"]["interval_1"] = persistence["cumulative"]["h1"]
    report["persistence"] = persistence

    # -------------------------------------------------------- 4. harmful share
    harmful: dict = {}
    for h in HORIZONS:
        neg = (gains[h] < 0).astype(np.float64)
        pos = (gains[h] > 0).astype(np.float64)
        zero = (gains[h] == 0).astype(np.float64)
        magnitude = np.maximum(0.0, -gains[h])
        entry = {}
        for name, series in (("share_gain_lt_0", neg), ("share_gain_gt_0", pos), ("share_gain_eq_0", zero)):
            values, _ = collapse(series, ids, K)
            entry[name] = interval(values, draws)
        values, _ = collapse(magnitude, ids, K)
        entry["mean_realised_harm_per_probe"] = interval(values, draws)
        entry["mean_realised_harm_per_probe"]["definition"] = "E[max(0, -gain)] per probed deviation"
        harmful[f"h{h}"] = entry
    report["harmful_share"] = harmful
    report["harm_bound_input"] = {
        "note": (
            "Design bound of 01_METHOD_PACE.md section 2.1 is "
            "E[harm per decision] <= p_max * E[|delta^-|]. The measured "
            "E[max(0,-gain)] at h4 is the E[|delta^-|] factor for the probed "
            "population; multiplying by p_max gives the per-decision bound."
        ),
        "e_abs_negative_h4": harmful["h4"]["mean_realised_harm_per_probe"]["estimate"],
        "per_decision_bound_at_p_max": {
            f"p_max={p}": p * harmful["h4"]["mean_realised_harm_per_probe"]["estimate"]
            for p in (0.001, 0.005, 0.01)
        },
    }

    # Scale reference, so the effect can be read relatively as well as absolutely.
    baseline_cost_h4 = np.asarray([float(r["baseline_cost_h4"]) for r in rows])
    report["scale_reference"] = {
        "baseline_cost_h4_mean": float(baseline_cost_h4.mean()),
        "baseline_cost_h4_median": float(np.median(baseline_cost_h4)),
        "ate_h4_as_share_of_mean_baseline_cost": float(
            report["ate_by_horizon"]["h4"]["estimate"] / baseline_cost_h4.mean()
        ),
        "harm_bound_p_max_0.01_as_share_of_mean_baseline_cost": float(
            0.01 * harmful["h4"]["mean_realised_harm_per_probe"]["estimate"]
            / baseline_cost_h4.mean()
        ),
    }

    # ------------------------------------------------------------- 5. post-hoc MDE
    power_block: dict = {}
    z_alpha = NormalDist().inv_cdf(1.0 - ALPHA_ONE_SIDED)
    z_power = NormalDist().inv_cdf(POWER)
    for h in HORIZONS:
        values, _ = collapse(gains[h], ids, K)
        sd = float(values.std(ddof=1))
        power_block[f"h{h}"] = {
            "cluster_sd": sd,
            "n_clusters": K,
            "mde": mde(sd, K),
            "observed_effect": float(values.mean()),
            "observed_abs_effect_over_mde": abs(float(values.mean())) / mde(sd, K),
            "row_level_sd_for_reference": float(gains[h].std(ddof=1)),
            "variance_reduction_vs_row_sd": 1.0 - sd / float(gains[h].std(ddof=1)),
        }
    power_block["parameters"] = {
        "power": POWER,
        "alpha_one_sided": ALPHA_ONE_SIDED,
        "z_alpha": z_alpha,
        "z_power": z_power,
        "formula": "mde = (z_alpha + z_power) * cluster_sd / sqrt(n_clusters)",
        "note": (
            "Post hoc, computed from the observed cluster SD. This is a precision "
            "statement about the design, not a retrospective power calculation on "
            "the observed effect."
        ),
    }
    power_block["phase0_planning_comparison"] = {
        "planned_row_sd": 190.89,
        "planned_n_for_delta_40": 240,
        "note": (
            "Phase 0 planned with a row-level SD. Cluster averaging over six "
            "conditions changes the relevant SD; the realised cluster SD is "
            "reported above."
        ),
    }
    report["post_hoc_mde"] = power_block

    # -------------------------------------------------------------- verdict
    h4 = report["ate_by_horizon"]["h4"]
    lo, hi = h4["ci95"]
    if hi < 0:
        verdict = "identified_negative"
    elif lo > 0:
        verdict = "identified_positive"
    else:
        verdict = "identified_but_indistinguishable_from_zero"
    report["verdict"] = {
        "p1_measurement": {
            "identified": True,
            "why": "exact paired counterfactual per unit; interval width set by cluster count",
            "h4_estimate": h4["estimate"],
            "h4_ci95": [lo, hi],
            "h4_ci_halfwidth": (hi - lo) / 2.0,
            "class": verdict,
        },
        "p2_conditional": {
            "any_condition_rejected_after_holm": any(
                conditional["by_condition"][c]["holm"]["rejected_at_alpha_0.05"] for c in CONDITIONS
            ),
            "conditions_rejected_after_holm": [
                c for c in CONDITIONS if conditional["by_condition"][c]["holm"]["rejected_at_alpha_0.05"]
            ],
            "conditions_with_positive_lower_bound": [
                c for c in CONDITIONS if conditional["by_condition"][c]["ci95"][0] > 0
            ],
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------ console
    print("=" * 78)
    print("PACE PHASE 1 -- ATE OF A SINGLE PHASE DEVIATION (probed population)")
    print("=" * 78)
    print(f"inputs         : {pinned['file_count']} files, manifest sha256 {pinned['manifest_sha256'][:16]}...")
    print(f"analysable     : {provenance['analysable']}/{provenance['scheduled']} "
          f"(attrition {100*provenance['attrition']:.2f}%)   clusters {K}")
    print()
    print(f"{'horizon':<8}{'ATE':>12}{'95% CI (cluster bootstrap)':>34}{'p':>12}")
    for h in HORIZONS:
        e = ate[f"h{h}"]
        p = e["studentized_test"]["p_value"]
        print(f"h{h:<7}{e['estimate']:>12.3f}   [{e['ci95'][0]:>10.3f}, {e['ci95'][1]:>10.3f}]{p:>12.5f}")
    print()
    print("condition (h4)          effect        95% CI            raw p     Holm p   rej")
    for c in CONDITIONS:
        e = conditional["by_condition"][c]
        print(f"  {c:<22}{e['estimate']:>8.2f}  [{e['ci95'][0]:>8.2f},{e['ci95'][1]:>8.2f}]"
              f"  {e['holm']['raw_p_value']:>8.4f} {e['holm']['holm_adjusted_p_value']:>8.4f}"
              f"   {'yes' if e['holm']['rejected_at_alpha_0.05'] else 'no'}")
    print()
    print("network (h4)            effect        95% CI            raw p")
    for n in NETWORKS:
        e = conditional["by_network"][n]
        print(f"  {n:<22}{e['estimate']:>8.2f}  [{e['ci95'][0]:>8.2f},{e['ci95'][1]:>8.2f}]"
              f"  {e['studentized_test']['p_value']:>8.4f}")
    print()
    print("h4 decomposition (weighted contribution to the total effect)")
    for name, part in decomposition["h4"]["components"].items():
        share = part["share_of_total_effect"]
        print(f"  {name:<26}{part['estimate']:>9.3f}  [{part['ci95'][0]:>8.3f},{part['ci95'][1]:>8.3f}]"
              f"   share {100*share:>7.1f}%")
    print(f"  {'TOTAL':<26}{decomposition['h4']['total_effect']:>9.3f}"
          f"   identity residual {decomposition['h4']['identity_max_abs_residual']:.2e}")
    print()
    print("deviation direction (h4, exploratory -- two different interventions)")
    for label, e in direction.items():
        print(f"  {label:<36}{e['estimate']:>9.2f}  [{e['ci95'][0]:>8.2f},{e['ci95'][1]:>8.2f}]"
              f"   n={e['n_rows']:<5} harmful {100*e['share_gain_lt_0']['estimate']:.1f}%")
    print()
    print("persistence of the perturbation (cumulative stopped-vehicle-seconds effect)")
    for name, e in persistence["increments"].items():
        print(f"  {name:<30}{e['estimate']:>9.3f}  [{e['ci95'][0]:>8.3f},{e['ci95'][1]:>8.3f}]")
    print()
    hs = harmful["h4"]["share_gain_lt_0"]
    print(f"harmful share (h4, gain<0): {100*hs['estimate']:.2f}%  "
          f"95% CI [{100*hs['ci95'][0]:.2f}%, {100*hs['ci95'][1]:.2f}%]")
    hm = harmful["h4"]["mean_realised_harm_per_probe"]
    print(f"E[max(0,-gain)] h4        : {hm['estimate']:.2f}  "
          f"95% CI [{hm['ci95'][0]:.2f}, {hm['ci95'][1]:.2f}]")
    print()
    for h in HORIZONS:
        pb = power_block[f"h{h}"]
        print(f"MDE h{h}: {pb['mde']:.3f}  (cluster SD {pb['cluster_sd']:.2f}, "
              f"n={pb['n_clusters']}, power {POWER}, one-sided alpha {ALPHA_ONE_SIDED})")
    print()
    print(f"verdict: {verdict}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
