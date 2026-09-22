"""Is the baseline-ladder overstatement factor invariant to incumbent strength?

The manuscript's Result 1 says the apparent value of learned temporal and graph
structure is an artifact of the reference: an eight-step history adds +0.82 pp
over a capacity-matched no-history control, while the same model gains
+16.38 pp against the naive zero-effect reference -- an overstatement factor of
19.9.

A reviewer will object that the incumbent throughout the corpus is a PPO
ensemble that loses to fixed-time control in all four V1 demand regimes, and
will ask whether the whole finding is an artifact of a weak incumbent.

The paper's claims are about an ESTIMATOR (how well a model predicts the paired
gain), a DECISION RULE, and the composition of the decision stream -- not about
control quality. If that reading is right, the overstatement factor must not
depend on how well the incumbent is coping in the state where the decision is
taken. This script tests exactly that.

Protocol, inherited verbatim from `v10_proposal/phase0/baseline_ladder.py`:
    * one dataset, one split, one fold protocol, one metric
    * train on representation_train + development (960 rows / 160 clusters)
    * evaluate on validation (1800 rows / 300 clusters)
    * RMSE gain over the training-set mean, the V4/V5 Stage-B reference
    * 95% cluster bootstrap over (network, seed)

What is added: the evaluation split is partitioned by a PRE-OUTCOME measure of
incumbent strength, and every rung is recomputed inside each stratum. Models
are fitted ONCE on the full training split and never refitted per stratum --
refitting would change the question and leave 320 training rows per stratum.

The cluster bootstrap resamples clusters ONCE per replicate and recomputes all
strata inside that same resample, so between-stratum contrasts are valid.

Read-only with respect to everything outside this directory.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PAIRED = Path(
    os.environ.get(
        "V6_PAIRED_PATH",
        REPO / "artifacts" / "derived" / "v6" / "paired_branches.csv",
    )
).resolve()
EXPECTED_SHA = "24231bba4eadff8051da0af7705ff429d86c93ec10c554ede9247836aea55d8e"
SEALED_RAW = Path(
    os.environ.get(
        "SEALED_RAW_ROOT",
        REPO / "external" / "sealed" / "raw",
    )
).resolve()
OUT = HERE / "results"

TARGET = "gain_h4"
SEED = 20260805          # same seed as phase0/baseline_ladder.py
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
REPLICATES = 2000
MIN_ROWS_PER_STRATUM = 30   # a bootstrap replicate below this is discarded

# features.SignalObservation.as_array() -- the ten-dimensional PPO observation
OBS_NAMES = ["green_queue", "red_queue", "total_queue", "max_lane_queue",
             "mean_speed", "downstream_queue", "green_age", "phase_progress",
             "queue_delta", "mean_occupancy"]

# Rung index in the ladder, following phase0/baseline_ladder.py exactly.
RUNG_ZERO, RUNG_MEAN, RUNG_CONTRAST_TREES, RUNG_HISTORY_TREES = 0, 1, 5, 6

STRATIFIERS = {
    "local_total_queue": "PRIMARY. Vehicles queued at the controlled signal at the "
                         "decision instant, from the incumbent's own observation "
                         "(context_observation[2] x 100). Higher = incumbent coping worse.",
    "observed_cost": "SignalObservation.observed_cost (features.py:55) evaluated at the "
                     "decision instant: total_queue + 0.35*downstream + 0.10*max(dq,0). "
                     "The repository's own cost functional, one step before the outcome.",
    "policy_entropy": "Entropy of the incumbent PPO ensemble's action distribution.",
    "policy_disagreement": "Disagreement among incumbent PPO ensemble members.",
    "severity": "Semantic event severity at the signal (exogenous task difficulty).",
    "baseline_cost_h4": "DISQUALIFIED as a pre-outcome measure, reported only as a "
                        "leaky diagnostic: it is realised over the same post-decision "
                        "window as the target and gain_h4 = baseline_cost_h4 - "
                        "candidate_cost_h4 exactly.",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_vector(value) -> np.ndarray:
    if isinstance(value, str):
        value = json.loads(value)
    return np.asarray(value, dtype=np.float64).ravel()


def block(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    return np.hstack([np.vstack([as_vector(v) for v in frame[c]]) for c in columns])


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    obs = np.vstack([as_vector(v) for v in df.context_observation])
    for i, name in enumerate(OBS_NAMES):
        df[f"obs_{name}"] = obs[:, i]
    df["local_total_queue"] = obs[:, 2] * 100.0
    downstream = obs[:, 5] * 50.0
    queue_delta = obs[:, 8] * 60.0 - 30.0
    df["observed_cost"] = (df.local_total_queue + 0.35 * downstream
                           + 0.10 * np.clip(queue_delta, 0.0, None))
    df["cluster"] = df.network + "|" + df.seed.astype(str)
    return df


def assign_strata(frame: pd.DataFrame, column: str, k: int, within_network: bool):
    """Return integer stratum labels 0..k-1 and the cutpoints used.

    Ranking breaks ties in row order so that heavily tied columns (severity
    takes six distinct values) still yield balanced strata.
    """
    if within_network:
        labels = frame.groupby("network")[column].transform(
            lambda x: pd.qcut(x.rank(method="first"), k, labels=False)).to_numpy(int)
        cuts = {net: [float(v) for v in np.quantile(g, np.linspace(0, 1, k + 1))]
                for net, g in frame.groupby("network")[column]}
    else:
        labels = pd.qcut(frame[column].rank(method="first"), k, labels=False).to_numpy(int)
        cuts = [float(v) for v in np.quantile(frame[column], np.linspace(0, 1, k + 1))]
    return labels, cuts


def fit_ladder(train: pd.DataFrame, evalu: pd.DataFrame):
    """Fit every rung once on the training split; return predictions on `evalu`."""
    y_tr = train[TARGET].to_numpy(float)
    obs_tr, obs_ev = block(train, ("context_observation",)), block(evalu, ("context_observation",))
    con_tr, con_ev = block(train, ("feature_vector",)), block(evalu, ("feature_vector",))
    hist_cols = ("feature_vector", "context_history", "context_action_history")
    hist_tr, hist_ev = block(train, hist_cols), block(evalu, hist_cols)

    mean_ref = float(np.mean(y_tr))
    n = len(evalu)
    rungs = [
        ("0. zero effect", None, np.zeros(n)),
        ("1. training mean (Stage-B reference)", None, np.full(n, mean_ref)),
    ]
    for name, xt, xe in (("2. ridge, current observation", obs_tr, obs_ev),
                         ("3. ridge, action contrast", con_tr, con_ev)):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(xt, y_tr)
        rungs.append((name, xt.shape[1], m.predict(xe)))
    for name, xt, xe in (("4. trees, current observation", obs_tr, obs_ev),
                         ("5. trees, action contrast", con_tr, con_ev),
                         ("6. trees, contrast + 8-step history", hist_tr, hist_ev)):
        m = ExtraTreesRegressor(n_estimators=300, random_state=SEED, n_jobs=-1).fit(xt, y_tr)
        rungs.append((name, xt.shape[1], m.predict(xe)))
    return rungs, mean_ref


FIELLER_GRID = np.round(np.arange(-400.0, 400.0 + 1e-9, 0.5), 3)


def fieller_set(num_draws: np.ndarray, den_draws: np.ndarray, grid=FIELLER_GRID) -> dict:
    """95% confidence set for the ratio num/den, by inversion.

    A percentile interval on the ratio itself is meaningless when the
    denominator's own interval covers zero -- the ratio then has no bounded
    95% interval and a percentile summary of the draws silently hides that.
    Inversion is the standard repair (Fieller): r belongs to the set iff the
    hypothesis num - r*den = 0 is not rejected, i.e. iff zero lies inside the
    95% bootstrap interval of num - r*den.  The result may be a bounded
    interval, an unbounded set, or the whole line, and each of those is a
    different and honest statement about identification.
    """
    ok = np.isfinite(num_draws) & np.isfinite(den_draws)
    num_draws, den_draws = num_draws[ok], den_draws[ok]
    if num_draws.size < 50:
        return {"kind": "insufficient_replicates", "interval": [None, None]}
    stat = num_draws[:, None] - grid[None, :] * den_draws[:, None]
    lo = np.percentile(stat, 2.5, axis=0)
    hi = np.percentile(stat, 97.5, axis=0)
    inside = (lo <= 0.0) & (hi >= 0.0)
    if not inside.any():
        return {"kind": "empty", "interval": [None, None],
                "note": "no ratio value is consistent with the data at 95%"}
    lo_i, hi_i = int(np.argmax(inside)), int(len(inside) - 1 - np.argmax(inside[::-1]))
    touches_edge = bool(inside[0] or inside[-1])
    contiguous = bool(inside[lo_i:hi_i + 1].all())
    return {
        "kind": ("unbounded" if touches_edge else
                 "bounded" if contiguous else "bounded_with_gap"),
        "interval": [float(grid[lo_i]), float(grid[hi_i])],
        "grid": [float(grid[0]), float(grid[-1])],
        "covers_fraction_of_grid": float(inside.mean()),
        "note": ("the set reaches the edge of the search grid, so the factor is not "
                 "identified at 95% -- the matched-control increment is not "
                 "significantly different from zero here"
                 if touches_edge else "bounded 95% confidence set"),
    }


def stratified_ladder(rungs, mean_ref, y, clusters, strata, k, *,
                      replicates=REPLICATES, seed=SEED):
    """Point estimates and a shared cluster bootstrap for every stratum.

    Squared errors are precomputed per row, so a bootstrap replicate is a
    handful of gathers rather than a refit.
    """
    sq = np.vstack([(pred - y) ** 2 for _, _, pred in rungs])       # (n_rungs, n)
    sq_ref = (mean_ref - y) ** 2

    def gains(idx: np.ndarray) -> np.ndarray:
        d = np.sqrt(sq_ref[idx].mean())
        if d <= 0:
            raise ValueError("degenerate reference")
        return 1.0 - np.sqrt(sq[:, idx].mean(axis=1)) / d

    def summarise(idx: np.ndarray) -> dict:
        g = gains(idx)
        num = g[RUNG_CONTRAST_TREES] - g[RUNG_ZERO]           # vs naive reference
        den = g[RUNG_HISTORY_TREES] - g[RUNG_CONTRAST_TREES]  # vs matched control
        return {"gains": g, "numerator": num, "denominator": den,
                "ratio": num / den if den != 0 else np.nan}

    strata_idx = [np.flatnonzero(strata == s) for s in range(k)]
    point = [summarise(ix) for ix in strata_idx]
    point_all = summarise(np.arange(len(y)))

    uniq = np.unique(clusters)
    index_of = {c: np.flatnonzero(clusters == c) for c in uniq}
    rng = np.random.default_rng(seed)
    draws = {"numerator": [[] for _ in range(k)], "denominator": [[] for _ in range(k)],
             "ratio": [[] for _ in range(k)], "gains": [[] for _ in range(k)]}
    draws_all = {"numerator": [], "denominator": [], "ratio": []}
    contrasts = {"numerator": [], "denominator": [], "ratio": [],
                 "numerator_range": [], "denominator_range": [], "ratio_range": []}
    discarded = 0

    for _ in range(replicates):
        picked = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([index_of[c] for c in picked])
        strat_here = strata[idx]
        per = []
        ok = True
        for s in range(k):
            sub = idx[strat_here == s]
            if sub.size < MIN_ROWS_PER_STRATUM:
                ok = False
                break
            per.append(summarise(sub))
        if not ok:
            discarded += 1
            continue
        for s in range(k):
            for key in ("numerator", "denominator", "ratio"):
                draws[key][s].append(per[s][key])
            draws["gains"][s].append(per[s]["gains"])
        a = summarise(idx)
        for key in ("numerator", "denominator", "ratio"):
            draws_all[key].append(a[key])
            vals = np.array([per[s][key] for s in range(k)])
            contrasts[key].append(vals[-1] - vals[0])            # top stratum - bottom
            contrasts[f"{key}_range"].append(np.nanmax(vals) - np.nanmin(vals))

    def ci(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return [float("nan"), float("nan")]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    out_strata = []
    for s in range(k):
        gd = np.vstack(draws["gains"][s]) if draws["gains"][s] else np.zeros((1, len(rungs)))
        den_draws = np.asarray(draws["denominator"][s], float)
        out_strata.append({
            "stratum": s,
            "rows": int(strata_idx[s].size),
            "clusters": int(np.unique(clusters[strata_idx[s]]).size),
            "rungs": [{"rung": rungs[r][0], "n_features": rungs[r][1],
                       "rmse_gain": float(point[s]["gains"][r]),
                       "rmse_gain_ci95": ci(gd[:, r])} for r in range(len(rungs))],
            "numerator_pp": float(100 * point[s]["numerator"]),
            "numerator_pp_ci95": [100 * v for v in ci(draws["numerator"][s])],
            "denominator_pp": float(100 * point[s]["denominator"]),
            "denominator_pp_ci95": [100 * v for v in ci(draws["denominator"][s])],
            "denominator_sign_flip_rate": float(np.mean(den_draws <= 0)) if den_draws.size else float("nan"),
            "overstatement_factor": float(point[s]["ratio"]),
            "overstatement_factor_ci95": ci(draws["ratio"][s]),
            "overstatement_factor_fieller95": fieller_set(
                np.asarray(draws["numerator"][s], float), den_draws),
        })

    return {
        "k": k,
        "bootstrap_replicates_used": int(replicates - discarded),
        "bootstrap_replicates_discarded": int(discarded),
        "pooled": {
            "rows": int(len(y)), "clusters": int(np.unique(clusters).size),
            "numerator_pp": float(100 * point_all["numerator"]),
            "numerator_pp_ci95": [100 * v for v in ci(draws_all["numerator"])],
            "denominator_pp": float(100 * point_all["denominator"]),
            "denominator_pp_ci95": [100 * v for v in ci(draws_all["denominator"])],
            "overstatement_factor": float(point_all["ratio"]),
            "overstatement_factor_ci95": ci(draws_all["ratio"]),
            "overstatement_factor_fieller95": fieller_set(
                np.asarray(draws_all["numerator"], float),
                np.asarray(draws_all["denominator"], float)),
            "denominator_sign_flip_rate": float(np.mean(
                np.asarray(draws_all["denominator"], float) <= 0)),
        },
        "strata": out_strata,
        "contrasts_top_minus_bottom": {
            "numerator_pp": float(100 * (point[-1]["numerator"] - point[0]["numerator"])),
            "numerator_pp_ci95": [100 * v for v in ci(contrasts["numerator"])],
            "denominator_pp": float(100 * (point[-1]["denominator"] - point[0]["denominator"])),
            "denominator_pp_ci95": [100 * v for v in ci(contrasts["denominator"])],
            "overstatement_factor": float(point[-1]["ratio"] - point[0]["ratio"]),
            "overstatement_factor_ci95": ci(contrasts["ratio"]),
        },
        "range_across_strata": {
            "numerator_pp": float(100 * (max(p["numerator"] for p in point)
                                         - min(p["numerator"] for p in point))),
            "numerator_pp_ci95": [100 * v for v in ci(contrasts["numerator_range"])],
            "denominator_pp": float(100 * (max(p["denominator"] for p in point)
                                           - min(p["denominator"] for p in point))),
            "denominator_pp_ci95": [100 * v for v in ci(contrasts["denominator_range"])],
            "overstatement_factor": float(max(p["ratio"] for p in point)
                                          - min(p["ratio"] for p in point)),
            "overstatement_factor_ci95": ci(contrasts["ratio_range"]),
        },
    }


def describe_strata(frame: pd.DataFrame, strata: np.ndarray, k: int, column: str) -> list:
    rows = []
    for s in range(k):
        sub = frame[strata == s]
        rows.append({
            "stratum": s, "rows": int(len(sub)),
            "clusters": int(sub.cluster.nunique()),
            "networks": {n: int(c) for n, c in sub.network.value_counts().items()},
            f"{column}_median": float(sub[column].median()),
            f"{column}_range": [float(sub[column].min()), float(sub[column].max())],
            "incumbent_realised_cost_median": float(sub.baseline_cost_h4.median()),
            "gain_h4_mean": float(sub[TARGET].mean()),
            "gain_h4_sd": float(sub[TARGET].std()),
            "share_gain_positive": float((sub[TARGET] > 0).mean()),
        })
    return rows


def auroc_point(pred: np.ndarray, y: np.ndarray, idx: np.ndarray) -> float:
    labels = (y[idx] > 0).astype(int)
    if labels.min() == labels.max() or np.allclose(pred[idx], pred[idx][0]):
        return float("nan")
    return float(roc_auc_score(labels, pred[idx]))


def fmt_fieller(f: dict) -> str:
    if f["kind"] in ("insufficient_replicates", "empty"):
        return f["kind"]
    lo, hi = f["interval"]
    tag = {"bounded": "bounded", "unbounded": "UNBOUNDED (factor not identified)",
           "bounded_with_gap": "bounded, non-contiguous"}[f["kind"]]
    return f"[{lo:>8.1f}, {hi:>8.1f}]  {tag}"


def print_table(title: str, res: dict, desc: list, column: str) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")
    print(f"{'stratum':<9}{'rows':>6}{'clus':>6}{'med(meas)':>12}{'med cost':>10}"
          f"{'naive num':>22}{'matched den':>22}{'factor':>20}")
    p = res["pooled"]
    print(f"{'POOLED':<9}{p['rows']:>6}{p['clusters']:>6}{'-':>12}{'-':>10}"
          f"{p['numerator_pp']:>9.2f} [{p['numerator_pp_ci95'][0]:6.2f},{p['numerator_pp_ci95'][1]:6.2f}]"
          f"{p['denominator_pp']:>9.2f} [{p['denominator_pp_ci95'][0]:6.2f},{p['denominator_pp_ci95'][1]:6.2f}]"
          f"{p['overstatement_factor']:>8.1f} [{p['overstatement_factor_ci95'][0]:5.1f},{p['overstatement_factor_ci95'][1]:6.1f}]")
    for s, d in zip(res["strata"], desc):
        print(f"{'T' + str(s['stratum'] + 1):<9}{s['rows']:>6}{s['clusters']:>6}"
              f"{d[column + '_median']:>12.1f}{d['incumbent_realised_cost_median']:>10.0f}"
              f"{s['numerator_pp']:>9.2f} [{s['numerator_pp_ci95'][0]:6.2f},{s['numerator_pp_ci95'][1]:6.2f}]"
              f"{s['denominator_pp']:>9.2f} [{s['denominator_pp_ci95'][0]:6.2f},{s['denominator_pp_ci95'][1]:6.2f}]"
              f"{s['overstatement_factor']:>8.1f} [{s['overstatement_factor_ci95'][0]:5.1f},{s['overstatement_factor_ci95'][1]:6.1f}]")
    print("\n  95% confidence SET for the factor (inversion, not a percentile summary):")
    print(f"    {'POOLED':<7} {fmt_fieller(p['overstatement_factor_fieller95'])}")
    for s in res["strata"]:
        print(f"    {'T' + str(s['stratum'] + 1):<7} {fmt_fieller(s['overstatement_factor_fieller95'])}")
    c = res["contrasts_top_minus_bottom"]
    print(f"\n  top-minus-bottom  numerator {c['numerator_pp']:+7.2f} pp "
          f"CI[{c['numerator_pp_ci95'][0]:+.2f},{c['numerator_pp_ci95'][1]:+.2f}]"
          f"   {'HETEROGENEOUS' if c['numerator_pp_ci95'][0] * c['numerator_pp_ci95'][1] > 0 else 'no evidence of heterogeneity'}")
    print(f"  top-minus-bottom  denominator {c['denominator_pp']:+7.2f} pp "
          f"CI[{c['denominator_pp_ci95'][0]:+.2f},{c['denominator_pp_ci95'][1]:+.2f}]"
          f"   {'HETEROGENEOUS' if c['denominator_pp_ci95'][0] * c['denominator_pp_ci95'][1] > 0 else 'no evidence of heterogeneity'}")
    print(f"  top-minus-bottom  factor {c['overstatement_factor']:+8.1f} "
          f"CI[{c['overstatement_factor_ci95'][0]:+.1f},{c['overstatement_factor_ci95'][1]:+.1f}]"
          f"   {'HETEROGENEOUS' if c['overstatement_factor_ci95'][0] * c['overstatement_factor_ci95'][1] > 0 else 'no evidence of heterogeneity'}")
    flips = [s["denominator_sign_flip_rate"] for s in res["strata"]]
    print(f"  denominator sign-flip rate per stratum: "
          + ", ".join(f"T{i+1}={100*f:.1f}%" for i, f in enumerate(flips)))
    if max(flips) > 0.025:
        print("  WARNING: the matched-control increment is not significantly positive in at "
              "least one stratum; the ratio there is a ratio with a near-zero denominator "
              "and its interval is not a bounded quantity.")
    if res["bootstrap_replicates_discarded"]:
        print(f"  {res['bootstrap_replicates_discarded']} of {REPLICATES} replicates discarded "
              f"(a stratum fell below {MIN_ROWS_PER_STRATUM} rows).")


def load_sealed() -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(str(SEALED_RAW / "*.json"))):
        r = json.loads(Path(path).read_text(encoding="utf-8"))
        if r.get("status") != "complete" or r.get(TARGET) is None:
            continue
        rows.append(r)
    df = pd.DataFrame(rows)
    for c in ("context_observation", "feature_vector"):
        df[c] = df[c].apply(lambda v: json.dumps(v) if not isinstance(v, str) else v)
    return df


def main() -> None:
    actual = sha256_file(PAIRED)
    if actual != EXPECTED_SHA:
        raise SystemExit(f"input hash changed: {actual}")
    OUT.mkdir(parents=True, exist_ok=True)

    df = add_derived(pd.read_csv(PAIRED))
    train = df[df.split.isin(("representation_train", "development"))].reset_index(drop=True)
    evalu = df[df.split == "validation"].reset_index(drop=True)
    print(f"input sha256 verified: {actual[:16]}...")
    print(f"train {len(train)} rows / {train.cluster.nunique()} clusters  ->  "
          f"eval {len(evalu)} rows / {evalu.cluster.nunique()} clusters")

    rungs, mean_ref = fit_ladder(train, evalu)
    y = evalu[TARGET].to_numpy(float)
    clusters = evalu.cluster.to_numpy()
    gp = [1.0 - np.sqrt(((p - y) ** 2).mean()) / np.sqrt(((mean_ref - y) ** 2).mean())
          for _, _, p in rungs]
    print("\nreplication of phase0/baseline_ladder.py on the pooled validation split:")
    for (name, w, _), g in zip(rungs, gp):
        print(f"  {name:<40} {str(w):>5}  gain {100*g:6.2f}%")
    print(f"  naive-reference numerator (rung5 - rung0) : {100*(gp[5]-gp[0]):+.2f} pp   "
          f"[manuscript: +16.38]")
    print(f"  matched-control denominator (rung6 - rung5): {100*(gp[6]-gp[5]):+.2f} pp   "
          f"[manuscript: +0.82]")
    print(f"  overstatement factor                       : {(gp[5]-gp[0])/(gp[6]-gp[5]):.2f}    "
          f"[manuscript: 19.9]")

    report = {
        "question": "Does the baseline-ladder overstatement factor depend on how well the "
                    "incumbent PPO ensemble is coping in the state where the decision is taken?",
        "evidence_class": "retrospective_diagnostic_not_confirmatory",
        "protocol_inherited_from": "v10_proposal/phase0/baseline_ladder.py",
        "input": "v6/results/cfra/paired_branches.csv",
        "input_sha256": actual,
        "target": TARGET,
        "numerator_definition": "rung 5 (trees, action contrast) minus rung 0 (zero effect)",
        "denominator_definition": "rung 6 (trees, contrast + 8-step history) minus rung 5",
        "models_refitted_per_stratum": False,
        "bootstrap": {"kind": "cluster bootstrap over (network, seed)",
                      "replicates": REPLICATES, "seed": SEED,
                      "note": "clusters are resampled once per replicate and every stratum is "
                              "recomputed inside that same resample, so between-stratum "
                              "contrasts are valid"},
        "stratifier_catalogue": STRATIFIERS,
        "pooled_replication": {
            "ladder_gains": {rungs[i][0]: float(gp[i]) for i in range(len(rungs))},
            "numerator_pp": float(100 * (gp[5] - gp[0])),
            "denominator_pp": float(100 * (gp[6] - gp[5])),
            "overstatement_factor": float((gp[5] - gp[0]) / (gp[6] - gp[5])),
        },
        "validation": {},
        "sealed": {},
    }

    table_rows = []
    schemes = [("within_network_tercile", True, 3), ("global_tercile", False, 3)]
    for column in STRATIFIERS:
        for scheme_name, within, k in schemes:
            strata, cuts = assign_strata(evalu, column, k, within)
            res = stratified_ladder(rungs, mean_ref, y, clusters, strata, k)
            desc = describe_strata(evalu, strata, k, column)
            for s in range(k):
                res["strata"][s]["auroc"] = auroc_point(
                    rungs[RUNG_HISTORY_TREES][2], y, np.flatnonzero(strata == s))
            res["cutpoints"] = cuts
            res["stratum_description"] = desc
            report["validation"].setdefault(column, {})[scheme_name] = res
            title = (f"VALIDATION SPLIT | stratifier = {column} | {scheme_name} "
                     f"| naive num = rung5-rung0, matched den = rung6-rung5")
            print_table(title, res, desc, column)
            for s in range(k):
                table_rows.append({
                    "block": "validation", "stratifier": column, "scheme": scheme_name,
                    "stratum": f"T{s+1}", "rows": res["strata"][s]["rows"],
                    "clusters": res["strata"][s]["clusters"],
                    "measure_median": desc[s][f"{column}_median"],
                    "incumbent_cost_median": desc[s]["incumbent_realised_cost_median"],
                    "gain_h4_mean": desc[s]["gain_h4_mean"],
                    "numerator_pp": res["strata"][s]["numerator_pp"],
                    "numerator_lo": res["strata"][s]["numerator_pp_ci95"][0],
                    "numerator_hi": res["strata"][s]["numerator_pp_ci95"][1],
                    "denominator_pp": res["strata"][s]["denominator_pp"],
                    "denominator_lo": res["strata"][s]["denominator_pp_ci95"][0],
                    "denominator_hi": res["strata"][s]["denominator_pp_ci95"][1],
                    "factor": res["strata"][s]["overstatement_factor"],
                    "factor_lo": res["strata"][s]["overstatement_factor_ci95"][0],
                    "factor_hi": res["strata"][s]["overstatement_factor_ci95"][1],
                    "den_sign_flip_rate": res["strata"][s]["denominator_sign_flip_rate"],
                    "factor_fieller_kind": res["strata"][s]["overstatement_factor_fieller95"]["kind"],
                    "factor_fieller_lo": res["strata"][s]["overstatement_factor_fieller95"]["interval"][0],
                    "factor_fieller_hi": res["strata"][s]["overstatement_factor_fieller95"]["interval"][1],
                })

    # ---------------------------------------------------------------- sealed
    print(f"\n\n{'#' * 100}\nSEALED BLOCK (seeds 9801-9900, never opened when the ladder was "
          f"written)\n{'#' * 100}")
    sealed = add_derived(load_sealed())
    print(f"sealed rows {len(sealed)} / clusters {sealed.cluster.nunique()}  "
          f"networks {sorted(sealed.network.unique())}")
    rungs_s, mean_ref_s = fit_ladder(train, sealed)
    y_s = sealed[TARGET].to_numpy(float)
    clusters_s = sealed.cluster.to_numpy()
    gs = [1.0 - np.sqrt(((p - y_s) ** 2).mean()) / np.sqrt(((mean_ref_s - y_s) ** 2).mean())
          for _, _, p in rungs_s]
    print("\npooled ladder on the sealed block (same training split, fresh evaluation data):")
    for (name, w, _), g in zip(rungs_s, gs):
        print(f"  {name:<40} {str(w):>5}  gain {100*g:6.2f}%")
    print(f"  numerator {100*(gs[5]-gs[0]):+.2f} pp   denominator {100*(gs[6]-gs[5]):+.2f} pp   "
          f"factor {(gs[5]-gs[0])/(gs[6]-gs[5]):.2f}")
    report["sealed_pooled_replication"] = {
        "rows": int(len(sealed)), "clusters": int(sealed.cluster.nunique()),
        "ladder_gains": {rungs_s[i][0]: float(gs[i]) for i in range(len(rungs_s))},
        "numerator_pp": float(100 * (gs[5] - gs[0])),
        "denominator_pp": float(100 * (gs[6] - gs[5])),
        "overstatement_factor": float((gs[5] - gs[0]) / (gs[6] - gs[5])),
        "note": "The ladder is refitted on the SAME training split and applied to fresh "
                "evaluation seeds. This is an independent evaluation sample, not an "
                "independent training sample, and it is not the preregistered "
                "confirmatory test in phase0/evaluate_sealed.py.",
    }

    for column in STRATIFIERS:
        for scheme_name, within, k in schemes:
            strata, cuts = assign_strata(sealed, column, k, within)
            res = stratified_ladder(rungs_s, mean_ref_s, y_s, clusters_s, strata, k)
            desc = describe_strata(sealed, strata, k, column)
            res["cutpoints"] = cuts
            res["stratum_description"] = desc
            report["sealed"].setdefault(column, {})[scheme_name] = res
            print_table(f"SEALED BLOCK | stratifier = {column} | {scheme_name}",
                        res, desc, column)
            for s in range(k):
                table_rows.append({
                    "block": "sealed", "stratifier": column, "scheme": scheme_name,
                    "stratum": f"T{s+1}", "rows": res["strata"][s]["rows"],
                    "clusters": res["strata"][s]["clusters"],
                    "measure_median": desc[s][f"{column}_median"],
                    "incumbent_cost_median": desc[s]["incumbent_realised_cost_median"],
                    "gain_h4_mean": desc[s]["gain_h4_mean"],
                    "numerator_pp": res["strata"][s]["numerator_pp"],
                    "numerator_lo": res["strata"][s]["numerator_pp_ci95"][0],
                    "numerator_hi": res["strata"][s]["numerator_pp_ci95"][1],
                    "denominator_pp": res["strata"][s]["denominator_pp"],
                    "denominator_lo": res["strata"][s]["denominator_pp_ci95"][0],
                    "denominator_hi": res["strata"][s]["denominator_pp_ci95"][1],
                    "factor": res["strata"][s]["overstatement_factor"],
                    "factor_lo": res["strata"][s]["overstatement_factor_ci95"][0],
                    "factor_hi": res["strata"][s]["overstatement_factor_ci95"][1],
                    "den_sign_flip_rate": res["strata"][s]["denominator_sign_flip_rate"],
                    "factor_fieller_kind": res["strata"][s]["overstatement_factor_fieller95"]["kind"],
                    "factor_fieller_lo": res["strata"][s]["overstatement_factor_fieller95"]["interval"][0],
                    "factor_fieller_hi": res["strata"][s]["overstatement_factor_fieller95"]["interval"][1],
                })

    # ------------------------------------------------- identified verdict
    print(f"\n\n{'#' * 100}\nVERDICT: what IS and what IS NOT identified\n{'#' * 100}")
    verdicts = {}
    for blockname, blockres in (("validation", report["validation"]),
                                ("sealed", report["sealed"])):
        for column in STRATIFIERS:
            for scheme_name, _, _ in schemes:
                res = blockres[column][scheme_name]
                st = res["strata"]
                v = {
                    "numerator_pp_min": min(s["numerator_pp"] for s in st),
                    "numerator_pp_max": max(s["numerator_pp"] for s in st),
                    "every_numerator_ci_excludes_zero":
                        all(s["numerator_pp_ci95"][0] > 0 for s in st),
                    "denominator_pp_min": min(s["denominator_pp"] for s in st),
                    "denominator_pp_max": max(s["denominator_pp"] for s in st),
                    "every_denominator_ci_covers_zero":
                        all(s["denominator_pp_ci95"][0] <= 0 <= s["denominator_pp_ci95"][1]
                            for s in st),
                    "any_denominator_ci_excludes_zero":
                        any(s["denominator_pp_ci95"][0] > 0 or s["denominator_pp_ci95"][1] < 0
                            for s in st),
                    "factor_identified_in_any_stratum":
                        any(s["overstatement_factor_fieller95"]["kind"] == "bounded" for s in st),
                    "numerator_heterogeneous_top_vs_bottom": bool(
                        res["contrasts_top_minus_bottom"]["numerator_pp_ci95"][0]
                        * res["contrasts_top_minus_bottom"]["numerator_pp_ci95"][1] > 0),
                    "denominator_heterogeneous_top_vs_bottom": bool(
                        res["contrasts_top_minus_bottom"]["denominator_pp_ci95"][0]
                        * res["contrasts_top_minus_bottom"]["denominator_pp_ci95"][1] > 0),
                    "min_clusters_in_a_stratum": min(s["clusters"] for s in st),
                }
                verdicts[f"{blockname}|{column}|{scheme_name}"] = v
                print(f"\n{blockname:<11}{column:<22}{scheme_name}")
                print(f"   numerator (vs naive)   range over strata "
                      f"[{v['numerator_pp_min']:6.2f}, {v['numerator_pp_max']:6.2f}] pp   "
                      f"all CIs exclude 0: {v['every_numerator_ci_excludes_zero']}   "
                      f"heterogeneous: {v['numerator_heterogeneous_top_vs_bottom']}")
                print(f"   denominator (vs matched) range over strata "
                      f"[{v['denominator_pp_min']:6.2f}, {v['denominator_pp_max']:6.2f}] pp   "
                      f"all CIs cover 0: {v['every_denominator_ci_covers_zero']}   "
                      f"heterogeneous: {v['denominator_heterogeneous_top_vs_bottom']}")
                print(f"   factor identified in any stratum: "
                      f"{v['factor_identified_in_any_stratum']}   "
                      f"min clusters in a stratum: {v['min_clusters_in_a_stratum']}")
    report["verdicts"] = verdicts

    pd.DataFrame(table_rows).to_csv(OUT / "strata_ladder.csv", index=False)
    (OUT / "incumbent_invariance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'incumbent_invariance.json'}")
    print(f"wrote {OUT / 'strata_ladder.csv'}")


if __name__ == "__main__":
    main()
