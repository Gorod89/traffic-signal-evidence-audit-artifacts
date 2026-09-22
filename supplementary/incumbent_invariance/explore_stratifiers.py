"""Exploration: which observable measures 'how well the incumbent copes'?

Read-only. Prints diagnostics used to justify the stratifier choice in
`incumbent_invariance.py`. Nothing here is a result; it is the selection audit.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PAIRED = Path(
    os.environ.get(
        "V6_PAIRED_PATH",
        REPO / "artifacts" / "derived" / "v6" / "paired_branches.csv",
    )
).resolve()
EXPECTED_SHA = "24231bba4eadff8051da0af7705ff429d86c93ec10c554ede9247836aea55d8e"

OBS_NAMES = [
    "green_queue", "red_queue", "total_queue", "max_lane_queue", "mean_speed",
    "downstream_queue", "green_age", "phase_progress", "queue_delta", "mean_occupancy",
]
OBS_SCALES = np.array([50.0, 50.0, 100.0, 30.0, 15.0, 50.0, 90.0, 1.0, 60.0, 1.0])


def main() -> None:
    actual = hashlib.sha256(PAIRED.read_bytes()).hexdigest()
    print(f"input sha256 {actual}  match={actual == EXPECTED_SHA}")

    df = pd.read_csv(PAIRED)
    df["cluster"] = df.network + "|" + df.seed.astype(str)

    obs = np.vstack([np.asarray(json.loads(v), dtype=np.float64).ravel()
                     for v in df.context_observation])
    for i, name in enumerate(OBS_NAMES):
        df[f"obs_{name}"] = obs[:, i]
    # de-normalised, matching features.SignalObservation.as_array()
    df["local_total_queue"] = obs[:, 2] * 100.0
    df["local_downstream_queue"] = obs[:, 5] * 50.0
    df["local_queue_delta"] = obs[:, 8] * 60.0 - 30.0
    # the repository's own single-observation cost functional (features.py:55)
    df["observed_cost"] = (df.local_total_queue
                           + 0.35 * df.local_downstream_queue
                           + 0.10 * df.local_queue_delta.clip(lower=0.0))

    ev = df[df.split == "validation"].reset_index(drop=True)
    print(f"\neval rows {len(ev)}  clusters {ev.cluster.nunique()}  "
          f"networks {sorted(ev.network.unique())}")
    print(f"gain_h4 == baseline_cost_h4 - candidate_cost_h4 : "
          f"{np.allclose(df.gain_h4, df.baseline_cost_h4 - df.candidate_cost_h4)}")

    cands = ["observed_cost", "local_total_queue", "obs_max_lane_queue",
             "local_downstream_queue", "obs_mean_occupancy", "obs_mean_speed",
             "severity", "policy_entropy", "policy_disagreement",
             "sensor_reliability"]

    print("\n--- Spearman rho with the incumbent's REALISED window cost "
          "(baseline_cost_h4), eval split ---")
    print(f"{'measure':<26} {'rho(all)':>9} {'|rho| within-network (mean)':>29}")
    for c in cands:
        rho = stats.spearmanr(ev[c], ev.baseline_cost_h4).statistic
        wn = [abs(stats.spearmanr(g[c], g.baseline_cost_h4).statistic)
              for _, g in ev.groupby("network") if g[c].nunique() > 1]
        print(f"{c:<26} {rho:>9.3f} {np.mean(wn):>29.3f}")

    print("\n--- does the measure leak the TARGET (gain_h4)? "
          "|Spearman rho| should be small ---")
    for c in cands + ["baseline_cost_h4"]:
        rho = stats.spearmanr(ev[c], ev.gain_h4).statistic
        print(f"{c:<26} rho(gain_h4) = {rho:>7.3f}")

    print("\n--- network confounding of a global tercile split ---")
    for c in ["observed_cost", "local_total_queue", "policy_entropy",
              "policy_disagreement", "severity", "baseline_cost_h4"]:
        q = pd.qcut(ev[c].rank(method="first"), 3, labels=["T1", "T2", "T3"])
        tab = pd.crosstab(q, ev.network)
        ncl = ev.groupby(q, observed=True).cluster.nunique()
        print(f"\n{c}:")
        print(tab.to_string())
        print("  clusters per stratum: " + ", ".join(f"{k}={v}" for k, v in ncl.items()))

    print("\n--- within-network tercile: clusters per stratum ---")
    for c in ["observed_cost", "local_total_queue", "policy_entropy",
              "policy_disagreement"]:
        s = ev.groupby("network")[c].transform(lambda x: pd.qcut(x.rank(method="first"), 3,
                                                                labels=[0, 1, 2]).astype(int))
        ncl = ev.groupby(s).cluster.nunique()
        tab = pd.crosstab(s, ev.network)
        print(f"\n{c}: clusters per stratum " + ", ".join(f"T{k+1}={v}" for k, v in ncl.items()))
        print(tab.to_string())

    print("\n--- spread of the incumbent's realised cost across strata "
          "(is the stratifier separating anything?) ---")
    for c in ["observed_cost", "local_total_queue", "policy_entropy",
              "policy_disagreement", "severity"]:
        s = ev.groupby("network")[c].transform(lambda x: pd.qcut(x.rank(method="first"), 3,
                                                                labels=[0, 1, 2]).astype(int))
        med = ev.groupby(s).baseline_cost_h4.median()
        print(f"{c:<24} median baseline_cost_h4 by within-network tercile: "
              + "  ".join(f"T{k+1}={v:,.0f}" for k, v in med.items()))

    print("\n--- per-network scale of baseline_cost_h4 ---")
    print(ev.groupby("network").baseline_cost_h4.describe()[
        ["count", "25%", "50%", "75%", "max"]].round(0).to_string())


if __name__ == "__main__":
    main()
