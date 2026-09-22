"""Put a paired interval on the history increment -- the denominator of the 19.9 factor.

The paper reports two RMSE gains, 14.32% without history and 15.15% with it, and
divides the distance from the naive reference by the distance between them. The
two gains carry intervals; their *difference* never did. If that difference is
not distinguishable from zero the ratio is not identified, and reporting it as a
point estimate overstates what the corpus can say.

This resamples clusters once per replicate and recomputes both models' errors on
the same resample, so the interval is paired rather than a comparison of two
marginal intervals.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from integrity import write_json_atomic
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

REPO = Path(__file__).resolve().parents[2]
PAIRED = REPO / "artifacts" / "derived" / "v6" / "paired_branches.csv"
EXPECTED_SHA = "24231bba4eadff8051da0af7705ff429d86c93ec10c554ede9247836aea55d8e"
OUT = REPO / "artifacts" / "derived" / "phase0" / "history_increment_ci.json"
SEED = 20260805
REPLICATES = 4000


def parse(series: pd.Series) -> np.ndarray:
    return np.vstack([np.asarray(json.loads(v), dtype=np.float64).ravel() for v in series])


def main() -> None:
    actual = hashlib.sha256(PAIRED.read_bytes()).hexdigest()
    if actual != EXPECTED_SHA:
        raise SystemExit(f"input hash changed: {actual}")

    df = pd.read_csv(PAIRED)
    df["cluster"] = df.network + "|" + df.seed.astype(str)
    train = df[df.split.isin(("representation_train", "development"))].reset_index(drop=True)
    evalu = df[df.split == "validation"].reset_index(drop=True)

    def block(frame, cols):
        return np.hstack([parse(frame[c]) for c in cols])

    con_tr = block(train, ("feature_vector",))
    con_ev = block(evalu, ("feature_vector",))
    hist_tr = block(train, ("feature_vector", "context_history", "context_action_history"))
    hist_ev = block(evalu, ("feature_vector", "context_history", "context_action_history"))
    y_tr = train.gain_h4.to_numpy(dtype=np.float64)
    y_ev = evalu.gain_h4.to_numpy(dtype=np.float64)
    clusters = evalu.cluster.to_numpy()

    m_con = ExtraTreesRegressor(n_estimators=300, random_state=SEED, n_jobs=-1).fit(con_tr, y_tr)
    m_hist = ExtraTreesRegressor(n_estimators=300, random_state=SEED, n_jobs=-1).fit(hist_tr, y_tr)
    p_con, p_hist = m_con.predict(con_ev), m_hist.predict(hist_ev)
    reference = float(np.mean(y_tr))

    def gains(idx):
        d = float(np.sqrt(np.mean((reference - y_ev[idx]) ** 2)))
        g_con = 1.0 - float(np.sqrt(np.mean((p_con[idx] - y_ev[idx]) ** 2))) / d
        g_hist = 1.0 - float(np.sqrt(np.mean((p_hist[idx] - y_ev[idx]) ** 2))) / d
        g_zero = 1.0 - float(np.sqrt(np.mean((0.0 - y_ev[idx]) ** 2))) / d
        return g_con, g_hist, g_zero

    all_idx = np.arange(len(y_ev))
    g_con, g_hist, g_zero = gains(all_idx)
    increment = g_hist - g_con
    naive = g_con - g_zero

    rng = np.random.default_rng(SEED)
    unique = np.unique(clusters)
    index_of = {c: np.flatnonzero(clusters == c) for c in unique}
    inc_draws, naive_draws, ratio_draws = [], [], []
    for _ in range(REPLICATES):
        picked = rng.choice(unique, size=unique.size, replace=True)
        idx = np.concatenate([index_of[c] for c in picked])
        a, b, z = gains(idx)
        inc_draws.append(b - a)
        naive_draws.append(a - z)
        if abs(b - a) > 1e-12:
            ratio_draws.append((a - z) / (b - a))

    inc = np.asarray(inc_draws)
    nai = np.asarray(naive_draws)
    rat = np.asarray(ratio_draws)

    report = {
        "input_sha256": actual,
        "replicates": REPLICATES,
        "cluster_unit": "(network, seed)",
        "gain_no_history": g_con,
        "gain_with_history": g_hist,
        "gain_zero_effect": g_zero,
        "history_increment_pp": 100 * increment,
        "history_increment_ci95_pp": [100 * np.percentile(inc, 2.5), 100 * np.percentile(inc, 97.5)],
        "history_increment_sign_flip_fraction": float(np.mean(inc <= 0)),
        "naive_increment_pp": 100 * naive,
        "naive_increment_ci95_pp": [100 * np.percentile(nai, 2.5), 100 * np.percentile(nai, 97.5)],
        "ratio_point": naive / increment if increment else None,
        "ratio_ci95_percentile": [float(np.percentile(rat, 2.5)), float(np.percentile(rat, 97.5))],
        "ratio_identified": bool(np.percentile(inc, 2.5) > 0),
        "verdict": (
            "The denominator's interval covers zero, so the ratio is not identified "
            "and must not be reported as a point estimate."
            if np.percentile(inc, 2.5) <= 0 else
            "The denominator is bounded away from zero; the ratio is identified."
        ),
    }
    write_json_atomic(OUT, report)

    print(f"gain, no history      {100*g_con:6.2f}%")
    print(f"gain, with history    {100*g_hist:6.2f}%")
    print(f"history increment     {100*increment:+6.2f} pp   "
          f"95% CI [{100*np.percentile(inc,2.5):+.2f}, {100*np.percentile(inc,97.5):+.2f}]")
    print(f"  sign flips in       {100*np.mean(inc<=0):.1f}% of replicates")
    print(f"naive increment       {100*naive:+6.2f} pp   "
          f"95% CI [{100*np.percentile(nai,2.5):+.2f}, {100*np.percentile(nai,97.5):+.2f}]")
    print(f"ratio point           {naive/increment:.1f}")
    print(f"  percentile CI       [{np.percentile(rat,2.5):.1f}, {np.percentile(rat,97.5):.1f}]")
    print(f"\n{report['verdict']}")


if __name__ == "__main__":
    main()
