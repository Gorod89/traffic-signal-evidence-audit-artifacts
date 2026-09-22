"""Does the baseline-artifact result survive a different cost function?

Every gain in this work is measured against one utility,
    J = stopped_vehicle_seconds + 2.0*terminal_queue + 0.5*hard_braking + 30.0*teleports,
inherited from V6 and never justified in the papers that used it. If the two
increments that carry the argument -- large against a naive reference, null
against a capacity-matched control -- were an artifact of those four weights,
the finding would be about the utility rather than about the estimator.

The paired records store the four component deltas separately, so the target can
be rebuilt under any weighting and the ladder refitted. That is what this does,
over the registered weights, three single-component targets, an equal-weight
variant, and two perturbations of each weight.
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
OUT = REPO / "artifacts" / "derived" / "phase0" / "cost_weight_sensitivity.json"
SEED = 20260813
REPLICATES = 2000

COMPONENTS = ("stopped", "queue", "braking", "teleports")
REGISTERED = {"stopped": 1.0, "queue": 2.0, "braking": 0.5, "teleports": 30.0}

SCHEMES = {
    "registered (1, 2, 0.5, 30)": REGISTERED,
    "stopped only": {"stopped": 1.0, "queue": 0.0, "braking": 0.0, "teleports": 0.0},
    "queue only": {"stopped": 0.0, "queue": 1.0, "braking": 0.0, "teleports": 0.0},
    "teleports only": {"stopped": 0.0, "queue": 0.0, "braking": 0.0, "teleports": 1.0},
    "equal weights": {c: 1.0 for c in COMPONENTS},
    "queue x4 (1, 8, 0.5, 30)": {**REGISTERED, "queue": 8.0},
    "queue /4 (1, 0.5, 0.5, 30)": {**REGISTERED, "queue": 0.5},
    "teleports x4 (1, 2, 0.5, 120)": {**REGISTERED, "teleports": 120.0},
    "teleports /30 (1, 2, 0.5, 1)": {**REGISTERED, "teleports": 1.0},
    "braking x10 (1, 2, 5, 30)": {**REGISTERED, "braking": 5.0},
}

DELTA_COLS = {
    "stopped": "delta_stopped_vehicle_seconds_h4",
    "queue": "delta_terminal_queue_h4",
    "braking": "delta_hard_braking_h4",
    "teleports": "delta_teleports_h4",
}


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

    con_tr, con_ev = block(train, ("feature_vector",)), block(evalu, ("feature_vector",))
    hist_cols = ("feature_vector", "context_history", "context_action_history")
    hist_tr, hist_ev = block(train, hist_cols), block(evalu, hist_cols)
    clusters = evalu.cluster.to_numpy()
    unique = np.unique(clusters)
    index_of = {c: np.flatnonzero(clusters == c) for c in unique}

    # The registered gain is a signed cost difference, so a target under new
    # weights is the same linear combination of the recorded component deltas.
    # Verify that identity on the registered weights before trusting the rest.
    def target(frame: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
        return sum(
            weights[c] * frame[DELTA_COLS[c]].to_numpy(dtype=np.float64) for c in COMPONENTS
        )

    rebuilt = target(evalu, REGISTERED)
    recorded = evalu.gain_h4.to_numpy(dtype=np.float64)
    residual = float(np.max(np.abs(rebuilt - recorded)))
    print(f"identity check on registered weights: max |rebuilt - recorded| = {residual:.3e}")
    if residual > 1e-6:
        raise SystemExit("component deltas do not reconstruct the recorded gain; aborting")

    rows = []
    for name, weights in SCHEMES.items():
        y_tr, y_ev = target(train, weights), target(evalu, weights)
        if np.allclose(y_ev, 0.0):
            continue
        reference = float(np.mean(y_tr))

        preds = {}
        for key, xt, xe in (("con", con_tr, con_ev), ("hist", hist_tr, hist_ev)):
            model = ExtraTreesRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
            model.fit(xt, y_tr)
            preds[key] = model.predict(xe)

        def gains(idx):
            d = float(np.sqrt(np.mean((reference - y_ev[idx]) ** 2)))
            g = {k: 1.0 - float(np.sqrt(np.mean((p[idx] - y_ev[idx]) ** 2))) / d
                 for k, p in preds.items()}
            g["zero"] = 1.0 - float(np.sqrt(np.mean((0.0 - y_ev[idx]) ** 2))) / d
            return g

        all_idx = np.arange(len(y_ev))
        g = gains(all_idx)
        increment, naive = g["hist"] - g["con"], g["con"] - g["zero"]

        rng = np.random.default_rng(SEED)
        inc_draws, nai_draws = [], []
        for _ in range(REPLICATES):
            picked = rng.choice(unique, size=unique.size, replace=True)
            idx = np.concatenate([index_of[c] for c in picked])
            gb = gains(idx)
            inc_draws.append(gb["hist"] - gb["con"])
            nai_draws.append(gb["con"] - gb["zero"])
        inc, nai = np.asarray(inc_draws), np.asarray(nai_draws)

        row = {
            "scheme": name,
            "weights": weights,
            "gain_no_history": g["con"],
            "history_increment_pp": 100 * increment,
            "history_increment_ci95_pp": [100 * np.percentile(inc, 2.5),
                                          100 * np.percentile(inc, 97.5)],
            "history_increment_covers_zero": bool(np.percentile(inc, 2.5) <= 0 <= np.percentile(inc, 97.5)),
            "naive_increment_pp": 100 * naive,
            "naive_increment_ci95_pp": [100 * np.percentile(nai, 2.5),
                                        100 * np.percentile(nai, 97.5)],
            "naive_increment_positive": bool(np.percentile(nai, 2.5) > 0),
        }
        rows.append(row)
        print(
            "%-30s naive %+6.2f [%+6.2f,%+6.2f]   history %+5.2f [%+5.2f,%+5.2f]  covers0=%s"
            % (name, row["naive_increment_pp"], *row["naive_increment_ci95_pp"],
               row["history_increment_pp"], *row["history_increment_ci95_pp"],
               row["history_increment_covers_zero"])
        )

    n = len(rows)
    report = {
        "input_sha256": actual,
        "identity_residual": residual,
        "replicates": REPLICATES,
        "schemes": rows,
        "summary": {
            "schemes_tested": n,
            "history_increment_covers_zero_in": sum(r["history_increment_covers_zero"] for r in rows),
            "naive_increment_positive_in": sum(r["naive_increment_positive"] for r in rows),
        },
    }
    write_json_atomic(OUT, report)
    s = report["summary"]
    print(f"\nhistory increment covers zero in {s['history_increment_covers_zero_in']}/{n} schemes")
    print(f"naive increment strictly positive in {s['naive_increment_positive_in']}/{n} schemes")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
