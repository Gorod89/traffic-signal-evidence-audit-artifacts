"""How much of the overstatement factor is the tree seed?

`v10_proposal/phase0/baseline_ladder.py` reports 19.9 with
ExtraTreesRegressor(random_state=20260805). A legacy run under Python 3.13 /
scikit-learn 1.7.2 is retained separately because it does not reproduce the
tree rungs under the canonical environment.

That is not a bug in either script. The factor is a ratio whose denominator is
the difference of two RMSEs that agree to within 1%, so anything that perturbs
a tree ensemble at the third significant figure moves the factor a long way.
This script measures that directly by refitting the ladder under many tree
seeds and leaving everything else fixed. It is a precision statement about the
estimator, and it is a precondition for asking whether the factor varies with
incumbent strength: a quantity that is not stable to its own RNG cannot be
shown stable or unstable across strata.

Generated output goes to ``build/``. Replacing the canonical reference result
requires the explicit ``--update-reference`` switch.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_info, threadpool_limits

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PAIRED = Path(
    os.environ.get(
        "V6_PAIRED_PATH",
        REPO / "artifacts" / "derived" / "v6" / "paired_branches.csv",
    )
).resolve()
EXPECTED_SHA = "24231bba4eadff8051da0af7705ff429d86c93ec10c554ede9247836aea55d8e"
REFERENCE = HERE / "results" / "seed_sensitivity.json"
TARGET = "gain_h4"
PHASE0_SEED = 20260805
N_SEEDS = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paired-path",
        type=Path,
        default=Path(os.environ.get("V6_PAIRED_PATH", PAIRED)),
        help="V6 paired derivative to analyse",
    )
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "--output",
        type=Path,
        help="generated result path (default: build/seed_sensitivity.json)",
    )
    destination.add_argument(
        "--update-reference",
        action="store_true",
        help="explicitly replace results/seed_sensitivity.json with this canonical run",
    )
    args = parser.parse_args()
    args.paired_path = args.paired_path.resolve()
    args.output = (
        REFERENCE if args.update_reference else args.output or HERE / "build" / "seed_sensitivity.json"
    ).resolve()
    if not args.update_reference and args.output == REFERENCE.resolve():
        parser.error("writing the canonical reference requires --update-reference")
    return args


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def sanitize_threadpools(entries: list[dict]) -> list[dict]:
    """Retain reproducibility fields without leaking host-specific paths."""
    fields = (
        "user_api",
        "internal_api",
        "prefix",
        "version",
        "threading_layer",
        "architecture",
        "num_threads",
    )
    return [
        {field: entry.get(field) for field in fields if field in entry}
        for entry in entries
    ]


def runtime_metadata(threadpools_before: list[dict], threadpools_during: list[dict]) -> dict:
    return {
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            name: package_version(name)
            for name in (
                "numpy",
                "pandas",
                "scikit-learn",
                "scipy",
                "matplotlib",
                "joblib",
                "threadpoolctl",
                "PyYAML",
                "pytest",
            )
        },
        "threadpools_before_limit": sanitize_threadpools(threadpools_before),
        "threadpools_during_limit": sanitize_threadpools(threadpools_during),
        "thread_limit": 1,
    }


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json.tmp", prefix=f".{path.name}.",
        dir=path.parent, delete=False, encoding="utf-8", newline="\n",
    )
    staged = Path(handle.name)
    try:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(staged, path)
    except BaseException:
        handle.close()
        staged.unlink(missing_ok=True)
        raise


def as_vector(v):
    if isinstance(v, str):
        v = json.loads(v)
    return np.asarray(v, dtype=np.float64).ravel()


def block(frame, columns):
    return np.hstack([np.vstack([as_vector(v) for v in frame[c]]) for c in columns])


def main() -> int:
    args = parse_args()
    actual = hashlib.sha256(args.paired_path.read_bytes()).hexdigest()
    if actual != EXPECTED_SHA:
        raise SystemExit(f"input hash changed: {actual}")

    df = pd.read_csv(args.paired_path)
    train = df[df.split.isin(("representation_train", "development"))].reset_index(drop=True)
    evalu = df[df.split == "validation"].reset_index(drop=True)
    y_tr = train[TARGET].to_numpy(float)
    y_ev = evalu[TARGET].to_numpy(float)
    mean_ref = float(np.mean(y_tr))
    denom = float(np.sqrt(np.mean((mean_ref - y_ev) ** 2)))

    con_tr, con_ev = block(train, ("feature_vector",)), block(evalu, ("feature_vector",))
    hcols = ("feature_vector", "context_history", "context_action_history")
    hist_tr, hist_ev = block(train, hcols), block(evalu, hcols)
    zero_gain = 1.0 - float(np.sqrt(np.mean(y_ev ** 2))) / denom

    seeds = [PHASE0_SEED] + [PHASE0_SEED + i for i in range(1, N_SEEDS)]
    rows = []
    threadpools_before = threadpool_info()
    with threadpool_limits(limits=1):
        threadpools_during = threadpool_info()
        for sd in seeds:
            g = {}
            for tag, xt, xe in (("contrast", con_tr, con_ev), ("history", hist_tr, hist_ev)):
                m = ExtraTreesRegressor(n_estimators=300, random_state=sd, n_jobs=1).fit(xt, y_tr)
                g[tag] = 1.0 - float(np.sqrt(np.mean((m.predict(xe) - y_ev) ** 2))) / denom
            num = 100 * (g["contrast"] - zero_gain)
            den = 100 * (g["history"] - g["contrast"])
            rows.append({"seed": sd, "gain_contrast_pct": 100 * g["contrast"],
                         "gain_history_pct": 100 * g["history"],
                         "numerator_pp": num, "denominator_pp": den,
                         "factor": num / den if den != 0 else float("nan")})
            print(f"seed {sd}  contrast {100*g['contrast']:6.2f}%  history {100*g['history']:6.2f}%"
                  f"  num {num:+6.2f}pp  den {den:+6.2f}pp  factor {num/den:>10.1f}")

    t = pd.DataFrame(rows)
    print("\n" + "=" * 84)
    print("Everything below varies ONLY the ExtraTrees random_state. Data, split, fold "
          "protocol,\nfeature blocks and metric are byte-identical across rows.")
    print("=" * 84)
    for col, fmt in (("numerator_pp", "{:+.2f}"), ("denominator_pp", "{:+.2f}"),
                     ("factor", "{:.1f}")):
        v = t[col].to_numpy()
        print(f"{col:<16} median {fmt.format(np.median(v)):>10}   "
              f"min {fmt.format(v.min()):>10}   max {fmt.format(v.max()):>10}   "
              f"sd {np.std(v):>10.2f}")
    print(f"\ndenominator <= 0 in {int((t.denominator_pp <= 0).sum())} of {len(t)} seeds")
    print(f"factor spans {t.factor.min():.1f} to {t.factor.max():.1f} "
          f"across {len(t)} seeds of the same estimator")
    print(f"\nphase0/baseline_ladder.py published: numerator +16.38 pp, denominator "
          f"+0.82 pp, factor 19.9")
    print(f"this environment at the same seed {PHASE0_SEED}: numerator "
          f"{t.numerator_pp.iloc[0]:+.2f} pp, denominator {t.denominator_pp.iloc[0]:+.2f} pp, "
          f"factor {t.factor.iloc[0]:.1f}")

    report = {
        "input_sha256": actual,
        "what_varies": "ExtraTreesRegressor random_state only",
        "canonical_result": bool(args.update_reference),
        "runtime": runtime_metadata(threadpools_before, threadpools_during),
        "n_seeds": len(t),
        "phase0_published": {"numerator_pp": 16.376550268185774,
                             "denominator_pp": 0.8243617337080789,
                             "factor": 19.87,
                             "environment_note": "historical environment was not fully preserved"},
        "per_seed": rows,
        "summary": {c: {"median": float(t[c].median()), "min": float(t[c].min()),
                        "max": float(t[c].max()), "sd": float(t[c].std())}
                    for c in ("numerator_pp", "denominator_pp", "factor")},
        "denominator_nonpositive_seeds": int((t.denominator_pp <= 0).sum()),
        "evidence_class": "retrospective_diagnostic_not_confirmatory",
    }
    write_json_atomic(args.output, report)
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
