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
import contextlib
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
    parser.add_argument(
        "--estimator-jobs",
        type=int,
        default=1,
        help="ExtraTrees n_jobs setting; canonical reference uses 1",
    )
    thread_control = parser.add_mutually_exclusive_group()
    thread_control.add_argument(
        "--thread-limit",
        type=int,
        dest="thread_limit",
        help="numerical thread-pool limit; canonical reference uses 1",
    )
    thread_control.add_argument(
        "--no-thread-limit",
        action="store_const",
        const=None,
        dest="thread_limit",
        help="do not apply threadpoolctl limits",
    )
    parser.set_defaults(thread_limit=1)
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
    if args.estimator_jobs == 0 or args.estimator_jobs < -1:
        parser.error("--estimator-jobs must be -1 or a positive integer")
    if args.thread_limit is not None and args.thread_limit < 1:
        parser.error("--thread-limit must be a positive integer")
    if args.update_reference and (args.estimator_jobs != 1 or args.thread_limit != 1):
        parser.error(
            "the canonical reference requires --estimator-jobs 1 and --thread-limit 1"
        )
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


def runtime_metadata(
    threadpools_before: list[dict],
    threadpools_during: list[dict],
    *,
    estimator_jobs: int,
    thread_limit: int | None,
) -> dict:
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
        "estimator_jobs": estimator_jobs,
        "thread_limit": thread_limit,
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


def compute_report(
    paired_path: Path,
    *,
    estimator_jobs: int,
    thread_limit: int | None,
    canonical_result: bool,
    print_progress: bool = True,
) -> dict:
    """Fit the forty seeded ladders and return a fully described result."""

    actual = hashlib.sha256(paired_path.read_bytes()).hexdigest()
    if actual != EXPECTED_SHA:
        raise SystemExit(f"input hash changed: {actual}")

    df = pd.read_csv(paired_path)
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
    thread_context = (
        threadpool_limits(limits=thread_limit)
        if thread_limit is not None
        else contextlib.nullcontext()
    )
    with thread_context:
        threadpools_during = threadpool_info()
        for sd in seeds:
            g = {}
            for tag, xt, xe in (("contrast", con_tr, con_ev), ("history", hist_tr, hist_ev)):
                m = ExtraTreesRegressor(
                    n_estimators=300,
                    random_state=sd,
                    n_jobs=estimator_jobs,
                ).fit(xt, y_tr)
                g[tag] = 1.0 - float(np.sqrt(np.mean((m.predict(xe) - y_ev) ** 2))) / denom
            num = 100 * (g["contrast"] - zero_gain)
            den = 100 * (g["history"] - g["contrast"])
            rows.append({"seed": sd, "gain_contrast_pct": 100 * g["contrast"],
                         "gain_history_pct": 100 * g["history"],
                         "numerator_pp": num, "denominator_pp": den,
                         "factor": num / den if den != 0 else float("nan")})
            if print_progress:
                print(
                    f"seed {sd}  contrast {100*g['contrast']:6.2f}%  "
                    f"history {100*g['history']:6.2f}%  num {num:+6.2f}pp  "
                    f"den {den:+6.2f}pp  factor {num/den:>10.1f}"
                )

    t = pd.DataFrame(rows)
    if print_progress:
        print("\n" + "=" * 84)
        print(
            "Everything below varies ONLY the ExtraTrees random_state. Data, split, "
            "fold protocol,\nfeature blocks and metric are byte-identical across rows."
        )
        print("=" * 84)
        for col, fmt in (
            ("numerator_pp", "{:+.2f}"),
            ("denominator_pp", "{:+.2f}"),
            ("factor", "{:.1f}"),
        ):
            values = t[col].to_numpy()
            print(
                f"{col:<16} median {fmt.format(np.median(values)):>10}   "
                f"min {fmt.format(values.min()):>10}   "
                f"max {fmt.format(values.max()):>10}   "
                f"sd {np.std(values):>10.2f}"
            )
        print(
            f"\ndenominator <= 0 in {int((t.denominator_pp <= 0).sum())} "
            f"of {len(t)} seeds"
        )
        print(
            f"factor spans {t.factor.min():.1f} to {t.factor.max():.1f} "
            f"across {len(t)} seeds of the same estimator"
        )
        print(
            "\nphase0/baseline_ladder.py published: numerator +16.38 pp, "
            "denominator +0.82 pp, factor 19.9"
        )
        print(
            f"this environment at the same seed {PHASE0_SEED}: numerator "
            f"{t.numerator_pp.iloc[0]:+.2f} pp, denominator "
            f"{t.denominator_pp.iloc[0]:+.2f} pp, factor {t.factor.iloc[0]:.1f}"
        )

    report = {
        "input_sha256": actual,
        "what_varies": "ExtraTreesRegressor random_state only",
        "canonical_result": canonical_result,
        "runtime": runtime_metadata(
            threadpools_before,
            threadpools_during,
            estimator_jobs=estimator_jobs,
            thread_limit=thread_limit,
        ),
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
    return report


def main() -> int:
    args = parse_args()
    report = compute_report(
        args.paired_path,
        estimator_jobs=args.estimator_jobs,
        thread_limit=args.thread_limit,
        canonical_result=bool(args.update_reference),
    )
    write_json_atomic(args.output, report)
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
