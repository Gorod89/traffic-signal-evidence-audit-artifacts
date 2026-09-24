"""Compare canonical and unconstrained-parallel forty-seed results.

The canonical seed-sensitivity record uses one estimator job and a one-thread
numerical pool.  This control keeps the data, software environment and forty
tree seeds fixed while changing only execution controls to ``n_jobs=-1`` and no
``threadpoolctl`` limit.  Generated output goes to ``build/``; replacing the
retained control requires ``--update-reference``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from seed_sensitivity import (
    EXPECTED_SHA,
    PAIRED,
    REFERENCE,
    compute_report,
    write_json_atomic,
)

CONTROL_REFERENCE = HERE / "results" / "parallelism_control.json"
COMPARISON_FIELDS = (
    "gain_contrast_pct",
    "gain_history_pct",
    "numerator_pp",
    "denominator_pp",
    "factor",
)


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
        help="generated result path (default: build/parallelism_control.json)",
    )
    destination.add_argument(
        "--update-reference",
        action="store_true",
        help="explicitly replace results/parallelism_control.json",
    )
    args = parser.parse_args()
    args.paired_path = args.paired_path.resolve()
    args.output = (
        CONTROL_REFERENCE
        if args.update_reference
        else args.output or HERE / "build" / "parallelism_control.json"
    ).resolve()
    if not args.update_reference and args.output == CONTROL_REFERENCE.resolve():
        parser.error("writing the retained control requires --update-reference")
    return args


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_reports(canonical: dict, parallel: dict) -> dict:
    """Return an exact per-seed comparison of two seed-sensitivity reports."""

    if canonical.get("input_sha256") != parallel.get("input_sha256"):
        raise ValueError("input digests differ")
    if canonical.get("input_sha256") != EXPECTED_SHA:
        raise ValueError("unexpected paired-derivative digest")

    canonical_rows = {int(row["seed"]): row for row in canonical.get("per_seed", [])}
    parallel_rows = {int(row["seed"]): row for row in parallel.get("per_seed", [])}
    if canonical_rows.keys() != parallel_rows.keys():
        raise ValueError("seed sets differ")

    mismatches: list[dict] = []
    per_field_max = {field: 0.0 for field in COMPARISON_FIELDS}
    for seed in sorted(canonical_rows):
        for field in COMPARISON_FIELDS:
            expected = float(canonical_rows[seed][field])
            actual = float(parallel_rows[seed][field])
            delta = abs(actual - expected)
            per_field_max[field] = max(per_field_max[field], delta)
            if actual != expected:
                mismatches.append(
                    {
                        "seed": seed,
                        "field": field,
                        "canonical": expected,
                        "parallel": actual,
                        "absolute_delta": delta,
                    }
                )

    return {
        "n_seeds": len(canonical_rows),
        "fields": list(COMPARISON_FIELDS),
        "scalar_comparisons": len(canonical_rows) * len(COMPARISON_FIELDS),
        "mismatch_count": len(mismatches),
        "max_abs_delta": max(per_field_max.values(), default=0.0),
        "per_field_max_abs_delta": per_field_max,
        "mismatches": mismatches,
    }


def build_control(canonical: dict, parallel: dict) -> dict:
    comparison = compare_reports(canonical, parallel)
    return {
        "schema_version": "seed-parallelism-control-v1",
        "evidence_class": "review_time_reproducibility_control",
        "input_sha256": parallel["input_sha256"],
        "canonical_result_path": (
            "supplementary/incumbent_invariance/results/seed_sensitivity.json"
        ),
        "canonical_result_sha256": sha256(REFERENCE),
        "canonical_execution": {
            "estimator_jobs": canonical.get("runtime", {}).get("estimator_jobs"),
            "thread_limit": canonical.get("runtime", {}).get("thread_limit"),
        },
        "parallel_execution": {
            "estimator_jobs": parallel.get("runtime", {}).get("estimator_jobs"),
            "thread_limit": parallel.get("runtime", {}).get("thread_limit"),
        },
        "comparison": comparison,
        "interpretation": (
            "On this recorded host and software environment, changing from one "
            "estimator job and one numerical thread to n_jobs=-1 without an "
            "explicit numerical thread limit changed none of the 200 compared "
            "per-seed scalars. This controls the scheduling explanation on the "
            "tested host; it is not a cross-platform determinism guarantee."
        ),
        "parallel_run": parallel,
    }


def main() -> int:
    args = parse_args()
    canonical = json.loads(REFERENCE.read_text(encoding="utf-8"))
    parallel = compute_report(
        args.paired_path,
        estimator_jobs=-1,
        thread_limit=None,
        canonical_result=False,
    )
    control = build_control(canonical, parallel)
    mismatches = control["comparison"]["mismatch_count"]
    if args.update_reference and mismatches:
        print(f"control mismatch: {mismatches}; retained reference not replaced")
        return 1
    write_json_atomic(args.output, control)
    print(
        f"parallelism control: {control['comparison']['scalar_comparisons']} comparisons, "
        f"{mismatches} mismatches, max abs delta "
        f"{control['comparison']['max_abs_delta']}"
    )
    print(f"wrote {args.output}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
