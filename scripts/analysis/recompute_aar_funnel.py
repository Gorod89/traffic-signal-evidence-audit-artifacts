#!/usr/bin/env python3
"""Recompute the V3 full-method activity funnel with both AAR estimands.

The trace directory is supplied explicitly because the large historical traces
are held in the external artifact bundle rather than in this Git repository.
The script pins the byte manifest before reading records and retains the legacy
pooled fields while adding separately named pooled and equal-weight-cluster
results from the corrected public AAR implementation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evidence_tools.aar import authority_activity_ratio  # noqa: E402
from integrity import pin_manifest, write_json_atomic  # noqa: E402


TRACE_MANIFEST_SHA256 = (
    "f88a69bff6d57149098a135da683b2028dbfd537cc2e591550366955d30d5e8a"
)
TRACE_FILE_COUNT = 72
METHOD = "ergs_v3"
BOOTSTRAP_SEED = 20_260_813
BOOTSTRAP_REPLICATES = 4_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(args.trace_dir.glob("*_traces.json"))
    manifest = pin_manifest(
        files,
        TRACE_MANIFEST_SHA256,
        label="V3 raw traces",
        count=TRACE_FILE_COUNT,
    )

    pooled_total = 0
    pooled_accepted = 0
    per_method_accepted: Counter[str] = Counter()
    executed: list[str] = []
    incumbent: list[str] = []
    clusters: list[tuple[str, int]] = []
    disagreed: list[bool] = []
    reasons: Counter[str] = Counter()

    for path in files:
        records = json.loads(path.read_text(encoding="utf-8"))
        for record in records:
            method = record.get("method")
            pooled_total += 1
            if record.get("accepted"):
                pooled_accepted += 1
                per_method_accepted[method] += 1
            if method != METHOD:
                continue

            baseline = record["baseline_action"]
            candidate = record["candidate_action"]
            accepted = bool(record.get("accepted"))
            differs = baseline != candidate
            executed.append(candidate if accepted else baseline)
            incumbent.append(baseline)
            clusters.append((str(record["network"]), int(record["seed"])))
            disagreed.append(differs)
            if differs:
                reasons[record.get("shield_block_reason") or "accepted"] += 1

    report = authority_activity_ratio(
        executed,
        incumbent,
        clusters,
        scope="V3 full method over all recorded proposal evaluations",
        opportunity_unit="one recorded proposal evaluation",
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED,
    )

    # The disagreement interval uses the same corrected engine with a binary
    # surrogate action. Its meaning is explicitly separate from the AAR.
    disagreement_report = authority_activity_ratio(
        ["different" if value else "same" for value in disagreed],
        ["same"] * len(disagreed),
        clusters,
        scope="V3 full-method proposer disagreement",
        opportunity_unit="one recorded proposal evaluation",
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED,
    )

    pooled = report.pooled
    cluster_mean = report.equal_weight_cluster_mean
    assert pooled.ci95 is not None
    assert cluster_mean is not None and cluster_mean.ci95 is not None
    disagreement = disagreement_report.pooled
    assert disagreement.ci95 is not None

    output = {
        "schema_version": "aar-funnel-v2",
        "evidence_status": "retrospective diagnostic from archived V3 traces",
        "input_manifest_sha256": manifest,
        "bootstrap": {
            "unit": "(network, seed) cluster",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
        },
        "pooled_figure": {
            "evaluations": pooled_total,
            "accepted": pooled_accepted,
            "accepted_share": pooled_accepted / pooled_total,
            "accepted_by_method": dict(per_method_accepted.most_common()),
            "note": "The dormancy figure pools all eight V3 method variants.",
        },
        "full_method_funnel": {
            "method": METHOD,
            "evaluations": report.decisions,
            "proposer_disagreed": disagreement.numerator,
            "disagreement_rate": disagreement.estimate,
            "disagreement_rate_ci95": list(disagreement.ci95),
            "refusals_among_disagreements": dict(reasons.most_common()),
            "executed_deviations": report.active_decisions,
            "aar": pooled.estimate,
            "aar_ci95": list(pooled.ci95),
            "aar_estimands": {
                "pooled": {
                    "numerator": pooled.numerator,
                    "denominator": pooled.denominator,
                    "estimate": pooled.estimate,
                    "ci95": list(pooled.ci95),
                },
                "equal_weight_cluster_mean": {
                    "clusters": cluster_mean.clusters,
                    "estimate": cluster_mean.estimate,
                    "ci95": list(cluster_mean.ci95),
                },
            },
            "clusters": report.clusters,
            "active_clusters": report.active_clusters,
        },
    }
    write_json_atomic(args.output, output)
    print(
        "V3 AAR pooled "
        f"{100 * pooled.estimate:.6f}% [{100 * pooled.ci95[0]:.6f}, "
        f"{100 * pooled.ci95[1]:.6f}]%; cluster mean "
        f"{100 * cluster_mean.estimate:.6f}% "
        f"[{100 * cluster_mean.ci95[0]:.6f}, {100 * cluster_mean.ci95[1]:.6f}]%"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
