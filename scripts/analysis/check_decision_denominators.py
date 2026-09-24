"""Validate the published V3/V4 decision-denominator aggregate.

The row-level sources are intentionally absent from the public package.  This
checker therefore verifies the frozen aggregate counts, their arithmetic
partitions, reported rates and source digests.  It never writes to the tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts" / "derived" / "decision_denominators.json"

EXPECTED_SOURCE_HASHES = {
    "v3_run_table": "1fd36882fe0e42dc38bacdafbe06a0ae53d84b70a73ef612d5e39c9b63e14297",
    "v3_aggregate_check": "8b5f6735ef3d9089ea3e10aaf5f89394fd169ba943b8281d8924a8370c170ee9",
    "v4_stage_c_summary": "469c08f6abef6c300ecd0d33c8f0506fbf46b20d8972b6feb21e915d9ca74a7c",
    "v4_run_totals": "5249437e827382c7ca8ae4cac066ef4f79910f1e0bbdd4451d8d3b0b7300900a",
}

EXPECTED_SOURCE_PATHS = {
    "v3_run_table": Path("v3/results/analysis/all_runs.csv"),
    "v3_aggregate_check": Path(
        "paper_v1_v8/revision_v10/analysis_v10/FP8_system_setup/v3_opportunities_check.json"
    ),
    "v4_stage_c_summary": Path("v4/results/cfra/stage_c_shadow.json"),
    "v4_run_totals": Path(
        "paper_v1_v8/revision_v10/analysis_v10/FP3_dormancy_aar/v4_run_level_totals.json"
    ),
}

EXPECTED_COUNTS = {
    "v3_full_method": {
        "controller_decisions": 1_123_200,
        "not_selected_for_proposal_evaluation": 1_063_121,
        "selected_for_proposal_evaluation": 60_079,
        "candidate_matched_incumbent": 50_032,
        "candidate_differed_from_incumbent": 10_047,
        "executed_deviations": 70,
    },
    "v4_shadow": {
        "controller_decisions": 280_800,
        "candidate_matched_incumbent": 199_673,
        "candidate_differed_from_incumbent": 81_127,
        "queried_after_disagreement": 824,
        "not_queried_after_disagreement": 80_303,
        "accepted_overrides": 133,
    },
}


def close(actual: float, expected: float) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=5e-16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        help=(
            "optional authorised private-archive root; when supplied, verify "
            "the recorded source digests against the source bytes"
        ),
    )
    args = parser.parse_args()
    env_root = os.environ.get("ARTICLE_SOURCE_ROOT")
    if args.source_root is None and env_root:
        args.source_root = Path(env_root)
    if args.source_root is not None:
        args.source_root = args.source_root.resolve()
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    data = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    errors: list[str] = []

    if data.get("schema_version") != "decision-denominators-v1":
        errors.append("unexpected schema_version")

    for label, expected_hash in EXPECTED_SOURCE_HASHES.items():
        actual = data.get("sources", {}).get(label, {}).get("sha256")
        if actual != expected_hash:
            errors.append(f"{label}: source digest mismatch")

    if args.source_root is not None:
        for label, relative_path in EXPECTED_SOURCE_PATHS.items():
            source_path = args.source_root / relative_path
            if not source_path.is_file():
                errors.append(f"{label}: private source missing at {relative_path.as_posix()}")
                continue
            actual_hash = sha256(source_path)
            if actual_hash != EXPECTED_SOURCE_HASHES[label]:
                errors.append(f"{label}: private source bytes do not match recorded digest")

    for section, expected in EXPECTED_COUNTS.items():
        actual_section = data.get(section, {})
        for field, expected_value in expected.items():
            if actual_section.get(field) != expected_value:
                errors.append(f"{section}.{field}: expected {expected_value}")

    v3 = data["v3_full_method"]
    if (
        v3["not_selected_for_proposal_evaluation"]
        + v3["selected_for_proposal_evaluation"]
        != v3["controller_decisions"]
    ):
        errors.append("V3 selection partition does not close")
    if v3["candidate_matched_incumbent"] + v3["candidate_differed_from_incumbent"] != v3["selected_for_proposal_evaluation"]:
        errors.append("V3 proposal partition does not close")
    v3_rates = v3["rates"]
    expected_v3_rates = {
        "selection_rate_all_decisions": v3["selected_for_proposal_evaluation"] / v3["controller_decisions"],
        "incumbent_match_rate_selected": v3["candidate_matched_incumbent"] / v3["selected_for_proposal_evaluation"],
        "executed_deviation_rate_all_decisions": v3["executed_deviations"] / v3["controller_decisions"],
        "executed_deviation_rate_selected": v3["executed_deviations"] / v3["selected_for_proposal_evaluation"],
    }
    for field, expected_value in expected_v3_rates.items():
        if not close(v3_rates.get(field), expected_value):
            errors.append(f"v3_full_method.rates.{field}: arithmetic mismatch")

    v4 = data["v4_shadow"]
    if v4["candidate_matched_incumbent"] + v4["candidate_differed_from_incumbent"] != v4["controller_decisions"]:
        errors.append("V4 proposal partition does not close")
    if v4["queried_after_disagreement"] + v4["not_queried_after_disagreement"] != v4["candidate_differed_from_incumbent"]:
        errors.append("V4 query partition does not close")
    if (
        v4["candidate_matched_incumbent"]
        + v4["not_queried_after_disagreement"]
        + v4["queried_after_disagreement"]
        != v4["controller_decisions"]
    ):
        errors.append("V4 decision-path partition does not close")
    v4_rates = v4["rates"]
    expected_v4_rates = {
        "incumbent_match_rate_all_decisions": v4["candidate_matched_incumbent"] / v4["controller_decisions"],
        "disagreement_rate_all_decisions": v4["candidate_differed_from_incumbent"] / v4["controller_decisions"],
        "query_rate_disagreements": v4["queried_after_disagreement"] / v4["candidate_differed_from_incumbent"],
        "accepted_override_rate_all_decisions": v4["accepted_overrides"] / v4["controller_decisions"],
        "accepted_override_rate_disagreements": v4["accepted_overrides"] / v4["candidate_differed_from_incumbent"],
    }
    for field, expected_value in expected_v4_rates.items():
        if not close(v4_rates.get(field), expected_value):
            errors.append(f"v4_shadow.rates.{field}: arithmetic mismatch")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    source_status = "source bytes verified" if args.source_root is not None else "source bytes not supplied"
    print(f"decision denominator artifact: OK ({source_status})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
