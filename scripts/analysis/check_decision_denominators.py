"""Validate the published V3/V4 decision-denominator aggregate.

Without a private source root, this checker validates the public artifact's
schema, pinned digest strings, arithmetic partitions and reported rates.  That
mode does not independently verify the omitted source bytes.  With
``--source-root`` (or ``ARTICLE_SOURCE_ROOT``), it additionally hashes all four
private sources and re-aggregates the V3 run table and the available V4 summary
fields against those bytes.  It never writes to the tree.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts" / "derived" / "decision_denominators.json"
DORMANCY_ARTIFACT = ROOT / "artifacts" / "derived" / "phase0" / "dormancy_anatomy.json"

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
        "budget_block_flag_recorded": 815_845,
        "other_nonquery_decisions": 247_276,
        "minimum_gate_active_decisions": 875_924,
        "candidate_matched_incumbent": 50_032,
        "candidate_differed_from_incumbent": 10_047,
        "executed_deviations": 70,
    },
    "v4_shadow": {
        "controller_decisions": 280_800,
        "candidate_matched_incumbent": 199_673,
        "candidate_differed_from_incumbent": 81_127,
        "scheduler_queries": 831,
        "queried_after_disagreement": 824,
        "queries_returning_incumbent_action": 7,
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


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def csv_integer_sum(rows: list[dict[str, str]], field: str) -> int:
    """Sum integer-valued CSV fields that may be serialised as decimals."""

    return sum(int(float(row[field])) for row in rows)


def main() -> int:
    args = parse_args()
    data = load_json(ARTIFACT)
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

        v3_run_table = args.source_root / EXPECTED_SOURCE_PATHS["v3_run_table"]
        if v3_run_table.is_file():
            with v3_run_table.open(encoding="utf-8", newline="") as stream:
                v3_rows = [
                    row for row in csv.DictReader(stream) if row.get("method") == "ergs_v3"
                ]
            if len(v3_rows) != 360:
                errors.append(f"v3_run_table: expected 360 ergs_v3 rows, found {len(v3_rows)}")
            else:
                direct_v3 = {
                    "controller_decisions": csv_integer_sum(v3_rows, "controller_decisions"),
                    "selected_for_proposal_evaluation": csv_integer_sum(v3_rows, "proposal_count"),
                    "candidate_differed_from_incumbent": csv_integer_sum(
                        v3_rows, "proposal_disagreement_count"
                    ),
                    "executed_deviations": csv_integer_sum(v3_rows, "accepted_override_count"),
                    "budget_block_flag_recorded": csv_integer_sum(
                        v3_rows, "budget_block_count"
                    ),
                }
                direct_v3["not_selected_for_proposal_evaluation"] = (
                    direct_v3["controller_decisions"]
                    - direct_v3["selected_for_proposal_evaluation"]
                )
                direct_v3["candidate_matched_incumbent"] = (
                    direct_v3["selected_for_proposal_evaluation"]
                    - direct_v3["candidate_differed_from_incumbent"]
                )
                direct_v3["other_nonquery_decisions"] = (
                    direct_v3["not_selected_for_proposal_evaluation"]
                    - direct_v3["budget_block_flag_recorded"]
                )
                direct_v3["minimum_gate_active_decisions"] = (
                    direct_v3["budget_block_flag_recorded"]
                    + direct_v3["selected_for_proposal_evaluation"]
                )
                for field, actual_value in direct_v3.items():
                    if data["v3_full_method"].get(field) != actual_value:
                        errors.append(
                            f"v3_run_table: {field}={actual_value} does not match artifact"
                        )

        v3_aggregate = args.source_root / EXPECTED_SOURCE_PATHS["v3_aggregate_check"]
        if v3_aggregate.is_file():
            aggregate = load_json(v3_aggregate).get("ergs_v3", {})
            aggregate_fields = {
                "controller_decisions": "controller_decisions",
                "selected_for_proposal_evaluation": "proposal_count",
                "candidate_differed_from_incumbent": "proposal_disagreement_count",
                "executed_deviations": "accepted_override_count",
            }
            for artifact_field, source_field in aggregate_fields.items():
                actual_value = int(float(aggregate.get(source_field, -1)))
                if data["v3_full_method"].get(artifact_field) != actual_value:
                    errors.append(
                        f"v3_aggregate_check: {source_field}={actual_value} does not match artifact"
                    )

        v4_totals_path = args.source_root / EXPECTED_SOURCE_PATHS["v4_run_totals"]
        if v4_totals_path.is_file():
            v4_totals = load_json(v4_totals_path).get("by_method", {}).get("ergs_cfra", {})
            direct_v4 = {
                "controller_decisions": int(float(v4_totals.get("controller_decisions", -1))),
                "scheduler_queries": int(float(v4_totals.get("llm_calls_count", -1))),
                "candidate_differed_from_incumbent": int(
                    float(v4_totals.get("proposal_disagreement_count", -1))
                ),
                "accepted_overrides": int(float(v4_totals.get("accepted_override_count", -1))),
            }
            direct_v4["candidate_matched_incumbent"] = (
                direct_v4["controller_decisions"]
                - direct_v4["candidate_differed_from_incumbent"]
            )
            for field, actual_value in direct_v4.items():
                if data["v4_shadow"].get(field) != actual_value:
                    errors.append(
                        f"v4_run_totals: {field}={actual_value} does not match artifact"
                    )

        v4_stage_c_path = args.source_root / EXPECTED_SOURCE_PATHS["v4_stage_c_summary"]
        if v4_stage_c_path.is_file():
            v4_stage_c = load_json(v4_stage_c_path)
            for artifact_field, source_field in (
                ("controller_decisions", "controller_decisions"),
                ("accepted_overrides", "counterfactual_acceptances"),
            ):
                actual_value = int(v4_stage_c.get(source_field, -1))
                if data["v4_shadow"].get(artifact_field) != actual_value:
                    errors.append(
                        f"v4_stage_c_summary: {source_field}={actual_value} does not match artifact"
                    )

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
    if (
        v3["budget_block_flag_recorded"] + v3["other_nonquery_decisions"]
        != v3["not_selected_for_proposal_evaluation"]
    ):
        errors.append("V3 nonquery partition does not close")
    if (
        v3["budget_block_flag_recorded"] + v3["selected_for_proposal_evaluation"]
        != v3["minimum_gate_active_decisions"]
    ):
        errors.append("V3 minimum-gate-active identity does not close")
    if (
        v3["budget_block_flag_recorded"]
        + v3["other_nonquery_decisions"]
        + v3["selected_for_proposal_evaluation"]
        != v3["controller_decisions"]
    ):
        errors.append("V3 three-way decision partition does not close")
    v3_rates = v3["rates"]
    expected_v3_rates = {
        "selection_rate_all_decisions": v3["selected_for_proposal_evaluation"] / v3["controller_decisions"],
        "budget_flag_rate_nonquery_decisions": v3["budget_block_flag_recorded"] / v3["not_selected_for_proposal_evaluation"],
        "budget_flag_rate_all_decisions": v3["budget_block_flag_recorded"] / v3["controller_decisions"],
        "minimum_gate_active_rate_all_decisions": v3["minimum_gate_active_decisions"] / v3["controller_decisions"],
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
        v4["queried_after_disagreement"]
        + v4["queries_returning_incumbent_action"]
        != v4["scheduler_queries"]
    ):
        errors.append("V4 scheduler-query partition does not close")
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

    dormancy = load_json(DORMANCY_ARTIFACT).get("v4", {})
    if dormancy.get("queried") != v4["queried_after_disagreement"]:
        errors.append("V4 queried-disagreement count does not match dormancy artifact")
    if dormancy.get("accepted") != v4["accepted_overrides"]:
        errors.append("V4 accepted-override count does not match dormancy artifact")
    if dormancy.get("authority_decisions") != v4["candidate_differed_from_incumbent"]:
        errors.append("V4 disagreement count does not match dormancy artifact")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    source_status = "source bytes verified" if args.source_root is not None else "source bytes not supplied"
    print(f"decision denominator artifact: OK ({source_status})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
