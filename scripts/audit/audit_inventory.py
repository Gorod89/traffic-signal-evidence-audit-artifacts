#!/usr/bin/env python3
"""Validate the explicit public-support state of every artifact-map row."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MAP = ROOT / "docs" / "ARTIFACT_MAP.csv"
EXPECTED_ROWS = 17
EXPECTED_COUNTS = {
    "inspection_only_public": 9,
    "not_covered": 0,
    "private_archive_only": 2,
    "runnable_public": 6,
}
ALLOWED = {
    "runnable_public",
    "inspection_only_public",
    "private_archive_only",
    "not_covered",
}


def local_path(value: str) -> Path | None:
    """Return a repository path when a field is a single relative path."""

    value = value.strip()
    if not value or ":" in value or ";" in value or value.startswith("private"):
        return None
    candidate = ROOT / value
    return candidate if "/" in value or "\\" in value else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map",
        type=Path,
        default=MAP,
        help="artifact-map CSV to validate (default: docs/ARTIFACT_MAP.csv)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    problems: list[str] = []
    counts = {status: 0 for status in sorted(ALLOWED)}
    with args.map.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    for index, row in enumerate(rows, start=2):
        status = row.get("status", "").strip()
        analysis = row.get("analysis", f"row-{index}")
        if status not in ALLOWED:
            problems.append(f"{analysis}: invalid status {status!r}")
            continue
        counts[status] += 1

        script = local_path(row.get("script", ""))
        output = local_path(row.get("public_output", ""))
        public_input = local_path(row.get("public_inputs", ""))

        if status in {"runnable_public", "inspection_only_public"}:
            for label, path in (("script", script), ("output", output)):
                if path is None or not path.is_file():
                    problems.append(f"{analysis}: public {label} is missing")
        if status == "runnable_public" and public_input is not None:
            if not public_input.exists():
                problems.append(f"{analysis}: declared public input is missing")

    if len(rows) != EXPECTED_ROWS:
        problems.append(
            f"artifact map has {len(rows)} rows; expected the fixed inventory {EXPECTED_ROWS}"
        )
    if counts != EXPECTED_COUNTS:
        problems.append(
            f"artifact-map status counts {counts!r} != expected {EXPECTED_COUNTS!r}"
        )

    print(f"artifact map: {len(rows)} rows")
    for status, count in counts.items():
        print(f"  {status}: {count}")
    if problems:
        print("inventory audit failed:")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print("inventory audit passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
