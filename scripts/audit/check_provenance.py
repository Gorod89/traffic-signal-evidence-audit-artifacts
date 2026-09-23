"""Verify the public side of docs/PROVENANCE.csv without private inputs."""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "docs" / "PROVENANCE.csv"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_COLUMNS = {
    "public_path",
    "logical_source",
    "source_sha256",
    "public_sha256",
    "transformation",
    "evidence_role",
    "storage_policy",
    "rights_holder",
    "upstream_license",
    "release_license",
    "redistribution_status",
}

ALLOWED_RELEASE_LICENSES = {"MIT", "CC-BY-4.0", "NOASSERTION"}
ALLOWED_REDISTRIBUTION = {
    "include_in_open_deposit",
    "exclude_from_open_deposit",
}


def digest_for_policy(path: Path, policy: str) -> str:
    payload = path.read_bytes()
    if policy == "git_lf":
        payload = payload.replace(b"\r\n", b"\n")
    elif policy != "exact":
        raise ValueError(f"unknown storage policy: {policy}")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    errors: list[str] = []
    with REGISTRY.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != REQUIRED_COLUMNS:
            errors.append("PROVENANCE.csv has unexpected columns")
        rows = list(reader)

    seen: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        public_path = row.get("public_path", "")
        pure = PurePosixPath(public_path)
        if not public_path or pure.is_absolute() or ".." in pure.parts:
            errors.append(f"line {line_number}: unsafe public_path")
            continue
        if public_path in seen:
            errors.append(f"line {line_number}: duplicate public_path {public_path}")
        seen.add(public_path)

        logical_source = row.get("logical_source", "")
        if not logical_source.startswith("private-archive:"):
            errors.append(f"line {line_number}: source is not a logical archive label")
        if re.search(r"(^|[\\/])[A-Za-z]:[\\/]", logical_source) or "C:\\" in logical_source:
            errors.append(f"line {line_number}: absolute developer path disclosed")

        source_sha = row.get("source_sha256", "")
        public_sha = row.get("public_sha256", "")
        if not SHA256_RE.fullmatch(source_sha):
            errors.append(f"line {line_number}: invalid source_sha256")
        if not SHA256_RE.fullmatch(public_sha):
            errors.append(f"line {line_number}: invalid public_sha256")

        rights_holder = row.get("rights_holder", "").strip()
        upstream_license = row.get("upstream_license", "").strip()
        release_license = row.get("release_license", "").strip()
        redistribution = row.get("redistribution_status", "").strip()
        if not rights_holder:
            errors.append(f"line {line_number}: empty rights_holder")
        if not upstream_license:
            errors.append(f"line {line_number}: empty upstream_license")
        if release_license not in ALLOWED_RELEASE_LICENSES:
            errors.append(f"line {line_number}: invalid release_license")
        if redistribution not in ALLOWED_REDISTRIBUTION:
            errors.append(f"line {line_number}: invalid redistribution_status")
        if release_license == "NOASSERTION" and redistribution != "exclude_from_open_deposit":
            errors.append(
                f"line {line_number}: NOASSERTION must be excluded from open deposit"
            )
        if upstream_license == "NOASSERTION" and redistribution != "exclude_from_open_deposit":
            errors.append(
                f"line {line_number}: unresolved upstream rights must be excluded"
            )

        role = row.get("evidence_role", "")
        if role in {"historical_source", "portable_source"} and release_license != "MIT":
            errors.append(f"line {line_number}: author-created software must use MIT")
        if role in {
            "historical_output",
            "corrected_rerun",
            "rerun",
            "portable_historical_output",
        } and release_license != "CC-BY-4.0":
            errors.append(
                f"line {line_number}: author-created research record must use CC-BY-4.0"
            )

        path = ROOT.joinpath(*pure.parts)
        if not path.is_file():
            errors.append(f"line {line_number}: missing {public_path}")
            continue
        try:
            actual_sha = digest_for_policy(path, row.get("storage_policy", ""))
        except ValueError as exc:
            errors.append(f"line {line_number}: {exc}")
            continue
        if actual_sha != public_sha:
            errors.append(
                f"line {line_number}: {public_path} digest {actual_sha} != {public_sha}"
            )

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(f"provenance registry: OK ({len(rows)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
