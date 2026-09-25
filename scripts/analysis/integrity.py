"""Input pinning and atomic output for the Phase-0 analyses.

The paper claims that every number it reports comes from a script that pins the
SHA-256 of its input, refuses to run if the input changed, and writes its result
atomically.  That claim was true of the scripts reading a single CSV and false
of the two reading directories of trace archives, which globbed whatever was on
disk; and no script wrote atomically, so an interrupted run could leave a
truncated JSON that a later analysis would read as valid.  This module supplies
both properties in one place so the claim holds for the whole pipeline.

A directory of archives is pinned by a manifest hash: file names and bytes in
sorted order, folded into one digest.  That detects a changed file, a removed
file and an added file, which a per-file hash list checked loosely would not.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Sequence


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_sha256(paths: Sequence[Path]) -> str:
    """Fold names and bytes of an ordered file set into a single digest."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def pin_file(path: Path, expected: str, *, label: str | None = None) -> str:
    actual = file_sha256(path)
    if actual != expected:
        raise SystemExit(
            f"input hash changed for {label or path}: expected {expected}, got {actual}"
        )
    return actual


def pin_manifest(
    paths: Iterable[Path], expected: str, *, label: str, count: int | None = None
) -> str:
    paths = sorted(paths)
    if not paths:
        raise SystemExit(f"no input files found for {label}")
    if count is not None and len(paths) != count:
        raise SystemExit(f"{label}: expected {count} files, found {len(paths)}")
    actual = manifest_sha256(paths)
    if actual != expected:
        raise SystemExit(
            f"input manifest changed for {label}: expected {expected}, got {actual}"
        )
    return actual


def write_json_atomic(path: Path, payload: object, *, indent: int = 2) -> None:
    """Write JSON through a temporary file in the same directory, then rename.

    os.replace is atomic within a filesystem, so a reader never observes a
    partially written result and an interrupted run leaves the previous result
    intact rather than a truncated one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=indent)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
