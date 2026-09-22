"""Hashing, strict JSON loading, and immutable output primitives.

``atomic rename`` and ``write once`` are different guarantees.  The historical
pipeline used ``os.replace``: that prevented a truncated JSON file, but allowed
an earlier analysis to be overwritten.  The helpers here never replace an
existing evidence artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable


class IntegrityError(RuntimeError):
    """An input or immutable-output contract was violated."""


class DuplicateJSONKeyError(IntegrityError):
    """A JSON object contained the same key more than once."""


class ImmutableOutputError(IntegrityError):
    """An existing write-once path contains different bytes."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_bytes_with_sha256(path: Path, *, label: str) -> tuple[bytes, str]:
    """Read a regular non-symlink file once and hash those exact bytes."""

    if path.is_symlink() or not path.is_file():
        raise IntegrityError(f"{label} is missing, not a regular file, or a symlink: {path}")
    payload = path.read_bytes()
    return payload, hashlib.sha256(payload).hexdigest()


def read_verified_bytes(path: Path, expected: str, *, label: str) -> tuple[bytes, str]:
    if len(expected) != 64 or any(char not in "0123456789abcdefABCDEF" for char in expected):
        raise IntegrityError(f"{label} expected SHA-256 must be 64 hexadecimal characters")
    payload, actual = read_bytes_with_sha256(path, label=label)
    if actual.lower() != expected.lower():
        raise IntegrityError(
            f"{label} SHA-256 mismatch: expected {expected.lower()}, got {actual.lower()}"
        )
    return payload, actual.lower()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJSONKeyError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise IntegrityError(f"non-finite JSON number is forbidden: {value}")


def loads_json_strict(text: str, *, label: str = "JSON payload") -> Any:
    """Parse JSON while rejecting duplicate object keys."""
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except json.JSONDecodeError as exc:
        raise IntegrityError(f"invalid JSON in {label}: {exc}") from exc


def load_json_strict(path: Path) -> Any:
    """Load JSON while rejecting duplicate object keys."""

    return loads_json_strict(path.read_text(encoding="utf-8"), label=str(path))


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def manifest_sha256(paths: Iterable[Path], *, root: Path) -> str:
    """Hash relative names and file digests in a platform-independent order."""

    if root.is_symlink():
        raise IntegrityError(f"manifest root may not be a symlink: {root}")
    root = root.resolve()
    entries: list[tuple[str, str]] = []
    for path in paths:
        if path.is_symlink():
            raise IntegrityError(f"manifest entries may not be symlinks: {path}")
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise IntegrityError(f"manifest entry escapes root {root}: {resolved}") from exc
        if not resolved.is_file():
            raise IntegrityError(f"manifest entry is not a regular file: {resolved}")
        entries.append((relative, sha256_file(resolved)))
    if not entries:
        raise IntegrityError("manifest may not be empty")
    names = [relative for relative, _ in entries]
    if len(names) != len(set(names)):
        raise IntegrityError("manifest contains duplicate relative names")
    digest = hashlib.sha256()
    for relative, file_digest in sorted(entries):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str, *, label: str) -> str:
    _, actual = read_verified_bytes(path, expected, label=label)
    return actual


def write_bytes_once(path: Path, payload: bytes) -> bool:
    """Publish bytes atomically without ever replacing an existing path.

    Returns ``True`` when this call creates the file and ``False`` when an
    identical file already exists.  A different existing payload is an error.
    The temporary file is hard-linked to the destination, which makes creation
    atomic and fails if another process won the race.
    """

    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise ImmutableOutputError(f"write-once path may not traverse a symlink: {ancestor}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return False
        raise ImmutableOutputError(f"refusing to overwrite immutable output: {path}")

    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            if path.read_bytes() == payload:
                return False
            raise ImmutableOutputError(
                f"concurrent writer created different immutable output: {path}"
            )
        return True
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def write_json_once(path: Path, payload: Any) -> bool:
    return write_bytes_once(path, canonical_json_bytes(payload))
