#!/usr/bin/env python3
"""Write or verify a deterministic SHA-256 release manifest.

Use ``--index`` immediately before a commit so hashes describe the exact Git
blobs, including attribute-driven end-of-line handling. Use ``--check`` after a
checkout to verify both file coverage and bytes without changing the tree.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "MANIFEST.sha256"
EXCLUDED_PARTS = {".git", "__pycache__"}


def excluded(relative: Path) -> bool:
    """Return true for VCS or generated paths outside the release payload."""
    return any(
        part in EXCLUDED_PARTS or part.endswith(".egg-info")
        for part in relative.parts
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def worktree_entries() -> list[tuple[str, str]]:
    paths: list[tuple[str, Path]] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == OUTPUT:
            continue
        relative = path.relative_to(ROOT)
        if excluded(relative):
            continue
        paths.append((relative.as_posix(), path))
    return [(relative, sha256(path)) for relative, path in sorted(paths)]


def index_entries() -> list[tuple[str, str]]:
    command = ["git", "-C", str(ROOT), "ls-files", "-z"]
    listed = subprocess.run(command, check=True, capture_output=True).stdout
    paths = [raw.decode("utf-8") for raw in listed.split(b"\0") if raw]
    entries: list[tuple[str, str]] = []
    for relative in sorted(paths):
        if relative == OUTPUT.name or excluded(Path(relative)):
            continue
        blob = subprocess.run(
            ["git", "-C", str(ROOT), "show", f":{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        entries.append((relative, sha256_bytes(blob)))
    return entries


def render(entries: list[tuple[str, str]]) -> str:
    return "".join(f"{digest}  {relative}\n" for relative, digest in entries)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--index", action="store_true", help="hash staged Git blobs")
    mode.add_argument("--check", action="store_true", help="verify a checkout")
    args = parser.parse_args()

    if args.check:
        if not OUTPUT.is_file():
            print(f"missing {OUTPUT.name}", file=sys.stderr)
            return 1
        expected = OUTPUT.read_text(encoding="utf-8")
        actual = render(worktree_entries())
        if actual != expected:
            expected_lines = set(expected.splitlines())
            actual_lines = set(actual.splitlines())
            for line in sorted(expected_lines - actual_lines)[:10]:
                print(f"missing or changed: {line}", file=sys.stderr)
            for line in sorted(actual_lines - expected_lines)[:10]:
                print(f"unexpected or changed: {line}", file=sys.stderr)
            return 1
        print(f"manifest check passed: {len(actual.splitlines())} files")
        return 0

    entries = index_entries() if args.index else worktree_entries()
    OUTPUT.write_text(render(entries), encoding="utf-8", newline="\n")
    source = "Git index" if args.index else "working tree"
    print(f"wrote {OUTPUT.name}: {len(entries)} files from {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
