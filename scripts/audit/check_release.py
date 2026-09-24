#!/usr/bin/env python3
"""Fail closed on accidental local, credential, or private working material."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEXT_SUFFIXES = {
    ".bib", ".cff", ".cfg", ".cls", ".csv", ".json", ".md", ".ps1",
    ".py", ".sty", ".tex", ".toml", ".txt", ".yaml", ".yml",
}
SKIP_PARTS = {".git", ".pytest_cache", "__pycache__"}
MAX_FILE_BYTES = 50 * 1024 * 1024
FORBIDDEN_PATH_PARTS = {"manuscript", "Definitions", "versions"}
FORBIDDEN_SUFFIXES = {".bib", ".bst", ".cls", ".eps", ".pdf", ".sty", ".tex"}

# Store only one-way signatures so the release guard does not reproduce the
# very product names that it is intended to keep out of the public package.
BLOCKED_TOKEN_DIGESTS = frozenset(
    {
        "57de4cf40144bdf7d00010f2f5557a7d642c2b9705309bfade167dd313e2ca93",
        "60965168ce762e949600281ba6d01fee136e5b6e8257b1f216f9025ed324474c",
        "c857d09db23e6822e3600bc06ad8d58f92ed62bc8efd81c753f77048662cb97d",
        "7d3194f79e645c42e4396dda38be04766810ec6a00d00aced3ffc2a0a1f1a9ef",
        "c70eca6b0f88f44d81a41311647e50fda1ac454ec04ffd442b0eb4743a993131",
        "5d72436256ada53828b51895a94bb8489e9f1ac4fe937a8024ef1594e7045ff6",
        "3ea125d0bff386e6754b3782b300016fc79a9cf8f8669c0a5c3db64467ddb681",
    }
)
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")

ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:\\Users\\", re.I),
    re.compile(r"[A-Za-z]:\\\\Users\\\\", re.I),
    re.compile(r"/c/Users/", re.I),
    re.compile(r"/home/[^/]+/", re.I),
)
SECRET_PATTERNS = (
    re.compile(r"\bghp_[A-Za-z0-9]+\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]+\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
)
FORBIDDEN_TEXT_PATTERNS = (
    re.compile(r"\bT-ITS\b", re.I),
    re.compile(r"Transactions\s+on\s+Intelligent\s+Transportation\s+Systems", re.I),
    re.compile(r"AUTHOR[- ]INPUT", re.I),
    re.compile(r"XX\.XXXX", re.I),
)


def contains_blocked_token(value: str) -> bool:
    """Match blocked product tokens without storing them in release text."""
    for token in TOKEN_PATTERN.findall(value):
        digest = hashlib.sha256(token.casefold().encode("utf-8")).hexdigest()
        if digest in BLOCKED_TOKEN_DIGESTS:
            return True
    return False


def main() -> int:
    problems: list[str] = []
    discovered_files = [path for path in ROOT.rglob("*") if path.is_file()]
    release_files: list[Path] = []
    for path in sorted(discovered_files):
        relative = path.relative_to(ROOT)
        if any(part in SKIP_PARTS for part in relative.parts):
            continue
        release_files.append(path)
        if path.is_symlink():
            problems.append(f"symbolic link is not allowed: {relative.as_posix()}")
        if any(part in FORBIDDEN_PATH_PARTS for part in relative.parts):
            problems.append(f"forbidden release path: {relative.as_posix()}")
        if contains_blocked_token(relative.as_posix()):
            problems.append(f"blocked product token in release path: {relative.as_posix()}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"forbidden release file type: {relative.as_posix()}")
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            problems.append(f"file exceeds 50 MiB: {relative.as_posix()} ({size} bytes)")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if relative.as_posix() != "scripts/audit/check_release.py":
                for pattern in ABSOLUTE_PATH_PATTERNS:
                    if pattern.search(line):
                        problems.append(
                            f"developer absolute path: {relative.as_posix()}:{line_number}"
                        )
                        break
                for pattern in FORBIDDEN_TEXT_PATTERNS:
                    if pattern.search(line):
                        problems.append(
                            f"forbidden release text: {relative.as_posix()}:{line_number}"
                        )
                        break
                if contains_blocked_token(line):
                    problems.append(
                        f"blocked product token in release text: "
                        f"{relative.as_posix()}:{line_number}"
                    )
            for pattern in SECRET_PATTERNS:
                if pattern.search(line):
                    problems.append(
                        f"secret-like value: {relative.as_posix()}:{line_number}"
                    )
                    break
    if problems:
        print("release check failed:")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print(f"release check passed: {len(release_files)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
