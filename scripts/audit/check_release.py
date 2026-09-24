#!/usr/bin/env python3
"""Fail closed on accidental local, credential, or private working material."""

from __future__ import annotations

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
    re.compile(r"\bChatGPT\b", re.I),
    re.compile(r"\bClaude\b", re.I),
    re.compile(r"\bOpenAI\b", re.I),
    re.compile(r"\bAnthropic\b", re.I),
)


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
