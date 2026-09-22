#!/usr/bin/env python3
"""Create a clearly labelled portable view of an immutable historical JSON file.

The source is never modified. Absolute developer paths are replaced recursively
in string values and dictionary keys. The command prints the source byte count
and SHA-256 so that the portable view cannot be confused with the original.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_paths(value: Any, source_root: str, marker: str) -> Any:
    if isinstance(value, str):
        return value.replace(source_root, marker)
    if isinstance(value, list):
        return [replace_paths(item, source_root, marker) for item in value]
    if isinstance(value, dict):
        return {
            replace_paths(key, source_root, marker): replace_paths(item, source_root, marker)
            for key, item in value.items()
        }
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--marker", default="<SOURCE_ROOT>")
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    payload = json.loads(source_bytes.decode("utf-8-sig"))
    normalized = replace_paths(payload, args.source_root, args.marker)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "source_name": args.source.name,
                "source_bytes": len(source_bytes),
                "source_sha256": sha256_bytes(source_bytes),
                "portable_view": args.destination.name,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
