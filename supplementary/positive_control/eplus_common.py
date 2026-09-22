"""Shared plumbing for the E+ positive-control run of the V8 graph search.

Read-only with respect to ``v8/``: this module only inserts ``v8/src`` on
``sys.path`` and imports.  Nothing under ``v8/`` is created or modified.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

EPLUS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EPLUS_DIR.parents[1]
V8_ROOT = Path(
    os.environ.get("V8_SOURCE_ROOT", REPO_ROOT / "external" / "v8")
).resolve()
V8_SRC = V8_ROOT / "src"

if str(V8_SRC) not in sys.path:
    sys.path.insert(0, str(V8_SRC))


def repo_paths() -> dict[str, Path]:
    return {
        "eplus": EPLUS_DIR,
        "repo": REPO_ROOT,
        "v8_root": V8_ROOT,
        "v8_src": V8_SRC,
    }
