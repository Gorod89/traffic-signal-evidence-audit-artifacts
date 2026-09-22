"""E+ step 1: run the preregistered V8 graph search at its locked default budget.

Writes SEARCH_PROTOCOL.json / SELECTED_AUDIT.json / SUMMARY.json / trials/*.json
into ``v10_proposal/phase0/eplus/runs/locked``.  Nothing under ``v8/`` is touched.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import eplus_common  # noqa: F401  (path setup side effect)
import numpy as np
import torch

from v8search.graph_search import GraphSearchConfig, run_graph_search


def main() -> int:
    output = Path(__file__).resolve().parent / "runs" / "locked"
    output.mkdir(parents=True, exist_ok=True)
    config = GraphSearchConfig()  # untouched locked defaults
    start = time.perf_counter()
    summary = run_graph_search(output, config)
    elapsed = time.perf_counter() - start
    environment = {
        "wall_clock_seconds": elapsed,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "config": asdict(config),
        "config_source": "v8search.graph_search.GraphSearchConfig() defaults, unmodified",
    }
    (output / "RUN_ENVIRONMENT.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"locked search finished in {elapsed:.1f}s")
    print(json.dumps(summary["matched_comparisons"], indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
