"""Fast smoke test: does the untouched v8 graph search run at all on this GPU?"""

from __future__ import annotations

import json
import time
from pathlib import Path

import eplus_common  # noqa: F401  (path setup side effect)
import torch

from v8search.graph_search import GraphSearchConfig, run_graph_search


def main() -> int:
    output = Path(__file__).resolve().parent / "runs" / "smoke"
    config = GraphSearchConfig(
        graph_sizes=(5, 7, 9),
        graph_seeds=(101, 211, 307),
        samples_per_graph=2,
        corruption_modes=("mcar", "burst"),
        generation_seed=71,
        split_seed=73,
        search_seed=79,
        parameter_targets=(60_000,),
        depths=(2,),
        temporal_depths=(1,),
        learning_rates=(1.0e-3,),
        weight_decays=(0.0,),
        max_trials=3,
        epochs=1,
        patience=1,
        batch_size=2,
        mixed_precision=False,
        latency_warmups=1,
        latency_repetitions=3,
    )
    start = time.perf_counter()
    summary = run_graph_search(output, config)
    elapsed = time.perf_counter() - start
    print(f"smoke ok in {elapsed:.1f}s, torch={torch.__version__}")
    print(json.dumps({k: summary[k] for k in ("selected_trial_id", "trial_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
