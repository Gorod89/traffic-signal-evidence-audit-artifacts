"""E+ side-check: is the SCM label itself topology-dependent?

Everything else in this folder asks whether a *model* uses edges.  This script
asks the prior question about the *generator*: re-simulate both branches of every
sample from the identical fork queue and identical future arrivals, but on the
adjacency-shuffled graph and on the edge-free graph, and measure how far the
resulting candidate-minus-baseline target moves.

Pure NumPy, CPU only.
"""

from __future__ import annotations

import json
from pathlib import Path

import eplus_common  # noqa: F401  (path setup side effect)
import numpy as np

from v8search.scm import (
    HORIZONS,
    MovementGraph,
    adjacency_shuffle,
    generate_scm_dataset,
    simulate_branch,
)


def branch_target(sample, graph: MovementGraph) -> np.ndarray:
    baseline = simulate_branch(
        graph,
        sample.fork_queue,
        sample.baseline_action,
        sample.continuation_actions,
        sample.future_arrivals,
    )
    candidate = simulate_branch(
        graph,
        sample.fork_queue,
        sample.candidate_action,
        sample.continuation_actions,
        sample.future_arrivals,
    )
    return np.stack(
        [candidate[horizon - 1] - baseline[horizon - 1] for horizon in HORIZONS], axis=0
    )


def main() -> int:
    output = Path(__file__).resolve().parent / "runs"
    output.mkdir(parents=True, exist_ok=True)
    sizes = tuple(6 + (index % 9) for index in range(18))
    seeds = tuple(1103 + 17 * index for index in range(18))
    dataset = generate_scm_dataset(
        graph_sizes=sizes,
        graph_seeds=seeds,
        samples_per_graph=8,
        corruption_modes=("mcar", "burst", "spatial", "mnar"),
        seed=20260817,
    )

    accumulators = {
        name: {"target_energy": 0.0, "shift_energy": 0.0, "slots": 0.0}
        for name in ("adjacency_shuffle", "no_edge")
    }
    for index, sample in enumerate(dataset.samples):
        capacity = sample.graph.capacity[None, :]
        reference = sample.paired_queue_delta / capacity
        variants = {
            "adjacency_shuffle": adjacency_shuffle(sample, 0x5EED ^ index).graph,
            "no_edge": MovementGraph(
                seed=sample.graph.seed,
                routing=np.zeros_like(sample.graph.routing),
                capacity=sample.graph.capacity,
                saturation_flow=sample.graph.saturation_flow,
                phases=sample.graph.phases,
            ),
        }
        for name, graph in variants.items():
            altered = branch_target(sample, graph) / capacity
            accumulators[name]["target_energy"] += float(np.sum(reference**2))
            accumulators[name]["shift_energy"] += float(np.sum((altered - reference) ** 2))
            accumulators[name]["slots"] += float(reference.size)

    report = {
        "samples": len(dataset.samples),
        "interpretation": (
            "shift_relative_to_target = |target(perturbed graph) - target(true graph)|^2 "
            "/ |target(true graph)|^2, capacity normalized, pooled over H1/H2/H4. "
            "A value near zero would mean the label does not depend on topology."
        ),
        "variants": {
            name: {
                "target_energy": values["target_energy"],
                "shift_energy": values["shift_energy"],
                "shift_relative_to_target": values["shift_energy"]
                / max(values["target_energy"], 1.0e-30),
            }
            for name, values in accumulators.items()
        },
    }
    destination = output / "GENERATOR_GRAPH_DEPENDENCE.json"
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
