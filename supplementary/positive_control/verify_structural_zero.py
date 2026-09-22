"""E+ side-check: what part of the SCM target can the model class even represent?

``GraphPairPredictor`` builds both action branches from one shared node embedding
``common`` plus a single scalar phase indicator per node.  For a movement served
by neither the baseline nor the candidate phase both branch inputs are
identical, so the predicted candidate-minus-baseline delta is exactly zero there
by construction, for every family.

Those are precisely the movements whose true delta can only arrive through
routing edges.  This script measures (a) that the zero is exact, and (b) how
much label energy sits in that structurally unreachable region.

Runs on CPU so it does not contend with a GPU search.
"""

from __future__ import annotations

import json
from pathlib import Path

import eplus_common  # noqa: F401  (path setup side effect)
import numpy as np
import torch

from v8search.graph_models import (
    GraphModelConfig,
    GraphPairPredictor,
    collate_graph_samples,
)
from v8search.scm import HORIZONS, generate_scm_dataset


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
    batch = collate_graph_samples(dataset.samples)

    # phases must partition the movements for "served by neither" to be meaningful
    partition_violations = 0
    for sample in dataset.samples:
        served_counts = sample.graph.phases.sum(axis=0)
        if not np.all(served_counts == 1.0):
            partition_violations += 1

    baseline_phase = batch.phases.gather(
        1, batch.baseline_action[:, None, None].expand(-1, 1, batch.max_nodes)
    ).squeeze(1)
    candidate_phase = batch.phases.gather(
        1, batch.candidate_action[:, None, None].expand(-1, 1, batch.max_nodes)
    ).squeeze(1)
    served = ((baseline_phase + candidate_phase) > 0.5) & batch.node_mask
    graph_only = batch.node_mask & ~served

    report: dict[str, object] = {
        "samples": int(batch.batch_size),
        "phase_partition_violations": partition_violations,
        "node_slots": {
            "real": int(batch.node_mask.sum().item()),
            "served_by_baseline_or_candidate": int(served.sum().item()),
            "served_by_neither": int(graph_only.sum().item()),
        },
        "structural_zero_max_abs_prediction_on_unserved": {},
        "label_energy": {},
        "achievable_skill_ceiling": {},
    }

    for family in ("edge_null", "directed_mpnn", "edge_attention"):
        worst = 0.0
        for seed in (11, 23, 37):
            torch.manual_seed(seed)
            model = GraphPairPredictor(
                GraphModelConfig(family=family, width=48, depth=2, dropout=0.05)
            )
            for mode in ("train", "eval"):
                model.train(mode == "train")
                with torch.no_grad():
                    prediction = model(batch)
                masked = prediction * graph_only.unsqueeze(1).to(prediction.dtype)
                worst = max(worst, float(masked.abs().max().item()))
        report["structural_zero_max_abs_prediction_on_unserved"][family] = worst

    target = batch.target_queue_delta
    normalized = (target / batch.capacity.unsqueeze(1)) ** 2
    for horizon_index, horizon in enumerate(HORIZONS):
        values = normalized[:, horizon_index, :]
        total = float((values * batch.node_mask).sum().item())
        unreachable = float((values * graph_only).sum().item())
        report["label_energy"][f"H{horizon}"] = {
            "total_normalized_energy": total,
            "unreachable_normalized_energy": unreachable,
            "unreachable_share": unreachable / max(total, 1.0e-30),
        }
    pooled_total = float((normalized * batch.node_mask.unsqueeze(1)).sum().item())
    pooled_unreachable = float((normalized * graph_only.unsqueeze(1)).sum().item())
    report["label_energy"]["pooled"] = {
        "total_normalized_energy": pooled_total,
        "unreachable_normalized_energy": pooled_unreachable,
        "unreachable_share": pooled_unreachable / max(pooled_total, 1.0e-30),
    }
    report["achievable_skill_ceiling"]["pooled"] = 1.0 - pooled_unreachable / max(
        pooled_total, 1.0e-30
    )

    destination = output / "STRUCTURAL_ZERO_CHECK.json"
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
