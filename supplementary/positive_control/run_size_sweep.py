"""E+ step 4: how much training data does the graph detector need to wake up?

The locked V8 budget trains on 40 samples from 5 graph clusters.  This sweep
holds the audit clusters, the architectures and the capacity matching fixed and
varies only the training-set size, so the family contrast can be read as a
function of training data.

The development set is subsampled to 6 clusters purely to keep the per-epoch
early-stopping cost bounded; every reported number comes from the untouched
audit clusters.  Epoch counts are set so that each training size receives a
comparable number of gradient steps (small sets otherwise get starved).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import eplus_common  # noqa: F401  (path setup side effect)
import numpy as np
import torch

from run_scaled_search import ROLES, build_cluster_axes, evaluate_all_controls
from v8search.graph_models import GraphModelConfig, GraphPairPredictor
from v8search.graph_search import (
    GraphSearchConfig,
    GroupedGraphSplit,
    _train_trial,
    _trial_seed,
    deterministic_grouped_split,
    enumerate_graph_catalog,
    require_cuda_device,
    select_bounded_catalog,
)
from v8search.scm import generate_scm_dataset

# (training clusters, scenarios kept per cluster); the first point reproduces
# the locked V8 training-set size of 40 samples from 5 clusters.
SIZE_POINTS = ((5, 8), (12, 8), (12, 24), (24, 24), (48, 24), (72, 24))
TARGET_STEPS = 2400


def subset_split(
    full: GroupedGraphSplit, n_clusters: int, per_cluster: int, dev_clusters: int
) -> GroupedGraphSplit:
    train_seeds = tuple(sorted(full.train_graph_seeds)[:n_clusters])
    dev_seeds = tuple(sorted(full.development_graph_seeds)[:dev_clusters])
    by_seed: dict[int, list] = {seed: [] for seed in train_seeds}
    for sample in full.train:
        if sample.graph_seed in by_seed:
            by_seed[sample.graph_seed].append(sample)
    train = tuple(
        sample
        for seed in train_seeds
        for sample in sorted(by_seed[seed], key=lambda s: s.scenario_seed)[:per_cluster]
    )
    development = tuple(
        sample for sample in full.development if sample.graph_seed in set(dev_seeds)
    )
    return GroupedGraphSplit(
        train=train,
        development=development,
        audit=full.audit,
        train_graph_seeds=train_seeds,
        development_graph_seeds=dev_seeds,
        audit_graph_seeds=full.audit_graph_seeds,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Training-size sweep for the E+ run.")
    parser.add_argument("--clusters", type=int, default=120)
    parser.add_argument("--samples-per-graph", type=int, default=24)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--dev-clusters", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tag", type=str, default="sweep")
    args = parser.parse_args()

    output = Path(__file__).resolve().parent / "runs" / args.tag
    output.mkdir(parents=True, exist_ok=True)
    device = require_cuda_device("cuda")
    sizes, seeds = build_cluster_axes(args.clusters)
    dataset = generate_scm_dataset(
        graph_sizes=sizes,
        graph_seeds=seeds,
        samples_per_graph=args.samples_per_graph,
        corruption_modes=("mcar", "burst", "spatial", "mnar"),
        seed=20260817,
    )
    full_split = deterministic_grouped_split(dataset.samples, seed=8807)

    replicate_seeds = [19001, 24007, 31013][: args.replicates]
    rows: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    start = time.perf_counter()

    for n_clusters, per_cluster in SIZE_POINTS:
        split = subset_split(full_split, n_clusters, per_cluster, args.dev_clusters)
        steps_per_epoch = max(1, math.ceil(len(split.train) / args.batch_size))
        epochs = int(min(300, max(30, math.ceil(TARGET_STEPS / steps_per_epoch))))
        for replicate_index, search_seed in enumerate(replicate_seeds):
            config = GraphSearchConfig(
                graph_sizes=sizes,
                graph_seeds=seeds,
                samples_per_graph=args.samples_per_graph,
                generation_seed=20260817,
                split_seed=8807,
                search_seed=search_seed,
                parameter_targets=(60_000,),
                depths=(2,),
                temporal_depths=(1,),
                learning_rates=(1.0e-3,),
                weight_decays=(0.0,),
                max_trials=3,
                epochs=epochs,
                patience=15,
                batch_size=args.batch_size,
            )
            catalog = select_bounded_catalog(
                enumerate_graph_catalog(config), max_trials=3, seed=config.search_seed
            )
            for trial in catalog:
                trial_start = time.perf_counter()
                record, state = _train_trial(trial, split, config, device)
                model = GraphPairPredictor(GraphModelConfig(**record["model"])).to(device)
                model.load_state_dict(state)
                control_seed = _trial_seed(config.search_seed, trial.trial_id) ^ 0x2468ACE0
                key_prefix = f"n{len(split.train):04d}_r{replicate_index}_{trial.trial_id}"
                evaluated = evaluate_all_controls(model, split.audit, device, seed=control_seed)
                for control, payload in evaluated.items():
                    for name, value in payload.items():
                        arrays[f"{key_prefix}|audit|{control}|{name}"] = value
                elapsed = time.perf_counter() - trial_start
                rows.append(
                    {
                        "key_prefix": key_prefix,
                        "train_samples": len(split.train),
                        "train_clusters": n_clusters,
                        "scenarios_per_cluster": per_cluster,
                        "epochs_budget": epochs,
                        "replicate_index": replicate_index,
                        "search_seed": search_seed,
                        "trial_id": trial.trial_id,
                        "comparison_group": trial.comparison_group,
                        "family": record["model"]["family"],
                        "width": record["model"]["width"],
                        "depth": record["model"]["depth"],
                        "parameter_count": record["parameter_count"],
                        "best_epoch": record["best_epoch"],
                        "epochs_completed": record["epochs_completed"],
                        "development_intact_nmse": record["development_controls"]["intact"][
                            "metrics"
                        ]["capacity_normalized_mse"],
                        "train_seconds": elapsed,
                    }
                )
                print(
                    f"[n={len(split.train):5d} r{replicate_index}] "
                    f"{record['model']['family']:15s} ep={record['best_epoch']}/"
                    f"{record['epochs_completed']} (budget {epochs}) "
                    f"dev_nmse={rows[-1]['development_intact_nmse']:.6f} ({elapsed:.1f}s)",
                    flush=True,
                )

    manifest = {
        "schema_version": "eplus-size-sweep-1",
        "size_points": [list(point) for point in SIZE_POINTS],
        "target_gradient_steps": TARGET_STEPS,
        "audit_graph_seeds": list(full_split.audit_graph_seeds),
        "development_clusters_used_for_early_stopping": args.dev_clusters,
        "roles": list(ROLES),
        "replicate_search_seeds": replicate_seeds,
        "total_seconds": time.perf_counter() - start,
        "trials": rows,
    }
    (output / "SWEEP_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(output / "per_sample_residuals.npz", **arrays)
    print(f"sweep done in {manifest['total_seconds']:.1f}s -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
