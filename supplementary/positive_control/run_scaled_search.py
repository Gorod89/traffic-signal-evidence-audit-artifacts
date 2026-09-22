"""E+ step 2: scaled positive-control run of the V8 graph search.

The locked V8 budget trains on 40 samples and validates on a single graph
cluster, which is far too small to place an interval on a family contrast.  This
script keeps every V8 component byte-identical (imported, never edited) and only
enlarges the data/compute budget, then records per-sample residuals so that a
cluster bootstrap can be run afterwards.

Extra decomposition, not present in V8: every node is labelled ``served`` when
the baseline or candidate phase serves it, and ``graph_only`` otherwise.  Phase
tables in the SCM are a partition of the movements, so a ``graph_only`` node can
only acquire a nonzero candidate-minus-baseline delta through routing edges.
That subset is the sharpest positive control available.

Outputs land in ``v10_proposal/phase0/eplus/runs/scaled``.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import eplus_common  # noqa: F401  (path setup side effect)
import numpy as np
import torch

from v8search.graph_models import (
    GraphModelConfig,
    GraphPairPredictor,
    collate_graph_samples,
)
from v8search.graph_search import (
    CONTROL_NAMES,
    GraphSearchConfig,
    _train_trial,
    _trial_seed,
    apply_control,
    deterministic_grouped_split,
    enumerate_graph_catalog,
    require_cuda_device,
    select_bounded_catalog,
)
from v8search.scm import HORIZONS, generate_scm_dataset

ROLES = ("all", "served", "graph_only")


def build_cluster_axes(n_clusters: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Deterministic aligned (sizes, seeds); sizes cycle the locked 6..14 range."""

    sizes = tuple(6 + (index % 9) for index in range(n_clusters))
    seeds = tuple(1103 + 17 * index for index in range(n_clusters))
    if len(set(seeds)) != n_clusters:
        raise RuntimeError("graph seeds must be unique")
    return sizes, seeds


@torch.no_grad()
def per_sample_residuals(
    model: GraphPairPredictor, samples, device: torch.device
) -> dict[str, np.ndarray]:
    """Return per-sample / per-horizon / per-role residual sums.

    ``ss_resid`` and ``ss_zero`` are capacity-normalized squared sums; ``raw_ss``
    is the unnormalized squared error used for the movement RMSE.  Summing these
    over any cluster subset reproduces the V8 metric exactly.
    """

    model.eval()
    batch = collate_graph_samples(samples).to(device)
    prediction = model(batch).float()
    target = batch.target_queue_delta.float()
    capacity = batch.capacity.float()
    node_mask = batch.node_mask.bool()

    normalized_residual = ((prediction - target) / capacity.unsqueeze(1)) ** 2
    normalized_zero = (target / capacity.unsqueeze(1)) ** 2
    raw_residual = (prediction - target) ** 2

    baseline_phase = batch.phases.gather(
        1, batch.baseline_action[:, None, None].expand(-1, 1, batch.max_nodes)
    ).squeeze(1)
    candidate_phase = batch.phases.gather(
        1, batch.candidate_action[:, None, None].expand(-1, 1, batch.max_nodes)
    ).squeeze(1)
    served = ((baseline_phase + candidate_phase) > 0.5) & node_mask
    graph_only = node_mask & ~served

    role_masks = {"all": node_mask, "served": served, "graph_only": graph_only}
    horizon_count = len(HORIZONS)
    batch_size = batch.batch_size
    output = {
        "ss_resid": np.zeros((batch_size, horizon_count, len(ROLES)), dtype=np.float64),
        "ss_zero": np.zeros((batch_size, horizon_count, len(ROLES)), dtype=np.float64),
        "raw_ss": np.zeros((batch_size, horizon_count, len(ROLES)), dtype=np.float64),
        "node_count": np.zeros((batch_size, horizon_count, len(ROLES)), dtype=np.float64),
    }
    for role_index, role in enumerate(ROLES):
        mask = role_masks[role].unsqueeze(1).to(prediction.dtype)
        output["ss_resid"][:, :, role_index] = (
            (normalized_residual * mask).sum(dim=2).double().cpu().numpy()
        )
        output["ss_zero"][:, :, role_index] = (
            (normalized_zero * mask).sum(dim=2).double().cpu().numpy()
        )
        output["raw_ss"][:, :, role_index] = (
            (raw_residual * mask).sum(dim=2).double().cpu().numpy()
        )
        output["node_count"][:, :, role_index] = (
            mask.expand(-1, horizon_count, -1).sum(dim=2).double().cpu().numpy()
        )
    output["graph_seed"] = batch.graph_seed.cpu().numpy()
    output["scenario_seed"] = batch.scenario_seed.cpu().numpy()
    return output


def evaluate_all_controls(
    model: GraphPairPredictor, samples, device: torch.device, *, seed: int
) -> dict[str, dict[str, np.ndarray]]:
    records: dict[str, dict[str, np.ndarray]] = {}
    for name in CONTROL_NAMES:
        controlled = samples if name == "intact" else apply_control(samples, name, seed=seed)
        records[name] = per_sample_residuals(model, controlled, device)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scaled E+ positive control run.")
    parser.add_argument("--clusters", type=int, default=90)
    parser.add_argument("--samples-per-graph", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--tag", type=str, default="scaled")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(__file__).resolve().parent / "runs" / args.tag
    output.mkdir(parents=True, exist_ok=True)
    device = require_cuda_device("cuda")
    sizes, seeds = build_cluster_axes(args.clusters)

    base_kwargs = dict(
        graph_sizes=sizes,
        graph_seeds=seeds,
        samples_per_graph=args.samples_per_graph,
        corruption_modes=("mcar", "burst", "spatial", "mnar"),
        generation_seed=20260817,
        split_seed=8807,
        parameter_targets=(60_000, 100_000),
        depths=(1, 2),
        temporal_depths=(1,),
        learning_rates=(1.0e-3,),
        weight_decays=(0.0,),
        max_trials=12,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
    )

    generation_start = time.perf_counter()
    reference_config = GraphSearchConfig(search_seed=19001, **base_kwargs)
    dataset = generate_scm_dataset(
        graph_sizes=reference_config.graph_sizes,
        graph_seeds=reference_config.graph_seeds,
        samples_per_graph=reference_config.samples_per_graph,
        corruption_modes=reference_config.corruption_modes,
        seed=reference_config.generation_seed,
    )
    split = deterministic_grouped_split(dataset.samples, seed=reference_config.split_seed)
    generation_seconds = time.perf_counter() - generation_start
    print(
        f"dataset: {len(dataset)} samples / {args.clusters} clusters "
        f"in {generation_seconds:.1f}s; split={split.manifest()['sample_counts']}"
    )

    replicate_seeds = [19001, 24007, 31013, 40009, 51001][: args.replicates]
    manifest: dict[str, object] = {
        "schema_version": "eplus-scaled-graph-search-1",
        "purpose": "positive control: synthetic SCM where graph causality is guaranteed",
        "generation_seconds": generation_seconds,
        "split": split.manifest(),
        "replicate_search_seeds": replicate_seeds,
        "base_config": asdict(reference_config),
        "roles": list(ROLES),
        "horizons": list(HORIZONS),
        "controls": list(CONTROL_NAMES),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
        },
    }

    trial_rows: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    total_start = time.perf_counter()

    for replicate_index, search_seed in enumerate(replicate_seeds):
        config = GraphSearchConfig(search_seed=search_seed, **base_kwargs)
        catalog = select_bounded_catalog(
            enumerate_graph_catalog(config),
            max_trials=config.max_trials,
            seed=config.search_seed,
        )
        for trial in catalog:
            trial_start = time.perf_counter()
            record, state = _train_trial(trial, split, config, device)
            model = GraphPairPredictor(GraphModelConfig(**record["model"])).to(device)
            model.load_state_dict(state)
            control_seed = _trial_seed(config.search_seed, trial.trial_id) ^ 0x2468ACE0
            key_prefix = f"r{replicate_index}_{trial.trial_id}"
            for role_split, samples in (("audit", split.audit), ("development", split.development)):
                evaluated = evaluate_all_controls(model, samples, device, seed=control_seed)
                for control, payload in evaluated.items():
                    for name, value in payload.items():
                        arrays[f"{key_prefix}|{role_split}|{control}|{name}"] = value
            elapsed = time.perf_counter() - trial_start
            trial_rows.append(
                {
                    "key_prefix": key_prefix,
                    "replicate_index": replicate_index,
                    "search_seed": search_seed,
                    "trial_id": trial.trial_id,
                    "comparison_group": trial.comparison_group,
                    "family": record["model"]["family"],
                    "width": record["model"]["width"],
                    "depth": record["model"]["depth"],
                    "parameter_target": record["parameter_target"],
                    "parameter_count": record["parameter_count"],
                    "parameter_relative_error": record["parameter_relative_error"],
                    "learning_rate": trial.learning_rate,
                    "weight_decay": trial.weight_decay,
                    "best_epoch": record["best_epoch"],
                    "epochs_completed": record["epochs_completed"],
                    "development_intact_nmse": record["development_controls"]["intact"][
                        "metrics"
                    ]["capacity_normalized_mse"],
                    "structural_contract": record["structural_contract"],
                    "permutation_equivariance_max_abs": record[
                        "permutation_equivariance_max_abs"
                    ],
                    "latency_milliseconds_p95": record["latency"]["milliseconds_p95"],
                    "train_seconds": elapsed,
                }
            )
            print(
                f"[r{replicate_index}] {record['model']['family']:15s} "
                f"w={record['model']['width']:3d} d={record['model']['depth']} "
                f"P={record['parameter_count']:6d} "
                f"ep={record['best_epoch']}/{record['epochs_completed']} "
                f"dev_nmse={trial_rows[-1]['development_intact_nmse']:.6f} "
                f"({elapsed:.1f}s)",
                flush=True,
            )

    manifest["total_seconds"] = time.perf_counter() - total_start
    manifest["trials"] = trial_rows
    (output / "SCALED_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(output / "per_sample_residuals.npz", **arrays)
    print(f"done in {manifest['total_seconds']:.1f}s -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
