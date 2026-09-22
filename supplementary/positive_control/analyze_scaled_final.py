"""E+ final analysis: does a graph architecture detect graph causality when it is guaranteed?

The synthetic SCM is a positive control. `GENERATOR_GRAPH_DEPENDENCE.json`
establishes that its labels genuinely depend on topology: shuffling adjacency
moves the target by 85% of its energy. A graph model that cannot beat a
capacity-matched graph-blind control here, or whose predictions do not move when
the adjacency is shuffled, is not measuring topology at all -- which would mean
the identically-zero topology-shuffle degradation reported on real traffic data
in V7 says nothing about traffic.

This reads the scaled run (120 graph clusters, 576 audit samples, 60 epochs,
3 replicates) and computes two contrasts with intervals that resample whole
graph seeds:

  graph gain        (MSE_edge_null - MSE_graph) / MSE_edge_null
  control response  (MSE_control  - MSE_intact) / MSE_intact

Read-only; writes only into this directory.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE / "runs" / "scaled"
OUT = Path(
    os.environ.get("EPLUS_OUTPUT", HERE / "runs" / "EPLUS_FINAL_RERUN.json")
).resolve()
SEED = 20260805
REPLICATES = 2000


def load() -> tuple[dict, np.lib.npyio.NpzFile]:
    manifest = json.loads((RUN / "SCALED_MANIFEST.json").read_text(encoding="utf-8"))
    return manifest, np.load(RUN / "per_sample_residuals.npz", allow_pickle=True)


def pooled_nmse(z, prefix: str, control: str, idx: np.ndarray | None = None) -> float:
    resid = z[f"{prefix}|audit|{control}|ss_resid"]
    zero = z[f"{prefix}|audit|{control}|ss_zero"]
    if idx is not None:
        resid, zero = resid[idx], zero[idx]
    return float(resid.sum() / zero.sum())


def cluster_boot(fn, clusters: np.ndarray, replicates: int = REPLICATES) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    uniq = np.unique(clusters)
    index_of = {c: np.flatnonzero(clusters == c) for c in uniq}
    draws = []
    for _ in range(replicates):
        picked = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([index_of[c] for c in picked])
        draws.append(fn(idx))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    manifest, z = load()
    trials = manifest["trials"]

    by_group: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for t in trials:
        by_group[t["comparison_group"]][t["family"]].append(t)

    any_prefix = trials[0]["key_prefix"]
    clusters = z[f"{any_prefix}|audit|intact|graph_seed"]
    n_clusters = int(np.unique(clusters).size)
    n_samples = int(clusters.size)
    print(f"audit: {n_samples} samples across {n_clusters} graph clusters")
    print(f"{len(trials)} trials in {len(by_group)} capacity-matched groups\n")

    # ---- contrast 1: graph families against the graph-blind control ---------
    gains = []
    for group, families in sorted(by_group.items()):
        if "edge_null" not in families:
            continue
        null_trials = families["edge_null"]
        for family in ("directed_mpnn", "edge_attention"):
            if family not in families:
                continue
            for gt, nt in zip(families[family], null_trials):
                gp, np_ = gt["key_prefix"], nt["key_prefix"]

                def gain(idx: np.ndarray) -> float:
                    graph = pooled_nmse(z, gp, "intact", idx)
                    blind = pooled_nmse(z, np_, "intact", idx)
                    return (blind - graph) / blind

                point = gain(np.arange(n_samples))
                lo, hi = cluster_boot(gain, clusters)
                gains.append(
                    {
                        "comparison_group": group,
                        "family": family,
                        "replicate": gt["replicate_index"],
                        "graph_params": gt["parameter_count"],
                        "blind_params": nt["parameter_count"],
                        "graph_gain": point,
                        "ci95": [lo, hi],
                    }
                )

    print("graph gain over the capacity-matched graph-blind control")
    print("  (positive = the graph family predicts better)")
    for family in ("directed_mpnn", "edge_attention"):
        rows = [g for g in gains if g["family"] == family]
        if not rows:
            continue
        pts = np.array([r["graph_gain"] for r in rows])
        cross = sum(1 for r in rows if r["ci95"][0] > 0)
        print(
            f"  {family:<16} n={len(rows):2d}  mean={100*pts.mean():+6.2f}%  "
            f"range=[{100*pts.min():+6.2f}%, {100*pts.max():+6.2f}%]  "
            f"CI excludes 0 in {cross}/{len(rows)}"
        )

    # ---- contrast 2: does the model respond to breaking the topology? ------
    print("\ncontrol response: relative MSE increase when the input is corrupted")
    print("  (a model that uses topology must degrade under adjacency_shuffle)")
    responses = []
    for family in ("edge_null", "directed_mpnn", "edge_attention"):
        family_trials = [t for t in trials if t["family"] == family]
        if not family_trials:
            continue
        row = {"family": family, "n_trials": len(family_trials)}
        for control in ("adjacency_shuffle", "no_edge", "action_shuffle"):
            vals = []
            for t in family_trials:
                p = t["key_prefix"]
                intact = pooled_nmse(z, p, "intact")
                corrupted = pooled_nmse(z, p, control)
                vals.append((corrupted - intact) / intact)
            arr = np.array(vals)
            row[control] = {"mean": float(arr.mean()), "max": float(arr.max())}
        responses.append(row)
        print(
            f"  {family:<16} adjacency={100*row['adjacency_shuffle']['mean']:+8.3f}%  "
            f"no_edge={100*row['no_edge']['mean']:+8.3f}%  "
            f"action={100*row['action_shuffle']['mean']:+9.2f}%"
        )

    generator = json.loads((HERE / "runs" / "GENERATOR_GRAPH_DEPENDENCE.json").read_text(encoding="utf-8"))

    report = {
        "question": "On data where graph causality is guaranteed, do graph architectures detect it?",
        "audit_samples": n_samples,
        "audit_graph_clusters": n_clusters,
        "trials": len(trials),
        "capacity_matched_groups": len(by_group),
        "generator_graph_dependence": {
            "adjacency_shuffle_target_shift": generator["variants"]["adjacency_shuffle"][
                "shift_relative_to_target"
            ],
            "no_edge_target_shift": generator["variants"]["no_edge"]["shift_relative_to_target"],
            "reading": "the label genuinely depends on topology; the positive control is real",
        },
        "graph_gain_vs_blind": gains,
        "control_response": responses,
        "cluster_unit": "graph_seed",
        "bootstrap_replicates": REPLICATES,
        "evidence_class": "synthetic_positive_control_only; not traffic evidence",
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
