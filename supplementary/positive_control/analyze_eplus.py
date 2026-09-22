"""E+ step 3: cluster bootstrap over the scaled positive-control residuals.

Every interval is a nonparametric bootstrap over ``graph_seed`` clusters (the
declared dependence unit).  All contrasts reuse the same cluster draw, so
family-vs-control and family-vs-family differences are paired.

Estimator per draw: ratio of sums, ``nmse = sum(ss_resid) / sum(node_count)``,
skill ``= 1 - nmse / nmse_zero``, which reproduces the V8 metric exactly on the
full sample.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROLES = ("all", "served", "graph_only")
HORIZONS = (1, 2, 4)
CONTROLS = ("intact", "adjacency_shuffle", "no_edge", "action_shuffle")
FAMILIES = ("edge_null", "directed_mpnn", "edge_attention")
# action_shuffle permutes the phase table, so the served / graph_only partition
# it induces is not the partition that generated the labels.  Role-resolved
# numbers are therefore only reported for the topology controls.
ROLE_SAFE_CONTROLS = ("intact", "adjacency_shuffle", "no_edge")


def horizon_slice(name: str) -> slice | list[int]:
    if name == "pooled":
        return [0, 1, 2]
    return [HORIZONS.index(int(name[1:]))]


def cluster_sums(
    archive: dict[str, np.ndarray],
    key_prefix: str,
    split: str,
    control: str,
    role_index: int,
    horizons: list[int],
    cluster_index: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate per-sample residual sums into per-cluster sums."""

    base = f"{key_prefix}|{split}|{control}|"
    seeds = archive[base + "graph_seed"]
    take = np.asarray(horizons, dtype=int)
    resid = archive[base + "ss_resid"][:, take, role_index].sum(axis=1)
    zero = archive[base + "ss_zero"][:, take, role_index].sum(axis=1)
    raw = archive[base + "raw_ss"][:, take, role_index].sum(axis=1)
    count = archive[base + "node_count"][:, take, role_index].sum(axis=1)
    n_clusters = len(cluster_index)
    out = [np.zeros(n_clusters, dtype=np.float64) for _ in range(4)]
    positions = np.asarray([cluster_index[int(seed)] for seed in seeds], dtype=int)
    for target, values in zip(out, (resid, zero, raw, count)):
        np.add.at(target, positions, values)
    return tuple(out)  # type: ignore[return-value]


def skill_from_weights(
    resid: np.ndarray, zero: np.ndarray, count: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Skill (1 - nmse/nmse_zero) for each bootstrap draw; weights is (B, C)."""

    numerator = weights @ resid
    denominator = weights @ zero
    return 1.0 - numerator / np.maximum(denominator, 1.0e-30)


def nmse_from_weights(
    resid: np.ndarray, count: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    return (weights @ resid) / np.maximum(weights @ count, 1.0e-30)


def summarize(values: np.ndarray, point: float) -> dict[str, float]:
    lower, upper = np.percentile(values, [2.5, 97.5])
    return {
        "point": float(point),
        "ci_low": float(lower),
        "ci_high": float(upper),
        "bootstrap_mean": float(np.mean(values)),
        "fraction_positive": float(np.mean(values > 0.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Cluster bootstrap for the E+ run.")
    parser.add_argument("--tag", type=str, default="scaled")
    parser.add_argument("--split", type=str, default="audit", choices=("audit", "development"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()

    run_dir = Path(__file__).resolve().parent / "runs" / args.tag
    manifest = json.loads((run_dir / "SCALED_MANIFEST.json").read_text("utf-8"))
    archive = dict(np.load(run_dir / "per_sample_residuals.npz"))
    trials = manifest["trials"]

    seeds_key = f"{args.split}_graph_seeds"
    clusters = sorted(int(value) for value in manifest["split"][seeds_key])
    cluster_index = {seed: position for position, seed in enumerate(clusters)}
    n_clusters = len(clusters)
    rng = np.random.default_rng(args.seed)
    counts = rng.multinomial(
        n_clusters, np.full(n_clusters, 1.0 / n_clusters), size=args.bootstrap
    ).astype(np.float64)
    unit = np.ones((1, n_clusters), dtype=np.float64)

    # ---- exactness check: edge_null must be bit-identical under topology controls
    exactness: list[dict[str, object]] = []
    for trial in trials:
        if trial["family"] != "edge_null":
            continue
        base = f"{trial['key_prefix']}|{args.split}|"
        intact = archive[base + "intact|ss_resid"]
        for control in ("adjacency_shuffle", "no_edge"):
            other = archive[base + control + "|ss_resid"]
            exactness.append(
                {
                    "trial": trial["key_prefix"],
                    "control": control,
                    "bitwise_identical": bool(np.array_equal(intact, other)),
                    "max_abs_difference": float(np.max(np.abs(intact - other))),
                }
            )

    results: dict[str, object] = {
        "split": args.split,
        "n_clusters": n_clusters,
        "bootstrap_draws": args.bootstrap,
        "edge_null_topology_invariance": exactness,
    }

    # ---- per (role, horizon) slice
    slices: list[tuple[str, str]] = [
        (role, horizon)
        for role in ROLES
        for horizon in ("pooled", "H1", "H2", "H4")
    ]
    per_slice: dict[str, object] = {}
    target_energy: dict[str, object] = {}

    for role, horizon_name in slices:
        role_index = ROLES.index(role)
        horizons = list(horizon_slice(horizon_name))
        controls = CONTROLS if role == "all" else ROLE_SAFE_CONTROLS
        cache: dict[tuple[str, str], tuple[np.ndarray, ...]] = {}
        for trial in trials:
            for control in controls:
                cache[(trial["key_prefix"], control)] = cluster_sums(
                    archive,
                    trial["key_prefix"],
                    args.split,
                    control,
                    role_index,
                    horizons,
                    cluster_index,
                )

        # target energy decomposition (label-side only, model independent)
        any_trial = trials[0]["key_prefix"]
        resid0, zero0, raw0, count0 = cache[(any_trial, "intact")]
        target_energy[f"{role}|{horizon_name}"] = {
            "sum_normalized_target_energy": float(zero0.sum()),
            "node_slots": float(count0.sum()),
            "mean_normalized_target_energy_per_node": float(
                zero0.sum() / max(count0.sum(), 1.0)
            ),
        }

        by_family_control: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
        by_group: dict[tuple[int, str, str, str], tuple[np.ndarray, ...]] = {}
        for trial in trials:
            for control in controls:
                sums = cache[(trial["key_prefix"], control)]
                by_family_control[(trial["family"], control)].append(
                    np.asarray(sums, dtype=np.float64)
                )
                by_group[
                    (
                        int(trial["replicate_index"]),
                        str(trial["comparison_group"]),
                        str(trial["family"]),
                        control,
                    )
                ] = sums

        # absolute levels per family x control (mean over the family's trials)
        levels: dict[str, dict[str, object]] = {}
        for family in FAMILIES:
            levels[family] = {}
            for control in controls:
                stacked = by_family_control[(family, control)]
                skills = []
                nmses = []
                rmses = []
                for resid, zero, raw, count in stacked:
                    skills.append(skill_from_weights(resid, zero, count, counts))
                    nmses.append(nmse_from_weights(resid, count, counts))
                    rmses.append(np.sqrt(nmse_from_weights(raw, count, counts)))
                skill_draws = np.mean(skills, axis=0)
                nmse_draws = np.mean(nmses, axis=0)
                rmse_draws = np.mean(rmses, axis=0)
                point_skill = float(
                    np.mean(
                        [
                            skill_from_weights(r, z, c, unit)[0]
                            for r, z, _, c in stacked
                        ]
                    )
                )
                point_nmse = float(
                    np.mean([nmse_from_weights(r, c, unit)[0] for r, _, _, c in stacked])
                )
                point_rmse = float(
                    np.mean(
                        [
                            float(np.sqrt(nmse_from_weights(raw, c, unit)[0]))
                            for _, _, raw, c in stacked
                        ]
                    )
                )
                levels[family][control] = {
                    "skill_vs_zero": summarize(skill_draws, point_skill),
                    "capacity_normalized_mse": summarize(nmse_draws, point_nmse),
                    "movement_rmse": summarize(rmse_draws, point_rmse),
                    "n_trials": len(stacked),
                }

        # paired contrast vs edge_null, matched inside each comparison group
        contrasts: dict[str, object] = {}
        group_keys = sorted({(int(t["replicate_index"]), str(t["comparison_group"])) for t in trials})
        for family in ("directed_mpnn", "edge_attention"):
            per_control: dict[str, object] = {}
            for control in controls:
                deltas = []
                point_deltas = []
                for replicate, group in group_keys:
                    a = by_group.get((replicate, group, family, control))
                    b = by_group.get((replicate, group, "edge_null", control))
                    if a is None or b is None:
                        continue
                    deltas.append(
                        skill_from_weights(a[0], a[1], a[3], counts)
                        - skill_from_weights(b[0], b[1], b[3], counts)
                    )
                    point_deltas.append(
                        float(
                            skill_from_weights(a[0], a[1], a[3], unit)[0]
                            - skill_from_weights(b[0], b[1], b[3], unit)[0]
                        )
                    )
                per_control[control] = summarize(
                    np.mean(deltas, axis=0), float(np.mean(point_deltas))
                ) | {"n_matched_pairs": len(deltas)}
            contrasts[family] = per_control

        # degradation: intact skill minus control skill, paired per trial
        degradation: dict[str, object] = {}
        for family in FAMILIES:
            per_control = {}
            for control in controls:
                if control == "intact":
                    continue
                deltas = []
                point_deltas = []
                for trial in trials:
                    if trial["family"] != family:
                        continue
                    intact_sums = cache[(trial["key_prefix"], "intact")]
                    control_sums = cache[(trial["key_prefix"], control)]
                    deltas.append(
                        skill_from_weights(intact_sums[0], intact_sums[1], intact_sums[3], counts)
                        - skill_from_weights(
                            control_sums[0], control_sums[1], control_sums[3], counts
                        )
                    )
                    point_deltas.append(
                        float(
                            skill_from_weights(intact_sums[0], intact_sums[1], intact_sums[3], unit)[0]
                            - skill_from_weights(
                                control_sums[0], control_sums[1], control_sums[3], unit
                            )[0]
                        )
                    )
                per_control[control] = summarize(
                    np.mean(deltas, axis=0), float(np.mean(point_deltas))
                ) | {"n_trials": len(deltas)}
            degradation[family] = per_control

        per_slice[f"{role}|{horizon_name}"] = {
            "levels": levels,
            "contrast_vs_edge_null": contrasts,
            "degradation_vs_intact": degradation,
        }

    results["target_energy"] = target_energy
    results["slices"] = per_slice

    destination = run_dir / f"BOOTSTRAP_{args.split}.json"
    destination.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # ---- console digest
    print(f"split={args.split}  clusters={n_clusters}  draws={args.bootstrap}")
    bad = [row for row in exactness if not row["bitwise_identical"]]
    print(
        f"edge_null topology invariance: {len(exactness) - len(bad)}/{len(exactness)} "
        f"bitwise identical (max abs diff = "
        f"{max(float(row['max_abs_difference']) for row in exactness):.3e})"
    )
    for key in ("all|pooled", "graph_only|pooled", "served|pooled"):
        block = per_slice[key]
        print(f"\n=== {key} ===")
        for family in FAMILIES:
            entry = block["levels"][family]["intact"]["skill_vs_zero"]
            print(
                f"  {family:15s} intact skill = {entry['point']:+.4f} "
                f"[{entry['ci_low']:+.4f}, {entry['ci_high']:+.4f}]"
            )
        for family in ("directed_mpnn", "edge_attention"):
            entry = block["contrast_vs_edge_null"][family]["intact"]
            print(
                f"  {family:15s} - edge_null  = {entry['point']*100:+.3f} pp "
                f"[{entry['ci_low']*100:+.3f}, {entry['ci_high']*100:+.3f}] "
                f"P(>0)={entry['fraction_positive']:.4f}"
            )
        for family in FAMILIES:
            for control in ("adjacency_shuffle", "no_edge"):
                entry = block["degradation_vs_intact"][family][control]
                print(
                    f"  degrade {family:15s} {control:18s} = {entry['point']*100:+.3f} pp "
                    f"[{entry['ci_low']*100:+.3f}, {entry['ci_high']*100:+.3f}]"
                )
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
