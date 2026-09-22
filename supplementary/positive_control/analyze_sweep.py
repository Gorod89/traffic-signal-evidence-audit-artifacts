"""E+ step 5: read the training-size sweep with the same cluster bootstrap."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_eplus import ROLES, cluster_sums, skill_from_weights, summarize

FAMILIES = ("edge_null", "directed_mpnn", "edge_attention")


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap the E+ training-size sweep.")
    parser.add_argument("--tag", type=str, default="sweep")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()

    run_dir = Path(__file__).resolve().parent / "runs" / args.tag
    manifest = json.loads((run_dir / "SWEEP_MANIFEST.json").read_text("utf-8"))
    archive = dict(np.load(run_dir / "per_sample_residuals.npz"))
    trials = manifest["trials"]

    clusters = sorted(int(value) for value in manifest["audit_graph_seeds"])
    cluster_index = {seed: position for position, seed in enumerate(clusters)}
    n_clusters = len(clusters)
    rng = np.random.default_rng(args.seed)
    counts = rng.multinomial(
        n_clusters, np.full(n_clusters, 1.0 / n_clusters), size=args.bootstrap
    ).astype(np.float64)
    unit = np.ones((1, n_clusters), dtype=np.float64)
    role_index = ROLES.index("all")
    horizons = [0, 1, 2]

    by_size: dict[int, dict[tuple[str, str, int], tuple[np.ndarray, ...]]] = defaultdict(dict)
    size_labels: dict[int, dict[str, object]] = {}
    for trial in trials:
        size = int(trial["train_samples"])
        size_labels[size] = {
            "train_samples": size,
            "train_clusters": trial["train_clusters"],
            "scenarios_per_cluster": trial["scenarios_per_cluster"],
            "epochs_budget": trial["epochs_budget"],
        }
        for control in ("intact", "adjacency_shuffle", "no_edge", "action_shuffle"):
            by_size[size][(trial["family"], control, int(trial["replicate_index"]))] = (
                cluster_sums(
                    archive,
                    trial["key_prefix"],
                    "audit",
                    control,
                    role_index,
                    horizons,
                    cluster_index,
                )
            )

    results: dict[str, object] = {
        "audit_clusters": n_clusters,
        "bootstrap_draws": args.bootstrap,
        "sizes": {},
    }
    replicates = sorted({int(trial["replicate_index"]) for trial in trials})

    print(f"audit clusters={n_clusters}  draws={args.bootstrap}\n")
    header = (
        f"{'train_n':>8} {'family':<15} {'intact skill':>22} "
        f"{'vs edge_null (pp)':>26} {'adj_shuffle degr (pp)':>26} {'no_edge degr (pp)':>26}"
    )
    print(header)
    print("-" * len(header))

    for size in sorted(by_size):
        entry: dict[str, object] = dict(size_labels[size])
        table = by_size[size]
        for family in FAMILIES:
            intact_draws = []
            intact_points = []
            contrast_draws = []
            contrast_points = []
            degradation: dict[str, list] = {"adjacency_shuffle": [], "no_edge": [], "action_shuffle": []}
            degradation_points: dict[str, list] = {k: [] for k in degradation}
            for replicate in replicates:
                a = table[(family, "intact", replicate)]
                intact_draws.append(skill_from_weights(a[0], a[1], a[3], counts))
                intact_points.append(float(skill_from_weights(a[0], a[1], a[3], unit)[0]))
                if family != "edge_null":
                    b = table[("edge_null", "intact", replicate)]
                    contrast_draws.append(
                        skill_from_weights(a[0], a[1], a[3], counts)
                        - skill_from_weights(b[0], b[1], b[3], counts)
                    )
                    contrast_points.append(
                        float(
                            skill_from_weights(a[0], a[1], a[3], unit)[0]
                            - skill_from_weights(b[0], b[1], b[3], unit)[0]
                        )
                    )
                for control in degradation:
                    c = table[(family, control, replicate)]
                    degradation[control].append(
                        skill_from_weights(a[0], a[1], a[3], counts)
                        - skill_from_weights(c[0], c[1], c[3], counts)
                    )
                    degradation_points[control].append(
                        float(
                            skill_from_weights(a[0], a[1], a[3], unit)[0]
                            - skill_from_weights(c[0], c[1], c[3], unit)[0]
                        )
                    )
            block: dict[str, object] = {
                "intact_skill": summarize(
                    np.mean(intact_draws, axis=0), float(np.mean(intact_points))
                )
            }
            if contrast_draws:
                block["skill_minus_edge_null"] = summarize(
                    np.mean(contrast_draws, axis=0), float(np.mean(contrast_points))
                )
            for control in degradation:
                block[f"degradation_{control}"] = summarize(
                    np.mean(degradation[control], axis=0),
                    float(np.mean(degradation_points[control])),
                )
            entry[family] = block

            contrast_text = "        (reference)        "
            if contrast_draws:
                value = block["skill_minus_edge_null"]
                contrast_text = (
                    f"{value['point']*100:+7.3f} [{value['ci_low']*100:+6.3f},{value['ci_high']*100:+6.3f}]"
                )
            adjacency = block["degradation_adjacency_shuffle"]
            no_edge = block["degradation_no_edge"]
            intact = block["intact_skill"]
            print(
                f"{size:>8} {family:<15} "
                f"{intact['point']:+7.4f} [{intact['ci_low']:+6.4f},{intact['ci_high']:+6.4f}] "
                f"{contrast_text:>26} "
                f"{adjacency['point']*100:+7.3f} [{adjacency['ci_low']*100:+6.3f},{adjacency['ci_high']*100:+6.3f}] "
                f"{no_edge['point']*100:+7.3f} [{no_edge['ci_low']*100:+6.3f},{no_edge['ci_high']*100:+6.3f}]"
            )
        results["sizes"][str(size)] = entry
        print()

    destination = run_dir / "BOOTSTRAP_sweep.json"
    destination.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
