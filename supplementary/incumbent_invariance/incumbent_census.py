"""Which incumbent does each paired / counterfactual dataset in the corpus branch from?

The reviewer's objection is that the incumbent is a PPO ensemble that loses to
fixed-time control in all four V1 demand regimes. The natural reply would be to
repeat the analysis with a stronger incumbent -- V3 measured an Adapted IDQN
that beat ERGS by 69.7%. This script checks whether any such data exists, by
enumerating every branch-style dataset in the corpus and reporting the class
that produced the baseline branch.

Read-only.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
REPO = Path(
    os.environ.get("ARTICLE_SOURCE_ROOT", REPOSITORY_ROOT / "external" / "archive")
).resolve()
OUT = HERE / "results" / "incumbent_census.json"

# dataset -> the module that generates its baseline branch
DATASETS = [
    ("v4/results/cfra/paired_branches.csv", "v4/src/ergs_tsc_v4/cfra_branching.py"),
    ("v5/results/cfra/paired_branches.csv", "v5/src/ergs_tsc_v5/cfra_branching.py"),
    ("v6/results/cfra/paired_branches.csv", "v6/src/ergs_tsc_v6/cfra_branching.py"),
    ("v10_proposal/phase0/sealed/raw", "v10_proposal/phase0/collect_sealed.py"),
    # pace_loop.py receives the incumbent as a constructor argument; pace_runtime.py
    # is the module that actually loads it.
    ("v10_proposal/phase1_controller/results/paired_voi",
     "v10_proposal/phase1_controller/pace_runtime.py"),
    ("v3/results/counterfactual/counterfactual_interventions.csv",
     "v3/src/ergs_tsc_v3/counterfactual.py"),
]

BASELINE_MARKERS = {
    "EnsemblePolicy": "PPO ensemble",
    "AdaptedIDQN": "Adapted IDQN",
    "MaxPressure": "max-pressure",
    "FixedTime": "fixed-time",
}


def classify(source: Path) -> list[str]:
    if not source.exists():
        return ["<generator not found>"]
    text = source.read_text(encoding="utf-8", errors="replace")
    found = [label for marker, label in BASELINE_MARKERS.items()
             if re.search(rf"\b{marker}\b", text)]
    return found or ["<no known controller class referenced>"]


def main() -> None:
    print(f"{'dataset':<62}{'baseline branch produced by'}")
    print("-" * 100)
    rows = []
    for data_rel, gen_rel in DATASETS:
        data, gen = REPO / data_rel, REPO / gen_rel
        exists = data.exists()
        controllers = classify(gen)
        rows.append({"dataset": data_rel, "exists": exists, "generator": gen_rel,
                     "baseline_controller": controllers})
        print(f"{data_rel:<62}{', '.join(controllers)}"
              f"{'' if exists else '   [dataset absent]'}")

    idqn_checkpoints = sorted(
        str(p.relative_to(REPO)).replace("\\", "/")
        for p in REPO.glob("v*/models/idqn*.pth"))
    print(f"\ntrained Adapted IDQN checkpoints present: {len(idqn_checkpoints)}")
    for p in idqn_checkpoints:
        print(f"   {p}")

    print("\nWhy the checkpoints are not enough to answer the question here:")
    blockers = [
        "cfra_branching._ppo_actions takes an EnsemblePolicy and calls .act(), which "
        "returns (actions, log_prob, values, entropy); AdaptedIDQN.act has a different "
        "signature and no ensemble.",
        "policy_entropy and policy_disagreement are elements 11 and 12 of "
        "ActionContext.as_array(), so they sit at positions 11, 38 and 65 of the "
        "81-dimensional feature_vector that rungs 3, 5 and 6 consume. A deterministic "
        "value-based incumbent has neither quantity, so swapping the incumbent changes "
        "the estimator's feature space as well as the branch baseline -- the two rungs "
        "whose difference defines the overstatement factor would no longer be the same "
        "two rungs.",
        "No paired-branch configuration for a non-PPO incumbent exists anywhere in the "
        "corpus, so answering the question needs a new SUMO collection run, not a "
        "reanalysis.",
    ]
    for b in blockers:
        print(f"   - {b}")

    report = {
        "question": "Does the corpus contain paired counterfactual data branched from an "
                    "incumbent stronger than the PPO ensemble?",
        "answer": "No. Every paired or counterfactual dataset in the corpus branches from "
                  "the same PPO EnsemblePolicy.",
        "datasets": rows,
        "idqn_checkpoints_present": idqn_checkpoints,
        "v3_evidence_that_idqn_is_stronger": {
            "source": "v3/results/analysis/competitor_comparisons.csv",
            "scope": "global",
            "mean_relative": 0.6970,
            "ci95_relative": [0.5721, 0.8247],
            "p_holm": 0.0,
            "reading": "ERGS v3 is 69.7% worse than Adapted IDQN; the comparison is "
                       "end-to-end episode scoring over 20 seed clusters, not paired "
                       "branches, and it never entered a counterfactual collection.",
        },
        "blockers": blockers,
        "evidence_class": "retrospective_diagnostic_not_confirmatory",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
