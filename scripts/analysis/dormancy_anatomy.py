"""Phase 0 / Analysis C: why the supervisory channel stayed silent.

Across V1-V4 the guarded supervisor almost never changed the incumbent action:
V1 accepted 0.056% of decision opportunities, V2 accepted 0.000%, V3 accepted 70
out of 187,200 decisions, V4 permitted 133 out of 280,800.  Every published
report noted the rate; none decomposed it.  The per-decision evidence needed for
that decomposition was written to disk and never opened:

    v3/results/raw/*_traces.json                     ~427k proposal evaluations
    v4/results/cfra/policy/internal_test/*_traces.json ~81k authority decisions
    results/raw/confirmatory/*/decisions_*.json        ~360k V1 decisions

This script streams those files and answers one question: of all the places the
supervisor could have acted and did not, what stopped it?  The answer separates
"the proposer agreed with the incumbent, so no authority decision was ever
required" from "the proposer disagreed and the authority refused" -- a
distinction that determines whether a non-inferiority result carries any
information at all.

Read-only with respect to the repository.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from integrity import pin_manifest, write_json_atomic

CURATED_ROOT = Path(__file__).resolve().parents[2]
REPO = Path(os.environ.get("ARTICLE_SOURCE_ROOT", CURATED_ROOT)).resolve()
OUT = CURATED_ROOT / "artifacts" / "derived" / "phase0" / "dormancy_anatomy.json"

# Each archive directory is pinned by a manifest digest over names and bytes, so
# a changed, added or removed trace file stops the run rather than silently
# shifting a denominator.
CORPORA = {
    "v3_traces": (
        REPO / "v3" / "results" / "raw", "*_traces.json", 72,
        "f88a69bff6d57149098a135da683b2028dbfd537cc2e591550366955d30d5e8a",
    ),
    "v4_policy_traces": (
        REPO / "v4" / "results" / "cfra" / "policy" / "internal_test", "*_traces.json", 72,
        "a1a05cb0672da5891a3c00eee50e0804940554491b1ccfc621168d238b018223",
    ),
    "v1_decisions": (
        REPO / "results" / "raw" / "confirmatory", "*/decisions_*.json", 100,
        "786c1ffeea926d6da38b89bd828854497759bd755d1e5e078adbaddd3e023a9a",
    ),
}


def pin_all() -> dict[str, str]:
    pinned = {}
    for label, (root, pattern, count, expected) in CORPORA.items():
        pinned[label] = pin_manifest(
            root.glob(pattern), expected, label=label, count=count
        )
    return pinned


def analyse_v3() -> dict:
    files = sorted((REPO / "v3" / "results" / "raw").glob("*_traces.json"))
    if not files:
        return {"error": "no V3 trace files found"}

    by_reason: Counter[str] = Counter()
    by_method_reason: dict[str, Counter[str]] = defaultdict(Counter)
    by_condition_reason: dict[str, Counter[str]] = defaultdict(Counter)
    total = 0
    disagreed = 0
    accepted = 0
    gains_when_disagreed: list[float] = []

    for path in files:
        for rec in json.loads(path.read_text(encoding="utf-8")):
            total += 1
            reason = rec.get("shield_block_reason") or "accepted"
            by_reason[reason] += 1
            by_method_reason[rec["method"]][reason] += 1
            by_condition_reason[rec["condition"]][reason] += 1
            if rec["baseline_action"] != rec["candidate_action"]:
                disagreed += 1
                value = rec.get("calibrated_lower_gain")
                if value is not None:
                    gains_when_disagreed.append(float(value))
            if rec.get("accepted"):
                accepted += 1

    gains = np.asarray(gains_when_disagreed, dtype=np.float64)
    no_disagreement = by_reason.get("no_disagreement", 0)

    return {
        "files": len(files),
        "proposal_evaluations": total,
        "proposer_disagreed_with_incumbent": disagreed,
        "disagreement_rate": disagreed / total if total else None,
        "accepted": accepted,
        "acceptance_rate_of_evaluations": accepted / total if total else None,
        "acceptance_rate_of_disagreements": accepted / disagreed if disagreed else None,
        "block_reasons": dict(by_reason.most_common()),
        "share_blocked_before_any_authority_decision": (
            no_disagreement / total if total else None
        ),
        "calibrated_lower_gain_when_disagreed": {
            "n": int(gains.size),
            "mean": float(gains.mean()) if gains.size else None,
            "median": float(np.median(gains)) if gains.size else None,
            "p90": float(np.percentile(gains, 90)) if gains.size else None,
            "max": float(gains.max()) if gains.size else None,
            "fraction_strictly_positive": (
                float(np.mean(gains > 0)) if gains.size else None
            ),
        },
        "block_reasons_by_method": {
            m: dict(c.most_common()) for m, c in sorted(by_method_reason.items())
        },
        "block_reasons_by_condition": {
            k: dict(c.most_common()) for k, c in sorted(by_condition_reason.items())
        },
    }


def analyse_v4() -> dict:
    root = REPO / "v4" / "results" / "cfra" / "policy" / "internal_test"
    files = sorted(root.glob("*_traces.json"))
    if not files:
        return {"error": "no V4 trace files found"}

    reasons: Counter[str] = Counter()
    total = 0
    queried = 0
    accepted = 0
    voi_values: list[float] = []
    gain_lcb_when_queried: list[float] = []

    for path in files:
        for rec in json.loads(path.read_text(encoding="utf-8")):
            total += 1
            if rec.get("queried"):
                queried += 1
                if rec.get("gain_lcb") is not None:
                    gain_lcb_when_queried.append(float(rec["gain_lcb"]))
            if rec.get("accepted"):
                accepted += 1
            reasons[rec.get("reason") or "accepted"] += 1
            if rec.get("voi") is not None:
                voi_values.append(float(rec["voi"]))

    lcb = np.asarray(gain_lcb_when_queried, dtype=np.float64)
    voi = np.asarray(voi_values, dtype=np.float64)
    return {
        "files": len(files),
        "authority_decisions": total,
        "queried": queried,
        "query_rate": queried / total if total else None,
        "accepted": accepted,
        "acceptance_rate_of_decisions": accepted / total if total else None,
        "acceptance_rate_of_queries": accepted / queried if queried else None,
        "refusal_reasons": dict(reasons.most_common()),
        "gain_lcb_when_queried": {
            "n": int(lcb.size),
            "mean": float(lcb.mean()) if lcb.size else None,
            "median": float(np.median(lcb)) if lcb.size else None,
            "fraction_strictly_positive": float(np.mean(lcb > 0)) if lcb.size else None,
        },
        "value_of_information": {
            "n": int(voi.size),
            "mean": float(voi.mean()) if voi.size else None,
            "median": float(np.median(voi)) if voi.size else None,
            "fraction_zero": float(np.mean(voi == 0)) if voi.size else None,
        },
    }


def analyse_v1() -> dict:
    root = REPO / "results" / "raw" / "confirmatory"
    files = sorted(root.glob("*/decisions_*.json"))
    if not files:
        return {"error": "no V1 decision files found"}

    per_method: dict[str, Counter[str]] = defaultdict(Counter)
    reason_text: dict[str, Counter[str]] = defaultdict(Counter)
    total = 0

    for path in files:
        method = path.stem.split("_")[-1]
        for rec in json.loads(path.read_text(encoding="utf-8")):
            total += 1
            counter = per_method[method]
            counter["decisions"] += 1
            if rec.get("triggered"):
                counter["triggered"] += 1
            if rec.get("baseline_action") != rec.get("action"):
                counter["proposed_different"] += 1
            if rec.get("accepted"):
                counter["accepted"] += 1
            if rec.get("fallback"):
                counter["fallback"] += 1
            text = (rec.get("reason") or "").strip()
            if text and rec.get("triggered"):
                reason_text[method][text[:110]] += 1

    summary = {}
    for method, counter in sorted(per_method.items()):
        decisions = counter["decisions"]
        summary[method] = {
            "decisions": decisions,
            "triggered": counter["triggered"],
            "trigger_rate": counter["triggered"] / decisions if decisions else None,
            "proposed_different_from_incumbent": counter["proposed_different"],
            "accepted": counter["accepted"],
            "acceptance_rate": counter["accepted"] / decisions if decisions else None,
            "authority_activity_ratio": counter["accepted"] / decisions if decisions else None,
            "acceptance_rate_of_triggers": (
                counter["accepted"] / counter["triggered"] if counter["triggered"] else None
            ),
        }

    return {
        "files": len(files),
        "decision_records": total,
        "by_method": summary,
        "most_common_llm_rationales": {
            m: dict(c.most_common(8)) for m, c in sorted(reason_text.items())
        },
    }


def main() -> None:
    report: dict = {
        "evidence_class": "retrospective_diagnostic_not_confirmatory",
        "input_manifest_sha256": pin_all(),
    }

    print("V3 proposal-evaluation traces ...")
    report["v3"] = analyse_v3()
    v3 = report["v3"]
    if "error" not in v3:
        print(f"  evaluations={v3['proposal_evaluations']:,}  "
              f"disagreements={v3['proposer_disagreed_with_incumbent']:,} "
              f"({100*v3['disagreement_rate']:.2f}%)  accepted={v3['accepted']:,}")
        print(f"  blocked before any authority decision (no_disagreement): "
              f"{100*v3['share_blocked_before_any_authority_decision']:.2f}%")
        for reason, count in list(v3["block_reasons"].items())[:8]:
            print(f"    {reason:<34} {count:>9,} ({100*count/v3['proposal_evaluations']:5.2f}%)")

    print("\nV4 authority-decision traces ...")
    report["v4"] = analyse_v4()
    v4 = report["v4"]
    if "error" not in v4:
        print(f"  decisions={v4['authority_decisions']:,}  queried={v4['queried']:,}  "
              f"accepted={v4['accepted']:,}")
        for reason, count in list(v4["refusal_reasons"].items())[:8]:
            print(f"    {reason:<34} {count:>9,}")

    print("\nV1 per-decision records ...")
    report["v1"] = analyse_v1()
    v1 = report["v1"]
    if "error" not in v1:
        print(f"  decision records={v1['decision_records']:,}")
        for method, row in v1["by_method"].items():
            print(f"    {method:<22} trigger={100*row['trigger_rate']:6.3f}%  "
                  f"accepted={100*row['acceptance_rate']:6.3f}%  (AAR)")

    write_json_atomic(OUT, report)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
