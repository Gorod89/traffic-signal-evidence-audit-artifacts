"""Verify every numeric claim in the manuscript against a file on disk.

Claims arrive from three places and carry different risk:
  own      - computed by the Phase-0 scripts in this directory
  agent - reported by an inventory subagent and checked against the named file
  archive  - quoted from a V1-V8 result file

The `agent` class preserves the historical provenance label for values reported
by an inventory subagent. It must not be trusted without a check because any
secondary transcription can contain an error. This
script re-reads the primary source for each claim and reports MATCH or MISMATCH.
It fails loudly rather than silently passing when a source file is missing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

CURATED_ROOT = Path(__file__).resolve().parents[2]
RESULTS = CURATED_ROOT / "artifacts" / "derived" / "phase0"
SUPPLEMENTARY = CURATED_ROOT / "supplementary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=("public", "all"),
        default="public",
        help="public checks released artifacts only; all also checks the private V1--V8 archive",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help="root of the private V1--V8 archive (required for --scope all unless ARTICLE_SOURCE_ROOT is set)",
    )
    args = parser.parse_args()
    env_root = os.environ.get("ARTICLE_SOURCE_ROOT")
    if args.scope == "all" and args.source_root is None and not env_root:
        parser.error("--scope all requires --source-root or ARTICLE_SOURCE_ROOT")
    args.source_root = (args.source_root or (Path(env_root) if env_root else CURATED_ROOT)).resolve()
    return args


ARGS = parse_args()
INCLUDE_ARCHIVE = ARGS.scope == "all"
REPO = ARGS.source_root

results: list[tuple[str, str, str, object, object, bool]] = []
EXPECTED_CHECKS = {"public": 98, "all": 141}


def check(claim: str, source: str, origin: str, expected, actual, tol: float = 5e-3) -> None:
    if actual is None:
        ok = False
    elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        denom = max(abs(expected), 1e-9)
        ok = abs(expected - actual) / denom <= tol
    else:
        ok = expected == actual
    results.append((claim, source, origin, expected, actual, ok))


PUBLIC_REQUIRED = (
    RESULTS / "baseline_ladder.json",
    RESULTS / "sample_size_curve_robust.json",
    RESULTS / "sample_size_curve.json",
    RESULTS / "history_increment_ci.json",
    RESULTS / "cost_weight_sensitivity.json",
    RESULTS / "aar_funnel.json",
    RESULTS / "dormancy_anatomy.json",
    RESULTS / "fact2_verification.json",
    RESULTS / "sealed_confirmation.json",
    SUPPLEMENTARY / "positive_control" / "runs" / "EPLUS_FINAL.json",
    SUPPLEMENTARY / "dplus" / "headline_comparison.csv",
    SUPPLEMENTARY / "dplus" / "trials_candidate_level.csv",
    SUPPLEMENTARY / "dplus" / "matched_worst_condition.csv",
)
for required_path in PUBLIC_REQUIRED:
    check(
        f"required public source exists: {required_path.relative_to(CURATED_ROOT)}",
        str(required_path.relative_to(CURATED_ROOT)),
        "own",
        True,
        required_path.is_file(),
    )


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def dig(obj, *keys, default=None):
    """Walk a nested structure by mapping key or sequence index."""
    for k in keys:
        if obj is None:
            return default
        if isinstance(obj, dict):
            obj = obj.get(k)
        elif isinstance(obj, (list, tuple)) and isinstance(k, int) and -len(obj) <= k < len(obj):
            obj = obj[k]
        else:
            return default
    return obj if obj is not None else default


def read_yaml_number(path: Path, keys: tuple):
    """Read one scalar out of a preregistration YAML, or None if absent.

    Registered thresholds live in the protocol files rather than in the result
    artifacts, so the requirement column of the gate table has a different
    primary source from the observed column and must be checked against it.
    """
    if not path.exists():
        return None
    import yaml

    return dig(yaml.safe_load(path.read_text(encoding="utf-8")), *keys)


# ---------------------------------------------------------------- own results
ladder = load_json(RESULTS / "baseline_ladder.json")
if ladder is not None:
    rungs = {r["rung"][3:]: r for r in ladder["ladder"]}
    check("ladder: ridge on current observation = 10.90%",
          "results/baseline_ladder.json", "own",
          0.1090, dig(rungs.get("ridge, current observation"), "rmse_gain"))
    check("ladder: trees on action contrast = 14.32%",
          "results/baseline_ladder.json", "own",
          0.1432, dig(rungs.get("trees, action contrast"), "rmse_gain"))
    check("ladder: trees + history = 15.15%",
          "results/baseline_ladder.json", "own",
          0.1515, dig(rungs.get("trees, contrast + 8-step history"), "rmse_gain"))
    check("history increment = +0.82 pp",
          "results/baseline_ladder.json", "own",
          0.824, ladder.get("history_increment_pp"), tol=0.02)
    check("naive-reference increment = +16.38 pp",
          "results/baseline_ladder.json", "own",
          16.377, ladder.get("zero_reference_increment_pp"), tol=0.02)

# The published curve is the 200-resample one; the 12-resample artifact it
# superseded is retained but no longer backs any claim in the text.
curve = load_json(RESULTS / "sample_size_curve_robust.json")
if curve is not None:
    tc = {row["train_clusters"]: row for row in curve["training_curve"]}
    check("curve: 200 resamples per budget", "results/sample_size_curve_robust.json", "own",
          200, curve.get("repeats"))
    check("curve: median at 10 clusters = 2.15%", "results/sample_size_curve_robust.json", "own",
          0.0215, dig(tc.get(10), "median"), tol=0.02)
    check("curve: median at 20 clusters (V5 budget) = 5.10%",
          "results/sample_size_curve_robust.json", "own",
          0.0510, dig(tc.get(20), "median"), tol=0.02)
    check("curve: median at 30 clusters (V4 budget) = 7.16%",
          "results/sample_size_curve_robust.json", "own",
          0.0716, dig(tc.get(30), "median"), tol=0.02)
    check("curve: median at 160 clusters = 14.64%",
          "results/sample_size_curve_robust.json", "own",
          0.1464, dig(tc.get(160), "median"), tol=0.02)
    check("curve: band at 20 clusters covers V5's observed 8.99%",
          "results/sample_size_curve_robust.json", "own",
          True, bool(dig(tc.get(20), "q05") <= 0.0899 <= dig(tc.get(20), "q95")))
    check("curve: band at 30 clusters covers V4's observed 5.34%",
          "results/sample_size_curve_robust.json", "own",
          True, bool(dig(tc.get(30), "q05") <= 0.0534 <= dig(tc.get(30), "q95")))
    check("curve: pass frequency at 20 clusters = 10.5%",
          "results/sample_size_curve_robust.json", "own",
          0.105, dig(tc.get(20), "share_clearing_gate"), tol=0.02)
    check("curve: pass frequency at 30 clusters = 24.5%",
          "results/sample_size_curve_robust.json", "own",
          0.245, dig(tc.get(30), "share_clearing_gate"), tol=0.02)
    check("curve: pass frequency at 160 clusters = 100%",
          "results/sample_size_curve_robust.json", "own",
          1.0, dig(tc.get(160), "share_clearing_gate"), tol=0.01)

# The evaluation-side half-widths are still quoted, as a hypothetical
# bound-form diagnostic rather than as the registered gate.
legacy = load_json(RESULTS / "sample_size_curve.json")
if legacy is not None:
    ec = {row["eval_clusters"]: row for row in legacy["evaluation_curve"]}
    check("CI halfwidth at 15 eval clusters = 11.23 pp (hypothetical bound diagnostic)",
          "results/sample_size_curve.json", "own",
          0.1123, dig(ec.get(15), "ci_halfwidth_mean"), tol=0.02)
    check("CI halfwidth at 300 eval clusters = 2.74 pp",
          "results/sample_size_curve.json", "own",
          0.0274, dig(ec.get(300), "ci_halfwidth_mean"), tol=0.02)

# The paired interval on the history increment is what replaced the withdrawn
# ratio, so it must be audited.
inc = load_json(RESULTS / "history_increment_ci.json")
if inc is not None:
    check("history increment = +0.82 pp", "results/history_increment_ci.json", "own",
          0.82, inc.get("history_increment_pp"), tol=0.02)
    check("history increment CI covers zero", "results/history_increment_ci.json", "own",
          False, inc.get("ratio_identified"))
    check("naive increment = +16.38 pp", "results/history_increment_ci.json", "own",
          16.38, inc.get("naive_increment_pp"), tol=0.02)

# Historical fitting budgets, read from the V4/V5 corpora themselves.
import pandas as _pd

if INCLUDE_ARCHIVE:
    for version, split, expected in (("v4", "training", 30), ("v5", "representation_train", 20)):
        path = REPO / version / "results" / "cfra" / "paired_branches.csv"
        if path.exists():
            frame = _pd.read_csv(path)
            frame = frame[frame.split == split]
            check(f"{version.upper()} fitted on {expected} clusters",
                  f"{version}/results/cfra/paired_branches.csv", "archive",
                  expected, int(frame.groupby(["network", "seed"]).ngroups))

# Cost-weight sensitivity: the null must not depend on the utility.
cw = load_json(RESULTS / "cost_weight_sensitivity.json")
if cw is not None:
    check("cost weights: component deltas reconstruct the gain",
          "results/cost_weight_sensitivity.json", "own",
          True, bool(cw["identity_residual"] < 1e-6))
    check("cost weights: history increment covers zero in 9 of 10 schemes",
          "results/cost_weight_sensitivity.json", "own",
          9, dig(cw, "summary", "history_increment_covers_zero_in"))
    check("cost weights: schemes tested = 10",
          "results/cost_weight_sensitivity.json", "own",
          10, dig(cw, "summary", "schemes_tested"))

# AAR funnel: the pooled figure and the single-method ratio must reconcile.
fun = load_json(RESULTS / "aar_funnel.json")
if fun is not None:
    full = fun["full_method_funnel"]
    check("funnel: full-method evaluations = 60,079", "results/aar_funnel.json", "own",
          60079, full["evaluations"])
    check("funnel: disagreements = 10,047", "results/aar_funnel.json", "own",
          10047, full["proposer_disagreed"])
    check("funnel: executed deviations = 70", "results/aar_funnel.json", "own",
          70, full["executed_deviations"])
    check("funnel: stages close exactly", "results/aar_funnel.json", "own",
          full["proposer_disagreed"],
          sum(full["refusals_among_disagreements"].values()))
    # The reported estimand is the pooled ratio, so the check is that the
    # published number equals numerator over denominator exactly, not that it
    # equals a remembered constant.
    check("funnel: pooled AAR = 0.117%", "results/aar_funnel.json", "own",
          full["executed_deviations"] / full["evaluations"], full["aar"], tol=1e-9)
    check("funnel: pooled disagreement = 16.72%", "results/aar_funnel.json", "own",
          full["proposer_disagreed"] / full["evaluations"],
          full["disagreement_rate"], tol=1e-9)
    check("funnel: AAR interval brackets the point estimate",
          "results/aar_funnel.json", "own", True,
          bool(full["aar_ci95"][0] <= full["aar"] <= full["aar_ci95"][1]))
    check("funnel: pooled figure accepted = 15,844", "results/aar_funnel.json", "own",
          15844, dig(fun, "pooled_figure", "accepted"))

dorm = load_json(RESULTS / "dormancy_anatomy.json")
if dorm is not None:
    v3 = dorm["v3"]
    check("V3 proposal evaluations = 427,342", "results/dormancy_anatomy.json", "own",
          427342, v3["proposal_evaluations"])
    check("V3 no_disagreement = 342,262", "results/dormancy_anatomy.json", "own",
          342262, v3["block_reasons"].get("no_disagreement"))
    check("V3 share blocked before authority = 80.09%", "results/dormancy_anatomy.json", "own",
          0.8009, v3["share_blocked_before_any_authority_decision"], tol=0.01)
    check("V3 model_uncertainty = 7.90% of evaluations", "results/dormancy_anatomy.json", "own",
          0.0790, v3["block_reasons"]["model_uncertainty"] / v3["proposal_evaluations"], tol=0.01)
    check("V3 calibrated_gain = 3.32% of evaluations", "results/dormancy_anatomy.json", "own",
          0.0332, v3["block_reasons"]["calibrated_gain"] / v3["proposal_evaluations"], tol=0.01)
    check("V3 disagreements with LCB recorded = 59,083", "results/dormancy_anatomy.json", "own",
          59083, dig(v3, "calibrated_lower_gain_when_disagreed", "n"))
    check("V3 LCB positive share = 0.20%", "results/dormancy_anatomy.json", "own",
          0.0020, dig(v3, "calibrated_lower_gain_when_disagreed", "fraction_strictly_positive"), tol=0.05)
    check("V3 LCB median = -1.135", "results/dormancy_anatomy.json", "own",
          -1.135, dig(v3, "calibrated_lower_gain_when_disagreed", "median"), tol=0.01)
    ergs = dig(v3, "block_reasons_by_method", "ergs_v3", default={})
    check("V3 ergs_v3 accepted = 70", "results/dormancy_anatomy.json", "own",
          70, ergs.get("accepted"))
    method_gain = v3["calibrated_lower_gain_by_method"]
    check("V3 positive LCBs in no-calibration ablation = 121",
          "results/dormancy_anatomy.json", "own", 121,
          method_gain["ergs_no_calibration"]["strictly_positive"])
    check("V3 positive LCBs across calibrated variants = 0",
          "results/dormancy_anatomy.json", "own", 0,
          sum(row["strictly_positive"] for method, row in method_gain.items()
              if method != "ergs_no_calibration"))
    accepted_full = v3["full_method_accepted_deviations"]
    check("V3 full-method accepted raw LCB minimum = -0.835390",
          "results/dormancy_anatomy.json", "own", -0.835390,
          accepted_full["calibrated_lower_gain_range"][0], tol=1e-6)
    check("V3 full-method accepted raw LCB maximum = -0.084852",
          "results/dormancy_anatomy.json", "own", -0.084852,
          accepted_full["calibrated_lower_gain_range"][1], tol=1e-5)
    check("V3 full-method accepted HOLD-to-ADVANCE = 70",
          "results/dormancy_anatomy.json", "own", 70,
          accepted_full["directions"].get("0->1"))
    check("V3 full-method accepted emergency events = 70",
          "results/dormancy_anatomy.json", "own", 70,
          accepted_full["event_types"].get("EMERGENCY_VEHICLE"))
    v4 = dorm["v4"]
    check("V4 authority decisions = 81,127", "results/dormancy_anatomy.json", "own",
          81127, v4["authority_decisions"])
    check("V4 not_queried = 80,303", "results/dormancy_anatomy.json", "own",
          80303, v4["refusal_reasons"].get("not_queried"))
    check("V4 support_reject = 438", "results/dormancy_anatomy.json", "own",
          438, v4["refusal_reasons"].get("support_reject"))
    check("V4 guard_reject = 127", "results/dormancy_anatomy.json", "own",
          127, v4["refusal_reasons"].get("guard_reject"))
    v1 = dorm["v1"]
    check("V1 decision records = 360,000", "results/dormancy_anatomy.json", "own",
          360000, v1["decision_records"])
    check("V1 full method AAR = 0.056%", "results/dormancy_anatomy.json", "own",
          0.00056, dig(v1, "by_method", "ergs", "authority_activity_ratio"), tol=0.02)
    check("V1 full method trigger = 0.856%", "results/dormancy_anatomy.json", "own",
          0.00856, dig(v1, "by_method", "ergs", "trigger_rate"), tol=0.02)
    check("V1 ungated AAR = 6.533%", "results/dormancy_anatomy.json", "own",
          0.06533, dig(v1, "by_method", "ungated", "authority_activity_ratio"), tol=0.02)

fact2 = load_json(RESULTS / "fact2_verification.json")
if fact2 is not None:
    zs = dig(fact2, "results", "gain_h4_zero_shot_ingolstadt21", default={})
    check("zero-shot ingolstadt21 gain = 10.82%", "results/fact2_verification.json", "own",
          0.1082, zs.get("rmse_gain"), tol=0.02)
    check("zero-shot AUROC = 0.715", "results/fact2_verification.json", "own",
          0.715, zs.get("auroc"), tol=0.02)
    check("zero-shot: ingolstadt21 absent from training", "results/fact2_verification.json", "own",
          False, zs.get("in_training"))

confirm = load_json(RESULTS / "sealed_confirmation.json")
if confirm is not None:
    check("sealed: observed RMSE gain = 13.99%", "results/sealed_confirmation.json", "own",
          0.1399, dig(confirm, "primary", "observed_rmse_gain"), tol=0.02)
    check("sealed: CI lower = 10.87%", "results/sealed_confirmation.json", "own",
          0.1087, dig(confirm, "primary", "ci95", default=[None])[0], tol=0.02)
    check("sealed: primary prediction met", "results/sealed_confirmation.json", "own",
          True, dig(confirm, "primary", "met"))
    check("sealed: secondary prediction met", "results/sealed_confirmation.json", "own",
          True, dig(confirm, "secondary", "met"))
    check("sealed: analysable pairs = 1800", "results/sealed_confirmation.json", "own",
          1800, dig(confirm, "sealed_data", "analysable"))
    check("sealed: attrition = 0.00%", "results/sealed_confirmation.json", "own",
          0.0, dig(confirm, "sealed_data", "attrition"), tol=1.0)
    check("sealed: independent clusters = 300", "results/sealed_confirmation.json", "own",
          300, dig(confirm, "sealed_data", "clusters"))
    check("sealed: estimator was not refitted", "results/sealed_confirmation.json", "own",
          False, confirm.get("estimator_refitted"))
    check("sealed: benefit AUROC = 0.762", "results/sealed_confirmation.json", "own",
          0.762, dig(confirm, "benefit_auroc", "observed"), tol=0.02)
    check("sealed: zero-shot ingolstadt21 gain = 11.08%", "results/sealed_confirmation.json", "own",
          0.1108, dig(confirm, "per_network", "ingolstadt21", "rmse_gain"), tol=0.02)
    # The prediction must pre-date the data it was tested on; otherwise the
    # confirmatory status is void regardless of the numbers.
    lock_path = Path(__file__).resolve().parent / "sealed" / "PREDICTION_LOCK.json"
    raw_dir = Path(__file__).resolve().parent / "sealed" / "raw"
    if lock_path.exists() and raw_dir.is_dir():
        import datetime as _dt

        written = _dt.datetime.fromisoformat(
            json.loads(lock_path.read_text(encoding="utf-8"))["written_utc"]
        ).timestamp()
        earliest = min(p.stat().st_mtime for p in raw_dir.glob("*.json"))
        check("sealed: prediction written before any sealed datum existed",
              "sealed/PREDICTION_LOCK.json vs sealed/raw/", "own",
              True, bool(written < earliest))

eplus = load_json(SUPPLEMENTARY / "positive_control" / "runs" / "EPLUS_FINAL.json")
if eplus is not None:
    gains = eplus["graph_gain_vs_blind"]
    mpnn = [g["graph_gain"] for g in gains if g["family"] == "directed_mpnn"]
    attn = [g["graph_gain"] for g in gains if g["family"] == "edge_attention"]
    check("E+ directed_mpnn gain = 18.8%", "eplus/runs/EPLUS_FINAL.json", "own",
          0.1877, sum(mpnn) / len(mpnn) if mpnn else None, tol=0.02)
    check("E+ edge_attention gain = 19.5%", "eplus/runs/EPLUS_FINAL.json", "own",
          0.1948, sum(attn) / len(attn) if attn else None, tol=0.02)
    check("E+ mpnn CI excludes zero in 12/12", "eplus/runs/EPLUS_FINAL.json", "own",
          12, sum(1 for g in gains if g["family"] == "directed_mpnn" and g["ci95"][0] > 0))
    resp = {r["family"]: r for r in eplus["control_response"]}
    check("E+ mpnn adjacency response = +71.8%", "eplus/runs/EPLUS_FINAL.json", "own",
          0.718, dig(resp.get("directed_mpnn"), "adjacency_shuffle", "mean"), tol=0.02)
    check("E+ edge_null adjacency response = 0.000%", "eplus/runs/EPLUS_FINAL.json", "own",
          0.0, dig(resp.get("edge_null"), "adjacency_shuffle", "mean"), tol=1.0)
    check("E+ audit samples = 576", "eplus/runs/EPLUS_FINAL.json", "own",
          576, eplus["audit_samples"])
    check("E+ audit clusters = 24", "eplus/runs/EPLUS_FINAL.json", "own",
          24, eplus["audit_graph_clusters"])
    check("E+ generator adjacency shift = 85%", "eplus/runs/EPLUS_FINAL.json", "own",
          0.853, dig(eplus, "generator_graph_dependence", "adjacency_shuffle_target_shift"), tol=0.02)

# --------------------------------------------------------- archive quotations
v4gate = load_json(REPO / "v4" / "results" / "cfra" / "stage_b_verifier.json") if INCLUDE_ARCHIVE else None
if v4gate is not None:
    blob = json.dumps(v4gate)
    m = re.search(r'"(?:rmse_improvement[a-z_]*|improvement_over_mean)"\s*:\s*([0-9.eE-]+)', blob)
    check("V4 Stage-B RMSE gain = 5.34%", "v4/results/cfra/stage_b_verifier.json", "archive",
          0.0534, float(m.group(1)) if m else None, tol=0.02)

v5gate = load_json(REPO / "v5" / "results" / "protocol" / "stage_b_verifier.json") if INCLUDE_ARCHIVE else None
if v5gate is not None:
    blob = json.dumps(v5gate)
    m = re.search(r'"(?:rmse_improvement[a-z_]*|improvement_over_mean)"\s*:\s*([0-9.eE-]+)', blob)
    check("V5 Stage-B RMSE gain = 8.99%", "v5/results/protocol/stage_b_verifier.json", "archive",
          0.0899, float(m.group(1)) if m else None, tol=0.02)

# -------------------------------------------------- every row of the gate table
# The gate table quotes an observed value for each registered criterion.  Each
# is read here from the same field the original go/no-go branch compares against
# its threshold, because the artifacts store several intervals per quantity and
# reading a neighbouring one silently changes the number.  An earlier draft did
# exactly that for V5 triggered coverage.
GATE_SRC_V4 = "v4/results/cfra/stage_b_verifier.json"
GATE_SRC_V5 = "v5/results/protocol/stage_b_verifier.json"

if v4gate is not None:
    for label, expected, observed, tol in (
        ("V4 gate: benefit AUROC = 0.695", 0.695, v4gate.get("benefit_auroc"), 0.002),
        ("V4 gate: marginal coverage LB = 84.61%", 0.8461,
         dig(v4gate, "coverage_ci95", 0), 0.0002),
        ("V4 gate: triggered coverage LB = 88.41%", 0.8841,
         dig(v4gate, "trigger_coverage_ci95", 0), 0.0002),
        ("V4 gate: harmful selection UB = 56.15%", 0.5615,
         dig(v4gate, "harmful_selected_ci95", 1), 0.0002),
        ("V4 gate: extreme-OOD score = 7.1e5", 710863.3,
         v4gate.get("synthetic_extreme_ood_score"), 1.0),
    ):
        check(label, GATE_SRC_V4, "archive", expected, observed, tol=tol)
    check("V4 gate: extreme-OOD threshold = 2.3", "v4/configs/study.yaml", "archive",
          2.3, read_yaml_number(REPO / "v4" / "configs" / "study.yaml",
                                ("cfra", "verifier", "maximum_ood")), tol=1e-9)
    check("V4 gate: terminal decision was STOP", GATE_SRC_V4, "archive",
          False, v4gate.get("go"))

if v5gate is not None:
    full = v5gate.get("ergs_cfra_jepa", {})
    for label, expected, observed, tol in (
        ("V5 gate: benefit AUROC = 0.777", 0.777, full.get("benefit_auroc"), 0.002),
        ("V5 gate: marginal coverage LB = 82.78%", 0.8278,
         dig(full, "marginal_coverage_ci95", 0), 0.0002),
        ("V5 gate: triggered coverage LB = 82.62%", 0.8262,
         dig(full, "trigger_coverage_ci95", 0), 0.0002),
        ("V5 gate: synthetic-OOD rejection = 100%", 1.0,
         full.get("synthetic_ood_rejection_rate"), 1e-9),
    ):
        check(label, GATE_SRC_V5, "archive", expected, observed, tol=tol)
    for label, expected, path in (
        ("V5 gate: RMSE requirement = 10%", 0.1, ("rmse_improvement_over_mean",)),
        ("V5 gate: AUROC requirement = 0.65", 0.65, ("benefit_auroc",)),
        ("V5 gate: coverage requirement = 85%", 0.85, ("marginal_coverage_lower_95",)),
        ("V5 gate: intervention requirement = 30", 30, ("selected_interventions",)),
        ("V5 gate: harm requirement = 10%", 0.1, ("harmful_selection_upper_95",)),
        ("V5 gate: synthetic-OOD requirement = 95%", 0.95, ("synthetic_ood_rejection_rate",)),
    ):
        check(label, "v5/configs/preregistered_protocol.yaml", "archive", expected,
              read_yaml_number(
                  REPO / "v5" / "configs" / "preregistered_protocol.yaml",
                  ("protocol", "gates", "B_verifier", "criteria") + path + ("threshold",),
              ), tol=1e-9)

v6gate = load_json(REPO / "v6" / "results" / "protocol" / "stage_b_representation.json") if INCLUDE_ARCHIVE else None
if v6gate is not None:
    metrics = v6gate.get("metrics", {})
    check("V6.1 gain over persistence = 21.80%", "v6/.../stage_b_representation.json", "archive",
          0.2180, metrics.get("h4_innovation_rmse_improvement_over_persistence"), tol=0.02)
    check("V6.1 gain over no-history = 1.40%", "v6/.../stage_b_representation.json", "archive",
          0.01398, metrics.get("h4_rmse_improvement_over_no_history_ablation"), tol=0.05)

# ------------------------------------------------- corpus sizes and V8 trials
import csv as _csv


def count_csv_rows(path: Path) -> int | None:
    return sum(1 for _ in path.open(encoding="utf-8")) - 1 if path.exists() else None


if INCLUDE_ARCHIVE:
    required_archive_sources = (
        REPO / "v4" / "results" / "cfra" / "paired_branches.csv",
        REPO / "v5" / "results" / "cfra" / "paired_branches.csv",
        REPO / "v4" / "results" / "cfra" / "stage_b_verifier.json",
        REPO / "v5" / "results" / "protocol" / "stage_b_verifier.json",
        REPO / "v4" / "configs" / "study.yaml",
        REPO / "v5" / "configs" / "preregistered_protocol.yaml",
        REPO / "v6" / "results" / "protocol" / "stage_b_representation.json",
        REPO / "results" / "raw" / "confirmatory" / "metrics.json",
        REPO / "v2" / "results" / "analysis" / "all_runs.csv",
        REPO / "v3" / "results" / "analysis" / "all_runs.csv",
        REPO / "v6" / "results" / "cfra" / "paired_branches.csv",
        REPO / "results" / "analysis" / "summary.csv",
    )
    for path in required_archive_sources:
        check(f"required archive source exists: {path.relative_to(REPO)}",
              str(path.relative_to(REPO)), "archive", True, path.exists())
    v8_trials = list((REPO / "v8" / "results").glob("*/trials/*.json"))
    check("required archive source exists: V8 trial files", "v8/results/*/trials/*.json",
          "archive", True, bool(v8_trials))

    v1_metrics = load_json(REPO / "results" / "raw" / "confirmatory" / "metrics.json")
    check("V1 corpus = 160 runs", "results/raw/confirmatory/metrics.json", "archive",
          160, len(v1_metrics) if v1_metrics is not None else None)
    check("V2 corpus = 1600 runs", "v2/results/analysis/all_runs.csv", "archive",
          1600, count_csv_rows(REPO / "v2" / "results" / "analysis" / "all_runs.csv"))
    check("V3 corpus = 4680 runs", "v3/results/analysis/all_runs.csv", "archive",
          4680, count_csv_rows(REPO / "v3" / "results" / "analysis" / "all_runs.csv"))
    check("V8 = 376 architecture-search trials", "v8/results/*/trials/*.json", "archive",
          376, len(v8_trials))

    paired = REPO / "v6" / "results" / "cfra" / "paired_branches.csv"
    check("V6.1 paired records = 3480", "v6/results/cfra/paired_branches.csv", "archive",
          3480, count_csv_rows(paired))

    # V2 authority activity ratio is claimed to be exactly zero for the full method.
    v2_runs = REPO / "v2" / "results" / "analysis" / "all_runs.csv"
    if v2_runs.exists():
        rows = list(_csv.DictReader(v2_runs.open(encoding="utf-8")))
        ergs = [float(r["override_accept_rate"]) for r in rows if r.get("method") == "ergs_v2"]
        check("V2 full-method override rate = exactly 0", "v2/results/analysis/all_runs.csv", "archive",
              0.0, max(ergs) if ergs else None, tol=1.0)

    # V1: the incumbent PPO is beaten by fixed-time in every demand regime.
    v1_summary = REPO / "results" / "analysis" / "summary.csv"
    if v1_summary.exists():
        rows = [r for r in _csv.DictReader(v1_summary.open(encoding="utf-8"))
                if "wait" in r.get("metric", "").lower()]
        beaten = 0
        for scenario in ("low", "medium", "high", "variable"):
            cell = {r["method"]: float(r["mean"]) for r in rows if r.get("scenario") == scenario}
            if cell.get("ppo") and cell.get("fixed") and cell["ppo"] > cell["fixed"]:
                beaten += 1
        check("V1: fixed-time beats PPO in all 4 regimes", "results/analysis/summary.csv", "archive",
              4, beaten)

# ------------------------------------------------------------ D+ (inventory subagent)
dplus = None
for name in ("dplus_summary.json", "headline_comparison.csv"):
    p = SUPPLEMENTARY / "dplus" / name
    if p.exists():
        dplus = p
        break
dplus = SUPPLEMENTARY / "dplus" / "headline_comparison.csv"
if dplus.exists():
    rows = list(_csv.DictReader(dplus.open(encoding="utf-8")))
    def find(sub):
        for r in rows:
            if sub.lower() in json.dumps(r).lower():
                return r
        return None
    best = find("m-e27868141735")
    ridge = find("history_ridge_alpha=100")
    cur = find("current_ridge_alpha=100")
    def num(row, *cands):
        if not row:
            return None
        for c in cands:
            for k, v in row.items():
                if c in k.lower():
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        pass
        return None
    check("D+ best neural overall = 0.17694", "dplus/headline_comparison.csv", "agent",
          0.17694, num(best, "overall"), tol=0.02)
    check("D+ history_ridge overall = 0.20145", "dplus/headline_comparison.csv", "agent",
          0.20145, num(ridge, "overall"), tol=0.02)
    check("D+ current_ridge overall = 0.18057", "dplus/headline_comparison.csv", "agent",
          0.18057, num(cur, "overall"), tol=0.02)

    trials_csv = SUPPLEMENTARY / "dplus" / "trials_candidate_level.csv"
    if trials_csv.exists():
        tr = list(_csv.DictReader(trials_csv.open(encoding="utf-8")))
        h4 = [float(r["h4_improvement"]) for r in tr if r.get("h4_improvement")]
        ov = [float(r["overall_improvement"]) for r in tr if r.get("overall_improvement")]
        check("D+ trials analysed = 94", "dplus/trials_candidate_level.csv", "agent", 94, len(tr))
        check("D+ max h4 across trials = 0.00366", "dplus/trials_candidate_level.csv", "agent",
              0.003661, max(h4) if h4 else None, tol=0.02)
        check("D+ trials beating current_ridge(100) = 0", "dplus/trials_candidate_level.csv", "agent",
              0, sum(1 for x in ov if x > 0.1805747640975042))

    mw = SUPPLEMENTARY / "dplus" / "matched_worst_condition.csv"
    if mw.exists():
        rows_mw = list(_csv.DictReader(mw.open(encoding="utf-8")))
        def row_for(sub):
            return next((r for r in rows_mw if sub in r["system"]), None)
        best_n = row_for("m-49b072af5e16")
        best_b = row_for("current_ridge_alpha=100")
        ridge_h = row_for("history_ridge_alpha=100")
        neural_g = row_for("m-e27868141735")
        check("D+ matched worst-condition, best neural = 0.1265",
              "dplus/matched_worst_condition.csv", "agent",
              0.12653, float(best_n["min_mean_condition"]) if best_n else None, tol=0.02)
        check("D+ matched worst-condition, best ridge = 0.0748",
              "dplus/matched_worst_condition.csv", "agent",
              0.074757, float(best_b["min_mean_condition"]) if best_b else None, tol=0.02)
        check("D+ ridge on degraded sensing = -0.153",
              "dplus/matched_worst_condition.csv", "agent",
              -0.15254, float(ridge_h["cond_sensor_delay_noise"]) if ridge_h else None, tol=0.02)
        check("D+ neural on degraded sensing = +0.111",
              "dplus/matched_worst_condition.csv", "agent",
              0.111366, float(neural_g["cond_sensor_delay_noise"]) if neural_g else None, tol=0.02)

# --------------------------------------------------------------------- report
def main() -> int:
    if not results:
        print("no claims were evaluated")
        return 1
    width = max(len(c) for c, *_ in results)
    n_ok = sum(1 for *_, ok in results if ok)
    print(f"{'claim':<{width}}  origin   status   expected      actual")
    print("-" * (width + 45))
    for claim, source, origin, expected, actual, ok in results:
        status = "MATCH" if ok else "MISMATCH"
        exp = f"{expected:.5g}" if isinstance(expected, (int, float)) else str(expected)
        act = f"{actual:.5g}" if isinstance(actual, (int, float)) else str(actual)
        print(f"{claim:<{width}}  {origin:<7}  {status:<8} {exp:>10}  {act:>10}")
    print("-" * (width + 45))
    print(f"{n_ok}/{len(results)} verified against a file on disk (scope={ARGS.scope})")
    expected_count = EXPECTED_CHECKS[ARGS.scope]
    count_mismatch = len(results) != expected_count
    if count_mismatch:
        print(
            f"ERROR: scope={ARGS.scope} instantiated {len(results)} checks; "
            f"expected the fixed denominator {expected_count}"
        )
    missing = [r for r in results if not r[5]]
    if missing:
        print("\nUNVERIFIED OR MISMATCHED:")
        for claim, source, origin, expected, actual, _ in missing:
            print(f"  [{origin}] {claim}  (source: {source}, read back: {actual})")
    return 1 if missing or count_mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main())
