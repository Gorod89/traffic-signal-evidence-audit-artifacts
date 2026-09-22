#!/usr/bin/env python3
"""Read-only audit of the v10 sealed test and conformal harm analysis.

The script reads the original artifacts under ``v10_proposal`` and writes one
JSON report beside this file.  It never writes into the source tree.

Two audits are reproduced:

1. Reconstruct the frozen-estimator score at the file-system creation time of
   the first ``sealed_confirmation.json`` and at the final 1,800-record state.
2. Refit the published post-opening ExtraTrees model, reproduce its conformal
   margin, and measure simultaneous cluster coverage overall and by network.

The first audit necessarily reconstructs the overwritten interim result from
file-system creation times; it is not a recovered copy of the first JSON file.
On Windows, ``st_ctime`` is the file creation time used for that reconstruction.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import platform
import re
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor


DEFAULT_OUTPUT = Path(__file__).with_name("audit_sealed_interim_and_conformal.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_iso(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds).astimezone().isoformat()


def json_vector(value: Any) -> np.ndarray:
    if isinstance(value, str):
        value = json.loads(value)
    return np.asarray(value, dtype=np.float64).ravel()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def cluster_bootstrap_interval(
    statistic: Callable[[np.ndarray], float],
    clusters: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    unique = np.unique(clusters)
    index_of = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
    draws: list[float] = []
    for _ in range(replicates):
        picked = rng.choice(unique, size=unique.size, replace=True)
        index = np.concatenate([index_of[cluster] for cluster in picked])
        draws.append(float(statistic(index)))
    return tuple(float(x) for x in np.percentile(draws, [2.5, 97.5]))


def load_complete_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") == "complete" and record.get("gain_h4") is not None:
            records.append(record)
    return records


def score_frozen_records(
    records: list[dict[str, Any]],
    *,
    model: Any,
    reference: float,
    bootstrap_seed: int,
    bootstrap_replicates: int,
    threshold: float,
) -> dict[str, Any]:
    features = np.vstack([json_vector(row["feature_vector"]) for row in records])
    outcome = np.asarray([float(row["gain_h4"]) for row in records])
    clusters = np.asarray(
        [f"{row['network']}|{row['seed']}" for row in records], dtype=str
    )
    prediction = np.asarray(model.predict(features), dtype=np.float64)

    def rmse_gain(index: np.ndarray) -> float:
        model_rmse = float(np.sqrt(np.mean((prediction[index] - outcome[index]) ** 2)))
        reference_rmse = float(
            np.sqrt(np.mean((reference - outcome[index]) ** 2))
        )
        return 1.0 - model_rmse / reference_rmse

    all_index = np.arange(outcome.size)
    point = float(rmse_gain(all_index))
    lower, upper = cluster_bootstrap_interval(
        rmse_gain,
        clusters,
        seed=bootstrap_seed,
        replicates=bootstrap_replicates,
    )
    counts = Counter(clusters.tolist())
    return {
        "analysable_rows": int(outcome.size),
        "analysis_clusters": int(len(counts)),
        "rows_per_cluster_min": int(min(counts.values())),
        "rows_per_cluster_max": int(max(counts.values())),
        "incomplete_cluster_keys": sorted(
            key for key, count in counts.items() if count != 6
        ),
        "rmse_gain": point,
        "ci95": [lower, upper],
        "decision_rule": {
            "threshold": threshold,
            "point_at_least_threshold": bool(point >= threshold),
            "lower_bound_above_zero": bool(lower > 0.0),
            "confirmed": bool(point >= threshold and lower > 0.0),
        },
    }


def parse_seed_range(seed_block: Any) -> tuple[int, int]:
    if isinstance(seed_block, (list, tuple)) and len(seed_block) == 2:
        return int(seed_block[0]), int(seed_block[1])
    match = re.search(r"seeds?\s+(\d+)\s*[-–]\s*(\d+)", str(seed_block))
    if not match:
        raise ValueError(f"cannot parse seed range from {seed_block!r}")
    return int(match.group(1)), int(match.group(2))


def key_of(record: dict[str, Any]) -> tuple[str, int, str]:
    return str(record["network"]), int(record["seed"]), str(record["condition"])


def key_text(key: tuple[str, int, str]) -> str:
    network, seed, condition = key
    return f"{network}|{seed}|{condition}"


def raw_manifest_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def audit_sealed_interim(source_root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    phase0 = source_root / "v10_proposal" / "phase0"
    sealed = phase0 / "sealed"
    lock_path = sealed / "PREDICTION_LOCK.json"
    model_path = sealed / "frozen_estimator.pkl"
    result_path = phase0 / "results" / "sealed_confirmation.json"
    evaluator_path = phase0 / "evaluate_sealed.py"

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    stored_result = json.loads(result_path.read_text(encoding="utf-8"))
    raw_paths = sorted(Path(path) for path in glob.glob(str(sealed / "raw" / "*.json")))
    if not raw_paths:
        raise FileNotFoundError(f"no sealed records under {sealed / 'raw'}")

    result_stat = result_path.stat()
    first_score_epoch = result_stat.st_ctime
    final_score_epoch = result_stat.st_mtime
    interim_paths = [path for path in raw_paths if path.stat().st_ctime <= first_score_epoch]
    late_paths = [path for path in raw_paths if path.stat().st_ctime > first_score_epoch]

    all_records = load_complete_records(raw_paths)
    interim_records = load_complete_records(interim_paths)

    training_path = source_root / lock["training"]["source"]
    training_sha = sha256_file(training_path)
    training_frame = pd.read_csv(training_path)
    training_rows = training_frame[
        training_frame.split.isin(lock["training"]["splits"])
    ]
    reference = float(training_rows[lock["training"]["target"]].mean())

    model_sha = sha256_file(model_path)
    if model_sha != lock["frozen_estimator"]["sha256"]:
        raise RuntimeError("frozen estimator does not match prediction lock")
    model = pickle.loads(model_path.read_bytes())
    fixed = lock["fixed_analysis"]
    threshold = float(lock["predictions"]["primary"]["threshold"])

    interim_score = score_frozen_records(
        interim_records,
        model=model,
        reference=reference,
        bootstrap_seed=int(fixed["bootstrap_seed"]),
        bootstrap_replicates=int(fixed["bootstrap_replicates"]),
        threshold=threshold,
    )
    final_score = score_frozen_records(
        all_records,
        model=model,
        reference=reference,
        bootstrap_seed=int(fixed["bootstrap_seed"]),
        bootstrap_replicates=int(fixed["bootstrap_replicates"]),
        threshold=threshold,
    )

    seed_start, seed_end = parse_seed_range(lock["evaluation_population"]["seed_block"])
    expected_keys = {
        (network, seed, condition)
        for network in lock["evaluation_population"]["networks"]
        for seed in range(seed_start, seed_end + 1)
        for condition in lock["evaluation_population"]["conditions"]
    }
    actual_counter = Counter(key_of(record) for record in all_records)
    interim_counter = Counter(key_of(record) for record in interim_records)

    evaluator_text = evaluator_path.read_text(encoding="utf-8")
    checks_in_evaluator = {
        "verifies_frozen_model_sha": "actual_model_sha" in evaluator_text,
        "mentions_locked_training_source_sha": "source_sha256" in evaluator_text,
        "checks_duplicate_population_keys": "duplicate" in evaluator_text.lower(),
        "checks_expected_cartesian_population": "itertools.product" in evaluator_text
        or "expected_keys" in evaluator_text,
        "checks_six_rows_per_cluster": "rows_per_cluster" in evaluator_text,
    }

    late = []
    for path in late_paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        late.append(
            {
                "file": path.name,
                "network": record.get("network"),
                "seed": record.get("seed"),
                "condition": record.get("condition"),
                "creation_time_local": local_iso(path.stat().st_ctime),
                "mtime_local": local_iso(path.stat().st_mtime),
                "sha256": sha256_file(path),
            }
        )

    stored_primary = stored_result["primary"]
    audit = {
        "scope": (
            "read-only reconstruction from current frozen estimator, current raw records, "
            "the prediction lock, and Windows file-system times"
        ),
        "filesystem": {
            "os_name": os.name,
            "platform": platform.platform(),
            "ctime_interpretation": (
                "creation time on Windows; the interim JSON was overwritten and is not recovered"
            ),
            "first_score_file_creation_local": local_iso(first_score_epoch),
            "final_score_file_mtime_local": local_iso(final_score_epoch),
        },
        "late_records_after_first_score_creation": late,
        "late_record_count": len(late),
        "records_present_by_first_score_creation": len(interim_paths),
        "records_in_final_directory": len(raw_paths),
        "interim_reconstruction": interim_score,
        "final_reconstruction": final_score,
        "stored_final_result": {
            "analysable_rows": stored_result["sealed_data"]["analysable"],
            "analysis_clusters": stored_result["sealed_data"]["clusters"],
            "rmse_gain": stored_primary["observed_rmse_gain"],
            "ci95": stored_primary["ci95"],
            "met": stored_primary["met"],
        },
        "final_reconstruction_minus_stored": {
            "rmse_gain": final_score["rmse_gain"] - stored_primary["observed_rmse_gain"],
            "ci95_lower": final_score["ci95"][0] - stored_primary["ci95"][0],
            "ci95_upper": final_score["ci95"][1] - stored_primary["ci95"][1],
        },
        "population_integrity_current_final": {
            "expected_key_count": len(expected_keys),
            "actual_unique_key_count": len(actual_counter),
            "missing_keys": sorted(key_text(key) for key in expected_keys - set(actual_counter)),
            "unexpected_keys": sorted(key_text(key) for key in set(actual_counter) - expected_keys),
            "duplicate_keys": sorted(
                key_text(key) for key, count in actual_counter.items() if count != 1
            ),
        },
        "population_integrity_at_first_score_creation": {
            "actual_unique_key_count": len(interim_counter),
            "missing_keys": sorted(
                key_text(key) for key in expected_keys - set(interim_counter)
            ),
            "unexpected_keys": sorted(
                key_text(key) for key in set(interim_counter) - expected_keys
            ),
            "duplicate_keys": sorted(
                key_text(key) for key, count in interim_counter.items() if count != 1
            ),
        },
        "locked_training_source": {
            "path": training_path.relative_to(source_root).as_posix(),
            "locked_sha256": lock["training"].get("source_sha256"),
            "actual_sha256": training_sha,
            "matches_lock": training_sha == lock["training"].get("source_sha256"),
            "reference_mean": reference,
        },
        "evaluator_static_checks": checks_in_evaluator,
        "audit_finding": (
            "The first score file was created before three expected records existed. "
            "The reconstructed interim and final analyses both meet the locked rule, "
            "but the outcome was exposed before the fixed 1,800-cell population completed."
        ),
    }
    hashes = {
        "prediction_lock": sha256_file(lock_path),
        "frozen_estimator": model_sha,
        "training_source": training_sha,
        "stored_sealed_confirmation": sha256_file(result_path),
        "evaluate_sealed_py": sha256_file(evaluator_path),
        "sealed_raw_manifest_name_and_content_sha256": raw_manifest_sha256(raw_paths),
    }
    return audit, hashes


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> tuple[int, float]:
    order = math.ceil((1.0 - alpha) * (scores.size + 1))
    if order > scores.size:
        return order, math.inf
    return order, float(np.sort(scores)[order - 1])


def simultaneous_scores(
    prediction: np.ndarray,
    outcome: np.ndarray,
    clusters: np.ndarray,
) -> np.ndarray:
    residual = prediction - outcome
    return np.asarray(
        [float(np.max(residual[clusters == cluster])) for cluster in np.unique(clusters)]
    )


def coverage_summary(scores: np.ndarray, q: float, alpha: float) -> dict[str, Any]:
    order, empirical_q = finite_sample_quantile(scores, alpha)
    covered = int(np.sum(scores <= q))
    return {
        "clusters": int(scores.size),
        "covered_clusters": covered,
        "simultaneous_cluster_coverage": float(covered / scores.size),
        "nominal_coverage": 1.0 - alpha,
        "finite_sample_order_statistic_k": order,
        "finite_sample_q_at_nominal_level": empirical_q,
        "published_q_shortfall": float(empirical_q - q),
    }


def audit_conformal(source_root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    phase2 = source_root / "v10_proposal" / "phase2"
    sys.path.insert(0, str(phase2))
    try:
        from phase2_common import SEED, load_csv_blocks, load_sealed_block
    finally:
        # Imports keep their module bindings; remove only our path mutation.
        sys.path.pop(0)

    train, calibration, _ = load_csv_blocks()
    sealed, _ = load_sealed_block()
    model = ExtraTreesRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
    model.fit(train.con, train.y)
    calibration_prediction = np.asarray(model.predict(calibration.con), dtype=np.float64)
    sealed_prediction = np.asarray(model.predict(sealed.con), dtype=np.float64)

    alpha = 0.10
    calibration_scores = simultaneous_scores(
        calibration_prediction, calibration.y, calibration.clusters
    )
    order, q = finite_sample_quantile(calibration_scores, alpha)

    sealed_scores = simultaneous_scores(sealed_prediction, sealed.y, sealed.clusters)
    by_network: dict[str, Any] = {}
    for network in sorted(set(sealed.network.tolist())):
        mask = sealed.network == network
        scores = simultaneous_scores(
            sealed_prediction[mask], sealed.y[mask], sealed.clusters[mask]
        )
        by_network[network] = coverage_summary(scores, q, alpha)

    selection = sealed_prediction - q > 0.0
    selection_by_network: dict[str, Any] = {}
    for network in sorted(set(sealed.network.tolist())):
        mask = selection & (sealed.network == network)
        selection_by_network[network] = {
            "selected_rows": int(mask.sum()),
            "active_clusters": int(np.unique(sealed.clusters[mask]).size),
            "harmful_rows": int(np.sum(sealed.y[mask] < 0.0)),
            "mean_realised_gain": float(np.mean(sealed.y[mask])) if mask.any() else None,
        }

    result_path = phase2 / "results" / "selective_policy.json"
    stored = json.loads(result_path.read_text(encoding="utf-8"))
    stored_margin = stored["conformal_calibration"]
    calibration_networks = Counter(calibration.network.tolist())
    sealed_networks = Counter(sealed.network.tolist())

    audit = {
        "published_model": "ExtraTreesRegressor(n_estimators=300, random_state=20260811)",
        "model_selection_recorded_in_artifact": stored["protocol"].get("model_selection"),
        "calibration": {
            "rows": calibration.n_rows,
            "clusters": calibration.n_clusters,
            "network_rows": dict(sorted(calibration_networks.items())),
            "alpha": alpha,
            "order_statistic_k": order,
            "recomputed_q": q,
            "stored_q": stored_margin["q"],
            "q_difference": q - stored_margin["q"],
            "coverage_on_same_calibration_clusters": coverage_summary(
                calibration_scores, q, alpha
            ),
        },
        "sealed": {
            "rows": sealed.n_rows,
            "clusters": sealed.n_clusters,
            "network_rows": dict(sorted(sealed_networks.items())),
            "coverage_pooled": coverage_summary(sealed_scores, q, alpha),
            "coverage_by_network": by_network,
            "default_rule_selection": {
                "selected_rows": int(selection.sum()),
                "active_clusters": int(np.unique(sealed.clusters[selection]).size),
                "harmful_rows": int(np.sum(sealed.y[selection] < 0.0)),
                "by_network": selection_by_network,
            },
        },
        "validity_checks": {
            "model_selected_using_same_calibration_labels_used_for_conformal_scores": bool(
                "calibration-split" in str(stored["protocol"].get("model_selection", ""))
            ),
            "calibration_network_set": sorted(set(calibration.network.tolist())),
            "sealed_network_set": sorted(set(sealed.network.tolist())),
            "sealed_contains_network_absent_from_calibration": sorted(
                set(sealed.network.tolist()) - set(calibration.network.tolist())
            ),
            "sealed_pooled_coverage_meets_nominal": bool(
                np.mean(sealed_scores <= q) >= 1.0 - alpha
            ),
        },
        "audit_finding": (
            "The published q is exactly reproduced, but pooled sealed simultaneous "
            "cluster coverage is below 90%, with the largest failure on the network "
            "absent from calibration. Model selection also used the same calibration "
            "labels later reused for conformal scores. The descriptive selection and "
            "harm counts remain reproducible; the nominal conformal guarantee does not."
        ),
    }
    hashes = {
        "selective_policy_result": sha256_file(result_path),
        "selective_policy_py": sha256_file(phase2 / "selective_policy.py"),
        "phase2_common_py": sha256_file(phase2 / "phase2_common.py"),
    }
    return audit, hashes


def build_report(source_root: Path) -> dict[str, Any]:
    interim, interim_hashes = audit_sealed_interim(source_root)
    conformal, conformal_hashes = audit_conformal(source_root)
    return {
        "schema_version": "audit-sealed-interim-and-conformal-v1",
        "source_root": "<external-artifact-root>",
        "script": {
            "path": Path(__file__).name,
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "input_sha256": {**interim_hashes, **conformal_hashes},
        "sealed_interim_audit": interim,
        "conformal_audit": conformal,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Directory containing v10_proposal and v6",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON report path (default: beside this script)",
    )
    args = parser.parse_args()

    report = build_report(args.source_root)
    atomic_write_json(args.output, report)
    print(args.output.resolve())
    print(f"script_sha256={report['script']['sha256']}")
    interim = report["sealed_interim_audit"]
    print(
        "interim="
        f"{100 * interim['interim_reconstruction']['rmse_gain']:.4f}% "
        f"CI={interim['interim_reconstruction']['ci95']} "
        f"rows={interim['interim_reconstruction']['analysable_rows']}"
    )
    conformal = report["conformal_audit"]
    print(
        "conformal="
        f"q={conformal['calibration']['recomputed_q']:.10f} "
        "sealed_coverage="
        f"{conformal['sealed']['coverage_pooled']['simultaneous_cluster_coverage']:.6f}"
    )


if __name__ == "__main__":
    main()
