"""D+ step 1: read all 94 primary_search_v2 trial JSONs into tidy tables.

Read-only with respect to v8/.  Generated outputs go to ``build/`` by
default; replacing the released reference CSVs requires ``--update-reference``.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CORRUPTIONS = ("mcar", "burst", "spatial", "mnar")
EXPECTED_TRIALS = 94
REFERENCE_NAMES = ("trials_candidate_level.csv", "trials_fold_level.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(os.environ.get("V8_SOURCE_ROOT", HERE.parents[1] / "external" / "v8")),
        help="root of the V8 archive (default: V8_SOURCE_ROOT or external/v8)",
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output-dir", type=Path, help="directory for generated CSVs")
    output.add_argument(
        "--update-reference",
        action="store_true",
        help="explicitly replace the two released reference CSVs after validation",
    )
    return parser.parse_args()


def validate_trial(name: str, doc: object) -> None:
    if not isinstance(doc, dict):
        raise ValueError(f"{name}: top-level JSON value must be an object")
    required = ("rung", "candidate_id", "status", "config", "metrics", "fold_results")
    absent = [key for key in required if key not in doc]
    if absent:
        raise ValueError(f"{name}: missing required fields: {', '.join(absent)}")
    if doc["rung"] not in {"R1", "R2"}:
        raise ValueError(f"{name}: unexpected rung {doc['rung']!r}")
    if not isinstance(doc["candidate_id"], str) or not doc["candidate_id"]:
        raise ValueError(f"{name}: candidate_id must be a non-empty string")
    if not isinstance(doc["config"], dict) or not isinstance(doc["metrics"], dict):
        raise ValueError(f"{name}: config and metrics must be objects")
    if not isinstance(doc["fold_results"], list) or len(doc["fold_results"]) != 5:
        raise ValueError(f"{name}: fold_results must contain exactly five folds")
    if any(not isinstance(fold, dict) for fold in doc["fold_results"]):
        raise ValueError(f"{name}: every fold result must be an object")


def load_all(trials: Path) -> list[tuple[str, dict]]:
    if not trials.is_dir():
        raise FileNotFoundError(f"trial directory does not exist: {trials}")
    paths = sorted(trials.glob("*.json"))
    if len(paths) != EXPECTED_TRIALS:
        raise ValueError(f"expected {EXPECTED_TRIALS} trial JSONs in {trials}, found {len(paths)}")
    records = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            doc = json.load(stream)
        validate_trial(path.name, doc)
        records.append((path.name, doc))
    return records


def load_promotion(v8: Path) -> list[str]:
    path = v8 / "results" / "primary_search_v2" / "PROMOTION_R1.json"
    if not path.is_file():
        raise FileNotFoundError(f"promotion record does not exist: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    promoted = doc.get("promoted") if isinstance(doc, dict) else None
    if not isinstance(promoted, list) or any(not isinstance(item, str) for item in promoted):
        raise ValueError(f"{path}: promoted must be a list of candidate identifiers")
    return promoted


def atomic_write_pair(candidates: pd.DataFrame, folds: pd.DataFrame, output_dir: Path) -> None:
    """Stage both CSVs, then replace the pair with rollback on failure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = [output_dir / name for name in REFERENCE_NAMES]
    frames = [candidates, folds]
    staged: list[Path] = []
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        for destination, frame in zip(destinations, frames, strict=True):
            handle = tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv.tmp", prefix=f".{destination.name}.",
                dir=output_dir, delete=False, encoding="utf-8", newline="",
            )
            staged_path = Path(handle.name)
            try:
                frame.to_csv(handle, index=False)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                handle.close()
            staged.append(staged_path)

        for destination in destinations:
            if destination.exists():
                backup_handle = tempfile.NamedTemporaryFile(
                    prefix=f".{destination.name}.", suffix=".previous",
                    dir=output_dir, delete=False,
                )
                backup = Path(backup_handle.name)
                backup_handle.close()
                backup.unlink()
                os.replace(destination, backup)
                backups[destination] = backup
        for staged_path, destination in zip(staged, destinations, strict=True):
            os.replace(staged_path, destination)
            installed.append(destination)
    except BaseException:
        for destination in installed:
            destination.unlink(missing_ok=True)
        for destination, backup in backups.items():
            if backup.exists():
                os.replace(backup, destination)
        raise
    finally:
        for staged_path in staged:
            staged_path.unlink(missing_ok=True)
    for backup in backups.values():
        backup.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    v8 = args.source_root.resolve()
    trials = v8 / "results" / "primary_search_v2" / "trials"
    raw = load_all(trials)
    promoted = load_promotion(v8)
    print(f"trial files: {len(raw)}")

    # ---- schema census -------------------------------------------------
    top_keys: dict[str, int] = {}
    for _, doc in raw:
        for key in doc:
            top_keys[key] = top_keys.get(key, 0) + 1
    print("\nTOP-LEVEL KEYS (count / 94):")
    for key, count in sorted(top_keys.items()):
        print(f"  {key:34s} {count}")

    metric_keys: dict[str, int] = {}
    for _, doc in raw:
        for key in doc.get("metrics", {}):
            metric_keys[key] = metric_keys.get(key, 0) + 1
    print("\nmetrics.* KEYS:")
    for key, count in sorted(metric_keys.items()):
        print(f"  {key:34s} {count}")

    # ---- candidate-level table -----------------------------------------
    rows = []
    for name, doc in raw:
        met = doc.get("metrics", {}) or {}
        cfg = doc.get("config", {}) or {}
        corr = doc.get("corruption_h4_improvement", {}) or {}
        struct = doc.get("structural", {}) or {}
        cond = met.get("condition_improvements", {}) or {}
        net = met.get("network_improvements", {}) or {}
        row = {
            "file": name,
            "rung": doc.get("rung"),
            "candidate_id": doc.get("candidate_id"),
            "recipe_id": doc.get("recipe_id"),
            "status": doc.get("status"),
            "error": doc.get("error"),
            "model_seed": doc.get("model_seed"),
            "parameter_count": doc.get("parameter_count"),
            "encoder": cfg.get("encoder"),
            "paired_head": cfg.get("paired_head"),
            "width": cfg.get("width"),
            "depth": cfg.get("depth"),
            "dropout": cfg.get("dropout"),
            "activation": cfg.get("activation"),
            "action_dim": cfg.get("action_dim"),
            "max_actions": cfg.get("max_actions"),
            "overall_improvement": met.get("overall_improvement"),
            "h4_improvement": met.get("h4_improvement"),
            "overall_nrmse": met.get("overall_nrmse"),
            "h4_nrmse": met.get("h4_nrmse"),
            "worst_condition_improvement": met.get("worst_condition_improvement"),
            "worst_network_improvement": met.get("worst_network_improvement"),
            "groupcv_folds": met.get("groupcv_folds"),
            "action_permutation_degradation": doc.get("action_permutation_degradation"),
            "effective_rank_fraction": doc.get("effective_rank_fraction"),
            "equal_action_max_abs": struct.get("equal_action_max_abs"),
            "swap_antisymmetry_max_abs": struct.get("swap_antisymmetry_max_abs"),
            "peak_vram_mib": doc.get("peak_vram_mib"),
            "p95_batch1_ms": doc.get("p95_batch1_ms"),
            "wall_seconds": doc.get("wall_seconds"),
            "train_rows": doc.get("train_rows"),
            "tune_rows": doc.get("tune_rows"),
            "n_train_clusters": len(doc.get("train_clusters", []) or []),
            "n_folds_recorded": len(doc.get("fold_results", []) or []),
            "model_sha256": doc.get("model_sha256"),
        }
        for mode in CORRUPTIONS:
            row[f"corr_{mode}"] = corr.get(mode)
        vals = [corr.get(m) for m in CORRUPTIONS if corr.get(m) is not None]
        row["corr_min"] = min(vals) if vals else np.nan
        row["corr_mean"] = float(np.mean(vals)) if vals else np.nan
        for key, value in cond.items():
            row[f"cond_{key}"] = value
        for key, value in net.items():
            row[f"net_{key}"] = value
        rows.append(row)
    candidates = pd.DataFrame(rows)

    # ---- fold-level table ----------------------------------------------
    fold_rows = []
    for name, doc in raw:
        cfg = doc.get("config", {}) or {}
        for fold in doc.get("fold_results", []) or []:
            met = fold.get("metrics", {}) or {}
            corr = fold.get("corruption_h4_improvement", {}) or {}
            struct = fold.get("structural", {}) or {}
            frow = {
                "rung": doc.get("rung"),
                "candidate_id": doc.get("candidate_id"),
                "encoder": cfg.get("encoder"),
                "paired_head": cfg.get("paired_head"),
                "width": cfg.get("width"),
                "depth": cfg.get("depth"),
                "dropout": cfg.get("dropout"),
                "activation": cfg.get("activation"),
                "fold": fold.get("fold"),
                "status": fold.get("status"),
                "overall_improvement": met.get("overall_improvement"),
                "h4_improvement": met.get("h4_improvement"),
                "overall_nrmse": met.get("overall_nrmse"),
                "h4_nrmse": met.get("h4_nrmse"),
                "worst_condition_improvement": met.get("worst_condition_improvement"),
                "worst_network_improvement": met.get("worst_network_improvement"),
                "action_permutation_degradation": fold.get("action_permutation_degradation"),
                "effective_rank_fraction": fold.get("effective_rank_fraction"),
                "equal_action_max_abs": struct.get("equal_action_max_abs"),
                "swap_antisymmetry_max_abs": struct.get("swap_antisymmetry_max_abs"),
                "peak_vram_mib": fold.get("peak_vram_mib"),
                "p95_batch1_ms": fold.get("p95_batch1_ms"),
                "n_epochs": len(fold.get("training_curve", []) or []),
                "final_train_loss": (
                    (fold.get("training_curve") or [{}])[-1].get("mean_train_loss")
                    if fold.get("training_curve")
                    else None
                ),
            }
            for mode in CORRUPTIONS:
                frow[f"corr_{mode}"] = corr.get(mode)
            for key, value in (met.get("condition_improvements") or {}).items():
                frow[f"cond_{key}"] = value
            for key, value in (met.get("network_improvements") or {}).items():
                frow[f"net_{key}"] = value
            fold_rows.append(frow)
    folds = pd.DataFrame(fold_rows)

    # Everything that could invalidate the outputs is checked before either
    # destination is opened.
    if len(candidates) != EXPECTED_TRIALS or len(folds) != EXPECTED_TRIALS * 5:
        raise ValueError(
            f"unexpected output shape: {len(candidates)} candidates and {len(folds)} folds"
        )
    for frame_name, frame, columns in (
        ("candidate", candidates, ("rung", "candidate_id", "status")),
        ("fold", folds, ("rung", "candidate_id", "fold", "status")),
    ):
        missing_columns = [column for column in columns if column not in frame]
        if missing_columns:
            raise ValueError(f"{frame_name} table lacks columns: {', '.join(missing_columns)}")
        if frame[list(columns)].isnull().any().any():
            raise ValueError(f"{frame_name} table has nulls in required columns")

    print(f"\ncandidate rows: {len(candidates)}  fold rows: {len(folds)}")
    print("\nrung / status:")
    print(candidates.groupby(["rung", "status"]).size().to_string())
    print("\nepochs per fold by rung:")
    print(folds.groupby("rung")["n_epochs"].agg(["min", "max", "mean"]).to_string())

    r1 = set(candidates.loc[candidates.rung == "R1", "candidate_id"])
    r2 = set(candidates.loc[candidates.rung == "R2", "candidate_id"])
    print(f"\nR1 trials: {len(r1)}  R2 trials: {len(r2)}  promoted at R1: {len(promoted)}")
    print(f"promoted but MISSING an R2 trial: {[c for c in promoted if c not in r2]}")
    print(f"R2 trials not in promoted list: {sorted(r2 - set(promoted))}")

    print("\nTOP 12 R2 by overall_improvement:")
    cols = [
        "candidate_id", "encoder", "paired_head", "width", "depth",
        "overall_improvement", "h4_improvement", "worst_condition_improvement", "corr_min",
        "action_permutation_degradation",
    ]
    print(
        candidates[candidates.rung == "R2"]
        .sort_values("overall_improvement", ascending=False)[cols]
        .to_string(index=False)
    )
    print("\nTOP 10 R1 by overall_improvement:")
    print(
        candidates[candidates.rung == "R1"]
        .sort_values("overall_improvement", ascending=False)
        .head(10)[cols]
        .to_string(index=False)
    )

    output_dir = HERE if args.update_reference else (args.output_dir or HERE / "build")
    atomic_write_pair(candidates, folds, output_dir.resolve())
    print(f"\nwrote validated CSV pair to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
