"""Fail-closed collection core for the article's sealed Cartesian population.

The historical collection script allowed smoke/limit/seed overrides and treated
the mere existence of a filename as a completed task.  This replacement has no
partial-grid mode: it validates every pre-existing record, collects exactly the
missing registered keys, and publishes a summary only after the full population
passes the same validator used by the scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .integrity import (
    IntegrityError,
    canonical_json_bytes,
    loads_json_strict,
    read_verified_bytes,
    sha256_file,
    write_bytes_once,
    write_json_once,
)
from .sealed import (
    PopulationContract,
    PopulationIntegrityError,
    PopulationKey,
    validate_raw_population,
    validate_record,
    verify_named_executables,
    _parse_named_paths,
)


TaskRunner = Callable[[PopulationKey], Mapping[str, Any]]


def _load_collection_contract(
    *,
    contract_path: Path,
    trusted_contract_sha256: str,
    lock_path: Path,
    model_path: Path,
) -> tuple[PopulationContract, dict[str, Any], dict[str, str]]:
    contract_bytes, contract_sha = read_verified_bytes(
        contract_path, trusted_contract_sha256, label="trusted population contract"
    )
    contract_payload = loads_json_strict(
        contract_bytes.decode("utf-8"), label=str(contract_path)
    )
    contract = PopulationContract.from_mapping(contract_payload)
    lock_bytes, lock_sha = read_verified_bytes(
        lock_path, contract.prediction_lock_sha256, label="prediction lock"
    )
    lock = loads_json_strict(lock_bytes.decode("utf-8"), label=str(lock_path))
    if not isinstance(lock, dict):
        raise IntegrityError("prediction lock must be a JSON object")
    try:
        model_sha_expected = str(lock["frozen_estimator"]["sha256"])
        population = lock["evaluation_population"]
    except (KeyError, TypeError) as exc:
        raise IntegrityError(f"prediction lock is incomplete: {exc}") from exc
    _, model_sha = read_verified_bytes(
        model_path, model_sha_expected, label="frozen estimator"
    )
    if list(population.get("networks", [])) != list(contract.networks):
        raise IntegrityError("collector contract networks disagree with prediction lock")
    if list(population.get("conditions", [])) != list(contract.conditions):
        raise IntegrityError("collector contract conditions disagree with prediction lock")
    if population.get("seed_block") != contract.locked_seed_block:
        raise IntegrityError("collector contract seed block disagrees with prediction lock")
    return contract, lock, {
        "population_contract_sha256": contract_sha,
        "prediction_lock_sha256": lock_sha,
        "frozen_estimator_sha256": model_sha,
    }


def collect_exact_population(
    *,
    contract_path: Path,
    trusted_contract_sha256: str,
    lock_path: Path,
    model_path: Path,
    raw_dir: Path,
    summary_dir: Path,
    executables: Mapping[str, Path],
    run_task: TaskRunner,
    runner_sha256: str,
) -> Path:
    """Collect all and only missing registered tasks, then validate exact closure."""

    contract, _lock, provenance = _load_collection_contract(
        contract_path=contract_path,
        trusted_contract_sha256=trusted_contract_sha256,
        lock_path=lock_path,
        model_path=model_path,
    )
    executable_receipts = verify_named_executables(
        contract.required_collection_executables, executables
    )
    if raw_dir.is_symlink():
        raise PopulationIntegrityError(f"raw directory may not be a symlink: {raw_dir}")
    raw_resolved = raw_dir.resolve()
    summary_resolved = summary_dir.resolve()
    if summary_resolved == raw_resolved:
        raise IntegrityError("summary directory must be outside the exact raw directory")
    try:
        summary_resolved.relative_to(raw_resolved)
    except ValueError:
        pass
    else:
        raise IntegrityError("summary directory may not be inside the raw directory")
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    existing = validate_raw_population(raw_dir, contract, require_complete=False)
    existing_keys = set(existing.keys)
    expected_keys = contract.expected_keys()
    missing = [key for key in expected_keys if key not in existing_keys]
    created = 0
    for key in missing:
        record = dict(run_task(key))
        target = raw_dir / key.filename()
        observed_key = validate_record(record, target, contract)
        if observed_key != key:
            raise PopulationIntegrityError(
                f"runner returned {observed_key}, but collector requested {key}"
            )
        if write_bytes_once(target, canonical_json_bytes(record)):
            created += 1

    final = validate_raw_population(raw_dir, contract, require_complete=True)
    summary = {
        "schema_version": "sealed-collection-summary-v2",
        "evidence_status": contract.evidence_status,
        "historical_limitations": list(contract.historical_limitations),
        "expected_tasks": final.expected_rows,
        "preexisting_valid_tasks": len(existing_keys),
        "newly_created_tasks": created,
        "final_valid_tasks": final.observed_rows,
        "analysis_clusters": final.clusters,
        "rows_per_cluster": final.rows_per_cluster,
        "raw_manifest_sha256": final.raw_manifest_sha256,
        "provenance": {
            **provenance,
            "runner_sha256": runner_sha256,
            "collector_module_sha256": sha256_file(Path(__file__)),
            "python_executable_name": Path(sys.executable).name,
            "python_executable_sha256": sha256_file(Path(sys.executable)),
            "python_version": platform.python_version(),
            "verified_executables": executable_receipts,
        },
    }
    collector_module_sha = summary["provenance"]["collector_module_sha256"]
    python_executable_sha = summary["provenance"]["python_executable_sha256"]
    identity_material = {
        "raw": final.raw_manifest_sha256,
        **provenance,
        "runner": runner_sha256,
        "executables": executable_receipts,
        "preexisting_valid_tasks": len(existing_keys),
        "newly_created_tasks": created,
        "collector_module_sha256": collector_module_sha,
        "python_executable_sha256": python_executable_sha,
    }
    identity = hashlib.sha256(
        json.dumps(identity_material, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    target = summary_dir / f"collection_summary_{identity}.json"
    write_json_once(target, summary)
    return target


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--trusted-contract-sha256", required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument(
        "--executable",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="repeat for every executable required by the population contract",
    )
    parser.add_argument("--runner", required=True, help="dotted callable MODULE:NAME")
    parser.add_argument(
        "--trusted-runner-sha256",
        required=True,
        help="out-of-band SHA-256 of the module containing the runner callable",
    )
    args = parser.parse_args(argv)

    if ":" not in args.runner:
        raise IntegrityError("runner must be written as MODULE:NAME")
    module_name, attribute = args.runner.split(":", 1)
    module = importlib.import_module(module_name)
    module_path = Path(module.__file__ or "")
    read_verified_bytes(
        module_path, args.trusted_runner_sha256, label="trusted runner module"
    )
    run_task = getattr(module, attribute, None)
    if not callable(run_task):
        raise IntegrityError(f"runner is not callable: {args.runner}")
    summary = collect_exact_population(
        contract_path=args.contract,
        trusted_contract_sha256=args.trusted_contract_sha256,
        lock_path=args.lock,
        model_path=args.model,
        raw_dir=args.raw_dir,
        summary_dir=args.summary_dir,
        executables=_parse_named_paths(args.executable),
        run_task=run_task,
        runner_sha256=args.trusted_runner_sha256.lower(),
    )
    print(summary)


if __name__ == "__main__":
    main()
