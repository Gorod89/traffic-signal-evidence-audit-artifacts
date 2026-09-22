"""Fail-closed validation and scoring of a sealed paired-branch population."""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import math
import pickle
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .integrity import (
    IntegrityError,
    load_json_strict,
    loads_json_strict,
    read_bytes_with_sha256,
    read_verified_bytes,
    sha256_file,
    verify_sha256,
    write_json_once,
)


class PopulationIntegrityError(IntegrityError):
    """The raw directory is not the exact population declared by the contract."""


@dataclass(frozen=True, order=True)
class PopulationKey:
    network: str
    condition: str
    seed: int
    snapshot_index: int

    def filename(self) -> str:
        return f"{self.network}_{self.condition}_{self.seed}.json"

    def cluster(self) -> tuple[str, int, int]:
        return (self.network, self.seed, self.snapshot_index)


@dataclass(frozen=True)
class PopulationContract:
    schema_version: str
    evidence_status: str
    prediction_lock_sha256: str
    networks: tuple[str, ...]
    conditions: tuple[str, ...]
    seed_start: int
    seed_count: int
    snapshot_indices: tuple[int, ...]
    split: str
    record_schema_version: int
    feature_dimension: int
    rows_per_cluster: int
    locked_seed_block: str
    required_collection_executables: dict[str, str]
    historical_limitations: tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PopulationContract":
        if not isinstance(raw, dict):
            raise IntegrityError("population contract must be a JSON object")
        required = {
            "schema_version",
            "evidence_status",
            "prediction_lock_sha256",
            "networks",
            "conditions",
            "seed_start",
            "seed_count",
            "snapshot_indices",
            "split",
            "record_schema_version",
            "feature_dimension",
            "rows_per_cluster",
            "locked_seed_block",
            "required_collection_executables",
            "historical_limitations",
        }
        missing = sorted(required - raw.keys())
        if missing:
            raise IntegrityError(f"population contract missing fields: {missing}")
        scalar_types = {
            "schema_version": str,
            "evidence_status": str,
            "prediction_lock_sha256": str,
            "seed_start": int,
            "seed_count": int,
            "split": str,
            "record_schema_version": int,
            "feature_dimension": int,
            "rows_per_cluster": int,
            "locked_seed_block": str,
        }
        for field, expected_type in scalar_types.items():
            if type(raw[field]) is not expected_type:
                raise IntegrityError(
                    f"population contract {field!r} must be {expected_type.__name__}, "
                    f"got {type(raw[field]).__name__}"
                )
        for field, item_type in (
            ("networks", str),
            ("conditions", str),
            ("snapshot_indices", int),
            ("historical_limitations", str),
        ):
            if not isinstance(raw[field], list) or any(
                type(item) is not item_type for item in raw[field]
            ):
                raise IntegrityError(
                    f"population contract {field!r} must be a list of {item_type.__name__}"
                )
        if not isinstance(raw["required_collection_executables"], dict) or any(
            type(name) is not str or type(digest) is not str
            for name, digest in raw["required_collection_executables"].items()
        ):
            raise IntegrityError(
                "required_collection_executables must map strings to SHA-256 strings"
            )
        if raw["schema_version"] != "sealed-population-contract-v2":
            raise IntegrityError(f"unsupported population contract: {raw['schema_version']!r}")
        contract = cls(
            schema_version=str(raw["schema_version"]),
            evidence_status=str(raw["evidence_status"]),
            prediction_lock_sha256=str(raw["prediction_lock_sha256"]).lower(),
            networks=tuple(map(str, raw["networks"])),
            conditions=tuple(map(str, raw["conditions"])),
            seed_start=int(raw["seed_start"]),
            seed_count=int(raw["seed_count"]),
            snapshot_indices=tuple(map(int, raw["snapshot_indices"])),
            split=str(raw["split"]),
            record_schema_version=int(raw["record_schema_version"]),
            feature_dimension=int(raw["feature_dimension"]),
            rows_per_cluster=int(raw["rows_per_cluster"]),
            locked_seed_block=str(raw["locked_seed_block"]),
            required_collection_executables={
                str(name): str(digest).lower()
                for name, digest in dict(raw["required_collection_executables"]).items()
            },
            historical_limitations=tuple(map(str, raw["historical_limitations"])),
        )
        contract.validate()
        return contract

    @classmethod
    def from_path(cls, path: Path) -> "PopulationContract":
        raw = load_json_strict(path)
        return cls.from_mapping(raw)

    def validate(self) -> None:
        if not self.networks or len(set(self.networks)) != len(self.networks):
            raise IntegrityError("networks must be a non-empty unique list")
        if not self.conditions or len(set(self.conditions)) != len(self.conditions):
            raise IntegrityError("conditions must be a non-empty unique list")
        if self.seed_count <= 0 or self.feature_dimension <= 0:
            raise IntegrityError("seed_count and feature_dimension must be positive")
        if not self.snapshot_indices or len(set(self.snapshot_indices)) != len(self.snapshot_indices):
            raise IntegrityError("snapshot_indices must be a non-empty unique list")
        if self.snapshot_indices != (0,):
            raise IntegrityError("this record filename schema supports exactly snapshot_index 0")
        expected_rows = len(self.conditions) * len(self.snapshot_indices)
        if self.rows_per_cluster != expected_rows:
            raise IntegrityError(
                f"rows_per_cluster={self.rows_per_cluster}, expected {expected_rows} "
                "from conditions x snapshots"
            )
        _validate_digest(self.prediction_lock_sha256, label="prediction_lock_sha256")
        for name, digest in self.required_collection_executables.items():
            _validate_digest(digest, label=f"required executable {name}")

    def expected_keys(self) -> tuple[PopulationKey, ...]:
        seeds = range(self.seed_start, self.seed_start + self.seed_count)
        return tuple(
            PopulationKey(network, condition, seed, snapshot)
            for network, condition, seed, snapshot in itertools.product(
                self.networks, self.conditions, seeds, self.snapshot_indices
            )
        )


@dataclass(frozen=True)
class PopulationValidation:
    records: tuple[dict[str, Any], ...]
    paths: tuple[Path, ...]
    keys: tuple[PopulationKey, ...]
    expected_rows: int
    observed_rows: int
    clusters: int
    rows_per_cluster: int
    raw_manifest_sha256: str


def _validate_digest(value: str, *, label: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value):
        raise IntegrityError(f"{label} must be exactly 64 hexadecimal characters")


def _finite_number(value: Any, *, field: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PopulationIntegrityError(f"{path.name}: {field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise PopulationIntegrityError(f"{path.name}: {field} must be finite")
    return number


def validate_record(
    record: Mapping[str, Any], path: Path, contract: PopulationContract
) -> PopulationKey:
    required = {
        "schema_version",
        "network",
        "condition",
        "seed",
        "snapshot_index",
        "split",
        "status",
        "feature_vector",
        "gain_h4",
    }
    missing = sorted(required - record.keys())
    if missing:
        raise PopulationIntegrityError(f"{path.name}: missing record fields {missing}")
    if type(record["schema_version"]) is not int or record["schema_version"] != contract.record_schema_version:
        raise PopulationIntegrityError(
            f"{path.name}: schema_version={record['schema_version']!r}, "
            f"expected {contract.record_schema_version}"
        )
    for field in ("network", "condition", "split", "status"):
        if type(record[field]) is not str:
            raise PopulationIntegrityError(f"{path.name}: {field} must be a string")
    if type(record["seed"]) is not int:
        raise PopulationIntegrityError(f"{path.name}: seed must be an integer")
    if type(record["snapshot_index"]) is not int:
        raise PopulationIntegrityError(f"{path.name}: snapshot_index must be an integer")
    key = PopulationKey(
        record["network"],
        record["condition"],
        record["seed"],
        record["snapshot_index"],
    )
    if path.name != key.filename():
        raise PopulationIntegrityError(
            f"filename/record mismatch: {path.name!r} != {key.filename()!r}"
        )
    if record["split"] != contract.split:
        raise PopulationIntegrityError(
            f"{path.name}: split={record['split']!r}, expected {contract.split!r}"
        )
    if record["status"] != "complete":
        raise PopulationIntegrityError(f"{path.name}: status is not 'complete'")
    vector = record["feature_vector"]
    if not isinstance(vector, list) or len(vector) != contract.feature_dimension:
        size = len(vector) if isinstance(vector, list) else None
        raise PopulationIntegrityError(
            f"{path.name}: feature_vector length={size}, expected {contract.feature_dimension}"
        )
    for index, value in enumerate(vector):
        _finite_number(value, field=f"feature_vector[{index}]", path=path)
    _finite_number(record["gain_h4"], field="gain_h4", path=path)
    return key


def validate_raw_population(
    raw_dir: Path,
    contract: PopulationContract,
    *,
    require_complete: bool = True,
) -> PopulationValidation:
    if raw_dir.is_symlink() or not raw_dir.is_dir():
        raise PopulationIntegrityError(f"raw directory is missing or is a symlink: {raw_dir}")
    entries = tuple(sorted(raw_dir.iterdir(), key=lambda item: item.name))
    unexpected_entries = [
        path.name
        for path in entries
        if not path.is_file() or path.suffix.lower() != ".json" or path.is_symlink()
    ]
    if unexpected_entries:
        raise PopulationIntegrityError(
            f"unexpected entries in exact raw population: {unexpected_entries[:10]}"
        )
    paths = entries
    expected = set(contract.expected_keys())
    seen: dict[PopulationKey, Path] = {}
    records_by_key: dict[PopulationKey, dict[str, Any]] = {}
    digest_by_key: dict[PopulationKey, str] = {}
    for path in paths:
        payload, digest = read_bytes_with_sha256(path, label=f"raw record {path.name}")
        record = loads_json_strict(payload.decode("utf-8"), label=str(path))
        if not isinstance(record, dict):
            raise PopulationIntegrityError(f"{path.name}: record must be a JSON object")
        key = validate_record(record, path, contract)
        if key in seen:
            raise PopulationIntegrityError(
                f"duplicate population key {key}: {seen[key].name}, {path.name}"
            )
        seen[key] = path
        records_by_key[key] = record
        digest_by_key[key] = digest

    actual = set(seen)
    unexpected = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unexpected:
        raise PopulationIntegrityError(f"unexpected population keys: {unexpected[:10]}")
    if require_complete and missing:
        raise PopulationIntegrityError(
            f"missing {len(missing)} of {len(expected)} expected keys: {missing[:10]}"
        )

    cluster_conditions: dict[tuple[str, int, int], set[str]] = {}
    for key in actual:
        cluster_conditions.setdefault(key.cluster(), set()).add(key.condition)
    if require_complete:
        expected_conditions = set(contract.conditions)
        bad = {
            cluster: sorted(conditions)
            for cluster, conditions in cluster_conditions.items()
            if conditions != expected_conditions
        }
        expected_clusters = (
            len(contract.networks) * contract.seed_count * len(contract.snapshot_indices)
        )
        if len(cluster_conditions) != expected_clusters or bad:
            raise PopulationIntegrityError(
                f"cluster structure invalid: clusters={len(cluster_conditions)}/{expected_clusters}, "
                f"bad condition sets={list(bad.items())[:5]}"
            )

    ordered_keys = tuple(sorted(actual))
    ordered_paths = tuple(seen[key] for key in ordered_keys)
    ordered_records = tuple(records_by_key[key] for key in ordered_keys)
    manifest_digest = hashlib.sha256()
    for key in ordered_keys:
        manifest_digest.update(seen[key].name.encode("utf-8"))
        manifest_digest.update(b"\0")
        manifest_digest.update(digest_by_key[key].encode("ascii"))
        manifest_digest.update(b"\n")
    raw_manifest = manifest_digest.hexdigest()
    return PopulationValidation(
        records=ordered_records,
        paths=ordered_paths,
        keys=ordered_keys,
        expected_rows=len(expected),
        observed_rows=len(actual),
        clusters=len(cluster_conditions),
        rows_per_cluster=contract.rows_per_cluster,
        raw_manifest_sha256=raw_manifest,
    )


def verify_named_executables(
    required: Mapping[str, str], supplied: Mapping[str, Path]
) -> dict[str, dict[str, str]]:
    missing = sorted(set(required) - set(supplied))
    extra = sorted(set(supplied) - set(required))
    if missing or extra:
        raise IntegrityError(f"executable mapping mismatch: missing={missing}, extra={extra}")
    result: dict[str, dict[str, str]] = {}
    for name, expected in sorted(required.items()):
        path = supplied[name]
        if path.is_symlink() or not path.is_file():
            raise IntegrityError(f"executable {name!r} is missing or a symlink: {path}")
        actual = verify_sha256(path, expected, label=f"executable {name}")
        result[name] = {"path_name": path.name, "sha256": actual}
    return result


def _resolve_inside(root: Path, relative: str, *, label: str) -> Path:
    if root.is_symlink():
        raise IntegrityError(f"repository root may not be a symlink: {root}")
    root = root.resolve()
    lexical = root / relative
    if lexical.is_symlink():
        raise IntegrityError(f"{label} may not be a symlink: {lexical}")
    candidate = lexical.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise IntegrityError(f"{label} escapes repository root: {relative!r}") from exc
    return candidate


def _cluster_bootstrap(
    fn, clusters: np.ndarray, *, seed: int, replicates: int
) -> tuple[float, float]:
    unique = np.unique(clusters)
    if unique.size < 2:
        raise IntegrityError("at least two analysis clusters are required")
    index_of = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(replicates):
        picked = rng.choice(unique, size=unique.size, replace=True)
        indices = np.concatenate([index_of[cluster] for cluster in picked])
        value = float(fn(indices))
        if math.isfinite(value):
            draws.append(value)
    if len(draws) != replicates:
        raise IntegrityError(
            f"bootstrap produced {len(draws)} valid draws, expected exactly {replicates}"
        )
    quantiles = np.quantile(np.asarray(draws), [0.025, 0.975], method="linear")
    return float(quantiles[0]), float(quantiles[1])


def _load_model(payload: bytes) -> Any:
    # Loading pickle can execute code.  The caller MUST verify its locked hash
    # first; keeping this in a separate function makes fail-before-load testable.
    return pickle.loads(payload)


def evaluate(
    *,
    lock_path: Path,
    contract_path: Path,
    raw_dir: Path,
    model_path: Path,
    repository_root: Path,
    trusted_contract_sha256: str,
) -> dict[str, Any]:
    """Validate every input before loading the model or calculating a score."""

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
        locked_model_sha = str(lock["frozen_estimator"]["sha256"]).lower()
        training = lock["training"]
        training_sha = str(training["source_sha256"]).lower()
        training_source = str(training["source"])
        training_splits = list(training["splits"])
        training_target = str(training["target"])
        fixed = lock["fixed_analysis"]
        threshold = float(lock["predictions"]["primary"]["threshold"])
        secondary_interval = list(lock["predictions"]["secondary_interval"]["interval"])
        evaluation_population = lock["evaluation_population"]
    except (KeyError, TypeError, ValueError) as exc:
        raise IntegrityError(f"prediction lock schema is incomplete: {exc}") from exc
    _validate_digest(locked_model_sha, label="locked model SHA-256")
    _validate_digest(training_sha, label="locked training source SHA-256")

    if list(evaluation_population.get("networks", [])) != list(contract.networks):
        raise IntegrityError("population contract networks disagree with prediction lock")
    if list(evaluation_population.get("conditions", [])) != list(contract.conditions):
        raise IntegrityError("population contract conditions disagree with prediction lock")
    if evaluation_population.get("seed_block") != contract.locked_seed_block:
        raise IntegrityError("population contract seed block disagrees with prediction lock")
    if training_target != "gain_h4":
        raise IntegrityError(f"unsupported locked target: {training_target!r}")
    required_fixed = {
        "reference_predictor": "mean of the training target, as in the V4/V5 Stage-B gate",
        "statistic": "1 - RMSE_model / RMSE_mean",
        "cluster_unit": "(network, seed)",
        "confidence_level": 0.95,
    }
    for field, expected in required_fixed.items():
        if fixed.get(field) != expected:
            raise IntegrityError(
                f"locked analysis {field!r}={fixed.get(field)!r}, expected {expected!r}"
            )
    if int(fixed.get("bootstrap_replicates", 0)) < 1:
        raise IntegrityError("locked bootstrap replicate count must be positive")

    # All of the following checks occur before _load_model and before predict().
    model_bytes, model_sha = read_verified_bytes(
        model_path, locked_model_sha, label="frozen estimator"
    )
    training_path = _resolve_inside(repository_root, training_source, label="training source")
    training_bytes, actual_training_sha = read_verified_bytes(
        training_path, training_sha, label="training source"
    )
    population = validate_raw_population(raw_dir, contract, require_complete=True)

    import pandas as pd

    training_frame = pd.read_csv(io.BytesIO(training_bytes))
    required_training_columns = {"split", "network", "seed", training_target}
    if not required_training_columns.issubset(training_frame.columns):
        raise IntegrityError("locked training columns are absent")
    selected = training_frame[training_frame["split"].isin(training_splits)]
    if len(selected) != int(training["rows"]):
        raise IntegrityError(
            f"training row count mismatch: {len(selected)} != {training['rows']}"
        )
    cluster_count = int(selected.groupby(["network", "seed"]).ngroups)
    if cluster_count != int(training["clusters"]):
        raise IntegrityError(
            f"training cluster count mismatch: {cluster_count} != {training['clusters']}"
        )
    target_values = selected[training_target].to_numpy(dtype=np.float64)
    if not np.isfinite(target_values).all():
        raise IntegrityError("training target contains non-finite values")
    reference = float(target_values.mean())

    model = _load_model(model_bytes)
    x = np.asarray(
        [record["feature_vector"] for record in population.records], dtype=np.float64
    )
    y = np.asarray([record["gain_h4"] for record in population.records], dtype=np.float64)
    if not hasattr(model, "n_features_in_"):
        raise IntegrityError("frozen model does not expose n_features_in_")
    if int(model.n_features_in_) != contract.feature_dimension:
        raise IntegrityError("model feature dimension disagrees with population contract")
    prediction = np.asarray(model.predict(x), dtype=np.float64)
    if prediction.shape != y.shape or not np.isfinite(prediction).all():
        raise IntegrityError("model returned a non-finite or wrongly shaped prediction")
    cluster_keys = sorted({(key.network, key.seed) for key in population.keys})
    cluster_id = {key: index for index, key in enumerate(cluster_keys)}
    clusters = np.asarray(
        [cluster_id[(key.network, key.seed)] for key in population.keys], dtype=np.int64
    )

    def rmse_gain(indices: np.ndarray) -> float:
        model_rmse = float(np.sqrt(np.mean((prediction[indices] - y[indices]) ** 2)))
        reference_rmse = float(np.sqrt(np.mean((reference - y[indices]) ** 2)))
        if reference_rmse == 0.0:
            raise ValueError("reference RMSE is zero")
        return 1.0 - model_rmse / reference_rmse

    indices = np.arange(y.size)
    point = rmse_gain(indices)
    lo, hi = _cluster_bootstrap(
        rmse_gain,
        clusters,
        seed=int(fixed["bootstrap_seed"]),
        replicates=int(fixed["bootstrap_replicates"]),
    )
    primary_met = bool(point >= threshold and lo > 0.0)

    from sklearn.metrics import roc_auc_score

    def auroc(indices: np.ndarray) -> float:
        labels = (y[indices] > 0.0).astype(int)
        if labels.min() == labels.max():
            raise ValueError("degenerate AUROC bootstrap draw")
        return float(roc_auc_score(labels, prediction[indices]))

    auroc_point = auroc(indices)
    auroc_lo, auroc_hi = _cluster_bootstrap(
        auroc,
        clusters,
        seed=int(fixed["bootstrap_seed"]),
        replicates=int(fixed["bootstrap_replicates"]),
    )

    per_network: dict[str, Any] = {}
    for network in contract.networks:
        mask = np.asarray([key.network == network for key in population.keys])
        per_network[network] = {
            "rows": int(mask.sum()),
            "clusters": int(np.unique(clusters[mask]).size),
            "rmse_gain": rmse_gain(np.flatnonzero(mask)),
        }

    result = {
        "schema_version": "sealed-evaluation-v2",
        "evidence_status": contract.evidence_status,
        "historical_limitations": list(contract.historical_limitations),
        "population": {
            "expected_rows": population.expected_rows,
            "observed_rows": population.observed_rows,
            "analysis_clusters": population.clusters,
            "rows_per_cluster": population.rows_per_cluster,
            "raw_manifest_sha256": population.raw_manifest_sha256,
        },
        "reference_predictor_value": reference,
        "primary": {
            "threshold": threshold,
            "observed_rmse_gain": point,
            "ci95": [lo, hi],
            "met": primary_met,
        },
        "secondary": {
            "predicted_interval": secondary_interval,
            "observed": point,
            "met": bool(secondary_interval[0] <= point <= secondary_interval[1]),
        },
        "benefit_auroc": {
            "observed": auroc_point,
            "ci95": [auroc_lo, auroc_hi],
        },
        "per_network": per_network,
        "provenance": {
            "prediction_lock_sha256": lock_sha,
            "population_contract_sha256": contract_sha,
            "frozen_estimator_sha256": model_sha,
            "training_source_sha256": actual_training_sha,
            "evaluator_module_sha256": sha256_file(Path(__file__)),
            "python_executable_name": Path(sys.executable).name,
            "python_executable_sha256": sha256_file(Path(sys.executable)),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "collection_executables_required_by_contract": dict(
                contract.required_collection_executables
            ),
        },
    }
    return result


def _parse_named_paths(values: Iterable[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise IntegrityError(f"expected NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        if not name or name in result:
            raise IntegrityError(f"invalid or duplicate executable name: {name!r}")
        result[name] = Path(raw_path)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--trusted-contract-sha256",
        required=True,
        help="out-of-band SHA-256 of the reviewed population contract",
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    raw_dir = args.raw_dir.resolve()
    try:
        output_dir.relative_to(raw_dir)
    except ValueError:
        pass
    else:
        raise IntegrityError("output directory may not be inside the raw evidence directory")

    report = evaluate(
        lock_path=args.lock,
        contract_path=args.contract,
        raw_dir=args.raw_dir,
        model_path=args.model,
        repository_root=args.repository_root,
        trusted_contract_sha256=args.trusted_contract_sha256,
    )
    identity_payload = {
        name: report["provenance"][name]
        for name in (
            "prediction_lock_sha256",
            "population_contract_sha256",
            "frozen_estimator_sha256",
            "training_source_sha256",
        )
    }
    identity_payload["raw_manifest_sha256"] = report["population"]["raw_manifest_sha256"]
    identity_payload["evaluator_module_sha256"] = report["provenance"][
        "evaluator_module_sha256"
    ]
    identity_payload["python_executable_sha256"] = report["provenance"][
        "python_executable_sha256"
    ]
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    target = output_dir / f"sealed_confirmation_{identity}.json"
    write_json_once(target, report)
    print(target)


if __name__ == "__main__":
    main()
