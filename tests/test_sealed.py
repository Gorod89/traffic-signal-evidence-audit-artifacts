from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import numpy as np
from sklearn.linear_model import LinearRegression

from evidence_tools.collector import collect_exact_population
from evidence_tools.integrity import IntegrityError, sha256_file
from evidence_tools.sealed import (
    PopulationContract,
    PopulationIntegrityError,
    PopulationKey,
    evaluate,
    validate_raw_population,
)


@dataclass
class Fixture:
    root: Path
    lock: Path
    contract: Path
    contract_sha: str
    model: Path
    training: Path
    raw: Path
    summary: Path
    executable: Path


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _record(key: PopulationKey) -> dict:
    gain = -1.0 if key.condition == "negative" else 1.0
    return {
        "schema_version": 1,
        "network": key.network,
        "condition": key.condition,
        "seed": key.seed,
        "snapshot_index": key.snapshot_index,
        "split": "sealed_probe",
        "status": "complete",
        "feature_vector": [gain, 0.0],
        "gain_h4": gain,
    }


def make_fixture(root: Path, *, populate_raw: bool = True) -> Fixture:
    training = root / "train.csv"
    training.write_text(
        "network,seed,split,gain_h4\n"
        "fit1,101,fit,-1\n"
        "fit1,102,fit,1\n"
        "fit2,103,fit,-2\n"
        "fit2,104,fit,2\n",
        encoding="utf-8",
    )
    model = LinearRegression().fit(
        np.asarray([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        np.asarray([-2.0, -1.0, 1.0, 2.0]),
    )
    model_path = root / "model.pkl"
    model_path.write_bytes(pickle.dumps(model))
    executable = root / "sumo.bin"
    executable.write_bytes(b"fixture executable")

    lock = root / "lock.json"
    lock_payload = {
        "schema_version": 1,
        "training": {
            "source": "train.csv",
            "source_sha256": sha256_file(training),
            "splits": ["fit"],
            "rows": 4,
            "clusters": 4,
            "target": "gain_h4",
        },
        "frozen_estimator": {
            "path": "model.pkl",
            "sha256": sha256_file(model_path),
        },
        "evaluation_population": {
            "seed_block": "fixture seeds 1-2",
            "networks": ["n1", "n2"],
            "conditions": ["negative", "positive"],
        },
        "predictions": {
            "primary": {"threshold": 0.1},
            "secondary_interval": {"interval": [0.0, 1.1]},
        },
        "fixed_analysis": {
            "reference_predictor": "mean of the training target, as in the V4/V5 Stage-B gate",
            "statistic": "1 - RMSE_model / RMSE_mean",
            "cluster_unit": "(network, seed)",
            "bootstrap_replicates": 40,
            "bootstrap_seed": 123,
            "confidence_level": 0.95,
        },
    }
    _json(lock, lock_payload)

    contract = root / "contract.json"
    contract_payload = {
        "schema_version": "sealed-population-contract-v2",
        "evidence_status": "synthetic unit-test contract",
        "prediction_lock_sha256": sha256_file(lock),
        "networks": ["n1", "n2"],
        "conditions": ["negative", "positive"],
        "seed_start": 1,
        "seed_count": 2,
        "snapshot_indices": [0],
        "split": "sealed_probe",
        "record_schema_version": 1,
        "feature_dimension": 2,
        "rows_per_cluster": 2,
        "locked_seed_block": "fixture seeds 1-2",
        "required_collection_executables": {"sumo": sha256_file(executable)},
        "historical_limitations": ["synthetic fixture"],
    }
    _json(contract, contract_payload)
    raw = root / "raw"
    raw.mkdir()
    parsed_contract = PopulationContract.from_path(contract)
    if populate_raw:
        for key in parsed_contract.expected_keys():
            _json(raw / key.filename(), _record(key))
    summary = root / "summaries"
    summary.mkdir()
    return Fixture(
        root=root,
        lock=lock,
        contract=contract,
        contract_sha=sha256_file(contract),
        model=model_path,
        training=training,
        raw=raw,
        summary=summary,
        executable=executable,
    )


class SealedTests(unittest.TestCase):
    def test_exact_population_and_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = make_fixture(Path(tmp))
            contract = PopulationContract.from_path(fixture.contract)
            validation = validate_raw_population(fixture.raw, contract)
            self.assertEqual(validation.observed_rows, 8)
            self.assertEqual(validation.clusters, 4)
            self.assertEqual(validation.rows_per_cluster, 2)
            report = evaluate(
                lock_path=fixture.lock,
                contract_path=fixture.contract,
                raw_dir=fixture.raw,
                model_path=fixture.model,
                repository_root=fixture.root,
                trusted_contract_sha256=fixture.contract_sha,
            )
            self.assertEqual(report["population"]["observed_rows"], 8)
            self.assertAlmostEqual(report["primary"]["observed_rmse_gain"], 1.0)
            self.assertEqual(report["primary"]["ci95"], [1.0, 1.0])
            self.assertTrue(report["primary"]["met"])

    def test_missing_population_fails_before_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = make_fixture(Path(tmp))
            next(fixture.raw.glob("*.json")).unlink()
            with mock.patch("evidence_tools.sealed._load_model") as loader:
                with self.assertRaises(PopulationIntegrityError):
                    evaluate(
                        lock_path=fixture.lock,
                        contract_path=fixture.contract,
                        raw_dir=fixture.raw,
                        model_path=fixture.model,
                        repository_root=fixture.root,
                        trusted_contract_sha256=fixture.contract_sha,
                    )
                loader.assert_not_called()

    def test_missing_plus_unexpected_with_same_count_still_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = make_fixture(Path(tmp))
            victim = next(fixture.raw.glob("n1_negative_1.json"))
            victim.unlink()
            unexpected = _record(PopulationKey("n1", "other", 1, 0))
            _json(fixture.raw / "n1_other_1.json", unexpected)
            contract = PopulationContract.from_path(fixture.contract)
            with self.assertRaises(PopulationIntegrityError):
                validate_raw_population(fixture.raw, contract)

    def test_schema_filename_status_vector_and_nonfinite_fail(self) -> None:
        mutations = (
            ("schema_version", 2),
            ("status", "failed"),
            ("feature_vector", [1.0]),
            ("gain_h4", float("nan")),
        )
        for field, value in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                fixture = make_fixture(Path(tmp))
                path = next(fixture.raw.glob("*.json"))
                record = json.loads(path.read_text(encoding="utf-8"))
                record[field] = value
                _json(path, record)
                with self.assertRaises((PopulationIntegrityError, IntegrityError)):
                    validate_raw_population(
                        fixture.raw, PopulationContract.from_path(fixture.contract)
                    )

    def test_unexpected_directory_or_temporary_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = make_fixture(Path(tmp))
            (fixture.raw / "scratch").mkdir()
            with self.assertRaises(PopulationIntegrityError):
                validate_raw_population(
                    fixture.raw, PopulationContract.from_path(fixture.contract)
                )

    def test_modified_model_or_training_fails_before_unpickle(self) -> None:
        for target_name in ("model", "training"):
            with self.subTest(target=target_name), tempfile.TemporaryDirectory() as tmp:
                fixture = make_fixture(Path(tmp))
                target = getattr(fixture, target_name)
                target.write_bytes(target.read_bytes() + b"changed")
                with mock.patch("evidence_tools.sealed._load_model") as loader:
                    with self.assertRaises(IntegrityError):
                        evaluate(
                            lock_path=fixture.lock,
                            contract_path=fixture.contract,
                            raw_dir=fixture.raw,
                            model_path=fixture.model,
                            repository_root=fixture.root,
                            trusted_contract_sha256=fixture.contract_sha,
                        )
                    loader.assert_not_called()

    def test_contract_requires_exact_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = make_fixture(Path(tmp))
            payload = json.loads(fixture.contract.read_text(encoding="utf-8"))
            payload["seed_count"] = "2"
            _json(fixture.contract, payload)
            with self.assertRaises(IntegrityError):
                PopulationContract.from_path(fixture.contract)

    def test_collector_validates_resume_executables_and_closure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = make_fixture(Path(tmp), populate_raw=False)
            calls: list[PopulationKey] = []

            def runner(key: PopulationKey) -> dict:
                calls.append(key)
                return _record(key)

            summary = collect_exact_population(
                contract_path=fixture.contract,
                trusted_contract_sha256=fixture.contract_sha,
                lock_path=fixture.lock,
                model_path=fixture.model,
                raw_dir=fixture.raw,
                summary_dir=fixture.summary,
                executables={"sumo": fixture.executable},
                run_task=runner,
                runner_sha256="a" * 64,
            )
            self.assertEqual(len(calls), 8)
            payload = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(payload["expected_tasks"], 8)
            self.assertEqual(payload["newly_created_tasks"], 8)
            self.assertEqual(payload["final_valid_tasks"], 8)
            calls.clear()
            second = collect_exact_population(
                contract_path=fixture.contract,
                trusted_contract_sha256=fixture.contract_sha,
                lock_path=fixture.lock,
                model_path=fixture.model,
                raw_dir=fixture.raw,
                summary_dir=fixture.summary,
                executables={"sumo": fixture.executable},
                run_task=runner,
                runner_sha256="a" * 64,
            )
            self.assertEqual(calls, [])
            self.assertNotEqual(summary, second)
            bad = next(fixture.raw.glob("*.json"))
            bad.write_text("{", encoding="utf-8")
            with self.assertRaises(IntegrityError):
                collect_exact_population(
                    contract_path=fixture.contract,
                    trusted_contract_sha256=fixture.contract_sha,
                    lock_path=fixture.lock,
                    model_path=fixture.model,
                    raw_dir=fixture.raw,
                    summary_dir=fixture.summary,
                    executables={"sumo": fixture.executable},
                    run_task=runner,
                    runner_sha256="a" * 64,
                )

    def test_collector_rejects_wrong_executable_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = make_fixture(Path(tmp), populate_raw=False)
            fixture.executable.write_bytes(b"different")
            with self.assertRaises(IntegrityError):
                collect_exact_population(
                    contract_path=fixture.contract,
                    trusted_contract_sha256=fixture.contract_sha,
                    lock_path=fixture.lock,
                    model_path=fixture.model,
                    raw_dir=fixture.raw,
                    summary_dir=fixture.summary,
                    executables={"sumo": fixture.executable},
                    run_task=_record,
                    runner_sha256="a" * 64,
                )


if __name__ == "__main__":
    unittest.main()
