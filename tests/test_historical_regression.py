from __future__ import annotations

import glob
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from evidence_tools.aar import authority_activity_ratio
from evidence_tools.integrity import sha256_file
from evidence_tools.sealed import (
    PopulationContract,
    PopulationIntegrityError,
    evaluate,
    validate_raw_population,
)


SOURCE_ROOT = os.environ.get("ARTICLE_SOURCE_ROOT")


@unittest.skipUnless(SOURCE_ROOT, "set ARTICLE_SOURCE_ROOT to run archived-artifact regression")
class HistoricalRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = Path(SOURCE_ROOT)
        cls.tool_root = Path(__file__).resolve().parents[1]
        cls.contract_path = (
            cls.tool_root
            / "artifacts"
            / "protocols"
            / "sealed_population_contract.historical-v1.json"
        )
        cls.contract_sha = sha256_file(cls.contract_path)

    def test_v1_and_v3_aar_regression(self) -> None:
        executed: list = []
        incumbent: list = []
        clusters: list = []
        pattern = self.source / "results" / "raw" / "confirmatory" / "*" / "decisions_*_ergs.json"
        for raw_path in glob.glob(str(pattern)):
            path = Path(raw_path)
            for row in json.loads(path.read_text(encoding="utf-8")):
                executed.append(row["action"])
                incumbent.append(row["baseline_action"])
                clusters.append(path.stem)
        v1 = authority_activity_ratio(
            executed,
            incumbent,
            clusters,
            scope="V1 full-method all controlled junction-steps",
            opportunity_unit="controlled junction at one decision step",
            bootstrap_replicates=4_000,
            bootstrap_seed=20_260_813,
        )
        self.assertEqual((v1.decisions, v1.active_decisions, v1.active_clusters), (72_000, 40, 5))
        self.assertEqual(v1.pooled.estimate, 40 / 72_000)
        self.assertEqual(v1.equal_weight_cluster_mean.estimate, v1.pooled.estimate)

        executed.clear()
        incumbent.clear()
        clusters.clear()
        for path in sorted((self.source / "v3" / "results" / "raw").glob("*_traces.json")):
            for row in json.loads(path.read_text(encoding="utf-8")):
                if row.get("method") != "ergs_v3":
                    continue
                executed.append(row["executed_action"])
                incumbent.append(row["baseline_action"])
                clusters.append((row["network"], row["seed"]))
        v3 = authority_activity_ratio(
            executed,
            incumbent,
            clusters,
            scope="V3 ergs_v3 proposal traces after event gate",
            opportunity_unit="proposal evaluation passed by event gate",
            bootstrap_replicates=4_000,
            bootstrap_seed=20_260_813,
        )
        self.assertEqual((v3.decisions, v3.active_decisions, v3.clusters), (60_079, 70, 60))
        self.assertAlmostEqual(v3.pooled.estimate, 0.0011651325754423343)
        self.assertEqual(
            v3.pooled.ci95,
            (0.0008000944202948882, 0.0016000621408320786),
        )
        self.assertAlmostEqual(
            v3.equal_weight_cluster_mean.estimate, 0.0014353503032949763
        )

    def test_hardened_final_sealed_regression_and_interim_rejection(self) -> None:
        phase0 = self.source / "v10_proposal" / "phase0"
        report = evaluate(
            lock_path=phase0 / "sealed" / "PREDICTION_LOCK.json",
            contract_path=self.contract_path,
            raw_dir=phase0 / "sealed" / "raw",
            model_path=phase0 / "sealed" / "frozen_estimator.pkl",
            repository_root=self.source,
            trusted_contract_sha256=self.contract_sha,
        )
        self.assertEqual(report["population"]["observed_rows"], 1_800)
        self.assertEqual(report["population"]["analysis_clusters"], 300)
        self.assertEqual(report["population"]["rows_per_cluster"], 6)
        self.assertEqual(report["primary"]["observed_rmse_gain"], 0.13992387139559503)
        self.assertEqual(
            report["primary"]["ci95"],
            [0.10870764488427018, 0.17073282757063235],
        )

        late = {
            "ingolstadt21_demand_shock_9900.json",
            "ingolstadt21_emergency_vehicle_9900.json",
            "ingolstadt21_normal_9900.json",
        }
        contract = PopulationContract.from_path(self.contract_path)
        with tempfile.TemporaryDirectory() as tmp:
            interim = Path(tmp)
            for path in (phase0 / "sealed" / "raw").glob("*.json"):
                if path.name not in late:
                    shutil.copyfile(path, interim / path.name)
            with self.assertRaises(PopulationIntegrityError):
                validate_raw_population(interim, contract, require_complete=True)


if __name__ == "__main__":
    unittest.main()
