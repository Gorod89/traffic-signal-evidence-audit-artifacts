from __future__ import annotations

import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from evidence_tools.aar import authority_activity_ratio, main
from evidence_tools.integrity import ImmutableOutputError, IntegrityError


class AARTests(unittest.TestCase):
    def test_unequal_clusters_keep_pooled_and_cluster_mean_separate(self) -> None:
        executed = [1] + [0] * 297
        incumbent = [0] * 298
        clusters = ["small"] + ["large-1"] * 99 + ["large-2"] * 99 + ["large-3"] * 99
        report = authority_activity_ratio(
            executed,
            incumbent,
            clusters,
            scope="synthetic unequal clusters",
            opportunity_unit="row",
            bootstrap_replicates=200,
            bootstrap_seed=7,
        )
        self.assertAlmostEqual(report.pooled.estimate, 1 / 298)
        self.assertAlmostEqual(report.equal_weight_cluster_mean.estimate, 0.25)
        self.assertNotEqual(report.pooled.ci95, report.equal_weight_cluster_mean.ci95)

    def test_equal_cluster_sizes_make_estimands_coincide(self) -> None:
        report = authority_activity_ratio(
            [1, 0, 0, 1],
            [0, 0, 0, 0],
            ["a", "a", "b", "b"],
            scope="equal clusters",
            opportunity_unit="row",
            bootstrap_replicates=50,
        )
        self.assertEqual(report.pooled.estimate, 0.5)
        self.assertEqual(report.equal_weight_cluster_mean.estimate, 0.5)

    def test_composite_cluster_keys_do_not_collide(self) -> None:
        report = authority_activity_ratio(
            [0, 1],
            [0, 0],
            [("a|b", "c"), ("a", "b|c")],
            scope="tuple keys",
            opportunity_unit="row",
            bootstrap_replicates=20,
        )
        self.assertEqual(report.clusters, 2)

    def test_identity_diagnostic_requires_all_assumptions(self) -> None:
        report = authority_activity_ratio(
            [0, 0],
            [0, 0],
            ["a", "b"],
            scope="zero activity",
            opportunity_unit="row",
            bootstrap_replicates=20,
        )
        self.assertIn("cannot be inferred", report.identity_diagnostic)
        report = authority_activity_ratio(
            [0, 0],
            [0, 0],
            ["a", "b"],
            scope="zero activity",
            opportunity_unit="row",
            bootstrap_replicates=20,
            assumptions={
                "same_initial_conditions": True,
                "shared_stochastic_realisation": True,
                "no_other_advisor_channel": True,
            },
        )
        self.assertIn("identical", report.identity_diagnostic)

    def test_assumptions_and_eligibility_are_strict(self) -> None:
        with self.assertRaises(ValueError):
            authority_activity_ratio(
                [0],
                [0],
                ["a"],
                scope="bad assumption",
                opportunity_unit="row",
                assumptions={"same_initial_conditions": "false"},
            )
        with self.assertRaises(ValueError):
            authority_activity_ratio(
                [1, 0],
                [0, 0],
                ["a", "b"],
                [False, True],
                scope="outside eligible",
                opportunity_unit="row",
                bootstrap_replicates=20,
            )
        with self.assertRaises(ValueError):
            authority_activity_ratio(
                [0, 0],
                [0, 0],
                ["a", "b"],
                [False, False],
                scope="zero eligible",
                opportunity_unit="row",
                bootstrap_replicates=20,
            )

    def test_invalid_inputs_fail(self) -> None:
        common = {"scope": "invalid", "opportunity_unit": "row"}
        with self.assertRaises(ValueError):
            authority_activity_ratio([], [], [], **common)
        with self.assertRaises(ValueError):
            authority_activity_ratio([0], [0, 1], ["a"], **common)
        with self.assertRaises(ValueError):
            authority_activity_ratio([math.nan], [0], ["a"], **common)
        with self.assertRaises(ValueError):
            authority_activity_ratio([0], [0], ["a"], bootstrap_replicates=0, **common)
        with self.assertRaises(ValueError):
            authority_activity_ratio([0], [0], ["a"], confidence_level=0.9, **common)

    def test_cli_parses_false_strictly_and_output_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "log.json"
            out = root / "report.json"
            log.write_text(
                json.dumps(
                    [
                        {"action": 0, "baseline": 0, "network": "n", "seed": 1, "ok": "false"},
                        {"action": 1, "baseline": 0, "network": "n", "seed": 2, "ok": "true"},
                    ]
                ),
                encoding="utf-8",
            )
            args = [
                str(log),
                "--incumbent",
                "baseline",
                "--cluster",
                "network",
                "seed",
                "--eligible",
                "ok",
                "--scope",
                "fixture",
                "--opportunity-unit",
                "row",
                "--bootstrap-replicates",
                "20",
                "--output",
                str(out),
            ]
            with redirect_stdout(StringIO()):
                main(args)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["eligible_decisions"], 1)
            with redirect_stdout(StringIO()):
                main(args)  # same bytes are idempotent
            log.write_text(log.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaises(ImmutableOutputError):
                with redirect_stdout(StringIO()):
                    main(args)

    def test_cli_rejects_missing_fields_and_duplicate_json_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing.json"
            missing.write_text('[{"action": 0, "network": "n", "seed": 1}]', encoding="utf-8")
            args = [
                str(missing), "--incumbent", "baseline", "--cluster", "network", "seed",
                "--scope", "fixture", "--opportunity-unit", "row",
            ]
            with self.assertRaises(IntegrityError):
                main(args)
            duplicate = root / "duplicate.json"
            duplicate.write_text('[{"action": 0, "action": 1, "baseline": 0, "network": "n", "seed": 1}]', encoding="utf-8")
            args[0] = str(duplicate)
            with self.assertRaises(IntegrityError):
                main(args)


if __name__ == "__main__":
    unittest.main()
