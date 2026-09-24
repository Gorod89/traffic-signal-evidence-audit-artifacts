from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "supplementary" / "incumbent_invariance"
sys.path.insert(0, str(MODULE_DIR))

import parallelism_control  # noqa: E402


CANONICAL = MODULE_DIR / "results" / "seed_sensitivity.json"
CONTROL = MODULE_DIR / "results" / "parallelism_control.json"


class ParallelismControlTests(unittest.TestCase):
    def test_retained_control_matches_all_canonical_scalars_exactly(self) -> None:
        canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
        control = json.loads(CONTROL.read_text(encoding="utf-8"))
        comparison = parallelism_control.compare_reports(
            canonical, control["parallel_run"]
        )

        self.assertEqual(comparison, control["comparison"])
        self.assertEqual(comparison["n_seeds"], 40)
        self.assertEqual(comparison["scalar_comparisons"], 200)
        self.assertEqual(comparison["mismatch_count"], 0)
        self.assertEqual(comparison["max_abs_delta"], 0.0)
        self.assertEqual(control["canonical_execution"], {
            "estimator_jobs": 1,
            "thread_limit": 1,
        })
        self.assertEqual(control["parallel_execution"], {
            "estimator_jobs": -1,
            "thread_limit": None,
        })
        self.assertEqual(
            control["canonical_result_sha256"],
            hashlib.sha256(CANONICAL.read_bytes()).hexdigest(),
        )

    def test_comparison_detects_one_perturbed_scalar(self) -> None:
        canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
        perturbed = copy.deepcopy(canonical)
        perturbed["per_seed"][0]["factor"] += 0.25

        comparison = parallelism_control.compare_reports(canonical, perturbed)

        self.assertEqual(comparison["mismatch_count"], 1)
        self.assertEqual(comparison["mismatches"][0]["field"], "factor")
        self.assertAlmostEqual(comparison["max_abs_delta"], 0.25)

    def test_noncanonical_controls_cannot_replace_canonical_reference(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(MODULE_DIR / "seed_sensitivity.py"),
                "--update-reference",
                "--estimator-jobs",
                "-1",
                "--no-thread-limit",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("canonical reference requires", completed.stderr)


if __name__ == "__main__":
    unittest.main()
