from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ClaimAuditCliTests(unittest.TestCase):
    def test_public_scope_succeeds(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/analysis/audit_claims.py", "--scope", "public"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("scope=public", completed.stdout)

    def test_all_scope_fails_closed_when_archive_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/analysis/audit_claims.py",
                    "--scope",
                    "all",
                    "--source-root",
                    tmp,
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("MISMATCH", completed.stdout)
        self.assertIn("required archive source exists", completed.stdout)


class DecisionDenominatorCliTests(unittest.TestCase):
    def test_public_aggregate_check_succeeds_without_private_sources(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/analysis/check_decision_denominators.py"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("source bytes not supplied", completed.stdout)

    def test_source_verification_fails_closed_when_archive_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/analysis/check_decision_denominators.py",
                    "--source-root",
                    tmp,
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("private source missing", completed.stdout)


class ExtractTrialsCliTests(unittest.TestCase):
    def test_missing_input_cannot_truncate_existing_outputs(self) -> None:
        released = [
            ROOT / "supplementary" / "dplus" / "trials_candidate_level.csv",
            ROOT / "supplementary" / "dplus" / "trials_fold_level.csv",
        ]
        released_before = {path: sha256(path) for path in released}

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "output"
            output_dir.mkdir()
            sentinels = {
                output_dir / "trials_candidate_level.csv": b"candidate sentinel\n",
                output_dir / "trials_fold_level.csv": b"fold sentinel\n",
            }
            for path, payload in sentinels.items():
                path.write_bytes(payload)

            env = os.environ.copy()
            env.pop("V8_SOURCE_ROOT", None)
            completed = subprocess.run(
                [
                    sys.executable,
                    "supplementary/dplus/extract_trials.py",
                    "--source-root",
                    str(tmp_path / "missing-v8"),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(completed.returncode, 0)
            for path, payload in sentinels.items():
                self.assertEqual(path.read_bytes(), payload)

        for path, digest in released_before.items():
            self.assertEqual(sha256(path), digest)


if __name__ == "__main__":
    unittest.main()
