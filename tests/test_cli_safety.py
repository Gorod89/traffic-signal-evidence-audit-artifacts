from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import textwrap
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
        self.assertIn("92/92 verified against a file on disk (scope=public)", completed.stdout)

    def test_public_scope_rejects_structurally_empty_json(self) -> None:
        probe = textwrap.dedent(
            """
            import pathlib
            import runpy
            import sys

            original_read_text = pathlib.Path.read_text

            def substituted_read_text(self, *args, **kwargs):
                if self.name == "history_increment_ci.json":
                    return "{}"
                return original_read_text(self, *args, **kwargs)

            pathlib.Path.read_text = substituted_read_text
            sys.argv = ["audit_claims.py", "--scope", "public"]
            runpy.run_path("scripts/analysis/audit_claims.py", run_name="__main__")
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("89/92 verified", completed.stdout)
        self.assertIn("MISMATCH", completed.stdout)

    def test_public_scope_rejects_a_reduced_check_denominator(self) -> None:
        probe = textwrap.dedent(
            """
            import pathlib
            import runpy
            import sys

            original_exists = pathlib.Path.exists

            def hidden_from_loader(self):
                if self.name == "history_increment_ci.json":
                    return False
                return original_exists(self)

            pathlib.Path.exists = hidden_from_loader
            sys.argv = ["audit_claims.py", "--scope", "public"]
            runpy.run_path("scripts/analysis/audit_claims.py", run_name="__main__")
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("89/89 verified", completed.stdout)
        self.assertIn("expected the fixed denominator 92", completed.stdout)

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
        env = os.environ.copy()
        env.pop("ARTICLE_SOURCE_ROOT", None)
        completed = subprocess.run(
            [sys.executable, "scripts/analysis/check_decision_denominators.py"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            env=env,
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

    def test_scheduler_query_partition_fails_closed(self) -> None:
        probe = textwrap.dedent(
            """
            import json
            import pathlib
            import runpy
            import sys

            original_read_text = pathlib.Path.read_text

            def substituted_read_text(self, *args, **kwargs):
                text = original_read_text(self, *args, **kwargs)
                if self.name == "decision_denominators.json":
                    payload = json.loads(text)
                    payload["v4_shadow"]["scheduler_queries"] = 830
                    return json.dumps(payload)
                return text

            pathlib.Path.read_text = substituted_read_text
            sys.argv = ["check_decision_denominators.py"]
            runpy.run_path(
                "scripts/analysis/check_decision_denominators.py", run_name="__main__"
            )
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("scheduler-query partition", completed.stdout)


class RegistryCardinalityCliTests(unittest.TestCase):
    def test_artifact_map_rejects_a_removed_row(self) -> None:
        source = ROOT / "docs" / "ARTIFACT_MAP.csv"
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / source.name
            lines = source.read_text(encoding="utf-8").splitlines()
            candidate.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "scripts/audit/audit_inventory.py", "--map", str(candidate)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("expected the fixed inventory 17", completed.stdout)

    def test_provenance_registry_rejects_a_removed_row(self) -> None:
        source = ROOT / "docs" / "PROVENANCE.csv"
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / source.name
            lines = source.read_text(encoding="utf-8").splitlines()
            candidate.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit/check_provenance.py",
                    "--registry",
                    str(candidate),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("expected the fixed register 15", completed.stdout)


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
