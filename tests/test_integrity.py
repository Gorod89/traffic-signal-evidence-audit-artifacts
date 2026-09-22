from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evidence_tools.integrity import (
    ImmutableOutputError,
    IntegrityError,
    loads_json_strict,
    verify_sha256,
    write_bytes_once,
)


class IntegrityTests(unittest.TestCase):
    def test_write_once_is_idempotent_but_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            self.assertTrue(write_bytes_once(path, b"first"))
            self.assertFalse(write_bytes_once(path, b"first"))
            with self.assertRaises(ImmutableOutputError):
                write_bytes_once(path, b"second")
            self.assertEqual(path.read_bytes(), b"first")

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_values(self) -> None:
        with self.assertRaises(IntegrityError):
            loads_json_strict('{"x": 1, "x": 2}')
        with self.assertRaises(IntegrityError):
            loads_json_strict('{"x": NaN}')
        with self.assertRaises(IntegrityError):
            loads_json_strict('{"x": Infinity}')

    def test_expected_digest_must_be_well_formed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x"
            path.write_bytes(b"x")
            with self.assertRaises(IntegrityError):
                verify_sha256(path, "not-a-digest", label="fixture")


if __name__ == "__main__":
    unittest.main()
