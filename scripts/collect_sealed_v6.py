"""Collect the exact historical sealed grid through the V6 paired-branch path.

This is a hardened, post-hoc public replacement.  It is not represented as the
exact collector executed in August 2026: the surviving historical collector was
created after the recorded collection.  No smoke, limit, or seed-range override
is provided for the reserved block.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evidence_tools.collector import collect_exact_population  # noqa: E402
from evidence_tools.integrity import (  # noqa: E402
    IntegrityError,
    loads_json_strict,
    read_verified_bytes,
    sha256_file,
)
from evidence_tools.sealed import PopulationKey  # noqa: E402


class V6Runner:
    def __init__(self, source_root: Path, sumo_binary: Path) -> None:
        self.source_root = source_root.resolve()
        self.sumo_binary = sumo_binary.resolve()
        self._factory = None
        self._ensemble = None
        self._policy = None
        self._params: dict[str, Any] = {}

    def _initialize(self) -> None:
        if self._factory is not None:
            return
        v6 = self.source_root / "v6"
        sys.path.insert(0, str(v6 / "src"))
        from ergs_tsc_v6.multi_llm import load_policy_provider
        from ergs_tsc_v6.ppo import EnsemblePolicy
        from ergs_tsc_v6.study import EnvironmentFactory, load_study

        root, payload = load_study(v6 / "configs" / "study.yaml")
        study = payload["study"]
        os.environ["SUMO_HOME"] = str(self.sumo_binary.parents[1])
        self._factory = EnvironmentFactory(
            root, study, payload["ood"], str(self.sumo_binary), extended_metrics=False
        )
        checkpoints = [
            root / "models" / f"ppo_seed{seed}.pth"
            for seed in study["training_replicates"]
        ]
        self._ensemble = EnsemblePolicy.load(checkpoints)
        panel = root / "models" / "multi_llm_policy_panel.json"
        artifact = panel if panel.exists() else root / "models" / "lightgpt_policy_table.json"
        self._policy = load_policy_provider(artifact)
        self._params = dict(payload["cfra"]["objective"])

    def __call__(self, key: PopulationKey) -> dict[str, Any]:
        self._initialize()
        from ergs_tsc_v6.cfra_branching import run_paired_branch

        result = run_paired_branch(
            self._factory,
            self._ensemble,
            self._policy,
            key.network,
            key.condition,
            key.seed,
            key.snapshot_index,
            "sealed_probe",
            horizon_decisions=int(self._params["horizon_decisions"]),
        )
        if not isinstance(result, dict):
            raise IntegrityError(f"V6 runner returned {type(result).__name__}, expected dict")
        return result


def verify_source_manifest(
    source_root: Path, manifest_path: Path, trusted_digest: str
) -> str:
    payload_bytes, digest = read_verified_bytes(
        manifest_path, trusted_digest, label="trusted V6 source manifest"
    )
    manifest = loads_json_strict(
        payload_bytes.decode("utf-8"), label=str(manifest_path)
    )
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise IntegrityError("V6 source manifest must contain a files object")
    root = source_root.resolve()
    for relative, expected in manifest["files"].items():
        lexical = root / relative
        resolved = lexical.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise IntegrityError(f"V6 source manifest path escapes root: {relative}") from exc
        read_verified_bytes(resolved, expected, label=f"V6 source {relative}")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--trusted-contract-sha256", required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sumo", type=Path, required=True)
    parser.add_argument("--v6-source-manifest", type=Path, required=True)
    parser.add_argument("--trusted-v6-source-manifest-sha256", required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, required=True)
    args = parser.parse_args()

    runner_manifest_sha = verify_source_manifest(
        args.source_root,
        args.v6_source_manifest,
        args.trusted_v6_source_manifest_sha256,
    )
    adapter_sha = sha256_file(Path(__file__))
    runner_identity_sha = hashlib.sha256(
        f"{runner_manifest_sha}\0{adapter_sha}".encode("ascii")
    ).hexdigest()
    runner = V6Runner(args.source_root, args.sumo)
    summary = collect_exact_population(
        contract_path=args.contract,
        trusted_contract_sha256=args.trusted_contract_sha256,
        lock_path=args.lock,
        model_path=args.model,
        raw_dir=args.raw_dir,
        summary_dir=args.summary_dir,
        executables={"sumo": args.sumo},
        run_task=runner,
        runner_sha256=runner_identity_sha,
    )
    print(summary)


if __name__ == "__main__":
    main()
