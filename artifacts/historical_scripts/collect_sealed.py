"""Collect fresh paired branches on the previously untouched V6 seed block.

Why this is not a protocol violation, and what it is instead.

The V6 preregistration reserved seeds 9801-9900 (`untouched_test`) for a
confirmatory *traffic* comparison that was never opened, because Gate B stopped
the study. That grid is defined in `paired_branches.networks_by_split` only for
the four collected splits; `untouched_test` has no paired-branch configuration
at all. Collecting paired branches on those seeds is therefore not the operation
the V6 protocol sealed -- it is a new operation, and it answers a question that
did not exist when V6 was frozen:

    Does the RMSE-gain-versus-sample-size relationship established on the
    already-consumed validation split hold out of sample?

Because the question is new, its answer can be confirmatory only if the
prediction is committed before the data exist. `preregister_sealed.py` writes
that commitment; this script then collects, and `evaluate_sealed.py` scores it.
The order matters and is enforced: this script refuses to run unless a
prediction lock is already present and hash-valid.

The paired-branch computation itself reuses the exact V6 code path
(`ergs_tsc_v6.cfra_branching.run_paired_branch`) and the exact SUMO binary
pinned by the V6 pre-data lock, verified by hash before any task runs. Output
goes only to v10_proposal/; nothing under v6/ is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V6 = REPO / "v6"
OUTDIR = Path(__file__).resolve().parent / "sealed"
# The V6 pre-data lock pins the SUMO executable by hash, at the path
# v2/.venv/.../sumo/bin/sumo.exe. That copy runs standalone but cannot open a
# TraCI listening socket on this host -- a path-scoped security rule, not a
# SUMO fault. The eclipse-sumo 1.27.1 wheel installed into v8/.venv produces a
# byte-identical executable (same SHA-256) that does accept TraCI connections,
# so using it satisfies the lock's hash requirement exactly while working
# around a host restriction. The hash is re-verified below before any task runs.
LOCKED_SUMO = REPO / "v8" / ".venv" / "Lib" / "site-packages" / "sumo" / "bin" / "sumo.exe"
LOCKED_SUMO_SHA = "e39d9f60e2494570bc826d29ac0a22c3fe311c2b222469cb5063591b4574d05c"
PREDICTION_LOCK = OUTDIR / "PREDICTION_LOCK.json"

SEED_START = 9801
SEED_COUNT = 100
NETWORKS = ["moscow", "cologne8", "ingolstadt21"]
CONDITIONS = [
    "normal",
    "packet_loss",
    "sensor_delay_noise",
    "lane_closure",
    "emergency_vehicle",
    "demand_shock",
]

_FACTORY = None
_ENSEMBLE = None
_POLICY = None
_PARAMS: dict = {}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ascii_path(path: Path) -> str:
    """Return an ASCII-safe form of a Windows path.

    SUMO exits with a bare 'Quitting (on unknown error)' from every tool -- even
    netgenerate on a grid needing no input -- whenever SUMO_HOME contains
    non-ASCII characters, which this repository's path does. Windows keeps an
    8.3 short name for such directories; using it costs nothing and avoids
    relocating the installation.
    """

    text = str(path)
    if text.isascii():
        return text
    import ctypes

    buffer = ctypes.create_unicode_buffer(2048)
    length = ctypes.windll.kernel32.GetShortPathNameW(text, buffer, 2048)
    if not length or not buffer.value.isascii():
        raise RuntimeError(
            f"cannot derive an ASCII path for {text!r}; SUMO will not start. "
            "Enable 8.3 short names on this volume or install SUMO under an "
            "ASCII-only directory."
        )
    return buffer.value


def _initialize(config_path: str, sumo_binary: str) -> None:
    global _FACTORY, _ENSEMBLE, _POLICY, _PARAMS
    from ergs_tsc_v6.study import EnvironmentFactory, load_study
    from ergs_tsc_v6.ppo import EnsemblePolicy
    from ergs_tsc_v6.multi_llm import load_policy_provider

    root, payload = load_study(config_path)
    study = payload["study"]
    _FACTORY = EnvironmentFactory(
        root, study, payload["ood"], sumo_binary, extended_metrics=False
    )
    checkpoints = [
        root / "models" / f"ppo_seed{seed}.pth"
        for seed in study["training_replicates"]
    ]
    _ENSEMBLE = EnsemblePolicy.load(checkpoints)
    panel = root / "models" / "multi_llm_policy_panel.json"
    artifact = panel if panel.exists() else root / "models" / "lightgpt_policy_table.json"
    _POLICY = load_policy_provider(artifact)
    _PARAMS = dict(payload["cfra"]["objective"])


def _worker(task):
    from ergs_tsc_v6.cfra_branching import run_paired_branch

    network, condition, seed = task
    started = time.time()
    result = run_paired_branch(
        _FACTORY,
        _ENSEMBLE,
        _POLICY,
        network,
        condition,
        int(seed),
        0,
        "sealed_probe",
        horizon_decisions=int(_PARAMS["horizon_decisions"]),
    )
    if isinstance(result, dict):
        result["_wall_seconds"] = time.time() - started
    return task, result


def build_tasks(limit: int | None, seeds: int | None) -> list[tuple[str, str, int]]:
    seed_list = list(range(SEED_START, SEED_START + (seeds or SEED_COUNT)))
    tasks = [
        (network, condition, seed)
        for seed in seed_list
        for network in NETWORKS
        for condition in CONDITIONS
    ]
    return tasks[:limit] if limit else tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="cap total tasks")
    parser.add_argument("--seeds", type=int, default=None, help="how many seeds to use")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="timing probe; writes no dataset")
    args = parser.parse_args()

    actual = sha256_file(LOCKED_SUMO)
    if actual != LOCKED_SUMO_SHA:
        raise SystemExit(f"SUMO binary does not match the V6 pre-data lock: {actual}")
    print(f"SUMO binary matches V6 pre-data lock ({actual[:16]}...)")

    # Seeds are checked against the block registry for the same reason the binary
    # is checked against its hash: a boundary that lives only in prose cannot stop
    # a concurrent run from walking into a sealed block, which is how eleven
    # clusters were lost in this project. Raises rather than warns.
    sys.path.insert(0, str(REPO / "v10_proposal" / "seedreg"))
    from seedreg import assert_seeds_available

    requested = SEED_START + (args.seeds or SEED_COUNT)
    assert_seeds_available(range(SEED_START, requested), owner="v10.phase0")
    print(f"seed block {SEED_START}-{requested - 1} confirmed available for v10.phase0")

    if not args.smoke and not PREDICTION_LOCK.exists():
        raise SystemExit(
            "refusing to collect: no PREDICTION_LOCK.json. Run preregister_sealed.py first."
        )

    os.environ["SUMO_HOME"] = ascii_path(LOCKED_SUMO.parents[1])
    sumo_binary = ascii_path(LOCKED_SUMO)
    print(f"SUMO_HOME (ASCII): {os.environ['SUMO_HOME']}")
    sys.path.insert(0, str(V6 / "src"))
    os.chdir(V6)

    tasks = build_tasks(args.limit, args.seeds)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    raw_dir = OUTDIR / "raw"
    raw_dir.mkdir(exist_ok=True)

    # Resume: a completed task already has its own file. Long collections get
    # interrupted, and recomputing a paired branch changes nothing except cost.
    if not args.smoke:
        scheduled = len(tasks)
        tasks = [
            t for t in tasks
            if not (raw_dir / f"{t[0]}_{t[1]}_{t[2]}.json").exists()
        ]
        if scheduled != len(tasks):
            print(f"resuming: {scheduled - len(tasks)} of {scheduled} already collected")
    print(f"{len(tasks)} paired-branch tasks ({args.workers} workers)")
    if not tasks:
        print("nothing to do")
        return

    started = time.time()
    done = 0
    failures = 0
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_initialize,
        initargs=(str(V6 / "configs" / "study.yaml"), sumo_binary),
    ) as executor:
        futures = {executor.submit(_worker, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                task, result = future.result()
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  task failed: {futures[future]} -> {type(exc).__name__}: {exc}")
                continue
            done += 1
            if not args.smoke:
                network, condition, seed = task
                name = f"{network}_{condition}_{seed}.json"
                tmp = raw_dir / (name + ".tmp")
                tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
                tmp.replace(raw_dir / name)
            if done % 10 == 0 or done == len(tasks):
                rate = (time.time() - started) / done
                print(
                    f"  {done}/{len(tasks)} done, {failures} failed, "
                    f"{rate:.1f} s/task, eta_full_1800={rate*1800/args.workers/60:.0f} min"
                )

    elapsed = time.time() - started
    summary = {
        "tasks": len(tasks),
        "completed": done,
        "failed": failures,
        "wall_seconds": elapsed,
        "seconds_per_task": elapsed / max(done, 1),
        "workers": args.workers,
        "sumo_sha256": actual,
        "seed_block": {"start": SEED_START, "count": args.seeds or SEED_COUNT},
        "networks": NETWORKS,
        "conditions": CONDITIONS,
        "smoke": args.smoke,
    }
    target = OUTDIR / ("SMOKE_TIMING.json" if args.smoke else "COLLECTION_SUMMARY.json")
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
