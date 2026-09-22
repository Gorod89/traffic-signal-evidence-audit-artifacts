#!/usr/bin/env python3
"""Verify the estimand-specific V3 AAR artifact used by the manuscript."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts" / "derived" / "phase0" / "aar_funnel.json"
EXPECTED_MANIFEST = (
    "f88a69bff6d57149098a135da683b2028dbfd537cc2e591550366955d30d5e8a"
)


def close(actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15):
        raise AssertionError(f"expected {expected!r}, observed {actual!r}")


def main() -> None:
    report = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    if report["schema_version"] != "aar-funnel-v2":
        raise AssertionError("unexpected AAR artifact schema")
    if report["input_manifest_sha256"] != EXPECTED_MANIFEST:
        raise AssertionError("V3 trace-manifest digest changed")
    if report["bootstrap"] != {
        "unit": "(network, seed) cluster",
        "replicates": 4000,
        "seed": 20260813,
        "confidence_level": 0.95,
    }:
        raise AssertionError("unexpected bootstrap specification")

    funnel = report["full_method_funnel"]
    pooled = funnel["aar_estimands"]["pooled"]
    cluster_mean = funnel["aar_estimands"]["equal_weight_cluster_mean"]
    if (pooled["numerator"], pooled["denominator"]) != (70, 60079):
        raise AssertionError("unexpected pooled numerator or denominator")
    close(pooled["estimate"], 0.0011651325754423343)
    close(pooled["ci95"][0], 0.0008000944202948882)
    close(pooled["ci95"][1], 0.0016000621408320786)
    if cluster_mean["clusters"] != 60:
        raise AssertionError("unexpected cluster count")
    close(cluster_mean["estimate"], 0.0014353503032949763)
    close(cluster_mean["ci95"][0], 0.001004123625174304)
    close(cluster_mean["ci95"][1], 0.0019257369711930005)
    print("AAR artifact check passed: pooled and cluster-mean estimands are separate")


if __name__ == "__main__":
    main()
