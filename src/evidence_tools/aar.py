"""Authority Activity Ratio with explicit, internally consistent estimands.

The primary estimand is the pooled ratio

    sum(active decisions) / sum(decision opportunities).

When clusters have unequal sizes, this differs from the equal-weight mean of
cluster-specific rates.  This module reports both quantities separately and
bootstraps both from the same sampled clusters.  It never attaches an interval
for one estimand to the point estimate of the other.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .integrity import (
    IntegrityError,
    loads_json_strict,
    sha256_file,
    write_json_once,
)


@dataclass(frozen=True)
class PooledRate:
    estimand: str
    numerator: int
    denominator: int
    estimate: float
    ci95: tuple[float, float] | None


@dataclass(frozen=True)
class ClusterMeanRate:
    estimand: str
    clusters: int
    estimate: float
    ci95: tuple[float, float] | None


@dataclass(frozen=True)
class BootstrapSpec:
    unit: str
    replicates: int
    seed: int
    confidence_level: float
    quantile_method: str


@dataclass(frozen=True)
class AARReport:
    schema_version: str
    scope: str
    opportunity_unit: str
    decisions: int
    active_decisions: int
    eligible_decisions: int | None
    active_outside_eligible: int | None
    clusters: int
    active_clusters: int
    pooled: PooledRate
    equal_weight_cluster_mean: ClusterMeanRate | None
    pooled_among_eligible: PooledRate | None
    bootstrap: BootstrapSpec | None
    identity_diagnostic: str
    assumptions: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_bool_vector(values: Sequence[bool], *, label: str) -> np.ndarray:
    bad = [index for index, value in enumerate(values) if not isinstance(value, (bool, np.bool_))]
    if bad:
        shown = ", ".join(map(str, bad[:5]))
        raise ValueError(f"{label} must contain booleans; invalid row(s): {shown}")
    return np.asarray(values, dtype=bool)


def _cluster_counts(
    active: np.ndarray, clusters: Sequence[Any], mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    if len(clusters) != active.size:
        raise ValueError("clusters must have the same length as executed")
    grouped: dict[Any, list[int]] = {}
    for index, raw_key in enumerate(clusters):
        if raw_key is None:
            raise ValueError(f"cluster is null at row {index}")
        try:
            hash(raw_key)
        except TypeError as exc:
            raise ValueError(f"cluster is not hashable at row {index}: {raw_key!r}") from exc
        if mask is not None and not bool(mask[index]):
            continue
        counts = grouped.setdefault(raw_key, [0, 0])
        counts[0] += int(active[index])
        counts[1] += 1
    keys = sorted(grouped, key=lambda item: (type(item).__name__, repr(item)))
    numerators = np.asarray([grouped[key][0] for key in keys], dtype=np.int64)
    denominators = np.asarray([grouped[key][1] for key in keys], dtype=np.int64)
    return numerators, denominators, keys


def _bootstrap_two_estimands(
    numerators: np.ndarray,
    denominators: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence level must be between zero and one")
    if confidence_level != 0.95:
        raise ValueError("this report schema names ci95 explicitly; confidence_level must be 0.95")
    clusters = numerators.size
    if clusters < 2:
        return None, None
    rng = np.random.default_rng(seed)
    pooled_draws = np.empty(replicates, dtype=np.float64)
    cluster_draws = np.empty(replicates, dtype=np.float64)
    rates = numerators / denominators
    for index in range(replicates):
        picked = rng.integers(0, clusters, size=clusters)
        pooled_draws[index] = numerators[picked].sum() / denominators[picked].sum()
        cluster_draws[index] = rates[picked].mean()
    alpha = 1.0 - confidence_level
    quantiles = [alpha / 2.0, 1.0 - alpha / 2.0]
    pooled_q = np.quantile(pooled_draws, quantiles, method="linear")
    cluster_q = np.quantile(cluster_draws, quantiles, method="linear")
    return (
        (float(pooled_q[0]), float(pooled_q[1])),
        (float(cluster_q[0]), float(cluster_q[1])),
    )


_IDENTITY_ASSUMPTIONS = (
    "same_initial_conditions",
    "shared_stochastic_realisation",
    "no_other_advisor_channel",
)


def _normalise_action(value: Any, *, row: int, field: str) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, (str, int, float, bool)):
        raise ValueError(
            f"{field} action must be a scalar string/number/bool at row {row}, "
            f"got {type(value).__name__}"
        )
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError(f"{field} action must be finite at row {row}")
    return value


def authority_activity_ratio(
    executed: Sequence[Any],
    incumbent: Sequence[Any],
    clusters: Sequence[Any] | None = None,
    eligible: Sequence[bool] | None = None,
    *,
    scope: str,
    opportunity_unit: str,
    bootstrap_replicates: int = 4_000,
    bootstrap_seed: int = 20_260_813,
    confidence_level: float = 0.95,
    assumptions: Mapping[str, bool] | None = None,
    strict_eligibility: bool = True,
) -> AARReport:
    """Compute pooled and equal-weight-cluster AAR without mixing estimands.

    ``scope`` and ``opportunity_unit`` are mandatory because a ratio over all
    junction-steps is not comparable with a ratio over proposals that passed an
    event gate.  ``eligible`` must describe the same unfiltered row population.
    """

    if not scope.strip() or not opportunity_unit.strip():
        raise ValueError("scope and opportunity_unit must be non-empty")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    if confidence_level != 0.95:
        raise ValueError("this report schema names ci95 explicitly; confidence_level must be 0.95")
    n = len(executed)
    if n == 0:
        raise ValueError("at least one decision is required")
    if len(incumbent) != n:
        raise ValueError("executed and incumbent must have equal length")
    missing = [i for i, (left, right) in enumerate(zip(executed, incumbent)) if left is None or right is None]
    if missing:
        shown = ", ".join(map(str, missing[:5]))
        raise ValueError(f"executed/incumbent action is null at row(s): {shown}")

    normalised_executed = [
        _normalise_action(value, row=index, field="executed")
        for index, value in enumerate(executed)
    ]
    normalised_incumbent = [
        _normalise_action(value, row=index, field="incumbent")
        for index, value in enumerate(incumbent)
    ]
    active = np.asarray(
        [left != right for left, right in zip(normalised_executed, normalised_incumbent)],
        dtype=bool,
    )
    pooled_point = float(active.mean())

    if clusters is None:
        numerators = denominators = np.asarray([], dtype=np.int64)
        cluster_keys: list[Any] = []
        pooled_ci = cluster_ci = None
    else:
        numerators, denominators, cluster_keys = _cluster_counts(active, clusters)
        pooled_ci, cluster_ci = _bootstrap_two_estimands(
            numerators,
            denominators,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
            confidence_level=confidence_level,
        )

    pooled = PooledRate(
        estimand="pooled ratio: total active decisions / total opportunities",
        numerator=int(active.sum()),
        denominator=n,
        estimate=pooled_point,
        ci95=pooled_ci,
    )
    cluster_mean = None
    if cluster_keys:
        cluster_mean = ClusterMeanRate(
            estimand="equal-weight mean of within-cluster activity rates",
            clusters=len(cluster_keys),
            estimate=float(np.mean(numerators / denominators)),
            ci95=cluster_ci,
        )

    eligible_summary = None
    eligible_count: int | None = None
    outside: int | None = None
    if eligible is not None:
        if len(eligible) != n:
            raise ValueError("eligible must have the same length as executed")
        eligible_mask = _validate_bool_vector(eligible, label="eligible")
        eligible_count = int(eligible_mask.sum())
        if eligible_count == 0:
            raise ValueError("eligible was supplied but marks zero decision opportunities")
        outside = int(np.logical_and(active, ~eligible_mask).sum())
        if outside and strict_eligibility:
            raise ValueError(
                f"{outside} executed deviation(s) occur outside the declared eligible population"
            )
        eligible_active = int(np.logical_and(active, eligible_mask).sum())
        eligible_estimate = eligible_active / eligible_count
        eligible_ci = None
        if clusters is not None and eligible_count:
            e_num, e_den, _ = _cluster_counts(active, clusters, eligible_mask)
            eligible_ci, _ = _bootstrap_two_estimands(
                e_num,
                e_den,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
                confidence_level=confidence_level,
            )
        eligible_summary = PooledRate(
            estimand="pooled ratio among rows explicitly marked eligible",
            numerator=eligible_active,
            denominator=eligible_count,
            estimate=float(eligible_estimate),
            ci95=eligible_ci,
        )

    supplied_assumptions = assumptions or {}
    invalid_assumptions = {
        key: supplied_assumptions[key]
        for key in supplied_assumptions
        if key not in _IDENTITY_ASSUMPTIONS
        or not isinstance(supplied_assumptions[key], (bool, np.bool_))
    }
    if invalid_assumptions:
        raise ValueError(f"assumptions must be known boolean flags: {invalid_assumptions}")
    stated = {key: bool(supplied_assumptions.get(key, False)) for key in _IDENTITY_ASSUMPTIONS}
    if int(active.sum()) == 0 and all(stated.values()):
        diagnostic = (
            "No executed action differs from the incumbent within the stated scope. "
            "Under all three explicitly affirmed coupling assumptions, the two action "
            "streams are identical; this is an identity diagnostic, not a safety claim."
        )
    elif int(active.sum()) == 0:
        diagnostic = (
            "No executed action differs from the incumbent within the stated scope. "
            "Outcome identity cannot be inferred because one or more coupling assumptions "
            "were not affirmed."
        )
    else:
        diagnostic = (
            "The log contains executed deviations. AAR describes activity only; it does "
            "not establish benefit, safety, non-inferiority, or statistical power."
        )

    bootstrap = None
    if clusters is not None:
        bootstrap = BootstrapSpec(
            unit="whole cluster",
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
            confidence_level=confidence_level,
            quantile_method="numpy.quantile(method='linear')",
        )

    active_clusters = 0
    if cluster_keys:
        active_clusters = int((numerators > 0).sum())
    return AARReport(
        schema_version="aar-report-v2",
        scope=scope,
        opportunity_unit=opportunity_unit,
        decisions=n,
        active_decisions=int(active.sum()),
        eligible_decisions=eligible_count,
        active_outside_eligible=outside,
        clusters=len(cluster_keys),
        active_clusters=active_clusters,
        pooled=pooled,
        equal_weight_cluster_mean=cluster_mean,
        pooled_among_eligible=eligible_summary,
        bootstrap=bootstrap,
        identity_diagnostic=diagnostic,
        assumptions=stated,
    )


def _load_records(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8")
    if path.suffix.lower() == ".jsonl":
        rows = [
            loads_json_strict(line, label=f"{path}:{index}")
            for index, line in enumerate(text.splitlines(), start=1)
            if line.strip()
        ]
    else:
        payload = loads_json_strict(text, label=str(path))
        rows = payload if isinstance(payload, list) else [payload]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise IntegrityError("decision log must contain one or more JSON objects")
    return rows, digest


def _required(row: Mapping[str, Any], field: str, index: int) -> Any:
    if field not in row or row[field] is None:
        raise IntegrityError(f"missing required field {field!r} at row {index}")
    return row[field]


def _strict_bool(value: Any, *, field: str, index: int) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise IntegrityError(f"field {field!r} is not a strict boolean at row {index}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log", type=Path)
    parser.add_argument("--executed", default="action")
    parser.add_argument("--incumbent", default="baseline_action")
    parser.add_argument("--cluster", nargs="+", required=True)
    parser.add_argument("--eligible")
    parser.add_argument("--scope", required=True)
    parser.add_argument("--opportunity-unit", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=4_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_813)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--assume-same-initial-conditions", action="store_true")
    parser.add_argument("--assume-shared-stochastic-realisation", action="store_true")
    parser.add_argument("--assume-no-other-advisor-channel", action="store_true")
    args = parser.parse_args(argv)

    records, input_digest = _load_records(args.log)
    executed = [_required(row, args.executed, index) for index, row in enumerate(records)]
    incumbent = [_required(row, args.incumbent, index) for index, row in enumerate(records)]
    clusters = [
        tuple(_required(row, field, index) for field in args.cluster)
        for index, row in enumerate(records)
    ]
    eligible = None
    if args.eligible:
        eligible = [
            _strict_bool(_required(row, args.eligible, index), field=args.eligible, index=index)
            for index, row in enumerate(records)
        ]
    report = authority_activity_ratio(
        executed,
        incumbent,
        clusters,
        eligible,
        scope=args.scope,
        opportunity_unit=args.opportunity_unit,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        assumptions={
            "same_initial_conditions": args.assume_same_initial_conditions,
            "shared_stochastic_realisation": args.assume_shared_stochastic_realisation,
            "no_other_advisor_channel": args.assume_no_other_advisor_channel,
        },
    )
    payload = report.to_dict()
    payload["provenance"] = {
        "input_log_sha256": input_digest,
        "aar_module_sha256": sha256_file(Path(__file__)),
    }
    if args.output:
        write_json_once(args.output, payload)
        print(args.output)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
