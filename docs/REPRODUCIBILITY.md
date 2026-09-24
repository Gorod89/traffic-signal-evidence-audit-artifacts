# Reproducibility guide

## Inspection-only reproduction

The committed JSON, CSV, and NPZ files allow reviewers to inspect selected values used by the article and to compare future reruns against archived outputs. `MANIFEST.sha256` records the released file hashes. This inspection layer is not a claim that every result can be regenerated from this checkout.

## Numerical reproduction

Full numerical reproduction additionally requires the external trace/model bundle described in `ARTIFACT_SCOPE.md`. Place that bundle outside the Git checkout and pass its root explicitly to the relevant scripts. Scripts must never silently fall back to a developer-specific absolute path.

The Phase-0 analyses and the canonical forty-tree-seed sensitivity based only on the committed V6 derivative are runnable in place. Scripts that need V1/V3/V4 raw traces use `ARTICLE_SOURCE_ROOT`; Phase 1, Phase 2, and incumbent-invariance scripts accept `SEALED_RAW_ROOT` and `V6_PAIRED_PATH`. D+ regeneration accepts `V8_SOURCE_ROOT`. For example:

```powershell
$env:ARTICLE_SOURCE_ROOT = '<external-artifact-root>'
python scripts/analysis/dormancy_anatomy.py
python scripts/analysis/audit_claims.py --scope all --source-root '<external-artifact-root>'
$env:SEALED_RAW_ROOT = '<sealed-raw-directory>'
python supplementary/phase1/estimate_pace_ate.py
```

The repository-only claim audit is deliberately narrower and fails on every
public mismatch:

```powershell
python scripts/analysis/audit_claims.py --scope public
```

Private-archive claims are reported only by `--scope all`; their absence is not
counted as a public success or hidden inside a zero exit status.

The D+ extractor and the tree-seed sensitivity script write new candidates
below their local `build/` directories. They validate inputs before publishing
output and require the explicit `--update-reference` flag before changing a
committed reference file. This separates a normal reproduction run from a
maintainer-approved evidence update.

The historical V1-V8 source snapshots are deliberately not distributed here pending file-level licensing review. Corrected portable interfaces are under `scripts/`, `src/`, and `tests/`; retained analysis modules and outputs are under `supplementary/`. See `ARTIFACT_MAP.csv` for the support level of every released analysis group.

## Historical identity

For every rerun, retain:

- the Git commit and SHA-256 of the executable script;
- the input manifest and schema version;
- the Python and dependency lock;
- the simulator version and configuration;
- the seed registry;
- start/end timestamps and an append-only access record;
- a write-once result path.

Do not overwrite a previous result. If a population is incomplete or contains duplicate/unexpected keys, the sealed evaluator must exit before reading outcomes or fitting/scoring a model.

## Remaining environment work

`requirements-lock.txt` pins the complete CPU audit environment verified with
CPython 3.12.14; `.python-version` pins the interpreter patch release. It is the
canonical reproduction environment, not proof of the exact historical runtime.
The committed canonical forty-tree-seed output,
`supplementary/incumbent_invariance/results/seed_sensitivity.json`, was
regenerated in that environment with one estimator job and one numerical
thread. The earlier Python 3.13.2 / scikit-learn 1.7.2 output is retained as
`supplementary/incumbent_invariance/results/seed_sensitivity_py313_sklearn172.json`
because the ratio changes materially across toolchains; it is not the
manuscript reference.

GPU positive-control training remains a separate historical environment and is
not covered by the CPU lock. Every future archival release should keep that
distinction, record the SUMO version and executable digest, and attach a full
runtime inventory to newly generated results.
