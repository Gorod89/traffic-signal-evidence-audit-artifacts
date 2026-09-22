# Reproducibility guide

## Inspection-only reproduction

The committed JSON, CSV, and NPZ files allow reviewers to inspect selected values used by the article and to compare future reruns against archived outputs. `MANIFEST.sha256` records the released file hashes. This inspection layer is not a claim that every result can be regenerated from this checkout.

## Numerical reproduction

Full numerical reproduction additionally requires the external trace/model bundle described in `ARTIFACT_SCOPE.md`. Place that bundle outside the Git checkout and pass its root explicitly to the relevant scripts. Scripts must never silently fall back to a developer-specific absolute path.

The Phase-0 analyses and the forty-tree-seed sensitivity based only on the committed V6 derivative are runnable in place. Scripts that need V1/V3/V4 raw traces use `ARTICLE_SOURCE_ROOT`; Phase 1, Phase 2, and incumbent-invariance scripts accept `SEALED_RAW_ROOT` and `V6_PAIRED_PATH`. D+ regeneration accepts `V8_SOURCE_ROOT`. For example:

```powershell
$env:ARTICLE_SOURCE_ROOT = '<external-artifact-root>'
python scripts/analysis/dormancy_anatomy.py
python scripts/analysis/audit_claims.py
$env:SEALED_RAW_ROOT = '<sealed-raw-directory>'
python supplementary/phase1/estimate_pace_ate.py
```

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

`requirements-lock.txt` pins the environment in which the corrected evidence tools and their regression suite were verified. It does not prove the exact historical package versions. Before archival release, retain this corrected-tool lock separately and add a reconstructed historical environment record that explicitly labels its evidential status and records SUMO, Python, NumPy, pandas, scikit-learn, and Matplotlib versions.
