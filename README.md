# Traffic-Signal Evidence Audit: Code and Derived Artifacts

Public companion repository for the analysis behind:

> Mikhail Gorodnichev, “Baselines, Sample Size, and Supervisory Activity in Guarded Reinforcement-Learning Traffic-Signal Control: A SUMO Evidence Audit.”

This repository intentionally contains no manuscript source, typeset article PDF, journal template, or `Definitions/` directory. It was created with a fresh Git history; it is not a fork or history rewrite of the private working repository.

## What this repository is

This is a curated inspection and partial-reproduction package. It contains corrected evidence tools, tests, protocol records, selected derived data, and the retained scripts and outputs for several analyses added during manuscript review. It is not a complete execution ledger and does not contain every raw simulation trace, model, intermediate array, command, or historical environment.

| Analysis group | Script | Derived output | Required input in Git | Public support |
|---|---:|---:|---:|---|
| Baseline ladder, history increment, fitting-budget and cost-weight analyses | yes | yes | yes: V6 paired derivative | rerunnable |
| Phase 1 paired contrast, Equation (4), and breakdowns | yes | yes | no: sealed raw branches omitted | inspection only |
| Phase 2 model ladder and harm sweep | yes | yes | partial: V6 derivative present; sealed block omitted | inspection only |
| Synthetic graph positive control | yes | yes | retained residual array present | summary rerunnable; training environment not bundled |
| V8 D+ re-scoring | yes | yes | no: complete trial-JSON population omitted | inspection only |
| Incumbent invariance | yes | yes | partial: sealed block omitted | partial |
| Forty-tree-seed sensitivity | yes | yes | yes: V6 paired derivative | rerunnable in the canonical Python 3.12.14 environment; the earlier Python 3.13 result is retained only as a toolchain-sensitivity record |

The machine-readable version of this table is [`docs/ARTIFACT_MAP.csv`](docs/ARTIFACT_MAP.csv).

## Layout

- `src/evidence_tools/` — corrected activity-ratio, collector, integrity, and sealed-population tools.
- `tests/` — regression tests for the corrected tools.
- `scripts/analysis/` — portable Phase-0 reanalysis and figure-data scripts.
- `scripts/audit/` — release, provenance, and manifest checks.
- `artifacts/derived/` — selected derived tables and JSON outputs, including the V6 paired derivative.
- `artifacts/protocols/` — preserved prediction-lock and population-contract records.
- `artifacts/audit/` — post-opening integrity audit output.
- `artifacts/historical_scripts/` — byte-exact historical programs retained for provenance; these are not the hardened public interfaces.
- `supplementary/` — Phase 1, Phase 2, positive-control, D+, incumbent-invariance, and tree-seed modules and retained outputs.
- `docs/` — exact inclusion boundary, file-level provenance, licensing and attribution records, known deviations, and reproduction instructions.

## Deliberate omissions

The following are not distributed here:

- article source, article PDF, bibliography, journal class/style files, and journal logos;
- raw paired-branch and per-decision traces;
- the frozen estimator and model weights;
- road-network and route files;
- the complete historical V1-V8 source snapshots; only five provenance-relevant
  historical programs are retained byte-for-byte under
  `artifacts/historical_scripts/` after file-level review;
- the complete V8 trial-JSON population;
- local review notes, prompts, caches, environments, telemetry, and unrelated follow-up studies.

See [`docs/ARTIFACT_SCOPE.md`](docs/ARTIFACT_SCOPE.md) and [`docs/DATA_LICENSES.md`](docs/DATA_LICENSES.md) for the reasons and access boundary.

## Verify the repository

```powershell
# Use CPython 3.12.14; `.python-version` records the exact patch release.
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
python scripts/audit/check_aar_artifact.py
python scripts/analysis/audit_claims.py --scope public
python scripts/analysis/check_decision_denominators.py
python scripts/audit/audit_inventory.py
python scripts/audit/check_provenance.py
python scripts/audit/check_release.py
python scripts/audit/build_manifest.py --check
```

The repository-only suite skips tests that require private raw artifacts. Set `ARTICLE_SOURCE_ROOT` to an authorised local archive to enable those checks. Absence is reported as unavailable; it must not be read as a successful reproduction.

When that archive is available, the denominator checker also validates the
four recorded source digests against the source bytes:

```powershell
python scripts/analysis/check_decision_denominators.py --source-root C:\path\to\archive
```

To rerun the forty-tree-seed sensitivity from the committed V6 derivative:

```powershell
python supplementary/incumbent_invariance/seed_sensitivity.py
```

The command writes a candidate result below `build/`; it never overwrites the
committed reference. Updating that reference is a deliberate maintainer action
requiring `--update-reference`. The canonical reference was generated with one
estimator job and a one-thread numerical thread-pool limit. It is
`supplementary/incumbent_invariance/results/seed_sensitivity.json`; the retained
legacy record is
`supplementary/incumbent_invariance/results/seed_sensitivity_py313_sklearn172.json`.
The latter demonstrates that the unstable ratio also changes with the Python
and scikit-learn toolchain. To rerun the same-environment scheduling control:

```powershell
python supplementary/incumbent_invariance/parallelism_control.py
```

It compares the canonical one-job/one-thread result with `n_jobs=-1` and no
explicit numerical thread limit, then writes a candidate below `build/`. The
retained `results/parallelism_control.json` records the full parallel per-seed
result, runtime metadata and an exact 200-scalar comparison.

The D+ trial extractor follows the same rule: it validates the complete
94-trial population and promotion record before writing, emits candidates below
`build/` by default, and requires `--update-reference` to replace released CSVs.

To recompute the positive-control summary from the retained residual array:

```powershell
python supplementary/positive_control/analyze_scaled_final.py
```

## Integrity and citation

`MANIFEST.sha256` covers every tracked release file other than the manifest itself. `docs/PROVENANCE.csv` separately distinguishes byte-exact historical objects, portable adaptations, reconstructed outputs, and derived records. The two headline decision denominators are recorded with their arithmetic and source digests in `artifacts/derived/decision_denominators.json`. Continuous integration rejects manuscript/template material, private paths, placeholder metadata, unreviewed assistant provenance, and secret-like values.

Citation metadata are in [`CITATION.cff`](CITATION.cff).
No GitHub release or Zenodo DOI exists yet. [`docs/ZENODO_RELEASE.md`](docs/ZENODO_RELEASE.md) gives the controlled release procedure; no DOI should be cited until that record is published.

Mikhail Gorodnichev<br>
Faculty of Information Technology, Moscow Technical University of Communication and Informatics<br>
Moscow 111024, Russia<br>
m.g.gorodnichev@mtuci.ru

## Licensing and attribution

This is a mixed-license package. Author-owned code is MIT licensed; eligible
author-created derived research records are CC BY 4.0; upstream rights and
redistribution restrictions override those defaults. The controlling path
defaults and file-level overrides are described in [`LICENSE`](LICENSE),
[`LICENSES.md`](LICENSES.md), and the audited-transform register
[`docs/PROVENANCE.csv`](docs/PROVENANCE.csv). Source attribution for
RESCO, TAPASCologne, InTAS, OpenStreetMap and the two compact language models is
recorded in [`docs/DATA_LICENSES.md`](docs/DATA_LICENSES.md). No rights in
omitted road networks, routes, model weights, or journal-template material are
granted here.
