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
| Forty-tree-seed sensitivity | yes | yes | yes: V6 paired derivative | rerunnable |

The machine-readable version of this table is [`docs/ARTIFACT_MAP.csv`](docs/ARTIFACT_MAP.csv).

## Layout

- `src/evidence_tools/` — corrected activity-ratio, collector, integrity, and sealed-population tools.
- `tests/` — regression tests for the corrected tools.
- `scripts/analysis/` — portable Phase-0 reanalysis and figure-data scripts.
- `scripts/audit/` — release, provenance, and manifest checks.
- `artifacts/derived/` — selected derived tables and JSON outputs, including the V6 paired derivative.
- `artifacts/protocols/` — preserved prediction-lock and population-contract records.
- `artifacts/audit/` — post-opening integrity audit output.
- `supplementary/` — Phase 1, Phase 2, positive-control, D+, incumbent-invariance, and tree-seed modules and retained outputs.
- `docs/` — exact inclusion boundary, licensing notes, known deviations, and reproduction instructions.

## Deliberate omissions

The following are not distributed here:

- article source, article PDF, bibliography, journal class/style files, and journal logos;
- raw paired-branch and per-decision traces;
- the frozen estimator and model weights;
- road-network and route files;
- the complete historical V1-V8 source snapshots, pending file-level licensing review;
- the complete V8 trial-JSON population;
- local review notes, prompts, caches, environments, telemetry, and unrelated follow-up studies.

See [`docs/ARTIFACT_SCOPE.md`](docs/ARTIFACT_SCOPE.md) and [`docs/DATA_LICENSES.md`](docs/DATA_LICENSES.md) for the reasons and access boundary.

## Verify the repository

```powershell
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
python scripts/audit/check_aar_artifact.py
python scripts/audit/check_release.py
python scripts/audit/build_manifest.py --check
```

The repository-only suite skips tests that require private raw artifacts. Set `ARTICLE_SOURCE_ROOT` to an authorised local archive to enable those checks. Absence is reported as unavailable; it must not be read as a successful reproduction.

To rerun the forty-tree-seed sensitivity from the committed V6 derivative:

```powershell
python supplementary/incumbent_invariance/seed_sensitivity.py
```

To recompute the positive-control summary from the retained residual array:

```powershell
python supplementary/positive_control/analyze_scaled_final.py
```

## Integrity and citation

`MANIFEST.sha256` covers every tracked release file other than the manifest itself. Continuous integration rejects manuscript/template material, private paths, placeholder metadata, assistant-brand provenance, and secret-like values.

Citation metadata are in [`CITATION.cff`](CITATION.cff).

Mikhail Gorodnichev<br>
Faculty of Information Technology, Moscow Technical University of Communication and Informatics<br>
Moscow 111024, Russia<br>
m.g.gorodnichev@mtuci.ru

## Licensing

The root MIT license applies only to the author-owned software scopes named in [`LICENSES.md`](LICENSES.md). Derived research records are provided for verification and retain the provenance and license statements attached to them. No road-network, route, model-weight, or journal-template rights are granted here.
