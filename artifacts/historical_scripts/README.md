# Retained historical scripts

The Python files in this directory are byte-exact retained copies. They are
kept outside the importable package so that a historical source file cannot be
mistaken for the corrected tooling in `src/evidence_tools` or for a portable
script under `scripts` and `supplementary`.

| File | SHA-256 | Retained role |
|---|---|---|
| `preregister_sealed.py` | `d634ac2fa8aecbf269c96883ef76369fcf800844af6a0072929f20b246892914` | Wrote the prediction-lock record in the retained workflow |
| `collect_sealed.py` | `ecf1e08ad81316160e60118047489e74d879ef7491fb8920b56bbaaa9068fe5d` | Retained collector source |
| `evaluate_sealed.py` | `81ba1f12771c273e3578860a939bb4bdf616a147b0b70da64ee991f6ae56b333` | Retained evaluator source cited by the integrity audit |
| `phase2_common.py` | `dc60b9a3be9af9dfb31e0facfa74ca4a8b79cb21c4f4189d5a233190be9426fa` | Retained common Phase-2 analysis source |
| `audit_claims.py` | `87e3945f2703f8c7170987b99d8d3e4b9645ca5e847b8c75ff396354d1f776ac` | Retained claim-audit source, including its original provenance labels |

These files are evidence, not entry points. Their original relative layout,
raw sealed records, frozen estimator and complete historical environment are
not distributed here, so they are not runnable from this directory.

Byte identity does not by itself establish execution identity. In particular,
the retained evaluator was modified after the first stored sealed result and
the source tree was not version-controlled at that time. The integrity audit
therefore treats it as the retained evaluator source, not as proof of the exact
bytes used for the first overwritten evaluation.

The mapping from retained sources to portable or regenerated public files is
recorded in `docs/PROVENANCE.csv`.
