# Known deviations and integrity limits

This document prevents corrected release code from being confused with historical execution.

## Sealed evaluation

- The first stored evaluation was created when 1,797 of 1,800 expected rows were present.
- Three `ingolstadt21`, seed `9900` rows arrived later.
- The result path was overwritten after the block became complete; the first JSON was not retained.
- The historical evaluator checked the frozen model hash but did not fail closed on the exact Cartesian key set, duplicate or unexpected keys, six conditions per cluster, or the locked training-source hash.
- The evaluator file now present in the working archive is newer than the stored result, and the source tree was not version-controlled at that point. Its exact execution identity is therefore not proved.

Any corrected evaluator in this repository is prospective tooling. It must not be described as the exact program that produced the August 2026 historical result.

The surviving historical registration, collection and evaluation files are now
released byte-for-byte in `artifacts/historical_scripts/`. Their hashes prove
identity with the surviving private-archive copies, not identity with the
executables that actually ran; no contemporaneous version-control record closes
that evidential gap.

## Interval and conformal claims

- The original AAR tool paired a pooled point estimate with a bootstrap interval for a different, equal-weight cluster estimand.
- Corrected code reports pooled and cluster-mean estimands separately and recomputes numerator and denominator within every resampled cluster draw.
- The post-opening conformal analysis reproduced its reported margin, but the nominal simultaneous coverage claim did not hold on the sealed population. The manuscript treats that analysis as invalid for a formal guarantee.

## Missing provenance

The exact historical package lock, the executable hash of every collector/evaluator run, an append-only access log, and the overwritten first result are unavailable. No later reconstruction can recreate those missing primary records. The repository documents the gap instead of silently backfilling it.

## Review-time transformations and toolchain sensitivity

An earlier public export normalised line endings in three evidence files and
ported paths in `phase2_common.py`, so four hashes recorded by the audit no
longer described the public bytes. The three evidence files are now restored
byte-for-byte; the portable module remains active and its byte-exact historical
counterpart is stored separately. Five Phase-0 JSON records also retain their
historical filenames despite review-time recomputation or path normalisation.
Each of these nine transformed-or-restored relationships is explicit in the
current 15-entry `docs/PROVENANCE.csv`; that curated register is not a claim to
enumerate every file covered by the release manifest. Filename alone is not
evidence of byte identity.

The original forty-tree-seed record was generated under Python 3.13.2 and
scikit-learn 1.7.2, not the repository's canonical CPU environment. It is
retained as
`supplementary/incumbent_invariance/results/seed_sensitivity_py313_sklearn172.json`.
The canonical reference,
`supplementary/incumbent_invariance/results/seed_sensitivity.json`, is
regenerated under Python 3.12.14 and scikit-learn 1.9.0 with `n_jobs=1` and a
one-thread numerical thread pool. Holding those concurrency settings fixed
reproduced the canonical record, so the observed discrepancy is attributed to
the software toolchain rather than parallel scheduling. The difference is
substantive because the reported factor divides by a history increment close
to zero; the manuscript therefore reports the canonical result and treats the
legacy run only as a toolchain-sensitivity diagnostic.
