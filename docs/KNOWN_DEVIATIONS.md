# Known deviations and integrity limits

This document prevents corrected release code from being confused with historical execution.

## Sealed evaluation

- The first stored evaluation was created when 1,797 of 1,800 expected rows were present.
- Three `ingolstadt21`, seed `9900` rows arrived later.
- The result path was overwritten after the block became complete; the first JSON was not retained.
- The historical evaluator checked the frozen model hash but did not fail closed on the exact Cartesian key set, duplicate or unexpected keys, six conditions per cluster, or the locked training-source hash.
- The evaluator file now present in the working archive is newer than the stored result, and the source tree was not version-controlled at that point. Its exact execution identity is therefore not proved.

Any corrected evaluator in this repository is prospective tooling. It must not be described as the exact program that produced the August 2026 historical result.

## Interval and conformal claims

- The original AAR tool paired a pooled point estimate with a bootstrap interval for a different, equal-weight cluster estimand.
- Corrected code reports pooled and cluster-mean estimands separately and recomputes numerator and denominator within every resampled cluster draw.
- The post-opening conformal analysis reproduced its reported margin, but the nominal simultaneous coverage claim did not hold on the sealed population. The manuscript treats that analysis as invalid for a formal guarantee.

## Missing provenance

The exact historical package lock, the executable hash of every collector/evaluator run, an append-only access log, and the overwritten first result are unavailable. No later reconstruction can recreate those missing primary records. The repository documents the gap instead of silently backfilling it.
