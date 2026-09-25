# Public Tooling Corrections

This document accompanies the corrected tools in the curated public research
compendium. The private historical source tree was not modified. These tools
are a post-hoc hardened release, not evidence that the historical run was
already fail-closed.

## 1. Authority Activity Ratio

### Historical defect

The historical `v10_proposal/aar/aar.py` returned a pooled point estimate at
lines 138--151, but `_cluster_bootstrap` at lines 81--102 bootstrapped the
equal-weight mean of per-cluster rates. For V3 this paired:

- pooled point: `70 / 60079 = 0.0011651325754423343` (`0.116513%`);
- equal-weight cluster-mean interval: approximately `[0.099%, 0.192%]`.

Those are different estimands because the 60 clusters contain 620--1621 rows.
The previously untraced `0.144%` is the point estimate of the second estimand:

```text
(1 / 60) * sum_c(active_c / opportunities_c)
= 0.0014353503032949763
= 0.143535%, rounded to 0.144%.
```

### Correction

`src/evidence_tools/aar.py` reports two explicitly named quantities:

1. `pooled`: total active decisions divided by total opportunities; and
2. `equal_weight_cluster_mean`: the mean of within-cluster rates.

Each bootstrap draw resamples whole clusters once, then computes both
estimands from that same draw. A point estimate is therefore never paired with
the other estimand's interval. The archived V3 regression gives:

- pooled AAR `0.116513%`, pooled 95% CI `[0.080009%, 0.160006%]`;
- cluster-mean AAR `0.143535%`, cluster-mean 95% CI
  `[0.100412%, 0.192574%]` under the same 4000 draws and seed `20260813`.

The API requires a textual `scope` and `opportunity_unit`; it does not pretend
that V1 all-junction-step, V3 post-event-gate, and V4 differing-proposal
denominators are interchangeable. Eligibility is a true row mask, missing
actions are errors, composite cluster keys remain tuples, and JSON booleans are
parsed strictly.

The former fixed `VACUOUS_THRESHOLD`, unconditional
`non_inferiority_informative` flag, and decision-level minimum-detectable-effect
formula are removed. A zero-AAR identity diagnostic is emitted only when the
caller explicitly affirms all three assumptions used in the manuscript:
matched initial conditions, a shared stochastic realisation, and no other
advisor-to-environment channel. AAR itself remains an activity measure, not a
safety, benefit, or power statistic.

## 2. Sealed Evaluator

### Historical defects

The surviving `v10_proposal/phase0/evaluate_sealed.py`:

- globbed every `raw/*.json` and silently skipped incomplete records
  (lines 63--70);
- hard-coded `scheduled = 1800` without checking the exact Cartesian set
  (line 72);
- did not detect missing, unexpected, or duplicate logical keys, filename and
  record disagreement, or the six-condition closure of each cluster;
- recomputed the training mean without checking the locked training-source hash
  (lines 90--93);
- verified the model hash, but loaded/scored before a complete population
  integrity receipt existed;
- used `os.replace` through `write_json_atomic`, so a score could overwrite a
  previous score at the same path.

The first score path was created with 1797 records. Three
`ingolstadt21|9900` records arrived later, and the path was overwritten with the
1800-record result. The surviving evaluator was itself modified after the
stored result. These facts cannot be repaired retrospectively.

### Correction

`src/evidence_tools/sealed.py` requires an out-of-band SHA-256 for the reviewed
population contract and then, before `pickle.loads` or `predict`, verifies:

- the byte-exact historical prediction-lock SHA;
- the frozen-model SHA;
- the locked training-source SHA and its row/cluster counts;
- the exact `3 networks x 6 conditions x 100 seeds x 1 snapshot = 1800`
  Cartesian population;
- unique logical keys, filename/content identity, schema version, split,
  `status=complete`, 81 finite features, and a finite target;
- exactly 300 `(network, seed)` analysis clusters with six conditions each;
- absence of unexpected files, temporary files, and subdirectories.

All bytes used for parsing/scoring are the same bytes that are hashed, avoiding
a hash-then-reread time-of-check/time-of-use gap. The result filename is keyed
by the combined lock, contract, raw-manifest, model, and training-source hashes.
The write-once primitive refuses to replace different bytes.

The historical contract is deliberately labelled:

> post-hoc hardened reanalysis of an internally locked test with an interim look

On the complete archive, the corrected evaluator exactly reproduces:

- 1800 rows, 300 clusters, six rows per cluster;
- RMSE gain `0.13992387139559503`;
- 95% CI `[0.10870764488427018, 0.17073282757063235]`;
- AUROC `0.7622089314194578`, 95% CI
  `[0.7389476539306005, 0.7870855100555321]`.

It refuses the reconstructed 1797-row interim directory before loading the
model. That does not erase the historical interim look; it prevents the same
failure mode in a new run.

## 3. Collector

The collection core in `src/evidence_tools/collector.py` has no `--smoke`,
`--limit`, or shortened-seed mode for the reserved block. Resume validates each
existing JSON record rather than trusting its filename. It verifies every
executable named by the contract (SUMO for this study), the trusted runner hash,
the lock and model, collects all and only missing keys, and writes a summary only
after the exact-set validator closes the population.

`scripts/collect_sealed_v6.py` is the V6 adapter. It additionally requires an
out-of-band hash of a V6 source manifest. It is intentionally described as a
hardened replacement: the current historical `collect_sealed.py` was created
after the recorded collection and cannot establish what code performed that
collection.

## 4. Immutable Evidence Outputs

`src/evidence_tools/integrity.py` distinguishes atomicity from immutability.
Evidence files are written to a flushed temporary file and published through an
exclusive hard link. An identical existing result is idempotent; different
bytes at the same path raise `ImmutableOutputError`. JSON duplicate keys,
`NaN`, and infinity are rejected.

The frozen estimator is a pickle and can execute code during loading. Use only
the artifact with locked SHA-256
`e4f2dea4100555241d0c17e8564e4c78189dd05eb8fd0ad356ccad6aa64302dc`
under the separately reviewed contract. Never pass an untrusted lock/pickle
pair merely because the pair is internally self-consistent.

## 5. Tests

Unit and synthetic integration tests:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src)
C:\path\to\python.exe -m unittest discover -s .\tests -v
```

Archived-artifact regression (adds V1/V3 and the 1800/1797 sealed checks):

```powershell
$env:ARTICLE_SOURCE_ROOT = '<historical-artifact-root>'
C:\path\to\python.exe -m unittest discover -s .\tests -v
```

The reviewed historical population-contract digest is:

```text
bc3b8b7ecd3de42d1e04bae3dcfe0e1e27ddc16eeeb663cd5bd59c37002d8199
```

Example scoring invocation:

```powershell
evaluate-sealed `
  --lock <exact-historical-PREDICTION_LOCK.json> `
  --contract artifacts/protocols/sealed_population_contract.historical-v1.json `
  --trusted-contract-sha256 bc3b8b7ecd3de42d1e04bae3dcfe0e1e27ddc16eeeb663cd5bd59c37002d8199 `
  --raw-dir <sealed/raw> `
  --model <sealed/frozen_estimator.pkl> `
  --repository-root <artifact-root> `
  --output-dir <new-empty-results-directory>
```

The repository carries the byte-exact historical lock at
`artifacts/protocols/PREDICTION_LOCK.json`; its SHA-256 is
`fd328c7a32057134fdb3c42e75a908de6d7b1b81796b8958fcd841dc9b9633cf`.
The adjacent digest file permits a byte-identity check without normalising line
endings.

## 6. Git Repository and External Deposit Boundary

This Git repository includes:

- this package, tests, and the reviewed population contract;
- the byte-exact prediction lock and its digest;
- the pinned `artifacts/derived/v6/paired_branches.csv` derivative used by the paper;
- the reconstructed interim/final audit, explicitly labelled retrospective;
- retained supplementary analysis scripts and selected derived outputs;
- licences, `CITATION.cff`, an artifact map, and a release manifest.

Article source, article PDF, bibliography, journal template files, and journal
logos are intentionally excluded.

The external DOI deposit must additionally carry the large or licence-sensitive
materials required for full numerical reproduction: the frozen estimator and
its digest, all 1{,}800 sealed raw records and their filename/content manifest,
the V1 and V3 decision traces, the V2 run-level summary, the exact V6
source/config/model/network bundle, an environment record, the SUMO executable
digest, and a claim-to-artifact map. The executable itself is redistributable
only where its licence permits. The pickle warning in Section 4 applies to the
external model artifact.

Do not publish virtual environments, caches, failed scratch runs, reviewer
working directories, local paths, credentials, or stale diagnosis/provenance
notes. The existing `v10_proposal/README.md` and `aar/REPORTING_STANDARD.md`
must not be released unchanged: they contain superseded statements and a stale
V1 row count.

## 7. Claim-Origin Labels and Historical Source Identity

The active public checker again uses the historical origin class `agent` for
values reported by an inventory subagent, including the D+ checks. Earlier
public-release commits replaced this label with `inventory` without recording
the change clearly enough. Making that substitution without recording it
obscured provenance, even though expected values, source paths, tolerances and
numerical logic did not change. The retained historical checker preserves its
original labels byte-for-byte under `artifacts/historical_scripts/`; the active
checker additionally provides explicit public/full scopes and a non-zero exit
status for mismatches. The release checker no longer supports hidden
SHA-256-encoded product-name filters. The repository instead discloses the
assistive tools used in preparing the manuscript and maintaining this package
in `docs/AI_USE_DISCLOSURE.md`.

Several other historical programs and evidence objects had also been
normalised to LF by Git attributes. Three released evidence files have been
restored byte-for-byte and marked `-text`; the original sealed registration,
collection and evaluation programs and the unmodified Phase-2 common module
are retained in the same non-normalising historical directory. The active
Phase-2 module remains the portable path adaptation. `docs/PROVENANCE.csv`
records which object is byte-exact and which was transformed, including five
Phase-0 JSON files kept under their historical filenames but regenerated or
path-normalised during review.

## 8. Fail-Closed Analysis Utilities

The earlier D+ extractor wrote both committed CSVs before it had established
that any trial population or promotion record existed. A missing input could
therefore truncate both references and then raise an exception. The corrected
interface validates the 94 JSON records, required fields, rungs and promotion
record first, builds both tables in memory, and publishes them atomically below
`build/` by default. Replacing the released references requires the explicit
`--update-reference` flag.

The claim audit previously printed five mismatches but returned process status
zero, and optional `if artifact:` branches could omit checks silently. Its
public and full-archive denominators are now fixed at 98 and 141 checks,
respectively; the artifact map and provenance register likewise assert 17 and
15 rows. Missing, structurally empty, mismatched or reduced inputs return a
non-zero status. Full-archive mode requires an explicit source root.
Continuous integration runs the public mode.

## 9. Tree-Seed and Toolchain Sensitivity

The first forty-seed JSON was generated under Python 3.13 and
scikit-learn 1.7.2, whereas the repository declared Python 3.12 and later pinned
scikit-learn 1.9.0. Because the reported overstatement factor divides by a
history increment close to zero, this was not a harmless packaging difference.
The old result is retained as
`supplementary/incumbent_invariance/results/seed_sensitivity_py313_sklearn172.json`,
a non-canonical toolchain-sensitivity record. The canonical manuscript
reference is
`supplementary/incumbent_invariance/results/seed_sensitivity.json`.

The released reference is regenerated in the canonical CPython 3.12.14 lock,
with `n_jobs=1` and a one-thread numerical thread pool. The script records the
interpreter, package versions, platform and thread-pool state rather than a
hard-coded environment sentence. Like the D+ extractor, a normal run writes a
candidate below `build/`; changing the reference requires
`--update-reference`.

The seed script now exposes `--estimator-jobs`, `--thread-limit` and
`--no-thread-limit` while refusing to replace the canonical reference under
non-canonical controls. `parallelism_control.py` uses that interface to rerun
all 40 seeds with `n_jobs=-1` and no explicit numerical thread limit in the
canonical toolchain. The retained control records 200 exact scalar comparisons,
zero mismatches and maximum absolute difference 0, along with the parallel
runtime inventory and per-seed values.

## 10. Decision-Denominator Source Verification

The first published denominator register contained two incorrect hexadecimal
characters in the recorded SHA-256 of `v4_run_level_totals.json`. The corrected digest is
`5249437e827382c7ca8ae4cac066ef4f79910f1e0bbdd4451d8d3b0b7300900a`;
the other three recorded source digests were independently rechecked and
already matched their private source bytes.

The public checker originally compared the register with constants embedded in
the checker. That detects later alteration of the released aggregate but cannot
authenticate an omitted primary source. The corrected checker retains that
public integrity check and adds `--source-root` (or `ARTICLE_SOURCE_ROOT`) to
hash all four authorised private sources directly. The released V4 aggregate
also prints the complete decision-path accounting identity
`199673 + 80303 + 824 = 280800`.
