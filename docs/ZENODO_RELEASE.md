# Zenodo deposit and correction record

## Published records

Zenodo published three versioned records on 24--25 September 2026:

| Release | Version DOI | Record |
|---|---|---|
| `v1.0.0` | `10.5281/zenodo.22938827` | <https://zenodo.org/records/22938827> |
| `v1.0.1` | `10.5281/zenodo.22938960` | <https://zenodo.org/records/22938960> |
| `v1.0.2` | `10.5281/zenodo.22960291` | <https://zenodo.org/records/22960291> |

The concept DOI `10.5281/zenodo.22938826` represents the version chain and
resolves to the latest version. Reproducibility claims should cite the exact
version DOI rather than the concept DOI.

## Current GitHub release

Release `v1.0.6` completes Marina Moseva's second-creator metadata in
`CITATION.cff`, package metadata, and citation guidance by adding her supplied
email address and ORCID iD. Her recorded contribution is software, validation,
formal analysis, and data curation. This release supersedes `v1.0.5` for
citation because the identifiers were supplied only after `v1.0.5` had already
been published. It does not alter code behaviour, experimental records,
derived numerical results, or claim-audit expectations.

Zenodo's GitHub integration reads the two ordered creators and their shared
affiliation and Marina Moseva's ORCID iD from `CITATION.cff`. The
version-specific DOI for `v1.0.6` must be
recorded in the GitHub release notes after Zenodo finishes processing; until
then, the concept DOI remains the stable link to the version chain.

## Known archive/tag mismatch in the first two records

The files currently attached to the `v1.0.0` and `v1.0.1` Zenodo records were
created during post-publication correction from the later repository tree at
commit `926b8c105e5b4b37b986b270f7b6b3d0a45d56bf`, not from the commits named by
the Git tags. After normalising the archive root, the two corrected deposits
differ only in `CITATION.cff` and `MANIFEST.sha256`; they must not be treated as
byte-exact snapshots of tags `v1.0.0` and `v1.0.1`. The record metadata for both
versions should carry this note. Version `v1.0.2` is therefore built from and
verified against its exact tag, and supersedes those records for reproduction.

The records contain a curated partial inspection and reproduction package, not
a complete raw-data execution ledger. Road networks, routes, raw paired traces,
model weights, and other restricted or unavailable inputs remain outside the
open deposit as documented in `docs/ARTIFACT_SCOPE.md`.

## License metadata

This is a mixed-license package. Author-created software is MIT licensed;
eligible author-created derived research records are CC BY 4.0; upstream
notices and `NOASSERTION` entries override those defaults. `LICENSES.md`,
`docs/DATA_LICENSES.md`, and `docs/PROVENANCE.csv` are controlling for the
path-level mapping. A record-level licence selection is only a summary and must
not be read as relicensing every archive member.

## Post-publication correction procedure

For an eligible minor correction to an already published Zenodo version:

1. open the record as its owner and select the published-file correction
   workflow;
2. replace the archive with a locally verified correction carrying the same
   version number and version-specific DOI;
3. update stale record notes and record-level rights metadata;
4. verify filenames, sizes, Zenodo checksums, title, creator, affiliation,
   version, DOI, related identifiers, and rights before publishing the
   correction;
5. publish the correction within Zenodo's applicable correction window;
6. download the public file again and compare its checksum and content with the
   approved local correction archive.

The correction must be limited to release hygiene, citation metadata,
documentation, or other changes permitted by Zenodo's published-file
correction policy. A scientifically material change requires **New version**,
a new version DOI, regenerated manifest, new archive checksum, and explicit
release notes.

## Release verification

Before uploading any corrected or new archive:

1. ensure every tracked release file is covered by `MANIFEST.sha256` and the
   path-level licence records;
2. run the pinned test suite, release audit, provenance audit, claim audit, and
   manifest check;
3. confirm that the archive contains no manuscript, journal template, private
   path, credential, cache, local environment, or temporary working record;
4. inspect the archive itself, including filenames and nested text members;
5. retain the archive SHA-256 and the Zenodo-reported checksum in the release
   log.

Do not require reviewers to access a private repository. The manuscript Data
Availability statement should link the public version DOI and should describe
the package boundary without claiming that omitted raw inputs were deposited.
