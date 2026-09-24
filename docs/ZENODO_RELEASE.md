# Zenodo deposit and correction record

## Published records

Zenodo published two versioned records on 24 September 2026:

| Release | Version DOI | Record |
|---|---|---|
| `v1.0.0` | `10.5281/zenodo.22938827` | <https://zenodo.org/records/22938827> |
| `v1.0.1` | `10.5281/zenodo.22938960` | <https://zenodo.org/records/22938960> |

The concept DOI `10.5281/zenodo.22938826` represents the version chain and
resolves to the latest version. Reproducibility claims should cite the exact
version DOI rather than the concept DOI.

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
