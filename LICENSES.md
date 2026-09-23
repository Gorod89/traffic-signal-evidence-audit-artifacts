# License scope and precedence

This is a mixed-license repository. The root `LICENSE` is a scope notice, not a
declaration that one license covers the entire tree.

## Author-created software: MIT

Unless a file carries a different notice, the MIT License in
`LICENSES/MIT.txt` applies to software authored by Mikhail Gorodnichev in:

- `scripts/`;
- `src/`;
- `tests/`;
- Python source under `supplementary/`;
- author-created build and continuous-integration configuration.

The MIT grant applies to the software itself. It does not grant rights in the
software's inputs, model weights, network data, route data, or third-party code.

## Author-created derived records: CC BY 4.0

The Creative Commons Attribution 4.0 International license identified in
`LICENSES/CC-BY-4.0.txt` applies to author-created CSV, JSON, NPZ, PNG,
Markdown, protocol, checksum, and similar research records in:

- `artifacts/audit/`, `artifacts/derived/`, and `artifacts/protocols/`;
- non-software outputs under `supplementary/`;
- author-created documentation in `docs/`, except where a file-level notice or
  upstream right applies.

The file-level provenance register can confirm or override this path default.
An upstream notice or `NOASSERTION` entry always takes precedence. Thus every
included author-created data record has an explicit reuse license even when it
is not one of the historically transformed objects enumerated in the current
provenance audit.

Attribution should include the record title, Mikhail Gorodnichev as creator,
the released version or commit, the repository or deposit URL, and an
indication of modifications. A citation is not a substitute for satisfying a
copyleft, share-alike, database-right, or notice requirement attached to an
upstream source.

## Upstream material and NOASSERTION

The following take precedence over the repository defaults, in descending
order:

1. a license or copyright notice inside the file;
2. a file-level entry in `docs/PROVENANCE.csv` naming an upstream license;
3. a file-level `NOASSERTION` entry;
4. the path defaults above.

`NOASSERTION` means that the release does not make a license determination or
grant reuse rights for that file. It must not be interpreted as public domain,
fair use, permission to redistribute, or a waiver of upstream rights. Such a
file should be excluded from an open archival deposit until its rights have
been resolved. See `LICENSES/NOASSERTION.txt`.

## Material not distributed here

This repository intentionally excludes:

- the article source, article PDF, bibliography, article figures, and journal
  template files;
- unreviewed historical source snapshots;
- road networks, route files, raw traces, model weights, and other
  license-sensitive third-party assets;
- private review records, credentials, caches, and local environments.

No rights in omitted material are granted. `docs/DATA_LICENSES.md` records the
known source and attribution boundary. A release must include a complete
`docs/PROVENANCE.csv`; where redistribution is not permitted or provenance is
incomplete, it should provide an upstream acquisition URL and checksum instead
of copying the asset.
