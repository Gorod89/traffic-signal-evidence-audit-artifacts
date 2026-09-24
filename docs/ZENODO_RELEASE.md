# Zenodo release procedure

## Current status

This repository does not currently assert a GitHub Release, Zenodo record,
archived version, or DOI. Do not place a fictional DOI, a placeholder DOI, or a
reserved-but-unpublished DOI in a public availability statement. Until a
versioned deposit is published, cite the exact Git commit used.

The repository is mixed-license: author-created software is MIT; eligible
author-created derived records are CC BY 4.0; upstream notices and
`NOASSERTION` entries override those defaults. A release must preserve this
mapping rather than label the entire archive MIT.

## 1. Complete the release boundary

Before creating a deposit draft:

1. Disable the Zenodo--GitHub integration for this repository and confirm that
   no automatic Zenodo draft, duplicate deposit, or previously reserved DOI
   already exists. The first production record must be the single manual
   deposit described below.
2. Ensure every tracked release file has an entry in `docs/PROVENANCE.csv` with
   its SHA-256, creator or source, transformation, redistribution status, and
   release license.
3. Exclude every file marked `NOASSERTION` from the open archive. Where useful,
   retain only an upstream URL, version or commit, lawful acquisition
   instructions, and a checksum.
4. Confirm that the archive contains no manuscript, journal template, road
   network, route file, raw trace, model weight, private path, credential,
   review record, cache, or local environment.
5. Run the repository tests, release audit, provenance audit, and manifest
   check from the pinned environment.
6. Regenerate `MANIFEST.sha256` only after the release tree is final, then
   verify every entry from a clean checkout.

## 2. Test in Zenodo Sandbox

Use <https://sandbox.zenodo.org> for a disposable dry run. Verify that:

- the resource type and title render correctly;
- Mikhail Gorodnichev is the creator and the institutional affiliation is
  correct;
- the description calls the package partial rather than fully reproducible;
- both MIT and CC BY 4.0 are represented as mixed licenses;
- upstream and `NOASSERTION` boundaries are visible in the description;
- related identifiers point to the exact GitHub repository or release;
- no DOI from the sandbox is copied into production metadata.

## 3. Create a production draft manually

Use a manual Zenodo upload for the first release. Manual upload makes the
mixed-license and file-level provenance boundary explicit; an automatic GitHub
archive can otherwise make the whole repository appear to carry one license.

In the production draft:

1. Select the resource type appropriate to the released package (normally
   `Software` for this combined code-and-derived-record release).
2. Enter the repository title, creator, affiliation, contact information,
   abstract, keywords, and the exact planned release version.
3. State that the deposit is a curated partial inspection package, not a full
   raw-data reproduction archive.
4. Add MIT and CC BY 4.0 as mixed licenses. Describe the path-level mapping and
   the precedence of upstream notices in the rights or description field.
5. Link the exact GitHub repository as a related identifier. Add an article DOI
   only after the article has one, and never reuse the article DOI as the
   deposit DOI.
6. Save the draft without publishing it.

Do not upload license-sensitive upstream assets merely to make the deposit more
complete. A documented omission is preferable to an unsupported redistribution.

## 4. Reserve, but do not yet advertise, the DOI

In the saved production draft, answer that the upload does not already have a
DOI and use Zenodo's **Get a DOI now** function. Record the reserved DOI exactly.

A reserved DOI may be inserted into the release files before upload, but it is
not a live archival citation until the record is published. While the draft is
unpublished:

- do not describe the package as publicly archived;
- do not give the reserved DOI to reviewers as if it resolved;
- do not delete the draft after distributing files containing the reservation,
  because deleting the draft loses that reservation.

After reservation, update the local release-candidate copy of `CITATION.cff`,
the availability statement, and release notes with the exact DOI. Do not push
or otherwise advertise those DOI-bearing files while the record remains a
draft. Rerun the release checks and rebuild the manifest locally.

## 5. Freeze the Git release

1. Commit the final metadata, licensing, provenance, and manifest changes in
   the local release candidate.
2. Re-run all checks from a clean checkout of that commit.
3. Create a local annotated version tag such as `v0.1.0` only when that version
   is genuinely ready; do not push the tag while the DOI is still a draft.
4. Build a deterministic archive from the tagged tree. Include
   `MANIFEST.sha256`, `LICENSE`, `LICENSES/`, `LICENSES.md`,
   `docs/DATA_LICENSES.md`, `docs/PROVENANCE.csv`, `CITATION.cff`, and release
   notes.
5. Compute and record the archive SHA-256 outside the archive.
6. Retain the exact archive and checksum for the later GitHub Release. Do not
   rely only on GitHub's automatically generated source ZIP if deposited byte
   identity is part of the claim.

## 6. Upload and verify the Zenodo draft

Upload the exact release-candidate archive and its checksum to the existing
Zenodo draft. In Preview, verify:

- filename, byte size, and SHA-256 against the locally frozen release asset;
- title, creator, affiliation, version, publication date, and keywords;
- mixed-license declarations and file-level provenance;
- related link to the exact tag or GitHub Release;
- the reserved DOI in every file that claims one;
- the absence of manuscript/template material and all excluded assets;
- that the description does not promise raw inputs or reruns that are absent.

A second person should independently check the preview and the checksums before
publication.

## 7. Publish and update public statements

Only after the preview and checksum audit passes should the Zenodo draft be
published. Publication registers the DOI and makes the record citable.

After publication:

1. Open the DOI in a logged-out browser and confirm that it resolves.
2. Push the already-audited commit and annotated tag without changing their
   bytes, create the GitHub Release, and attach the exact archive and checksum
   that Zenodo received.
3. Update the GitHub Release notes and repository citation metadata if the live
   Zenodo metadata differs from the reserved draft.
4. Replace the manuscript's provisional availability statement with the live
   DOI, the exact released version, and an honest description of the omitted
   inputs.
5. Do not require reviewer access to a private repository.

## 8. Immutability and later corrections

Treat the published files, version-specific DOI, Git tag, release assets, and
manifest as immutable. Metadata-only corrections may be made through Zenodo's
record editor where supported, but a change to code, data, provenance, license
mapping, or archive bytes requires a new version.

For a new version:

1. create a new release commit and tag;
2. regenerate and verify the manifest and archive checksum;
3. use Zenodo's **New version** workflow;
4. preserve the prior version-specific record and DOI;
5. describe every material change in release notes;
6. cite the concept DOI only when a citation intentionally refers to all
   versions, and cite the version DOI for exact reproducibility.

Never silently replace published evidence bytes or reuse a version tag for
different content.
