# Source, license, and attribution register

This public register documents known upstream sources used by the research
program. It is not a claim that every named asset is distributed in this
repository. Road-network files, route files, raw traces, and model weights are
not included in the public package.

The repository contains author-created code and selected derived records. Their
release licenses are assigned according to `LICENSES.md` and the file-level
`docs/PROVENANCE.csv`. Upstream licenses and `NOASSERTION` entries override
repository defaults.

## RESCO benchmark

- **Source:** Pi-Star-Lab, *Reinforcement Signal Control (RESCO) Benchmark*.
- **Upstream URL:** <https://github.com/Pi-Star-Lab/RESCO>
- **Recorded revision:**
  `f1ed9a174f8de41fc9d8689373b836bc882570dc`.
- **Upstream license:** GNU General Public License v3.0 (`GPL-3.0-only` as
  indicated by the upstream repository).
- **Citation:** James Ault and Guni Sharon, “Reinforcement Learning Benchmarks
  for Traffic Signal Control,” NeurIPS 2021 Datasets and Benchmarks Track.
- **Use in the program:** benchmark interfaces and the provenance path for the
  Cologne and Ingolstadt scenario copies used by the historical program.

The RESCO license does not replace the licenses or database-right obligations
of scenario data incorporated from other sources.

## Cologne / TAPASCologne

- **Local scenario name:** `cologne8`.
- **Provenance:** obtained through the recorded RESCO revision; the scenario is
  derived from TAPASCologne demand and an OpenStreetMap-based road network.
- **Upstream information:**
  <https://sumo.dlr.de/docs/Data/Scenarios/TAPASCologne.html>
- **License evidence retained in the working archive:** the adjacent scenario
  `LICENSE` identifies Creative Commons Attribution-NonCommercial-ShareAlike
  3.0 Unported (`CC-BY-NC-SA-3.0`).
- **Required attribution:** identify TAPASCologne and its source URL, retain the
  CC BY-NC-SA notice, indicate modifications, and credit OpenStreetMap and its
  contributors for the road-network component.

The scenario and observed route file are not distributed here. Before any
future redistribution, the exact upstream snapshot, modifications, and the
interaction between CC BY-NC-SA 3.0 and OpenStreetMap database rights must be
reviewed at file level.

## Ingolstadt / InTAS

- **Local scenario name:** `ingolstadt21`.
- **Provenance:** obtained through the recorded RESCO revision from the InTAS
  traffic scenario.
- **Upstream URL:** <https://github.com/silaslobo/InTAS>
- **Upstream license:** GNU General Public License v3.0 (`GPL-3.0-or-later` in
  the retained InTAS scenario notice).
- **Authors to credit:** Silas C. Lobo, Stefan Neumeier, Evelio M. G.
  Fernandez, and Christian Facchi.
- **Citation:** “InTAS -- The Ingolstadt Traffic Scenario for SUMO,” SUMO User
  Conference Proceedings, DOI <https://doi.org/10.52825/scp.v1i.102>.

The network and observed route file are not distributed here. A future deposit
must preserve the applicable GPL text, source attribution, modification notice,
and the exact upstream revision or acquisition checksum.

## Moscow / OpenStreetMap-derived network

- **Local scenario name:** `moscow`.
- **Recovered generation evidence:** `moscow.net.xml` states that Eclipse SUMO
  `netconvert` 1.27.1 generated it on 2026-07-24 from
  `data/osm/khoroshevo_mnevniki_bbox.osm.xml` using UTM projection and passenger
  road filtering.
- **Recovered network SHA-256:**
  `fb7b8cfdb56962b7313e18b12e1998c82fa9825532dc4f5002845549645693b2`.
- **Upstream attribution:** “© OpenStreetMap contributors.”
- **Upstream license:** Open Data Commons Open Database License 1.0 (`ODbL-1.0`).
- **License and attribution URL:** <https://www.openstreetmap.org/copyright>

The source OSM extract, its acquisition URL or query, timestamp, and checksum
were not recovered. Consequently, exact snapshot identity and share-alike
compliance cannot presently be demonstrated. The network must not be
redistributed in an open deposit until that provenance is resolved. Its current
status is `NOASSERTION`, not MIT or CC BY 4.0.

## Language-model sources

The model weights are not distributed. The retained policy-table metadata names
the following model sources, but does not preserve an immutable model revision.

### LightGPT-0.5B-Qwen2

- **Model identifier:** `lightgpt/LightGPT-0.5B-Qwen2`.
- **Source:** <https://huggingface.co/lightgpt/LightGPT-0.5B-Qwen2>
- **License shown by the upstream model repository:** MIT.
- **Provenance limit:** no immutable historical revision, weight checksum, or
  contemporaneous license snapshot was retained. The current upstream page
  must not be used to claim byte identity with the historical weights.

### SmolLM2-360M-Instruct

- **Model identifier:** `HuggingFaceTB/SmolLM2-360M-Instruct`.
- **Source:** <https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct>
- **License shown by the upstream model repository:** Apache License 2.0.
- **Provenance limit:** no immutable historical revision, weight checksum, or
  contemporaneous license snapshot was retained. The current upstream page
  must not be used to claim byte identity with the historical weights.

## Redistribution rule

A public release may include only files with a reviewed provenance entry and a
license compatible with redistribution. When an upstream asset cannot be
redistributed, publish its acquisition instructions, exact version or commit,
checksum, citation, and transformation record instead. A `NOASSERTION` entry is
a stop condition for open redistribution, not a permissive license.
