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
- **Retained license evidence:** the adjacent environment copy contains the
  GNU General Public License v3 legal text. The archive does not retain an
  InTAS-specific copyright notice or an election between `GPL-3.0-only` and
  `GPL-3.0-or-later`, so this register does not assert either SPDX variant for
  the historical scenario copy.
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
- **Recovered source extract:** the working archive retains that OSM XML. Its
  embedded `osmGet.py` record names bbox `37.5117,55.7770,37.5426,55.7948`, the
  Overpass endpoint `https://overpass-api.de/api/interpreter`, generation time
  2026-07-24 09:49:02, and OSM base time `2026-07-24T06:46:58Z`.
- **Recovered source-extract SHA-256:**
  `2bad5f73338b69706c9120382539544253d004bb99d97734500731680ae89c87`.
- **Recovered network SHA-256:**
  `fb7b8cfdb56962b7313e18b12e1998c82fa9825532dc4f5002845549645693b2`.
- **Upstream attribution:** “© OpenStreetMap contributors.”
- **Upstream license:** Open Data Commons Open Database License 1.0 (`ODbL-1.0`).
- **License and attribution URL:** <https://www.openstreetmap.org/copyright>

The recovered extract closes the earlier source-identification gap, but neither
the extract nor the derived network is distributed here. Any future
redistribution must preserve ODbL attribution, provide the applicable database
and produced-work notices, and document the transformation. Until that
file-level review is complete, the derived network remains excluded from the
open deposit; it is not MIT or CC BY 4.0 material.

## Language-model sources

The model weights are not distributed. The private working archive preserves
download metadata for the following two snapshots; the public package records
their identities and the licensing limits without redistributing the files.

### Qwen2.5-0.5B-Instruct (V1)

- **Model identifier recorded by the V1 configuration:**
  `Qwen/Qwen2.5-0.5B-Instruct`.
- **Source:** <https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct>
- **Configuration evidence:** the retained private V1
  `configs/experiment.yaml` has SHA-256
  `ed7ce75cb6f6af299655dc35485e7177239b23e43eca99147a20ec9a124049b0`
  and names this model explicitly.
- **Provenance limit:** no byte-exact V1 weight snapshot, immutable model
  revision, download record, adjacent model card or adjacent license file was
  retained. The identifier establishes the configured model name, not the exact
  weights that executed or their contemporaneous licensing state.

### LightGPT-0.5B-Qwen2

- **Model identifier:** `lightgpt/LightGPT-0.5B-Qwen2`.
- **Source:** <https://huggingface.co/lightgpt/LightGPT-0.5B-Qwen2>
- **Retained revision:**
  `f566851abf69131d49117cda06fc2e4c6ddbb459`.
- **Retained weight SHA-256:**
  `7f43bc8a498a64a5a169469fd2e0a22f58a02e8b2269b884564a6aa66351c242`.
- **Retained metadata:** `README.md` (model card) declares MIT, while the
  adjacent `LICENSE` contains Apache License 2.0 text attributed to Alibaba
  Cloud. This conflict is not resolved by the archive and is a stop condition
  for redistributing the weights.

### SmolLM2-360M-Instruct

- **Model identifier:** `HuggingFaceTB/SmolLM2-360M-Instruct`.
- **Source:** <https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct>
- **Retained revision:**
  `a10cc1512eabd3dde888204e902eca88bddb4951`.
- **Retained weight SHA-256:**
  `e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e`.
- **Provenance limit:** no adjacent model card or license file survives in the
  archived snapshot. A present-day upstream license statement must not be used
  as a contemporaneous license record for these historical bytes.

## Redistribution rule

A public release may include only files with a reviewed provenance entry and a
license compatible with redistribution. When an upstream asset cannot be
redistributed, publish its acquisition instructions, exact version or commit,
checksum, citation, and transformation record instead. A `NOASSERTION` entry is
a stop condition for open redistribution, not a permissive license.
