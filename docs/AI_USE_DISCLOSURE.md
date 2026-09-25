# Assistive-tool disclosure

During manuscript preparation and maintenance of this public reproducibility
package, the author used ChatGPT 5.6 Sol (OpenAI) and Claude Opus 5 (Anthropic)
as assistive tools for drafting and language revision, code refactoring,
literature discovery, bibliographic formatting, internal consistency checks,
and command orchestration for post hoc audit and reproduction tasks.

These tools were not used to generate, synthesize, or replace data,
measurements, or experimental results. Reported quantities were produced by
archived simulation and analysis programs and were recomputed where retained
inputs permit or checked against retained artifacts, configurations, and
recorded seeds. All assisted material was critically reviewed, verified, and
edited by the author, who takes full responsibility for the repository and the
associated publication.

The term `agent` in `scripts/analysis/audit_claims.py` is a provenance class
retained from the historical checker: it marks values first reported by an
inventory subagent and then checked against the named primary artifact. It does
not identify the software product that produced a number.
