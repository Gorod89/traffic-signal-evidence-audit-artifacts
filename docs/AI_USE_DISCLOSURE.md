# Assistive-tool disclosure

Generative-AI coding agents were used under the first author's direction in both the
original research programme and the later reanalysis. Codex (OpenAI; model
GPT-5.6 Sol) executed commands used to generate routes, train controllers and
verifiers, run SUMO evaluations and paired-branch collections, and compute
version-level analyses. Claude Code (Anthropic; model Claude Opus 5, including
inventory subagents) was used for the V1--V8 inventory, subsequent analyses,
the internal prediction lock, collection and scoring of the sealed seed block,
and the incumbent-invariance analysis. Both systems were also used for code
development and refactoring, drafting and language revision, literature
discovery, bibliographic formatting, figures, internal consistency checks,
post hoc audits, and reproduction runs.

Experimental records are outputs of the SUMO simulator and the research
programs; no observation was synthesized, edited, or replaced by a language
model. Agent session logs retained by the first author record commands executed in
those sessions, but cannot establish the origin of commands run elsewhere or
who made every historical design decision. Both authors reviewed the code,
configurations, protocols, outputs, analyses, and text and take full
responsibility for the repository and associated publication.

The term `agent` in `scripts/analysis/audit_claims.py` is a provenance class
retained from the historical checker: it marks values first reported by a
Claude Code inventory subagent and then checked against the named primary
artifact. The label describes provenance, not the program that produced the
underlying experimental record.
