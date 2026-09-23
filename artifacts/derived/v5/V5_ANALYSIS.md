# V5 pre-freeze analysis

## Decision

- Stage A: **FAIL**.
- Stage B: **FAIL**.
- Stage C, freeze, and untouched Stage D: **not run** because the prerequisite gates are conjunctive and irreversible for this protocol.
- Traffic superiority, non-inferiority, and deployment claims: **not evaluated**.

## Data firewall

- Representation training: 120 paired branches.
- Development: 60 paired branches.
- Conformal calibration: 60 paired branches.
- Validation: 180 paired branches.
- Untouched test accessed: `false`.

## Stage A — representation

| Criterion | Observed | Threshold | Decision |
|---|---:|---:|---|
| Prediction improvement over persistence | -63.05% | ≥10.00% | FAIL |
| Effective-rank fraction | 0.114 | ≥0.250 | FAIL |
| Active latent dimensions | 100.00% | ≥50.00% | PASS |
| Ingolstadt zero-shot improvement | -88.91% | >0.00% | FAIL |

The action-conditioned predictor was worse than the persistence baseline, including on the held-out Ingolstadt network. Numerically active dimensions did not prevent a low effective-rank representation.

## Stage B — H4 verifier

| Criterion | Observed | Threshold | Decision |
|---|---:|---:|---|
| RMSE improvement over mean | 8.99% | ≥10.00% | FAIL |
| Benefit AUROC | 0.777 | ≥0.650 | PASS |
| Marginal coverage lower 95% | 82.78% | ≥85.00% | FAIL |
| Trigger coverage lower 95% | 82.62% | ≥85.00% | FAIL |
| Selected interventions | 0 | ≥30 | FAIL |
| Harmful-selection upper 95% | 100.00% | ≤10.00% | FAIL |
| Synthetic-OOD rejection | 100.00% | ≥95.00% | PASS |

The AUROC and OOD checks passed, but the conjunctive verifier gate failed. With zero selected H4 interventions, the harmful-selection upper bound is uninformative (100%), not evidence of safety.

## JEPA comparisons

- CFRA H4 RMSE: JEPA 189.938, no-JEPA 186.510 (JEPA difference +3.427).
- CFRA H4 AUROC: JEPA 0.777, no-JEPA 0.806 (difference -0.029).
- JEPA H1 RMSE/AUROC: 46.095/0.857; 3 selections, insufficient for the predeclared minimum.
- Legacy one-step cost RMSE: JEPA 5.624, ExtraTrees 5.113.
- Legacy gain AUROC: JEPA 0.822, ExtraTrees 0.846.

Neither integration improved its matched validation comparator.

## Development-only ablations

| Variant | RMSE | AUROC |
|---|---:|---:|
| `full_jepa` | 141.605 | 0.848 |
| `no_action_condition` | 140.684 | 0.848 |
| `no_target_ema` | 142.724 | 0.841 |
| `no_variance_covariance` | 143.046 | 0.832 |
| `random_encoder` | 145.483 | 0.830 |
| `handcrafted_only` | 143.481 | 0.814 |

The no-action-conditioning variant had slightly lower development RMSE than full JEPA. Therefore the experiment does not demonstrate that the learned representation uses the action in a beneficial way. Operational no-conformal/no-support ablations and the legacy representation-gate removal were reserved for Stage D and were not run after the gate failure.

## Multi-LLM proposer

- Families: 2 (LightGPT-0.5B-Qwen2, SmolLM2-360M-Instruct).
- Cross-model direct disagreement: 0.00%.
- Qwen useful-opportunity coverage: 34.72%.
- SmolLM useful-opportunity coverage: 34.72%.

Both families produced the same actions on the common prompt grid. The second family therefore increases model provenance diversity, but not behavioral complementarity in this experiment. The panel remains proposal-only.

## Why thresholds were not lowered

Lowering coverage, sample-count, or harmful-selection requirements after seeing these outcomes would turn validation into model selection, change the accepted risk level, and invalidate the predeclared error guarantees. A future attempt requires a new protocol version and new sealed seeds, not a V5 threshold edit.
