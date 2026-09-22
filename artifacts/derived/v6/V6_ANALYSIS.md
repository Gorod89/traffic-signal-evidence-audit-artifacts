# V6 analysis

## Outcome

The complete V6.1 pre-test SUMO cycle reached a protocol-valid terminal state.
Gate A passed. Gate B failed three of six preregistered conjunctive criteria
and issued `STOP`. Gates C and D, protocol freeze, untouched test, confirmatory
traffic comparison, and runtime ablations were therefore not run. The
untouched test remained sealed.

## Protocol Amendment 1

The first fresh branch launch was invalidated before model fitting or gate
evaluation. Inspection of only the status and target fields showed that targets
199 and 204 could not complete inside a 200-decision measured episode with a
four-decision estimand. Completed raw records also contain outcome fields; the
operator attests that those fields were not inspected before the amendment.
The first lock remains in Git and all associated seed blocks are retired. V6.1
uses wholly new disjoint seeds and a target bound derived from the finite
episode. This correction did not change hypotheses, gates, methods, endpoints,
sample sizes, or multiplicity.

## Fresh paired-data inventory

Exact key-set validation passed for every pre-test split:

| Split | Observed | Expected | Duplicates | Lock/routes bound |
|---|---:|---:|---:|---|
| Representation training | 600 | 600 | 0 | yes |
| Development | 360 | 360 | 0 | yes |
| Calibration | 720 | 720 | 0 | yes |
| Validation | 1800 | 1800 | 0 | yes |
| **Total** | **3480** | **3480** | **0** | **yes** |

The validation grid contains 3 networks, 6 scenarios, and 100 route-seed
clusters per network/scenario. The paired-branch data SHA-256 recorded by the
temporal evaluation is
`24231bba4eadff8051da0af7705ff429d86c93ec10c554ede9247836aea55d8e`.

## Gate A: corrected measurement path

Gate A used 60 development episodes across Moscow and Cologne, 30 route-seed
clusters, and the emergency-vehicle condition.

| Criterion | Value | Requirement | Result |
|---|---:|---:|---|
| Warm-up duration | 600 s | = 600 s | pass |
| Per-second metric sampling | true | true | pass |
| Movement-pressure environment path | true | true | pass |
| Censored trip endpoint | true | true | pass |
| Emergency advance notice | 60 s | >= 60 s | pass |

The canonical decision was `GO`.

## Gate B: temporal representation

Five fixed model seeds (6007, 6011, 6029, 6037, and 6043) were fitted for 40
epochs on representation-training plus development data and evaluated on the
locked validation split. The evaluation contained 1800 paired histories and
5400 H1/H2/H4 rows.

| Criterion | Value | Requirement | Result |
|---|---:|---:|---|
| H4 innovation RMSE improvement over persistence | 21.80% | >= 10% | pass |
| H4 RMSE improvement over no-history | 1.40% | >= 5% | **fail** |
| Latent effective-rank fraction | 18.80% | >= 25% | **fail** |
| Standardized action sensitivity | 0.790 | >= 0.05 | pass |
| Shuffled-action relative degradation | 1.51% | >= 5% | **fail** |
| Held-out Ingolstadt H4 improvement | 16.68% | > 0% | pass |

The canonical Gate B decision was `STOP`. Its metric source SHA-256 is
`3d2f53d31a5e74c5a644958bc19a87d015b57063a9debb0978ae1a70575b588d`.

This is a mixed representation result rather than evidence that JEPA had no
predictive value. The model passed the persistence, action-sensitivity, and
held-out-network criteria. It did not establish that history contributed the
minimum relevant improvement, that the latent used enough independent
dimensions, or that predictions depended sufficiently on the correct action
ordering. Because Gate B is conjunctive, these three failures require STOP.

## Terminal decision and unopened stages

The fail-closed validator rejected progression at `B_representation`.
Filesystem and canonical-artifact checks confirm:

- Gate C verifier/calibration analysis was not run;
- Gate D shadow execution was not run;
- no V6 protocol freeze was created;
- no untouched-test record was created;
- no confirmatory traffic grid was opened; and
- no runtime-ablation grid or Holm analysis was opened.

The raw-results directory contains 720 calibration and 1800 validation records,
but zero shadow, test, or ablation records.

## Controlled and retrospective diagnostics

The controlled temporal experiment remains diagnostic evidence:

| Model | RMSE |
|---|---:|
| Persistence | 0.4556 |
| Temporal residual JEPA | 0.1916 |
| No history | 0.3463 |
| No action | 0.2752 |

The full model improves over persistence by 57.9% in a controlled system with
known dynamics. This is not SUMO traffic-performance evidence.

The separate retrospective diagnostic on immutable V5 paired branches produced:

| Metric | Result |
|---|---:|
| Fit rows | 180 |
| Validation rows | 180 |
| H4 RMSE | 184.90 |
| Training-mean RMSE | 208.71 |
| Improvement over mean | 11.4% |
| Benefit AUROC | 0.797 |
| Ingolstadt H4 RMSE | 256.86 |

The historical V5 result comes from a different split and is not a same-split
competitor comparison for the fresh V6.1 cycle.

## Claims permitted

- The corrected SUMO measurement path passed Gate A.
- The complete preregistered pre-test paired dataset was collected and passed
  exact key, duplicate, route, and lock validation.
- Temporal JEPA beat persistence at H4 and improved on held-out Ingolstadt
  under the locked validation protocol.
- Action sensitivity exceeded its gate threshold.
- The fail-closed protocol worked as designed and stopped after three other
  representation criteria failed.

## Claims not permitted

- Gate B or the representation as a whole did not pass.
- The experiment did not establish a practically sufficient history benefit,
  latent-rank margin, or shuffled-action dependence.
- The verifier, conformal authority path, shadow runtime, and intervention
  safety/activity were not validated.
- There is no untouched-test or runtime-ablation result.
- No claim of superiority or non-inferiority to PPO, Advanced-MP, IDQN, V3,
  V4, V5, or another traffic controller is permitted.

## Interpretation

V6.1 resolves the earlier measurement and sample-size blockers and provides a
complete negative gate result rather than an unfinished experiment. The next
version should improve temporal/action identifiability and latent diversity on
development data, then repeat Gate B with new preregistered data. It must not
reuse the sealed V6 test or lower the failed thresholds.
