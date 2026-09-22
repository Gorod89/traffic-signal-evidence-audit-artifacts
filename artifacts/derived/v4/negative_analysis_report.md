# V4 negative-study report

## Protocol decision

The predeclared protocol stopped at Stage B. Stage A passed on
360 paired internal-test branches, with an oracle
opportunity rate of 0.427778. Stage B
did not pass (3 of 6 criteria failed), so the
protocol was not frozen and Stage D was not authorized.

- `rmse_improvement`: observed 0.0534236 > 0.1 — FAIL
- `benefit_auroc`: observed 0.694947 > 0.6 — PASS
- `coverage_lower_bound`: observed 0.846053 >= 0.85 — FAIL
- `trigger_coverage_lower_bound`: observed 0.88412 >= 0.85 — PASS
- `harmful_selected_upper_bound`: observed 0.561497 <= 0.1 — FAIL
- `synthetic_extreme_ood_rejection`: observed 710863 > 2.3 — PASS

## Diagnostic V3 context

These values summarize immutable historical V3 rows. They are descriptive
context only; they are not a V4-versus-V3 treatment comparison.

| Historical method | Rows | Mean waiting (s) | Mean speed (m/s) | Mean throughput |
|---|---:|---:|---:|---:|
| ergs_v3 | 360 | 12.430506 | 7.477456 | 2542.269 |
| ppo_ensemble | 360 | 12.416229 | 7.479237 | 2542.100 |

No V4 controlled-authority traffic rows exist because the gate failed.
Consequently, a V4-versus-V3 traffic effect, confidence interval, p-value,
non-inferiority conclusion, or competitor ranking is not estimable.
