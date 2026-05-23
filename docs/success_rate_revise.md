# Success Rate Definition — Unified Design

**Last updated:** 2026-05-13
**Status:** Implemented

---

## 1. Paper Primary Success Rate

**Paper primary success = `success_pose_1cm_5deg_rate`**

```
primary_success = (final_pos_error <= 0.01m) AND (final_theta_error_deg <= 5.0deg)
```

This is the main success metric for Paper 1's push-to-pose / OOD generalization claim.

---

## 2. Unified Success Metric Hierarchy

| Semantic Name | Definition | Use Case |
|---------------|-----------|----------|
| `primary_success` | `success_pose_1cm_5deg` | **Paper main success rate** |
| `coarse_success` | `success_pose_2cm_15deg` | Coarse task completion |
| `precision_success` | `success_pose_0p5cm_5deg` | High-precision completion |
| `strict_completion` | `success_pose_0p15cm_3deg` | Ultra-strict (1.5mm + 3deg) |
| `legacy_pos_5cm` | `success_pos_5cm` | Debug/capacity only, NOT paper metric |

---

## 3. Field Aliases

- `success_rate` = `primary_success_rate` (backward-compatible alias)
- `success_rate_definition` = `"success_pose_1cm_5deg"` (always written to summary)
- `strict_pose_success` = whether strict pose early stop was triggered (NOT paper primary success)

---

## 4. What NOT to Use as Paper Primary Success

1. **`success_rate` in open-loop mode** — compound condition (improved_cost AND improved_dist AND final_dist < 5cm AND no collision). Marked as `success_rate_definition = "open_loop_compound_legacy"`.

2. **`strict_pose_success`** — only indicates whether the strict pose early stop (1.5mm + 3deg) was triggered during MPC execution. This is an execution control flag, not the paper success metric.

3. **`success_pos_5cm` / `legacy_pos_5cm`** — position-only 5cm threshold, too loose for paper claims.

---

## 5. OOD Gap Reporting

OOD gap should be reported mainly with:

1. `final_pose_cost` gap (continuous, most statistically robust)
2. `final_pos_error` gap (mm)
3. `final_theta_error_deg` gap (degrees)
4. `primary_success_rate` gap (= success_pose_1cm_5deg_rate gap)

Binary success_rate alone is insufficient for OOD gap analysis due to high variance at small sample sizes.

---

## 6. Code Locations

### Core helper
`src/metrics/mujoco_oracle_capacity.py` — `compute_pose_success_metrics(final_pos_error, final_theta_error_deg)`

### Summary generation
- `run_mujoco_oracle_mpc_closed_loop_capacity()` — adds `success_rate_definition`, `primary_success_rate`, `coarse_success_rate`, `precision_success_rate`, `strict_completion_rate`, `legacy_pos_5cm_rate`
- `run_mujoco_oracle_mpc_capacity()` — open-loop, same fields but `success_rate_definition = "open_loop_compound_legacy"`

### Sweep scripts
- `scripts/run_c23_strictstop_eval.py` — `build_report()`, `summarize_reports()`, `write_compact_summary()`
- `scripts/run_c23_obstacle_sixpack_sweep.py` — `build_report()`, `summarize_by_budget()`, `summarize_by_layout()`, `write_compact_summary()`
- `scripts/run_c23_obstacle_sixpack_sweep_parallel.py` — same structure as serial version
- `scripts/check_mpc_capacity.py` — prints `success_rate_definition` and all semantic rates
- `scripts/run_closed_loop_sweep.py` — `extract_metrics_from_report()` includes all semantic rates

---

## 7. Threshold Convention

All thresholds use `<=` (inclusive). This is consistent across `compute_pose_success_metrics()` and all downstream consumers.

---

## 8. Backward Compatibility

- `result["success"]` is retained but now equals `primary_success` for closed-loop modes
- `result["success_definition"]` field added to clarify which definition applies
- All old per-threshold fields (`success_pos_5cm`, `success_pose_2cm_15deg`, etc.) are preserved
- `strict_pose_success` field preserved (still means early-stop trigger)
- No fields deleted

---

**End of Document**
