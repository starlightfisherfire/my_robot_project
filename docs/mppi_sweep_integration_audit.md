# MPPI Sweep Integration Audit

**Date:** 2026-05-17  
**Auditor:** 管家助手 (zhushou)  
**Context:** Phase 0.2 — verify MPPI runner integration before sweep

---

## 1. `make_planner("mppi")` — PASS ✅

```python
from src.planners import make_planner
mppi = make_planner('mppi', horizon=10, action_dim=2, temperature=1.0)
# → MPPI instance, type=MPPI, T=1.0
```

MPPI is registered in `src/planners/__init__.py` and callable via `make_planner("mppi")`.

## 2. MPPIResult diagnostics — PASS ✅

```python
@dataclass
class MPPIResult:
    ...
    diagnostics: dict | None = None  # effective_sample_size, weight_entropy, collapse_rate
```

Diagnostics dict includes:
- `effective_sample_size_mean`
- `effective_sample_size_min`
- `weight_entropy_mean`
- `weight_entropy_min`
- `collapse_rate`

All computed per-iteration, aggregated in `optimize()`.

## 3. Runner supports planner_mode=mppi — PASS ✅

`scripts/render_closed_loop_rollout_mppi_with_data.py`:
- `--planner-mode mppi` ✅
- `--mppi-temperature FLOAT` ✅
- `--mppi-init-std FLOAT` ✅
- `--mppi-smoothing FLOAT` ✅
- Reads `result.diagnostics` ✅
- JSON summary includes all diagnostics ✅

## 4. Episode/transition data saving — PASS ✅

`src/data/episode_writer.py`:
- Saves transition-level `.npz` (states, actions, next_states, object_poses, ee_positions, velocities, contact/collision flags, goal_pose, obstacle_features) ✅
- Saves metadata `.jsonl` per episode ✅
- Success AND failure data saved ✅

## 5. Proximity cost / obstacle geometry — UNCERTAIN ⚠️

The current cost function (`MujocoPushEnv.compute_cost`) computes based on goal distance + contact + collision. Obstacle avoidance is implicit in the MPC rollout (collision penalized), not via explicit obstacle_positions/radii proximity cost.

- `step_env()` calls `env._check_obstacle_collision()` ✅
- Obstacle features are extracted from template and saved to npz ✅
- No explicit proximity cost with `obstacle_positions/obstacle_radii` is passed to the cost function

**Verdict:** UNCERTAIN — MPPI explores via sampling and collision is penalized in the rollout, but there is no explicit proximity geometry passed to the cost function. The circle-approximation used by `_check_obstacle_collision()` is the only obstacle geometry input.

*Mitigation:* First version uses collision-based penalty through rollout. This is acceptable for initial sweep. Can add proximity cost later if needed.

## 6. MPPI sample evaluation — parallel NOT used ⚠️

Current MPPI implementation evaluates samples **serially**:
```python
costs = np.asarray([cost_fn(seq) for seq in samples], dtype=np.float64)
```

**Verdict:** slow_path=true — Each MPPI optimization step evaluates 1024 samples × 5 iterations = 5120 cost evaluations per MPC step, all serial. With ~100 MPC steps per run, this is ~500K serial cost evaluations.

*Impact:* Single run ~200-400s. 100 runs would need 5-11 hours.

This is expected and not a blocker for Phase 1.

## 7. NaN check in render script — PASS ✅

`render_closed_loop_rollout_mppi_with_data.py` loads the saved npz after episode save and checks states/actions/next_states for NaN. Result included in JSON summary as `nan_check: "PASS"/"FAIL"`.

## 8. 10-family template file — PASS ✅

`data/sim/metadata/reset_templates_obstacle_10family_v0.json`:
- 73 templates total ✅
- All 10 families present ✅
- Inventory CSV and provenance MD generated ✅

## 9. Shell runner checkpoint support — PASS ✅

`scripts/run_mppi_param_sweep_checkpoint8h_v1.sh`:
- `--dry-run` ✅
- `--smoke` ✅
- `--phase1` ✅
- `--resume RUN_ROOT` ✅
- 8h checkpoint guard (STOP_LAUNCH_AFTER=7h45m) ✅
- Balanced scheduling (Priority A→B→C, interleaved temperature) ✅

---

## Final Verdict

| Check | Status |
|-------|--------|
| 1. MPPI creation | ✅ PASS |
| 2. MPPI diagnostics | ✅ PASS |
| 3. Runner integration | ✅ PASS |
| 4. Episode data saving | ✅ PASS |
| 5. Obstacle proximity cost | ⚠️ UNCERTAIN (collision-based only) |
| 6. Serial sample eval | ⚠️ slow_path=true |
| 7. NaN check | ✅ PASS |
| 8. 10-family templates | ✅ PASS |
| 9. Shell checkpoint runner | ✅ PASS |

**Overall: ✅ PASS — Ready for Phase 1 sweep.**

Slow eval path is expected. Obstacle proximity is collision-based in first version, which is acceptable. No blockers.
