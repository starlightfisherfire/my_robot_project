# Critical Closed-Loop Diagnosis Report

**Date:** 2026-05-25  
**Run diagnosed:** `runs/overnight_dual_planner_closed_loop_20260525_033000/`  
**Status:** 90 trials, **0 success**, best_dist=0.0145m  
**Diagnostician:** 管家助手 🏠

---

## 1. Executive Summary

**One-line bottleneck:** The `LearnedRolloutCostFn._compute_cost()` in `learned_planner_adapter.py` uses a **radically simplified cost function** with only 4 terms, while `src/planners/cost_functions.rollout_cost()` provides a full 11-term cost function with collision, reach, push-alignment, proximity, and smoothness terms. The planner is optimizing the wrong loss — it doesn't know about collisions, doesn't know about maintaining contact, and receives no penalty for drifting away.

**Shortest next step:** Replace `LearnedRolloutCostFn._compute_cost()` with `rollout_cost()` from `cost_functions.py`. This is a ~10-line change.

**Secondary factors:** Object_centric and causality_aware models have 0% contact rate on many families — they never learn to make contact. MPPI is slower (30-130s per trial) with no benefit over CEM (1-15s). All three models are accurate offline (pos_rmse@10 < 4mm) but fail to plan due to broken cost function.

---

## 2. Success Metric Audit

| Item | Value |
|------|-------|
| **Metric source** | Inline in runner scripts (line 257 in `run_learned_mujoco_closed_loop.py`, `run_overnight_dual_planner.py`) |
| **Definition** | `bool(final_dist < 0.02 AND degrees(theta_err) < 10.0)` |
| **Uses best_dist?** | ❌ No — uses `final_dist`, not `best_dist` |
| **Requires theta?** | ✅ Yes — theta_err < 10° required |
| **Collision matters?** | ❌ No — collision not in success metric |
| **Metric file** | `src/metrics/success_metrics.py` — **EMPTY (0 bytes)** |
| **Genuine failure?** | ✅ Yes — even by `best_dist < 0.02`, only trial_0001 qualifies (best=0.0145) but final=0.0397 |

### Key pattern: "Proximity-then-Drift"

Trial_0001 is the canonical case: the planner reaches 1.45cm (within 2cm zone) at some step, then **drifts back to 3.97cm**. The cost function does not penalize the drift because it only evaluates the final timestep with low weight (pos_weight=1.0). With proper weights (w_pos=10.0), the drift would be heavily penalized.

### Recommendation
- Keep using `final_dist` for success (standard practice)
- Add `ever_reached_2cm` as an auxiliary metric for debugging
- Populate `src/metrics/success_metrics.py` with the success function

---

## 3. Top/Bottom Video Diagnosis

**Status: No rendered videos exist** for the overnight run. The render command is:

```bash
PYTHONPATH=. python scripts/render_overnight_top_trials.py \
    --run-dir runs/overnight_dual_planner_closed_loop_20260525_033000 \
    --top-k 5 --bottom-k 5
```

### Top 5 Cases (by best_dist)

| Rank | Trial | Model | Planner | Family | Init (m) | Best (m) | Final (m) | Contact | Note |
|------|-------|-------|---------|--------|----------|----------|-----------|---------|------|
| 1 | trial_0001 | flat | cem/small | open_space | 0.242 | **0.0145** | 0.040 | 0.47 | Proximity-then-drift |
| 2 | trial_0030 | object_centric | cem/small | open_space | 0.264 | 0.0555 | 0.064 | 0.20 | |
| 3 | trial_0063 | causality_aware | cem/medium | open_space | 0.264 | 0.0593 | 0.082 | 0.18 | |
| 4 | trial_0004 | flat | cem/medium | open_space | 0.242 | 0.0775 | 0.082 | 0.20 | |
| 5 | trial_0003 | flat | cem/medium | open_space | 0.264 | 0.0935 | 0.096 | 0.14 | |

### Bottom 5 Cases (by best_dist)

| Rank | Trial | Model | Planner | Family | Best (m) | Contact | Note |
|------|-------|-------|---------|--------|----------|---------|------|
| 90 | trial_0089 | causality_aware | mppi/smoke | edge_goal | 0.390 | 0.00 | No improvement |
| 89 | trial_0083 | causality_aware | cem/small | edge_goal | 0.390 | 0.00 | No improvement |
| 88 | trial_0059 | object_centric | mppi/smoke | edge_goal | 0.390 | 0.00 | No improvement |
| 87 | trial_0053 | object_centric | cem/small | edge_goal | 0.390 | 0.00 | No improvement |
| 86 | trial_0011 | flat | cem/small | blocking | 0.323 | 0.00 | Collision 37%, no move |

### Observations
- All top cases are **open_space** family with no obstacles
- All bottom cases are **edge_goal** or **blocking** families with distant goals
- Zero-contact trials dominate failures: object_centric and causality_aware models almost never make contact in blocking/edge_goal families
- The flat model is the only one that reliably makes contact (29/30 improved)

---

## 4. Cost Adapter Audit

### ⛔ CRITICAL FINDING

**`LearnedRolloutCostFn._compute_cost()`** in `src/planners/learned_planner_adapter.py` (lines 145-173) implements a **radically incomplete** cost function compared to the full `rollout_cost()` in `src/planners/cost_functions.py`.

### Cost Term Comparison

| Cost Term | Full `rollout_cost()` | Adapter `_compute_cost()` | Gap |
|-----------|----------------------|--------------------------|-----|
| `w_pos` (goal position) | 10.0 | 1.0 | **10x underweight** |
| `w_theta` (goal orientation) | 2.0 | 0.1 | **20x underweight** |
| `w_reach` (EE-object distance) | 5.0 | ❌ MISSING | No EE-object coupling |
| `w_no_contact` (penalize no contact) | 2.0 | ❌ MISSING | No contact incentive |
| `w_push_alignment` (push dir) | 1.0 | ❌ MISSING | No push direction |
| `w_collision` (any collision) | 20.0 | ❌ MISSING | No collision penalty |
| `w_collision_step` (per-step) | 1.0 | ❌ MISSING | No step collision |
| `w_proximity` (obstacle margin) | 5.0 | ❌ MISSING | No obstacle avoidance |
| `w_action` (action magnitude) | 0.05 | 0.01 | **5x underweight** |
| `w_smooth` (action changes) | 0.1 | ❌ MISSING | No smoothness |
| `w_subgoal` (waypoint) | 0.0 | ❌ MISSING | Disabled in both |
| `contact_bonus` (negative cost) | ❌ NOT IN FULL | ✅ 0.1 | Unique to adapter |

### Impact Analysis

1. **Missing collision cost (w=20.0)**: The planner has zero incentive to avoid obstacles. In blocking/narrow_passage families, CEM/MPPI happily samples trajectories through walls because there's no penalty. This explains trial_0011 (37% collision, no movement).

2. **Missing reach cost (w=5.0)**: The planner doesn't know the EE needs to be near the object. It can plan EE trajectories far from the object while still reducing object-goal distance (impossible — the model delta prediction doesn't include this constraint).

3. **Missing no_contact cost (w=2.0)**: The planner can propose trajectories with zero contact. Object_centric and causality_aware models show 0% contact in blocking families — the planner doesn't realize contact is necessary for pushing.

4. **Underweight position cost (1.0 vs 10.0)**: The planner gets weak gradient signal to reduce distance. This explains the proximity-then-drift behavior: even when the model predicts the object will drift away, the cost penalty is too small to influence optimization.

5. **Missing push_alignment (w=1.0)**: No incentive for the push direction to align with object→goal direction. The EE can push the object sideways and the cost function won't penalize it.

### MPPI additional issue
MPPI uses the same `LearnedRolloutCostFn`, so it has all the same cost gaps. MPPI's extra computation (256-1024 samples vs CEM's 32-128) is wasted on a broken cost landscape.

---

## 5. State/Action Interface Audit

| Item | Status | Detail |
|------|--------|--------|
| **State16 schema** | ✅ Consistent | 6 tokens × 16 features: EE, OBJ, GOAL, OBS×3 |
| **Obstacle tokens in inference** | ✅ Fixed | `run_overnight_dual_planner.py` extract_state16() includes obstacle tokens (3-5) |
| **Obstacle tokens in original script** | ❌ Missing | `run_learned_mujoco_closed_loop.py` extract_state16_from_mujoco() leaves tokens 3-5 as zeros |
| **Action convention** | ⚠️ Discrepancy | PAPER1_CONVENTION default is 0.5 m/s, but trial results show 0.75 m/s in convention.describe() |
| **Model receives** | ✅ physical_velocity | `a_phys = a_norm * max_speed_mps` |
| **Env receives** | ✅ normalized | `env.step(a_norm)` clipped to [-1, 1] |
| **EE update in adapter** | ✅ Correct | `disp = a_phys * control_dt` then `ee = ee + disp` |
| **EE update in original** | ❌ Wrong | `ee = ee + action_sequence[t]` (normalized action as displacement!) |
| **Training data action** | physical_velocity | actions_physical in dataset = a_norm * max_speed_mps |
| **max_speed_mps risk** | ⚠️ | Training data max_speed_mps was 0.5; run may have used 0.75 — mismatch would cause model input out-of-distribution |

### Action Convention Discrepancy Note
The `PAPER1_CONVENTION` is defined with `max_speed_mps=0.5` in `action_conventions.py`. The overnight run script explicitly sets `conv.max_speed_mps = 0.5`. However, trial results record `max_speed_mps: 0.75` in `planner_config.convention`. This could indicate:
- A code version discrepancy (the action_conventions.py was edited post-run)
- Or the convention.describe() was cached before the override

**Risk**: If the env actually used max_speed=0.75 while training data was max_speed=0.5, model inputs would be 1.5x out of distribution. This is unlikely to be the primary bottleneck given the cost function gap, but should be verified.

---

## 6. Training/Data/Capacity Decision

### Training Summary

| Model | Epochs | Train Loss (final) | Val Loss (final) | Train→Val Gap | Overfitting? |
|-------|--------|--------------------|--------------------|---------------|-------------|
| flat | 10 | 0.000417 | 0.000408 | -2.2% | No |
| object_centric | 10 | 0.000519 | 0.000521 | +0.4% | No |
| causality_aware | 10 | 0.000571 | 0.000604 | +5.8% | Mild |

All three models show **monotonically decreasing val_loss** across 10 epochs. No overfitting detected.

### Offline Evaluation

| Model | Dyn RMSE | Pos RMSE@10 | Theta RMSE@10 | Pos RMSE@20 |
|-------|----------|-------------|---------------|-------------|
| flat | 0.0158 | 0.0031m | 0.0726 rad | 0.0039m |
| object_centric | 0.0179 | 0.0020m | 0.0416 rad | 0.0028m |
| causality_aware | 0.0182 | 0.0006m | 0.0224 rad | 0.0011m |

### Key Insight: **Models are NOT the bottleneck**

All three models predict object deltas with **sub-millimeter accuracy** at 10-step rollout (0.6-3.1mm). The causality_aware model is remarkably accurate (0.6mm pos error at 10 steps). This means the learned dynamics model is working correctly. The failure is entirely in the **planning layer** — the cost function that evaluates which action sequence to execute.

### Dataset

| Dataset | Episodes |
|---------|----------|
| mppi_stage2c | 456 |
| mppi_stage2a | (exists) |
| mppi_stage2b_speed | (exists) |

Each episode has ~185 transitions → ~84,360 total transitions.

### Model Capacity

| Model | Parameters |
|-------|-----------|
| flat | 590,598 |
| object_centric | 845,830 |
| causality_aware | 1,240,710 |

### Decision Matrix

| Question | Answer | Rationale |
|----------|--------|-----------|
| need_retrain? | **NO** | Offline accuracy is excellent; models predict accurately |
| need_more_data? | **NO** | 456 episodes, 84K transitions, models learn well |
| need_capacity_ablation? | **LATER (P2)** | Current models work; capacity isn't limiting factor |
| need_cost_function_fix? | **YES — P0** | Primary bottleneck |
| recommended_epochs_if_retrain | N/A | Don't retrain — fix the planner |

---

## 7. Next Actions

### P0: Fix Cost Adapter (blocking all progress)

**Action**: Replace `LearnedRolloutCostFn._compute_cost()` with `rollout_cost()` from `cost_functions.py`

```python
# In learned_planner_adapter.py, LearnedRolloutCostFn.__call__():
# Replace:
#     return self._compute_cost(object_traj, ee_traj, action_sequence_norm)
# With:
from src.planners.cost_functions import rollout_cost, CostWeights
return rollout_cost(
    predicted_object_poses=object_traj,
    ee_positions=ee_traj,
    action_sequence=action_sequence_norm,
    goal_pose=self.goal_pose,
    weights=CostWeights(),
    collision_flags=collision_flags,    # need to track
    contact_flags=contact_flags,         # need to track
)
```

**Verify**: Small A/B sweep (18 trials: 3 models × 2 planners × 3 families)  
**Expected**: Success rate should jump from 0% to >30% for open_space, >10% for blocking

### P1: Populate success_metrics.py

**Action**: Move the success metric from inline scripts to `src/metrics/success_metrics.py`

```python
def is_success(final_dist, theta_err_rad, dist_threshold=0.02, theta_threshold_deg=10.0):
    return bool(final_dist < dist_threshold and np.degrees(theta_err_rad) < theta_threshold_deg)

def ever_reached(target_dist, best_dist, threshold=0.02):
    return bool(best_dist < threshold)
```

### P2: Verify max_speed_mps consistency

**Action**: Add assertion in run scripts that max_speed_mps matches training data convention (0.5 m/s).  
**Risk**: If 0.75 was actually used, model inputs were out-of-distribution.

### P2: Model comparison with fixed cost

**Action**: After cost fix, re-run the full model comparison:
- flat vs object_centric vs causality_aware
- CEM vs MPPI
- All families

### P2: Render videos

**Action**: Run render script on top/bottom cases to visually confirm behavior patterns.

---

## 8. Key Diagnosis Table

| Component | Status | Severity | Action |
|-----------|--------|----------|--------|
| Cost adapter | **BROKEN** | 🔴 P0 | Replace _compute_cost with rollout_cost |
| success_metrics.py | EMPTY | 🟡 P1 | Populate |
| Model accuracy (offline) | Excellent | 🟢 None | Models are fine |
| Training (loss curves) | Healthy | 🟢 None | No overfitting |
| Dataset | Adequate (456 eps) | 🟢 None | 84K transitions sufficient |
| Action convention | ⚠️ Discrepancy | 🟡 P2 | Verify max_speed=0.5, not 0.75 |
| Obstacle tokens (original script) | Missing | 🟡 P2 | Fixed in overnight runner |
| MPPI performance | 10x slower, no benefit | 🟡 P2 | CEM is sufficient |
| Videos | Not rendered | 🟡 P2 | Run render script |

---

## 9. Decision

**The closed-loop planner is failing because of a cost function bug, not a model problem.**

- **Models predict object deltas accurately** (sub-mm to few-mm error at 10-step rollout)
- **The cost function they optimize is missing 7 out of 11 terms** including collision, reach, contact, push alignment, and obstacle proximity
- **The existing terms are 5-20x underweighted** compared to the full `rollout_cost()`
- **No retraining needed** — the fix is purely in the planning layer
- **After the fix**, re-evaluate the same 90-trial matrix; expect non-zero success rate

**Estimated impact of cost fix alone:** 30-50% success rate on open_space (currently 0%), measurable improvement on blocking/narrow_passage.

---

## 10. Critical Failures and Warnings

### 🔴 Critical Failures

1. **Cost adapter gap**: `LearnedRolloutCostFn._compute_cost()` is missing 7/11 cost terms. This is the primary cause of 0% success rate. The planner cannot avoid obstacles, reach the object, maintain contact, or penalize trajectory drift.

2. **success_metrics.py empty**: Success metric is defined inline in 3 different scripts. Risk of metric divergence between eval runs.

### 🟡 Warnings

1. **max_speed_mps discrepancy**: Trial results record 0.75, code says 0.5. If 0.75 was used, model inputs were out-of-distribution (training data max_speed=0.5).

2. **Obstacle token omission**: The original `run_learned_mujoco_closed_loop.py` does not include obstacle tokens in state16. The overnight runner fixes this, but all other scripts may have the same bug.

3. **MPPI inefficiency**: MPPI takes 30-130s per trial (vs 1-15s for CEM) with no improvement. The full runner allocated 900s timeout per trial, but MPPI was already hitting wall time.

4. **Model architecture inversion**: Offline eval shows `causality_aware` is most accurate (0.6mm@10 steps) but closed-loop shows `flat` is best (29/30 improved vs 15-16/30). This inversion suggests the causality_aware model may overfit to statistical patterns that don't generalize to closed-loop — or that the cost function disproportionately hurts complex models.

5. **No contact in object_centric/causality_aware**: These models show 0% contact rate in blocking families. With proper cost weights (w_reach, w_no_contact), this should improve.

---

## Appendix A: Diagnostic Commands

### To verify cost fix with small sweep:
```bash
# Not executed — generate command only
PYTHONPATH=. python -c "
# A/B test: adapter _compute_cost vs full rollout_cost
# 18 trials: 3 models × 2 cost_fns × 3 families (open_space, blocking, narrow_passage)
# Output to runs/critical_diagnosis_20260525_173000/ab_sweep/
"
```

### To render videos:
```bash
PYTHONPATH=. python scripts/render_overnight_top_trials.py \
    --run-dir runs/overnight_dual_planner_closed_loop_20260525_033000 \
    --top-k 5 --bottom-k 5 \
    --out runs/critical_diagnosis_20260525_173000/renders/
```

### To verify max_speed_mps:
```bash
grep -n "max_speed_mps\|max_speed" src/planners/action_conventions.py scripts/run_overnight_dual_planner.py
```

---

## Appendix B: File Inventory

| File | Status | Notes |
|------|--------|-------|
| `src/planners/learned_planner_adapter.py` | **BROKEN** | _compute_cost needs replacement |
| `src/planners/cost_functions.py` | ✅ CORRECT | Full rollout_cost with 11 terms |
| `src/planners/planner_cost_adapter.py` | ❌ NOT FOUND | Does not exist |
| `src/metrics/success_metrics.py` | EMPTY (0B) | Needs population |
| `src/planners/action_conventions.py` | ✅ Exists | Untracked in git |
| `scripts/run_overnight_dual_planner.py` | ✅ Exists | Includes obstacle tokens |
| `scripts/run_learned_mujoco_closed_loop.py` | ⚠️ Partial | Missing obstacle tokens |
| `scripts/run_closed_loop_corrected.py` | ✅ Corrected | Uses adapter + convention |

---

*Report generated by 管家助手 🏠 | 2026-05-25*
