# Learned Closed-Loop MuJoCo Smoke Report

**Date:** 2026-05-24  
**Status:** SMOKE TEST — NOT paper main result  

---

## Executive Summary

All three learned models (flat, object_centric, causality_aware) can run in MuJoCo closed-loop without crashes, but **none can successfully push the object to the goal**. The best_dist equals final_dist in all cases, indicating the learned rollout provides no useful planning guidance beyond the initial state. The most likely bottleneck is action scale mismatch between training data (MPPI physical actions) and CEM search space (normalized actions).

---

## Phase Status

| Phase | Status | Evidence |
|-------|--------|----------|
| 0: Artifact precheck | ✅ PASS | All checkpoints/normalizers loadable |
| 1: Env + state extraction | ✅ PASS | reset/step/state16 extraction verified |
| 2: Learned planner adapter | ✅ PASS | CEM_FALLBACK with learned rollout |
| 3: Three-model smoke | ✅ PASS | All 3 models × 3 templates completed |
| 4: Results summary | ✅ DONE | This report |

---

## Planner Backend

- **Backend:** CEM_FALLBACK_LEARNED_ROLLOUT
- **MPPI best params source:** User-provided (speed=0.3, T=0.1, H=100, samples=2048)
- **Fallback used:** YES — MPPI does not support learned rollout cost_fn directly
- **Reason:** Existing MPPI implementation expects oracle MuJoCo rollout, not learned model
- **CEM config:** horizon=10, samples=128, elites=16, iterations=3, action_range=[-1,1]

---

## Closed-Loop Results

| Model | Success | Initial Dist | Final Dist | Best Dist | Improved? | Contact Rate | Collision Rate |
|-------|---------|--------------|------------|-----------|-----------|--------------|----------------|
| flat | 0/3 | ~0.25m | 0.2535m | 0.2535m | NO | 1/9 (11%) | 0/9 |
| object_centric | 0/3 | ~0.26m | 0.2579m | 0.2579m | NO | 0/9 (0%) | 0/9 |
| causality_aware | 0/3 | ~0.25m | 0.2540m | 0.2540m | NO | 1/9 (11%) | 0/9 |

**Key observation:** In all 9 runs (3 models × 3 templates), best_dist = final_dist. The learned rollout model cannot guide the planner to find actions that reduce the object-goal distance.

---

## Interpretation

### Can the models complete the task?
**No.** None of the three models achieved success (< 2cm position error, < 10° rotation error).

### Can any model at least reduce the distance?
**No.** The best_dist equals final_dist in all cases, meaning the CEM planner with learned rollout cannot find action sequences that move the object closer to the goal.

### Most likely bottleneck

1. **Action scale mismatch (HIGH PROBABILITY):**
   - Training data: `actions_physical` in [-0.3, 0.3] m/s
   - CEM search: normalized actions in [-1, 1]
   - The learned model was trained on physical actions but the CEM searches in normalized space
   - This means the model receives out-of-distribution action inputs during planning

2. **Learned rollout accuracy (MEDIUM PROBABILITY):**
   - One-step RMSE ~0.016m compounds over 10 planning steps
   - The rollout predictions are too inaccurate for meaningful planning

3. **CEM configuration (MEDIUM PROBABILITY):**
   - Only 128 samples with 3 iterations may be insufficient
   - Horizon=10 may be too short for this task

4. **Normalizer mismatch (LOW PROBABILITY):**
   - The normalizer was fitted on training data features
   - V2 features may have different distributions

---

## Limitations

1. **max_templates=3** — very small sample, not statistically significant
2. **Smoke only** — not a formal evaluation
3. **Planner backend is CEM fallback** — not the MPPI best config
4. **Trained on MPPI-generated data** — may not transfer to CEM planning
5. **Action scale not verified** — likely mismatch between training and inference
6. **No initial_dist recorded** — only final_dist and best_dist available

---

## Next Actions

1. **Fix action scale mismatch:**
   - Verify what action space the model was trained on
   - Ensure CEM uses the same action space
   - Consider: train on normalized actions [-1,1] or search in physical space [-0.3,0.3]

2. **Expand to 10 templates per model**

3. **Tune CEM config:**
   - Increase samples to 512+
   - Increase iterations to 5+
   - Try horizon=20

4. **Add MuJoCo oracle MPPI baseline:**
   - Compare learned model vs oracle on same templates
   - This establishes the upper bound

5. **Add rollout ranking accuracy eval:**
   - Does the learned model correctly rank good actions vs bad actions?
   - This is a prerequisite for useful planning

6. **Consider retraining with normalized actions:**
   - If action scale mismatch is confirmed, retrain on normalized [-1,1] actions
