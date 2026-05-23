# Experiment Gates: Paper 1 Implementation Milestones

**Last updated:** 2026-05-23

This document defines the Paper 1 experimental pipeline gates with clear pass criteria.

---

## Gate 0: Repo Skeleton / Configs ✅

**Goal:** Basic repo structure and config files

**Status:** ✅ PASS (2026-04-20)

**Evidence:** Repo structure exists, all config files created (train configs filled 2026-05-23)

**Pass criteria:**
- ✅ Repo structure exists
- ✅ Config files created
- ✅ Train configs non-empty (rig_world_shared.yaml, flat/object/causal yaml)

---

## Gate 1: Reset Template Sanity ✅

**Goal:** Generate and validate reset templates

**Status:** ✅ PASS (2026-04-21, 140/140 templates)

**Evidence:** reset_templates_v0.json, schema validation passed

**Pass criteria:**
- ✅ Templates generated
- ✅ Schema validation passed
- ✅ Geometric constraints respected

---

## Gate 2: Model Smoke Tests ✅

**Goal:** Verify model architecture forward/backward pass

**Status:** ✅ PASS (2026-04-26)

**Evidence:** debug_encoder_variants.py, debug_rig_world_model.py

**Pass criteria:**
- ✅ All three encoders forward/backward work
- ✅ DynamicsHead/SubgoalHead forward/backward work
- ✅ RIGWorldModel unified interface works

---

## Gate 3: Toy Oracle-MPC ✅

**Goal:** Verify oracle rollout + CEM-MPC interface with toy dynamics

**Status:** ✅ PASS (2026-04-26, 20/20 templates)

**Evidence:** ToyPushEnv + CEM-MPC interface verified

---

## Gate 4: Minimal MuJoCo Env ✅

**Goal:** MuJoCo environment scaffold with reset/step/clone/restore

**Status:** ✅ PASS (2026-05-06)

**Evidence:** mujoco_push_env.py interface verified

---

## Gate 5: MuJoCo Oracle Rollout ✅

**Goal:** Roll out action sequences using MuJoCo true dynamics

**Status:** ✅ PASS (2026-05-06)

**Evidence:** mujoco_oracle_rollout.py interface verified

---

## Gate 6: MuJoCo Oracle-MPC Interface Smoke ✅

**Goal:** Verify CEM-MPC + MuJoCo oracle rollout interface works

**Status:** ✅ PASS (2026-05-08, 5/5 interface checks)

**Evidence:** check_mpc_capacity.py interface smoke passed

---

## Gate 7: MuJoCo Oracle-MPC Task-Solving Capacity ⚠️ PARTIAL

**Goal:** Establish that Oracle-MPC can solve push-to-pose task with true dynamics

**Status:** ⚠️ PARTIAL

**Evidence:**
- ID/open_space: boundary_video_night2 mm-level precision ✅
- Layout OOD/blocking: horizon140_sweep partial success ⚠️
- Layout OOD/passage: mppi_stage2c 94.3% (sp050_T0.2) ✅
- Layout OOD/narrow_passage: not fully tested ❌
- Layout OOD/edge_goal: not tested ❌

**Pass criteria:**
- ✅ Oracle-MPC can solve ID/open_space with <1cm error
- ⚠️ Oracle-MPC can solve most layout OOD variants (>80%)
- ❌ All layout OOD families systematically tested

**Next:**
- Run Oracle-MPC capacity check on all layout OOD families
- Use consistent config (horizon, samples, iterations)

---

## Gate 8: canonical_state16 Dataset Coverage ❌ FAIL

**Goal:** Sufficient state16 data for all splits/families

**Status:** ❌ FAIL

**Evidence:**
- layout_ood_state16_v0: 36 episodes (open, blocking_easy, blocking_medium), all success=False
- layout_ood_state16_v1_smoke: 18 episodes (passage), all success=False
- Total: 54 episodes, 0 successful

**Pass criteria:**
- ❌ ≥500 episodes per split for meaningful training
- ❌ ≥50% success rate in collected data
- ❌ Coverage of all families: open_space, mild_offset, blocking, narrow_passage, edge_goal
- ❌ Shape OOD data (L-shape)

**Next:**
- Re-run data collection with better Oracle-MPC configs
- Use mppi_stage2c best config (sp050_T0.2) for data generation
- Target: 100+ episodes per family with >50% success

---

## Gate 9: Learned Rollout ID Smoke 🔧 IN PROGRESS

**Goal:** Verify learned rollout model can produce valid predictions

**Status:** 🔧 IN PROGRESS (2026-05-23)

**Evidence:** rollout_model.py implemented, check_model_interfaces.py created

**Pass criteria:**
- ⬜ LearnedRolloutModel.forward_step produces finite [3] output
- ⬜ LearnedRolloutModel.rollout_sequence produces finite [H+1, 3] trajectory
- ⬜ Model interfaces verified (check_model_interfaces.py passes)
- ⬜ Training pipeline works on existing data

**Next:**
1. Run `python scripts/check_model_interfaces.py`
2. Run `python scripts/train_high_level.py --config ... --model flat --smoke`
3. Run `python scripts/run_learned_mpc_eval.py ...`

---

## Gate 10: Learned Rollout + CEM-MPC ID ⬜ NEXT

**Goal:** learned rollout + CEM-MPC can solve ID task

**Status:** ⬜ NOT STARTED (blocked by Gate 9)

**Pass criteria:**
- Learned rollout model trained on ID data
- CEM-MPC with learned rollout achieves >50% success on ID/open_space
- Final position error <5cm

**Evidence required:**
- `runs/learned_mpc_eval/*/summary.json`
- Success rate, final_dist, best_min_dist

---

## Gate 11: Flat vs Object vs Causal ID Comparison ⬜

**Goal:** Compare three encoder variants on ID task

**Status:** ⬜ NOT STARTED (blocked by Gate 10)

**Pass criteria:**
- All three models trained with same data/config
- All three evaluated with same CEM-MPC config
- Report dynamics_rmse, subgoal_rmse, success_rate, final_dist

---

## Gate 12: OOD Gap and Representation Comparison ⬜

**Goal:** Causality-aware representation improves OOD generalization

**Status:** ⬜ NOT STARTED (blocked by Gate 11)

**Pass criteria:**
- Layout OOD evaluation on blocking, narrow_passage, edge_goal
- Shape OOD evaluation on L-shape
- OOD gap = ID_performance - OOD_performance
- Causality-aware has smaller OOD gap than flat and object_centric

---

## Gate 13: Real-ID Adapted OOD ⬜

**Goal:** Sim-to-real transfer and real robot OOD generalization

**Status:** ⬜ NOT STARTED (blocked by Gate 12)

**Pass criteria:**
- Zero-shot sim-to-real transfer on real ID
- Real-ID adapted OOD generalization
- No leakage of OOD test information in adaptation set
