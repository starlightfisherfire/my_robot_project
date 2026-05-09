# Experiment Gates: Paper 1 Implementation Milestones

**Last updated:** 2026-05-09

This document breaks down the Paper 1 experimental pipeline into discrete gates with clear pass criteria.

---

## Gate 0: Repo Skeleton / Configs ✅

**Goal:** Basic repo structure and config files

**Files:**
- `CLAUDE.md`
- `configs/planner/cem_mpc.yaml`
- `configs/planner/cost_weights.yaml`
- `configs/splits.yaml`
- `configs/train/*.yaml`
- `configs/eval/*.yaml`

**Command:** N/A (manual setup)

**Pass criteria:**
- ✅ Repo structure exists
- ✅ Config files created

**Status:** ✅ PASS (2026-04-20)

**Proves:** Repo is initialized

**Does NOT prove:** Configs are correct or final

---

## Gate 1: Reset Template Sanity ✅

**Goal:** Generate and validate reset templates

**Files:**
- `src/interventions/shape_families.py`
- `src/interventions/layout_families.py`
- `src/interventions/sampling_rules.py`
- `scripts/generate_reset_templates.py`
- `data/sim/metadata/reset_templates_v0.json`

**Command:**
```bash
PYTHONPATH=. python scripts/generate_reset_templates.py --num-per-split 20
PYTHONPATH=. python scripts/check_mpc_capacity.py --mode state_sanity
```

**Pass criteria:**
- ✅ 140 templates generated (20 per split × 7 splits)
- ✅ Schema validation passed
- ✅ No geometric errors
- ✅ Object-goal distances reasonable
- ✅ EE-object distances reasonable
- ✅ Workspace bounds respected

**Status:** ✅ PASS (2026-04-21, 140/140 templates)

**Proves:** Reset templates are geometrically reasonable

**Does NOT prove:** Templates are solvable by CEM-MPC

---

## Gate 2: Model Smoke Tests ✅

**Goal:** Verify model architecture forward/backward pass

**Files:**
- `src/models/encoders.py`
- `src/models/heads.py`
- `src/models/rig_world.py`
- `src/models/losses.py`
- `src/data/state_normalizer.py`

**Command:**
```bash
PYTHONPATH=. python scripts/debug_encoder_variants.py
PYTHONPATH=. python scripts/debug_rig_world_model.py
PYTHONPATH=. python scripts/debug_state_normalizer.py
```

**Pass criteria:**
- ✅ FlatEncoder forward/backward works
- ✅ ObjectCentricEncoder forward/backward works
- ✅ CausalityAwareEncoder forward/backward works
- ✅ DynamicsHead forward/backward works
- ✅ SubgoalHead forward/backward works
- ✅ RIGWorldModel unified interface works
- ✅ StateNormalizer fit/transform works

**Status:** ✅ PASS (2026-04-26)

**Proves:** Model architecture is syntactically correct, gradient flow works

**Does NOT prove:** Models can learn from real data, models generalize to OOD

---

## Gate 3: Toy Oracle-MPC ✅

**Goal:** Verify oracle rollout + CEM-MPC interface with toy dynamics

**Files:**
- `src/envs/toy_push_env.py`
- `src/planners/oracle_rollout.py`
- `src/planners/cem_mpc.py`
- `src/metrics/toy_oracle_capacity.py`

**Command:**
```bash
PYTHONPATH=. python scripts/check_mpc_capacity.py \
  --mode toy_oracle_mpc \
  --split train_sim_id \
  --max-templates 20
```

**Pass criteria:**
- ✅ 20/20 templates: planned_cost < zero_cost
- ✅ 20/20 templates: restore_state correct
- ✅ Program does not crash
- ✅ Costs are finite

**Status:** ✅ PASS (2026-04-26, 20/20 templates)

**Proves:** Oracle rollout + CEM-MPC interface is correct

**Does NOT prove:** MuJoCo oracle-MPC works, real layout OOD capacity (ToyPushEnv ignores obstacles)

---

## Gate 4: Minimal MuJoCo Env ✅

**Goal:** MuJoCo environment scaffold with reset/step/clone/restore

**Files:**
- `src/envs/mujoco_push_env.py`
- `scripts/debug_mujoco_env.py`

**Command:**
```bash
PYTHONPATH=. python scripts/debug_mujoco_env.py
```

**Pass criteria:**
- ✅ MuJoCo XML loads
- ✅ reset() works
- ✅ reset_from_template() works
- ✅ step(action) works
- ✅ clone_state() / restore_state() works
- ✅ get_object_pose() / get_ee_pos() / get_goal_pose() works
- ✅ get_contact_flag() detects pusher-object contact

**Status:** ✅ PASS (2026-05-06)

**Proves:** MuJoCo environment interface is correct

**Does NOT prove:** Obstacles are instantiated, collision detection works, task can be solved

**Limitation:** Obstacles not instantiated, collision detection placeholder

---

## Gate 5: MuJoCo Oracle Rollout ✅

**Goal:** Roll out action sequences using MuJoCo true dynamics

**Files:**
- `src/planners/mujoco_oracle_rollout.py`
- `scripts/debug_mujoco_oracle_rollout.py`

**Command:**
```bash
PYTHONPATH=. python scripts/debug_mujoco_oracle_rollout.py
```

**Pass criteria:**
- ✅ rollout_action_sequence_mujoco() runs
- ✅ restore_state works after rollout
- ✅ Hand-coded right push produces contact
- ✅ Object x increases as expected
- ✅ Rollout cost is finite

**Status:** ✅ PASS (2026-05-06)

**Proves:** MuJoCo oracle rollout interface works, true dynamics rollout is correct

**Does NOT prove:** CEM-MPC can find good action sequences, task can be solved

---

## Gate 6: MuJoCo Oracle-MPC Interface Smoke Test ✅

**Goal:** Verify CEM-MPC + MuJoCo oracle rollout interface works

**Files:**
- `src/metrics/mujoco_oracle_capacity.py`
- `scripts/check_mpc_capacity.py`

**Command:**
```bash
PYTHONPATH=. python scripts/check_mpc_capacity.py \
  --mode mujoco_oracle_mpc \
  --split train_sim_id \
  --max-templates 5 \
  --horizon 80 \
  --num-samples 1536 \
  --num-elites 128 \
  --num-iterations 7
```

**Pass criteria:**
- ✅ Program does not crash
- ✅ Costs are finite
- ✅ 5/5 templates: planned_cost < zero_cost
- ✅ 5/5 templates: best_min_dist < initial_dist
- ✅ 5/5 templates: restore_state correct
- ✅ Prints "mujoco oracle mpc capacity check ok"

**Status:** ✅ PASS (2026-05-08, 5/5 interface checks)

**Proves:** MuJoCo Oracle-MPC interface works, CEM can improve over zero-action baseline

**Does NOT prove:** Task success (success_rate = 0/5), layout OOD capacity (obstacles not instantiated)

**Limitation:** success_rate = 0, final_dist does not reach success threshold, obstacles not tested

---

## Gate 7: MuJoCo Oracle-MPC Task-Solving Capacity ⬜

**Goal:** Establish that Oracle-MPC can solve push-to-pose task with true dynamics

**Files:**
- `src/metrics/mujoco_oracle_capacity.py` (may need tuning)
- `configs/planner/oracle_capacity_strong.yaml` (new)
- `configs/planner/cost_weights.yaml` (may need tuning)

**Command:**
```bash
PYTHONPATH=. python scripts/check_mpc_capacity.py \
  --mode mujoco_oracle_mpc \
  --split train_sim_id \
  --max-templates 20 \
  --horizon 100 \
  --num-samples 2048 \
  --num-elites 256 \
  --num-iterations 10
```

**Pass criteria:**
- ⬜ success_rate > 80% on train_sim_id (20 templates)
- ⬜ mean_final_dist < 0.05 (success threshold)
- ⬜ No collisions (max_collision < 0.5)
- ⬜ Costs are finite and reasonable

**Status:** ⬜ NOT STARTED (waiting for user decision)

**Proves:** Task is solvable with true dynamics, establishes upper bound for learned model

**Does NOT prove:** Layout OOD capacity (obstacles not instantiated), learned model can achieve this

**Next action:** Tune horizon / num_samples / num_iterations / cost_weights until success_rate > 80%

---

## Gate 8: Obstacles-Enabled Layout Capacity ⬜

**Goal:** Test Oracle-MPC capacity on layout OOD with obstacles instantiated

**Files:**
- `src/envs/mujoco_push_env.py` (updated to instantiate obstacles)
- `src/metrics/mujoco_oracle_capacity.py` (updated for collision detection)

**Command:**
```bash
PYTHONPATH=. python scripts/check_mpc_capacity.py \
  --mode mujoco_oracle_mpc \
  --split test_sim_layout_ood_blocking \
  --max-templates 20

PYTHONPATH=. python scripts/check_mpc_capacity.py \
  --mode mujoco_oracle_mpc \
  --split test_sim_layout_ood_narrow_passage \
  --max-templates 20

PYTHONPATH=. python scripts/check_mpc_capacity.py \
  --mode mujoco_oracle_mpc \
  --split test_sim_layout_ood_edge_goal \
  --max-templates 20
```

**Pass criteria:**
- ⬜ Obstacles are instantiated in MuJoCo
- ⬜ Collision detection works
- ⬜ success_rate measured on blocking / narrow_passage / edge_goal
- ⬜ Layout OOD degradation quantified (e.g., train_sim_id: 80%, blocking: 50%)

**Status:** ⬜ NOT STARTED (blocked by obstacles not instantiated)

**Proves:** Layout OOD difficulty is real, Oracle-MPC capacity on layout OOD

**Does NOT prove:** Learned model can handle layout OOD

**Next action:** Implement obstacles instantiation in MujocoPushEnv

---

## Gate 9: Sim Data Collection ⬜

**Goal:** Collect training data for learned models

**Files:**
- `scripts/collect_sim_data.py` (new)
- `data/sim/episodes/train_sim_id/episode_*.npz` (new)
- `data/sim/metadata/train_sim_id_metadata.json` (new)

**Command:**
```bash
PYTHONPATH=. python scripts/collect_sim_data.py \
  --split train_sim_id \
  --num-episodes 1000 \
  --policy oracle_mpc
```

**Pass criteria:**
- ⬜ 1000+ episodes collected for train_sim_id
- ⬜ 200+ episodes collected for val_sim_id
- ⬜ Metadata saved correctly
- ⬜ StateNormalizer fitted ONLY on train_sim_id
- ⬜ No test data leakage

**Status:** ⬜ NOT STARTED (blocked by Oracle-MPC capacity)

**Proves:** Training data is available

**Does NOT prove:** Data quality is sufficient for learning

**Next action:** Implement collect_sim_data.py after Gate 7 or Gate 8

---

## Gate 10: Learned High-Level Model Training ⬜

**Goal:** Train flat/object/causal encoders + heads

**Files:**
- `scripts/train_high_level.py` (new)
- `runs/train/flat/checkpoint_best.pt` (new)
- `runs/train/object_centric/checkpoint_best.pt` (new)
- `runs/train/causality_aware/checkpoint_best.pt` (new)

**Command:**
```bash
PYTHONPATH=. python scripts/train_high_level.py \
  --model-type flat \
  --train-split train_sim_id \
  --val-split val_sim_id

PYTHONPATH=. python scripts/train_high_level.py \
  --model-type object_centric \
  --train-split train_sim_id \
  --val-split val_sim_id

PYTHONPATH=. python scripts/train_high_level.py \
  --model-type causality_aware \
  --train-split train_sim_id \
  --val-split val_sim_id
```

**Pass criteria:**
- ⬜ All three models train without crashing
- ⬜ Training loss decreases
- ⬜ Validation loss decreases
- ⬜ Checkpoints saved
- ⬜ StateNormalizer fitted ONLY on train_sim_id
- ⬜ Same hyperparameters for all three variants

**Status:** ⬜ NOT STARTED (blocked by data collection)

**Proves:** Models can learn from data

**Does NOT prove:** Models generalize to OOD, representation quality differences

**Next action:** Implement train_high_level.py after Gate 9

---

## Gate 11: Learned Model + MPC ⬜

**Goal:** Evaluate learned models with fixed CEM-MPC

**Files:**
- `src/planners/rollout_model.py` (new)
- `scripts/eval_learned_mpc.py` (new)
- `configs/eval/learned_eval_default.yaml` (new)

**Command:**
```bash
PYTHONPATH=. python scripts/eval_learned_mpc.py \
  --model-type flat \
  --checkpoint runs/train/flat/checkpoint_best.pt \
  --split test_sim_id

PYTHONPATH=. python scripts/eval_learned_mpc.py \
  --model-type object_centric \
  --checkpoint runs/train/object_centric/checkpoint_best.pt \
  --split test_sim_id

PYTHONPATH=. python scripts/eval_learned_mpc.py \
  --model-type causality_aware \
  --checkpoint runs/train/causality_aware/checkpoint_best.pt \
  --split test_sim_id
```

**Pass criteria:**
- ⬜ All three models run without crashing
- ⬜ Learned rollout produces finite predictions
- ⬜ CEM-MPC converges
- ⬜ success_rate measured on test_sim_id
- ⬜ Same planner config for all three variants

**Status:** ⬜ NOT STARTED (blocked by model training)

**Proves:** Learned model + MPC pipeline works

**Does NOT prove:** OOD gap, representation quality differences

**Next action:** Implement rollout_model.py and eval_learned_mpc.py after Gate 10

---

## Gate 12: OOD Gap and Representation Comparison ⬜

**Goal:** Compare flat/object/causal on ID and OOD splits

**Files:**
- `scripts/eval_learned_mpc.py` (run on all splits)
- `scripts/analyze_ood_gap.py` (new)

**Command:**
```bash
# Evaluate all three models on all splits
for model in flat object_centric causality_aware; do
  for split in test_sim_id test_sim_layout_ood_blocking test_sim_layout_ood_narrow_passage test_sim_layout_ood_edge_goal test_sim_shape_ood_L; do
    PYTHONPATH=. python scripts/eval_learned_mpc.py \
      --model-type $model \
      --checkpoint runs/train/$model/checkpoint_best.pt \
      --split $split
  done
done

# Analyze OOD gap
PYTHONPATH=. python scripts/analyze_ood_gap.py
```

**Pass criteria:**
- ⬜ All models evaluated on all splits
- ⬜ OOD gap quantified (ID success_rate - OOD success_rate)
- ⬜ Causality-aware shows smaller OOD gap than flat/object-centric
- ⬜ Statistical significance tested

**Status:** ⬜ NOT STARTED (blocked by learned model + MPC)

**Proves:** Causality-aware representation improves OOD generalization (Paper 1 main claim)

**Does NOT prove:** Real robot transfer, RGB input, other tasks

**Next action:** Run full evaluation after Gate 11

---

## Gate 13: Real-ID Adapted OOD ⬜

**Goal:** Adapt to real robot ID data, test on real robot OOD

**Files:**
- `scripts/adapt_to_real.py` (new)
- `scripts/eval_real_robot.py` (new)

**Command:**
```bash
# Adapt models to real robot ID data
PYTHONPATH=. python scripts/adapt_to_real.py \
  --model-type causality_aware \
  --sim-checkpoint runs/train/causality_aware/checkpoint_best.pt \
  --real-data data/real/train_real_id

# Evaluate on real robot OOD
PYTHONPATH=. python scripts/eval_real_robot.py \
  --model-type causality_aware \
  --checkpoint runs/adapt/causality_aware/checkpoint_adapted.pt \
  --split test_real_layout_ood
```

**Pass criteria:**
- ⬜ Adaptation to real robot ID data works
- ⬜ Real robot OOD evaluation works
- ⬜ Sim-to-real gap quantified
- ⬜ Causality-aware shows better real OOD than flat/object-centric

**Status:** ⬜ NOT STARTED (blocked by sim OOD evaluation)

**Proves:** Sim-to-real transfer, real robot OOD generalization

**Does NOT prove:** Other tasks, other robots, RGB input

**Next action:** Real robot integration after Gate 12

---

## Current Gate Status Summary

| Gate | Name | Status | Date | Blocker |
|------|------|--------|------|---------|
| 0 | Repo skeleton | ✅ PASS | 2026-04-20 | None |
| 1 | Reset template sanity | ✅ PASS | 2026-04-21 | None |
| 2 | Model smoke tests | ✅ PASS | 2026-04-26 | None |
| 3 | Toy oracle-MPC | ✅ PASS | 2026-04-26 | None |
| 4 | Minimal MuJoCo env | ✅ PASS | 2026-05-06 | None |
| 5 | MuJoCo oracle rollout | ✅ PASS | 2026-05-06 | None |
| 6 | MuJoCo Oracle-MPC interface | ✅ PASS | 2026-05-08 | None |
| 7 | Oracle-MPC task capacity | ⬜ NOT STARTED | - | User decision |
| 8 | Obstacles-enabled layout | ⬜ NOT STARTED | - | Obstacles not instantiated |
| 9 | Sim data collection | ⬜ NOT STARTED | - | Gate 7 or 8 |
| 10 | Learned model training | ⬜ NOT STARTED | - | Gate 9 |
| 11 | Learned model + MPC | ⬜ NOT STARTED | - | Gate 10 |
| 12 | OOD gap comparison | ⬜ NOT STARTED | - | Gate 11 |
| 13 | Real-ID adapted OOD | ⬜ NOT STARTED | - | Gate 12 |

---

## Critical Path

```
Gate 6 (DONE) → Gate 7 OR Gate 8 → Gate 9 → Gate 10 → Gate 11 → Gate 12 → Gate 13
                     ↓                ↓
              Task capacity    Layout capacity
```

**Current decision point:** Gate 7 (tune Oracle-MPC) OR Gate 8 (implement obstacles) first?

---

## Next Action

**User must decide:**
- **Option A:** Proceed to Gate 7 (tune Oracle-MPC for task success)
- **Option B:** Proceed to Gate 8 (implement obstacles-enabled MuJoCo env)

**Do NOT proceed without user confirmation.**
