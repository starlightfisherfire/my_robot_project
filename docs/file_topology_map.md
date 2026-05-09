# File Topology Map: Paper 1 Experiment Flow

**Last updated:** 2026-05-09

This document describes the file topology organized by Paper 1 experimental flow, not just by directory structure.

---

## Main Experimental Chain

```
A. Reset Template / Split Generation
    ↓
B. Environment Scaffold
    ↓
C. Oracle Rollout
    ↓
D. Oracle-MPC Smoke Test
    ↓
E. Future: Full Oracle-MPC Task Capacity
    ↓
F. Future: Sim Data Collection
    ↓
G. Future: Learned Model Training
    ↓
H. Future: Learned Model + MPC Eval
```

---

## A. Reset Template / Split Generation

**Purpose:** Generate reset templates with split/layout/shape family labels

```
src/interventions/shape_families.py
    - SHAPE_FAMILIES = {"T", "L", "other"}
    - get_shape_config(shape_name)
    ↓
src/interventions/layout_families.py
    - LAYOUT_FAMILIES = {"open_space", "mild_offset", "blocking", "narrow_passage", "edge_goal"}
    - get_layout_config(layout_name)
    ↓
src/interventions/sampling_rules.py
    - validate_reset_templates(templates)
    - check schema compliance
    ↓
scripts/generate_reset_templates.py
    - main()
    - CLI: --num-per-split, --seed
    ↓
data/sim/metadata/reset_templates_v0.json
    - 140 templates
    - train_sim_id: 20 templates (T-shape, open_space/mild_offset)
    - val_sim_id: 20 templates
    - test_sim_id: 20 templates
    - test_sim_layout_ood_blocking: 20 templates (T-shape, blocking)
    - test_sim_layout_ood_narrow_passage: 20 templates (T-shape, narrow_passage)
    - test_sim_layout_ood_edge_goal: 20 templates (T-shape, edge_goal)
    - test_sim_shape_ood_L: 20 templates (L-shape, open_space/mild_offset)
    ↓
src/interventions/reset_template_loader.py
    - load_reset_templates(path)
    - get_templates_by_split(templates, split)
    - template_to_episode_metadata(template)
    ↓
src/data/metadata_schema.py
    - EpisodeMetadata (dataclass)
    - Pose2D (dataclass)
    - ObstacleMetadata (dataclass)
```

**Debug command:**
```bash
PYTHONPATH=. python scripts/generate_reset_templates.py --num-per-split 3
PYTHONPATH=. python scripts/debug_reset_templates.py
```

**Status:** ✅ PASS (140 templates generated, schema validated)

**Limitation:** Obstacles not yet instantiated in MuJoCo

---

## B. Environment Scaffold

**Purpose:** MuJoCo environment with reset/step/clone/restore interface

```
src/envs/mujoco_push_env.py
    - MujocoPushEnv
    - MujocoPushState (dataclass)
    - MINIMAL_PUSH_XML (hardcoded XML)
    - reset(object_pose, goal_pose, ee_pos)
    - reset_from_template(template)
    - step(action)
    - clone_state() → MujocoPushState
    - restore_state(state)
    - get_object_pose() → [x, y, theta]
    - get_ee_pos() → [x, y]
    - get_goal_pose() → [x, y, theta]
    - get_contact_flag() → float
    - get_collision_flag() → float (placeholder)
    ↓
scripts/debug_mujoco_env.py
    - test reset / step / clone / restore
    - test reset_from_template
    - test contact detection
```

**Debug command:**
```bash
PYTHONPATH=. python scripts/debug_mujoco_env.py
```

**Status:** ✅ PASS (interface works, contact detection works)

**Limitation:**
- **Obstacles not instantiated** from templates
- goal_site is static XML site, not dynamically updated
- collision detection is placeholder (always False)

---

## C. Oracle Rollout

**Purpose:** Roll out action sequences using true MuJoCo dynamics

```
src/planners/cost_functions.py
    - CostWeights (dataclass)
    - rollout_cost(predicted_object_poses, ee_positions, action_sequence, goal_pose, weights, contact_flags, collision_flags)
    - wrap_angle(angle)
    ↓
src/planners/mujoco_oracle_rollout.py
    - MujocoOracleRolloutResult (dataclass)
    - rollout_action_sequence_mujoco(env, action_sequence, restore_state=True)
    - mujoco_oracle_rollout_cost(env, action_sequence, goal_pose, weights, restore_state=True)
    ↓
scripts/debug_mujoco_oracle_rollout.py
    - test rollout with hand-coded actions
    - test restore_state
    - test contact detection
    - test cost computation
```

**Debug command:**
```bash
PYTHONPATH=. python scripts/debug_mujoco_oracle_rollout.py
```

**Status:** ✅ PASS (rollout works, restore_state correct, cost finite)

**Limitation:** Does not test with obstacles

---

## D. Oracle-MPC Smoke Test

**Purpose:** Verify CEM-MPC + oracle rollout interface works

```
src/planners/cem_mpc.py
    - CEMMPC
    - CEMResult (dataclass)
    - optimize(cost_fn, init_mean, init_std) → CEMResult
    - plan(cost_fn) → (first_action, CEMResult)
    ↓
src/metrics/mujoco_oracle_capacity.py
    - make_default_mujoco_cost_weights() → CostWeights
    - evaluate_one_template_mujoco_oracle_mpc(template, horizon, num_samples, num_elites, num_iterations, seed, success_dist_threshold)
    - run_mujoco_oracle_mpc_capacity(templates, ...)
    - save_mujoco_oracle_mpc_report(report, path)
    ↓
scripts/check_mpc_capacity.py
    - parse_args()
    - run_mujoco_oracle_mpc_mode(args)
    - CLI: --mode mujoco_oracle_mpc --split train_sim_id --max-templates 5 --horizon 80 --num-samples 1536 --num-elites 128 --num-iterations 7
```

**Debug command:**
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

**Status:** ✅ PASS (interface smoke test)

**Result:**
- 5/5 templates: planned_cost < zero_cost
- 5/5 templates: best_min_dist < initial_dist
- 5/5 templates: restore_state correct
- Prints "mujoco oracle mpc capacity check ok"

**Limitation:**
- **success_rate = 0/5** (task not solved yet)
- final_dist does not reach success threshold (0.05)
- Obstacles not tested

---

## E. Future: Full Oracle-MPC Task Capacity

**Purpose:** Establish upper bound for learned model performance

**Required steps:**
1. **Option A:** Tune CEM-MPC parameters for task success
   - Increase horizon / num_samples / num_iterations
   - Tune cost weights
   - Goal: success_rate > 80% on train_sim_id

2. **Option B:** Implement obstacles-enabled MuJoCo env first
   - Modify MujocoPushEnv to instantiate obstacles from templates
   - Add collision detection
   - Then run full Oracle-MPC capacity on layout OOD

**Expected files:**
```
src/envs/mujoco_push_env.py (updated)
    - instantiate_obstacles_from_template(template)
    - dynamic XML generation for obstacles
    - collision detection with obstacles

configs/planner/oracle_capacity_strong.yaml (new)
    - horizon: 100+ (longer than learned eval)
    - num_samples: 2048+ (more than learned eval)
    - num_iterations: 10+ (more than learned eval)

runs/oracle_capacity/train_sim_id_report.json
runs/oracle_capacity/test_sim_layout_ood_blocking_report.json
runs/oracle_capacity/test_sim_layout_ood_narrow_passage_report.json
runs/oracle_capacity/test_sim_layout_ood_edge_goal_report.json
runs/oracle_capacity/test_sim_shape_ood_L_report.json
```

**Status:** ⬜ NOT STARTED (waiting for user decision)

---

## F. Future: Sim Data Collection

**Purpose:** Collect training data for learned models

**Required steps:**
1. Use Oracle-MPC or exploratory policy to collect episodes
2. Save episodes with metadata
3. Validate split protocol (no test data leakage)

**Expected files:**
```
scripts/collect_sim_data.py (new)
    - collect_episodes_with_oracle_mpc(templates, policy, num_episodes)
    - save_episode_data(episode, metadata, path)

data/sim/episodes/train_sim_id/episode_*.npz
data/sim/episodes/val_sim_id/episode_*.npz
data/sim/metadata/train_sim_id_metadata.json
data/sim/metadata/val_sim_id_metadata.json
```

**Status:** ⬜ NOT STARTED (blocked by Oracle-MPC capacity)

---

## G. Future: Learned Model Training

**Purpose:** Train flat/object/causal encoders + heads

**Required steps:**
1. Fit StateNormalizer on train data ONLY
2. Train RIGWorldModel (flat/object/causal) with same hyperparameters
3. Save checkpoints

**Expected files:**
```
src/data/state_normalizer.py (existing)
    - StateNormalizer.fit(train_data)  ⚠️ ONLY on train
    - StateNormalizer.transform(state)

scripts/train_high_level.py (new)
    - load_train_data(split="train_sim_id")
    - fit_normalizer(train_data)
    - train_rig_world_model(model_type, train_data, val_data, normalizer)
    - save_checkpoint(model, normalizer, path)

configs/train/flat.yaml (new)
configs/train/object_centric.yaml (new)
configs/train/causality_aware.yaml (new)

runs/train/flat/checkpoint_best.pt
runs/train/object_centric/checkpoint_best.pt
runs/train/causality_aware/checkpoint_best.pt
```

**Status:** ⬜ NOT STARTED (blocked by data collection)

---

## H. Future: Learned Model + MPC Eval

**Purpose:** Evaluate learned models with fixed CEM-MPC

**Required steps:**
1. Load trained model + normalizer
2. Implement learned rollout interface
3. Run CEM-MPC with learned rollout
4. Evaluate on ID / layout OOD / shape OOD
5. Compare flat / object / causal

**Expected files:**
```
src/planners/rollout_model.py (new)
    - rollout_with_learned_model(model, normalizer, env, action_sequence)
    - learned_rollout_cost(model, normalizer, env, action_sequence, weights)

scripts/eval_learned_mpc.py (new)
    - load_checkpoint(path)
    - evaluate_learned_mpc(model, normalizer, templates, planner_config)
    - save_eval_report(report, path)

configs/eval/learned_eval_default.yaml (new)
    - horizon: 40-60 (shorter than oracle strong)
    - num_samples: 1024 (same for all variants)
    - num_iterations: 5 (same for all variants)

runs/eval/flat/train_sim_id_report.json
runs/eval/flat/test_sim_layout_ood_blocking_report.json
runs/eval/object_centric/train_sim_id_report.json
runs/eval/causality_aware/train_sim_id_report.json
```

**Status:** ⬜ NOT STARTED (blocked by model training)

---

## Current Validated Chain

**What is actually working right now:**

```
reset_templates_v0.json
    ↓
reset_template_loader.load_reset_templates()
    ↓
MujocoPushEnv.reset_from_template()
    ↓
mujoco_oracle_rollout.rollout_action_sequence_mujoco()
    ↓
cost_functions.rollout_cost()
    ↓
CEMMPC.plan(cost_fn)
    ↓
mujoco_oracle_capacity.evaluate_one_template_mujoco_oracle_mpc()
    ↓
check_mpc_capacity.py --mode mujoco_oracle_mpc
    ↓
Result: interface smoke test passed (5/5)
```

**Status:** ✅ Interface works, cost improvement works, restore_state works

**Limitation:** success_rate = 0, obstacles not instantiated

---

## Current Missing Links

1. **Obstacles not instantiated** in MujocoPushEnv
   - Impact: Layout OOD capacity invalid
   - Fix: Implement obstacles-enabled MuJoCo env

2. **Task success not achieved** (success_rate = 0)
   - Impact: Cannot attribute learned model failure to representation
   - Fix: Tune Oracle-MPC or implement obstacles first

3. **No learned dynamics dataset**
   - Impact: Cannot train models
   - Fix: Collect sim data after Oracle-MPC capacity established

4. **No learned rollout validation**
   - Impact: Cannot evaluate learned model + MPC
   - Fix: Implement rollout_model.py after model training

5. **No OOD gap yet**
   - Impact: Cannot compare flat/object/causal
   - Fix: Complete full experimental pipeline

---

## File Importance Ranking

### 🔴 Critical Path (must work first)
1. `mujoco_push_env.py` - Environment interface
2. `mujoco_oracle_rollout.py` - Oracle rollout
3. `cem_mpc.py` - Fixed planner
4. `cost_functions.py` - Cost computation
5. `mujoco_oracle_capacity.py` - Capacity check
6. `check_mpc_capacity.py` - Unified entry

### 🟡 Data Path (needed for training)
7. `reset_template_loader.py` - Template loading
8. `state_normalizer.py` - Normalization (CRITICAL: fit only on train)
9. `layout_families.py` - Layout OOD definitions
10. `shape_families.py` - Shape OOD definitions

### 🟢 Model Path (needed for evaluation)
11. `encoders.py` - Flat/object/causal encoders
12. `heads.py` - Dynamics/subgoal heads
13. `rig_world.py` - Unified model interface
14. `losses.py` - Training losses
15. `rollout_model.py` - Learned rollout (future)

### 🔵 Config / Debug
16. `cem_mpc.yaml` - Planner config
17. `cost_weights.yaml` - Cost weights
18. `splits.yaml` - Split definitions
19. `debug_*.py` - Smoke tests

---

## Dependency Graph

```
                    reset_templates_v0.json
                            ↓
                    reset_template_loader.py
                            ↓
                    MujocoPushEnv
                    ↙           ↘
    mujoco_oracle_rollout    (future) learned_rollout
                ↓                       ↓
        cost_functions          cost_functions
                ↓                       ↓
            CEMMPC.plan()       CEMMPC.plan()
                ↓                       ↓
    mujoco_oracle_capacity    (future) learned_eval
                ↓                       ↓
    check_mpc_capacity.py     eval_learned_mpc.py
```

---

## Validation Commands

```bash
# State sanity (140/140 passed)
PYTHONPATH=. python scripts/check_mpc_capacity.py --mode state_sanity

# Toy oracle-MPC (20/20 passed)
PYTHONPATH=. python scripts/check_mpc_capacity.py --mode toy_oracle_mpc --split train_sim_id --max-templates 20

# MuJoCo oracle-MPC (5/5 interface passed, success_rate=0)
PYTHONPATH=. python scripts/check_mpc_capacity.py --mode mujoco_oracle_mpc --split train_sim_id --max-templates 5

# Model smoke tests (all passed)
PYTHONPATH=. python scripts/debug_rig_world_model.py
PYTHONPATH=. python scripts/debug_encoder_variants.py

# Environment tests (all passed)
PYTHONPATH=. python scripts/debug_mujoco_env.py
PYTHONPATH=. python scripts/debug_mujoco_oracle_rollout.py

# Data tests (all passed)
PYTHONPATH=. python scripts/debug_state_normalizer.py
PYTHONPATH=. python scripts/debug_reset_templates.py
PYTHONPATH=. python scripts/debug_metadata_schema.py
```

---

## Critical Rules

1. **CEM-MPC fixed**: All encoder variants use same planner config
2. **StateNormalizer rule**: Fit only on train/adaptation, never on test
3. **Split isolation**: Layout OOD / Shape OOD not in training
4. **Toy vs MuJoCo**: ToyPushEnv only for interface testing
5. **Oracle-MPC first**: Must establish capacity before training learned models

---

## Next Steps

**User must decide:**
- **Option A:** Tune Oracle-MPC for task success (success_rate > 80%)
- **Option B:** Implement obstacles-enabled MuJoCo env first

**Do NOT proceed without user confirmation.**
