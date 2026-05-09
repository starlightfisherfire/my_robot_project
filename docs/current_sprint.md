# Current Sprint Control Console

**Last updated:** 2026-05-09

---

## 1. Current Stage

**Phase:** MuJoCo Oracle-MPC interface smoke test PASSED → Topology audit → Next gate decision

**NOT in these phases:**
- ❌ collect_sim_data
- ❌ learned model training
- ❌ learned model + MPC evaluation
- ❌ OOD gap analysis
- ❌ real robot integration

---

## 2. Completed Milestones

### 2.1 Model Smoke Tests ✅

**Command:**
```bash
PYTHONPATH=. python scripts/debug_rig_world_model.py
PYTHONPATH=. python scripts/debug_encoder_variants.py
```

**Passed:**
- FlatEncoder forward/backward
- ObjectCentricEncoder forward/backward
- CausalityAwareEncoder forward/backward
- DynamicsHead forward/backward
- SubgoalHead forward/backward
- RIGWorldModel unified interface

**Proves:**
- Model architecture is syntactically correct
- Gradient flow works
- Tensor shapes match contracts

**Does NOT prove:**
- Models can learn from real data
- Models generalize to OOD
- Representation quality differences

---

### 2.2 StateNormalizer Smoke Test ✅

**Command:**
```bash
PYTHONPATH=. python scripts/debug_state_normalizer.py
```

**Passed:**
- fit() on dummy train data
- transform() produces finite normalized values
- inverse_transform() recovers original scale

**Proves:**
- Normalizer interface works

**Does NOT prove:**
- Normalizer is fitted on correct split
- No test data leakage

---

### 2.3 Reset Template Generation ✅

**Command:**
```bash
PYTHONPATH=. python scripts/generate_reset_templates.py --num-per-split 3
PYTHONPATH=. python scripts/debug_reset_templates.py
```

**Passed:**
- reset_templates_v0.json generated
- Schema validation passed
- Split distribution correct
- Layout family distribution correct
- Shape family distribution correct

**Proves:**
- Reset template generation pipeline works
- Metadata schema is valid

**Does NOT prove:**
- Templates are geometrically feasible for task success
- Obstacles will be instantiated correctly

---

### 2.4 State Sanity Check ✅

**Command:**
```bash
PYTHONPATH=. python scripts/check_mpc_capacity.py --mode state_sanity
```

**Result:** 140/140 templates passed

**Passed:**
- No geometric errors
- Object-goal distances reasonable
- EE-object distances reasonable
- Workspace bounds respected

**Proves:**
- Reset templates are geometrically reasonable

**Does NOT prove:**
- CEM-MPC can solve these templates
- MuJoCo can instantiate these templates

---

### 2.5 Toy Oracle-MPC ✅

**Command:**
```bash
PYTHONPATH=. python scripts/check_mpc_capacity.py \
  --mode toy_oracle_mpc \
  --split train_sim_id \
  --max-templates 20
```

**Result:** 20/20 passed

**Passed:**
- ToyPushEnv + oracle rollout + CEM-MPC interface works
- planned_cost < zero_cost for all templates
- restore_state works correctly

**Proves:**
- Oracle rollout + CEM-MPC interface is correct
- CEM can improve over zero-action baseline

**Does NOT prove:**
- MuJoCo oracle-MPC works
- Real layout OOD capacity (ToyPushEnv ignores obstacles)
- Task-solving capacity

---

### 2.6 Minimal MuJoCo Env Scaffold ✅

**Command:**
```bash
PYTHONPATH=. python scripts/debug_mujoco_env.py
```

**Passed:**
- MuJoCo XML loads
- reset() works
- reset_from_template() works
- step(action) works
- clone_state() / restore_state() works
- get_object_pose() / get_ee_pos() / get_goal_pose() works
- get_contact_flag() detects pusher-object contact

**Proves:**
- MuJoCo environment interface is correct
- State cloning/restoration works

**Does NOT prove:**
- Obstacles are instantiated
- Collision detection works
- Task can be solved

---

### 2.7 MuJoCo Oracle Rollout ✅

**Command:**
```bash
PYTHONPATH=. python scripts/debug_mujoco_oracle_rollout.py
```

**Passed:**
- rollout_action_sequence_mujoco() runs
- restore_state works after rollout
- Hand-coded right push produces contact
- Object x increases as expected
- Rollout cost is finite

**Proves:**
- MuJoCo oracle rollout interface works
- True dynamics rollout is correct
- Cost function is computable

**Does NOT prove:**
- CEM-MPC can find good action sequences
- Task can be solved

---

### 2.8 MuJoCo Oracle-MPC Interface Smoke Test ✅

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

**Result:** 5/5 templates passed interface checks

**Passed:**
- Program does not crash
- Costs are finite
- planned_cost < zero_cost: 5/5
- best_min_dist < initial_dist: 5/5
- restore_state: 5/5
- Prints "mujoco oracle mpc capacity check ok"

**Proves:**
- MuJoCo Oracle-MPC interface works
- CEM can improve over zero-action baseline
- State restoration works correctly

**Does NOT prove:**
- Task success (success_rate = 0/5)
- Final distance reaches success threshold
- Layout OOD capacity (obstacles not instantiated)
- Shape OOD capacity
- Full task-solving capacity

**Current Limitation:**
- v0.1 MujocoPushEnv does not instantiate obstacles from templates
- Therefore blocking / narrow_passage / edge_goal cannot be tested yet
- success_rate = 0 should be interpreted as "task-solving capacity not yet established", NOT as "representation failed"

---

## 3. Current Truth

### What is TRUE:
✅ MuJoCo oracle rollout path works
✅ MuJoCo oracle-MPC interface smoke test passed
✅ CEM can improve cost over zero-action baseline
✅ State cloning/restoration works correctly

### What is NOT YET TRUE:
❌ MuJoCo Oracle-MPC achieves task success
❌ Layout OOD planner capacity is valid (obstacles not instantiated)
❌ Shape OOD planner capacity is tested
❌ Learned dynamics model can be trained
❌ Learned model + MPC can run
❌ Flat / object-centric / causality-aware have performance differences
❌ OOD gap exists
❌ Sim-to-real transfer works

---

## 4. Next Gate Decision

**Two options:**

### Option A: Tune MuJoCo Oracle-MPC task-solving capacity
- Increase horizon / num_samples / num_iterations
- Tune cost weights
- Debug why final_dist does not reach success threshold
- Goal: success_rate > 80% on train_sim_id

**Pros:**
- Establishes upper bound for learned model
- Validates that task is solvable with true dynamics

**Cons:**
- Still cannot test layout OOD (obstacles not instantiated)
- May spend time tuning planner before obstacles are ready

---

### Option B: Implement obstacles-enabled MuJoCo env first
- Modify MujocoPushEnv to instantiate obstacles from templates
- Add collision detection
- Then run full Oracle-MPC capacity on layout OOD

**Pros:**
- Enables true layout-OOD capacity testing
- More complete experimental setup

**Cons:**
- Requires MuJoCo XML generation for obstacles
- May introduce new bugs

---

**Recommendation:** User should decide based on project priority.

---

## 5. Do NOT Do Yet

❌ **collect_sim_data.py** - No learned model training until Oracle-MPC capacity is established
❌ **train_high_level.py** - No training until data collection is ready
❌ **learned model + MPC** - No learned rollout until models are trained
❌ **real robot / SO-101 integration** - Paper 1 is simulation first
❌ **RGB input** - Paper 1 uses structured object-state input
❌ **Diffusion Policy** - Paper 1 uses fixed CEM-MPC
❌ **C-JEPA implementation** - Not in Paper 1 scope
❌ **VLM / VLA / LLM graph planner** - Not in Paper 1 scope
❌ **Full benchmark release** - Internal research first

---

## 6. Immediate Commands

### Check current status:
```bash
git status
git diff
```

### Re-run MuJoCo Oracle-MPC smoke test:
```bash
PYTHONPATH=. python scripts/check_mpc_capacity.py \
  --mode mujoco_oracle_mpc \
  --split train_sim_id \
  --max-templates 5
```

### Debug MuJoCo oracle rollout:
```bash
PYTHONPATH=. python scripts/debug_mujoco_oracle_rollout.py
```

### Check reset templates:
```bash
PYTHONPATH=. python scripts/debug_reset_templates.py
```

---

## 7. Critical Risks

### Risk 1: Obstacles not instantiated
**Impact:** Layout OOD capacity results are invalid
**Mitigation:** Implement obstacles-enabled MuJoCo env before claiming layout OOD results

### Risk 2: Task-solving capacity not established
**Impact:** Cannot attribute learned model failure to representation quality
**Mitigation:** Tune Oracle-MPC until success_rate > 80% on train_sim_id

### Risk 3: Planner config confusion
**Impact:** Different encoder variants may use different planner configs, invalidating comparison
**Mitigation:** Document planner config policy clearly (see docs/planner_config_policy.md)

---

## 8. Definition of Done for Current Sprint

Current sprint is complete when:
1. ✅ MuJoCo Oracle-MPC interface smoke test passed
2. ⬜ User decides next gate: Option A or Option B
3. ⬜ Documentation audit complete
4. ⬜ Next sprint plan approved

---

## 9. Sprint History

- **2026-04-20**: Repo skeleton created
- **2026-04-21**: Reset template generation + state sanity passed
- **2026-04-26**: Toy oracle-MPC passed
- **2026-05-06**: MuJoCo env scaffold + oracle rollout passed
- **2026-05-08**: MuJoCo Oracle-MPC interface smoke test passed
- **2026-05-09**: Topology audit + documentation update
