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
- **2026-05-11**: Wide sweep (5cm early stop) → 结论：5cm early stop 不适合精度边界判断
- **2026-05-11**: Boundary refine v1 → no-early-stop boundary search 确认可行
- **2026-05-12**: boundary_video_night2 → c23/c25 毫米级精度结果确认
- **2026-05-12**: strict pose stop 代码 bug 修复（legacy 5cm 截断问题）

---

## 10. Sprint Update — 2026-05-12

**Last updated:** 2026-05-12

### 当前阶段

**正在完成：** 第 1 关 — MuJoCo Oracle-MPC capacity gate（open_space + mild_offset）

**尚未进入：**
- ❌ dataset generation
- ❌ learned dynamics / world model 训练
- ❌ flat / object / causal representation 对比
- ❌ OOD final evaluation
- ❌ SO-101 real robot validation

### 已确认结论

**5cm early stop 不适合作为能力边界判断：**
- no-early-stop full budget 后，Oracle-MPC 可达到毫米级误差
- 5cm 只能作为粗成功统计，不应作为正式 pose-to-goal 完成标准

**boundary_video_night2 结果（train_sim_id, 5 templates）：**

c23_precise：mean_final_pos_error ≈ 2.70mm，success_pos_1cm_rate = 1.0，success_pos_0p5cm_rate = 0.8

c25_fast：mean_final_pos_error ≈ 2.38mm，success_pos_0p5cm_rate = 1.0，success_pose_0p5cm_5deg_rate = 1.0

c23 更快进入高精度区（~405 steps 到 1cm），更适合作为主 baseline。c25 更稳但更慢，适合保守备用。

**strict pose stop 代码修复（2026-05-12）：**
- 修复了 legacy 5cm success 截断 strict pose stop 的 bug
- 引入 `should_stop`、`legacy_success_reached`、`strict_pose_stop_active`
- py_compile 通过，正式 smoke test 待执行

### 当前主线任务

1. 执行 strict pose stop smoke test（1 template，c23_strict600）
2. 执行 c23_strict600 confirmatory eval（3 open_space + 3 mild_offset）
3. 接入真实 MuJoCo obstacles（blocking / narrow_passage / edge_goal）
4. 在真实 obstacle 下用 config23 做 gate test
5. obstacle gate 通过后做小范围局部 sweep

### 暂时不要做

❌ 继续大范围 blind sweep
❌ 只围绕 open-space 调漂亮结果
❌ 把 mild_offset 说成真实 obstacle
❌ 改 cost function（先确保 obstacle 真正进入 MuJoCo）
❌ 进入 learned model 训练（直到 oracle-MPC capacity + obstacle gate 基本通过）
❌ 改 Paper 1 主线去做 VLM/VLA/LLM
❌ 让 AI agent 自动运行长实验

### 下一步 smoke test 命令（人工执行）

```bash
cd ~/my_robot_project
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

PYTHONPATH=. python scripts/check_mpc_capacity.py \
  --mode mujoco_oracle_mpc_closed_loop \
  --split train_sim_id \
  --max-templates 1 \
  --horizon 80 \
  --execute-steps 20 \
  --max-mpc-steps 30 \
  --num-samples 1024 \
  --num-elites 96 \
  --num-iterations 5 \
  --strict-pose-stop \
  --stop-pos-threshold 0.0015 \
  --stop-theta-threshold-deg 3.0 \
  --out runs/debug/c23_strict600_smoke.json
```

检查重点：不能在 5cm 处提前停止；只有 `pos<=1.5mm 且 theta<=3°` 才触发 STRICT POSE EARLY STOP。
