# AI Handoff Document

**Last updated:** 2026-05-09

This document provides a quick handoff for future AI assistants (Claude, ChatGPT, etc.) to understand the project state and continue work.

---

## 1. Project Identity

### What this project IS:
- **Paper 1:** Embodied OOD generalization study under fixed CEM-MPC
- **Research question:** Do causality-aware object-level high-level representations improve structural OOD generalization compared to flat/object-centric representations?
- **Task:** Single-arm planar push-to-pose (Push-T-style)
- **Input:** Structured object-state tensors (NOT RGB)
- **Planner:** Fixed CEM-MPC (NOT learned policy)
- **OOD axes:** Layout OOD (blocking, narrow_passage, edge_goal) + Shape OOD (T→L)

### What this project is NOT:
- ❌ NOT "causal world model solves embodied AI"
- ❌ NOT RGB-based visual world model
- ❌ NOT C-JEPA / V-JEPA / VLM / VLA
- ❌ NOT Diffusion Policy / Flow Matching
- ❌ NOT LLM graph planner
- ❌ NOT foundation model architecture
- ❌ NOT large-scale training infrastructure

### Claim scope:
- ✅ Causality-aware representation improves OOD generalization under same MPC planner
- ❌ NOT claiming "true causal discovery"
- ❌ NOT claiming "slots are guaranteed causal"
- Use terminology: "causality-aware", "mechanism-aware", "factorized slots", "representation diagnostic"

---

## 2. Current Technical State

### What is WORKING:
✅ Model forward/backward smoke tests (flat/object/causal encoders + heads)
✅ StateNormalizer interface
✅ Reset template generation (140 templates)
✅ State sanity check (140/140 passed)
✅ Toy oracle-MPC (20/20 passed)
✅ MuJoCo env scaffold (reset/step/clone/restore/contact)
✅ MuJoCo oracle rollout (true dynamics rollout)
✅ MuJoCo Oracle-MPC interface smoke test (5/5 passed)

### What is NOT YET WORKING:
❌ MuJoCo Oracle-MPC task success (success_rate = 0/5)
❌ Obstacles not instantiated in MuJoCo env
❌ Layout OOD capacity not valid (obstacles missing)
❌ Shape OOD capacity not tested
❌ Learned dynamics model training
❌ Learned model + MPC evaluation
❌ OOD gap analysis
❌ Real robot integration

### Critical limitation:
**v0.1 MujocoPushEnv does not instantiate obstacles from reset templates.**
- Therefore blocking / narrow_passage / edge_goal cannot be tested yet
- success_rate = 0 should be interpreted as "task-solving capacity not yet established", NOT as "representation failed"

---

## 3. Current Repo Path

```
~/my_robot_project
```

Activate environment:
```bash
cd ~/my_robot_project
conda activate lerobot
```

---

## 4. Current Main Command

### Re-run MuJoCo Oracle-MPC smoke test:
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

**Expected result:**
- Program does not crash
- planned_cost < zero_cost: 5/5
- best_min_dist < initial_dist: 5/5
- restore_state: 5/5
- Prints "mujoco oracle mpc capacity check ok"
- success_rate = 0 (task not solved yet)

### Other important commands:
```bash
# Check git status
git status
git diff

# Debug MuJoCo oracle rollout
PYTHONPATH=. python scripts/debug_mujoco_oracle_rollout.py

# Debug MuJoCo env
PYTHONPATH=. python scripts/debug_mujoco_env.py

# Check reset templates
PYTHONPATH=. python scripts/debug_reset_templates.py

# Model smoke tests
PYTHONPATH=. python scripts/debug_rig_world_model.py
PYTHONPATH=. python scripts/debug_encoder_variants.py
```

---

## 5. Next Decision Point

**User must decide between two options:**

### Option A: Tune MuJoCo Oracle-MPC for task success
- Increase horizon / num_samples / num_iterations
- Tune cost weights
- Debug why final_dist does not reach success threshold
- Goal: success_rate > 80% on train_sim_id

**Pros:** Establishes upper bound for learned model
**Cons:** Still cannot test layout OOD (obstacles not instantiated)

### Option B: Implement obstacles-enabled MuJoCo env first
- Modify MujocoPushEnv to instantiate obstacles from templates
- Add collision detection
- Then run full Oracle-MPC capacity on layout OOD

**Pros:** Enables true layout-OOD capacity testing
**Cons:** Requires MuJoCo XML generation for obstacles

**DO NOT proceed with either option without user confirmation.**

---

## 6. Hard Boundaries (DO NOT DO)

❌ **collect_sim_data.py** - No data collection until Oracle-MPC capacity is established
❌ **train_high_level.py** - No training until data collection is ready
❌ **learned model + MPC** - No learned rollout until models are trained
❌ **real robot / SO-101** - Paper 1 is simulation first
❌ **RGB input** - Paper 1 uses structured object-state input
❌ **Diffusion Policy** - Paper 1 uses fixed CEM-MPC
❌ **C-JEPA implementation** - Not in Paper 1 scope
❌ **VLM / VLA / LLM graph planner** - Not in Paper 1 scope
❌ **Full benchmark release** - Internal research first

---

## 7. Planner Config Caution

**IMPORTANT:** There are THREE types of CEM-MPC configs:

### A. smoke_test config (current)
- Used for interface verification
- Lightweight and fast
- Example: horizon=80, num_samples=1536, num_iterations=7

### B. oracle_capacity_strong config (future)
- Used for MuJoCo true-dynamics upper bound
- Can be stronger than learned eval config
- Purpose: establish what is possible with perfect dynamics

### C. learned_eval_default config (future)
- Used for flat/object/causal fair comparison
- **MUST be identical for all three encoder variants**
- May use shorter horizon than oracle_capacity_strong
- Reason: learned model accumulates prediction error over long horizons

**Critical rules:**
1. Oracle strong config ≠ learned final config
2. Learned model long horizon (e.g., horizon=80) may accumulate too much error
3. Flat/object/causal MUST use identical planner config
4. Can do horizon ablation later, but don't decide final config now
5. Do NOT assume horizon=80 / num_samples=1536 / num_iterations=7 is the learned eval default

**See:** `docs/planner_config_policy.md` for details

---

## 8. Key Files to Know

### Core environment / oracle / planner:
- `src/envs/mujoco_push_env.py` - MuJoCo env (obstacles not instantiated yet)
- `src/planners/mujoco_oracle_rollout.py` - MuJoCo oracle rollout
- `src/planners/cem_mpc.py` - CEM optimizer (fixed)
- `src/planners/cost_functions.py` - Cost computation
- `src/metrics/mujoco_oracle_capacity.py` - Oracle-MPC capacity check
- `scripts/check_mpc_capacity.py` - Unified capacity check entry

### Reset / split / metadata:
- `src/interventions/shape_families.py` - Shape OOD definitions
- `src/interventions/layout_families.py` - Layout OOD definitions
- `src/interventions/sampling_rules.py` - Validation rules
- `src/interventions/reset_template_loader.py` - Template loader
- `data/sim/metadata/reset_templates_v0.json` - 140 templates

### Normalizer / model:
- `src/data/state_normalizer.py` - State normalization (CRITICAL: fit only on train)
- `src/models/encoders.py` - Flat/object/causal encoders
- `src/models/heads.py` - Dynamics/subgoal heads
- `src/models/rig_world.py` - Unified model interface
- `src/models/losses.py` - Training losses

### Configs:
- `configs/planner/cem_mpc.yaml` - CEM-MPC config
- `configs/planner/cost_weights.yaml` - Cost weights
- `configs/splits.yaml` - Split definitions

### Documentation:
- `CLAUDE.md` - Project instructions (READ THIS FIRST)
- `docs/current_sprint.md` - Current sprint status
- `docs/code_audit.md` - File-level audit table
- `docs/file_topology_map.md` - File topology
- `docs/planner_config_policy.md` - Planner config policy
- `docs/split_protocol.md` - Split rules

---

## 9. Critical Rules

### Data split rules:
1. StateNormalizer can ONLY be fitted on training/adaptation splits
2. StateNormalizer must NEVER be fitted on any test split
3. Layout OOD families must NOT appear in training
4. L-shaped objects must NOT appear in training
5. Real OOD test splits must NOT be used for model selection / hyperparameter tuning / early stopping

### Planner fairness rules:
1. CEM-MPC is fixed across encoder variants
2. Do NOT tune planner parameters separately for flat/object/causal models
3. All encoder variants must use identical planner config during evaluation

### Experiment attribution rules:
1. If Oracle-MPC fails under true dynamics, learned model failure cannot be attributed to representation quality
2. Must establish Oracle-MPC capacity BEFORE training learned models
3. Toy / MuJoCo / Real robot results must be clearly distinguished
4. Do NOT overclaim from toy or scaffold tests

---

## 10. Common Pitfalls

### Pitfall 1: Assuming obstacles are instantiated
**Reality:** v0.1 MujocoPushEnv does NOT instantiate obstacles from templates
**Impact:** Layout OOD capacity results are invalid
**Fix:** Implement obstacles-enabled MuJoCo env before claiming layout OOD results

### Pitfall 2: Interpreting success_rate=0 as representation failure
**Reality:** Task-solving capacity not yet established
**Impact:** Cannot attribute failure to representation quality
**Fix:** Tune Oracle-MPC until success_rate > 80% on train_sim_id

### Pitfall 3: Using different planner configs for different encoders
**Reality:** Unfair comparison
**Impact:** Cannot isolate representation quality
**Fix:** Use identical planner config for all encoder variants

### Pitfall 4: Fitting StateNormalizer on test data
**Reality:** Data leakage
**Impact:** Invalidates OOD evaluation
**Fix:** Fit only on train/adaptation splits

### Pitfall 5: Confusing toy/MuJoCo/real results
**Reality:** ToyPushEnv ignores obstacles, not real physics
**Impact:** Overclaiming from toy results
**Fix:** Clearly label toy vs MuJoCo vs real robot results

---

## 11. Quick Start for New AI

1. **Read CLAUDE.md** - Project instructions
2. **Read docs/current_sprint.md** - Current status
3. **Read docs/code_audit.md** - File-level audit
4. **Run smoke test:**
   ```bash
   PYTHONPATH=. python scripts/check_mpc_capacity.py --mode mujoco_oracle_mpc --split train_sim_id --max-templates 5
   ```
5. **Check git status:**
   ```bash
   git status
   git diff
   ```
6. **Ask user for next action** - Do NOT proceed without confirmation

---

## 12. Contact / Feedback

- User: Bruce Wu
- Repo: ~/my_robot_project
- Branch: main
- Last commit: "add mujoco oracle rollout debug path" (2026-05-08)

---

## 13. Version History

- **v0.1** (2026-05-09): Initial handoff document after topology audit
- MuJoCo Oracle-MPC interface smoke test passed
- Obstacles not yet instantiated
- Task success not yet achieved
- Next gate: user decision between Option A (tune Oracle-MPC) or Option B (implement obstacles)
- **v0.2** (2026-05-12): 重大进展更新 — Oracle-MPC 毫米级精度确认，strict pose stop 修复

---

## 14. 最新实验事实（2026-05-12 更新）

### 已确认：Oracle-MPC 在 open_space 上具备毫米级精度

来自 `boundary_video_night2`（`runs/video_sweeps/boundary_video_night2_20260512_001900`）：

**c23_precise（500 steps）：**
- mean_final_pos_error ≈ 2.70mm，median ≈ 1.55mm
- success_pos_1cm_rate = 1.0，success_pos_0p5cm_rate = 0.8
- success_pose_0p5cm_5deg_rate = 0.6

**c25_fast（600 steps）：**
- mean_final_pos_error ≈ 2.38mm，median ≈ 1.21mm
- success_pos_0p5cm_rate = 1.0，success_pose_0p5cm_5deg_rate = 1.0

**结论：** 当前问题不再是"planner 完全不会推"，而是高精度停止、预算分配和 obstacle gate。

### 已确认：5cm early stop 不适合作为能力边界判断

- 旧的 5cm early stop 让系统看起来只能到 4.8cm 左右
- no-early-stop full budget 后，Oracle-MPC 可达到毫米级误差
- 5cm 只能作为粗成功统计，不应作为正式 pose-to-goal 完成标准

### 已修复：strict pose stop 被 legacy 5cm 截断的 bug（2026-05-12）

**旧问题：** legacy 5cm success 设置 `success=True`，chunk end 的 `if success and not disable_early_stop: break` 导致 strict pose stop 开启时仍被 5cm 截断。

**修复：** 引入 `should_stop`、`legacy_success_reached`、`strict_pose_stop_active`；chunk end 改为 `if should_stop: break`；strict 阈值改为 `<=`。

**状态：** py_compile 通过，正式 smoke test 待人工执行。

### 当前技术状态更新

✅ Oracle-MPC 在 open_space / mild_offset 上具备毫米级精度（2.70mm / 2.38mm）
✅ Strict pose stop 代码逻辑已修复
✅ 两种运行模式已定义（boundary search / confirmatory eval）
❌ Strict pose stop smoke test 尚未执行
❌ Obstacles 仍未接入 MuJoCo（最大阻塞项）
❌ Layout OOD capacity 仍无效（obstacles missing）

---

## 15. 当前主线任务（2026-05-12）

1. **执行 strict pose stop smoke test**（1 template，c23_strict600）
2. **执行 c23_strict600 confirmatory eval**（3 open_space + 3 mild_offset）
3. **接入真实 MuJoCo obstacles**（blocking / narrow_passage / edge_goal）
4. **在真实 obstacle 下用 config23 做 gate test**
5. **obstacle gate 通过后做小范围局部 sweep**（16 configs）

### 暂时不要做

❌ 继续大范围 blind sweep
❌ 只围绕 open-space 调漂亮结果
❌ 把 mild_offset 说成真实 obstacle
❌ 改 cost function（先确保 obstacle 真正进入 MuJoCo）
❌ 进入 learned model 训练（直到 oracle-MPC capacity + obstacle gate 基本通过）
❌ 改 Paper 1 主线去做 VLM/VLA/LLM
❌ 让 AI agent 自动运行长实验

---

## 16. 新 AI 快速上手（更新版）

1. 读 `CLAUDE.md`
2. 读 `docs/current_sprint.md`（第 10 节是最新状态）
3. 读 `docs/known_issues.md`（关键风险）
4. 读 `docs/planner_capacity_protocol.md`（实验协议）
5. 读 `docs/experiment_log.md`（最新实验结果）
6. **不要自动运行任何实验**，先向用户确认当前任务

### 关键提醒

- mild_offset ≠ 真实 obstacle，不要混淆
- 当前所有结果均来自 train_sim_id，尚未在 OOD split 上测试
- strict pose stop smoke test 需要人工执行，不要让 AI agent 自动运行
- 不要修改 CEM、cost function、planner 参数、env、reset templates，除非用户明确要求
