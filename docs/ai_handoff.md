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
