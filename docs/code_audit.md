# Code Audit: File-Level Status Table

**Last updated:** 2026-05-11

This document provides a file-level audit of the Paper 1 codebase, tracking validation status and limitations.

---

## Reset / Split / Metadata Layer

| File | Layer | Role | Main classes/functions | Input | Output | Called by | Debug command | Status | Proven | Not proven / limitations | Next action |
|------|-------|------|------------------------|-------|--------|-----------|---------------|--------|--------|--------------------------|-------------|
| `src/interventions/shape_families.py` | Data | Define shape families for OOD | `SHAPE_FAMILIES`, `get_shape_config()` | shape_name | shape config dict | generate_reset_templates.py | N/A | PASS | Shape families defined correctly | Not tested in actual MuJoCo env | None needed |
| `src/interventions/layout_families.py` | Data | Define layout families for OOD | `LAYOUT_FAMILIES`, `get_layout_config()` | layout_name | layout config dict | generate_reset_templates.py | N/A | PASS | Layout families defined correctly | Obstacles not instantiated in MuJoCo yet | Implement obstacles in MuJoCo |
| `src/interventions/sampling_rules.py` | Data | Validate reset templates | `validate_reset_templates()` | templates list | raises on error | reset_template_loader.py | N/A | PASS | Schema validation works | Does not validate geometric feasibility | None needed |
| `scripts/generate_reset_templates.py` | Data | Generate reset templates | `main()` | CLI args | reset_templates_v0.json | User | `PYTHONPATH=. python scripts/generate_reset_templates.py` | PASS | Generates valid templates | Does not guarantee task solvability | None needed |
| `data/sim/metadata/reset_templates_v0.json` | Data | Reset template storage | N/A | N/A | JSON templates | reset_template_loader.py | N/A | PASS | 140 templates generated | Obstacles not yet instantiated | None needed |
| `src/interventions/reset_template_loader.py` | Data | Load and query templates | `load_reset_templates()`, `get_templates_by_split()` | JSON path | templates list | check_mpc_capacity.py | `PYTHONPATH=. python scripts/debug_reset_templates.py` | PASS | Loading and querying works | N/A | None needed |
| `src/data/metadata_schema.py` | Data | Episode metadata schema | `EpisodeMetadata`, `Pose2D`, `ObstacleMetadata` | dict | dataclass | reset_template_loader.py | `PYTHONPATH=. python scripts/debug_metadata_schema.py` | PASS | Schema validation works | Not used in actual data collection yet | Use in data collection |

---

## Environment Layer

| File | Layer | Role | Main classes/functions | Input | Output | Called by | Debug command | Status | Proven | Not proven / limitations | Next action |
|------|-------|------|------------------------|-------|--------|-----------|---------------|--------|--------|--------------------------|-------------|
| `src/envs/toy_push_env.py` | Env | Toy dynamics for interface testing | `ToyPushEnv` | action | next_state | oracle_rollout.py | `PYTHONPATH=. python scripts/debug_cem_mpc_toy.py` | PASS | Interface works, toy dynamics correct | Ignores obstacles, not MuJoCo, not real physics | None needed (toy only) |
| `src/envs/mujoco_push_env.py` | Env | MuJoCo dynamics scaffold | `MujocoPushEnv`, `MujocoPushState` | action | next_state | mujoco_oracle_rollout.py | `PYTHONPATH=. python scripts/debug_mujoco_env.py` | PARTIAL | reset/step/clone/restore works, contact detection works | **Obstacles not instantiated**, collision detection placeholder, goal_site not dynamic | Implement obstacles instantiation |
| `scripts/debug_mujoco_env.py` | Debug | Test MuJoCo env interface | `main()` | N/A | prints | User | `PYTHONPATH=. python scripts/debug_mujoco_env.py` | PASS | Env interface validated | Does not test obstacles | None needed |

---

## Planner Layer

| File | Layer | Role | Main classes/functions | Input | Output | Called by | Debug command | Status | Proven | Not proven / limitations | Next action |
|------|-------|------|------------------------|-------|--------|-----------|---------------|--------|--------|--------------------------|-------------|
| `src/planners/cost_functions.py` | Planner | Cost computation core | `rollout_cost()`, `CostWeights`, `wrap_angle()` | rollout result, weights | scalar cost | oracle_rollout.py, mujoco_oracle_rollout.py | N/A | PASS | Cost computation correct, angle wrapping safe | N/A | None needed |
| `src/planners/cem_mpc.py` | Planner | CEM optimizer (fixed) | `CEMMPC`, `CEMResult` | cost_fn | action_sequence | check_mpc_capacity.py | `PYTHONPATH=. python scripts/debug_cem_mpc_toy.py` | PASS | CEM optimization works | Not tuned for task success yet | Tune for task success |
| `src/planners/oracle_rollout.py` | Planner | Toy oracle rollout | `rollout_action_sequence()`, `oracle_rollout_cost()` | ToyPushEnv, action_sequence | rollout result, cost | toy_oracle_capacity.py | `PYTHONPATH=. python scripts/debug_oracle_rollout.py` | PASS | Toy oracle rollout works | Not MuJoCo, ignores obstacles | None needed (toy only) |
| `src/planners/mujoco_oracle_rollout.py` | Planner | MuJoCo oracle rollout | `rollout_action_sequence_mujoco()`, `mujoco_oracle_rollout_cost()` | MujocoPushEnv, action_sequence | rollout result, cost | mujoco_oracle_capacity.py | `PYTHONPATH=. python scripts/debug_mujoco_oracle_rollout.py` | PASS | MuJoCo oracle rollout works, restore_state correct | Does not test with obstacles | Test with obstacles |
| `scripts/debug_oracle_rollout.py` | Debug | Test toy oracle rollout | `main()` | N/A | prints | User | `PYTHONPATH=. python scripts/debug_oracle_rollout.py` | PASS | Toy oracle rollout validated | Not MuJoCo | None needed |
| `scripts/debug_mujoco_oracle_rollout.py` | Debug | Test MuJoCo oracle rollout | `main()` | N/A | prints | User | `PYTHONPATH=. python scripts/debug_mujoco_oracle_rollout.py` | PASS | MuJoCo oracle rollout validated | Does not test with obstacles | Test with obstacles |
| `scripts/debug_cem_mpc_toy.py` | Debug | Test CEM with toy env | `main()` | N/A | prints | User | `PYTHONPATH=. python scripts/debug_cem_mpc_toy.py` | PASS | CEM + toy oracle works | Not MuJoCo | None needed |

---

## Capacity Check Layer

| File | Layer | Role | Main classes/functions | Input | Output | Called by | Debug command | Status | Proven | Not proven / limitations | Next action |
|------|-------|------|------------------------|-------|--------|-----------|---------------|--------|--------|--------------------------|-------------|
| `src/metrics/planner_capacity.py` | Metrics | State sanity checks | `run_state_sanity()` | templates | report | check_mpc_capacity.py | N/A | PASS | Geometric sanity validated | Does not run CEM | None needed |
| `src/metrics/toy_oracle_capacity.py` | Metrics | Toy oracle-MPC capacity | `run_toy_oracle_mpc_capacity()` | templates | report | check_mpc_capacity.py | N/A | PASS | Toy oracle-MPC works | Not MuJoCo, ignores obstacles | None needed (toy only) |
| `src/metrics/mujoco_oracle_capacity.py` | Metrics | MuJoCo oracle-MPC capacity | `run_mujoco_oracle_mpc_capacity()`, `evaluate_one_template_mujoco_oracle_mpc()` | templates | report | check_mpc_capacity.py | N/A | PARTIAL | Interface smoke test passed (5/5), cost improvement works | **success_rate=0**, task success not achieved, obstacles not tested | Tune for task success OR implement obstacles first |
| `scripts/check_mpc_capacity.py` | Metrics | Unified capacity check entry | `main()`, `run_state_sanity_mode()`, `run_toy_oracle_mpc_mode()`, `run_mujoco_oracle_mpc_mode()` | CLI args | report JSON | User | `PYTHONPATH=. python scripts/check_mpc_capacity.py --mode mujoco_oracle_mpc` | PARTIAL | state_sanity: 140/140, toy_oracle_mpc: 20/20, mujoco_oracle_mpc: 5/5 interface | mujoco_oracle_mpc success_rate=0, task not solved | Tune or implement obstacles |

---

## Data Normalization Layer

| File | Layer | Role | Main classes/functions | Input | Output | Called by | Debug command | Status | Proven | Not proven / limitations | Next action |
|------|-------|------|------------------------|-------|--------|-----------|---------------|--------|--------|--------------------------|-------------|
| `src/data/state_normalizer.py` | Data | State normalization (CRITICAL) | `StateNormalizer`, `fit()`, `transform()` | raw state | normalized state | RIGWorldModel (future) | `PYTHONPATH=. python scripts/debug_state_normalizer.py` | PASS | Interface works, fit/transform correct | **Not tested with real data**, **split leakage not validated** | Validate split protocol in training |
| `scripts/debug_state_normalizer.py` | Debug | Test normalizer | `main()` | N/A | prints | User | `PYTHONPATH=. python scripts/debug_state_normalizer.py` | PASS | Normalizer interface validated | Dummy data only | None needed |

---

## Model Layer

| File | Layer | Role | Main classes/functions | Input | Output | Called by | Debug command | Status | Proven | Not proven / limitations | Next action |
|------|-------|------|------------------------|-------|--------|-----------|---------------|--------|--------|--------------------------|-------------|
| `src/models/encoders.py` | Model | Three encoder variants | `FlatEncoder`, `ObjectCentricEncoder`, `CausalityAwareEncoder` | [B,H,N,D] state | [B,256] z | rig_world.py | `PYTHONPATH=. python scripts/debug_encoder_variants.py` | PASS | Forward/backward works, shapes correct | Not trained, not tested on real data | Train on real data |
| `src/models/heads.py` | Model | Prediction heads | `DynamicsHead`, `SubgoalHead` | z, action | pred_delta, pred_subgoal | rig_world.py | `PYTHONPATH=. python scripts/debug_rig_world_model.py` | PASS | Forward/backward works, shapes correct | Not trained, not tested on real data | Train on real data |
| `src/models/losses.py` | Model | Training losses | `dynamics_loss()`, `subgoal_loss()` | pred, target | scalar loss | train script (future) | N/A | PASS | Loss computation works | Not used in training yet | Use in training |
| `src/models/rig_world.py` | Model | Unified model interface | `RIGWorldModel` | state, action | z, pred_delta, pred_subgoal | train/eval scripts (future) | `PYTHONPATH=. python scripts/debug_rig_world_model.py` | PASS | Unified interface works, all variants work | Not trained, not tested on real data | Train on real data |
| `scripts/debug_encoder_variants.py` | Debug | Test encoders | `main()` | N/A | prints | User | `PYTHONPATH=. python scripts/debug_encoder_variants.py` | PASS | All encoders validated | Dummy data only | None needed |
| `scripts/debug_rig_world_model.py` | Debug | Test RIGWorldModel | `main()` | N/A | prints | User | `PYTHONPATH=. python scripts/debug_rig_world_model.py` | PASS | RIGWorldModel validated | Dummy data only | None needed |

---

## Learned Rollout Layer (Future)

| File | Layer | Role | Main classes/functions | Input | Output | Called by | Debug command | Status | Proven | Not proven / limitations | Next action |
|------|-------|------|------------------------|-------|--------|-----------|---------------|--------|--------|--------------------------|-------------|
| `src/planners/rollout_model.py` | Planner | Learned model rollout | `rollout_with_learned_model()` (future) | RIGWorldModel, action_sequence | predicted states | CEM-MPC (future) | N/A | NOT_RUN | N/A | Not implemented yet | Implement after model training |

---

## Config Files

| File | Layer | Role | Main classes/functions | Input | Output | Called by | Debug command | Status | Proven | Not proven / limitations | Next action |
|------|-------|------|------------------------|-------|--------|-----------|---------------|--------|--------|--------------------------|-------------|
| `configs/planner/cem_mpc.yaml` | Config | CEM-MPC config | N/A | N/A | config dict | train/eval scripts (future) | N/A | PARTIAL | Config defined | **horizon=10 is for learned eval, not oracle strong**, config policy not clear | Document config policy |
| `configs/planner/cost_weights.yaml` | Config | Cost weights | N/A | N/A | weights dict | cost_functions.py | N/A | PARTIAL | Weights defined | Not tuned for task success | Tune for task success |
| `configs/splits.yaml` | Config | Split definitions | N/A | N/A | split config | generate_reset_templates.py | N/A | PASS | Splits defined correctly | N/A | None needed |

---

## Summary Statistics

### By Status:
- **PASS**: 23 files (fully validated for their scope)
- **PARTIAL**: 4 files (interface works, but limitations exist)
- **NOT_RUN**: 1 file (not implemented yet)
- **FUTURE**: Multiple files (planned but not started)

### Critical Blockers:
1. **Obstacles not instantiated** in MujocoPushEnv → blocks layout OOD capacity
2. **Task success not achieved** (success_rate=0) → blocks learned model attribution
3. **Planner config policy unclear** → risks unfair comparison

### Next Critical Path:
1. **Decision:** Tune Oracle-MPC for task success OR implement obstacles first
2. **After decision:** Complete chosen path
3. **Then:** Collect sim data
4. **Then:** Train learned models
5. **Then:** Evaluate learned model + MPC
6. **Then:** Analyze OOD gap

---

## File Dependency Graph

```
reset_templates_v0.json
    ↓
reset_template_loader.py
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
```

---

## Validation Commands Quick Reference

```bash
# State sanity
PYTHONPATH=. python scripts/check_mpc_capacity.py --mode state_sanity

# Toy oracle-MPC
PYTHONPATH=. python scripts/check_mpc_capacity.py --mode toy_oracle_mpc --split train_sim_id --max-templates 20

# MuJoCo oracle-MPC
PYTHONPATH=. python scripts/check_mpc_capacity.py --mode mujoco_oracle_mpc --split train_sim_id --max-templates 5

# Model smoke tests
PYTHONPATH=. python scripts/debug_rig_world_model.py
PYTHONPATH=. python scripts/debug_encoder_variants.py

# Environment tests
PYTHONPATH=. python scripts/debug_mujoco_env.py
PYTHONPATH=. python scripts/debug_mujoco_oracle_rollout.py

# Data tests
PYTHONPATH=. python scripts/debug_state_normalizer.py
PYTHONPATH=. python scripts/debug_reset_templates.py
PYTHONPATH=. python scripts/debug_metadata_schema.py
```

---

## Critical Findings and Decisions

### Finding 1: Sphere Pusher Invalid for Planar Side Pushing (completed 2026-05-10)

**Problem:**
- Original pusher geometry was a 25mm radius sphere at z=35mm
- Sphere pusher was invalid for planar side pushing scaffold with 12mm thick T/L shapes
- Video evidence showed pusher slipping over/past object, unable to form stable lateral contact

**Evidence:**
1. Pusher capacity diagnostic videos (artifacts/videos/pusher_capacity_*.mp4):
   - At max_speed=0.05/0.10: some contact and minimal object displacement
   - At max_speed=0.15/0.20: earlier contact but LOWER contact_rate and WORSE object displacement
   - Visual inspection: blue sphere pusher visibly slips over/bypasses T-shaped object
   - Higher commanded speed reduced contact stability instead of improving it

2. Quantitative metrics from pusher_capacity.json:
   - Increasing max_speed paradoxically reduced contact_rate
   - Object displacement decreased at higher speeds
   - Pusher tracking ratio remained high but object didn't move proportionally

**Root cause:**
- 25mm radius sphere contacting 12mm thick planar object creates unstable geometry
- Sphere can roll over thin edge rather than maintaining side contact
- Not suitable for planar pushing tasks requiring stable lateral force transmission

**Decision:**
- Changed official pusher to vertical cylinder/finger geometry
- Current official/default parameters:
  - `pusher_geom_type = "cylinder"`
  - `pusher_radius = 0.010` m (diameter = 20 mm)
  - `pusher_halfheight = 0.014` m (total height = 28 mm)
  - `pusher_z = 0.016` m (bottom clearance = 2 mm, top height = 30 mm)
  - `pusher_mass = 0.05` kg
- Geometric interpretation:
  - Cylinder diameter = 20 mm
  - Total height = 28 mm
  - Bottom clearance from floor = 2 mm
  - Top height from floor = 30 mm
- Real-robot implications for SO-101:
  - Future SO-101 real robot should try to match this vertical cylindrical fingertip geometry
  - Best achieved by gripping/fixing a 3D-printed cylindrical adapter in the gripper
  - Avoid using complex gripper jaw edges to push objects directly
- Sphere pusher retained only as legacy diagnostic option

**Files modified:**
- src/envs/mujoco_push_env.py: Changed default pusher to cylinder
- scripts/render_pusher_capacity.py: Updated defaults, added pusher type to JSON filenames
- scripts/check_mpc_capacity.py: Parameter passthrough (already supports both)
- scripts/render_closed_loop_rollout.py: Parameter passthrough (already supports both)

**Impact:**
- All future MuJoCo oracle-MPC capacity checks use cylinder pusher by default
- Existing sphere-based diagnostic results remain valid for comparison
- No changes to cost function, CEM/MPC, planner, or reset templates
- No changes to friction/damping/kv/max_speed physical parameters
- SO-101 real robot will need matching cylindrical fingertip

**Validation:**
- Cylinder pusher capacity video shows stable contact and consistent object displacement
- Visual inspection confirms no slipping/bypassing behavior
- Contact maintained throughout push phase

---

### Finding 2: Wide Sweep with 5cm Early Stop — Conclusion (2026-05-11)

**Sweep identity:**
- Directory: `runs/sweeps/wide_overnight_v2_20260511_012735`
- Command: `PYTHONPATH=. python scripts/run_wide_mpc_sweep_with_best_video.py --preset wide_overnight_v2 --jobs 20 --max-templates 5 --shuffle-configs --timeout-sec 36000`

**Results:**
- 117 configs completed (reached timeout)
- Summary rebuilt from completed configs

**Key observations:**
- Top configs all reached `success_rate = 1.0` under **5 cm position threshold**
- Top `final_pos_error` values cluster around 0.0487–0.0491 m
- `success_pose_2cm_15deg_rate` remains **0** across all configs
- This is caused by 5 cm early stop truncating stronger configs around the success boundary
- The 5 cm early stop terminates the rollout as soon as object enters the 5 cm ball around the goal, preventing measurement of how close the planner can really get

**Conclusion:**
- The old wide sweep is useful as a **5 cm coarse capacity map**
- It is NOT sufficient for near-goal precision or boundary analysis
- Stronger configs were truncated by early stop before reaching their true final precision

**Config family patterns extracted from old sweep:**
- `horizon = 60–120` is enough for the next boundary-search stage
- `execute_steps = 20–30` should be main range
- `max_mpc_steps = 20–25` should be tested
- `num_samples = 512–1024` is enough for first boundary sweep
- `num_iterations = 3–5` is enough for first boundary sweep
- `num_samples = 1536`, `num_iterations >= 7`, `horizon >= 140` should be reserved for **second-stage refinement only**

---

### Finding 3: Next Stage — No-Early-Stop Boundary Search (decision 2026-05-11)

**Current next stage is NOT about continuing to prove 5 cm success.**

Instead, the next stage does:
- Disable early stop
- Run fixed-budget rollout (execute full `max_mpc_steps`)
- Compare `final` and `best` pose error
- Record multi-threshold success rates (5cm / 3cm / 2cm / 1.5cm / 1cm / 0.5cm)
- Record `first_reach_steps` for each threshold
- Record post-reach trend: does the planner keep improving or drift away after reaching a threshold?

**Boundary search primary metrics:**
- `mean_best_pose_cost`
- `mean_final_pose_cost`
- `mean_best_pos_error`
- `mean_final_pos_error`
- `mean_best_theta_error_deg_at_best_pos`
- `mean_final_theta_error_deg`

**Pose cost definition:**
```
pos_scale = 0.01
theta_scale_deg = 5.0
pose_cost = (pos_error / pos_scale)^2 + (theta_error_deg / theta_scale_deg)^2
```
- `pose_cost` is only a continuous ranking metric for boundary search
- It is NOT a success threshold
- Success thresholds are reported separately

**Multi-threshold success reporting metrics:**

Position-only:
- `success_pos_5cm`
- `success_pos_3cm`
- `success_pos_2cm`
- `success_pos_1p5cm`
- `success_pos_1cm`
- `success_pos_0p5cm`

Pose-level:
- `success_pose_5cm_15deg`
- `success_pose_3cm_15deg`
- `success_pose_2cm_15deg`
- `success_pose_1p5cm_10deg`
- `success_pose_1cm_10deg`
- `success_pose_1cm_5deg`
- `success_pose_0p5cm_5deg`

Milestone tracking:
- `threshold_first_reach`: at which MPC step each threshold is first reached
- `threshold_post_reach_trace`: does error continue decreasing or drift after reaching?
- `mpc_step_error_trace`: full error-vs-step curve

**boundary_refine_v1 parameter grid:**

```
horizon = [60, 80, 100, 120]
execute_steps = [20, 30]
max_mpc_steps = [20, 25]
num_samples = [512, 1024]
num_iterations = [3, 5]
num_elites = {
    512: 48,
    1024: 96
}
total configs = 64
```

Notes:
- This is the first no-early-stop boundary-search sweep
- Derived from the old 5 cm early-stop wide sweep
- Intentionally excludes `horizon >= 140`, `num_samples = 1536`, `num_iterations >= 7` for first pass
- A second-stage refinement may test top configs with `samples = 1536`, `iterations = 7`, `horizon = 140`, or `execute_steps = 40`

**Current cautions / DO NOT:**
- Do NOT continue using 5 cm early-stop sweep to judge precision boundary
- Do NOT tune cost function yet — wait until no-early-stop boundary behavior is understood
- Do NOT change reset templates or split protocol
- Do NOT expand to learned dynamics / representation models until Oracle-MPC capacity boundary is understood
- Do NOT reintroduce sphere pusher as official pusher
- If using AI coding agents: they must NOT modify env/cost/CEM/planner unless explicitly requested

---

## Recent Progress — 2026-05-12

### Finding 4：No-Early-Stop Boundary Search 确认 Oracle-MPC 可达毫米级精度

**实验：** boundary_video_night2（`runs/video_sweeps/boundary_video_night2_20260512_001900`）

**结论：**
- c23_precise（500 steps）：mean_final_pos_error ≈ 2.70mm，success_pos_1cm_rate = 1.0
- c25_fast（600 steps）：mean_final_pos_error ≈ 2.38mm，success_pos_0p5cm_rate = 1.0
- Oracle-MPC 在 open_space / mild_offset 上已具备毫米级精度能力
- 当前问题不再是"planner 完全不会推"，而是高精度停止、预算分配和 obstacle gate

**代码状态更新：**

| File | Status | 更新内容 |
|------|--------|---------|
| `src/metrics/mujoco_oracle_capacity.py` | PARTIAL→IMPROVED | 新增 closed-loop 评估函数，strict pose stop 逻辑修复 |
| `scripts/check_mpc_capacity.py` | PARTIAL→IMPROVED | 新增 `mujoco_oracle_mpc_closed_loop` mode |
| `scripts/render_closed_loop_rollout.py` | NEW | 闭环 rollout 视频渲染 |
| `scripts/run_c23_strictstop_eval.py` | NEW | c23 strict stop confirmatory eval 入口 |

### Finding 5：Strict Pose Stop 代码 Bug 修复（2026-05-12）

**问题：** legacy 5cm success 设置 `success=True`，chunk end 的 `if success and not disable_early_stop: break` 导致 strict pose stop 开启时仍被 5cm 截断。

**修复（`src/metrics/mujoco_oracle_capacity.py`）：**
- 引入 `should_stop`、`legacy_success_reached`、`strict_pose_stop_active`
- strict pose active 时，legacy 5cm 只写 `legacy_success_reached=True`，不设 `should_stop`
- chunk end 改为 `if should_stop: break`
- strict 阈值判断改为 `<=`（`pos_error <= threshold AND theta_error_deg <= threshold`）
- py_compile 通过，正式 smoke test 待执行

### 当前 Critical Blockers 更新

1. **Obstacles not instantiated**（未变）→ 阻塞 layout OOD capacity
2. **Strict pose stop smoke test 待执行**（新增）→ 需人工执行确认修复有效
3. **Task success 已建立**（已解决）→ Oracle-MPC 在 open_space 上可达毫米级精度

### 下一步 Critical Path 更新

1. 执行 strict pose stop smoke test（1 template）
2. 执行 c23_strict600 confirmatory eval（3 open_space + 3 mild_offset）
3. 接入真实 MuJoCo obstacles
4. 在真实 obstacle 下用 config23 做 gate test
5. obstacle gate 通过后做小范围局部 sweep（16 configs）
6. 然后进入 dataset generation gate

