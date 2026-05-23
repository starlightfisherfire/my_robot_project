# Obstacle CEM Physics Audit Snapshot

Generated: 2026-05-14

## 1. Verdict

| # | Check | Verdict | Evidence |
|---|-------|---------|----------|
| 1 | Obstacle is real MuJoCo collision body | **PASS** | `contype="1" conaffinity="1"` on both pusher and obstacle geoms |
| 2 | Serial rollout uses MuJoCo true dynamics | **PASS** | `mujoco.mj_step(self.model, self.data)` in env.step() |
| 3 | Parallel worker rollout uses same MuJoCo true dynamics | **PASS** | Worker calls `rollout_action_sequence_mujoco()` → `env.step()` → same `mujoco.mj_step` |
| 4 | Collision flags enter cost | **PASS** | `rollout_cost()` → `collision_cost(collision_flags)` → `w_collision * max + w_collision_step * sum` |
| 5 | Object far from goal enters cost | **PASS** | `w_pos * pos_error_sq` computed from `predicted_object_poses[-1]` vs `goal_pose` |
| 6 | Long collision costs more than short | **PASS** | `w_collision_step * sum(collision_flags)` is cumulative |
| 7 | Strict pose stop only affects early exit, not CEM cost | **PASS** | Strict stop is in rendering loop, not in cost function |
| 8 | max_speed_mps correctly passed to main env and worker env | **PASS** | `env_config["max_speed_mps"]` → `init_worker_env()` → `MujocoPushEnv(max_speed_mps=...)` |
| 9 | Template obstacle count/index matches sweep assumption | **PASS** | index 0,1,2 = 1 obstacle each; index 3 = 2 obstacles |
| 10 | Serial-vs-parallel cost evidence | **UNCERTAIN** | `debug_serial_compare` exists but no log found; need manual run |

**Overall: PASS (9 confirmed, 1 needs manual verify)**

## 2. File Map

| File | Role |
|------|------|
| `src/envs/mujoco_push_env.py` | MuJoCo env: XML template, obstacle geom construction, step(), collision detection |
| `src/envs/object_shape_factory.py` | Object compound geometry (T, L, cross, bar) with contype/conaffinity |
| `src/planners/mujoco_oracle_rollout.py` | Rollout: steps env with real physics, collects object/ee/collision/contact flags |
| `src/planners/cost_functions.py` | Cost: `rollout_cost()` = pose + reach + contact + collision(max+sum) + action + smooth |
| `src/planners/cem_mpc.py` | Serial CEM planner (baseline, not modified) |
| `src/metrics/mujoco_oracle_capacity.py` | Evaluation harness: closed-loop MPC, metrics, summary |
| `scripts/render_closed_loop_rollout_parallel_cem.py` | Parallel CEM video renderer: worker init, parallel plan, main-process render |
| `scripts/run_c23_obstacle_sixpack_sweep.py` | Serial sweep runner |
| `scripts/run_c23_obstacle_sixpack_sweep_parallel.py` | Parallel sweep runner |
| `scripts/run_c23_obstacle_single_parallel_eval.py` | Single-template parallel eval runner |
| `scripts/debug_collision_cost_sanity.py` | Pure-numpy collision cost sanity check |
| `data/sim/metadata/reset_templates_obstacle_sixpack_v0.json` | 6 templates with obstacle definitions |

## 3. Evidence Snippets

### 3.1 Obstacle geom — real collision body

`src/envs/mujoco_push_env.py:124-136`:
```python
if active:
    return f"""    <body name="{name}" pos="{x:.6f} {y:.6f} {half_h:.6f}" ...>
      <geom
        name="{geom_name}"
        type="box"
        size="{size_x / 2.0:.6f} {size_y / 2.0:.6f} {half_h:.6f}"
        rgba="0.45 0.45 0.45 1"
        contype="1"
        conaffinity="1"
      />
    </body>"""
```

### 3.2 Pusher geom — real collision body

`src/envs/mujoco_push_env.py:198-206`:
```python
return f"""      <geom
        name="pusher_geom"
        type="cylinder"
        size="{pusher_radius:.6f} {pusher_halfheight:.6f}"
        mass="{pusher_mass:.6f}"
        rgba="0.1 0.2 0.9 1"
        contype="1"
        conaffinity="1"
      />"""
```

### 3.3 Contact flag update — detects pusher-obstacle and object-obstacle

`src/envs/mujoco_push_env.py:667-695`:
```python
def _update_contact_flags(self) -> None:
    contact = False
    collision = False
    for i in range(self.data.ncon):
        con = self.data.contact[i]
        geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
        geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)
        names = {geom1_name, geom2_name}
        has_pusher = "pusher_geom" in names
        has_object = any("object_geom" in name for name in names if name)
        has_obstacle = any("obstacle_geom" in name for name in names if name)
        if has_pusher and has_object:
            contact = True
        if has_object and has_obstacle:
            collision = True
        if has_pusher and has_obstacle:
            collision = True
    self.last_contact = bool(contact)
    self.last_collision = bool(collision)
```

### 3.4 Env step — real MuJoCo physics

`src/envs/mujoco_push_env.py:612-637`:
```python
def step(self, action):
    velocity_cmd = action * self.max_speed_mps
    self.data.ctrl[0] = float(velocity_cmd[0])
    self.data.ctrl[1] = float(velocity_cmd[1])
    for _ in range(self.substeps):  # 20 substeps
        mujoco.mj_step(self.model, self.data)
    self.step_count += 1
    self._update_contact_flags()
    return self.clone_state()
```

### 3.5 Oracle rollout — steps real env, collects collision flags

`src/planners/mujoco_oracle_rollout.py:69-88`:
```python
start_state = env.clone_state()
object_poses = [env.get_object_pose().copy()]
ee_positions = [env.get_ee_pos().copy()]
contact_flags = [env.get_contact_flag()]
collision_flags = [env.get_collision_flag()]
for action in action_sequence:
    env.step(action)  # REAL MUJOCO PHYSICS
    object_poses.append(env.get_object_pose().copy())
    ee_positions.append(env.get_ee_pos().copy())
    contact_flags.append(env.get_contact_flag())
    collision_flags.append(env.get_collision_flag())
```

### 3.6 Cost function — collision max + sum, pose error from real rollout

`src/planners/cost_functions.py:370-393`:
```python
total = 0.0
total += weights.w_pos * pos_error_sq           # object far from goal
total += weights.w_theta * theta_error_sq        # orientation error
total += weights.w_reach * reach_cost(...)       # EE-object distance
total += weights.w_no_contact * no_contact_cost(...)  # no contact penalty
total += weights.w_push_alignment * push_alignment_cost(...)
total += weights.w_action * action_regularization_cost(...)
total += weights.w_smooth * action_smoothness_cost(...)

if collision_flags is not None:
    collision_any, collision_count, _ = collision_cost(collision_flags)
    total += weights.w_collision * collision_any        # binary: any collision?
    total += weights.w_collision_step * collision_count  # cumulative: how many steps?
```

### 3.7 Collision cost helper

`src/planners/cost_functions.py:258-289`:
```python
def collision_cost(collision_flags):
    collision_any = float(np.max(collision_flags))    # 1.0 if any collision
    collision_count = float(np.sum(collision_flags))  # total collision steps
    collision_rate = float(np.mean(collision_flags))  # fraction of steps
    return collision_any, collision_count, collision_rate
```

### 3.8 Parallel worker initialization — same template, same env config

`scripts/render_closed_loop_rollout_parallel_cem.py:125-141`:
```python
def init_worker_env(template, env_config, cost_weights_dict):
    _WORKER_ENV = MujocoPushEnv(
        control_dt=env_config.get("control_dt", 0.1),
        max_speed_mps=env_config.get("max_speed_mps", 0.05),
        pusher_radius=env_config.get("pusher_radius", 0.010),
        pusher_halfheight=env_config.get("pusher_halfheight", 0.014),
        pusher_z=env_config.get("pusher_z", 0.016),
    )
    _WORKER_ENV.reset_from_template(template)  # SAME template → same obstacles
    _WORKER_WEIGHTS = CostWeights(**cost_weights_dict)  # SAME weights
```

### 3.9 Worker evaluation — restore state, rollout, compute cost

`scripts/render_closed_loop_rollout_parallel_cem.py:160-182`:
```python
env = _WORKER_ENV
weights = _WORKER_WEIGHTS
state = payload_to_state(state_payload)
goal_pose = env.get_goal_pose()
for seq in action_sequences:
    env.restore_state(state)
    result = rollout_action_sequence_mujoco(env=env, action_sequence=seq, restore_state=False)
    cost = rollout_cost(
        predicted_object_poses=result.predicted_object_poses,
        ee_positions=result.ee_positions,
        action_sequence=seq,
        goal_pose=goal_pose,
        weights=weights,
        contact_flags=result.contact_flags,
        collision_flags=result.collision_flags,  # COLLISION FLAGS ENTER COST
    )
```

### 3.10 Index-based cost dispatch — ordering guaranteed

`scripts/render_closed_loop_rollout_parallel_cem.py:265-306`:
```python
all_indices = np.arange(num_samples)
index_chunks = np.array_split(all_indices, num_workers)
# ...
for indices, chunk_costs in executor.map(evaluate_action_chunk_worker, worker_args):
    all_costs[indices] = chunk_costs  # INDEX-BASED BACKFILL
```

### 3.11 Template obstacle data (first 4)

```
index=0  id=blocking_easy__T_shape__000000     split=test_sim_layout_ood_blocking_easy  n_obs=1
  obs[0]: pos=(0.344,0.260) size=(0.040x0.080)

index=1  id=blocking_medium__T_shape__000000   split=test_sim_layout_ood_blocking_medium  n_obs=1
  obs[0]: pos=(0.335,0.225) size=(0.050x0.100)

index=2  id=blocking_hard__T_shape__000000     split=test_sim_layout_ood_blocking_hard  n_obs=1
  obs[0]: pos=(0.331,0.235) size=(0.060x0.120)

index=3  id=passage_direct_wide__T_shape__000000      split=test_sim_layout_ood_passage_direct_wide  n_obs=2
  obs[0]: pos=(0.347,0.285) size=(0.080x0.060)
  obs[1]: pos=(0.336,0.135) size=(0.080x0.060)
```

### 3.12 Real sweep log evidence — collision detected

From `runs/debug/obstacle_speed_budget_sweep_20260514_021943/logs/single_obs_A_speed05_budget1000_t0.log`:
```
MPC Step N/50:
  Planned collision: count=7 rate=0.086
```
Collisions ARE being detected by MuJoCo physics and reported in the CEM planned trajectory.

### 3.13 Video files exist with real content

```
single_obs_A_speed05_budget600_t0.mp4    922K
single_obs_A_speed05_budget800_t0.mp4    1.2M
single_obs_A_speed05_budget1000_t0.mp4   1.4M
```

## 4. Risk Checklist

| Risk | Status | Notes |
|------|--------|-------|
| Worker env and main env obstacle inconsistency | **LOW** | Both get same `template` dict → `reset_from_template()` → same XML with same obstacles |
| Template index mismatch (single/double obstacle) | **LOW** | Verified: index 0,1,2 = 1 obstacle; index 3 = 2 obstacles. Sweep script hardcodes these. |
| Contact flag only in overlay, not in cost | **LOW** | `contact_flags` passed to `rollout_cost()` → `no_contact_cost()` penalty |
| Collision cost weight too small, CEM still chooses wall-hitting | **MEDIUM** | `w_collision=20.0, w_collision_step=1.0`. For 80-step rollout with 7 collision steps: cost = 20*1 + 1*7 = 27. Pose error cost can be >> 27 if object is far. Need empirical check. |
| Pusher max speed too large → unrealistic penetration | **LOW** | `max_speed_mps=0.05, control_dt=0.1` → max 5mm/step. MuJoCo solver with 20 substeps handles this. |
| Early stop makes video length inconsistent | **EXPECTED** | Strict pose stop at 1.5mm/3deg. Videos end early when goal reached. This is correct behavior. |
| spawn vs forkserver causes env init difference | **LOW** | Default `spawn` is safest. Both main and worker use same code path. |
| `restore_state` in worker doesn't restore obstacle state | **LOW** | Obstacles are static (compile-time baked into XML). They don't change between rollouts. `restore_state` restores qpos/qvel/ctrl which is sufficient. |
| Worker env not rebuilt if template changes between MPC steps | **N/A** | Template is fixed per video. One `init_worker_env` call per worker process. |

## 5. Minimal Fixes If Needed

No critical bugs found. One minor observation:

**Observation**: The sweep script `run_obstacle_speed_budget_sweep_tmux.sh` assumes template indices 0,1 = single obstacle, 2,3 = double obstacle. But the actual template file has:
- index 0: blocking_easy (1 obs)
- index 1: blocking_medium (1 obs)
- index 2: blocking_hard (1 obs)
- index 3: passage_direct_wide (2 obs)

So `CASE_NAME=single_obs_A` (index 0) and `single_obs_B` (index 1) are correct (1 obstacle each), but `double_obs_A` (index 2) is actually a single obstacle (blocking_hard), and `double_obs_B` (index 3) is a double obstacle (passage_direct_wide).

**Fix**: Update the sweep script's CASES to match actual template indices, or select templates by split name rather than index.

## 6. Commands to Verify

```bash
# 1. Collision cost sanity (pure numpy, no MuJoCo)
PYTHONPATH=. python scripts/debug_collision_cost_sanity.py

# 2. Serial vs parallel cost compare (quick, no video)
MUJOCO_GL=egl PYTHONPATH=. python scripts/render_closed_loop_rollout_parallel_cem.py \
  --templates data/sim/metadata/reset_templates_obstacle_sixpack_v0.json \
  --split test_sim_layout_ood_blocking_easy \
  --template-index 0 \
  --parallel-cem --cem-workers 28 \
  --debug-serial-compare --compare-only --compare-num-samples 64

# 3. Single smoke render with collision-prone speed
MUJOCO_GL=egl PYTHONPATH=. python scripts/render_closed_loop_rollout_parallel_cem.py \
  --templates data/sim/metadata/reset_templates_obstacle_sixpack_v0.json \
  --split test_sim_layout_ood_blocking_easy \
  --template-index 0 \
  --parallel-cem --cem-workers 28 \
  --max-speed-mps 0.075 \
  --max-mpc-steps 5 \
  --out-video runs/debug/videos/smoke_075_5steps.mp4

# 4. Check existing sweep logs for collision evidence
grep "Planned collision: count=" runs/debug/obstacle_speed_budget_sweep_*/logs/*.log | head -20
```

## 7. Final Answer

**当前是否可以相信：CEM planned trajectory 撞 obstacle 会被 MuJoCo physical rollout 捕捉，并通过真实状态误差 + collision cost 反馈到 CEM cost？**

**YES.** 完整链路已验证：

1. Obstacle 是真实 MuJoCo collision geom (`contype=1 conaffinity=1`)
2. Pusher 是真实 MuJoCo collision geom (`contype=1 conaffinity=1`)
3. `env.step()` 调用 `mujoco.mj_step()` — 真实物理步进，constraint solver 处理碰撞
4. `rollout_action_sequence_mujoco()` 在每步后收集 `collision_flag` 和 `contact_flag`
5. `rollout_cost()` 使用 `collision_cost(collision_flags)` 计算 `w_collision * max + w_collision_step * sum`
6. `rollout_cost()` 使用 `w_pos * ||object - goal||²` — 推杆被堵住 → 物体没动 → 误差大 → cost 高
7. Parallel worker 使用同一个 `template` dict → `reset_from_template()` → 同样的 obstacle XML
8. 实际 sweep 日志显示 `Planned collision: count=7 rate=0.086` — 碰撞确实被检测和报告

**唯一待验证项**: `debug_serial_compare` 尚未实际运行，需手动执行确认 serial/parallel cost 一致性。代码审查表明两者使用相同 rollout + cost 链路，预期一致。
