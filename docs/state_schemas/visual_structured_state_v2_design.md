# visual_structured_state_v2_design.md

**Version:** 0.1  
**Date:** 2026-05-24  
**Status:** DESIGN SKELETON — not implemented  

---

## 1. Why v2? Why not keep state16?

canonical_state16 [H,6,16] 是 Paper 1 的第一个工作基线。它的设计原则是：
- 6 tokens: EE, Object, Goal, Obstacle×3
- 16 features: x,y,sinθ,cosθ,vx,vy,ω,size×2,shape×3,mass,friction,contact,valid

这个设计足以验证"因果分解的表征比扁平表征更好"这个核心命题。

但它有几个根本性的局限：

1. **信息密度不足**：16 维中大量字段是填零或静态值，实际有效维度远低于 16
2. **缺乏关系结构化**：所有 token 是独立编码的，EE-object 距离、object-goal 方向等关系信息需要 encoder 自己"重新发现"
3. **缺乏时域结构化**：速度是瞬时值，没有"过去几帧的趋势"或"最近是否卡住"等时域特征
4. **没有视觉 nuisance 建模**：灯光、纹理、颜色在 state16 中完全不存在，但视觉恢复会遇到
5. **物理参数和状态混在一起**：mass/friction 放在 token 里，它们是静态属性而非动态状态

v2 的目标不是替代 state16，而是在其之上建立一个更完整、可消融的实验框架。

## 2. Design Philosophy

v2 的核心设计原则：

1. **从视觉恢复出发**：问"如果从图像/点云重建这个场景，我能得到什么结构化信息？"
2. **分层组织**：按语义层次组织 token，不是按索引阵列
3. **关系显式化**：EE-object、object-goal、object-obstacle 等关系是 explicit features
4. **时域结构化**：速度、加速度、历史趋势、卡住标志是显式特征
5. **Privileged/Nuisance 分离**：物理参数和视觉 nuisance 明确标注，不与主状态混合

## 3. Token Taxonomy

### 3.1 Object Tokens

```
object_tokens:
  roles: [manipulated_object]
  identity:
    - shape_type: T/L/other (categorical)
    - object_id: fixed identifier
  pose:
    - x, y
    - sin_theta, cos_theta
  kinematics:
    - vx, vy, omega (from MuJoCo or finite diff)
    - ax, ay, alpha (optional, from diff)
  geometry:
    - size_x, size_y
    - bounding_box_corners (optional, 4×2)
    - area, aspect_ratio
  visibility:
    - is_occluded (proxy)
    - bbox_in_frame (optional, from render)
  appearance:  # optional, for nuisance ablation
    - color_rgb
    - texture_id
```

### 3.2 Relation Tokens

```
relation_tokens:
  roles:
    - ee_object: End-effector → Object
    - object_goal: Object → Goal
    - object_obstacle_i: Object → Obstacle i
    - ee_obstacle_i: EE → Obstacle i
    - object_boundary: Object → workspace boundary
  features:
    - relative_position: [dx, dy]
    - distance: scalar
    - bearing: angle from object to target
    - relative_velocity: scalar projection
    - clearance: distance to collision
    - alignment: cos similarity of motion → goal direction
    - contact_proxy: distance-based contact likelihood
    - collision_risk_proxy: distance-based collision likelihood
```

### 3.3 Temporal Tokens

```
temporal_tokens:
  features:
    - action_history: [H, 2] last H actions
    - delta_pose_history: [H, 3] last H object deltas
    - contact_proxy_history: [H] last H contact proxies
    - object_moved_history: [H] last H movement magnitudes
    - stuck_proxy: fraction of recent steps with < 1mm movement
    - oscillation_proxy: sign changes in velocity direction
```

### 3.4 Proprio / Action Tokens

```
proprio_action_tokens:
  features:
    - ee_pose: [x, y]
    - ee_velocity: [vx, vy]
    - previous_action: [dx, dy]
    - action_magnitude: ||action||
```

### 3.5 Visual Nuisance (NOT main input)

```
visual_nuisance:
  features:
    - object_color_rgb
    - obstacle_color_rgb
    - table_texture_id
    - background_texture_id
    - light_position
    - light_direction
    - ambient_light
    - diffuse_light
    - specular_light
    - shadow_strength
    - camera_position
    - camera_fovy
    - render_noise_seed
  usage: domain randomization, invariance test, z_nuisance diagnostic
  forbidden: do NOT use as dynamics main input
```

### 3.6 Privileged Physics (NOT main input)

```
privileged_physics:
  features:
    - object_mass
    - object_friction
    - object_inertia
    - true_contact_flag (from MuJoCo)
    - contact_point [x, y]
    - contact_normal [nx, ny]
    - contact_force [fx, fy]
    - contact_mu
    - solref, solimp
  usage: oracle ablation, diagnostics, label/probe
  forbidden: do NOT use as main claim input
```

### 3.7 Targets

```
targets:
  - target_delta_object_pose: [dx, dy, dtheta]
  - next_object_pose: [x, y, theta]
  - contact_label: optional 0/1
  - collision_label: optional 0/1
```

### 3.8 Masks

```
masks:
  - object_valid_mask
  - relation_valid_mask
  - visibility_mask
  - privileged_access_mask
```

## 4. What NOT to use from MuJoCo

以下 MuJoCo 内部状态**禁止**进入 v2 主输入，因为它们在实际机器人上无法获取：

| Forbidden | Reason |
|-----------|--------|
| data.qpos, data.qvel (full) | 完整关节状态不可观测 |
| data.qfrc_actuator | 执行器内力，真机不可测 |
| data.qM (mass matrix) | 完整质量矩阵不可估计 |
| data.contact full details | 只有力传感器可测部分接触 |
| model.body_mass (all) | 只有可称量物体的质量 |

以下可以从视觉恢复到一定程度：
| Recoverable | Method |
|-------------|--------|
| Object pose (x,y,θ) | Object detection + pose estimation |
| Object velocity (vx,vy,ω) | Temporal difference of pose |
| EE pose (x,y) | Forward kinematics (known) |
| Goal pose | Given by task |
| Obstacle positions | Object detection |
| Contact event | Force sensor / pose discontinuity |
| Distance to goal | Computed from poses |
