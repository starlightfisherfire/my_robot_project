# Topology / Geometry Protocol

**Date:** 2026-05-25  
**Purpose:** 诊断 Push-T 模板的结构难度，与闭环结果关联分析

---

## 1. 为什么需要 Topology / Geometry Audit

Learned closed-loop 实验中，部分模板成功、部分失败。失败原因可能是：
1. 模型/Planner 问题
2. 模板本身几何难度过高

本模块将两者分离，回答："这个模板几何上应该可解吗？"

---

## 2. 几何合法性检查

| 检查项 | 含义 |
|--------|------|
| workspace_bounds | object/goal/obstacles 是否在 workspace [0, 0.7] × [0, 0.5] 内 |
| object_pose_valid | object 在 workspace 内且 theta 有效 |
| goal_pose_valid | goal 在 workspace 内且 theta 有效 |
| obstacle_pose_valid | obstacles 在 workspace 内 |
| no_object_obstacle_overlap | 初始状态 object 与 obstacles 不重叠 |
| no_goal_obstacle_overlap | goal 与 obstacles 不重叠 |

---

## 3. 拓扑难度指标

### 3.1 基础几何

| 指标 | 含义 | 计算 |
|------|------|------|
| object_goal_distance | object 到 goal 的欧氏距离 | norm(obj_xy - goal_xy) |
| goal_edge_distance | goal 到最近边界的距离 | min(goal_x, 0.7-goal_x, goal_y, 0.5-goal_y) |
| min_object_obstacle_distance | object 到最近 obstacle 的距离 | min over obstacles |
| min_goal_obstacle_distance | goal 到最近 obstacle 的距离 | min over obstacles |
| obstacle_count | obstacle 数量 | count |

### 3.2 路径拓扑

| 指标 | 含义 | 计算 |
|------|------|------|
| direct_path_blocked | 直线路径是否被阻挡 | ray-cast obj→goal 检查碰撞 |
| obstacle_between_object_goal | obj-goal 连线上是否有 obstacle | 投影距离检查 |
| blocking_score | 阻挡程度 [0,1] | 综合 obstacle 位置和大小 |
| passage_width_estimate | 通道宽度估计 | obstacle 间距 - obstacle 尺寸 |
| edge_goal_score | goal 靠边程度 [0,1] | 1 - edge_distance / max_edge |

### 3.3 接触可达性

| 指标 | 含义 | 计算 |
|------|------|------|
| reachable_contact_sides | pusher 可从几个方向接触 object | 检查 4 个方向的 clearance |
| approach_feasibility_score | 接近可行性 [0,1] | 综合 clearance 和距离 |
| ee_to_object_distance | EE 到 object 的距离（如有 EE pose） | norm(ee_xy - obj_xy) |

---

## 4. 对 Failure Analysis 的用途

| 失败模式 | 可能对应的几何指标 |
|----------|-------------------|
| no_contact | approach_feasibility_score 低 |
| weak_contact | contact side / pusher path 难 |
| wrong_direction | goal angle 复杂 |
| stuck | narrowest_corridor_width 小 |
| horizon_too_short | path length / detour ratio 高 |

---

## 5. 使用原则

- 用于诊断和分层分析
- 不作为当前 learned model 主输入
- 不作为论文主 claim 的唯一证据
- 可用于后续 visual_state_v2 relation_tokens / profile ablation
