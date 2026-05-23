# Paper1 重要关卡 & Oracle-MPC 能力验证手册

> **最后更新：** 2026-05-14  
> **用途：** 简明记录 Paper1 各关卡进度、MPC 参数配置，以及 Oracle 能力示范视频。

---

## 一、关卡总览

| Gate | 名称 | 状态 | 关键结论 |
|------|------|------|----------|
| 0 | Repo 骨架 & Configs | ✅ PASS | 项目结构 + 配置文件就绪 |
| 1 | Reset Template 生成 | ✅ PASS | 140/140 模板通过几何校验 |
| 2 | 模型 Smoke Test | ✅ PASS | 三种 encoder + 两种 head forward/backward 通过 |
| 3 | Toy Oracle-MPC | ✅ PASS | 20/20 模板：planned_cost < zero_cost |
| 4 | MuJoCo Env 脚手架 | ✅ PASS | reset/step/clone/restore/contact 全部通过 |
| 5 | MuJoCo Oracle Rollout | ✅ PASS | 真实动力学 rollout + state restore 正确 |
| 6 | MuJoCo Oracle-MPC 接口 | ✅ PASS | 5/5 模板接口验证通过 |
| **7** | **Oracle-MPC 任务能力** | **🔄 进行中** | **毫米级精度确认，obstacle gate 待过** |
| 8 | Obstacle 布局容量 | ⬜ 待开始 | blocking/narrow_passage/edge_goal |
| 9 | Sim 数据收集 | ⬜ 待开始 | 依赖 Gate 7/8 |
| 10 | Learned Model 训练 | ⬜ 待开始 | 依赖 Gate 9 |
| 11 | Learned Model + MPC | ⬜ 待开始 | 依赖 Gate 10 |
| 12 | OOD Gap 对比 | ⬜ 待开始 | 依赖 Gate 11 |
| 13 | Real-ID Adapted OOD | ⬜ 待开始 | 依赖 Gate 12 |

---

## 二、MPC 参数配置（当前主配置）

### 2.1 c23_precise（主 baseline）

CEM-MPC + MuJoCo oracle dynamics，closed-loop 执行。

| 参数 | 值 | 说明 |
|------|-----|------|
| `horizon` | **80** | 规划视野（steps） |
| `execute_steps` | **20** | 每次 MPC 执行步数 |
| `max_mpc_steps` | **25** | 最大重规划次数 |
| `num_samples` | **1024** | CEM 每轮候选序列数 |
| `num_elites` | **96** | CEM 精英样本数 |
| `num_iterations` | **5** | CEM 迭代轮数 |
| **总预算** | **500 env steps** | 25 × 20 |

**结论：** mean_final_pos_error ≈ **2.70mm**，success_pos_1cm_rate = **1.0**

### 2.2 c23_strict600（confirmatory eval）

增加预算 + strict pose stop（1.5mm + 3° 同时满足才停止）。

| 参数 | 值 |
|------|-----|
| `horizon` | 80 |
| `execute_steps` | 20 |
| `max_mpc_steps` | **30** |
| `num_samples` | 1024 |
| `num_elites` | 96 |
| `num_iterations` | 5 |
| **总预算** | **600 env steps** |
| `stop_pos_threshold` | **0.0015** (1.5mm) |
| `stop_theta_threshold_deg` | **3.0** |

### 2.3 c25_fast（保守备用）

更轻量、更频繁 replan。

| 参数 | 值 |
|------|-----|
| `horizon` | 80 |
| `execute_steps` | **30** |
| `max_mpc_steps` | **20** |
| `num_samples` | **512** |
| `num_elites` | **48** |
| `num_iterations` | 5 |
| **总预算** | **600 env steps** |

**结论：** mean_final_pos_error ≈ **2.38mm**，success_pos_0p5cm_rate = **1.0**

---

## 三、Cost Weights（cost_weights.yaml）

| 项 | 权重 | 说明 |
|----|------|------|
| `goal_position` | 1.0 | 目标位置误差 |
| `goal_rotation` | 0.5 | 目标角度误差 |
| `collision` | 10.0 | 碰撞惩罚 |
| `out_of_bounds` | 20.0 | 越界惩罚（最高） |
| `action_smoothness` | 0.1 | 动作平滑 |
| `contact_affordance` | 0.0 | 接触 affordance（当前关闭） |

---

## 四、成功标准

### Paper 主成功定义

```
primary_success = (final_pos_error ≤ 1cm) AND (final_theta_error_deg ≤ 5°)
```

### 精度层级

| 语义名称 | 定义 | 用途 |
|----------|------|------|
| `primary_success` | pos ≤ 1cm, theta ≤ 5° | **Paper 主 success rate** |
| `coarse_success` | pos ≤ 2cm, theta ≤ 15° | 粗粒度完成 |
| `precision_success` | pos ≤ 0.5cm, theta ≤ 5° | 高精度完成 |
| `strict_completion` | pos ≤ 1.5mm, theta ≤ 3° | 超严格完成 |

### 当前 open_space / mild_offset 目标

| 指标 | 目标 |
|------|------|
| primary_success_rate | ≥ 0.8 |
| success_pos_0p5cm_rate | ≥ 0.6 |
| mean_final_pos_error | ≤ 5mm |
| mean_final_theta_error_deg | ≤ 5° |

---

## 五、ORACLE 能力示范视频

### 5.1 🎬 主示范：18秒推T成功视频

**视频：** [`runs/videos/non_blocking_demo.mp4`](../runs/videos/non_blocking_demo.mp4)  
**时长：** 18.13 秒  
**场景：** MuJoCo 推T任务（T-shape push-to-pose）  
**布局：** open_space（无阻挡物），橙色 T 形物体 + 蓝色球形 pusher → 绿色目标 T  
**展示内容：** CEM-MPC + Oracle dynamics 在 open-space 下完成 push-to-pose 的完整过程  
**意义：** 作为 ORACLE 能力检验管卡的示范视频，展示 task 在 true dynamics 下可解

### 5.2 更多示范视频

| 视频 | 时长 | 场景 | 亮点 |
|------|------|------|------|
| `artifacts/videos/pusher_capacity_speed010.mp4` | 42s | T-shape push | 慢速高精度推T |
| `artifacts/videos/pusher_capacity_speed005.mp4` | 32s | T-shape push | 极慢速超高精度 |
| `artifacts/videos/pusher_capacity_cylinder_speed010.mp4` | 13s | Cylinder push | 圆柱体推送到位 |
| `runs/horizon140_verify_20260515_164753/videos/h140_sp075_blocking_medium_t8.mp4` | 18s | Blocking medium | obstacle 下推T |
| `runs/horizon140_verify_20260515_164753/videos/h140_sp075_blocking_hard_t1.mp4` | 18s | Blocking hard | 更难 obstacle |
| `artifacts/videos/closed_loop_cylinder_strong_exec20_mpc15.mp4` | 30s | Cylinder strong | 闭环圆柱体强推 |
| `artifacts/videos/boundary_config23/config23_t000_noearly_600steps.mp4` | 60s | T-shape full budget | c23 全预算边界探索 |

### 5.3 关键发现

- **5cm early stop 不适合**：no-early-stop full budget 后，Oracle-MPC 可达**毫米级误差**
- **c23_precise**：更快进入 ≤1cm 精度区（~405 steps），适合作为主 baseline
- **c25_fast**：更稳但更慢，适合保守备用

---

## 六、后续关卡路线

```
Gate 6 (✅) → Gate 7 (🔄 当前：obstacle gate) → Gate 8 → Gate 9 → Gate 10 → Gate 11 → Gate 12 → Gate 13
                   ↓                              ↓
           Oracle-MPC capacity          Obstacle 布局容量
           (毫米级精度确认)              (blocking/narrow/edge)
```

**当前待办：**
1. ⬜ 接入真实 MuJoCo obstacles（blocking / narrow_passage / edge_goal）
2. ⬜ 在真实 obstacle 下用 config23 做 gate test
3. ⬜ obstacle gate 通过后做小范围局部 sweep（16 configs）

---

## 七、核心实验命令

### Oracle-MPC capacity check
```bash
cd ~/my_robot_project
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

PYTHONPATH=. python scripts/check_mpc_capacity.py \
  --mode mujoco_oracle_mpc_closed_loop \
  --split train_sim_id \
  --max-templates 20 \
  --horizon 80 \
  --execute-steps 20 \
  --max-mpc-steps 25 \
  --num-samples 1024 \
  --num-elites 96 \
  --num-iterations 5
```

### Confirmatory eval（strict pose stop）
```bash
MUJOCO_GL=egl PYTHONPATH=. python scripts/run_c23_strictstop_eval.py
```

---

## 八、Paper Claim

> **因果感知的物体级表征在推放任务中实现 OOD 泛化。**

- **主任务：** 单臂 planar push-to-pose
- **主 OOD：** Layout OOD（blocking / narrow_passage / edge_goal）
- **次 OOD：** Shape OOD（T → L）
- **固定执行器：** CEM-MPC（所有方法共用）
- **对比对象：** Flat vs Object-Centric vs Causality-Aware 三种 high-level representation
- **科学问题：** Causality-aware object-level representation 是否降低 structural OOD degradation？

---

*文档维护：Paper1 助手（每次 gate 状态变更后更新）*
