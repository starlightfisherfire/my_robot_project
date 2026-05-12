# Experiment Log

**Last updated:** 2026-05-12

本文档记录各阶段实验的参数、结果和结论。

---

## EXP-001：Wide Sweep with 5cm Early Stop（2026-05-11）

**目录：** `runs/sweeps/wide_overnight_v2_20260511_012735`

**参数：**
- 117 configs 完成（达到 timeout）
- 5cm early stop 开启

**结果：**
- 所有 top configs 的 `success_rate = 1.0`（5cm 阈值）
- top `final_pos_error` 集中在 0.0487–0.0491 m
- `success_pose_2cm_15deg_rate = 0`（全部为 0）

**结论：**
- 5cm early stop 截断了更强的 config，无法测量真实精度边界
- 此 sweep 只能作为 5cm 粗成功能力图，不能用于精度分析
- 需要 no-early-stop boundary search

---

## EXP-002：Boundary Refine v1（2026-05-11）

**目录：** `runs/video_sweeps/boundary_refine_v1_20260511_213206`

**参数：**
- `disable_early_stop=True`
- 64 configs（horizon × execute_steps × max_mpc_steps × num_samples × num_iterations）
- split: train_sim_id, max_templates: 5

**结论：**
- 确认 no-early-stop 下 Oracle-MPC 可达到毫米级误差
- 为 boundary_video_night2 提供了 config 筛选依据

---

## EXP-003：Boundary Video Night2（2026-05-12）

**目录：** `runs/video_sweeps/boundary_video_night2_20260512_001900`

**文件：**
- `c23_precise_train5.json`
- `c25_fast_train5.json`
- `compact_summary.txt`
- `c23_precise_t000~t004` 视频
- `c25_fast_t000~t004` 视频

### c23_precise 结果（train_sim_id, 5 templates）

| 指标 | 值 |
|------|-----|
| mean_final_pos_error | ≈ 2.70 mm |
| median_final_pos_error | ≈ 1.55 mm |
| mean_best_pos_error | ≈ 2.53 mm |
| mean_final_theta_error_deg | ≈ 2.48° |
| success_pos_1cm_rate | 1.0 |
| success_pos_0p5cm_rate | 0.8 |
| success_pose_0p5cm_5deg_rate | 0.6 |
| mean_first_reach_1cm_steps | ≈ 404.6 |
| mean_first_reach_0p5cm_steps | ≈ 414.0 |

### c25_fast 结果（train_sim_id, 5 templates）

| 指标 | 值 |
|------|-----|
| mean_final_pos_error | ≈ 2.38 mm |
| median_final_pos_error | ≈ 1.21 mm |
| mean_best_pos_error | ≈ 2.19 mm |
| mean_final_theta_error_deg | ≈ 1.22° |
| success_pos_0p5cm_rate | 1.0 |
| success_pose_0p5cm_5deg_rate | 1.0 |
| mean_first_reach_1cm_steps | ≈ 449.6 |
| mean_first_reach_0p5cm_steps | ≈ 522.6 |

### 对比分析

| 维度 | c23_precise | c25_fast |
|------|-------------|----------|
| 到达 1cm 的速度 | 更快（~405 steps） | 更慢（~450 steps） |
| 最终精度 | 略差 | 略好 |
| 预算利用 | 500 steps，部分 hard case 略不足 | 600 steps，更充裕 |
| 定位 | 主 baseline | 保守备用 |

**结论：**
- c23 更快进入高精度区，更适合作为主 baseline
- c25 更稳但更慢，适合保守备用
- c23 在 500 steps 下部分 hard case 预算略不足 → 需要 c23_strict600（600 steps）验证
- 视频显示 planner 已出现 near-goal refinement 和换侧修正行为
- 当前问题不再是"planner 完全不会推"，而是高精度停止、预算分配和 obstacle gate

---

## EXP-004：c23_strict600 Confirmatory Eval（待执行）

**状态：** 待执行（需先通过 smoke test）

**计划参数：**
- config: c23_strict600（horizon=80, execute_steps=20, max_mpc_steps=30, num_samples=1024, num_elites=96, num_iterations=5）
- strict_pose_stop: pos<=1.5mm AND theta<=3°
- templates: 3 open_space + 3 mild_offset（train_sim_id）
- 脚本: `scripts/run_c23_strictstop_eval.py`

**注意：**
- mild_offset 只能算 weak constrained，不是真正 obstacle
- 结果不能写成真实 obstacle performance

---

## EXP-005：真实 Obstacle Gate（待实现）

**状态：** 待实现（MujocoPushEnv 尚未接入 obstacle）

**前置条件：**
1. MujocoPushEnv 读取 template["obstacles"]
2. MuJoCo XML 动态生成 obstacle geom
3. obstacle 参与 collision
4. render 视频可视化确认

**计划：**
- 在 blocking / narrow_passage / edge_goal 模板上用 config23_strict600 测试
- 通过标准：不崩、不穿模、不乱飞，视频可视化确认
