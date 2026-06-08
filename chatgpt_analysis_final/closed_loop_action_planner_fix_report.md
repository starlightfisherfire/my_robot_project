# Closed-Loop Action Planner Fix Report

**Date:** 2026-05-25  
**Status:** SMOKE TEST — NOT paper main result

---

## 1. Executive Summary

**Action convention 已修正，learned model + CEM 在 open_space 和 classic templates 上均显示改善，但未达到 success（<2cm）。Flat 模型在所有 family 上表现最稳定，causality_aware 在 open_space 最好但在 classic templates 上较弱。**

---

## 2. Action Convention Contract

| Quantity | Meaning | Unit | Used By |
|----------|---------|------|---------|
| `a_norm` | 归一化动作 | dimensionless | CEM/MPPI/env.step() |
| `a_phys` | 物理速度 = `a_norm * max_speed_mps` | m/s | Model input |
| `disp` | 位移 = `a_phys * control_dt` | m | EE state update |

**max_speed_mps = 0.75, control_dt = 0.1**

---

## 3. Planner Configs

| Planner | Status | Source | Key Params |
|---------|--------|--------|------------|
| CEM (learned) | ✅ 可用 | 实验配置 | H=10, S=32, E=4, I=2, speed=0.75 |
| MPPI (oracle) | ✅ 已有 | mppi_stage2c | speed=0.3, T=0.1, H=100, samples=2048 |
| MPPI (learned) | ❌ NOT_READY | - | 现有 MPPI 不支持 learned rollout cost |

---

## 4. Open-space Results (Phase 5)

| Model | Template | Init Dist | Best Dist | Improved | Contact |
|-------|----------|-----------|-----------|----------|---------|
| **flat** | 1 | 0.264 | **0.174** | ✅ | 0.30 |
| flat | 2 | 0.242 | 0.242 | ❌ | 0.00 |
| flat | 3 | 0.269 | 0.255 | ✅ | 0.07 |
| **object_centric** | 1 | 0.264 | **0.118** | ✅ | 0.33 |
| object_centric | 2 | 0.242 | 0.242 | ❌ | 0.00 |
| object_centric | 3 | 0.269 | 0.262 | ✅ | 0.00 |
| **causality_aware** | 1 | 0.264 | **0.094** | ✅ | 0.43 |
| causality_aware | 2 | 0.242 | 0.242 | ❌ | 0.00 |
| causality_aware | 3 | 0.269 | 0.256 | ✅ | 0.10 |

**Best result: causality_aware on template 1 = 0.094m (9.4cm from goal)**

---

## 5. Classic Template Results (Phase 6)

| Model | Family | Improved | Mean Best | Contact |
|-------|--------|----------|-----------|---------|
| **flat** | blocking | 3/3 | 0.2792 | 0.10 |
| flat | narrow_passage | 2/3 | 0.2528 | 0.11 |
| flat | edge_goal | 3/3 | 0.3093 | 0.16 |
| **object_centric** | blocking | 1/3 | 0.3312 | 0.01 |
| object_centric | narrow_passage | 1/3 | 0.3102 | 0.01 |
| object_centric | edge_goal | 0/3 | 0.3769 | 0.02 |
| **causality_aware** | blocking | 1/3 | 0.3324 | 0.01 |
| causality_aware | narrow_passage | 1/3 | 0.3071 | 0.06 |
| causality_aware | edge_goal | 1/3 | 0.3625 | 0.06 |

---

## 6. Interpretation

### 是否 action mismatch 已修复？
✅ 是。使用 `a_norm → a_phys = a_norm * 0.75 → model` 转换链。

### CEM 是否可用？
✅ 可用。H=10, S=32 每步 ~1-5 秒。

### MPPI 是否可用？
❌ NOT_READY。现有 MPPI 使用 oracle dynamics，不支持 learned rollout。

### 哪个 planner 更适合 learned rollout？
**CEM。** 但需要 exec_steps=10（执行整个计划序列），而不是每步重新规划。

### 哪个模型在 closed-loop 中最有信号？
| 场景 | 最佳模型 | 原因 |
|------|----------|------|
| open_space | **causality_aware** | mean_best=0.1972, contact=0.18 |
| classic templates | **flat** | 改善率最高 (8/9), 最稳定 |

### 当前最可能瓶颈
1. **EE 接近物体的能力不足** — 很多模板 contact_rate=0
2. **Planner horizon 太短** — H=10 只能规划 10 步，不足以完成长距离推动
3. **模型精度有限** — 多步 rollout 累积误差导致规划质量下降

---

## 7. Limitations

- smoke/pilot only
- 3 templates per family
- trained on MPPI-generated data
- CEM config 较小 (H=10, S=32)
- MPPI + learned rollout 未实现
- not paper main result
- not real robot

---

## 8. Next Actions

| 优先级 | 方向 | 描述 |
|--------|------|------|
| 1 | 增大 CEM 配置 | H=20, S=128, I=3，看是否能提升 |
| 2 | 实现 MPPI learned adapter | 让 MPPI 使用 learned dynamics |
| 3 | Cost 改进 | 加 EE-object distance 奖励 |
| 4 | 更多 templates | 扩展到 10 templates per family |
| 5 | Oracle baseline 对比 | 用 oracle MPPI 跑同一批 templates |

---

## 9. Topology / Geometry Audit Summary

### 9.1 Template Geometry by Family

| Family | Count | Mean Difficulty | Mean Blocking | Mean Edge Distance |
|--------|-------|----------------|---------------|-------------------|
| blocking | 20 | 0.261 | 0.000 | 0.194 |
| edge_goal | 20 | 0.347 | 0.000 | 0.119 |
| mild_offset | 40 | 0.234 | 0.000 | 0.177 |
| narrow_passage | 20 | 0.263 | 0.000 | 0.187 |
| non_blocking | 20 | 0.212 | 0.000 | 0.213 |
| open_space | 40 | 0.213 | 0.000 | 0.216 |

**发现:** 所有 family 的 blocking_score 都是 0.000，说明 obstacles 不在 object-goal 直线路径上。"blocking" family 的难度来自 EE 接近路径被阻挡，而不是 object-goal 路径被阻挡。

### 9.2 Topology Metrics in Closed-loop Manifest

✅ 已将 geometry metrics 写入 manifest  
✅ 已按 topology 选择 templates  
⏳ failure-vs-geometry analysis 需要更多数据

### 9.3 Failure vs Geometry (初步)

- open_space templates 中，approach_feasibility_score 高的模板 contact_rate 更高
- edge_goal templates 目前未在闭环中测试
- 需要 overnight run 数据来做完整分析

---

## 10. Modified / Created Files

| File | Action | Reason |
|------|--------|--------|
| `src/planners/action_conventions.py` | Created | Action convention helper |
| `src/planners/learned_planner_adapter.py` | Created | CEM learned rollout adapter |
| `configs/eval/closed_loop_smoke_templates.yaml` | Created | Template matrix |
| `run_phase5.py` | Created | Open-space experiment |
| `run_phase6.py` | Created | Classic template experiment |
| `docs/closed_loop_action_planner_fix_report.md` | Updated | This report |

---

## 10. Decision

- 是否可以进入更大 closed-loop eval：**YES**（改善已确认）
- 是否需要先完成 MPPI-compatible learned rollout：**YES**（对比 oracle）
- 当前结果等级：**pilot**
- 是否能作为论文主结果：**NO**
- 最短下一步命令：**增大 CEM 配置 + 实现 MPPI learned adapter**
