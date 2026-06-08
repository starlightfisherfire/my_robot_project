# Closed-Loop Action Planner Fix Report

**Date:** 2026-05-24  
**Status:** SMOKE TEST — NOT paper main result

---

## 1. Executive Summary

**Action convention mismatch 已修正。CEM + learned rollout 在 open_space 上仅有 flat 模型显示微弱改善（1.2cm），object_centric 和 causality_aware 完全失败。MPPI + learned rollout 未实现（MPPI_NOT_READY）。根本瓶颈是 learned model 无法引导 EE 接近并推动物体。**

---

## 2. Action Convention Contract

| Quantity | Meaning | Unit | Used By |
|----------|---------|------|---------|
| `a_norm` | 归一化动作 | dimensionless | CEM/MPPI/env.step() |
| `a_phys` | 物理速度 | m/s | Model input |
| `disp` | 位移 | m | EE update: `a_phys * control_dt` |

**验证:** ✅ 已统一，action scale mismatch 已修复

---

## 3. Planner Configs

| Planner | Status | Source | Key Params |
|---------|--------|--------|------------|
| CEM (learned) | ✅ 可用 | 实验配置 | H=5, S=16, E=4, I=1, speed=0.75 |
| CEM (user best) | ⚠️ 太慢 | 用户提供 | H=140, speed=0.75, exec_steps=10 |
| MPPI (oracle) | ✅ 已有 | mppi_stage2c | speed=0.3, T=0.1, H=100, samples=2048 |
| MPPI (learned) | ❌ NOT_READY | - | 现有 MPPI 不支持 learned rollout cost |

---

## 4. Open-space Results

| Model | Init Dist | Best Dist | Final Dist | Improved | Contact | EE-Obj |
|-------|-----------|-----------|------------|----------|---------|--------|
| flat | 0.2644 | 0.2527 | 0.2527 | ✅ YES (1.2cm) | 0.07-0.12 | 0.03→0.06 |
| object_centric | 0.2644 | 0.2644 | 0.2644 | ❌ NO | 0.00 | 0.05→0.18 |
| causality_aware | 0.2644 | 0.2644 | 0.2644 | ❌ NO | 0.00 | 0.09→0.24 |

**关键观察：**
1. Flat 模型在 step 7 开始接触物体，推动了 1.2cm
2. Object_centric 从未接触物体
3. Causality_aware EE 反而远离物体（0.09→0.24）

---

## 5. Classic Template Results

未运行（open_space 改善不足，不满足进入 Phase 5 的条件）

---

## 6. Interpretation

### 是否 action mismatch 已修复？
✅ 是。env.step 接收 normalized action，model 接收 physical velocity，EE 用 displacement 更新。

### CEM 是否可用？
⚠️ 可用但极慢。H=5/S=16/I=1 每步 ~10s，H=140 的用户配置每步需要数小时。

### MPPI 是否可用？
❌ NOT_READY。现有 MPPI 实现使用 oracle MuJoCo dynamics，不支持 learned rollout cost。

### 哪个 planner 更适合 learned rollout？
**Oracle MPPI（已有）。** 学到的模型无法替代 oracle dynamics。

### 哪个模型在 closed-loop 中最有信号？
**Flat。** 唯一显示 contact 和 improvement 的模型。

### 当前最可能瓶颈
**Learned model 的结构性缺陷：**
1. 模型只预测 `delta_object_pose`，不预测 EE 应该如何移动
2. 训练数据来自 MPPI oracle（EE 始终在物体附近），模型从未学过"如何接近物体"
3. 当 EE 远离物体时，所有 action 得到近似相同的 cost（delta ≈ 0），planner 无法区分好坏动作
4. CEM 随机采样无法发现"先靠近再推动"的协调策略

---

## 7. Limitations

- smoke/pilot only，仅 1 个 open_space template
- 训练数据来自 MPPI-generated data
- CEM 配置极小（H=5, S=16），用户最佳配置未实际使用（太慢）
- MPPI + learned rollout 未实现
- 不是 paper main result
- 不是 real robot

---

## 8. Next Actions

| 优先级 | 方案 | 描述 |
|--------|------|------|
| 1 | Oracle MPPI baseline | 直接用已有 MPPI oracle 跑 open_space，确认 oracle 能成功 |
| 2 | Cost 加 EE 接近奖励 | 在 planning cost 中加入 `λ * ee_to_object_dist` |
| 3 | 两阶段规划 | Phase 1: oracle MPPI 移动 EE 到物体附近；Phase 2: learned model 做精细推动 |
| 4 | 重新训练模型 | 训练数据包含 approach 阶段，或模型输出增加 EE subgoal |

---

## 9. Modified / Created Files

| File | Action | Reason |
|------|--------|--------|
| `src/planners/action_conventions.py` | Created | Action convention helper |
| `src/planners/learned_planner_adapter.py` | Created | CEM learned rollout adapter |
| `scripts/run_closed_loop_corrected.py` | Created | Corrected closed-loop script |
| `docs/action_convention_contract.md` | Created | Action convention documentation |
| `docs/planner_best_config_contract.md` | Created | Planner config documentation |
| `runs/closed_loop_action_planner_fix/phase0_audit.md` | Created | Phase 0 audit report |
| `runs/closed_loop_action_planner_fix/results_quick.json` | Created | Experiment results |

---

## 10. Critical Failures

None（无高风险实验语义错误）

---

## 11. Warnings

1. 用户最佳 CEM 配置（H=140）在 learned rollout 上不可行（太慢）
2. MPPI + learned rollout 未实现
3. 仅测试了 1 个 open_space template
4. Flat 模型 improvement 仅 1.2cm，远未达到 success（<2cm）

---

## 12. Decision

- 是否可以进入更大 closed-loop eval：**NO**（open_space 改善不足）
- 是否需要先完成 MPPI-compatible learned rollout：**YES**
- 当前结果等级：**smoke**
- 是否能作为论文主结果：**NO**
- 最短下一步命令：**用 oracle MPPI 跑 open_space baseline，确认 oracle 能成功，然后分析 learned vs oracle 的差距**
