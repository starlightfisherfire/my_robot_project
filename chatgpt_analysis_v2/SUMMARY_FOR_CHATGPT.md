# Paper1 V2 进度总结 — 交给 ChatGPT 分析

**日期:** 2026-05-24  
**项目:** Paper1 - Causal Representation for Push Manipulation  
**状态:** V2 闭环实验完成初步诊断，发现结构性瓶颈

---

## 一、项目背景

### 1.1 研究目标
比较三种世界模型架构在推块任务中的表现：
- **Flat**: 扁平编码器（baseline）
- **Object-centric**: 物体中心编码器
- **Causality-aware**: 因果感知编码器（slot factorization）

### 1.2 当前阶段
- ✅ 三个模型已训练完成（10 epochs, mppi_stage2c_state16 数据）
- ✅ 离线评估完成（one-step + multi-step rollout）
- ✅ 第一轮 MuJoCo 闭环失败（action scale mismatch）
- ✅ V2 修正实验完成（action scale 已修复）
- ⚠️ V2 结果：仅 flat 模型有微弱改善（1.2cm）

---

## 二、训练情况确认

### 2.1 训练配置
```
数据集: data/sim/mppi_stage2c_state16
训练比例: 70% train, 15% val
Epochs: 10
Batch size: 128
Learning rate: 0.0003
Loss: w_dynamics * dynamics_loss + w_subgoal * subgoal_loss (1.0 + 0.2)
```

### 2.2 训练 Loss 下降

| 模型 | 初始 val_loss | 最终 val_loss | 最佳 val_loss | 下降幅度 |
|------|--------------|--------------|---------------|----------|
| flat | 0.000748 | 0.000408 | 0.000408 (ep10) | 45.5% |
| object_centric | 0.000818 | 0.000521 | 0.000514 (ep8) | 36.3% |
| causality_aware | 0.000880 | 0.000604 | 0.000570 (ep8) | 31.4% |

**结论:** 训练正常收敛，无过拟合。

### 2.3 离线评估结果（5000 test samples）

| 模型 | 一步 RMSE | 10-step RMSE | 20-step RMSE |
|------|-----------|-------------|-------------|
| flat | 15.78 mm | 3.10 mm | 3.86 mm |
| object_centric | 17.87 mm | 2.04 mm | 2.84 mm |
| causality_aware | 18.18 mm | **0.59 mm** | **1.08 mm** |

**结论:** Causality_aware 多步 rollout 最稳，但闭环表现最差。

---

## 三、V1 失败原因（已确认）

### 3.1 Action Scale Mismatch
- 训练数据: `actions_physical` ∈ [-0.5, 0.5] m/s
- CEM 采样: `a_norm` ∈ [-1, 1]
- 模型收到 2-5x 的 OOD 输入

### 3.2 EE Update Bug
- 错误: `ee_xy += action_sequence[t]`（用归一化动作更新 EE）
- 正确: `ee_xy += a_phys * control_dt`

---

## 四、V2 修正与结果

### 4.1 已修正的问题
- ✅ Action scale: `a_phys = a_norm * max_speed_mps`
- ✅ EE update: `ee_xy += a_phys * control_dt`
- ✅ Planner → Model → Env 转换链统一

### 4.2 V2 闭环结果（open_space, 1 template）

| 模型 | Init Dist | Best Dist | Improved | Contact | EE-Obj 变化 |
|------|-----------|-----------|----------|---------|------------|
| **flat** | 0.2644 | **0.2527** | ✅ 1.2cm | 0.07-0.12 | 0.09→0.03→0.06 |
| object_centric | 0.2644 | 0.2644 | ❌ | 0.00 | 0.07→0.18 |
| causality_aware | 0.2644 | 0.2644 | ❌ | 0.00 | 0.09→0.24 |

### 4.3 详细过程（flat 模型）

```
step 0: dist=0.2644 ee_obj=0.0934 contact=0.00
step 1: dist=0.2644 ee_obj=0.1080 contact=0.00
...
step 6: dist=0.2644 ee_obj=0.0499 contact=0.00  ← EE 接近物体
step 7: dist=0.2570 ee_obj=0.0310 contact=0.12  ← 首次接触！距离开始下降
step 8: dist=0.2527 ee_obj=0.0302 contact=0.11  ← 最佳距离
step 9: dist=0.2527 ee_obj=0.0281 contact=0.10
...
step 14: dist=0.2527 ee_obj=0.0620 contact=0.07  ← EE 又远离了
```

**关键发现:** Flat 模型在 step 7 终于接触物体，推动了 1.2cm，但随后失去接触。

---

## 五、核心瓶颈分析

### 5.1 结构性问题（非工程问题）

**问题:** Learned model 只预测 `delta_object_pose`，不预测 EE 应该如何移动。

**后果:**
1. 当 EE 远离物体时，所有 action 的 `delta ≈ 0`，cost 几乎相同
2. CEM 随机采样无法发现"先靠近再推动"的协调策略
3. 即使偶然靠近物体，模型也不知道如何保持接触并推动

### 5.2 数据分布问题

**训练数据来源:** MPPI oracle rollout
- MPPI 本身就能成功推动物体
- 训练数据中 EE 始终在物体附近（因为 MPPI 知道要先靠近）
- 模型从未见过"EE 远离物体"的状态，也从未学过如何从远处接近

### 5.3 因果推理 vs 实用性

| 模型 | 离线 rollout | 闭环表现 | 矛盾原因 |
|------|-------------|----------|----------|
| flat | 最差 | **最好** | 简单模型更容易从有限数据中学到有用模式 |
| causality_aware | **最好** | 最差 | 复杂结构在 OOD 状态下更容易失效 |

---

## 六、Oracle MPPI 对比（已有数据）

### 6.1 MPPI Oracle 最佳配置
```
speed = 0.3 m/s
temperature = 0.1
horizon = 100
num_samples = 2048
init_std = 0.5
execute_steps = 10
max_mpc_steps = 100
success = pose_success_2mm_10deg
```

### 6.2 MPPI Oracle 成功率（mppi_stage2c sweep）

| Config | 成功率 | Mean Final Dist |
|--------|--------|-----------------|
| sp050_T0.2 | **92.4%** | 0.0107m |
| sp030_T0.1 | 90.3% | 0.0168m |
| sp020_T0.3 | 54.2% | 0.0693m |

**结论:** Oracle MPPI 可以达到 92% 成功率，而 learned model + CEM 几乎完全失败。

---

## 七、代码与文件清单

### 7.1 核心代码
```
src/planners/action_conventions.py      # Action convention helper
src/planners/learned_planner_adapter.py  # CEM learned rollout adapter
src/planners/cem_mpc.py                 # CEM planner
src/envs/mujoco_push_env.py             # MuJoCo 环境
src/models/rig_world.py                 # 三模型 wrapper
src/models/encoders.py                  # Flat/ObjectCentric/CausalityAware
src/models/heads.py                     # DynamicsHead + SubgoalHead
scripts/train_high_level.py             # 训练脚本
scripts/run_closed_loop_corrected.py    # 修正后的闭环脚本
```

### 7.2 训练产物
```
runs/pilot_state16_mppi_stage2c/train_flat/flat/checkpoints/best.pt
runs/pilot_state16_mppi_stage2c/train_object_centric/object_centric/checkpoints/best.pt
runs/pilot_state16_mppi_stage2c/train_causality_aware/causality_aware/checkpoints/best.pt
runs/pilot_state16_mppi_stage2c/train_*/normalizer_state16.json
```

### 7.3 评估结果
```
runs/pilot_state16_mppi_stage2c/offline_eval/offline_eval_summary.json
runs/pilot_state16_mppi_stage2c/training_loss_curves.png
docs/closed_loop_action_planner_fix_report.md
docs/action_convention_contract.md
docs/planner_best_config_contract.md
```

---

## 八、给 ChatGPT 的问题

### 8.1 核心问题
1. **如何让 learned model 引导 EE 接近物体？**
   - 修改 cost function？加入 EE-object distance 奖励？
   - 重新训练模型，加入 approach 阶段数据？
   - 用两阶段规划（oracle approach + learned push）？

2. **Causality-aware 模型为什么闭环最差？**
   - 离线 rollout 最好但闭环最差，说明什么？
   - Slot factorization 在 OOD 状态下是否更容易失效？
   - 如何改进？

3. **Oracle MPPI 92% vs Learned CEM ~0%，差距来自哪里？**
   - Dynamics prediction accuracy？
   - Planning horizon/samples？
   - 还是模型根本没学到有用的物理规律？

4. **下一步最有 ROI 的方向是什么？**
   - A: 改 cost function（加 EE 接近奖励）
   - B: 两阶段规划（oracle + learned）
   - C: 重新训练（更多 approach 数据）
   - D: 放弃 learned model，直接用 oracle MPPI
   - E: 其他？

### 8.2 技术细节问题
5. **MPPI 能否直接用 learned dynamics 替换 oracle？**
   - 现有 MPPI 实现使用 MuJoCo oracle rollout
   - 如何最小改动让 MPPI 使用 learned model？

6. **Horizon=140 的 CEM 是否对 learned model 可行？**
   - 用户指定的 CEM 最佳配置 H=140
   - 但 learned model 140 步累积误差太大
   - 有没有折中方案？

---

## 九、下一步建议

### 9.1 立即可做（1-2天）
1. **Oracle MPPI baseline:** 用已有 MPPI oracle 跑 open_space + classic templates，确认 oracle 能成功
2. **Cost function 修改:** 在 CEM cost 中加入 `λ * ee_to_object_dist`，引导 EE 接近物体
3. **更多 templates:** 跑 10 个 open_space templates，统计 flat model 的改善率

### 9.2 中期（1周）
4. **两阶段规划:** Oracle MPPI 做 approach，learned model 做 fine manipulation
5. **MPPI learned adapter:** 让 MPPI 使用 learned dynamics（而非 oracle）

### 9.3 长期（需要决策）
6. **重新训练:** 训练数据包含 approach 阶段，或模型增加 EE subgoal 输出
7. **架构改进:** Causality-aware 模型需要在 OOD 状态下更鲁棒

---

**压缩包内容:**
- `README.md` — 快速开始指南
- `SUMMARY_FOR_CHATGPT.md` — 本文件
- `core_files/` — 核心代码（12个文件）
- `configs/` — 训练配置（4个文件）
- `results/` — 评估结果（6个文件）
- `docs/` — 文档（4个文件）
