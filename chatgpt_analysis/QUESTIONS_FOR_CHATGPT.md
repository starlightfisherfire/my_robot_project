# ChatGPT 分析请求：Paper1 三模型闭环失败诊断

**日期:** 2026-05-24  
**项目:** Paper1 - Causal Representation for Push Manipulation

---

## 背景

我们训练了三个 learned rollout 模型（flat, object_centric, causality_aware），用于指导 MPPI/CEM 在 MuJoCo 中进行推块任务。离线评估表现尚可（10-step rollout RMSE ~0.6-3mm），但闭环实验全部失败。

---

## 核心问题

### 问题 1：Action Scale Mismatch（已确认）

**训练数据：**
```
actions_physical 范围: [-0.3, 0.3] m/s (物理单位)
```

**CEM 规划器：**
```python
cem = CEMMPC(
    action_low=-1.0,   # 归一化空间
    action_high=1.0,   # 归一化空间
)
```

**MuJoCo 环境内部：**
```python
velocity_cmd = action * max_speed_mps  # max_speed_mps = 0.3
```

**问题：** 模型在训练时学习的是物理尺度 action [-0.3, 0.3]，但 CEM 规划器在归一化空间 [-1, 1] 搜索。这导致模型收到 3x scale 的 OOD 输入。

**疑问：**
1. 这是否是闭环失败的主要原因？
2. 修复方案：修改 CEM 在物理空间搜索 vs 重新训练模型（归一化 action）？
3. 如果修改 CEM，是否需要重新调整 init_std 等参数？

---

### 问题 2：CEM 配置过保守

**当前 CEM 配置：**
```python
horizon = 10
num_samples = 128
num_elites = 16
num_iterations = 3
init_std = 0.5
```

**MPPI 最佳参数（用户提供）：**
```python
speed = 0.3
temperature = 0.1
horizon = 100
num_samples = 2048
init_std = 0.5
execute_steps = 10
max_mpc_steps = 100
```

**问题：**
1. CEM 的 128 samples × 3 iterations 是否足够？
2. Horizon=10 是否太短？
3. 在修复 action scale 后，推荐的 CEM 配置是什么？

---

### 问题 3：Dynamics Rollout 精度

**离线评估结果：**
```
flat:           10-step pos_rmse = 0.0031m (3.1mm)
object_centric: 10-step pos_rmse = 0.0020m (2.0mm)
causality_aware: 10-step pos_rmse = 0.0006m (0.6mm)  ← 最好
```

**闭环结果：**
```
所有 9 个实验 (3 models × 3 templates): best_dist = final_dist
含义: CEM 在整个 horizon 内找不到任何能减少距离的动作序列
```

**问题：**
1. 即使 causality_aware 离线最好（0.6mm），闭环仍然失败，为什么？
2. Action scale mismatch 是否是唯一原因？
3. 还有其他潜在问题吗？

---

### 问题 4：端到端训练 vs 分阶段训练

**当前方案：** 分阶段训练
1. 用 MPPI 收集数据
2. 训练 learned rollout model（监督学习）
3. 用 learned model 替换 oracle model 进行规划

**潜在问题：**
- 训练数据分布 vs 规划时的分布不匹配
- 模型在训练时从未见过自己预测的累积误差

**问题：**
1. 是否应该考虑端到端训练（用 MPC cost 作为 reward）？
2. 或者 DAgger-style 在线微调？
3. 当前阶段，哪个方向 ROI 最高？

---

## 附录：核心文件

### 1. 模型定义
- `src/models/rig_world.py` — 三模型统一 wrapper
- `src/models/encoders.py` — Flat/ObjectCentric/CausalityAware encoder
- `src/models/heads.py` — DynamicsHead + SubgoalHead

### 2. 训练
- `scripts/train_high_level.py` — 训练脚本
- `src/models/losses.py` — 损失函数
- `src/data/episode_loader.py` — 数据加载（确认 action 来源）

### 3. 规划器
- `src/planners/cem_mpc.py` — CEM 规划器实现

### 4. 闭环执行
- `scripts/run_learned_mujoco_closed_loop.py` — MuJoCo 闭环脚本

### 5. 环境
- `src/envs/mujoco_push_env.py` — MuJoCo 推块环境

### 6. 评估结果
- `runs/pilot_state16_mppi_stage2c/offline_eval/offline_eval_summary.json`
- `docs/learned_closed_loop_mujoco_smoke_report.md`

### 7. Normalizer
- `runs/pilot_state16_mppi_stage2c/train_*/normalizer_state16.json`

### 8. 训练配置
- `configs/train/rig_world_shared.yaml`
- `configs/train/flat_high_level.yaml`
- `configs/train/object_centric_noncausal.yaml`
- `configs/train/causality_aware.yaml`

---

## 请求

1. 请分析上述问题，给出诊断和建议
2. 特别是 action scale mismatch 的修复方案
3. 如果需要更多信息，请告知

---

**附注：** 所有核心文件已复制到 `chatgpt_analysis/` 目录，可以直接查看。
