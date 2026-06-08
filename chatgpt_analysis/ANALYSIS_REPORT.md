# Paper1 三模型闭环失败核心原因分析

**日期:** 2026-05-24  
**状态:** 初步分析完成，等待进一步诊断

---

## 一、Executive Summary

三个 learned rollout 模型（flat, object_centric, causality_aware）在 MuJoCo 闭环推块实验中全部失败，**核心原因是 Action Scale Mismatch + 规划精度不足**。模型在离线评估中表现尚可（RMSE ~1.5cm），但在闭环规划中无法找到有效动作序列。

---

## 二、核心失败原因分析

### 🔴 原因 1：Action Scale Mismatch（最可能，概率 70%）— 已确认

**问题描述：**
- 训练数据中的 action 是物理动作 `actions_physical`，范围约 [-0.3, 0.3] m/s
- CEM 规划器搜索空间是归一化动作 [-1, 1]
- 模型在训练时学习的是物理尺度的动作效果
- 规划时输入归一化动作，导致模型收到 OOD（分布外）输入

**铁证：**
```bash
# 训练数据中的 action 范围：
3833f81a-c8c.npz: shape=(772, 2), min=-0.3000, max=0.3000
0d7b2d8c-0a4.npz: shape=(1000, 2), min=-0.2000, max=0.2000
5c8b4575-5fe.npz: shape=(225, 2), min=-0.5000, max=0.5000

# CEM 规划器搜索空间：
cem = CEMMPC(
    action_low=-1.0,  # 归一化空间
    action_high=1.0,  # 归一化空间
)

# 差距：3x ~ 5x 的 scale mismatch！
```

**关键矛盾：**
- 训练时：模型看到的是 `action ∈ [-0.3, 0.3]`（物理单位）
- 规划时：CEM 采样 `action ∈ [-1, 1]`（归一化单位）
- 模型从未见过 scale=3.3x 的动作输入

---

### 🟠 原因 2：规划精度不足（概率 60%）

**问题描述：**
- CEM 配置过于保守：128 samples × 3 iterations × 16 elites
- Horizon=10 步太短，难以完成推块任务
- 初始 std=0.5 太大，采样过于分散

**当前配置 vs MPPI 最佳参数：**

| 参数 | CEM (当前) | MPPI (最佳) | 差距 |
|------|-----------|-------------|------|
| samples | 128 | 2048 | 16x |
| horizon | 10 | 100 | 10x |
| iterations | 3 | N/A (MPPI) | - |

---

### 🟡 原因 3：Dynamics Rollout 精度不足（概率 50%）

**问题描述：**
- 一步预测 RMSE ~1.5cm 看似不错
- 但 10 步 rollout 累积误差可达 ~5cm
- 物体目标精度要求 <2cm，被累积误差淹没

**离线评估数据：**
```
flat:           10-step pos_rmse = 0.0031m (3.1mm)
object_centric: 10-step pos_rmse = 0.0020m (2.0mm)  
causality_aware: 10-step pos_rmse = 0.0006m (0.6mm) ← 最好
```

**闭环失败但 causality_aware 离线最好：**
- Causality_aware 在离线 10-step rollout 中 RMSE 仅 0.6mm
- 但在闭环中仍然失败 → 说明不是唯一瓶颈
- Action scale mismatch 可能是更主要的原因

---

### 🟢 原因 4：Normalizer 不匹配（概率 20%）

**问题描述：**
- Normalizer 是在训练数据上拟合的
- V2 特征可能有不同的分布
- 但这个问题相对容易验证和修复

---

## 三、关键发现

### 1. best_dist = final_dist（所有 9 个实验）

**含义：** CEM 规划器在整个 MPC horizon 内找不到任何能减少距离的动作序列。

**可能解释：**
- Action scale mismatch 导致模型输出垃圾预测
- CEM 采样空间太大，128 个样本不足以找到好动作
- 或者两者叠加

### 2. Causality_aware 离线最好但闭环同样失败

**含义：** 离线 rollout 精度不是唯一瓶颈。

**推论：** 即使 causality_aware 有最好的 rollout 精度，如果 action scale 错了，规划仍然会失败。

---

## 四、诊断计划

### Step 1: 验证 Action Scale Mismatch

```python
# 检查训练数据中的 action 范围
# 检查模型在归一化 action 下的预测质量
```

### Step 2: 修复 Action Scale

**方案 A：** CEM 在物理空间 [-0.3, 0.3] 搜索
**方案 B：** 重新训练模型，使用归一化 action [-1, 1]

### Step 3: 增加 CEM 配置

```python
cem = CEMMPC(
    horizon=20,           # 10 → 20
    num_samples=512,      # 128 → 512
    num_elites=32,        # 16 → 32
    num_iterations=5,     # 3 → 5
    init_std=0.3,         # 0.5 → 0.3
)
```

### Step 4: 重新运行闭环实验

---

## 五、核心文件清单

### 模型定义
1. `src/models/rig_world.py` — 三模型统一 wrapper
2. `src/models/encoders.py` — Flat/ObjectCentric/CausalityAware encoder
3. `src/models/heads.py` — DynamicsHead + SubgoalHead

### 训练
4. `scripts/train_high_level.py` — 训练脚本
5. `src/models/losses.py` — 损失函数

### 规划器
6. `src/planners/cem_mpc.py` — CEM 规划器实现

### 闭环执行
7. `scripts/run_learned_mujoco_closed_loop.py` — MuJoCo 闭环脚本

### 环境
8. `src/envs/mujoco_push_env.py` — MuJoCo 推块环境

### 评估结果
9. `runs/pilot_state16_mppi_stage2c/offline_eval/offline_eval_summary.json` — 离线评估
10. `docs/learned_closed_loop_mujoco_smoke_report.md` — 闭环报告

### Normalizer
11. `runs/pilot_state16_mppi_stage2c/train_flat/normalizer_state16.json`
12. `runs/pilot_state16_mppi_stage2c/train_object_centric/normalizer_state16.json`
13. `runs/pilot_state16_mppi_stage2c/train_causality_aware/normalizer_state16.json`

---

## 六、给 ChatGPT 的问题清单

1. **Action Scale 验证：** 如何确认训练数据中的 action 范围？
2. **修复方案选择：** 修改 CEM 空间 vs 重新训练模型？
3. **CEM 配置优化：** 在给定计算预算下，如何平衡 samples/horizon/iterations？
4. **Rollout 精度 vs 规划精度：** 哪个更值得优先提升？
5. **端到端训练：** 是否应该考虑用 MPC 的 cost 作为 reward 进行 fine-tune？

---

## 七、结论

**最可能的核心原因：Action Scale Mismatch (70%)**

模型在训练时学习的是物理尺度 action [-0.3, 0.3]，但 CEM 规划器在归一化空间 [-1, 1] 搜索。这导致模型在规划时收到完全 OOD 的输入，输出垃圾预测。

**推荐修复顺序：**
1. 先验证 action scale mismatch 假说
2. 修改 CEM 在物理空间搜索（最快修复）
3. 增加 CEM 配置
4. 如果仍有问题，再考虑重新训练

---

**下一步：** 等待 ChatGPT 分析反馈，然后执行诊断和修复。
