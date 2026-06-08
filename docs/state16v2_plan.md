# State16 v2 方案：几何关系表征 + 反事实训练

> 日期: 2026-05-28
> 原则: 不改原文件，全部新增

---

## 1. 目标

让 learned world model 具备可规划的 action-conditioned dynamics：
- 状态表征编码几何关系（而非速度）
- 训练数据包含反事实覆盖（同一状态，多种动作）
- 训练损失包含对比项（区分不同动作的后果）

## 2. State16 v2 设计

保持 16 维，替换内容：

| 索引 | v1 (当前) | v2 (新) | 理由 |
|:---:|-----------|---------|------|
| 0 | x | x | 保留 |
| 1 | y | y | 保留 |
| 2 | sin(θ) | sin(θ) | 保留 |
| 3 | cos(θ) | cos(θ) | 保留 |
| 4 | **vx** | **dist_to_ee** | 去速度，加几何 |
| 5 | **vy** | **dist_to_goal** | 去速度，加几何 |
| 6 | **ω** | **angle_to_goal** | 去速度，加几何 |
| 7 | width | width | 保留 |
| 8 | valid | valid | 保留 |
| 9 | (备用) | **rel_x_to_ee** | 新增 |
| 10 | (备用) | **rel_y_to_ee** | 新增 |
| 11 | (备用) | **rel_x_to_goal** | 新增 |
| 12 | mass | **rel_y_to_goal** | 移位 |
| 13 | friction | mass | 移位 |
| 14 | (备用) | friction | 移位 |
| 15 | valid | valid | 保留 |

## 3. 新增文件清单

### 数据采集
- `scripts/collect_state16v2.py` — 新的状态提取器
- `configs/experiments/state16v2_collection.yaml` — 采集配置

### 数据加载
- `src/data/state16v2_normalizer.py` — 新的 normalizer
- `src/data/state16v2_dataset.py` — 新的 dataset

### 模型训练
- `scripts/train_state16v2.py` — 新的训练脚本（含对比损失）

### 评估
- `scripts/eval_state16v2.py` — 新的评估脚本

### 备份
- `backup/state16v1/` — 关键原文件的备份

## 4. 训练数据要求

### 当前数据的问题
- 每条轨迹只用最优动作
- 同一状态只见过一种动作
- 模型无法学到"不同动作 → 不同结果"

### 反事实数据采集方案

**方案 A: 随机扰动采集**
```
对每个时间步 t:
  1. 执行最优动作 a* → 记录 s_{t+1}
  2. 随机采样 2-3 个扰动动作 a' → 记录 s'_{t+1}
  3. 存储: (s_t, a*, s_{t+1}), (s_t, a', s'_{t+1})
```

**方案 B: 网格覆盖**
```
在动作空间 [-1,1]×[-1,1] 上均匀采样:
  8 个方向 × 3 个力度 = 24 种动作
  每个状态尝试 4-6 种 → 足够的反事实覆盖
```

**方案 C: 混合策略（推荐）**
```
70% 最优动作轨迹（保持任务完成能力）
30% 随机/扰动动作（增加反事实覆盖）
```

## 5. 训练损失设计

```python
# 当前损失
L_old = MSE(pred_delta, true_delta) + MSE(pred_subgoal, true_subgoal)

# 新损失（三项）
L_new = L_dynamics + λ_contrast * L_contrast + λ_multi * L_multistep

# 1. 单步预测损失（保持）
L_dynamics = MSE(pred_delta, true_delta)

# 2. 对比损失（新增）
# 同一状态，不同动作应导致不同预测
L_contrast = max(0, margin - ‖f(s,a1) - f(s,a2)‖)

# 3. 多步 rollout 损失（新增）
# 展开 K 步，每步误差累积
L_multistep = Σ_k w_k * MSE(ŝ_{t+k}, s_{t+k})
```

## 6. 实施顺序

1. **备份原文件** → `backup/state16v1/`
2. **写 state16v2 adapter** → `scripts/collect_state16v2.py`
3. **采集新数据** → `data/sim/state16v2/`
4. **写新 dataset/normalizer** → `src/data/state16v2_*.py`
5. **写新训练脚本** → `scripts/train_state16v2.py`
6. **训练 + 评估**

## 7. 风险评估

| 风险 | 等级 | 缓解 |
|------|:---:|------|
| 新数据采集耗时 | 中 | 复用现有 MuJoCo 环境，只改状态提取 |
| 模型不收敛 | 低 | 先用 flat 验证，再扩展到 oc/ca |
| 与现有代码冲突 | 低 | 全部新文件，不碰原文件 |
| checkpoint 不兼容 | 无 | 新数据 = 新 checkpoint，不复用旧的 |

---

**下一步**: 开始实施第 1-2 步（备份 + 写 adapter）
