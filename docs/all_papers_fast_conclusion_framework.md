# 全项目最速出结论框架

**Date:** 2026-05-24  
**Principle:** AI 辅助时代，结论先行，论文后写。最快验证核心假设。

---

## 总体策略

```
Paper 1 (因果表征) ─── 已有 pilot 结论 ─── 下周 formalize
Paper 2 (视觉 WM)  ─── 最小闭环验证 ─── 下周出初步结论
Paper 3 (执行器)   ─── Oracle baseline ─── 下周出初步结论
Paper 4 (LLM 接入) ─── 设计文档 ─── 下周确定方向
```

---

## Paper 1: 因果分解表征

**核心假设：** causality-aware representation 在 OOD 泛化上优于 flat 和 object-centric

**当前状态：** ✅ PILOT CONCLUSION READY

**已验证：**
- flat one-step 最好，但 multi-step rollout 最差
- causality_aware one-step 最差，但 multi-step rollout 最好 (3.6× 稳定)
- family_holdout 支持该 finding
- MuJoCo closed-loop smoke: 三模型都 ~0.25m，未达 success

**Preliminary Conclusion (可写入论文草稿)：**
> "One-step prediction accuracy does not fully predict long-horizon rollout stability. Causality-aware slot factorization provides structural regularization that reduces error accumulation over multi-step rollouts, despite worse single-step accuracy."

**下周任务：**
1. 正式数据采集 (MPPI best config, 每 family 100+ episodes)
2. 重训三模型 (更多 epochs, 更多数据)
3. MuJoCo closed-loop 正式 eval
4. 写 formal conclusion

---

## Paper 2: 视觉物体世界模型

**核心假设：** 从 RGB 学到的 visual slots 可以替代 structured state 用于 OOD 控制

**当前状态：** 🔴 框架 only，无代码运行

**最快出结论路径 (1 周)：**

### Step 1: 最小视觉数据 (1 天)
```bash
# 从 MuJoCo 渲染 top-down 视频
# 复用 Paper 1 的 mppi_stage2c 数据
# 每个 episode 渲染 RGB frames
```

### Step 2: 最小 slot encoder (1 天)
```bash
# 用现有 slot_encoder.py + patch_encoder.py
# 训练 slot encoder on rendered frames
# 输出: visual slots [B, N_slots, D_slot]
```

### Step 3: Slot → Paper 1 接口 (1 天)
```bash
# visual slots → flat encoder → dynamics head
# 复用 Paper 1 的 rollout_model.py
# 对比: visual slots vs structured state16
```

### Step 4: OOD eval (1 天)
```bash
# 同 Paper 1 的 OOD protocol
# 对比: visual slots vs state16 on layout OOD
```

### Step 5: 结论 (1 天)
- 如果 visual slots ≈ state16 → 支持假设
- 如果 visual slots << state16 → 需要更好的视觉编码
- 如果 visual slots > state16 → 意外发现，视觉信息更丰富

**Preliminary Conclusion Template：**
> "Visual object slots learned from RGB [can/cannot] replace structured object states for OOD manipulation control. The performance gap between visual slots and oracle structured states is [X]%, suggesting [interpretation]."

---

## Paper 3: 世界模型-执行器接口

**核心假设：** WM 的输出格式（接口）决定下游执行器的效果

**当前状态：** 🔴 框架 only

**最快出结论路径 (1 周)：**

### Step 1: Oracle subgoal baseline (1 天)
```bash
# 用 Paper 1 的 Oracle-MPC 生成 subgoal 序列
# 这是最简单的 "接口"：subgoal = next_object_pose
```

### Step 2: 最小执行器 (1 天)
```bash
# 实现 simplest executor:
# input: subgoal (next_object_pose)
# output: action (delta_ee)
# 方法: 简单的 PD controller to reach subgoal
```

### Step 3: 接口变体对比 (2 天)
```bash
# 变体 A: 只给 next_object_pose
# 变体 B: 给 next_object_pose + contact flag
# 变体 C: 给 next_object_pose + subgoal + affordance
# 对比: 哪种接口让执行器效果最好
```

### Step 4: OOD eval (1 天)
```bash
# 同 Paper 1 OOD protocol
# 对比: 不同接口在 OOD 下的表现
```

### Step 5: 结论 (1 天)
- 如果接口 B/C >> 接口 A → 支持假设
- 如果接口差异不大 → 接口不是瓶颈
- 如果 oracle subgoal 也做不好 → 执行器是瓶颈

**Preliminary Conclusion Template：**
> "WM output interface design [does/does not] significantly affect downstream executor performance. The best interface includes [X], achieving [Y]% improvement over the baseline interface."

---

## Paper 4: LLM 接入

**核心假设：** LLM 的语言先验可以加速因果分解的学习

**当前状态：** 🔴 未开始

**最快出结论路径 (1 周)：**

### Step 1: 设计文档 (1 天)
```bash
# 定义: LLM 如何提供因果先验
# 方案 A: LLM 生成因果图 → 正则化 encoder
# 方案 B: LLM 描述物体属性 → 条件输入
# 方案 C: LLM 生成任务指令 → 条件输入
```

### Step 2: 最小实验 (2 天)
```bash
# 方案 A 最简单:
# 用 LLM 生成 "T 形物体在 blocking 下应该绕行"
# 作为 encoder 的额外条件
# 对比: 有/无 LLM 先验的 OOD 表现
```

### Step 3: 结论 (1 天)
- 如果 LLM 先验显著提升 OOD → 支持假设
- 如果 LLM 先验无帮助 → 语言先验不够
- 如果 LLM 先验反而有害 → 先验不准确

**Preliminary Conclusion Template：**
> "LLM-derived causal priors [can/cannot] accelerate causal decomposition learning. The improvement from LLM priors is [X]%, suggesting [interpretation]."

---

## 总时间线

| 天数 | Paper 1 | Paper 2 | Paper 3 | Paper 4 |
|------|---------|---------|---------|---------|
| Day 1 | 正式数据采集 | 视觉数据渲染 | Oracle subgoal | 设计文档 |
| Day 2 | 重训模型 | Slot encoder | 最小执行器 | 最小实验 |
| Day 3 | Closed-loop eval | Paper 1 接口 | 接口对比 | 最小实验 |
| Day 4 | OOD eval | OOD eval | OOD eval | 结论 |
| Day 5 | Formal conclusion | 结论 | 结论 | - |
| Day 6 | 论文草稿 | 论文草稿 | 论文草稿 | 论文草稿 |
| Day 7 | Review | Review | Review | Review |

---

## 并行策略

所有项目可以并行推进，因为：
1. Paper 1 的数据/模型可被 Paper 2/3 复用
2. Paper 2 的 visual slots 可接 Paper 1 的接口
3. Paper 3 的执行器可接 Paper 1/2 的 WM 输出
4. Paper 4 的 LLM 先验可加到任何阶段

**关键：** 先出结论，后写论文。结论 = 最小实验 + 核心发现。
