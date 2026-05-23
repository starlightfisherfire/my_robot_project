# Bruce Wu - 机器人世界模型 / 因果推理 研究方向

**📍 南京 | 📧 [your_email] | 🔗 [your_github] | 📱 [your_phone]**

---

## 👨‍🎓 教育背景

**[你的大学]** - 硕士/博士/本科,[专业]
*20XX - 20XX*
研究方向:机器人操作、世界模型、因果表征学习

---

## 🔬 研究项目

### 1. 因果感知的表征学习用于机器人操作 OOD 泛化 (Paper 1)
*独立研究 | 2025 - 至今 | IEEE RA-L / CoRL 投稿准备中*

**根本出发点:**
当前机器人操控系统在分布外(OOD)场景下泛化能力不足。主流方法依赖大规模数据 + VLA/LLM 端到端训练,无法解释世界知识如何被编码、状态如何表征、动作如何作用于状态变化。我们主张:世界状态是可压缩的,动作可以指导状态变化,但压缩方式有好坏之分--按因果机制分解的表征优于无结构压缩。

**本文假设:**
在不依赖语言模型的结构化推放任务中,按因果机制分解的物体级表征(Causality-Aware)比按物体分解(Object-Centric)或无结构压缩(Flat)在 Layout OOD 和 Shape OOD 上实现更小的泛化差距。

**方法:**
- 提出 **CausalityAwareEncoder**:将潜空间分解为 `z_stable`(不变结构)、`z_dynamics`(运动状态)、`z_affordance`(可行性)、`z_nuisance`(无关因子) 四个因果因子
- 设计 **RIGWorldModel** 统一架构:支持 Flat / Object-Centric / Causality-Aware 三种编码器变体,在相同规划器和数据协议下公平对比
- 联合训练 DynamicsHead(前向预测)和 SubgoalHead(子目标预测),使潜空间同时支撑动力学推演和规划引导

**实验进度:**

| 关卡 | 内容 | 状态 |
|------|------|------|
| Gate 0-3 | Repo 骨架、Reset Template、Toy Oracle-MPC | ✅ |
| Gate 4-6 | MuJoCo Env 脚手架、Oracle Rollout、MPC 接口 | ✅ |
| **Gate 7** | **Oracle-MPC 任务能力验证** | ✅ 毫米级精度 |
| Gate 8 | Obstacle 布局容量(blocking/narrow/edge) | 🔄 |
| Gate 9-12 | 数据收集 → 模型训练 → OOD 对比 | ⬜ |

**关键发现:**
- 300+ 配置 × 6 种障碍物布局的系统性 sweep,发现 speed-budget 相变现象
- CEM-MPC + Oracle dynamics 在 blocking_hard 上达到 83% 成功率
- 物理可行性边界:passage 宽度 < pusher+object 截面时,任何 planner 均不可解

**技术栈:** Python, PyTorch, MuJoCo, CEM/MPPI MPC, 并行轨迹采样 (32-core)

---

### 2. C-JEPA 世界模型的因果解耦验证 (Paper 2)
*独立研究 | 2025 - 至今*

**根本出发点:**
JEPA(Joint Embedding Predictive Architecture)类架构声称通过 latent space 的 masked prediction 能自动学到因果解耦的表征。但"声称学到"和"真的学到"之间缺少验证--如果表征真的因果解耦了,那么在 nuisance 变化的 OOD 场景下应该保持泛化能力。

**本文假设:**
C-JEPA 在视觉推放任务上学到的潜空间表征如果实现了因果解耦,应在 Layout OOD 上显著优于非因果解耦的 baseline;反之则说明 JEPA 范式的架构本身存在缺陷。

**实验进度:** 🔄 架构设计中

---

### 3. 世界模型表征的执行器消费与真机迁移 (Paper 3)
*独立研究 | 2025 - 至今*

**根本出发点:**
世界模型学到的表征最终需要被执行器消费。如果表征无法被 MPC/RL 等规划器有效使用,或者无法迁移到真机,那么表征学习的价值就无法闭环。

**本文假设:**
JEPA 类表征可以直接接入 MPC/RL 执行器,并通过 sim-to-real 迁移到真实机器人上。

**实验进度:** ⬜ 待 Paper 1/2 结果确定后启动

---

### 4. 强化学习中的风险偏好涌现
*独立研究 | 2025*

**根本出发点:**
RL agent 在经济学风险决策场景中是否能涌现出与人类相似的风险偏好?如果能,说明 reward shaping 可以诱导出复杂的经济行为。

**本文假设:**
在特定 reward 结构下,RL agent 会自发涌现出风险厌恶或风险偏好行为。

**实验进度:** ✅ 完成

---

## 🛠 技术能力

| 类别 | 技能 |
|------|------|
| 语言 | Python (主力), Bash, LaTeX |
| 深度学习 | PyTorch, 自编码器, 因果表征学习, 世界模型 |
| 机器人仿真 | MuJoCo, 刚体动力学, 碰撞检测 |
| 规划与控制 | CEM-MPC, MPPI, 并行轨迹优化, 模型预测控制 |
| 工具链 | Git, Linux, Conda, 多核并行计算, Weights & Biases |
| 数据分析 | NumPy, Pandas, 大规模 sweep 分析与可视化 |

---

## 📝 论文 / 进行中的工作

1. **"Causality-Aware Object-Level Representations Improve Structural OOD Generalization in World Models"**
   *进行中 | 第一作者 | IEEE RA-L / CoRL*
   根本出发点:世界状态是可压缩的,但压缩方式决定泛化能力。
   Claim:因果机制分解的表征在 Layout OOD 上优于物体分解和无结构压缩。
   进度:Oracle-MPC 能力验证通过,obstacle 布局容量测试中。

2. **"C-JEPA 的因果解耦假设:OOD 验证"**
   *进行中 | 第一作者*
   根本出发点:JEPA 声称学到因果解耦,需要 OOD 任务来验证。
   Claim:如果 C-JEPA 真的因果解耦,应在 Layout OOD 上显著优于非因果 baseline。
   进度:架构设计中。

3. **"世界模型表征的执行器消费与真机迁移"**
   *计划中 | 第一作者*
   根本出发点:表征必须能被执行器消费并迁移到真机才有实际价值。
   Claim:JEPA 表征可直接接入 MPC/RL 并通过 sim-to-real 迁移。
   进度:待 Paper 1/2 结果。

---

## 🏆 研究视角与核心信念

**根本问题：世界知识存储在哪里？**
VLA 将感知、理解、动作揉为一体，无法定位知识、无法验证理解。本工作的核心主张：世界知识可以脱离动作而存在——表征存储"世界是什么样"，DynamicsHead 存储"世界会怎样变"，消费器负责"该怎么做"。三者可分离、可各自验证、可独立升级。

**why PushT？**
在最简化的设定下，先把"知识如何编码"这个问题彻底讲清楚。不需要 LLM、不需要多模态、不需要复杂视觉 pipeline——如果最简设定走不通，更复杂的系统只是在用算力掩盖结构缺陷；如果走通了，就为更大的系统提供了可验证的结构蓝图。

**结构化状态为什么还不够？**
结构化状态本身是语义上终极解耦的——每个维度含义明确。但它"太干净"了：维度和维度之间的关系不在表征里。Flat encoder 不知道"推杆向左加速"和"方块向右滑动"是因果相关的——它必须自己从数据中发现。OC/Causal encoder 做的就是把这些已知的语义结构翻译成模型可消费的架构先验。这不是压缩信息，是结构注入。

**自包含表征：从字典到句子**
语言是"自包含"的——一个 token 同时编码语义、语用、交互期待。LLM 的多层表征空间就是这种自包含的实现。世界状态的表征也应该如此：不是孤立快照，而是经过时序混合（历史帧）和关系混合（物体间 attention）之后的自包含片段。它编码的不仅是"现在是什么"，还有"正在发生什么趋势"和"动作作用后的预期变化"。

**消费器是测试点，不是终极答案**
世界模型本身不产生动作——它需要被消费。消费器（CEM-MPC / MPPI / RL+MPPI）的设计和表征结构同等重要。不同消费器可能给出不同的"哪种 encoder 好"的结论——这不是缺陷，是更诚实、更可辩护的实验框架。RL proposal + MPPI 精搜是对"表征优势能否在消费侧放大"的压力测试。

**LLM 是物理世界表征的特例**
LLM 的知识存储在 transformer block 的多层表征空间中——知识就是 attention 和 MLP 对表征的变换方式。世界模型是同一范式：知识存储在 transition function 对 latent state 的变换方式中。LLM 的离散 token 连续 embedding 空间是物理世界连续 latent 空间的一个特例。两者的训练逻辑也相通：层级不应手工设计，而应由预测目标诱导涌现。

**正例反例都是收获**
研究不是为了凑 positive result。知道自己在证明什么，然后让实验告诉你：假设成立、假设被推翻、还是验证不足——三种结果都指向明确的下一步。

---

## 📂 项目链接

- GitHub: [your_github]
- 个人主页: [your_website]

---

*最后更新: 2026-05-18*
