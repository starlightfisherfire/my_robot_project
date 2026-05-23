# Bruce Wu — 机器人算法工程师

**📍 南京 | 📧 [your_email] | 🔗 github.com/[your_github] | 📱 [your_phone]**

---

## 🛠 核心技能

**Python · PyTorch · MuJoCo · 机器人仿真 · 模型预测控制(MPC) · 强化学习 · 因果推断 · CEM/MPPI · Linux · Git · 并行计算 · 数据分析**

---

## 💼 项目经历

### 因果感知世界模型 — 机器人操作泛化
*独立研究 | 2025 – 至今*

- 提出 CausalityAwareEncoder，将潜空间分解为场景结构、运动、可行性、无关因子四个因果维度，提升世界模型在未见环境下的泛化能力
- 设计统一架构支持 Flat / Object-Centric / Causality-Aware 三种编码器对比实验
- 在 MuJoCo push-T 任务上完成 **300+ 配置 × 6 种障碍物布局** 的系统性扫参，识别最优 MPC 配置组合
- 最优配置在 blocking_hard 任务上达到 **49 秒完成 / 毫米级精度**
- 设计单障碍物→双障碍物的 **Layout OOD 泛化实验**，验证潜空间组合泛化能力
- 技术栈：PyTorch, MuJoCo, CEM/MPPI MPC, 32 核并行轨迹采样

### 视觉世界模型与执行器接口
*独立研究 | 2025 – 至今*

- 探索从视觉观测构建物体级世界模型的方法
- 设计世界模型输出与机器人执行器之间的标准化接口，支持跨硬件平台复用

### 强化学习中的风险偏好涌现
*独立研究 | 2025*

- 研究 RL agent 在经济学风险决策场景中风险偏好的涌现机制

---

## 🎓 教育背景

**[大学名称]** — 硕士/博士/本科 · [专业]  
*20XX – 20XX*

---

## 📝 论文

- **"Causality-Aware Object-Level Representations Improve Structural OOD Generalization in World Models"** (进行中，第一作者)
- **"From Single to Double: Compositional Layout Generalization via Factorized Latent World Models"** (计划中，第一作者)

---

## 🏆 自我评价

- 独立完成从问题定义、实验设计到代码实现的全流程研究
- 管理 8 个 AI 助手协作并行推进 3 个研究项目
- 擅长从大规模实验数据中提炼结构性洞察（300+ 配置 sweep 中识别最优参数组合和 MPPI 温度敏感性）
- 快速搭建并行仿真实验框架，工程效率高

---

*最后更新: 2026-05-17*
