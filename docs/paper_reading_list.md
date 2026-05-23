# Paper Reading List — 速读储备

> 每篇一句话核心观点，建立知识地图用。按优先级分组。

## 第一优先级：因果表征 + JEPA（直接相关）

| # | 论文 | 一句话 | 链接 |
|---|------|--------|------|
| 1 | iVAE | 用辅助变量（时间索引）识别因果潜变量 | https://arxiv.org/abs/1905.03401 |
| 2 | CausalWorld | 因果表征必须在 intervention 下验证，重建不够 | https://arxiv.org/abs/2005.04768 |
| 3 | CaRep | 因果表征 = 不变机制 + 变化上下文，invariance 是训练信号 | https://arxiv.org/abs/2310.02513 |
| 4 | CausalRep (ICLR 2024) | 通过 sparse mechanism shift 学因果表征 | https://arxiv.org/abs/2310.07214 |
| 5 | I-JEPA | latent space 掩码预测，不需像素重建就学到语义表征 | https://arxiv.org/abs/2301.08243 |
| 6 | V-JEPA | 视频版 I-JEPA，预测被遮挡区域的表征学时空结构 | https://arxiv.org/abs/2404.16367 |
| 7 | MC-JEPA | 在 JEPA 中引入 mechanism-aware factorized predictor | https://arxiv.org/abs/2402.13659 |
| 8 | A-JEPA | Audio JEPA，跨模态验证 JEPA 范式的通用性 | https://arxiv.org/abs/2409.18048 |

## 第二优先级：世界模型 + 规划

| # | 论文 | 一句话 | 链接 |
|---|------|--------|------|
| 9 | DreamerV1 | 最初的 latent dynamics + imagination 论文 | https://arxiv.org/abs/1912.01603 |
| 10 | DreamerV3 | 一个世界模型跨所有任务，latent dynamics 可泛化 | https://arxiv.org/abs/2301.04104 |
| 11 | TD-MPC2 | latent dynamics + MPC 在 100+ 任务 work，关键在接口设计 | https://arxiv.org/abs/2310.16828 |
| 12 | IRIS | 离散 token 世界模型做 planning，表征的离散/连续影响规划 | https://arxiv.org/abs/2211.06566 |
| 13 | TD-MPC | temporal difference learning + MPC 的结合 | https://arxiv.org/abs/2203.04955 |

## 第三优先级：OOD 泛化

| # | 论文 | 一句话 | 链接 |
|---|------|--------|------|
| 14 | COLA | 结构化 OOD 需要 controlled intervention，随机 augmentation 不够 | https://arxiv.org/abs/2310.04832 |
| 15 | GenSim | 自动生成多样化仿真环境提升泛化，但评估协议是瓶颈 | https://arxiv.org/abs/2310.01439 |
| 16 | OOD Survey | 机器人 OOD 泛化综述 | https://arxiv.org/abs/2403.01266 |

## 第四优先级：VLA / LLM + 机器人

| # | 论文 | 一句话 | 链接 |
|---|------|--------|------|
| 17 | RT-2 | VLM 直接输出动作 token，语言和动作在同一表征空间 | https://arxiv.org/abs/2307.15818 |
| 18 | OpenVLA | 开源 VLA，但 OOD 泛化有限 | https://arxiv.org/abs/2406.09246 |
| 19 | SayCan | LLM 高层规划 + 低层策略执行，接口是 hand-designed | https://arxiv.org/abs/2204.01691 |
| 20 | π0 | VLA + flow matching action head，diffusion 风格动作生成 | https://arxiv.org/abs/2410.24164 |
| 21 | Code as Policy | LLM 生成代码作为策略 | https://arxiv.org/abs/2209.07753 |

## 第五优先级：RL + 搜索 + 物理理解

| # | 论文 | 一句话 | 链接 |
|---|------|--------|------|
| 22 | AlphaZero | value network 粗评估 + MCTS 精搜 | https://arxiv.org/abs/1712.01815 |
| 23 | MuZero | 学习 world model + MCTS，不在原始状态空间搜索 | https://arxiv.org/abs/1911.08265 |
| 24 | Physion | 用物理推理评测物体表征质量 | https://arxiv.org/abs/2112.01473 |
| 25 | PHYRE | 物理推理 benchmark，需要理解因果 | https://arxiv.org/abs/1908.05656 |
| 26 | Structured WMs | 结构化世界模型综述 | https://arxiv.org/abs/2310.10083 |

---

*最后更新: 2026-05-18*
