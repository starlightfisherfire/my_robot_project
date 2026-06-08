# ChatGPT Analysis Package

**项目:** Paper1 - Causal Representation for Push Manipulation  
**日期:** 2026-05-24  
**目的:** 诊断三模型闭环失败原因

---

## 目录结构

```
chatgpt_analysis/
├── README.md                    # 本文件
├── ANALYSIS_REPORT.md           # 详细分析报告
├── QUESTIONS_FOR_CHATGPT.md     # 给 ChatGPT 的问题清单
│
├── core_files/                  # 核心代码文件
│   ├── rig_world.py            # 三模型统一 wrapper
│   ├── encoders.py             # Flat/ObjectCentric/CausalityAware encoder
│   ├── heads.py                # DynamicsHead + SubgoalHead
│   ├── losses.py               # 损失函数
│   ├── cem_mpc.py              # CEM 规划器
│   ├── run_learned_mujoco_closed_loop.py  # 闭环脚本
│   ├── mujoco_push_env.py      # MuJoCo 环境
│   └── train_high_level.py     # 训练脚本
│
├── configs/                     # 训练配置
│   ├── rig_world_shared.yaml   # 共享配置
│   ├── flat_high_level.yaml    # Flat 模型配置
│   ├── object_centric_noncausal.yaml  # Object-centric 配置
│   └── causality_aware.yaml    # Causality-aware 配置
│
└── results/                     # 评估结果
    ├── offline_eval_summary.json        # 离线评估结果
    ├── learned_closed_loop_mujoco_smoke_report.md  # 闭环报告
    ├── normalizer_flat.json             # Flat normalizer
    ├── normalizer_object_centric.json   # Object-centric normalizer
    └── normalizer_causality_aware.json  # Causality-aware normalizer
```

---

## 快速开始

1. **先读 ANALYSIS_REPORT.md** — 了解问题全貌
2. **再读 QUESTIONS_FOR_CHATGPT.md** — 具体问题清单
3. **查看 core_files/** — 理解代码实现
4. **查看 results/** — 查看评估数据

---

## 核心发现

### ✅ 已确认的问题

**Action Scale Mismatch (概率 70%)**

```bash
# 训练数据 action 范围:
[-0.3, 0.3] m/s (物理单位)

# CEM 规划器搜索空间:
[-1.0, 1.0] (归一化单位)

# 差距: 3x scale mismatch！
```

### 🔄 待诊断的问题

1. CEM 配置是否足够？
2. Dynamics rollout 精度是否足够？
3. 是否需要端到端训练？

---

## 关键数据

### 离线评估 (10-step rollout RMSE)

| 模型 | pos_rmse | theta_rmse |
|------|----------|------------|
| flat | 3.1mm | 7.3° |
| object_centric | 2.0mm | 4.2° |
| causality_aware | 0.6mm | 2.2° |

### 闭环结果

```
所有 9 个实验: best_dist = final_dist
含义: CEM 找不到任何能减少距离的动作序列
```

---

## 联系方式

如有问题，请通过 Telegram 联系 Paper1 助手。

---

**最后更新:** 2026-05-24 10:45 GMT+8
