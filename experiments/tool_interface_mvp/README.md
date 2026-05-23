# Tool Interface MVP 实验

> 验证 Agentic WM 的 Tool Interface 概念：一个 WM 面对不同场景时，能自适应选择最优的动作执行器。

## 实验假设

WM + Tool Selector > 单一 Action Expert (MLP 或 Diffusion)

## 实验设置

- **环境**: MuJoCo Push-T (2D 平面推物体)
- **Task**: 将 T 形物体推到随机目标位置
- **两种模式**:
  - 简单场景：物体离目标远，需要快速大范围移动 → 适合 MLP
  - 复杂场景：物体接近目标，需要精确对准 → 适合 Diffusion

## 三个 Baseline

| Baseline | 说明 |
|----------|------|
| MLP only | 纯 MLP 动作预测器，快速但单模 |
| Diffusion only | Diffusion Policy，精确但慢 |
| Tool Interface | MLP + Diffusion + Selector，自适应选择 |

## 文件结构

```
tool_interface_mvp/
├── README.md           # 本文档
├── config.yaml         # 配置文件
├── models.py           # 模型定义
├── envs.py             # MuJoCo Push-T 环境
├── data.py             # 数据生成与加载
├── train_baselines.py  # 训练 MLP/Diffusion baseline
├── train_tool_interface.py  # 训练 Tool Interface
└── eval.py             # 评估与可视化
```

## 快速开始

```bash
# 1. 生成数据
python data.py --mode generate --n_episodes 2000

# 2. 训练 baseline
python train_baselines.py --tool mlp --epochs 50
python train_baselines.py --tool diffusion --epochs 50

# 3. 训练 Tool Interface
python train_tool_interface.py --epochs 100

# 4. 评估对比
python eval.py --all
```
