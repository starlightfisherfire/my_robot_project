# Git Asset Manifest

> Generated: 2026-05-24 02:09 CST
> Project: my_robot_project (Paper 1 — MuJoCo Push + Learned Rollout)

## 已提交到 Git 的资产

### 代码 (`src/`)
- `src/data/` — episode loader, writer, template generator
- `src/envs/` — MuJoCo push environment
- `src/interventions/` — layout families, sampling rules
- `src/metrics/` — oracle capacity metrics
- `src/planners/` — MPPI, CEM-MPC, cost functions, rollout model, obstacle utils

### 脚本 (`scripts/`)
- 数据收集、模板生成、sweep 脚本、分析脚本、可视化脚本
- 共 60+ 脚本文件

### 配置 (`configs/`)
- `configs/train/` — 训练配置 (causality_aware, flat, object_centric, rig_world_shared)
- `configs/experiments/` — 实验配置
- `configs/splits/` — 数据分割配置
- `configs/state_schema/` — 状态 schema 定义

### 文档 (`docs/`)
- 实验报告、设计文档、审计报告
- `docs/state_schemas/` — 状态 schema 设计文档
- `docs/papers/` — 论文大纲

### 设计 (`design/`)
- 架构演化记录、FTWM/SCWM/ACWM 架构图
- Mermaid 源文件 + PNG 渲染

### 论文 (`papers/`)
- `agentic_wm_position_paper.md`
- `world_model_position_paper_v2.md`

### 实验记录 (`experiments/`)
- `tool_interface_mvp/` — Tool Interface MVP 实验
- `exp_20260517_layout_ood_state16_v0/` — Layout OOD 实验

### 元数据 (`data/sim/metadata/`)
- 所有 reset template JSON 文件
- 模板 inventory CSV
- 模板 provenance 文档

### 归档 (`archive/`)
- 历史迁移记录

## 未提交（本地保留，.gitignore 排除）

### 数据资产 (`data/sim/`)
| 目录 | 大小 | 说明 |
|------|------|------|
| `mppi_stage2c_state16/` | 164M | Stage2c state16 数据 |
| `mppi_stage2c/` | 15M | Stage2c sweep 数据 |
| `layout_ood_state16_v0/` | 1.4M | Layout OOD v0 |
| `mppi_stage2a/` | 8.6M | Stage2a sweep |
| `mppi_stage2b_speed/` | 5.2M | Stage2b speed sweep |
| `episodes/` | - | 原始 episode 数据 |

### 运行结果 (`runs/`)
| 目录 | 说明 |
|------|------|
| `mppi_stage2c_20260520_194856/` | Stage2c 主 sweep (240 logs) |
| `train_state16_poc/` | State16 POC 训练 |
| `pilot_state16_mppi_stage2c/` | Pilot 训练 |
| 各种 sweep 目录 | 参数搜索结果 |

### Checkpoints (`runs/*/checkpoints/`)
| 文件 | 大小 |
|------|------|
| `train_state16_poc/causality_aware/checkpoints/best.pt` | 15M |
| `train_state16_poc/object_centric/checkpoints/best.pt` | 10M |
| `train_state16_poc/flat/checkpoints/best.pt` | 7M |
| `pilot_state16_mppi_stage2c/*/checkpoints/best.pt` | 7-15M |

### 敏感文件（已排除）
| 文件 | 内容 | 状态 |
|------|------|------|
| `.env` | DEEPSEEK_API_KEY | .gitignore 排除 ✅ |
| `.aider.chat.history.md` | Aider 历史 | .gitignore 排除 ✅ |

## 推送状态

- 本地 commit: `46cd965` ✅
- GitHub remote: **未配置** — 需要 GitHub token 或 SSH key
