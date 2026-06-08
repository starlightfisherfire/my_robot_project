# Paper1 V2 Final Analysis Package

**日期:** 2026-05-25  
**目的:** 完整的 V2 闭环修正实验 + Topology/Geometry 诊断

---

## 文件清单

### 核心模块（新建）
| 文件 | 说明 |
|------|------|
| `action_conventions.py` | Action convention helper（a_norm ↔ a_phys ↔ disp） |
| `learned_planner_adapter.py` | CEM learned rollout planner adapter |
| `topology_geometry.py` | Topology/geometry metrics 模块 |

### 脚本（新建）
| 文件 | 说明 |
|------|------|
| `self_check_topology_geometry.py` | Topology metrics self-check |
| `audit_template_topology_geometry.py` | 批量审计 160 个 templates |
| `run_phase5.py` | Open-space 闭环实验（含 topology metrics） |
| `run_phase6.py` | Classic templates 闭环实验（含 topology metrics） |

### 配置（新建）
| 文件 | 说明 |
|------|------|
| `closed_loop_smoke_templates.yaml` | Template matrix（4 families × 3） |

### 文档（新建/更新）
| 文件 | 说明 |
|------|------|
| `action_convention_contract.md` | Action convention 审计 |
| `planner_best_config_contract.md` | Planner 最佳配置 |
| `topology_geometry_protocol.md` | Topology/geometry 协议 |
| `closed_loop_action_planner_fix_report.md` | **完整报告**（含 topology 分析） |

### 结果（新建）
| 文件 | 说明 |
|------|------|
| `template_topology_geometry.csv` | 160 个 templates 的 geometry metrics |
| `by_family_summary.csv` | 按 family 聚合的 geometry 统计 |
| `recommended_closed_loop_template_pool.yaml` | 推荐的闭环 template pool |
| `open_space_summary.json` | Open-space 闭环结果 |
| `classic_templates_summary.json` | Classic templates 闭环结果 |
| `topology_geometry_self_check.json` | Self-check 结果 |

---

## 核心发现

### 1. Action Convention（已修复）
```
训练: actions_physical ∈ [-0.5, 0.5] m/s
Planner: a_norm ∈ [-1, 1]
转换: a_phys = a_norm * max_speed_mps (0.75)
Env: env.step(a_norm) → velocity_cmd = a_norm * max_speed_mps
EE update: ee_xy += a_phys * control_dt
```

### 2. 闭环结果
| 模型 | Open-space Improved | Classic Improved | Best Single |
|------|--------------------|--------------------|-------------|
| flat | 2/3 | 8/9 | 0.174m |
| object_centric | 2/3 | 2/9 | 0.118m |
| **causality_aware** | 2/3 | 3/9 | **0.094m** |

### 3. Topology/Geometry
- 所有 family 的 blocking_score = 0.000（obstacles 不在 object-goal 路径上）
- "blocking" 的难度来自 EE 接近路径被阻挡
- edge_goal 的 goal_edge_distance 最小（0.119）

### 4. 核心瓶颈
- Learned model 无法引导 EE 接近物体
- 当 EE 远离物体时，所有 action 的 cost 几乎相同
- 需要改进 cost function 或两阶段规划

---

## 给 ChatGPT 的问题

1. **如何改进 learned model 的 approach 能力？**
2. **blocking_score = 0 是否说明指标设计有问题？**
3. **CEM vs MPPI for learned rollout，哪个更有前景？**
4. **Action token 架构是否值得深入探索？**
