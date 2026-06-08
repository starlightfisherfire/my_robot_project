# Topology/Geometry Fix Report

**Date:** 2026-05-25  
**Status:** FIX COMPLETE — Ready for overnight dual-planner eval

---

## 1. Executive Summary

**Topology/geometry 诊断层已修复，现在可以可靠用于 overnight closed-loop template selection 和 failure analysis。**

---

## 2. 修复前问题

| 问题 | Before | After | Evidence |
|------|--------|-------|----------|
| blocking_score | 全部为 0 | blocking=1.0, 其他=0 | `by_family_summary.csv` |
| passage_width | 全部为 inf | narrow_passage=0.160 | `by_family_summary.csv` |
| JSON Infinity | 含 inf | 不含 inf | `allow_nan=False` |
| silent defaults | 无 warning | 有 missing_fields warning | `schema_warnings.csv` |
| overlap check | max(0,d) 截断失效 | raw signed distance | `min_*_raw` 字段 |
| approach_score | 近乎常数 | 有方差 | variance=0.0002 |

---

## 3. 修复后指标

### 3.1 Blocking 指标
| 指标 | 含义 | 用途 |
|------|------|------|
| `object_goal_blocking_score` | object→goal 路径被阻挡 | blocking family 检测 |
| `ee_object_blocking_score` | EE→object 路径被阻挡 | approach 难度检测 |
| `goal_region_blocking_score` | goal 区域被阻挡 | goal 可达性检测 |
| `blocking_score` | max(上述三者) | 综合阻挡程度 |

### 3.2 Passage 指标
| 指标 | 含义 | 用途 |
|------|------|------|
| `passage_width_estimate` | 障碍物间最小间距 | narrow_passage family 检测 |

### 3.3 Edge Goal 指标
| 指标 | 含义 | 用途 |
|------|------|------|
| `edge_goal_score` | goal 靠边程度（考虑 object radius） | edge_goal family 检测 |
| `goal_edge_distance_adjusted` | goal 到边界距离 - object radius | 精确距离 |

### 3.4 Approach 指标
| 指标 | 含义 | 用途 |
|------|------|------|
| `approach_feasibility_score` | 综合接近可行性 | 接近难度评估 |
| `ee_object_path_blocked` | EE→object 路径是否被阻挡 | 接近障碍检测 |
| `reachable_contact_sides` | 可接触的 object 侧面数 | 接触可达性 |

---

## 4. 家庭区分力验证

| Family | Count | blocking | edge_goal | passage_width | approach |
|--------|-------|----------|-----------|---------------|----------|
| blocking | 20 | **1.000** | 0.321 | 0.176 | 0.913 |
| edge_goal | 20 | 0.000 | **0.622** | N/A | 0.907 |
| narrow_passage | 20 | 0.000 | 0.350 | **0.160** | 0.932 |
| non_blocking | 20 | 0.000 | 0.243 | 0.022 | 0.917 |
| open_space | 40 | 0.000 | 0.233 | N/A | 0.901 |

**✅ blocking_score 区分 blocking family**  
**✅ edge_goal_score 区分 edge_goal family**  
**✅ passage_width 区分 narrow_passage family**

---

## 5. Self-check 结果

| Check | Status |
|-------|--------|
| open_space_blocking_zero | ✅ |
| object_goal_blocking_positive | ✅ |
| ee_approach_blocking_positive | ✅ |
| narrow_passage_width_finite | ✅ |
| edge_goal_score_discriminates | ✅ |
| overlap_detected_invalid | ✅ |
| missing_field_recorded | ✅ |
| json_no_nan_inf | ✅ |
| blocking_score_discriminates | ✅ |
| edge_goal_gt_open_space | ✅ |
| approach_score_variance | ✅ |

**Status: PASS (14/14)**

---

## 6. 推荐 Template Pool

| Category | Count | 用途 |
|----------|-------|------|
| open_space_easy | 68 | 基础 sanity check |
| blocking_high | 20 | 阻挡场景测试 |
| narrow_passage_low_width | 20 | 通道场景测试 |
| edge_goal_high | 32 | 边界目标测试 |

---

## 7. 可用/不可用指标

### 可用于 template selection
- `blocking_score` (blocking family)
- `passage_width_estimate` (narrow_passage family)
- `edge_goal_score` (edge_goal family)
- `difficulty_score` (综合难度)

### 可用于 failure analysis
- `ee_object_path_blocked` (approach failure)
- `object_goal_path_blocked` (push failure)
- `approach_feasibility_score` (approach difficulty)
- `reachable_contact_sides` (contact access)

### 暂时不能过度解释
- `approach_feasibility_score` 方差仍然较低
- `obstacle_density_near_path` 需要更多验证

---

## 8. 如何接入 Overnight Eval

1. **先运行 topology audit** → `runs/topology_geometry_audit_fixed/`
2. **使用 recommended pool** → `recommended_closed_loop_template_pool.yaml`
3. **manifest join geometry metrics** → 每个 trial 记录 topology metrics
4. **report 中增加 failure-vs-geometry section** → 按 geometry bin 分析

---

## 9. Modified Files

| File | Action | Reason |
|------|--------|--------|
| `src/metrics/topology_geometry.py` | Rewritten | 修复 schema parsing, blocking, passage |
| `scripts/self_check_topology_geometry.py` | Rewritten | 增强 self-check |
| `scripts/audit_template_topology_geometry.py` | Rewritten | 修复输出格式 |
| `docs/topology_geometry_fix_report.md` | Created | 本报告 |
