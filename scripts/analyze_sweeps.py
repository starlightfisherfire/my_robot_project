#!/usr/bin/env python3
"""分析三个 sweep 数据，生成总结报告和可视化表格。
Usage: python3 analyze_sweeps.py
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict

BASE = Path("/home/brucewu/my_robot_project/runs")

SWEEPS = {
    "700g Pusher": {
        "path": None,  # auto-detect latest
        "pattern": "heavy_pusher_700g_sweep_*",
        "mass": "700g",
    },
    "250g Pusher": {
        "path": None,
        "pattern": "heavy_pusher_250g_sweep_*",
        "mass": "250g",
    },
    "Planner Trio 300g": {
        "path": None,
        "pattern": "planner_trio_300g_sweep_*",
        "mass": "300g",
    },
}

def find_latest(pattern):
    dirs = sorted(BASE.glob(pattern), reverse=True)
    return dirs[0] if dirs else None

def load_manifest(sweep_dir):
    mf = sweep_dir / "manifest.csv"
    if not mf.exists():
        return []
    with open(mf) as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

def calc_metrics(rows):
    """计算关键指标"""
    total = len(rows)
    completed = [r for r in rows if r["status"] == "completed"]
    successes = [r for r in completed if r["success"] == "True"]
    
    if not completed:
        return {
            "total": total,
            "completed": 0,
            "success_rate": 0,
            "avg_cost": 0,
            "min_cost": 0,
            "avg_best_dist": 0,
            "best_dist": float("inf"),
            "collision_total": 0,
            "collision_avg": 0,
            "avg_runtime": 0,
        }
    
    costs = [float(r["avg_cost"]) for r in completed]
    best_dists = [float(r["best_dist"]) for r in completed]
    collisions = [int(r["collision"]) for r in completed]
    runtimes = [float(r["runtime_sec"]) for r in completed]
    
    success_rate = len(successes) / len(completed) * 100
    
    return {
        "total": total,
        "completed": len(completed),
        "success_count": len(successes),
        "success_rate": success_rate,
        "avg_cost": sum(costs) / len(costs),
        "min_cost": min(costs),
        "max_cost": max(costs),
        "avg_best_dist": sum(best_dists) / len(best_dists),
        "best_dist": min(best_dists),
        "collision_total": sum(collisions),
        "collision_avg": sum(collisions) / len(collisions),
        "avg_runtime": sum(runtimes) / len(runtimes),
        "total_runtime": sum(runtimes),
    }

def analyze_by_speed(rows):
    """按速度分组的指标"""
    groups = defaultdict(list)
    for r in rows:
        if r["status"] == "completed":
            groups[r["speed"]].append(r)
    
    result = {}
    for speed, group in sorted(groups.items(), key=lambda x: float(x[0])):
        m = calc_metrics(group)
        result[speed] = m
    return result

def analyze_by_horizon(rows):
    """按 horizon 分组的指标"""
    groups = defaultdict(list)
    for r in rows:
        if r["status"] == "completed":
            groups[r["horizon"]].append(r)
    
    result = {}
    for h, group in sorted(groups.items(), key=lambda x: int(x[0])):
        m = calc_metrics(group)
        result[h] = m
    return result

def analyze_by_template(rows):
    """按模板 (label) 分组"""
    groups = defaultdict(list)
    for r in rows:
        if r["status"] == "completed":
            groups[r["label"]].append(r)
    
    result = {}
    for label, group in sorted(groups.items()):
        m = calc_metrics(group)
        # Annotate template type
        if "bh" in label:
            m["type"] = "blocking_hard"
        elif "be" in label:
            m["type"] = "blocking_easy"
        elif "bm" in label:
            m["type"] = "blocking_medium"
        elif "pm" in label:
            m["type"] = "passage_direct_medium"
        elif "ph" in label:
            m["type"] = "passage_hard"
        else:
            m["type"] = label
        result[label] = m
    return result

def format_table(headers, rows, col_widths=None):
    """生成 ASCII 表格"""
    if not rows:
        return "(空)"
    
    # Auto-width
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    
    def fmt_row(cells):
        return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |"
    
    sep = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    lines = [fmt_row(headers), sep]
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)

def main():
    output = []
    
    sweep_data = {}
    for name, cfg in SWEEPS.items():
        cfg["path"] = find_latest(cfg["pattern"])
        if not cfg["path"]:
            output.append(f"⚠️ {name}: 找不到数据目录")
            continue
        rows = load_manifest(cfg["path"])
        sweep_data[name] = {
            "rows": rows,
            "metrics": calc_metrics(rows),
            "path": cfg["path"],
            "mass": cfg["mass"],
        }
    
    # ═══════════════════════════════════
    # 1. 总体概览
    # ═══════════════════════════════════
    output.append("=" * 70)
    output.append("  三个 Sweep 综合总结")
    output.append("=" * 70)
    output.append("")
    
    overview = [["Sweep", "总任务", "完成", "成功率", "Avg Cost", "Best距离", "碰撞/步", "Avg耗时"]]
    for name, sd in sweep_data.items():
        m = sd["metrics"]
        overview.append([
            name,
            m["total"],
            m["completed"],
            f"{m['success_rate']:.1f}%",
            f"{m['avg_cost']:.2f}",
            f"{m['best_dist']:.4f}m",
            f"{m['collision_avg']:.1f}",
            f"{m['avg_runtime']:.0f}s",
        ])
    
    output.append("## 总体概览")
    output.append(format_table(overview[0], overview[1:]))
    output.append("")
    
    # ═══════════════════════════════════
    # 2. 按速度分析
    # ═══════════════════════════════════
    output.append("## 按速度 (Speed) 分析")
    output.append("")
    
    for name, sd in sweep_data.items():
        speed_data = analyze_by_speed(sd["rows"])
        if not speed_data:
            continue
        output.append(f"### {name} ({sd['mass']})")
        headers = ["Speed(m/s)", "成功%", "Avg Cost", "Best Dist", "碰撞/步"]
        rows = []
        for s, m in sorted(speed_data.items(), key=lambda x: float(x[0])):
            rows.append([
                s,
                f"{m['success_rate']:.1f}%",
                f"{m['avg_cost']:.2f}",
                f"{m['best_dist']:.5f}m",
                f"{m['collision_avg']:.1f}",
            ])
        output.append(format_table(headers, rows))
        output.append("")
    
    # ═══════════════════════════════════
    # 3. 按 Horizon 分析
    # ═══════════════════════════════════
    output.append("## 按 Horizon 分析")
    output.append("")
    
    for name, sd in sweep_data.items():
        h_data = analyze_by_horizon(sd["rows"])
        if not h_data:
            continue
        output.append(f"### {name}")
        headers = ["Horizon", "成功%", "Avg Cost", "Best Dist", "碰撞/步", "Avg Runtime"]
        rows = []
        for h, m in sorted(h_data.items(), key=lambda x: int(x[0])):
            rows.append([
                h,
                f"{m['success_rate']:.1f}%",
                f"{m['avg_cost']:.2f}",
                f"{m['best_dist']:.5f}m",
                f"{m['collision_avg']:.1f}",
                f"{m['avg_runtime']:.0f}s",
            ])
        output.append(format_table(headers, rows))
        output.append("")
    
    # ═══════════════════════════════════
    # 4. 按模板分析
    # ═══════════════════════════════════
    output.append("## 按障碍物模板分析")
    output.append("")
    
    for name, sd in sweep_data.items():
        t_data = analyze_by_template(sd["rows"])
        if not t_data:
            continue
        output.append(f"### {name}")
        headers = ["模板", "类型", "成功%", "Avg Cost", "Best Dist", "碰撞/步"]
        rows = []
        for label, m in sorted(t_data.items()):
            rows.append([
                label,
                m.get("type", label),
                f"{m['success_rate']:.1f}%",
                f"{m['avg_cost']:.2f}",
                f"{m['best_dist']:.5f}m",
                f"{m['collision_avg']:.1f}",
            ])
        output.append(format_table(headers, rows))
        output.append("")
    
    # ═══════════════════════════════════
    # 5. 跨 sweep 对比分析
    # ═══════════════════════════════════
    output.append("=" * 70)
    output.append("  跨 Sweep 分析结论")
    output.append("=" * 70)
    output.append("")
    
    # 5.1 Cost 对比
    output.append("### 💰 最终 Cost 对比")
    output.append("")
    for name, sd in sweep_data.items():
        m = sd["metrics"]
        output.append(f"- **{name}** — Avg Cost: {m['avg_cost']:.2f}, Min: {m['min_cost']:.2f}, Max: {m['max_cost']:.2f}")
    output.append("")
    
    # Find best sweep by cost
    best_cost = min(sweep_data.items(), key=lambda x: x[1]["metrics"]["avg_cost"])
    output.append(f"**Cost 最优:** {best_cost[0]} (Avg={best_cost[1]['metrics']['avg_cost']:.2f})")
    output.append("")
    
    # 5.2 实际效果 (best_dist, collision, success)
    output.append("### 🎯 实际效果对比")
    output.append("")
    for name, sd in sweep_data.items():
        m = sd["metrics"]
        output.append(f"- **{name}** — 成功率: {m['success_rate']:.1f}%, Best距离: {m['best_dist']:.4f}m, 平均碰撞/步: {m['collision_avg']:.1f}")
    output.append("")
    
    best_success = max(sweep_data.items(), key=lambda x: x[1]["metrics"]["success_rate"])
    output.append(f"**成功率最高:** {best_success[0]} ({best_success[1]['metrics']['success_rate']:.1f}%)")
    
    best_dist = min(sweep_data.items(), key=lambda x: x[1]["metrics"]["best_dist"])
    output.append(f"**最近距离:** {best_dist[0]} ({best_dist[1]['metrics']['best_dist']:.4f}m)")
    output.append("")
    
    # 5.3 Planner Trio 专项（如果有 planner 列）
    if "Planner Trio 300g" in sweep_data:
        sd = sweep_data["Planner Trio 300g"]
        rows = sd["rows"]
        # 按 planner 分组
        planner_groups = defaultdict(list)
        for r in rows:
            if r.get("planner") and r["status"] == "completed":
                planner_groups[r["planner"]].append(r)
        
        if planner_groups:
            output.append("### 🤖 Planner 对比 (Planner Trio only)")
            output.append("")
            headers = ["Planner", "任务数", "成功%", "Avg Cost", "Best Dist", "碰撞/步"]
            p_rows = []
            for planner, group in sorted(planner_groups.items()):
                m = calc_metrics(group)
                p_rows.append([
                    planner,
                    m["completed"],
                    f"{m['success_rate']:.1f}%",
                    f"{m['avg_cost']:.2f}",
                    f"{m['best_dist']:.5f}m",
                    f"{m['collision_avg']:.1f}",
                ])
            output.append(format_table(headers, p_rows))
            output.append("")
    
    # 5.4 最终建议
    output.append("### 📋 综合建议")
    output.append("")
    
    return "\n".join(output)

if __name__ == "__main__":
    print(main())
