#!/usr/bin/env python3
"""Planner Trio 300g 严格分析：三规划器 × 六模板对比
重点：cost 和实际效果（成功率、碰撞、最佳距离）
"""

import csv
from pathlib import Path
from collections import defaultdict

def load_manifest():
    dirs = sorted(Path("/home/brucewu/my_robot_project/runs").glob("planner_trio_300g_sweep_*"), reverse=True)
    if not dirs:
        print("❌ 找不到 planner_trio sweep 目录")
        return []
    mf = dirs[0] / "manifest.csv"
    with open(mf) as f:
        return list(csv.DictReader(f))

def metrics(rows):
    total = len(rows)
    completed = [r for r in rows if r["status"] == "completed"]
    if not completed:
        return {"total": total, "completed": 0}
    successes = [r for r in completed if r["success"] == "True"]
    costs = [float(r["avg_cost"]) for r in completed]
    dists = [float(r["best_dist"]) for r in completed]
    collisions = [int(r["collision"]) for r in completed]
    runtimes = [float(r["runtime_sec"]) for r in completed]
    return {
        "total": total,
        "completed": len(completed),
        "success_count": len(successes),
        "success_rate": len(successes) / len(completed) * 100,
        "avg_cost": sum(costs) / len(costs),
        "min_cost": min(costs),
        "max_cost": max(costs),
        "avg_best_dist": sum(dists) / len(dists),
        "best_dist": min(dists),
        "avg_collision": sum(collisions) / len(collisions),
        "avg_runtime": sum(runtimes) / len(runtimes),
    }

def fmt_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(str(c)))
    sep = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    lines = ["| " + " | ".join(str(c).ljust(w) for c, w in zip(headers, widths)) + " |", sep]
    for row in rows:
        lines.append("| " + " | ".join(str(c).ljust(w) for c, w in zip(row, widths)) + " |")
    return "\n".join(lines)

def main():
    rows = load_manifest()
    if not rows:
        return

    completed = [r for r in rows if r["status"] == "completed"]
    print(f"# Planner Trio 300g 严格分析")
    print(f"总任务: {len(rows)} | 完成: {len(completed)}")
    print()

    # ── 1. 三规划器总体对比 ──
    planners = defaultdict(list)
    for r in completed:
        planners[r["planner"]].append(r)

    print("## 一、三规划器总体对比")
    print()
    headers = ["Planner", "任务数", "成功率", "Avg Cost", "Best Dist", "碰撞/步", "Avg Runtime"]
    p_rows = []
    for pname, prows in sorted(planners.items()):
        m = metrics(prows)
        p_rows.append([pname, m["completed"], f"{m['success_rate']:.1f}%", f"{m['avg_cost']:.2f}",
                       f"{m['best_dist']:.5f}m", f"{m['avg_collision']:.1f}", f"{m['avg_runtime']:.0f}s"])
    print(fmt_table(headers, p_rows))
    print()

    # 找最优
    best_cost = min(planners.items(), key=lambda x: metrics(x[1])["avg_cost"])
    best_succ = max(planners.items(), key=lambda x: metrics(x[1])["success_rate"])
    print(f"**Cost 最优:** {best_cost[0]} (Avg={metrics(best_cost[1])['avg_cost']:.2f})")
    print(f"**成功率最高:** {best_succ[0]} ({metrics(best_succ[1])['success_rate']:.1f}%)")
    print()

    # ── 2. 按模板 × 规划器 ──
    print("## 二、按障碍物模板 × 规划器")
    print()

    templates = defaultdict(lambda: defaultdict(list))
    for r in completed:
        templates[r["label"]][r["planner"]].append(r)

    template_names = {
        "be00": "blocking_easy", "bm00": "blocking_medium", "bh00": "blocking_hard",
        "pw00": "passage_direct_wide", "pm00": "passage_direct_medium", "ph00": "passage_hard",
    }

    for label in sorted(templates.keys()):
        tname = template_names.get(label, label)
        print(f"### {label} ({tname})")
        print()
        headers = ["Planner", "成功率", "Avg Cost", "Best Dist", "碰撞/步"]
        trows = []
        for pname in sorted(planners.keys()):
            if pname in templates[label]:
                m = metrics(templates[label][pname])
                trows.append([pname, f"{m['success_rate']:.1f}%", f"{m['avg_cost']:.2f}",
                              f"{m['best_dist']:.5f}m", f"{m['avg_collision']:.1f}"])
        if trows:
            print(fmt_table(headers, trows))
        else:
            print("_(无数据)_")
        print()

    # ── 3. 按速度 × 规划器 ──
    print("## 三、按速度 × 规划器")
    print()
    speeds = defaultdict(lambda: defaultdict(list))
    for r in completed:
        speeds[r["speed"]][r["planner"]].append(r)

    for speed in sorted(speeds.keys(), key=float):
        print(f"### Speed={speed} m/s")
        headers = ["Planner", "成功率", "Avg Cost", "Best Dist", "碰撞/步"]
        srows = []
        for pname in sorted(planners.keys()):
            if pname in speeds[speed]:
                m = metrics(speeds[speed][pname])
                srows.append([pname, f"{m['success_rate']:.1f}%", f"{m['avg_cost']:.2f}",
                              f"{m['best_dist']:.5f}m", f"{m['avg_collision']:.1f}"])
        if srows:
            print(fmt_table(headers, srows))
        print()

    # ── 4. 关键结论 ──
    print("## 四、关键结论")
    print()

    # 4.1 是否有人攻克了 passage_hard
    ph_data = templates.get("ph00", {})
    if ph_data:
        print("### passage_hard 攻克情况")
        for pname, prows in sorted(ph_data.items()):
            m = metrics(prows)
            print(f"- **{pname}**: 成功率 {m['success_rate']:.1f}%, Cost {m['avg_cost']:.2f}")
        print()

    # 4.2 blocking_hard 对比
    bh_data = templates.get("bh00", {})
    if bh_data:
        print("### blocking_hard 对比")
        for pname, prows in sorted(bh_data.items()):
            m = metrics(prows)
            print(f"- **{pname}**: 成功率 {m['success_rate']:.1f}%, Cost {m['avg_cost']:.2f}")
        print()

    # 4.3 综合推荐
    print("### 综合推荐")
    print()
    for pname in sorted(planners.keys()):
        m = metrics(planners[pname])
        print(f"- **{pname}**: 成功率 {m['success_rate']:.1f}%, Avg Cost {m['avg_cost']:.2f}, Best Dist {m['best_dist']:.5f}m")
    print()

if __name__ == "__main__":
    main()
