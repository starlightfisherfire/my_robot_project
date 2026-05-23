#!/usr/bin/env python3
"""Analyze all blocking_hard (bh) runs across all experiments."""

import os
import csv
import glob
import sys
from collections import defaultdict

RUNS_DIR = "/home/brucewu/my_robot_project/runs"

def find_blocking_hard_manifests():
    """Find all manifests and extract blocking_hard rows."""
    manifests = glob.glob(os.path.join(RUNS_DIR, "**", "manifest.csv"), recursive=True)
    
    all_rows = []
    
    for mpath in sorted(manifests):
        rel = os.path.relpath(mpath, RUNS_DIR)
        try:
            with open(mpath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Check for blocking_hard indicators across different column names
                    text_fields = ' '.join(str(v).lower() for v in row.values())
                    
                    # Look for blocking_hard in any field
                    is_bh = False
                    for v in row.values():
                        if 'blocking_hard' in str(v).lower():
                            is_bh = True
                            break
                        if str(v) == 'bh00':
                            is_bh = True
                            break
                    
                    if is_bh:
                        row['_source'] = rel
                        row['_manifest'] = mpath
                        all_rows.append(row)
        except Exception as e:
            print(f"SKIP {rel}: {e}", file=sys.stderr)
    
    return all_rows

def extract_metrics(row):
    """Extract key metrics from a row, handling various column names."""
    metrics = {}
    
    # Success
    for col in ['success', 'Success', 'SUCCESS']:
        if col in row:
            val = str(row[col]).strip().lower()
            metrics['success'] = val in ('true', '1', 'yes')
            break
    
    # Distance
    for col in ['best_dist', 'Best dist', 'dist', 'final_dist', 'best_distance']:
        if col in row and row[col]:
            try:
                metrics['best_dist'] = float(row[col])
                break
            except:
                pass
    
    # Runtime
    for col in ['runtime_sec', 'Total runtime', 'runtime', 'duration']:
        if col in row and row[col]:
            try:
                metrics['runtime'] = float(row[col])
                break
            except:
                pass
    
    # MPC steps
    for col in ['mpc_steps', 'MPC steps', 'max_mpc_steps']:
        if col in row and row[col]:
            try:
                metrics['mpc_steps'] = int(row[col])
                break
            except:
                pass
    
    # Speed
    for col in ['speed', 'max_speed_mps', 'Speed']:
        if col in row and row[col]:
            try:
                metrics['speed'] = float(row[col])
                break
            except:
                pass
    
    # Horizon
    for col in ['horizon', 'Horizon', 'planning_horizon']:
        if col in row and row[col]:
            try:
                metrics['horizon'] = int(row[col])
                break
            except:
                pass
    
    # Budget
    for col in ['budget', 'total_budget', 'Budget']:
        if col in row and row[col]:
            try:
                metrics['budget'] = int(float(row[col]))
                break
            except:
                pass
    
    # Planner mode
    for col in ['planner_mode', 'Planner mode', 'mode', 'planner']:
        if col in row and row[col]:
            metrics['planner'] = str(row[col]).strip()
            break
    
    # Pusher mass
    for col in ['pusher_mass', 'Pusher mass', 'mass']:
        if col in row and row[col]:
            try:
                metrics['pusher_mass'] = float(row[col])
                break
            except:
                pass
    
    # Config name
    for col in ['config', 'Config', 'label', 'Label', 'cfg_name']:
        if col in row and row[col]:
            metrics['config_name'] = str(row[col]).strip()
            break
    
    # Collision count
    for col in ['collision_count', 'collision', 'Collision']:
        if col in row and row[col]:
            try:
                metrics['collision'] = int(row[col])
                break
            except:
                pass
    
    # Avg cost
    for col in ['avg_cost', 'Avg cost']:
        if col in row and row[col]:
            try:
                metrics['avg_cost'] = float(row[col])
                break
            except:
                pass
    
    # Min cost
    for col in ['min_cost', 'Min cost']:
        if col in row and row[col]:
            try:
                metrics['min_cost'] = float(row[col])
                break
            except:
                pass
    
    # Split
    for col in ['split', 'Split']:
        if col in row and row[col]:
            metrics['split'] = str(row[col]).strip()
            break
    
    # Template index
    for col in ['template_index', 'idx', 'Index']:
        if col in row and row[col]:
            try:
                metrics['template_index'] = int(row[col])
                break
            except:
                pass
    
    # Status
    for col in ['status', 'Status']:
        if col in row and row[col]:
            metrics['status'] = str(row[col]).strip()
            break
    
    return metrics

def scan_logs_for_blocking_hard():
    """Also scan MPPI blocking demo logs that might not be in manifests."""
    log_dirs = glob.glob(os.path.join(RUNS_DIR, "mppi_blocking_*", "logs"))
    extra_rows = []
    
    for log_dir in log_dirs:
        for logfile in sorted(glob.glob(os.path.join(log_dir, "*bh*.log"))):
            metrics = {}
            metrics['_source'] = os.path.relpath(logfile, RUNS_DIR)
            metrics['config_name'] = os.path.basename(logfile).replace('.log', '')
            
            with open(logfile, 'r') as f:
                content = f.read()
            
            if 'Success: True' in content:
                metrics['success'] = True
            elif 'Success: False' in content:
                metrics['success'] = False
            
            import re
            # Best dist
            m = re.search(r'Best dist:\s*([\d.]+)m', content)
            if m:
                metrics['best_dist'] = float(m.group(1))
            
            # Total runtime
            m = re.search(r'Total runtime:\s*([\d.]+)s', content)
            if m:
                metrics['runtime'] = float(m.group(1))
            
            # STRICT POSE
            m = re.search(r'STRICT POSE STOP.*dist=([\d.]+)mm', content)
            if m:
                metrics['strict_pose_dist_mm'] = float(m.group(1))
            
            # Speed
            m = re.search(r'max_speed_mps[= ]+([\d.]+)', content)
            if m:
                metrics['speed'] = float(m.group(1))
            
            # Horizon
            m = re.search(r'Planning horizon:\s*(\d+)', content)
            if m:
                metrics['horizon'] = int(m.group(1))
            
            # Budget (derived from execute*mmpc)
            m = re.search(r'Execute steps:\s*(\d+)', content)
            exec_steps = int(m.group(1)) if m else None
            m = re.search(r'max_mpc_steps[= ]+(\d+)', content)
            max_mpc = int(m.group(1)) if m else None
            if exec_steps and max_mpc:
                metrics['budget'] = exec_steps * max_mpc
            
            # Temperature
            m = re.search(r'mppi_temperature[= ]+([\d.]+)', content)
            if m:
                metrics['mppi_temperature'] = float(m.group(1))
            
            # Workers
            m = re.search(r'cem_workers[= ]+(\d+)', content)
            if m:
                metrics['workers'] = int(m.group(1))
            
            # Planner mode
            m = re.search(r'planner_mode[= ]+(\w+)', content)
            if m:
                metrics['planner'] = m.group(1)
            
            # Pusher mass
            m = re.search(r'Pusher mass:\s*([\d.]+)', content)
            if m:
                metrics['pusher_mass'] = float(m.group(1))
            
            # MPC steps
            m = re.search(r'MPC Step (\d+)/', content)
            if m:
                # Count the last step
                steps = [int(x) for x in re.findall(r'MPC Step (\d+)/', content)]
                if steps:
                    metrics['mpc_steps'] = max(steps)
            
            # Collision count
            collisions = re.findall(r'Planned collision: count=(\d+)', content)
            if collisions:
                metrics['total_collision'] = sum(int(c) for c in collisions)
            
            # STRICT POSE at frame
            m = re.search(r'STRICT POSE STOP at step (\d+)!', content)
            if m:
                metrics['stop_frame'] = int(m.group(1))
            
            extra_rows.append({'metrics': metrics, 'row_data': {}})
    
    return extra_rows


def main():
    print("=" * 90)
    print("BLOCKING HARD — 全实验数据分析")
    print("=" * 90)
    
    rows = find_blocking_hard_manifests()
    print(f"\n从 manifest 找到 {len(rows)} 条 blocking_hard 记录\n")
    
    # Parse metrics
    parsed = []
    for row in rows:
        m = extract_metrics(row)
        parsed.append((m, row))
    
    # Also scan logs
    log_results = scan_logs_for_blocking_hard()
    print(f"从日志找到 {len(log_results)} 条额外 blocking_hard 记录\n")
    
    for lr in log_results:
        parsed.append((lr['metrics'], lr['row_data']))
    
    # Separate by planner
    by_planner = defaultdict(list)
    for m, row in parsed:
        planner = m.get('planner', 'unknown')
        if not planner:
            planner = 'unknown'
        by_planner[planner].append((m, row))
    
    # Print all results sorted by best_dist
    print("=" * 90)
    print("全部 blocking_hard 记录 (按最终距离排序)")
    print("=" * 90)
    
    # Sort all by distance
    all_sorted = sorted(parsed, key=lambda x: x[0].get('best_dist', 999))
    
    print(f"\n{'来源':>35s} | {'Planner':>12s} | {'Speed':>6s} | {'H':>4s} | {'B':>5s} | {'Mass':>5s} | {'Temp':>5s} | {'Success':>8s} | {'BestDist':>9s} | {'Runtime':>8s} | {'MPC':>4s} | {'Collision':>9s}")
    print("-" * 140)
    
    for m, row in all_sorted:
        source = m.get('_source', '?')[-35:]
        planner = m.get('planner', '?')[:12]
        speed = m.get('speed', 0)
        hor = m.get('horizon', 0)
        budget = m.get('budget', 0)
        mass = m.get('pusher_mass', 0)
        temp = m.get('mppi_temperature', '-')
        succ = '✅' if m.get('success') else '❌'
        dist = m.get('best_dist', 999)
        runtime = m.get('runtime', 0)
        mpc = m.get('mpc_steps', 0)
        collision = m.get('collision', m.get('total_collision', '-'))
        
        temp_str = f"{temp:>5}" if isinstance(temp, (int, float)) else f"{str(temp):>5}"
        
        print(f"{source:>35s} | {planner:>12s} | {speed:>5.2f} | {hor:>4d} | {budget:>5d} | {mass:>5.3f} | {temp_str} | {succ:>8s} | {dist:>8.4f}m | {runtime:>7.1f}s | {mpc:>4d} | {str(collision):>9s}")
    
    # Summary by planner
    print("\n" + "=" * 90)
    print("按 Planner 汇总")
    print("=" * 90)
    
    for planner, items in sorted(by_planner.items()):
        succ_count = sum(1 for m, _ in items if m.get('success'))
        total = len(items)
        dists = [m.get('best_dist', 999) for m, _ in items]
        best_dist = min(dists)
        avg_dist = sum(dists) / len(dists) if dists else 0
        
        print(f"\n  {planner}: {total} runs | "
              f"Success: {succ_count}/{total} ({100*succ_count/total:.0f}%) | "
              f"Best: {best_dist:.4f}m | "
              f"Avg: {avg_dist:.4f}m")
    
    # Top 3 overall
    print("\n" + "=" * 90)
    print("🏆 TOP 3 blocking_hard 最佳配置")
    print("=" * 90)
    
    top3 = all_sorted[:3]
    for i, (m, row) in enumerate(top3):
        planner = m.get('planner', '?')
        speed = m.get('speed', '?')
        hor = m.get('horizon', '?')
        budget = m.get('budget', '?')
        temp = m.get('mppi_temperature', None)
        mass = m.get('pusher_mass', '?')
        succ = '✅ SUCCESS' if m.get('success') else '❌ FAILED'
        dist = m.get('best_dist', 999)
        runtime = m.get('runtime', 0)
        mpc = m.get('mpc_steps', '?')
        source = m.get('_source', '?')
        
        strict = ""
        if m.get('strict_pose_dist_mm'):
            strict = f" | PoseStop: {m['strict_pose_dist_mm']:.2f}mm"
        
        print(f"\n  {i+1}. {succ} — best_dist = {dist:.4f}m"
              f"\n     Config: {m.get('config_name', '?')}"
              f"\n     Planner={planner} | Speed={speed}m/s | Horizon={hor} | Budget={budget} | Mass={mass}kg"
              f"{' | Temp=' + str(temp) if temp else ''}"
              f"\n     Runtime={runtime:.0f}s | MPC steps={mpc}{strict}"
              f"\n     Source: {source}")
    
    # Key finding: success vs failure split
    print("\n" + "=" * 90)
    print("📊 成败分析")
    print("=" * 90)
    
    successes = [(m, r) for m, r in parsed if m.get('success')]
    failures = [(m, r) for m, r in parsed if not m.get('success')]
    
    print(f"\n  成功: {len(successes)} 次")
    print(f"  失败: {len(failures)} 次")
    print(f"  总成功率: {100*len(successes)/len(parsed):.1f}%" if parsed else "N/A")
    
    if successes:
        print(f"\n  成功配置特征:")
        speeds = [m.get('speed', 0) for m, _ in successes]
        hors = [m.get('horizon', 0) for m, _ in successes]
        budgets = [m.get('budget', 0) for m, _ in successes]
        print(f"    Speed:   avg={sum(speeds)/len(speeds):.2f} m/s, range=[{min(speeds):.2f}, {max(speeds):.2f}]")
        print(f"    Horizon: avg={sum(hors)/len(hors):.0f}, range=[{min(hors)}, {max(hors)}]")
        print(f"    Budget:  avg={sum(budgets)/len(budgets):.0f}, range=[{min(budgets)}, {max(budgets)}]")

if __name__ == '__main__':
    main()
