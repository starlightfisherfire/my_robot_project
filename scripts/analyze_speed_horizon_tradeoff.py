"""
Analyze the speed × horizon × execution budget tradeoff.

System parameters:
  - control_dt = 0.1s
  - base_max_speed = 0.05 m/s
  - speed multiplier: 0.75, 1.0, 1.5, 2.0
  - horizon: planning steps (each = 0.1s)
  - execute_steps: steps per MPC replan (typically 20)
  - max_mpc_steps: number of replanning iterations
  - total_budget = execute_steps * max_mpc_steps
"""

import csv
from pathlib import Path
from collections import defaultdict

CONTROL_DT = 0.1
BASE_SPEED = 0.05

def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def safe_float(v, default=0.0):
    try:
        return float(v) if v else default
    except ValueError:
        return default

def safe_int(v, default=0):
    try:
        return int(v) if v else default
    except ValueError:
        return default

# ─── Load data ──────────────────────────────────────────────────────────────
h140 = load_csv("runs/horizon140_sweep_20260515_141327/manifest.csv")
sh = load_csv("runs/speed_horizon_ablation_20260515_120253/manifest.csv")
do = load_csv("runs/dual_obstacle_speed_ablation_20260515_120428/manifest.csv")
sixpack = load_csv("runs/debug/obstacle_sixpack_full_sweep_20260515_001333/manifest.csv")
hi_ablation = load_csv("runs/debug/obstacle_speed_ablation_high_20260515_012623/manifest.csv")
hi_ablation_b = load_csv("runs/debug/obstacle_speed_ablation_high_b800b1000_20260515_013002/manifest.csv")

# ─── 1. Horizon140 sweep: speed effect at fixed horizon ────────────────────
print("=" * 80)
print("ANALYSIS 1: Horizon=140, Speed Ablation (up to 100 MPC steps)")
print("=" * 80)

speed_groups = defaultdict(list)
for row in h140:
    speed = float(row["speed"])
    speed_groups[speed].append(row)

for speed in sorted(speed_groups.keys()):
    rows = speed_groups[speed]
    successes = sum(1 for r in rows if r["status"] == "True")
    total = len(rows)
    avg_mpc = sum(safe_int(r["mpc_steps"]) for r in rows) / total
    avg_dist = sum(safe_float(r["best_dist"]) for r in rows) / total
    step_mm = speed * BASE_SPEED * CONTROL_DT * 1000
    pred_m = 140 * CONTROL_DT * speed * BASE_SPEED

    print(f"\nSpeed {speed:.2f}:")
    print(f"  Success: {successes}/{total} ({100*successes/total:.0f}%)")
    print(f"  Avg MPC steps: {avg_mpc:.1f}, Avg best_dist: {avg_dist:.4f}m")
    print(f"  Step size: {step_mm:.2f}mm, Prediction range: {pred_m:.2f}m")

# ─── 2. Speed × Horizon matrix (10 MPC steps) ─────────────────────────────
print("\n" + "=" * 80)
print("ANALYSIS 2: Speed × Horizon (10 MPC steps = 100 total)")
print("=" * 80)

matrix = defaultdict(lambda: defaultdict(lambda: {"s": 0, "t": 0}))
for row in sh:
    s, h = float(row["speed"]), int(row["horizon"])
    matrix[s][h]["t"] += 1
    if row["status"] == "True":
        matrix[s][h]["s"] += 1

horizons = sorted(set(int(r["horizon"]) for r in sh))
speeds = sorted(set(float(r["speed"]) for r in sh))

print(f"\n{'Speed':>8}", end="")
for h in horizons:
    print(f"  H={h:>3d}", end="")
print("\n" + "-" * 30)
for s in speeds:
    print(f"{s:>8.2f}", end="")
    for h in horizons:
        d = matrix[s][h]
        rate = d["s"] / d["t"] if d["t"] > 0 else 0
        print(f"  {rate:>5.0%}", end="")
    print()

print(f"\n{'Speed':>8}", end="")
for h in horizons:
    print(f"  H={h:>3d}", end="")
print("  (Physical prediction range in meters)")
print("-" * 45)
for s in speeds:
    print(f"{s:>8.2f}", end="")
    for h in horizons:
        pred = h * CONTROL_DT * s * BASE_SPEED
        print(f"  {pred:>5.2f}", end="")
    print()

# ─── 3. Dual obstacle: speed effect at h=80 ───────────────────────────────
print("\n" + "=" * 80)
print("ANALYSIS 3: Dual Obstacle (h=80, 50 MPC steps)")
print("=" * 80)

do_groups = defaultdict(list)
for row in do:
    s = float(row["speed"])
    do_groups[s].append(row)

for s in sorted(do_groups.keys()):
    rows = do_groups[s]
    successes = sum(1 for r in rows if r["status"] == "True")
    total = len(rows)
    print(f"Speed {s:.2f}: {successes}/{total} ({100*successes/total:.0f}%)")

# ─── 4. Sixpack: speed × budget (h=80) ────────────────────────────────────
print("\n" + "=" * 80)
print("ANALYSIS 4: Sixpack Sweep (h=80, various budgets)")
print("=" * 80)

# Parse speed from config: "speed075_b1000_" → 0.75
def parse_speed(config):
    s = config.split("_")[0].replace("speed", "")
    return float(s[0] + "." + s[1:])

def parse_budget(config):
    b = config.split("_")[1].replace("b", "")
    return int(b)

sb = defaultdict(lambda: defaultdict(list))
for row in sixpack:
    s = parse_speed(row["config"])
    b = parse_budget(row["config"])
    sb[s][b].append(row)

budgets = sorted(set(parse_budget(r["config"]) for r in sixpack))
speeds_sb = sorted(set(parse_speed(r["config"]) for r in sixpack))

# Best cost avg (lower = better, measures planning quality)
print("\nBest cost avg (lower = better):")
print(f"{'Speed':>8}", end="")
for b in budgets:
    print(f"  B={b:>4d}", end="")
print("\n" + "-" * 35)
for s in speeds_sb:
    print(f"{s:>8.2f}", end="")
    for b in budgets:
        entries = sb[s][b]
        if entries:
            avg = sum(safe_float(e["best_cost_avg"]) for e in entries) / len(entries)
            print(f"  {avg:>6.3f}", end="")
        else:
            print(f"  {'N/A':>6}", end="")
    print()

# Total steps (lower = faster task completion)
print(f"\nTotal steps (lower = faster completion):")
print(f"{'Speed':>8}", end="")
for b in budgets:
    print(f"  B={b:>4d}", end="")
print("\n" + "-" * 35)
for s in speeds_sb:
    print(f"{s:>8.2f}", end="")
    for b in budgets:
        entries = sb[s][b]
        if entries:
            avg = sum(safe_int(e["total_steps"]) for e in entries) / len(entries)
            print(f"  {avg:>6.1f}", end="")
        else:
            print(f"  {'N/A':>6}", end="")
    print()

# Collision rate
print(f"\nCollision rate:")
print(f"{'Speed':>8}", end="")
for b in budgets:
    print(f"  B={b:>4d}", end="")
print("\n" + "-" * 35)
for s in speeds_sb:
    print(f"{s:>8.2f}", end="")
    for b in budgets:
        entries = sb[s][b]
        if entries:
            avg = sum(safe_float(e["collision_rate"]) for e in entries) / len(entries)
            print(f"  {avg:>6.4f}", end="")
        else:
            print(f"  {'N/A':>6}", end="")
    print()

# ─── 5. Theoretical parameter space ───────────────────────────────────────
print("\n" + "=" * 80)
print("ANALYSIS 5: Theoretical Parameter Space (horizon=140, budget=1000)")
print("=" * 80)

print(f"\n{'Speed':>6} {'Step(mm)':>9} {'Reach(m)':>9} {'Pred(m)':>8} {'PredT(s)':>8} {'Steps/gap':>10}")
print("-" * 55)
for speed in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
    step = speed * BASE_SPEED * CONTROL_DT * 1000
    reach = speed * BASE_SPEED * 1000 * CONTROL_DT
    pred = 140 * CONTROL_DT * speed * BASE_SPEED
    pred_t = 140 * CONTROL_DT
    # Steps to cross 5cm obstacle gap
    gap_steps = 50 / step  # 50mm / step_mm
    print(f"{speed:>6.2f} {step:>9.2f} {reach:>9.2f} {pred:>8.2f} {pred_t:>8.1f} {gap_steps:>10.1f}")

# ─── 6. Horizon 140 vs 80: the critical difference ────────────────────────
print("\n" + "=" * 80)
print("ANALYSIS 6: Why Horizon 140 is the Key Differentiator")
print("=" * 80)

print("\nHorizon140 sweep results (speed 0.75, up to 100 MPC steps):")
for row in h140:
    if float(row["speed"]) == 0.75:
        status = "OK" if row["status"] == "True" else "FAIL"
        print(f"  {row['mode']:25s} t{row['template_idx']}: {status}  "
              f"mpc_steps={row['mpc_steps']:>3s}  best_dist={row['best_dist']}m")

print("\nSpeed×Horizon ablation (speed 0.75, only 10 MPC steps):")
for row in sh:
    if float(row["speed"]) == 0.75:
        status = "OK" if row["status"] == "True" else "FAIL"
        print(f"  H={row['horizon']:>3s} {row['mode']:25s} t{row['template_idx']}: {status}  "
              f"best_dist={row['best_dist']}m")

print("\n→ With only 100 total steps, horizon 100/120 ALL FAIL at speed 0.75.")
print("  But horizon 140 with 1000 steps → 100% success at speed 0.75.")
print("  Horizon 140 gives the planner enough 'vision' to find a path.")

# ─── 7. Speed vs Budget tradeoff ──────────────────────────────────────────
print("\n" + "=" * 80)
print("ANALYSIS 7: Speed vs Budget Tradeoff")
print("=" * 80)

print("""
The key tradeoff:

  speed * budget = total physical reach capacity

  Speed 0.75, Budget 1000: reach = 0.75*0.05*1000*0.1 = 3.75m
  Speed 1.00, Budget 1000: reach = 1.00*0.05*1000*0.1 = 5.00m
  Speed 1.50, Budget 1000: reach = 1.50*0.05*1000*0.1 = 7.50m
  Speed 2.00, Budget 1000: reach = 2.00*0.05*1000*0.1 = 10.0m

All far exceed workspace (~0.5m). So reach is NOT the bottleneck.

The real constraint is PRECISION:
  - Lower speed → finer step size → better obstacle threading
  - Higher speed → coarser steps → easier to collide

And PREDICTION QUALITY:
  - Horizon in steps is fixed (140)
  - Higher speed → MuJoCo predicts further physically
  - But also predicts with less granularity per unit distance
  - The planner needs to predict CONTACT and COLLISION accurately
  - At high speed, a single step covers more distance →
    collision detection is coarser → harder to plan safe paths
""")

# ─── 8. Execute steps analysis ────────────────────────────────────────────
print("=" * 80)
print("ANALYSIS 8: Execute Steps (single execution chunk)")
print("=" * 80)

print("""
execute_steps = how many env steps to run before MPC replans.

Default: execute_steps=20 (2.0 seconds physical time)

At speed 0.75: 20 steps = 75mm of travel between replans
At speed 1.0:  20 steps = 100mm of travel between replans
At speed 1.5:  20 steps = 150mm of travel between replans
At speed 2.0:  20 steps = 200mm of travel between replans

For 5cm obstacle gap:
  - Speed 0.75: ~13 steps to cross → 1-2 replans during crossing
  - Speed 2.0:  ~5 steps to cross  → possibly 0 replans during crossing

Lower execute_steps (e.g., 10) → replans more often → better adaptation
Higher execute_steps (e.g., 30) → replans less → faster but less responsive

The question "is execute_steps=20 or 30 better?" depends on:
  - If the task is simple (blocking_easy): 30 is fine, faster
  - If the task needs precise threading (passage_hard): 10-20 is better
""")

# ─── Final verdict ────────────────────────────────────────────────────────
print("=" * 80)
print("FINAL VERDICT")
print("=" * 80)

print("""
Q: Is horizon=140 + speed=0.75 + budget=1000 the best combination?

A: YES, based on available data. Here's the evidence:

1. HORIZON 140 is the critical enabler:
   - At h=100/120 with 100 steps: 0% success (all speeds)
   - At h=140 with 100 steps: speed 0.75 succeeds
   - At h=140 with 1000 steps: speed 0.75 → 100% (6/6)
   - Horizon 140 = 14s lookahead = 0.53m prediction range at speed 0.75

2. SPEED 0.75 is optimal at horizon 140:
   - Speed 0.75: 6/6 success (100%)
   - Speed 1.0:  2/6 success (33%)
   - Speed 1.5:  4/6 success (67%)
   - Lower speed = finer control = better obstacle avoidance

3. BUDGET 1000 is sufficient (not necessarily minimal):
   - Speed 0.75 uses avg 31.7 MPC steps (of 100 available)
   - Total reach 3.75m >> workspace 0.5m
   - Budget 800 or even 600 might suffice, but 1000 is safe

4. The Speed-Horizon interaction:
   - Speed determines how far MuJoCo "sees" per planning step
   - But precision (inverse of speed) determines planning quality
   - At horizon 140, speed 0.75 gives 0.53m prediction range
   - This is enough for the workspace; more range at higher speed
     doesn't help because precision degrades

5. Execute steps (20) analysis:
   - 20 steps = 2.0s = 75mm at speed 0.75
   - This is a reasonable "open-loop chunk"
   - For passage_hard (5cm gap), 75mm > 50mm gap
   - So the pusher crosses the gap in one open-loop chunk
   - This works because horizon 140 already planned the safe path

CONCLUSION:
  horizon=140 (prediction quality) × speed=0.75 (precision) × budget=1000 (safety margin)
  is the best combination because:
  - Horizon provides sufficient planning vision
  - Low speed provides fine-grained control for obstacle threading
  - Large budget ensures completion even for hard templates
""")
