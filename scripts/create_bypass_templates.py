#!/usr/bin/env python3
"""Create bypass double-obstacle templates.
Design: obs[0] blocks straight object→goal path; pusher must detour through gap.
Difficulty: wide=10cm, medium=8cm, narrow=6cm effective vertical gap.
Three templates → passage_bypass_wide / _medium / _narrow
"""
import json
from pathlib import Path

PROJECT = Path("/home/brucewu/my_robot_project")

def make_bypass(family, split, obs0_y, obs1_y, effective_gap):
    """obs0_y=top obstacle center Y, obs1_y=bottom obstacle center Y"""
    obs_half = 0.030  # half of size_y=0.06
    o0_bottom = obs0_y - obs_half
    o1_top = obs1_y + obs_half
    gap_center_y = (o0_bottom + o1_top) / 2
    passage_gap = o0_bottom - o1_top  # true inner gap
    
    return {
        "schema_version": "reset_template_v0.1",
        "reset_template_id": f"{split}__{family}__T_shape__000000",
        "domain": "sim",
        "split": split,
        "layout_family": family,
        "shape_family": "T_shape",
        "seed": 0,
        "object_shape": "T",
        "object_size_x": 0.048,
        "object_size_y": 0.048,
        "object_mass": 0.01905,
        "object_friction": 1.0,
        "object_initial_pose": {"x": 0.180, "y": 0.220, "theta": 0.0},
        "goal_pose": {"x": 0.480, "y": 0.220, "theta": 0.0},
        "ee_initial_pose": {"x": 0.080, "y": 0.220, "theta": 0.0},
        "obstacles": [
            {
                "obstacle_id": f"obs_{family}_top",
                "pose": {"x": 0.330, "y": obs0_y, "theta": 0.0},
                "size_x": 0.080, "size_y": 0.060,
                "valid": True,
            },
            {
                "obstacle_id": f"obs_{family}_bottom",
                "pose": {"x": 0.330, "y": obs1_y, "theta": 0.0},
                "size_x": 0.080, "size_y": 0.060,
                "valid": True,
            },
        ],
        "obstacle_size_x": 0.080,
        "obstacle_size_y": 0.060,
        "passage_gap": round(passage_gap, 4),
        "effective_passage_gap": round(effective_gap, 4),
        "passage_center_distance": round(obs0_y - obs1_y, 4),
        "passage_gap_definition": "vertical_edge_to_edge_bypass",
        "passage_center": {"x": 0.330, "y": round(gap_center_y, 4)},
    }

# Design: Object at y=0.22, obs_top blocks at y=0.255
# obs_top bottom = 0.255 - 0.03 = 0.225 → blocks obj_y=0.22
# gap = obs_top_bottom - obs_bottom_top
templates = {
    "passage_bypass_wide": make_bypass(
        "passage_bypass_wide",
        "test_sim_layout_ood_passage_bypass_wide",
        obs0_y=0.255, obs1_y=0.095,
        effective_gap=0.080,  # 8cm effective
    ),
    "passage_bypass_medium": make_bypass(
        "passage_bypass_medium",
        "test_sim_layout_ood_passage_bypass_medium",
        obs0_y=0.255, obs1_y=0.115,
        effective_gap=0.060,  # 6cm effective
    ),
    "passage_bypass_narrow": make_bypass(
        "passage_bypass_narrow",
        "test_sim_layout_ood_passage_bypass_narrow",
        obs0_y=0.255, obs1_y=0.135,
        effective_gap=0.040,  # 4cm effective
    ),
}

# --- 1. Save standalone bypass JSON ---
bypass_path = PROJECT / "data/sim/metadata/reset_templates_bypass_v0.json"
bypass_list = list(templates.values())
with open(bypass_path, "w") as f:
    json.dump(bypass_list, f, indent=2, ensure_ascii=False)
print(f"✅ Bypass templates: {bypass_path} ({len(bypass_list)} templates)")

# --- 2. Update sixpack (replace bypass slots) ---
sp = PROJECT / "data/sim/metadata/reset_templates_obstacle_sixpack_v0.json"
with open(sp) as f:
    sixpack = json.load(f)

# Replace bypass templates in slots 3-5 (or append if missing)
bypass_order = ["passage_bypass_wide", "passage_bypass_medium", "passage_bypass_narrow"]
bypass_count = 0
for i, t in enumerate(sixpack):
    if "bypass" in t.get("layout_family", ""):
        if bypass_count < len(bypass_order):
            sixpack[i] = templates[bypass_order[bypass_count]]
            bypass_count += 1

with open(sp, "w") as f:
    json.dump(sixpack, f, indent=2, ensure_ascii=False)

# --- 3. Print geometry ---
print()
for name, t in templates.items():
    o0, o1 = t["obstacles"][0], t["obstacles"][1]
    o0_bot = o0["pose"]["y"] - o0["size_y"] / 2
    o1_top = o1["pose"]["y"] + o1["size_y"] / 2
    gap = o0_bot - o1_top
    blocked = o0_bot <= 0.22 <= o0["pose"]["y"] + o0["size_y"] / 2
    print(f"  {name}: blocked={blocked}, gap={gap*100:.1f}cm, effective={t['effective_passage_gap']*100:.0f}cm")

print(f"\n✅ Sixpack updated: {sp}")
for i, t in enumerate(sixpack):
    print(f"  [{i}] {t['layout_family']}")
