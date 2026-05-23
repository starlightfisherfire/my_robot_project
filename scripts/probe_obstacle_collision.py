#!/usr/bin/env python3
"""
Minimal physics collision probe for MuJoCo obstacles.

Verifies whether obstacle geoms actually produce contact forces with
the pusher and object. Does NOT run MPC, does NOT render video.

Usage examples:
  # Object-obstacle probe on first blocking template:
  PYTHONPATH=. python scripts/probe_obstacle_collision.py \
    --templates data/sim/metadata/reset_templates_obstacle_sixpack_v0.json \
    --split test_sim_layout_ood_blocking_easy \
    --template-index 0 \
    --probe both

  # Pusher-obstacle probe on passage template:
  PYTHONPATH=. python scripts/probe_obstacle_collision.py \
    --templates data/sim/metadata/reset_templates_obstacle_sixpack_v0.json \
    --split test_sim_layout_ood_passage_direct_medium \
    --template-index 0 \
    --probe pusher_obstacle

  # Dynamic blocking probe:
  PYTHONPATH=. python scripts/probe_obstacle_collision.py \
    --templates data/sim/metadata/reset_templates_obstacle_sixpack_v0.json \
    --split test_sim_layout_ood_blocking_easy \
    --template-index 0 \
    --probe dynamic_blocking
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError:
    print("ERROR: mujoco not installed.")
    sys.exit(1)


def _load_template(args):
    """Load and select a template from the JSON file."""
    template_path = Path(args.templates)
    if not template_path.exists():
        print(f"ERROR: Template file not found: {template_path}")
        sys.exit(1)

    from src.interventions.reset_template_loader import load_reset_templates
    all_templates = load_reset_templates(template_path)

    if args.reset_template_id:
        matches = [t for t in all_templates
                   if t.get("reset_template_id") == args.reset_template_id]
        if not matches:
            print(f"ERROR: No template with reset_template_id={args.reset_template_id}")
            sys.exit(1)
        return matches[0]

    if args.split:
        filtered = [t for t in all_templates if t.get("split") == args.split]
        if not filtered:
            print(f"ERROR: No templates for split={args.split}")
            sys.exit(1)
        idx = args.template_index
        if idx < 0 or idx >= len(filtered):
            print(f"ERROR: template-index {idx} out of range [0, {len(filtered)-1}]")
            sys.exit(1)
        return filtered[idx]

    idx = args.template_index
    if idx < 0 or idx >= len(all_templates):
        print(f"ERROR: template-index {idx} out of range [0, {len(all_templates)-1}]")
        sys.exit(1)
    return all_templates[idx]


def _init_env(template):
    """Initialize MujocoPushEnv from template shape_family."""
    from src.envs.mujoco_push_env import MujocoPushEnv

    shape_family = template.get("shape_family", "T_shape")
    shape_type_map = {
        "T_shape": "T", "L_shape": "L", "cross_shape": "cross",
        "bar_shape": "bar", "square_shape": "square", "cylinder_shape": "cylinder",
    }
    shape_type = shape_type_map.get(shape_family, "T")
    env = MujocoPushEnv(shape_type=shape_type)
    return env


def _find_active_obstacle_geom(env) -> int | None:
    """Find the first active obstacle geom_id (contype > 0, alpha > 0). Returns -1 if none."""
    model = env.model
    data = env.data
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name and "obstacle_geom" in name:
            if int(model.geom_contype[i]) > 0 and float(model.geom_rgba[i][3]) > 0.01:
                return i
    return None


def _get_obstacle_center_from_geom(env, geom_id: int) -> tuple[float, float]:
    """Get obstacle center x/y from actual data.geom_xpos."""
    xpos = env.data.geom_xpos[geom_id]
    return float(xpos[0]), float(xpos[1])


def _get_obstacle_size_from_geom(env, geom_id: int) -> tuple[float, float, float]:
    """Get obstacle half-sizes from model.geom_size."""
    size = env.model.geom_size[geom_id]
    return float(size[0]), float(size[1]), float(size[2])


def _print_geom_info(env, label: str) -> dict:
    """Print detailed geom info and return a dict of geom data."""
    model = env.model
    data = env.data
    result = {}

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Total bodies: {model.nbody}, Total geoms: {model.ngeom}")

    # Collect all geom names
    geom_names = []
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        geom_names.append(name)

    # Identify key geoms
    pusher_geom_ids = [i for i, n in enumerate(geom_names) if n and "pusher" in n]
    object_geom_ids = [i for i, n in enumerate(geom_names) if n and "object_geom" in n]
    obstacle_geom_ids = [i for i, n in enumerate(geom_names) if n and "obstacle_geom" in n]

    def _print_geom(gid, category):
        name = geom_names[gid]
        body_id = int(model.geom_bodyid[gid])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        geom_pos_local = model.geom_pos[gid].copy()
        geom_xpos_world = data.geom_xpos[gid].copy()
        geom_size = model.geom_size[gid].copy()
        rgba = model.geom_rgba[gid].copy()
        contype = int(model.geom_contype[gid])
        conaffinity = int(model.geom_conaffinity[gid])

        geom_type = model.geom_type[gid]
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            z_half = geom_size[2]
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            z_half = geom_size[1]
        else:
            z_half = geom_size[2] if len(geom_size) > 2 else 0.0

        z_min = geom_xpos_world[2] - z_half
        z_max = geom_xpos_world[2] + z_half
        active = contype > 0 and rgba[3] > 0.01

        print(f"  [{category}] {name}:")
        print(f"    geom_id={gid}  body_id={body_id}  body_name={body_name}")
        print(f"    geom_pos(local)={np.array2string(geom_pos_local, precision=6)}")
        print(f"    geom_xpos(world)={np.array2string(geom_xpos_world, precision=6)}")
        print(f"    geom_size={np.array2string(geom_size, precision=6)}")
        print(f"    rgba={np.array2string(rgba, precision=3)}")
        print(f"    contype={contype}  conaffinity={conaffinity}")
        print(f"    z_range=[{z_min:.6f}, {z_max:.6f}]")
        print(f"    active={active}")

        return {
            "name": name, "geom_id": gid, "body_id": body_id,
            "body_name": body_name,
            "geom_pos_local": geom_pos_local.tolist(),
            "geom_xpos_world": geom_xpos_world.tolist(),
            "geom_size": geom_size.tolist(),
            "rgba": rgba.tolist(),
            "contype": contype, "conaffinity": conaffinity,
            "z_range": [z_min, z_max],
            "active": active,
        }

    print(f"\n  --- Pusher geoms ---")
    for gid in pusher_geom_ids:
        result["pusher"] = _print_geom(gid, "PUSHER")

    print(f"\n  --- Object geoms ---")
    for gid in object_geom_ids:
        key = f"object_{geom_names[gid]}"
        result[key] = _print_geom(gid, "OBJECT")

    print(f"\n  --- Obstacle geoms ---")
    active_obstacle_count = 0
    for gid in obstacle_geom_ids:
        key = f"obstacle_{geom_names[gid]}"
        info = _print_geom(gid, "OBSTACLE")
        result[key] = info
        if info["active"]:
            active_obstacle_count += 1

    print(f"\n  Active obstacle count: {active_obstacle_count}")

    # Print obstacle body info
    print(f"\n  --- Obstacle body info ---")
    for i in range(env.max_obstacles):
        body_id = env._obstacle_body_ids[i]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        body_pos = model.body_pos[body_id].copy()
        body_quat = model.body_quat[body_id].copy()
        xpos = data.xpos[body_id].copy()
        xquat = data.xquat[body_id].copy()
        print(f"  obstacle_{i} ({body_name}):")
        print(f"    model.body_pos={np.array2string(body_pos, precision=6)}")
        print(f"    data.xpos={np.array2string(xpos, precision=6)}")
        print(f"    model.body_quat={np.array2string(body_quat, precision=6)}")
        print(f"    data.xquat={np.array2string(xquat, precision=6)}")

    # Print current contacts
    _print_contacts(env, "After reset")

    return result


def _print_contacts(env, label: str):
    """Print current contact information."""
    mujoco.mj_forward(env.model, env.data)
    ncon = env.data.ncon
    print(f"\n  --- Contacts ({label}) ---")
    print(f"  data.ncon = {ncon}")
    for i in range(ncon):
        con = env.data.contact[i]
        g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
        g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)
        dist = con.dist
        print(f"    contact[{i}]: {g1} <-> {g2}  dist={dist:.6f}")


def _probe_object_obstacle(env, template) -> dict:
    """Place object directly on top of obstacle and check for contacts."""
    print(f"\n{'#'*60}")
    print(f"  OBJECT-OBSTACLE PROBE")
    print(f"{'#'*60}")

    # Reset from template (activates obstacles)
    env.reset_from_template(template)

    # Find first active obstacle geom
    obs_geom_id = _find_active_obstacle_geom(env)
    if obs_geom_id is None:
        print("  ERROR: No active obstacle geom found after reset_from_template.")
        return {"object_obstacle_contact_detected": False, "reason": "no_active_obstacle_after_reset"}

    obs_x, obs_y = _get_obstacle_center_from_geom(env, obs_geom_id)
    obs_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, obs_geom_id)
    obs_hx, obs_hy, obs_hz = _get_obstacle_size_from_geom(env, obs_geom_id)

    print(f"  Target obstacle: {obs_name} (geom_id={obs_geom_id})")
    print(f"    center=({obs_x:.4f}, {obs_y:.4f})")
    print(f"    half_size=({obs_hx:.4f}, {obs_hy:.4f}, {obs_hz:.4f})")

    # Set object qpos directly to obstacle center
    obj_adr = env.object_qpos_adr
    env.data.qpos[obj_adr] = obs_x
    env.data.qpos[obj_adr + 1] = obs_y
    # z stays at 0.006 (planar convention), yaw stays at 0

    mujoco.mj_forward(env.model, env.data)
    _print_contacts(env, "object on obstacle center")

    # Check if any contact involves object and obstacle
    object_obstacle_contact = False
    contact_pairs = []
    for i in range(env.data.ncon):
        con = env.data.contact[i]
        g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
        g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)
        names = {g1, g2}
        has_object = any("object_geom" in n for n in names if n)
        has_obstacle = any("obstacle_geom" in n for n in names if n)
        if has_object and has_obstacle:
            object_obstacle_contact = True
            contact_pairs.append(f"{g1} <-> {g2}")

    print(f"\n  RESULT: object_obstacle_contact_detected = {object_obstacle_contact}")
    if contact_pairs:
        for p in contact_pairs:
            print(f"    pair: {p}")

    return {
        "object_obstacle_contact_detected": object_obstacle_contact,
        "object_pos": [obs_x, obs_y],
        "obstacle_center": [obs_x, obs_y],
        "contact_pairs": contact_pairs,
    }


def _probe_pusher_obstacle(env, template) -> dict:
    """Place pusher directly on top of obstacle and check for contacts."""
    print(f"\n{'#'*60}")
    print(f"  PUSHER-OBSTACLE PROBE")
    print(f"{'#'*60}")

    # Reset from template (activates obstacles)
    env.reset_from_template(template)

    # Find first active obstacle geom
    obs_geom_id = _find_active_obstacle_geom(env)
    if obs_geom_id is None:
        print("  ERROR: No active obstacle geom found after reset_from_template.")
        return {"pusher_obstacle_contact_detected": False, "reason": "no_active_obstacle_after_reset"}

    obs_x, obs_y = _get_obstacle_center_from_geom(env, obs_geom_id)
    obs_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, obs_geom_id)

    print(f"  Target obstacle: {obs_name} (geom_id={obs_geom_id})")
    print(f"    center=({obs_x:.4f}, {obs_y:.4f})")

    # Move pusher to obstacle center
    env.data.qpos[env.pusher_x_qpos_adr] = obs_x
    env.data.qpos[env.pusher_y_qpos_adr] = obs_y

    # Move object far away to avoid pusher-object contact
    obj_adr = env.object_qpos_adr
    env.data.qpos[obj_adr] = 0.10
    env.data.qpos[obj_adr + 1] = 0.10

    mujoco.mj_forward(env.model, env.data)
    _print_contacts(env, "pusher on obstacle center")

    # Check if any contact involves pusher and obstacle
    pusher_obstacle_contact = False
    contact_pairs = []
    for i in range(env.data.ncon):
        con = env.data.contact[i]
        g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
        g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)
        names = {g1, g2}
        has_pusher = "pusher_geom" in names
        has_obstacle = any("obstacle_geom" in n for n in names if n)
        if has_pusher and has_obstacle:
            pusher_obstacle_contact = True
            contact_pairs.append(f"{g1} <-> {g2}")

    print(f"\n  RESULT: pusher_obstacle_contact_detected = {pusher_obstacle_contact}")
    if contact_pairs:
        for p in contact_pairs:
            print(f"    pair: {p}")

    return {
        "pusher_obstacle_contact_detected": pusher_obstacle_contact,
        "pusher_pos": [obs_x, obs_y],
        "obstacle_center": [obs_x, obs_y],
        "contact_pairs": contact_pairs,
    }


def _probe_dynamic_blocking(env, template) -> dict:
    """Push pusher toward obstacle without object, check if blocked."""
    print(f"\n{'#'*60}")
    print(f"  DYNAMIC BLOCKING PROBE (pusher-only)")
    print(f"{'#'*60}")

    # Reset from template (activates obstacles)
    env.reset_from_template(template)

    # Find first active obstacle geom
    obs_geom_id = _find_active_obstacle_geom(env)
    if obs_geom_id is None:
        print("  ERROR: No active obstacle geom found after reset_from_template.")
        return {"pusher_blocked": False, "reason": "no_active_obstacle_after_reset"}

    obs_x, obs_y = _get_obstacle_center_from_geom(env, obs_geom_id)
    obs_hx, obs_hy, obs_hz = _get_obstacle_size_from_geom(env, obs_geom_id)
    obs_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, obs_geom_id)

    print(f"  Target obstacle: {obs_name} (geom_id={obs_geom_id})")
    print(f"    center=({obs_x:.4f}, {obs_y:.4f})")
    print(f"    half_size=({obs_hx:.4f}, {obs_hy:.4f}, {obs_hz:.4f})")

    # Move object far away
    obj_adr = env.object_qpos_adr
    env.data.qpos[obj_adr] = 0.05
    env.data.qpos[obj_adr + 1] = 0.05

    # Place pusher to the left of obstacle
    start_x = obs_x - 0.08
    start_y = obs_y
    env.data.qpos[env.pusher_x_qpos_adr] = start_x
    env.data.qpos[env.pusher_y_qpos_adr] = start_y

    mujoco.mj_forward(env.model, env.data)
    print(f"  Pusher start: ({start_x:.4f}, {start_y:.4f})")
    print(f"  Pushing +x toward obstacle at ({obs_x:.4f}, {obs_y:.4f})")

    # Push rightward for 50 steps
    n_steps = 50
    contact_count = 0
    for step_i in range(n_steps):
        state = env.step([1.0, 0.0])  # full speed +x
        if env.last_collision:
            contact_count += 1

    final_x = env.data.xpos[env.pusher_body_id][0]
    final_y = env.data.xpos[env.pusher_body_id][1]

    # Check if pusher crossed obstacle center
    crossed = final_x > obs_x

    # Obstacle AABB from actual geom
    obs_x_min = obs_x - obs_hx
    obs_x_max = obs_x + obs_hx
    obs_y_min = obs_y - obs_hy
    obs_y_max = obs_y + obs_hy

    print(f"\n  Pusher final: ({final_x:.4f}, {final_y:.4f})")
    print(f"  Obstacle AABB: x=[{obs_x_min:.4f}, {obs_x_max:.4f}] y=[{obs_y_min:.4f}, {obs_y_max:.4f}]")
    print(f"  Pusher crossed obstacle center x? {crossed}")
    print(f"  Contact count (via last_collision): {contact_count}")

    # Also check raw contacts on final state
    _print_contacts(env, "after dynamic probe")

    # Check for pusher-obstacle contacts
    pusher_obs_contacts = 0
    for i in range(env.data.ncon):
        con = env.data.contact[i]
        g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
        g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)
        names = {g1, g2}
        has_pusher = "pusher_geom" in names
        has_obstacle = any("obstacle_geom" in n for n in names if n)
        if has_pusher and has_obstacle:
            pusher_obs_contacts += 1

    blocked = not crossed
    print(f"\n  RESULT: pusher blocked by obstacle? {blocked}")
    print(f"    pusher_obstacle_raw_contacts = {pusher_obs_contacts}")

    return {
        "pusher_blocked": blocked,
        "pusher_start": [start_x, start_y],
        "pusher_final": [final_x, final_y],
        "obstacle_center": [obs_x, obs_y],
        "obstacle_aabb": {
            "x_min": obs_x_min, "x_max": obs_x_max,
            "y_min": obs_y_min, "y_max": obs_y_max,
        },
        "pusher_crossed_obstacle_center": crossed,
        "contact_count_last_collision": contact_count,
        "pusher_obstacle_raw_contacts": pusher_obs_contacts,
    }


def _probe_dynamic_object_blocking(env, template) -> dict:
    """Push object toward obstacle, check if object passes through."""
    print(f"\n{'#'*60}")
    print(f"  DYNAMIC OBJECT BLOCKING PROBE")
    print(f"{'#'*60}")

    # Reset from template (activates obstacles)
    env.reset_from_template(template)

    # Find first active obstacle geom
    obs_geom_id = _find_active_obstacle_geom(env)
    if obs_geom_id is None:
        print("  ERROR: No active obstacle geom found after reset_from_template.")
        return {"object_passed_through_obstacle": False, "reason": "no_active_obstacle_after_reset"}

    obs_x, obs_y = _get_obstacle_center_from_geom(env, obs_geom_id)
    obs_hx, obs_hy, obs_hz = _get_obstacle_size_from_geom(env, obs_geom_id)
    obs_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, obs_geom_id)

    print(f"  Target obstacle: {obs_name} (geom_id={obs_geom_id})")
    print(f"    center=({obs_x:.4f}, {obs_y:.4f})")
    print(f"    half_size=({obs_hx:.4f}, {obs_hy:.4f}, {obs_hz:.4f})")

    # Place object to the left of obstacle, pusher behind object
    obj_start_x = obs_x - 0.06
    obj_start_y = obs_y
    env.data.qpos[env.object_qpos_adr] = obj_start_x
    env.data.qpos[env.object_qpos_adr + 1] = obj_start_y

    pusher_start_x = obj_start_x - 0.04
    env.data.qpos[env.pusher_x_qpos_adr] = pusher_start_x
    env.data.qpos[env.pusher_y_qpos_adr] = obj_start_y

    mujoco.mj_forward(env.model, env.data)
    print(f"  Object start: ({obj_start_x:.4f}, {obj_start_y:.4f})")
    print(f"  Pusher start: ({pusher_start_x:.4f}, {obj_start_y:.4f})")
    print(f"  Pushing +x toward obstacle at ({obs_x:.4f}, {obs_y:.4f})")

    n_steps = 80
    object_passed = False
    collision_count = 0
    contact_count = 0

    for step_i in range(n_steps):
        state = env.step([1.0, 0.0])
        if env.last_collision:
            collision_count += 1
        if env.last_contact:
            contact_count += 1

        obj_x = env.data.xpos[env.object_body_id][0]
        obs_x_max = obs_x + obs_hx
        if obj_x > obs_x_max + 0.01:
            object_passed = True
            print(f"  Object passed obstacle at step {step_i}, obj_x={obj_x:.4f}")
            break

    final_obj_x = env.data.xpos[env.object_body_id][0]
    final_obj_y = env.data.xpos[env.object_body_id][1]

    obs_x_min = obs_x - obs_hx
    obs_x_max = obs_x + obs_hx

    print(f"\n  Object final: ({final_obj_x:.4f}, {final_obj_y:.4f})")
    print(f"  Obstacle AABB: x=[{obs_x_min:.4f}, {obs_x_max:.4f}]")
    print(f"  Object passed through obstacle? {object_passed}")
    print(f"  Object-obstacle collision count: {collision_count}")
    print(f"  Pusher-object contact count: {contact_count}")

    _print_contacts(env, "after object blocking probe")

    return {
        "object_passed_through_obstacle": object_passed,
        "object_start": [obj_start_x, obj_start_y],
        "object_final": [final_obj_x, final_obj_y],
        "obstacle_center": [obs_x, obs_y],
        "object_obstacle_collision_count": collision_count,
        "pusher_object_contact_count": contact_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal physics collision probe for MuJoCo obstacles (no MPC, no video).",
    )
    parser.add_argument("--templates", type=str,
                        default="data/sim/metadata/reset_templates_obstacle_sixpack_v0.json")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--template-index", type=int, default=0)
    parser.add_argument("--reset-template-id", type=str, default=None)
    parser.add_argument("--probe", type=str, default="both",
                        choices=["object_obstacle", "pusher_obstacle",
                                 "dynamic_blocking", "dynamic_object",
                                 "both", "all"],
                        help="Which probe to run.")
    parser.add_argument("--out-json", type=str, default=None,
                        help="Output JSON path for probe results.")
    args = parser.parse_args()

    template = _load_template(args)
    template_id = template.get("reset_template_id", "unknown")
    print(f"Template: {template_id}")
    print(f"  split: {template.get('split', '?')}")
    print(f"  layout_family: {template.get('layout_family', '?')}")
    print(f"  shape_family: {template.get('shape_family', '?')}")
    print(f"  obstacles: {len(template.get('obstacles', []))}")

    env = _init_env(template)

    # Reset from template FIRST to activate obstacles, THEN print geom info
    env.reset_from_template(template)
    geom_info = _print_geom_info(env, "After reset_from_template")

    # Validate: if template has obstacles but none are active, flag error
    template_obstacles = template.get("obstacles", [])
    if template_obstacles:
        active_obs = _find_active_obstacle_geom(env)
        if active_obs is None:
            print("\n  *** ERROR: Template has obstacles but NO active obstacle geom after reset! ***")
            print("  *** Obstacle activation is broken. Check compile-time XML building. ***")

    # Validate: pusher initial position matches template ee_initial_pose
    ee_x = template["ee_initial_pose"]["x"]
    ee_y = template["ee_initial_pose"]["y"]
    pusher_xpos = env.data.xpos[env.pusher_body_id]
    dx = abs(float(pusher_xpos[0]) - ee_x)
    dy = abs(float(pusher_xpos[1]) - ee_y)
    print(f"\n  --- Pusher initial position check ---")
    print(f"    template ee_initial_pose: ({ee_x:.6f}, {ee_y:.6f})")
    print(f"    pusher geom_xpos:         ({pusher_xpos[0]:.6f}, {pusher_xpos[1]:.6f})")
    print(f"    delta: dx={dx:.6f}, dy={dy:.6f}")
    if dx > 0.001 or dy > 0.001:
        print("    *** WARNING: pusher position does not match ee_initial_pose! ***")
    else:
        print("    OK: pusher position matches ee_initial_pose")

    results = {"template_id": template_id, "geom_info": geom_info}

    probe = args.probe

    if probe in ("object_obstacle", "both", "all"):
        results["object_obstacle_probe"] = _probe_object_obstacle(env, template)

    if probe in ("pusher_obstacle", "both", "all"):
        results["pusher_obstacle_probe"] = _probe_pusher_obstacle(env, template)

    if probe in ("dynamic_blocking", "all"):
        results["dynamic_blocking_probe"] = _probe_dynamic_blocking(env, template)

    if probe in ("dynamic_object", "all"):
        results["dynamic_object_probe"] = _probe_dynamic_object_blocking(env, template)

    # Summary
    print(f"\n{'='*60}")
    print(f"  PROBE SUMMARY")
    print(f"{'='*60}")
    if "object_obstacle_probe" in results:
        r = results["object_obstacle_probe"]
        print(f"  object-obstacle contact detected: {r.get('object_obstacle_contact_detected', '?')}")
    if "pusher_obstacle_probe" in results:
        r = results["pusher_obstacle_probe"]
        print(f"  pusher-obstacle contact detected: {r.get('pusher_obstacle_contact_detected', '?')}")
    if "dynamic_blocking_probe" in results:
        r = results["dynamic_blocking_probe"]
        print(f"  pusher blocked by obstacle: {r.get('pusher_blocked', '?')}")
    if "dynamic_object_probe" in results:
        r = results["dynamic_object_probe"]
        print(f"  object passed through obstacle: {r.get('object_passed_through_obstacle', '?')}")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {out_path}")

    print("\nDone. No MPC was run. No video was rendered.")


if __name__ == "__main__":
    main()
