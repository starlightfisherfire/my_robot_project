#!/usr/bin/env python3
"""
MuJoCo reset-only scene inspector.

Resets MujocoPushEnv from a template once, renders a screenshot,
and prints all body/geom names — especially obstacle-related ones.

DO NOT run this script automatically. Run it manually to inspect
whether obstacle geoms are present in the MuJoCo model after reset.

Usage:
  PYTHONPATH=. python scripts/preview_mujoco_reset_scene.py \
    --split test_sim_layout_ood_blocking \
    --template-index 0 \
    --out runs/debug/template_previews/blocking_000000_mujoco_reset.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MuJoCo reset-only scene inspector (no step, no MPC)",
    )
    parser.add_argument(
        "--templates",
        type=str,
        default="data/sim/metadata/reset_templates_v0.json",
        help="Path to reset template JSON file.",
    )
    parser.add_argument("--split", type=str, default=None,
                        help="Filter by split name.")
    parser.add_argument("--template-index", type=int, default=0,
                        help="Index within filtered split (0-based).")
    parser.add_argument("--reset-template-id", type=str, default=None,
                        help="Select by reset_template_id (overrides --split/--template-index).")
    parser.add_argument("--out", type=str, default=None,
                        help="Output screenshot path (.png). If omitted, no image is saved.")
    parser.add_argument("--width", type=int, default=1280,
                        help="Screenshot width in pixels.")
    parser.add_argument("--height", type=int, default=720,
                        help="Screenshot height in pixels.")
    parser.add_argument("--camera", type=str, default="topdown",
                        help="Camera name for rendering (default: topdown). Falls back to camera=-1 if not found.")
    args = parser.parse_args()

    # --- load templates ---
    template_path = Path(args.templates)
    if not template_path.exists():
        print(f"ERROR: Template file not found: {template_path}")
        sys.exit(1)

    from src.interventions.reset_template_loader import load_reset_templates
    all_templates = load_reset_templates(template_path)
    print(f"Loaded {len(all_templates)} templates from {template_path}")

    # --- select template ---
    if args.reset_template_id:
        matches = [t for t in all_templates
                   if t.get("reset_template_id") == args.reset_template_id]
        if not matches:
            print(f"ERROR: No template with reset_template_id={args.reset_template_id}")
            sys.exit(1)
        template = matches[0]
    elif args.split:
        filtered = [t for t in all_templates if t.get("split") == args.split]
        if not filtered:
            print(f"ERROR: No templates for split={args.split}")
            sys.exit(1)
        idx = args.template_index
        if idx < 0 or idx >= len(filtered):
            print(f"ERROR: template-index {idx} out of range [0, {len(filtered)-1}]")
            sys.exit(1)
        template = filtered[idx]
    else:
        idx = args.template_index
        if idx < 0 or idx >= len(all_templates):
            print(f"ERROR: template-index {idx} out of range [0, {len(all_templates)-1}]")
            sys.exit(1)
        template = all_templates[idx]

    template_id = template.get("reset_template_id", "unknown")
    print(f"\nSelected template: {template_id}")
    print(f"  split:         {template.get('split', '?')}")
    print(f"  layout_family: {template.get('layout_family', '?')}")
    print(f"  shape_family:  {template.get('shape_family', '?')}")
    obstacles = template.get("obstacles", [])
    print(f"  obstacles in template metadata: {len(obstacles)}")
    for obs in obstacles:
        print(f"    obstacle_id={obs.get('obstacle_id', '?')}  "
              f"pose={obs.get('pose', {})}  "
              f"size_x={obs.get('size_x', '?')}  size_y={obs.get('size_y', '?')}")

    # --- init env ---
    shape_family = template.get("shape_family", "T_shape")
    shape_type_map = {
        "T_shape": "T", "L_shape": "L", "cross_shape": "cross",
        "bar_shape": "bar", "square_shape": "square", "cylinder_shape": "cylinder",
    }
    shape_type = shape_type_map.get(shape_family, "T")

    print(f"\nInitializing MujocoPushEnv with shape_type={shape_type} ...")
    from src.envs.mujoco_push_env import MujocoPushEnv
    env = MujocoPushEnv(shape_type=shape_type)

    print("Resetting from template (no step, no MPC) ...")
    env.reset_from_template(template)

    import mujoco
    model = env.model
    print(f"\n=== MuJoCo model: {model.nbody} bodies, {model.ngeom} geoms ===")

    print("\nAll body names:")
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"  body[{i}]: {name}")

    print("\nAll geom names:")
    obstacle_geom_ids = []
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        print(f"  geom[{i}]: {name}")
        if name and any(kw in name.lower()
                        for kw in ("obstacle", "obs", "block", "wall", "barrier")):
            obstacle_geom_ids.append(i)

    if obstacle_geom_ids:
        print(f"\nObstacle-related geoms ({len(obstacle_geom_ids)}):")
        active_obstacle_count = 0
        for i in obstacle_geom_ids:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            pos = model.geom_pos[i]
            xpos = env.data.geom_xpos[i]
            size = model.geom_size[i]
            rgba = model.geom_rgba[i]
            contype = int(model.geom_contype[i])
            conaffinity = int(model.geom_conaffinity[i])
            geom_type = model.geom_type[i]
            if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                z_half = size[2]
            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                z_half = size[1]
            else:
                z_half = size[2] if len(size) > 2 else 0.0
            z_min = xpos[2] - z_half
            z_max = xpos[2] + z_half
            active = contype > 0 and rgba[3] > 0.01
            if active:
                active_obstacle_count += 1
            print(f"  {name}:")
            print(f"    geom_pos(local)={pos}  geom_xpos(world)={xpos}")
            print(f"    geom_size={size}  rgba={rgba}")
            print(f"    contype={contype}  conaffinity={conaffinity}")
            print(f"    z_range=[{z_min:.6f}, {z_max:.6f}]")
            print(f"    active={active}")
        print(f"\n  Active obstacle count: {active_obstacle_count}")
        # Validate: template has obstacles but none are active
        if obstacles and active_obstacle_count == 0:
            print("\n  *** ERROR: Template defines obstacles but NO obstacle geom is active after reset! ***")
            print("  *** Obstacle activation may be broken. Check mj_setConst / model field updates. ***")
    else:
        print("\nWARNING: No obstacle geom/body found in MuJoCo model after reset.")
        print("  This confirms that obstacles from reset templates are NOT instantiated.")
        print("  MujocoPushEnv.reset_from_template() ignores template['obstacles'].")

    # --- Key geom z ranges ---
    print(f"\n--- Key geom z ranges ---")
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name and any(kw in name for kw in ("pusher_geom", "object_geom", "obstacle_geom")):
            xpos = env.data.geom_xpos[i]
            size = model.geom_size[i]
            geom_type = model.geom_type[i]
            if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                z_half = size[2]
            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                z_half = size[1]
            else:
                z_half = size[2] if len(size) > 2 else 0.0
            z_min = xpos[2] - z_half
            z_max = xpos[2] + z_half
            contype = int(model.geom_contype[i])
            conaffinity = int(model.geom_conaffinity[i])
            print(f"  {name}: z=[{z_min:.6f}, {z_max:.6f}]  contype={contype}  conaffinity={conaffinity}")

    # --- Contact info ---
    import mujoco as _mj
    _mj.mj_forward(model, env.data)
    print(f"\n--- Contacts after reset ---")
    print(f"  data.ncon = {env.data.ncon}")
    for i in range(env.data.ncon):
        con = env.data.contact[i]
        g1 = _mj.mj_id2name(model, _mj.mjtObj.mjOBJ_GEOM, con.geom1)
        g2 = _mj.mj_id2name(model, _mj.mjtObj.mjOBJ_GEOM, con.geom2)
        print(f"    contact[{i}]: {g1} <-> {g2}  dist={con.dist:.6f}")

    print(f"\nmodel.stat.center = {model.stat.center}")
    print(f"model.stat.extent = {model.stat.extent:.4f}")

    if args.out:
        try:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            renderer = mujoco.Renderer(model, height=args.height, width=args.width)
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
            if cam_id >= 0:
                renderer.update_scene(env.data, camera=cam_id)
            else:
                print(f"WARNING: Camera '{args.camera}' not found, falling back to camera=-1")
                renderer.update_scene(env.data, camera=-1)
            pixels = renderer.render()
            renderer.close()
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(args.width / 100, args.height / 100))
            ax.imshow(pixels)
            ax.axis("off")
            ax.set_title(f"MuJoCo reset: {template_id}", fontsize=8)
            fig.tight_layout()
            fig.savefig(out_path, dpi=100)
            plt.close(fig)
            print(f"\nScreenshot saved: {out_path}")
        except Exception as e:
            print(f"\nWARNING: Could not render screenshot: {e}")
    else:
        print("\n(No --out specified, skipping screenshot render)")

    print("\nDone. No step() was called. No MPC was run.")


if __name__ == "__main__":
    main()
