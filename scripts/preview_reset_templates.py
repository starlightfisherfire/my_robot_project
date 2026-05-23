#!/usr/bin/env python3
"""
Static 2D top-down preview of reset template layouts.

Usage:
  # Batch grid preview
  PYTHONPATH=. python scripts/preview_reset_templates.py \
    --split train_sim_id --max-templates 12

  # Single template preview
  PYTHONPATH=. python scripts/preview_reset_templates.py \
    --split train_sim_id --template-index 0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# shape drawing helpers (no MuJoCo import, no ObjectShapeFactory)
# ---------------------------------------------------------------------------

def _load_object_specs(config_path: str | Path | None = None) -> dict:
    """Load object_specs.yaml, returning the inner versioned dict."""
    if config_path is None:
        repo_root = Path(__file__).resolve().parent.parent
        config_path = repo_root / "configs" / "object_specs.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Object specs config not found: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    version_key = list(cfg.keys())[0]
    return cfg[version_key]


# Cache specs at module level so we only parse YAML once.
_SPECS: dict | None = None


def _get_specs() -> dict:
    global _SPECS
    if _SPECS is None:
        _SPECS = _load_object_specs()
    return _SPECS


def _rotated_rect_patch(
    rect_cx: float,
    rect_cy: float,
    half_w: float,
    half_h: float,
    theta: float,
    facecolor,
    edgecolor="k",
    linewidth=0.5,
) -> mpatches.Polygon:
    """Return a Polygon patch for a rectangle centered at (rect_cx, rect_cy),
    rotated by theta radians around its center."""
    local = np.array([
        [-half_w, -half_h],
        [ half_w, -half_h],
        [ half_w,  half_h],
        [-half_w,  half_h],
    ])
    rot = np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta),  math.cos(theta)],
    ])
    world = local @ rot.T + np.array([rect_cx, rect_cy])
    return mpatches.Polygon(
        world,
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )


def _build_shape_patches(
    shape_type: str,
    cx: float,
    cy: float,
    theta: float,
    base_rgba: tuple[float, ...],
    alpha: float = 1.0,
) -> list[mpatches.Patch]:
    """
    Return a list of matplotlib Patch objects for a shape_type
    positioned at (cx, cy) with rotation theta (radians).

    shape_type: "T" | "L" | "cross" | "bar" | "square" | "cylinder"

    All dimensions are converted from mm → m internally.
    """
    specs = _get_specs()
    thickness_mm = specs["thickness_mm"]  # 12.0 typical

    patches: list[mpatches.Patch] = []

    rgba = base_rgba[:3] + (base_rgba[3] * alpha,) if len(base_rgba) == 4 else base_rgba

    if shape_type in ("T", "L", "cross"):
        mo = specs["manipulated_objects"][shape_type]
        footprint_mm = mo["footprint_mm"]  # [W, H]
        arm_width_mm = mo["arm_width_mm"]
        W = footprint_mm[0] / 1000.0
        H = footprint_mm[1] / 1000.0
        w = arm_width_mm / 1000.0
        _t = thickness_mm / 1000.0  # ignored in 2D

        if shape_type == "T":
            # top bar: W × w, centered at (0, (H-w)/2) in local frame
            # stem:    w × (H-w), centered at (0, -w/2) in local frame
            top_hw = W / 2.0
            top_hh = w / 2.0
            top_ly = (H - w) / 2.0

            stem_hw = w / 2.0
            stem_hh = (H - w) / 2.0
            stem_ly = -w / 2.0

            for (lx, ly, hw, hh) in [
                (0.0, top_ly, top_hw, top_hh),
                (0.0, stem_ly, stem_hw, stem_hh),
            ]:
                # rotate local (lx, ly) to world
                wx = cx + lx * math.cos(theta) - ly * math.sin(theta)
                wy = cy + lx * math.sin(theta) + ly * math.cos(theta)
                patches.append(
                    _rotated_rect_patch(wx, wy, hw, hh, theta,
                                        facecolor=rgba, edgecolor="k", linewidth=0.5))

        elif shape_type == "L":
            # horiz bar: W × w, at y = -(H-w)/2
            # vert bar:  w × (H-w), at x = -(W-w)/2, y = +w/2
            h_hw = W / 2.0
            h_hh = w / 2.0
            h_ly = -(H - w) / 2.0

            v_hw = w / 2.0
            v_hh = (H - w) / 2.0
            v_lx = -(W - w) / 2.0
            v_ly = w / 2.0

            for (lx, ly, hw, hh) in [
                (0.0, h_ly, h_hw, h_hh),
                (v_lx, v_ly, v_hw, v_hh),
            ]:
                wx = cx + lx * math.cos(theta) - ly * math.sin(theta)
                wy = cy + lx * math.sin(theta) + ly * math.cos(theta)
                patches.append(
                    _rotated_rect_patch(wx, wy, hw, hh, theta,
                                        facecolor=rgba, edgecolor="k", linewidth=0.5))

        elif shape_type == "cross":
            # vert bar: w × H, centered at (0, 0)
            # left arm: w × w, at x = -(W-w)/2, y = 0
            # right arm: w × w, at x = +(W-w)/2, y = 0
            vb_hw = w / 2.0
            vb_hh = H / 2.0

            la_hw = w / 2.0
            la_hh = w / 2.0
            la_lx = -(W - w) / 2.0

            ra_hw = w / 2.0
            ra_hh = w / 2.0
            ra_lx = (W - w) / 2.0

            for (lx, ly, hw, hh) in [
                (0.0, 0.0, vb_hw, vb_hh),
                (la_lx, 0.0, la_hw, la_hh),
                (ra_lx, 0.0, ra_hw, ra_hh),
            ]:
                wx = cx + lx * math.cos(theta) - ly * math.sin(theta)
                wy = cy + lx * math.sin(theta) + ly * math.cos(theta)
                patches.append(
                    _rotated_rect_patch(wx, wy, hw, hh, theta,
                                        facecolor=rgba, edgecolor="k", linewidth=0.5))

    elif shape_type == "bar":
        mo = specs["manipulated_objects"]["bar"]
        footprint_mm = mo["footprint_mm"]
        W = footprint_mm[0] / 1000.0
        H = footprint_mm[1] / 1000.0
        patches.append(
            _rotated_rect_patch(cx, cy, W / 2.0, H / 2.0, theta,
                                facecolor=rgba, edgecolor="k", linewidth=0.5))

    elif shape_type == "square":
        obj = specs["obstacles"]["square"]
        size_mm = obj["size_mm"]
        S = size_mm[0] / 1000.0
        patches.append(
            _rotated_rect_patch(cx, cy, S / 2.0, S / 2.0, theta,
                                facecolor=rgba, edgecolor="k", linewidth=0.5))

    elif shape_type == "cylinder":
        obj = specs["obstacles"]["cylinder"]
        radius_m = obj["radius_mm"] / 1000.0
        patches.append(
            Circle((cx, cy), radius_m,
                   facecolor=rgba, edgecolor="k", linewidth=0.5))

    else:
        raise ValueError(f"Unsupported shape_type for drawing: {shape_type}")

    return patches


def _shape_family_to_type(shape_family: str) -> str:
    """Convert shape_family (e.g. 'T_shape') to shape_type (e.g. 'T')."""
    mapping = {
        "T_shape": "T",
        "L_shape": "L",
        "cross_shape": "cross",
        "bar_shape": "bar",
        "square_shape": "square",
        "cylinder_shape": "cylinder",
    }
    if shape_family in mapping:
        return mapping[shape_family]
    # fallback: try stripping _shape suffix
    for suffix in ("_shape",):
        if shape_family.endswith(suffix):
            return shape_family[:-len(suffix)]
    return shape_family


# ---------------------------------------------------------------------------
# robust template field reading
# ---------------------------------------------------------------------------

def _get_pose(template: dict, keys: list[str], pose_name: str) -> dict:
    """Try multiple keys, return the pose dict with x, y, theta."""
    for key in keys:
        if key in template and template[key] is not None:
            pose = template[key]
            if isinstance(pose, dict) and "x" in pose and "y" in pose:
                theta = pose.get("theta", 0.0)
                return {"x": float(pose["x"]), "y": float(pose["y"]), "theta": float(theta)}
    raise KeyError(
        f"Cannot find {pose_name} in template keys: {list(template.keys())}. "
        f"Tried: {keys}"
    )


def _get_pose_optional(template: dict, keys: list[str]) -> dict | None:
    """Try multiple keys, return pose or None."""
    for key in keys:
        if key in template and template[key] is not None:
            pose = template[key]
            if isinstance(pose, dict) and "x" in pose and "y" in pose:
                theta = pose.get("theta", 0.0)
                return {"x": float(pose["x"]), "y": float(pose["y"]), "theta": float(theta)}
    return None


# ---------------------------------------------------------------------------
# single-template rendering
# ---------------------------------------------------------------------------

def render_single_template(
    template: dict,
    pusher_radius: float = 0.010,
    workspace_x: float = 0.70,
    workspace_y: float = 0.50,
) -> plt.Figure:
    """Render one template as a single matplotlib figure."""

    # --- read template fields ---
    obj_pose = _get_pose(
        template,
        ["object_initial_pose", "object_pose", "initial_object_pose"],
        "object pose",
    )
    goal_pose = _get_pose(
        template,
        ["goal_pose", "object_goal_pose"],
        "goal pose",
    )
    ee_pose = _get_pose(
        template,
        ["ee_initial_pose", "ee_pos", "initial_ee_pos", "pusher_initial_pose"],
        "ee pose",
    )

    shape_family = template.get("shape_family", "T_shape")
    shape_type = _shape_family_to_type(shape_family)

    template_id = template.get("reset_template_id", "unknown")
    split = template.get("split", "unknown")
    layout_family = template.get("layout_family", "unknown")

    # --- distances ---
    obj_goal_dist = math.hypot(
        obj_pose["x"] - goal_pose["x"],
        obj_pose["y"] - goal_pose["y"],
    )
    ee_obj_dist = math.hypot(
        ee_pose["x"] - obj_pose["x"],
        ee_pose["y"] - obj_pose["y"],
    )

    # --- create figure ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))
    ax.set_xlim(-0.02, workspace_x + 0.02)
    ax.set_ylim(-0.02, workspace_y + 0.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle="--")

    # workspace boundary
    workspace_rect = Rectangle(
        (0, 0), workspace_x, workspace_y,
        fill=False, edgecolor="gray", linewidth=1.5, linestyle="--",
        label="Workspace",
    )
    ax.add_patch(workspace_rect)

    # --- goal ghost (draw first so it's behind) ---
    goal_patches = _build_shape_patches(
        shape_type,
        goal_pose["x"], goal_pose["y"], goal_pose["theta"],
        base_rgba=(0.1, 0.8, 0.1, 1.0),
        alpha=0.3,
    )
    for p in goal_patches:
        ax.add_patch(p)

    # --- object (red, semi-transparent) ---
    obj_patches = _build_shape_patches(
        shape_type,
        obj_pose["x"], obj_pose["y"], obj_pose["theta"],
        base_rgba=(0.9, 0.2, 0.1, 1.0),
        alpha=0.85,
    )
    for p in obj_patches:
        ax.add_patch(p)

    # --- pusher / EE (blue circle) ---
    pusher = Circle(
        (ee_pose["x"], ee_pose["y"]),
        pusher_radius,
        facecolor=(0.1, 0.2, 0.9, 0.9),
        edgecolor="k",
        linewidth=0.5,
        label="Pusher (EE)",
    )
    ax.add_patch(pusher)

    # --- mark centers ---
    ax.plot(obj_pose["x"], obj_pose["y"], "o", color="red", markersize=4)
    ax.plot(goal_pose["x"], goal_pose["y"], "o", color="green", markersize=4)
    ax.plot(ee_pose["x"], ee_pose["y"], "o", color="blue", markersize=4)

    # --- arrows: ee→object, object→goal ---
    ax.annotate(
        "", xy=(obj_pose["x"], obj_pose["y"]),
        xytext=(ee_pose["x"], ee_pose["y"]),
        arrowprops=dict(arrowstyle="->", color="blue", alpha=0.4, lw=1.0),
    )
    ax.annotate(
        "", xy=(goal_pose["x"], goal_pose["y"]),
        xytext=(obj_pose["x"], obj_pose["y"]),
        arrowprops=dict(arrowstyle="->", color="red", alpha=0.4, lw=1.0),
    )

    # --- obstructions (if any) ---
    obstacles = template.get("obstacles", [])
    for obs in obstacles:
        if not isinstance(obs, dict):
            continue
        obs_pose = obs.get("pose", {})
        ox = float(obs_pose.get("x", 0))
        oy = float(obs_pose.get("y", 0))
        otheta = float(obs_pose.get("theta", 0))
        osx = float(obs.get("size_x", 0.04))
        osy = float(obs.get("size_y", 0.04))
        # draw as gray rectangle
        cos_t = math.cos(otheta)
        sin_t = math.sin(otheta)
        corners = np.array([
            [-osx/2, -osy/2],
            [ osx/2, -osy/2],
            [ osx/2,  osy/2],
            [-osx/2,  osy/2],
        ])
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        world_corners = corners @ rot.T + np.array([ox, oy])
        obs_poly = mpatches.Polygon(
            world_corners,
            facecolor=(0.6, 0.6, 0.6, 0.5),
            edgecolor="gray",
            linewidth=0.5,
        )
        ax.add_patch(obs_poly)

    # --- title and info ---
    title_lines = [
        f"Template: {template_id}",
        f"split={split}  |  layout={layout_family}  |  shape={shape_family}",
        f"obj→goal: {obj_goal_dist:.4f}m  |  ee→obj: {ee_obj_dist:.4f}m",
        f"obj: x={obj_pose['x']:.4f} y={obj_pose['y']:.4f} θ={obj_pose['theta']:.4f}",
        f"goal: x={goal_pose['x']:.4f} y={goal_pose['y']:.4f} θ={goal_pose['theta']:.4f}",
        f"ee: x={ee_pose['x']:.4f} y={ee_pose['y']:.4f}",
    ]
    if obstacles:
        title_lines.append(f"obstacles: {len(obstacles)}")
    ax.set_title("\n".join(title_lines), fontsize=8, family="monospace")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    # legend
    legend_patches = [
        mpatches.Patch(facecolor=(0.1, 0.2, 0.9, 0.9), edgecolor="k", label="Pusher (EE)"),
        mpatches.Patch(facecolor=(0.9, 0.2, 0.1, 0.85), edgecolor="k", label="Object"),
        mpatches.Patch(facecolor=(0.1, 0.8, 0.1, 0.3), edgecolor="k", label="Goal (ghost)"),
    ]
    if obstacles:
        legend_patches.append(
            mpatches.Patch(facecolor=(0.6, 0.6, 0.6, 0.5), edgecolor="gray",
                          label="Obstacle")
        )
    ax.legend(handles=legend_patches, loc="upper right", fontsize=7)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# multi-template grid rendering
# ---------------------------------------------------------------------------

def render_template_grid(
    templates: list[dict],
    pusher_radius: float = 0.010,
    workspace_x: float = 0.70,
    workspace_y: float = 0.50,
    cols: int = 3,
) -> plt.Figure:
    """Render multiple templates as a grid of subplots."""

    n = len(templates)
    cols = min(cols, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(5 * cols, 5 * rows),
        squeeze=False,
    )

    for idx, (template, ax) in enumerate(zip(templates, axes.flat)):
        try:
            obj_pose = _get_pose(
                template,
                ["object_initial_pose", "object_pose", "initial_object_pose"],
                "object pose",
            )
            goal_pose = _get_pose(
                template,
                ["goal_pose", "object_goal_pose"],
                "goal pose",
            )
            ee_pose = _get_pose(
                template,
                ["ee_initial_pose", "ee_pos", "initial_ee_pos", "pusher_initial_pose"],
                "ee pose",
            )
        except KeyError as e:
            ax.text(0.5, 0.5, f"PARSE ERROR\n{e}",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=7, color="red")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_title(f"[{idx}] {template.get('reset_template_id', '?')}", fontsize=7)
            continue

        shape_family = template.get("shape_family", "T_shape")
        shape_type = _shape_family_to_type(shape_family)

        template_id = template.get("reset_template_id", f"idx_{idx}")
        split = template.get("split", "?")
        layout_family = template.get("layout_family", "?")
        obstacles = template.get("obstacles", [])

        obj_goal_dist = math.hypot(
            obj_pose["x"] - goal_pose["x"],
            obj_pose["y"] - goal_pose["y"],
        )
        ee_obj_dist = math.hypot(
            ee_pose["x"] - obj_pose["x"],
            ee_pose["y"] - obj_pose["y"],
        )

        ax.set_xlim(-0.02, workspace_x + 0.02)
        ax.set_ylim(-0.02, workspace_y + 0.02)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2, linestyle="--")

        # workspace
        ax.add_patch(Rectangle(
            (0, 0), workspace_x, workspace_y,
            fill=False, edgecolor="gray", linewidth=1.0, linestyle="--",
        ))

        # goal ghost
        for p in _build_shape_patches(
            shape_type,
            goal_pose["x"], goal_pose["y"], goal_pose["theta"],
            base_rgba=(0.1, 0.8, 0.1, 1.0), alpha=0.3,
        ):
            ax.add_patch(p)

        # object
        for p in _build_shape_patches(
            shape_type,
            obj_pose["x"], obj_pose["y"], obj_pose["theta"],
            base_rgba=(0.9, 0.2, 0.1, 1.0), alpha=0.85,
        ):
            ax.add_patch(p)

        # pusher
        ax.add_patch(Circle(
            (ee_pose["x"], ee_pose["y"]), pusher_radius,
            facecolor=(0.1, 0.2, 0.9, 0.9), edgecolor="k", linewidth=0.3,
        ))

        # center dots
        ax.plot(obj_pose["x"], obj_pose["y"], "o", color="red", markersize=3)
        ax.plot(goal_pose["x"], goal_pose["y"], "o", color="green", markersize=3)
        ax.plot(ee_pose["x"], ee_pose["y"], "o", color="blue", markersize=3)

        # arrows
        ax.annotate(
            "", xy=(obj_pose["x"], obj_pose["y"]),
            xytext=(ee_pose["x"], ee_pose["y"]),
            arrowprops=dict(arrowstyle="->", color="blue", alpha=0.3, lw=0.8),
        )
        ax.annotate(
            "", xy=(goal_pose["x"], goal_pose["y"]),
            xytext=(obj_pose["x"], obj_pose["y"]),
            arrowprops=dict(arrowstyle="->", color="red", alpha=0.3, lw=0.8),
        )

        # obstacles
        for obs in obstacles:
            if not isinstance(obs, dict):
                continue
            obs_pose = obs.get("pose", {})
            ox = float(obs_pose.get("x", 0))
            oy = float(obs_pose.get("y", 0))
            otheta = float(obs_pose.get("theta", 0))
            osx = float(obs.get("size_x", 0.04))
            osy = float(obs.get("size_y", 0.04))
            cos_t = math.cos(otheta)
            sin_t = math.sin(otheta)
            corners = np.array([
                [-osx/2, -osy/2], [osx/2, -osy/2], [osx/2, osy/2], [-osx/2, osy/2],
            ])
            rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            world_corners = corners @ rot.T + np.array([ox, oy])
            ax.add_patch(mpatches.Polygon(
                world_corners,
                facecolor=(0.6, 0.6, 0.6, 0.5),
                edgecolor="gray", linewidth=0.3,
            ))

        # title
        ax.set_title(
            f"#{idx} {template_id}\n"
            f"{split} | {layout_family} | {shape_family}\n"
            f"obj→goal {obj_goal_dist:.3f}m  ee→obj {ee_obj_dist:.3f}m",
            fontsize=6, family="monospace",
        )
        ax.set_xlabel("X (m)", fontsize=5)
        ax.set_ylabel("Y (m)", fontsize=5)
        ax.tick_params(labelsize=5)

    # hide unused subplots
    for j in range(n, len(axes.flat)):
        axes.flat[j].set_visible(False)

    fig.suptitle(
        f"Reset Template Layout Preview — {n} templates",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Static 2D preview of reset template layouts",
    )
    parser.add_argument(
        "--templates",
        type=str,
        default="data/sim/metadata/reset_templates_v0.json",
        help="Path to reset template JSON file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train_sim_id",
        help="Split to filter (e.g. train_sim_id, val_sim_id, test_sim_id, all).",
    )
    parser.add_argument(
        "--max-templates",
        type=int,
        default=12,
        help="Maximum number of templates to render.",
    )
    parser.add_argument(
        "--template-index",
        type=int,
        default=None,
        help="If set, render only this single template index (0-based after filtering).",
    )
    parser.add_argument(
        "--reset-template-id",
        type=str,
        default=None,
        help="If set, select template by reset_template_id (overrides --template-index).",
    )
    parser.add_argument(
        "--layout-family",
        type=str,
        default=None,
        help="If set, filter to this layout_family (e.g. open_space, mild_offset).",
    )
    parser.add_argument(
        "--shape-family",
        type=str,
        default=None,
        help="If set, filter to this shape_family (e.g. T_shape, L_shape).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/template_previews",
        help="Output directory for image files.",
    )
    parser.add_argument(
        "--out-image",
        type=str,
        default=None,
        help="If set, override output image path (single template only).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Alias for --out-image (single template only).",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        default=False,
        help="Print template fields to stdout (single template mode only).",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=3,
        help="Number of columns in grid preview.",
    )
    parser.add_argument(
        "--pusher-radius",
        type=float,
        default=0.010,
        help="Pusher radius in meters (default 0.010).",
    )
    parser.add_argument(
        "--workspace-x",
        type=float,
        default=0.70,
        help="Workspace X size in meters (default 0.70).",
    )
    parser.add_argument(
        "--workspace-y",
        type=float,
        default=0.50,
        help="Workspace Y size in meters (default 0.50).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Output image DPI.",
    )

    args = parser.parse_args()

    # --out is an alias for --out-image
    if args.out and not args.out_image:
        args.out_image = args.out

    # --- load templates ---
    template_path = Path(args.templates)
    if not template_path.exists():
        print(f"ERROR: Template file not found: {template_path}")
        print("Generate it first with:")
        print("  PYTHONPATH=. python scripts/generate_reset_templates.py")
        sys.exit(1)

    try:
        from src.interventions.reset_template_loader import load_reset_templates
        all_templates = load_reset_templates(template_path)
        print(f"Loaded {len(all_templates)} templates via loader from {template_path}")
    except ImportError:
        with open(template_path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            all_templates = data
        elif isinstance(data, dict) and "templates" in data:
            all_templates = data["templates"]
        else:
            all_templates = data  # fallback
        print(f"Loaded {len(all_templates)} templates directly from {template_path}")

    # --- filter ---
    templates = all_templates

    if args.split and args.split != "all":
        templates = [t for t in templates if t.get("split") == args.split]
        print(f"After split filter ({args.split}): {len(templates)} templates")

    if args.layout_family:
        templates = [t for t in templates if t.get("layout_family") == args.layout_family]
        print(f"After layout-family filter ({args.layout_family}): {len(templates)} templates")

    if args.shape_family:
        templates = [t for t in templates if t.get("shape_family") == args.shape_family]
        print(f"After shape-family filter ({args.shape_family}): {len(templates)} templates")

    if not templates:
        print("ERROR: No templates match the filter criteria.")
        sys.exit(1)

    # --- select single or slice ---
    if args.reset_template_id:
        matches = [t for t in templates if t.get("reset_template_id") == args.reset_template_id]
        if not matches:
            print(f"ERROR: No template with reset_template_id={args.reset_template_id}")
            sys.exit(1)
        templates = [matches[0]]
        print(f"Selected template by id: {args.reset_template_id}")
    elif args.template_index is not None:
        if args.template_index < 0 or args.template_index >= len(templates):
            print(f"ERROR: template-index {args.template_index} out of range [0, {len(templates)-1}]")
            sys.exit(1)
        templates = [templates[args.template_index]]
        print(f"Selected single template at index {args.template_index}")
    else:
        templates = templates[:args.max_templates]
        print(f"Using first {len(templates)} templates")

    # --- print-json (single template only) ---
    if args.print_json and len(templates) == 1:
        t = templates[0]
        print("\n=== Template fields ===")
        print(f"reset_template_id : {t.get('reset_template_id', '?')}")
        print(f"split             : {t.get('split', '?')}")
        print(f"layout_family     : {t.get('layout_family', '?')}")
        print(f"shape_family      : {t.get('shape_family', '?')}")
        for field, keys in [
            ("object_initial_pose", ["object_initial_pose", "object_pose"]),
            ("goal_pose",           ["goal_pose", "object_goal_pose"]),
            ("ee_initial_pose",     ["ee_initial_pose", "ee_pos", "pusher_initial_pose"]),
        ]:
            for k in keys:
                if k in t and t[k]:
                    p = t[k]
                    print(f"{field:20s}: x={p.get('x','?')}  y={p.get('y','?')}  theta={p.get('theta', 0.0)}")
                    break
        obstacles = t.get("obstacles", [])
        print(f"obstacles         : {len(obstacles)}")
        for obs in obstacles:
            print(f"  obstacle_id={obs.get('obstacle_id','?')}  pose={obs.get('pose',{})}  "
                  f"size_x={obs.get('size_x','?')}  size_y={obs.get('size_y','?')}  "
                  f"shape={obs.get('shape','?')}  valid={obs.get('valid','?')}")
        print("=== End fields ===\n")

    # --- render ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(templates) == 1:
        # single template mode
        template = templates[0]
        template_id = template.get("reset_template_id", "single")
        print(f"\nRendering single template: {template_id}")

        fig = render_single_template(
            template,
            pusher_radius=args.pusher_radius,
            workspace_x=args.workspace_x,
            workspace_y=args.workspace_y,
        )

        if args.out_image:
            out_path = Path(args.out_image)
        else:
            out_path = out_dir / f"template_{_safe_filename(template_id)}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=args.dpi)
        print(f"Saved: {out_path}")
        plt.close(fig)
    else:
        # grid mode
        print(f"\nRendering {len(templates)} templates in grid (cols={args.cols})...")
        fig = render_template_grid(
            templates,
            pusher_radius=args.pusher_radius,
            workspace_x=args.workspace_x,
            workspace_y=args.workspace_y,
            cols=args.cols,
        )

        if args.out_image:
            out_path = Path(args.out_image)
        else:
            out_path = out_dir / f"{args.split}_preview_grid.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=args.dpi)
        print(f"Saved: {out_path}")
        plt.close(fig)

    # --- summary ---
    print(f"\nDone. Rendered {len(templates)} template(s).")
    parsed_ok = 0
    parse_errors = 0
    for t in templates:
        try:
            _get_pose(t, ["object_initial_pose", "object_pose", "initial_object_pose"], "obj")
            _get_pose(t, ["goal_pose", "object_goal_pose"], "goal")
            _get_pose(t, ["ee_initial_pose", "ee_pos", "initial_ee_pos", "pusher_initial_pose"], "ee")
            parsed_ok += 1
        except KeyError as e:
            print(f"  PARSE ERROR in {t.get('reset_template_id', '?')}: {e}")
            parse_errors += 1
    print(f"Template parse: {parsed_ok} OK, {parse_errors} errors")


def _safe_filename(text: str) -> str:
    """Make a safe filename from a string."""
    return text.replace("/", "_").replace(" ", "_").replace(":", "_")


if __name__ == "__main__":
    main()
