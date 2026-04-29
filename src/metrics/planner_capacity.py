from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_WORKSPACE_BOUNDS = {
    "x_min": 0.0,
    "x_max": 0.70,
    "y_min": 0.0,
    "y_max": 0.50,
}


DEFAULT_THRESHOLDS = {
    # Object-goal distance should not be trivial or impossibly far.
    "min_object_goal_dist": 0.08,
    "max_object_goal_dist": 0.55,

    # EE should start near enough to make contact possible,
    # but not already on top of the object.
    "min_ee_object_dist": 0.04,
    "max_ee_object_dist": 0.18,
}


def _require_finite(value: float, name: str) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite, got {value}")


def validate_workspace_bounds(bounds: dict[str, float]) -> None:
    required = ["x_min", "x_max", "y_min", "y_max"]

    for key in required:
        if key not in bounds:
            raise ValueError(f"Missing workspace bound: {key}")
        _require_finite(bounds[key], key)

    if bounds["x_min"] >= bounds["x_max"]:
        raise ValueError(
            f"Invalid workspace x bounds: x_min={bounds['x_min']}, "
            f"x_max={bounds['x_max']}"
        )

    if bounds["y_min"] >= bounds["y_max"]:
        raise ValueError(
            f"Invalid workspace y bounds: y_min={bounds['y_min']}, "
            f"y_max={bounds['y_max']}"
        )


def validate_thresholds(thresholds: dict[str, float]) -> None:
    required = [
        "min_object_goal_dist",
        "max_object_goal_dist",
        "min_ee_object_dist",
        "max_ee_object_dist",
    ]

    for key in required:
        if key not in thresholds:
            raise ValueError(f"Missing threshold: {key}")
        _require_finite(thresholds[key], key)

    if thresholds["min_object_goal_dist"] < 0:
        raise ValueError("min_object_goal_dist must be non-negative")

    if thresholds["min_ee_object_dist"] < 0:
        raise ValueError("min_ee_object_dist must be non-negative")

    if thresholds["min_object_goal_dist"] >= thresholds["max_object_goal_dist"]:
        raise ValueError("object-goal distance thresholds are invalid")

    if thresholds["min_ee_object_dist"] >= thresholds["max_ee_object_dist"]:
        raise ValueError("ee-object distance thresholds are invalid")


def euclidean_distance(pose_a: dict, pose_b: dict) -> float:
    dx = float(pose_a["x"]) - float(pose_b["x"])
    dy = float(pose_a["y"]) - float(pose_b["y"])
    return math.sqrt(dx * dx + dy * dy)


def point_in_workspace(
    pose: dict,
    bounds: dict[str, float],
    margin: float = 0.0,
) -> bool:
    x = float(pose["x"])
    y = float(pose["y"])

    return (
        bounds["x_min"] + margin <= x <= bounds["x_max"] - margin
        and bounds["y_min"] + margin <= y <= bounds["y_max"] - margin
    )


def rect_bounds(
    pose: dict,
    size_x: float,
    size_y: float,
    margin: float = 0.0,
) -> tuple[float, float, float, float]:
    """
    Axis-aligned rectangle bounds.

    v0.1 ignores theta. This is only a conservative sanity check,
    not an exact collision checker.
    """
    x = float(pose["x"])
    y = float(pose["y"])
    size_x = float(size_x)
    size_y = float(size_y)

    _require_finite(x, "pose.x")
    _require_finite(y, "pose.y")
    _require_finite(size_x, "size_x")
    _require_finite(size_y, "size_y")

    if size_x <= 0 or size_y <= 0:
        raise ValueError(f"Invalid rectangle size: size_x={size_x}, size_y={size_y}")

    half_x = 0.5 * size_x + margin
    half_y = 0.5 * size_y + margin

    return (
        x - half_x,
        x + half_x,
        y - half_y,
        y + half_y,
    )


def rects_overlap(
    rect_a: tuple[float, float, float, float],
    rect_b: tuple[float, float, float, float],
) -> bool:
    ax_min, ax_max, ay_min, ay_max = rect_a
    bx_min, bx_max, by_min, by_max = rect_b

    overlap_x = ax_min < bx_max and ax_max > bx_min
    overlap_y = ay_min < by_max and ay_max > by_min

    return overlap_x and overlap_y


def check_template_sanity(
    template: dict[str, Any],
    workspace_bounds: dict[str, float] | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Check one reset template for obvious geometry / task issues.

    This does not guarantee MuJoCo feasibility.
    It only catches obvious bad templates before planner experiments.
    """
    if workspace_bounds is None:
        workspace_bounds = DEFAULT_WORKSPACE_BOUNDS

    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    validate_workspace_bounds(workspace_bounds)
    validate_thresholds(thresholds)

    reset_template_id = template["reset_template_id"]
    split = template["split"]
    layout_family = template["layout_family"]
    shape_family = template["shape_family"]

    errors: list[str] = []
    warnings: list[str] = []

    object_pose = template["object_initial_pose"]
    goal_pose = template["goal_pose"]
    ee_pose = template["ee_initial_pose"]

    object_size_x = float(template["object_size_x"])
    object_size_y = float(template["object_size_y"])

    # Workspace checks.
    if not point_in_workspace(object_pose, workspace_bounds, margin=0.02):
        errors.append("object_initial_pose outside workspace or too close to edge")

    if not point_in_workspace(goal_pose, workspace_bounds, margin=0.02):
        errors.append("goal_pose outside workspace or too close to edge")

    if not point_in_workspace(ee_pose, workspace_bounds, margin=0.00):
        errors.append("ee_initial_pose outside workspace")

    # Distance checks.
    object_goal_dist = euclidean_distance(object_pose, goal_pose)
    ee_object_dist = euclidean_distance(ee_pose, object_pose)

    if object_goal_dist < thresholds["min_object_goal_dist"]:
        warnings.append(f"object-goal distance too small: {object_goal_dist:.4f}")

    if object_goal_dist > thresholds["max_object_goal_dist"]:
        warnings.append(f"object-goal distance too large: {object_goal_dist:.4f}")

    if ee_object_dist < thresholds["min_ee_object_dist"]:
        warnings.append(
            f"ee-object distance very small, may start in contact/collision: "
            f"{ee_object_dist:.4f}"
        )

    if ee_object_dist > thresholds["max_ee_object_dist"]:
        warnings.append(
            f"ee-object distance large, CEM may fail to establish contact: "
            f"{ee_object_dist:.4f}"
        )

    # Obstacle checks.
    object_rect = rect_bounds(
        object_pose,
        object_size_x,
        object_size_y,
        margin=0.005,
    )
    goal_rect = rect_bounds(
        goal_pose,
        object_size_x,
        object_size_y,
        margin=0.005,
    )
    ee_rect = rect_bounds(
        ee_pose,
        size_x=0.025,
        size_y=0.025,
        margin=0.005,
    )

    obstacles = template.get("obstacles", [])
    valid_obstacles = []

    for obs in obstacles:
        if not bool(obs.get("valid", True)):
            continue

        valid_obstacles.append(obs)

        obs_id = obs["obstacle_id"]
        obs_pose = obs["pose"]
        obs_size_x = float(obs["size_x"])
        obs_size_y = float(obs["size_y"])

        if not point_in_workspace(obs_pose, workspace_bounds, margin=0.02):
            errors.append(f"obstacle {obs_id} outside workspace or too close to edge")

        obs_rect = rect_bounds(
            obs_pose,
            obs_size_x,
            obs_size_y,
            margin=0.005,
        )

        if rects_overlap(object_rect, obs_rect):
            errors.append(f"object initial pose overlaps obstacle {obs_id}")

        if rects_overlap(goal_rect, obs_rect):
            errors.append(f"goal pose overlaps obstacle {obs_id}")

        if rects_overlap(ee_rect, obs_rect):
            errors.append(f"ee initial pose overlaps obstacle {obs_id}")

    return {
        "reset_template_id": reset_template_id,
        "split": split,
        "layout_family": layout_family,
        "shape_family": shape_family,
        "object_goal_dist": object_goal_dist,
        "ee_object_dist": ee_object_dist,
        "num_obstacles": len(valid_obstacles),
        "errors": errors,
        "warnings": warnings,
        "ok": len(errors) == 0,
    }


def run_state_sanity(
    templates: list[dict[str, Any]],
    workspace_bounds: dict[str, float] | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    results = [
        check_template_sanity(
            template=t,
            workspace_bounds=workspace_bounds,
            thresholds=thresholds,
        )
        for t in templates
    ]

    num_errors = sum(len(r["errors"]) for r in results)
    num_warnings = sum(len(r["warnings"]) for r in results)
    num_ok_templates = sum(1 for r in results if r["ok"])

    summary = {
        "num_templates": len(results),
        "num_ok_templates": num_ok_templates,
        "num_error_messages": num_errors,
        "num_warning_messages": num_warnings,
        "by_split": dict(Counter(r["split"] for r in results)),
        "by_layout_family": dict(Counter(r["layout_family"] for r in results)),
        "by_shape_family": dict(Counter(r["shape_family"] for r in results)),
    }

    return {
        "summary": summary,
        "results": results,
    }


def save_state_sanity_report(report: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)