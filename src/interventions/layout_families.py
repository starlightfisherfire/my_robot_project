from __future__ import annotations

import random
from copy import deepcopy


def make_pose(x: float, y: float, theta: float = 0.0) -> dict:
    return {
        "x": float(x),
        "y": float(y),
        "theta": float(theta),
    }


def make_obstacle(
    obstacle_id: str,
    x: float,
    y: float,
    size_x: float,
    size_y: float,
    theta: float = 0.0,
    shape: str = "box",
    valid: bool = True,
) -> dict:
    return {
        "obstacle_id": obstacle_id,
        "pose": make_pose(x=x, y=y, theta=theta),
        "size_x": float(size_x),
        "size_y": float(size_y),
        "shape": shape,
        "valid": bool(valid),
    }


def _jitter(rng: random.Random, value: float, scale: float) -> float:
    return float(value + rng.uniform(-scale, scale))


def sample_open_space(rng: random.Random) -> dict:
    """
    ID layout.

    No obstacles. Object and goal are in an open workspace.
    """
    object_pose = make_pose(
        x=_jitter(rng, 0.20, 0.03),
        y=_jitter(rng, 0.18, 0.03),
        theta=rng.uniform(-0.25, 0.25),
    )

    goal_pose = make_pose(
        x=_jitter(rng, 0.45, 0.04),
        y=_jitter(rng, 0.28, 0.04),
        theta=rng.uniform(-0.25, 0.25),
    )

    ee_pose = make_pose(
        x=object_pose["x"] - 0.10,
        y=object_pose["y"],
        theta=0.0,
    )

    return {
        "layout_family": "open_space",
        "object_initial_pose": object_pose,
        "goal_pose": goal_pose,
        "ee_initial_pose": ee_pose,
        "obstacles": [],
    }


def sample_mild_offset(rng: random.Random) -> dict:
    """
    ID layout variant.

    Still no hard obstacle. Goal is mildly offset to create more variation.
    """
    object_pose = make_pose(
        x=_jitter(rng, 0.22, 0.03),
        y=_jitter(rng, 0.18, 0.03),
        theta=rng.uniform(-0.30, 0.30),
    )

    goal_pose = make_pose(
        x=_jitter(rng, 0.42, 0.04),
        y=_jitter(rng, 0.32, 0.05),
        theta=rng.uniform(-0.30, 0.30),
    )

    ee_pose = make_pose(
        x=object_pose["x"] - 0.10,
        y=object_pose["y"] - 0.02,
        theta=0.0,
    )

    return {
        "layout_family": "mild_offset",
        "object_initial_pose": object_pose,
        "goal_pose": goal_pose,
        "ee_initial_pose": ee_pose,
        "obstacles": [],
    }


def sample_blocking(rng: random.Random) -> dict:
    """
    Layout OOD.

    One obstacle roughly blocks the direct path between object and goal.
    """
    object_pose = make_pose(
        x=_jitter(rng, 0.18, 0.025),
        y=_jitter(rng, 0.20, 0.025),
        theta=rng.uniform(-0.25, 0.25),
    )

    goal_pose = make_pose(
        x=_jitter(rng, 0.48, 0.035),
        y=_jitter(rng, 0.30, 0.035),
        theta=rng.uniform(-0.25, 0.25),
    )

    mid_x = 0.5 * (object_pose["x"] + goal_pose["x"])
    mid_y = 0.5 * (object_pose["y"] + goal_pose["y"])

    obstacle = make_obstacle(
        obstacle_id="obs_blocking_0",
        x=_jitter(rng, mid_x, 0.02),
        y=_jitter(rng, mid_y, 0.02),
        size_x=0.06,
        size_y=0.12,
        theta=rng.uniform(-0.4, 0.4),
    )

    ee_pose = make_pose(
        x=object_pose["x"] - 0.10,
        y=object_pose["y"],
        theta=0.0,
    )

    return {
        "layout_family": "blocking",
        "object_initial_pose": object_pose,
        "goal_pose": goal_pose,
        "ee_initial_pose": ee_pose,
        "obstacles": [obstacle],
    }


def sample_narrow_passage(rng: random.Random) -> dict:
    """
    Layout OOD.

    Two obstacles create a narrow passage between object and goal.
    """
    object_pose = make_pose(
        x=_jitter(rng, 0.18, 0.025),
        y=_jitter(rng, 0.20, 0.025),
        theta=rng.uniform(-0.25, 0.25),
    )

    goal_pose = make_pose(
        x=_jitter(rng, 0.50, 0.035),
        y=_jitter(rng, 0.20, 0.03),
        theta=rng.uniform(-0.25, 0.25),
    )

    passage_x = 0.34
    center_y = 0.20
    gap = 0.12

    obstacles = [
        make_obstacle(
            obstacle_id="obs_narrow_top",
            x=_jitter(rng, passage_x, 0.015),
            y=center_y + gap,
            size_x=0.08,
            size_y=0.06,
            theta=0.0,
        ),
        make_obstacle(
            obstacle_id="obs_narrow_bottom",
            x=_jitter(rng, passage_x, 0.015),
            y=center_y - gap,
            size_x=0.08,
            size_y=0.06,
            theta=0.0,
        ),
    ]

    ee_pose = make_pose(
        x=object_pose["x"] - 0.10,
        y=object_pose["y"],
        theta=0.0,
    )

    return {
        "layout_family": "narrow_passage",
        "object_initial_pose": object_pose,
        "goal_pose": goal_pose,
        "ee_initial_pose": ee_pose,
        "obstacles": obstacles,
    }


def sample_edge_goal(rng: random.Random) -> dict:
    """
    Layout OOD.

    Goal is near the workspace edge, requiring more careful control.
    """
    object_pose = make_pose(
        x=_jitter(rng, 0.22, 0.025),
        y=_jitter(rng, 0.22, 0.025),
        theta=rng.uniform(-0.25, 0.25),
    )

    goal_pose = make_pose(
        x=_jitter(rng, 0.55, 0.02),
        y=_jitter(rng, 0.38, 0.02),
        theta=rng.uniform(-0.25, 0.25),
    )

    ee_pose = make_pose(
        x=object_pose["x"] - 0.10,
        y=object_pose["y"],
        theta=0.0,
    )

    return {
        "layout_family": "edge_goal",
        "object_initial_pose": object_pose,
        "goal_pose": goal_pose,
        "ee_initial_pose": ee_pose,
        "obstacles": [],
    }


def sample_non_blocking(rng: random.Random) -> dict:
    """
    ID layout with non-blocking obstacle.

    Same object/goal as open_space, but one obstacle is placed in the
    workspace corner far from the pushing path.  The obstacle is visible
    in the scene but should NOT interfere with ee → object → goal.

    Purpose: evaluate whether a learned model can ignore irrelevant obstacles.
    """
    object_pose = make_pose(
        x=_jitter(rng, 0.20, 0.03),
        y=_jitter(rng, 0.18, 0.03),
        theta=rng.uniform(-0.25, 0.25),
    )

    goal_pose = make_pose(
        x=_jitter(rng, 0.45, 0.04),
        y=_jitter(rng, 0.28, 0.04),
        theta=rng.uniform(-0.25, 0.25),
    )

    ee_pose = make_pose(
        x=object_pose["x"] - 0.10,
        y=object_pose["y"],
        theta=0.0,
    )

    # Place obstacle in upper-right corner, far from typical pushing path.
    # Typical path: x ∈ [0.10, 0.50], y ∈ [0.15, 0.35]
    # Obstacle center: (0.55, 0.42) — well outside this band.
    obstacle = make_obstacle(
        obstacle_id="obs_non_blocking_0",
        x=_jitter(rng, 0.55, 0.02),
        y=_jitter(rng, 0.42, 0.02),
        size_x=0.06,
        size_y=0.12,
        theta=rng.uniform(-0.3, 0.3),
    )

    return {
        "layout_family": "non_blocking",
        "object_initial_pose": object_pose,
        "goal_pose": goal_pose,
        "ee_initial_pose": ee_pose,
        "obstacles": [obstacle],
    }


LAYOUT_SAMPLERS = {
    "open_space": sample_open_space,
    "mild_offset": sample_mild_offset,
    "non_blocking": sample_non_blocking,
    "blocking": sample_blocking,
    "narrow_passage": sample_narrow_passage,
    "edge_goal": sample_edge_goal,
}


def get_layout_families() -> list[str]:
    return list(LAYOUT_SAMPLERS.keys())


def sample_layout_family(layout_family: str, rng: random.Random) -> dict:
    if layout_family not in LAYOUT_SAMPLERS:
        raise ValueError(
            f"Unknown layout_family={layout_family}. "
            f"Available: {list(LAYOUT_SAMPLERS.keys())}"
        )

    return deepcopy(LAYOUT_SAMPLERS[layout_family](rng))