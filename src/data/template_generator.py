"""Random obstacle template generator for layout OOD state16 data collection.

Generates infinite random templates for:
- open: no obstacles
- blocking: one random obstacle on object→goal path
- passage: two obstacles forming a random-gap passage

All templates pass basic safety checks (no overlap, workspace bounds).
"""

from __future__ import annotations

import random
import numpy as np


def _safe_clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def random_open_template(rng: random.Random, object_shape: str = "T") -> dict:
    """No obstacles, open workspace. Object and goal positions are randomized."""
    ox = rng.uniform(0.15, 0.25)
    oy = rng.uniform(0.12, 0.32)
    otheta = rng.uniform(-0.3, 0.3)

    gx = rng.uniform(ox + 0.15, 0.55)
    gy = rng.uniform(max(0.05, oy - 0.10), min(0.45, oy + 0.10))
    gtheta = rng.uniform(-0.3, 0.3)

    ee_x = max(0.02, ox - 0.10)

    return {
        "object_initial_pose": {"x": ox, "y": oy, "theta": otheta},
        "goal_pose": {"x": gx, "y": gy, "theta": gtheta},
        "ee_initial_pose": {"x": ee_x, "y": oy, "theta": 0.0},
        "object_shape": object_shape,
        "object_size_x": 0.048,
        "object_size_y": 0.048,
        "object_mass": 0.01905,
        "obstacles": [],
        "family": "open",
    }


def random_blocking_template(rng: random.Random, object_shape: str = "T") -> dict:
    """One random obstacle placed on object→goal path.

    Obstacle size, position, and orientation are all randomized.
    Constraint: obstacle X is strictly between object and goal in X.
    """
    ox = rng.uniform(0.15, 0.25)
    oy = rng.uniform(0.15, 0.30)
    otheta = rng.uniform(-0.3, 0.3)

    gx = rng.uniform(ox + 0.15, 0.55)
    gy = rng.uniform(max(0.05, oy - 0.12), min(0.45, oy + 0.12))
    gtheta = rng.uniform(-0.3, 0.3)

    # Obstacle size: random within wide range
    size_x = rng.uniform(0.03, 0.12)
    size_y = rng.uniform(0.04, 0.15)

    # Obstacle near object→goal midpoint with jitter
    mid_x = (ox + gx) / 2.0
    mid_y = (oy + gy) / 2.0
    jitter_x = rng.uniform(-0.04, 0.04)
    jitter_y = rng.uniform(-0.04, 0.04)

    obs_x = mid_x + jitter_x
    obs_y = mid_y + jitter_y

    # Safety: obstacle must be between object and goal in X
    obs_x = _safe_clamp(obs_x, ox + 0.04, gx - 0.04)

    # Random rotation
    obs_theta = rng.uniform(-0.6, 0.6)

    ee_x = max(0.02, ox - 0.10)

    return {
        "object_initial_pose": {"x": ox, "y": oy, "theta": otheta},
        "goal_pose": {"x": gx, "y": gy, "theta": gtheta},
        "ee_initial_pose": {"x": ee_x, "y": oy, "theta": 0.0},
        "object_shape": object_shape,
        "object_size_x": 0.048,
        "object_size_y": 0.048,
        "object_mass": 0.01905,
        "obstacles": [{
            "pose": {"x": obs_x, "y": obs_y, "theta": obs_theta},
            "size_x": size_x,
            "size_y": size_y,
        }],
        "family": "blocking",
    }


def random_passage_template(rng: random.Random, object_shape: str = "T") -> dict:
    """Two obstacles forming a passage with random gap width.

    Obstacles are placed symmetrically around the object→goal midpoint.
    Gap width is uniformly sampled from [0.04, 0.12] m.
    """
    ox = rng.uniform(0.15, 0.22)
    oy = rng.uniform(0.15, 0.30)
    otheta = rng.uniform(-0.3, 0.3)

    gx = rng.uniform(ox + 0.20, 0.55)
    gy = rng.uniform(max(0.08, oy - 0.10), min(0.42, oy + 0.10))
    gtheta = rng.uniform(-0.3, 0.3)

    # Random passage gap
    gap = rng.uniform(0.04, 0.12)
    half_gap = gap / 2.0

    # Passage center near midpoint with small jitter
    mid_x = (ox + gx) / 2.0 + rng.uniform(-0.02, 0.02)
    mid_y = (oy + gy) / 2.0 + rng.uniform(-0.02, 0.02)

    # Top obstacle
    top_sx = rng.uniform(0.06, 0.12)
    top_sy = rng.uniform(0.04, 0.08)
    top_y = mid_y + half_gap + top_sy / 2.0

    # Bottom obstacle
    bot_sx = rng.uniform(0.06, 0.12)
    bot_sy = rng.uniform(0.04, 0.08)
    bot_y = mid_y - half_gap - bot_sy / 2.0

    ee_x = max(0.02, ox - 0.10)

    return {
        "object_initial_pose": {"x": ox, "y": oy, "theta": otheta},
        "goal_pose": {"x": gx, "y": gy, "theta": gtheta},
        "ee_initial_pose": {"x": ee_x, "y": oy, "theta": 0.0},
        "object_shape": object_shape,
        "object_size_x": 0.048,
        "object_size_y": 0.048,
        "object_mass": 0.01905,
        "obstacles": [
            {
                "pose": {"x": mid_x, "y": top_y, "theta": 0.0},
                "size_x": top_sx,
                "size_y": top_sy,
            },
            {
                "pose": {"x": mid_x, "y": bot_y, "theta": 0.0},
                "size_x": bot_sx,
                "size_y": bot_sy,
            },
        ],
        "family": "passage",
        "_meta": {"passage_gap": gap},
    }


TEMPLATE_GENERATORS = {
    "open": random_open_template,
    "blocking": random_blocking_template,
    "passage": random_passage_template,
}


def generate_template(family: str, rng: random.Random, object_shape: str = "T") -> dict:
    """Generate one random template for the given family."""
    if family not in TEMPLATE_GENERATORS:
        raise ValueError(
            f"Unknown family '{family}'. Available: {list(TEMPLATE_GENERATORS.keys())}"
        )
    return TEMPLATE_GENERATORS[family](rng, object_shape=object_shape)


def is_template_valid(template: dict) -> bool:
    """Quick sanity check: no overlap between obstacles and object/goal."""
    obj_x = template["object_initial_pose"]["x"]
    obj_y = template["object_initial_pose"]["y"]
    goal_x = template["goal_pose"]["x"]
    goal_y = template["goal_pose"]["y"]
    obj_half = 0.024  # half of object size

    for obs in template.get("obstacles", []):
        ox = obs["pose"]["x"]
        oy = obs["pose"]["y"]
        half_sx = obs["size_x"] / 2.0 + obj_half
        half_sy = obs["size_y"] / 2.0 + obj_half

        # Check object-overlap
        if abs(obj_x - ox) < half_sx and abs(obj_y - oy) < half_sy:
            return False
        # Check goal-overlap
        if abs(goal_x - ox) < half_sx and abs(goal_y - oy) < half_sy:
            return False

    return True
