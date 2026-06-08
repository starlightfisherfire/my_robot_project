"""Convert v2 episode to state16v2 format [T,6,16].

State16 v2: geometric relationships instead of velocity.
No vx/vy/omega. Adds dist_to_ee, dist_to_goal, angle_to_goal, relative coordinates.

Token layout (16 dim):
  [0] x, [1] y, [2] sin(theta), [3] cos(theta)
  [4] dist_to_ee, [5] dist_to_goal, [6] angle_to_goal
  [7] width
  [8] valid_flag
  [9] rel_x_to_ee, [10] rel_y_to_ee
  [11] rel_x_to_goal, [12] rel_y_to_goal
  [13] mass, [14] friction
  [15] valid_flag (duplicate for compatibility)
"""

import numpy as np


def _angle(dx, dy):
    """Compute angle from dx, dy."""
    return np.arctan2(dy, dx)


def convert_v2_to_state16v2(
    v2_episode: dict[str, np.ndarray],
    history_len: int = 6,
) -> dict[str, np.ndarray]:
    """Convert a v2 episode dict to state16v2 format.

    Args:
        v2_episode: dict with keys like obj_x, obj_y, ee_x, ee_y, goal_x, etc.
        history_len: number of history frames (default 6)

    Returns:
        dict with keys: states [T,6,16], actions_physical [T,2],
        object_poses [T,3], next_object_poses [T,3], goal_pose [3]
    """
    required = ["obj_x", "obj_y", "obj_sin_theta", "obj_cos_theta",
                "ee_x", "ee_y", "goal_x", "goal_y", "goal_sin_theta", "goal_cos_theta",
                "object_poses", "goal_pose"]
    missing = [k for k in required if k not in v2_episode]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    T = len(v2_episode["obj_x"])
    states = np.zeros((T, 6, 16), dtype=np.float32)

    # Extract positions
    ee_x = v2_episode["ee_x"][:T]
    ee_y = v2_episode["ee_y"][:T]
    obj_x = v2_episode["obj_x"][:T]
    obj_y = v2_episode["obj_y"][:T]
    goal_x = v2_episode["goal_x"][:T]
    goal_y = v2_episode["goal_y"][:T]

    # Pre-compute geometric relationships for object token
    dx_ee = ee_x - obj_x
    dy_ee = ee_y - obj_y
    dist_ee = np.sqrt(dx_ee**2 + dy_ee**2)

    dx_goal = goal_x - obj_x
    dy_goal = goal_y - obj_y
    dist_goal = np.sqrt(dx_goal**2 + dy_goal**2)
    angle_goal = _angle(dx_goal, dy_goal)

    # Contact detection
    contact = (dist_ee < 0.06).astype(np.float32)

    # ── Token 0: EE ──
    states[:, 0, 0] = ee_x
    states[:, 0, 1] = ee_y
    states[:, 0, 3] = 1.0  # cos(theta)=1 (EE has no orientation)
    # EE geometry relative to object
    states[:, 0, 4] = 0.0  # dist_to_ee (self) = 0
    states[:, 0, 5] = np.sqrt((goal_x - ee_x)**2 + (goal_y - ee_y)**2)  # dist_to_goal
    states[:, 0, 6] = _angle(goal_x - ee_x, goal_y - ee_y)  # angle_to_goal
    states[:, 0, 7] = 0.0  # width (EE is point)
    states[:, 0, 8] = 1.0  # valid
    states[:, 0, 9] = 0.0  # rel_x_to_ee (self) = 0
    states[:, 0, 10] = 0.0  # rel_y_to_ee (self) = 0
    states[:, 0, 11] = goal_x - ee_x  # rel_x_to_goal
    states[:, 0, 12] = goal_y - ee_y  # rel_y_to_goal
    states[:, 0, 13] = 0.0  # mass (EE has no mass)
    states[:, 0, 14] = 0.0  # friction
    states[:, 0, 15] = 1.0  # valid

    # ── Token 1: Object ──
    states[:, 1, 0] = obj_x
    states[:, 1, 1] = obj_y
    states[:, 1, 2] = v2_episode["obj_sin_theta"][:T]
    states[:, 1, 3] = v2_episode["obj_cos_theta"][:T]
    states[:, 1, 4] = dist_ee
    states[:, 1, 5] = dist_goal
    states[:, 1, 6] = angle_goal
    states[:, 1, 7] = v2_episode.get("obj_size_x", np.ones(T) * 0.048)
    states[:, 1, 8] = 1.0  # valid
    states[:, 1, 9] = dx_ee
    states[:, 1, 10] = dy_ee
    states[:, 1, 11] = dx_goal
    states[:, 1, 12] = dy_goal
    states[:, 1, 13] = v2_episode.get("object_mass", np.ones(T) * 0.038)
    states[:, 1, 14] = v2_episode.get("object_friction", np.ones(T) * 0.8)
    states[:, 1, 15] = 1.0

    # ── Token 2: Goal ──
    states[:, 2, 0] = goal_x
    states[:, 2, 1] = goal_y
    states[:, 2, 2] = v2_episode["goal_sin_theta"][:T]
    states[:, 2, 3] = v2_episode["goal_cos_theta"][:T]
    states[:, 2, 4] = dist_goal  # dist_to_ee (from goal's perspective)
    states[:, 2, 5] = 0.0  # dist_to_goal (self) = 0
    states[:, 2, 6] = 0.0  # angle_to_goal (self) = 0
    states[:, 2, 7] = 0.048
    states[:, 2, 8] = 1.0
    states[:, 2, 9] = -dx_goal  # rel_x_to_ee (from goal)
    states[:, 2, 10] = -dy_goal
    states[:, 2, 11] = 0.0  # rel_x_to_goal (self) = 0
    states[:, 2, 12] = 0.0
    states[:, 2, 13] = 0.038
    states[:, 2, 14] = 0.8
    states[:, 2, 15] = 1.0

    # ── Token 3-5: Obstacles ──
    for oi in range(3):
        obs_x_key = f"obs{oi+1}_x" if oi > 0 else "obs_x"
        if obs_x_key not in v2_episode and oi > 0:
            break
        if "obs_x" in v2_episode and oi == 0:
            ox = v2_episode["obs_x"][:T]
            oy = v2_episode["obs_y"][:T]
            states[:, 3, 0] = ox
            states[:, 3, 1] = oy
            states[:, 3, 3] = 1.0
            states[:, 3, 4] = np.sqrt((ee_x - ox)**2 + (ee_y - oy)**2)
            states[:, 3, 5] = np.sqrt((goal_x - ox)**2 + (goal_y - oy)**2)
            states[:, 3, 6] = _angle(goal_x - ox, goal_y - oy)
            states[:, 3, 7] = v2_episode.get("obs_size_x", np.ones(T) * 0.1)
            states[:, 3, 8] = 1.0
            states[:, 3, 9] = ee_x - ox
            states[:, 3, 10] = ee_y - oy
            states[:, 3, 11] = goal_x - ox
            states[:, 3, 12] = goal_y - oy
            states[:, 3, 13] = 0.5
            states[:, 3, 14] = 0.8
            states[:, 3, 15] = 1.0

    return {
        "states": states,
        "actions_physical": v2_episode.get(
            "actions_physical", np.zeros((T, 2), dtype=np.float32))[:T],
        "object_poses": v2_episode["object_poses"][:T],
        "next_object_poses": v2_episode.get(
            "next_object_poses", np.zeros((T, 3), dtype=np.float32))[:T],
        "goal_pose": v2_episode["goal_pose"][:3],
    }
