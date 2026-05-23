"""Convert v2 episode to canonical_state16 format [T,6,16]."""

from typing import Optional
import numpy as np


def convert_v2_to_state16(
    v2_episode: dict[str, np.ndarray],
    history_len: int = 6,
) -> dict[str, np.ndarray]:
    """Convert a v2 episode dict to canonical_state16 format.
    
    Args:
        v2_episode: dict with keys like obj_x, obj_y, ee_x, ee_y, goal_x, etc.
        history_len: number of history frames (default 6)
    
    Returns:
        dict with keys: states [T,6,16], actions_physical [T,2],
        object_poses [T,3], next_object_poses [T,3], goal_pose [3]
    
    Raises:
        ValueError: if required fields are missing.
    """
    # Check required fields
    required = ["obj_x", "obj_y", "obj_sin_theta", "obj_cos_theta",
                "ee_x", "ee_y", "goal_x", "goal_y", "goal_sin_theta", "goal_cos_theta",
                "object_poses", "goal_pose"]
    missing = [k for k in required if k not in v2_episode]
    if missing:
        raise ValueError(f"Missing required fields for conversion: {missing}")
    
    T = len(v2_episode["obj_x"])
    states = np.zeros((T, 6, 16), dtype=np.float32)
    
    # Token 0: EE
    states[:, 0, 0] = v2_episode["ee_x"][:T]
    states[:, 0, 1] = v2_episode["ee_y"][:T]
    states[:, 0, 3] = 1.0  # cos(theta)=1
    if "ee_vx" in v2_episode:
        states[:, 0, 4] = v2_episode["ee_vx"][:T]
    if "ee_vy" in v2_episode:
        states[:, 0, 5] = v2_episode["ee_vy"][:T]
    if "contact_proxy" in v2_episode:
        states[:, 0, 14] = v2_episode["contact_proxy"][:T]
    states[:, 0, 15] = 1.0  # valid
    
    # Token 1: Object
    states[:, 1, 0] = v2_episode["obj_x"][:T]
    states[:, 1, 1] = v2_episode["obj_y"][:T]
    states[:, 1, 2] = v2_episode["obj_sin_theta"][:T]
    states[:, 1, 3] = v2_episode["obj_cos_theta"][:T]
    if "obj_vx" in v2_episode:
        states[:, 1, 4] = v2_episode["obj_vx"][:T]
    if "obj_vy" in v2_episode:
        states[:, 1, 5] = v2_episode["obj_vy"][:T]
    if "obj_omega" in v2_episode:
        states[:, 1, 6] = v2_episode["obj_omega"][:T]
    if "obj_size_x" in v2_episode:
        states[:, 1, 7] = v2_episode.get("obj_size_x", np.ones(T)*0.048)
    if "obj_size_y" in v2_episode:
        states[:, 1, 8] = v2_episode.get("obj_size_y", np.ones(T)*0.048)
    if "obj_shape_T" in v2_episode:
        states[:, 1, 9] = v2_episode["obj_shape_T"][:T]
    else:
        states[:, 1, 9] = 1.0
    if "object_mass" in v2_episode:
        states[:, 1, 12] = v2_episode["object_mass"][:T]
    else:
        states[:, 1, 12] = 0.038
    if "object_friction" in v2_episode:
        states[:, 1, 13] = v2_episode["object_friction"][:T]
    else:
        states[:, 1, 13] = 0.8
    states[:, 1, 15] = 1.0
    
    # Token 2: Goal
    states[:, 2, 0] = v2_episode["goal_x"][:T]
    states[:, 2, 1] = v2_episode["goal_y"][:T]
    states[:, 2, 2] = v2_episode["goal_sin_theta"][:T]
    states[:, 2, 3] = v2_episode["goal_cos_theta"][:T]
    states[:, 2, 7] = 0.048; states[:, 2, 8] = 0.048
    states[:, 2, 9] = 1.0; states[:, 2, 12] = 0.038; states[:, 2, 13] = 0.8
    states[:, 2, 15] = 1.0
    
    # Token 3-5: Obstacles
    for oi in range(3):
        obs_x_key = f"obs{oi+1}_x" if oi > 0 else "obs_x"
        if obs_x_key not in v2_episode and oi > 0:
            break  # No more obstacles
        if "obs_x" in v2_episode and oi == 0:
            states[:, 3, 0] = v2_episode["obs_x"][:T]
            states[:, 3, 1] = v2_episode["obs_y"][:T]
            if "obs_size_x" in v2_episode:
                states[:, 3, 7] = v2_episode["obs_size_x"][:T]
            if "obs_size_y" in v2_episode:
                states[:, 3, 8] = v2_episode["obs_size_y"][:T]
            states[:, 3, 3] = 1.0; states[:, 3, 12] = 0.5; states[:, 3, 13] = 0.8
            states[:, 3, 15] = 1.0
    
    return {
        "states": states,
        "actions_physical": v2_episode.get("actions_physical", np.zeros((T, 2), dtype=np.float32))[:T],
        "object_poses": v2_episode["object_poses"][:T],
        "next_object_poses": v2_episode.get("next_object_poses", np.zeros((T, 3), dtype=np.float32))[:T],
        "goal_pose": v2_episode["goal_pose"][:3],
    }
