"""Validation utilities for v2 episodes."""

from pathlib import Path
from typing import Optional
import numpy as np


def validate_v2_npz(
    path: str | Path,
    schema: dict,
) -> dict:
    """Validate a v2 .npz file against the schema.
    
    Returns dict with keys: valid, errors, warnings, shapes
    """
    result = {"valid": True, "errors": [], "warnings": [], "shapes": {}}
    
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"Cannot load: {e}")
        return result
    
    # Check required keys
    required_keys = ["obj_x", "obj_y", "ee_x", "ee_y", "goal_x", "goal_y",
                     "actions_physical", "object_poses", "goal_pose"]
    for key in required_keys:
        if key not in data:
            result["valid"] = False
            result["errors"].append(f"Missing key: {key}")
        else:
            result["shapes"][key] = str(data[key].shape)
    
    # Check no NaN/inf in numeric arrays
    for key in data.keys():
        arr = data[key]
        if isinstance(arr, np.ndarray) and arr.dtype.kind in "fi":
            if np.any(np.isnan(arr)):
                result["valid"] = False
                result["errors"].append(f"NaN in {key}")
            if np.any(np.isinf(arr)):
                result["valid"] = False
                result["errors"].append(f"Inf in {key}")
    
    # Check alignment
    T = None
    for key in ["obj_x", "ee_x", "actions_physical", "object_poses"]:
        if key in data and isinstance(data[key], np.ndarray) and data[key].ndim >= 1:
            if T is None:
                T = len(data[key])
            elif len(data[key]) != T:
                result["errors"].append(f"Length mismatch: {key} has {len(data[key])}, expected {T}")
    
    return result
