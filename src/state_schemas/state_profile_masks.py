"""Apply profile masks to v2 episode data."""

from typing import Any
import numpy as np

from src.state_schemas.schema_registry import expand_profile_features


def apply_profile_mask(
    episode_data: dict[str, np.ndarray],
    profile_name: str,
    profiles: dict,
) -> dict[str, np.ndarray]:
    """Filter episode_data to only include features from the given profile.
    
    Returns a dict with only the allowed keys and their arrays.
    Features not in the profile are set to zero.
    """
    allowed = expand_profile_features(profiles, profile_name)
    result = {}
    for key, arr in episode_data.items():
        if isinstance(arr, np.ndarray):
            if key in allowed:
                result[key] = arr.copy()
            else:
                result[key] = np.zeros_like(arr)
        else:
            result[key] = arr
    return result


def get_profile_feature_mask(
    profile_name: str,
    profiles: dict,
) -> dict[str, bool]:
    """Return a boolean mask: feature_name -> True if included in profile."""
    allowed = expand_profile_features(profiles, profile_name)
    mask = {}
    for feat in allowed:
        mask[feat] = True
    return mask
