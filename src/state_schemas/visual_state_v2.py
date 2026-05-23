"""Data structures for visual_structured_state_v2."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class VisualStateV2Episode:
    """A single v2 episode. All arrays are [T, ...]."""
    object_features: np.ndarray    # [T, N_obj_features]
    relation_features: np.ndarray  # [T, N_rel_features]
    temporal_features: np.ndarray  # [T, N_temp_features]
    proprio_features: np.ndarray   # [T, N_proprio_features]
    goal_features: np.ndarray      # [T, N_goal_features]
    obstacle_features: np.ndarray  # [T, N_obs * max_obstacles]
    nuisance_features: np.ndarray  # [T, N_nuisance] or empty
    privileged_features: np.ndarray  # [T, N_priv] or empty
    actions_physical: np.ndarray   # [T, 2]
    object_poses: np.ndarray       # [T, 3]
    next_object_poses: np.ndarray  # [T, 3]
    goal_pose: np.ndarray          # [3]
    masks: dict = field(default_factory=dict)
    schema_version: str = "0.1"
    metadata: dict = field(default_factory=dict)


@dataclass
class VisualStateV2Batch:
    """Batched v2 episode for training."""
    object_features: np.ndarray    # [B, T, N_obj]
    relation_features: np.ndarray
    temporal_features: np.ndarray
    proprio_features: np.ndarray
    goal_features: np.ndarray
    obstacle_features: np.ndarray
    nuisance_features: np.ndarray
    privileged_features: np.ndarray
    actions: np.ndarray            # [B, T, 2]
    dynamics_target: np.ndarray    # [B, T, 3]
