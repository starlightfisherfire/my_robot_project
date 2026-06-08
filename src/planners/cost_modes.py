"""
Cost mode variants for Paper 1 planner.

Extends cost_functions.py with selectable cost modes without replacing it.
Each mode is a thin wrapper that configures CostWeights and adds/removes terms.

Modes:
    current          — Original cost (as-is)
    no_redundant_contact — Remove no_contact_cost, keep reach/first/persistent
    push_efficiency  — Add push_efficiency term
    object_obstacle  — Add object-obstacle proximity term
    terminal_hold    — Add terminal hold term
    staged_full      — All new terms combined
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from src.planners.cost_functions import (
    CostWeights,
    rollout_cost as _original_rollout_cost,
    pose_error,
    wrap_angle,
    reach_cost,
    no_contact_cost,
    push_alignment_cost,
    first_contact_cost,
    persistent_contact_cost,
    action_regularization_cost,
    action_smoothness_cost,
    collision_cost,
    obstacle_proximity_cost,
)

CostMode = Literal[
    "current",
    "no_redundant_contact",
    "push_efficiency",
    "object_obstacle",
    "terminal_hold",
    "staged_full",
]


# ──────────────────────────────────────────────────────────
# New cost terms
# ──────────────────────────────────────────────────────────

def push_efficiency_cost(
    contact_flags: np.ndarray | None,
    object_poses: np.ndarray,
    goal_pose: np.ndarray,
) -> float:
    """
    Reward effective pushing: object displacement toward goal during contact.

    Returns:
        1 - (effective_push / total_object_move)   if contact occurs and object moves,
        1.0                                         if no contact or no movement.

    Range: [0, 1].
    0 = all movement was toward goal during contact.
    1 = no useful pushing (no contact, or moved away from goal).
    """
    if contact_flags is None:
        return 1.0

    contact_flags = np.asarray(contact_flags, dtype=np.float64)
    object_poses = np.asarray(object_poses, dtype=np.float64)
    goal_pose = np.asarray(goal_pose, dtype=np.float64)

    if not np.isfinite(contact_flags).all():
        return 1.0

    contact_indices = np.where(contact_flags > 0.5)[0]
    if len(contact_indices) < 2:
        return 1.0

    # Object displacement during contact phases
    goal_dir = goal_pose[:2] - object_poses[0, :2]
    goal_dir_norm = float(np.linalg.norm(goal_dir))
    if goal_dir_norm < 1e-8:
        return 0.0
    goal_dir = goal_dir / goal_dir_norm

    total_toward_goal = 0.0
    total_move = 0.0

    for i in range(len(contact_indices) - 1):
        t0 = contact_indices[i]
        t1 = contact_indices[i + 1]
        if t1 == t0 + 1:
            move = object_poses[t1, :2] - object_poses[t0, :2]
            move_norm = float(np.linalg.norm(move))
            if move_norm > 1e-8:
                total_move += move_norm
                total_toward_goal += float(np.dot(move, goal_dir))

    if total_move < 1e-8:
        return 1.0

    efficiency = total_toward_goal / total_move  # [-1, 1]
    return float(np.clip(1.0 - efficiency, 0.0, 2.0) / 2.0)  # [0, 1]


def object_obstacle_proximity_cost(
    object_positions: np.ndarray,
    obstacle_positions: np.ndarray,
    obstacle_radii: np.ndarray,
    margin: float = 0.05,
) -> float:
    """
    Penalize object being pushed into/near obstacles.

    Args:
        object_positions: [T, 2] object trajectory.
        obstacle_positions: [N, 2] obstacle centers.
        obstacle_radii: [N] obstacle radii.
        margin: extra clearance.

    Returns:
        Sum of squared penetrations into margin zone over all timesteps.
    """
    object_positions = np.asarray(object_positions, dtype=np.float64)
    obstacle_positions = np.asarray(obstacle_positions, dtype=np.float64)
    obstacle_radii = np.asarray(obstacle_radii, dtype=np.float64)

    if object_positions.ndim != 2 or object_positions.shape[-1] != 2:
        raise ValueError(f"Expected object_positions [T,2], got {object_positions.shape}")

    n_obs = len(obstacle_positions)
    total = 0.0
    for i in range(n_obs):
        diff = object_positions - obstacle_positions[i]
        dist = np.linalg.norm(diff, axis=-1)
        penetration = (obstacle_radii[i] + margin) - dist
        total += float(np.sum(np.maximum(penetration, 0.0) ** 2))

    return total


def terminal_hold_cost(
    ee_positions: np.ndarray,
    object_positions: np.ndarray,
    terminal_fraction: float = 0.2,
) -> float:
    """
    Encourage EE to stay near object during the final portion of rollout.

    Returns:
        Mean EE-object distance² over the last `terminal_fraction` of timesteps.
    """
    ee_positions = np.asarray(ee_positions, dtype=np.float64)
    object_positions = np.asarray(object_positions, dtype=np.float64)

    T = len(ee_positions)
    if T == 0:
        return 0.0

    start = max(0, int(T * (1.0 - terminal_fraction)))
    ee_tail = ee_positions[start:]
    obj_tail = object_positions[start:]

    d = ee_tail - obj_tail
    return float(np.mean(np.sum(d ** 2, axis=-1)))


def drift_after_best_cost(
    object_poses: np.ndarray,
    goal_pose: np.ndarray,
) -> float:
    """
    Penalize object drifting away from goal after reaching its best position.

    Returns:
        (final_dist_to_goal - best_dist_to_goal) / best_dist_to_goal   if positive,
        0.0                                                              otherwise.

    Range: [0, ∞). 0 = no drift or improved.
    """
    object_poses = np.asarray(object_poses, dtype=np.float64)
    goal_pose = np.asarray(goal_pose, dtype=np.float64)

    positions = object_poses[:, :2]
    goal_xy = goal_pose[:2]
    dists = np.linalg.norm(positions - goal_xy, axis=-1)

    best_dist = float(np.min(dists))
    final_dist = float(dists[-1])

    if best_dist < 1e-8:
        return max(0.0, final_dist)

    return max(0.0, (final_dist - best_dist) / best_dist)


# ──────────────────────────────────────────────────────────
# Cost mode weights
# ──────────────────────────────────────────────────────────

def get_mode_weights(mode: CostMode) -> CostWeights:
    """Return CostWeights configured for the given mode."""
    if mode == "current":
        return CostWeights()  # defaults as-is

    elif mode == "no_redundant_contact":
        w = CostWeights()
        w.w_no_contact = 0.0  # remove redundant no_contact
        return w

    elif mode == "push_efficiency":
        w = CostWeights()
        w.w_no_contact = 0.0
        w.w_reach = 3.0  # reduce from 5
        w.w_push_alignment = 0.0  # disable misleading
        return w

    elif mode == "object_obstacle":
        w = CostWeights()
        w.w_no_contact = 0.0
        w.w_reach = 3.0
        w.w_push_alignment = 0.0
        return w

    elif mode == "terminal_hold":
        w = CostWeights()
        w.w_no_contact = 0.0
        w.w_reach = 3.0
        w.w_push_alignment = 0.0
        return w

    elif mode == "staged_full":
        w = CostWeights()
        w.w_no_contact = 0.0       # removed (redundant)
        w.w_reach = 2.0            # reduced (gradient guidance only)
        w.w_push_alignment = 0.0   # removed (misleading)
        w.w_early_contact = 4.0    # increased (encourage fast find)
        w.w_persistent_contact = 3.0  # increased
        w.w_smooth = 0.15          # slightly increased
        return w

    else:
        raise ValueError(f"Unknown cost mode: {mode}")


# ──────────────────────────────────────────────────────────
# New weights dataclass for extended modes
# ──────────────────────────────────────────────────────────

@dataclass
class StagedCostWeights:
    """Extended weights for staged cost modes."""
    # All base CostWeights fields
    w_pos: float = 10.0
    w_theta: float = 2.0
    w_reach: float = 2.0
    w_no_contact: float = 0.0
    w_push_alignment: float = 0.0
    w_collision: float = 20.0
    w_collision_step: float = 1.0
    w_proximity: float = 5.0
    w_early_contact: float = 4.0
    w_persistent_contact: float = 3.0
    w_action: float = 0.05
    w_smooth: float = 0.15
    w_subgoal: float = 0.0
    # New terms
    w_push_efficiency: float = 3.0
    w_object_obstacle: float = 5.0
    w_terminal_hold: float = 2.0
    w_drift_after_best: float = 4.0


# ──────────────────────────────────────────────────────────
# Unified rollout cost with mode support
# ──────────────────────────────────────────────────────────

def rollout_cost_with_mode(
    predicted_object_poses: np.ndarray,
    ee_positions: np.ndarray,
    action_sequence: np.ndarray,
    goal_pose: np.ndarray,
    cost_mode: CostMode = "current",
    contact_flags: np.ndarray | None = None,
    collision_flags: np.ndarray | None = None,
    subgoal_pose: np.ndarray | None = None,
    subgoal_index: int | None = None,
    obstacle_positions: np.ndarray | None = None,
    obstacle_radii: np.ndarray | None = None,
    proximity_margin: float = 0.03,
) -> float:
    """
    Unified rollout cost with selectable cost mode.

    For modes "current" through "terminal_hold", delegates to original rollout_cost
    with adjusted weights.

    For mode "staged_full", uses all new terms.
    """
    predicted_object_poses = np.asarray(predicted_object_poses, dtype=np.float64)
    ee_positions = np.asarray(ee_positions, dtype=np.float64)
    action_sequence = np.asarray(action_sequence, dtype=np.float64)
    goal_pose = np.asarray(goal_pose, dtype=np.float64)

    # ── Modes that just change weights ──
    if cost_mode != "staged_full":
        weights = get_mode_weights(cost_mode)
        return _original_rollout_cost(
            predicted_object_poses=predicted_object_poses,
            ee_positions=ee_positions,
            action_sequence=action_sequence,
            goal_pose=goal_pose,
            weights=weights,
            contact_flags=contact_flags,
            collision_flags=collision_flags,
            subgoal_pose=subgoal_pose,
            subgoal_index=subgoal_index,
            obstacle_positions=obstacle_positions,
            obstacle_radii=obstacle_radii,
            proximity_margin=proximity_margin,
        )

    # ── staged_full: use all new terms ──
    sw = StagedCostWeights()

    object_initial_pose = predicted_object_poses[0]
    object_final_pose = predicted_object_poses[-1]
    object_positions = predicted_object_poses[:, :2]

    pos_error_sq, theta_error_sq = pose_error(object_final_pose, goal_pose)

    total = 0.0

    # Terminal objective
    total += sw.w_pos * pos_error_sq
    total += sw.w_theta * theta_error_sq

    # Contact guidance
    total += sw.w_reach * reach_cost(ee_positions, object_positions)
    total += sw.w_early_contact * first_contact_cost(
        contact_flags=contact_flags,
        ee_positions=ee_positions,
        object_positions=object_positions,
    )
    total += sw.w_persistent_contact * persistent_contact_cost(
        contact_flags=contact_flags,
        ee_positions=ee_positions,
        object_positions=object_positions,
    )

    # Push efficiency
    total += sw.w_push_efficiency * push_efficiency_cost(
        contact_flags=contact_flags,
        object_poses=predicted_object_poses,
        goal_pose=goal_pose,
    )

    # Object-obstacle proximity
    if obstacle_positions is not None and obstacle_radii is not None:
        total += sw.w_object_obstacle * object_obstacle_proximity_cost(
            object_positions=object_positions,
            obstacle_positions=obstacle_positions,
            obstacle_radii=obstacle_radii,
        )

    # Terminal hold
    total += sw.w_terminal_hold * terminal_hold_cost(
        ee_positions=ee_positions,
        object_positions=object_positions,
    )

    # Drift after best
    total += sw.w_drift_after_best * drift_after_best_cost(
        object_poses=predicted_object_poses,
        goal_pose=goal_pose,
    )

    # Action regularization
    total += sw.w_action * action_regularization_cost(action_sequence)
    total += sw.w_smooth * action_smoothness_cost(action_sequence)

    # Collision (if available)
    if collision_flags is not None:
        collision_any, collision_count, _ = collision_cost(collision_flags)
        total += sw.w_collision * collision_any
        total += sw.w_collision_step * collision_count

    # EE-obstacle proximity
    if obstacle_positions is not None and obstacle_radii is not None and sw.w_proximity > 0.0:
        total += sw.w_proximity * obstacle_proximity_cost(
            ee_positions=ee_positions,
            obstacle_positions=obstacle_positions,
            obstacle_radii=obstacle_radii,
            margin=proximity_margin,
        )

    if not np.isfinite(total):
        raise ValueError(f"Non-finite rollout cost: {total}")

    return float(total)


def list_cost_modes() -> list[str]:
    """Return all available cost mode names."""
    return ["current", "no_redundant_contact", "push_efficiency",
            "object_obstacle", "terminal_hold", "staged_full"]
