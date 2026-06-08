from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CostWeights:
    """
    Cost weights for Paper 1 planner.

    v0.1:
        final goal cost is the main objective.
        reach / no_contact / push_alignment are shaping terms for sparse-contact pushing.
        subgoal cost is included but should remain disabled until the learned rollout path is stable.

    Collision cost (v0.2):
        collision_cost = w_collision * max(collision_flags) + w_collision_step * sum(collision_flags)
        w_collision:    one-time penalty if ANY collision occurs (existence signal)
        w_collision_step: per-timestep penalty for each collision step (severity signal)

    Staged contact-obstacle-goal cost (v0.3):
        New weights for staged cost: terminal_hold, contact_duration, progress,
        object_obstacle, object_collision, object_collision_step, ee_obstacle.

    Later:
        Compare w_subgoal = 0.0 vs w_subgoal > 0.0 as a small ablation.
    """

    w_pos: float = 10.0
    w_theta: float = 2.0
    w_reach: float = 5.0
    w_no_contact: float = 2.0
    w_push_alignment: float = 1.0
    w_collision: float = 20.0
    w_collision_step: float = 1.0
    w_proximity: float = 5.0
    w_early_contact: float = 3.0
    w_persistent_contact: float = 2.0
    w_action: float = 0.05
    w_smooth: float = 0.1
    w_subgoal: float = 0.0
    # Staged contact-obstacle-goal cost weights (v0.3)
    w_terminal_hold: float = 10.0
    w_contact_duration: float = 2.0
    w_progress: float = 5.0
    w_object_obstacle: float = 25.0
    w_object_collision: float = 50.0
    w_object_collision_step: float = 3.0
    w_ee_obstacle: float = 8.0


def make_clean_default_cost_weights() -> CostWeights:
    """
    Simplified default cost: keep directional guidance, remove contact micromanagement.

    - Encourage early contact (but not persistent contact or contact duration)
    - Heavy penalty for obstacle collision
    - Reward proximity to goal (implicit "finish fast")
    - No terminal_hold (early stop handles this)
    """
    return CostWeights(
        w_pos=10.0,
        w_theta=2.0,
        w_reach=5.0,
        w_no_contact=2.0,
        w_push_alignment=1.0,
        w_collision=20.0,
        w_collision_step=1.0,
        w_proximity=5.0,
        w_early_contact=3.0,
        w_persistent_contact=0.0,   # removed: don't force persistent contact
        w_action=0.05,
        w_smooth=0.1,
        w_subgoal=0.0,
        w_terminal_hold=0.0,        # removed: early stop is enough
        w_contact_duration=0.0,     # removed: don't micromanage contact
        w_progress=5.0,
        w_object_obstacle=25.0,
        w_object_collision=50.0,
        w_object_collision_step=3.0,
        w_ee_obstacle=8.0,
    )


def make_staged_cost_weights() -> CostWeights:
    """
    Return CostWeights tuned for staged_contact_obstacle_goal_cost.

    Disables reach / no_contact / push_alignment (replaced by staged terms).
    """
    return CostWeights(
        w_pos=10.0,
        w_theta=2.0,
        w_reach=0.0,
        w_no_contact=0.0,
        w_push_alignment=0.0,
        w_collision=30.0,
        w_collision_step=2.0,
        w_proximity=0.0,  # replaced by object_obstacle in staged
        w_early_contact=4.0,
        w_persistent_contact=4.0,
        w_action=0.02,
        w_smooth=0.05,
        w_subgoal=0.0,
        w_terminal_hold=10.0,
        w_contact_duration=2.0,
        w_progress=5.0,
        w_object_obstacle=25.0,
        w_object_collision=50.0,
        w_object_collision_step=3.0,
        w_ee_obstacle=8.0,
    )


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    """
    Wrap angle to [-pi, pi].
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def pose_error(
    pose: np.ndarray,
    target_pose: np.ndarray,
) -> tuple[float, float]:
    """
    Compute planar pose error for a single 2D pose.

    Args:
        pose:
            Shape (3,), representing [x, y, theta].

        target_pose:
            Shape (3,), representing [x, y, theta].

    Returns:
        pos_error_sq:
            Squared x-y position error.

        theta_error_sq:
            Squared wrapped theta error.

    Note:
        This function intentionally does not support batched input.
        For rollout cost, pass one selected pose such as predicted_object_poses[-1].
    """
    pose = np.asarray(pose, dtype=np.float64)
    target_pose = np.asarray(target_pose, dtype=np.float64)

    if pose.shape != (3,) or target_pose.shape != (3,):
        raise ValueError(
            f"Expected pose and target_pose shape (3,), "
            f"got {pose.shape}, {target_pose.shape}"
        )

    if not np.isfinite(pose).all() or not np.isfinite(target_pose).all():
        raise ValueError("pose_error received non-finite pose values.")

    dxy = pose[:2] - target_pose[:2]
    dtheta = wrap_angle(pose[2] - target_pose[2])

    pos_error_sq = float(np.sum(dxy ** 2))
    theta_error_sq = float(dtheta ** 2)

    return pos_error_sq, theta_error_sq


def action_regularization_cost(action_sequence: np.ndarray) -> float:
    """
    Penalize large actions.

    Args:
        action_sequence:
            [H, action_dim]
    """
    action_sequence = np.asarray(action_sequence, dtype=np.float64)

    if action_sequence.ndim != 2:
        raise ValueError(f"Expected action_sequence [H,A], got {action_sequence.shape}")

    if not np.isfinite(action_sequence).all():
        raise ValueError("action_sequence contains non-finite values.")

    return float(np.mean(action_sequence ** 2))


def action_smoothness_cost(action_sequence: np.ndarray) -> float:
    """
    Penalize action changes.

    Args:
        action_sequence:
            [H, action_dim]
    """
    action_sequence = np.asarray(action_sequence, dtype=np.float64)

    if action_sequence.ndim != 2:
        raise ValueError(f"Expected action_sequence [H,A], got {action_sequence.shape}")

    if not np.isfinite(action_sequence).all():
        raise ValueError("action_sequence contains non-finite values.")

    if action_sequence.shape[0] <= 1:
        return 0.0

    diff = action_sequence[1:] - action_sequence[:-1]
    return float(np.mean(diff ** 2))


def reach_cost(
    ee_positions: np.ndarray,
    object_positions: np.ndarray,
) -> float:
    """
    Encourage end-effector to get close to the object.

    Args:
        ee_positions:
            [T, 2]

        object_positions:
            [T, 2]

    Returns:
        minimum EE-object distance squared over the rollout.
    """
    ee_positions = np.asarray(ee_positions, dtype=np.float64)
    object_positions = np.asarray(object_positions, dtype=np.float64)

    if ee_positions.shape != object_positions.shape:
        raise ValueError(
            f"Expected ee_positions and object_positions with same shape, "
            f"got {ee_positions.shape}, {object_positions.shape}"
        )

    if ee_positions.ndim != 2 or ee_positions.shape[-1] != 2:
        raise ValueError(f"Expected positions [T,2], got {ee_positions.shape}")

    if not np.isfinite(ee_positions).all() or not np.isfinite(object_positions).all():
        raise ValueError("reach_cost received non-finite positions.")

    d = ee_positions - object_positions
    dist_sq = np.sum(d ** 2, axis=-1)

    return float(np.min(dist_sq))


def no_contact_cost(
    contact_flags: np.ndarray | None = None,
    ee_positions: np.ndarray | None = None,
    object_positions: np.ndarray | None = None,
    contact_distance_threshold: float = 0.035,
) -> float:
    """
    Penalize no-contact rollouts.

    If contact_flags is provided:
        no_contact_cost = 1 - max(contact_flags)

    Otherwise use distance proxy:
        contact if min EE-object distance < threshold.
    """
    if contact_flags is not None:
        contact_flags = np.asarray(contact_flags, dtype=np.float64)

        if not np.isfinite(contact_flags).all():
            raise ValueError("contact_flags contains non-finite values.")

        has_contact = float(np.max(contact_flags) > 0.5)
        return 1.0 - has_contact

    if ee_positions is None or object_positions is None:
        return 0.0

    ee_positions = np.asarray(ee_positions, dtype=np.float64)
    object_positions = np.asarray(object_positions, dtype=np.float64)

    if ee_positions.shape != object_positions.shape:
        raise ValueError(
            f"Expected ee_positions and object_positions with same shape, "
            f"got {ee_positions.shape}, {object_positions.shape}"
        )

    if not np.isfinite(ee_positions).all() or not np.isfinite(object_positions).all():
        raise ValueError("no_contact_cost received non-finite positions.")

    d = ee_positions - object_positions
    min_dist = float(np.min(np.linalg.norm(d, axis=-1)))

    has_contact = float(min_dist < contact_distance_threshold)
    return 1.0 - has_contact


def push_alignment_cost(
    object_initial_pose: np.ndarray,
    object_final_pose: np.ndarray,
    goal_pose: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Encourage object motion direction to align with object->goal direction.

    Returns:
        0 when aligned;
        around 1 when orthogonal;
        around 2 when opposite.
    """
    object_initial_pose = np.asarray(object_initial_pose, dtype=np.float64)
    object_final_pose = np.asarray(object_final_pose, dtype=np.float64)
    goal_pose = np.asarray(goal_pose, dtype=np.float64)

    if not (
        np.isfinite(object_initial_pose).all()
        and np.isfinite(object_final_pose).all()
        and np.isfinite(goal_pose).all()
    ):
        raise ValueError("push_alignment_cost received non-finite poses.")

    move_vec = object_final_pose[:2] - object_initial_pose[:2]
    goal_vec = goal_pose[:2] - object_initial_pose[:2]

    move_norm = float(np.linalg.norm(move_vec))
    goal_norm = float(np.linalg.norm(goal_vec))

    if move_norm < eps or goal_norm < eps:
        return 1.0

    cos_sim = float(np.dot(move_vec, goal_vec) / (move_norm * goal_norm + eps))
    cos_sim = float(np.clip(cos_sim, -1.0, 1.0))

    return 1.0 - cos_sim


def first_contact_cost(
    contact_flags: np.ndarray | None = None,
    ee_positions: np.ndarray | None = None,
    object_positions: np.ndarray | None = None,
    contact_distance_threshold: float = 0.035,
) -> float:
    """
    Penalize late first contact.  Earlier contact -> lower cost.

    Returns:
        first_contact_timestep / T  if contact occurs,
        1.0                          if no contact at all.

    Range: [0, 1].
    0   = contact at t=0 (immediate).
    ~1  = no contact, or contact only at the final timestep.
    """
    if contact_flags is not None:
        contact_flags = np.asarray(contact_flags, dtype=np.float64)

        if not np.isfinite(contact_flags).all():
            raise ValueError("contact_flags contains non-finite values.")

        T = len(contact_flags)
        if T == 0:
            return 1.0

        contact_indices = np.where(contact_flags > 0.5)[0]
        if len(contact_indices) == 0:
            return 1.0

        first_t = int(contact_indices[0])
        return float(first_t) / float(T)

    # Distance-based proxy
    if ee_positions is None or object_positions is None:
        return 1.0

    ee_positions = np.asarray(ee_positions, dtype=np.float64)
    object_positions = np.asarray(object_positions, dtype=np.float64)

    if ee_positions.shape != object_positions.shape:
        raise ValueError(
            f"Expected ee_positions and object_positions with same shape, "
            f"got {ee_positions.shape} and {object_positions.shape}"
        )

    if not np.isfinite(ee_positions).all() or not np.isfinite(object_positions).all():
        raise ValueError("first_contact_cost received non-finite positions.")

    T = len(ee_positions)
    if T == 0:
        return 1.0

    dists = np.linalg.norm(ee_positions - object_positions, axis=-1)
    contact_indices = np.where(dists < contact_distance_threshold)[0]
    if len(contact_indices) == 0:
        return 1.0

    first_t = int(contact_indices[0])
    return float(first_t) / float(T)


def collision_cost(
    collision_flags: np.ndarray | None,
) -> tuple[float, float, float]:
    """
    Compute collision summary from per-timestep collision flags.

    Args:
        collision_flags:
            Optional [T], 0/1 flags per timestep. 1 = collision.

    Returns:
        collision_any: 1.0 if any timestep has collision, else 0.0
        collision_count: number of timesteps with collision
        collision_rate: fraction of timesteps with collision
    """
    if collision_flags is None:
        return 0.0, 0.0, 0.0

    collision_flags = np.asarray(collision_flags, dtype=np.float64)

    if not np.isfinite(collision_flags).all():
        raise ValueError("collision_cost received non-finite collision_flags.")

    t = len(collision_flags)
    if t == 0:
        return 0.0, 0.0, 0.0

    collision_any = float(np.max(collision_flags))
    collision_count = float(np.sum(collision_flags))
    collision_rate = float(np.mean(collision_flags))

    return collision_any, collision_count, collision_rate


def obstacle_proximity_cost(
    ee_positions: np.ndarray,
    obstacle_positions: np.ndarray,
    obstacle_radii: np.ndarray,
    margin: float = 0.03,
) -> float:
    """
    Smooth repulsive field around obstacles. Gives directional gradient signal
    even when no collision occurs -- guides CEM/MPPI samples away from obstacles.

    Args:
        ee_positions: [T, 2] end-effector trajectory.
        obstacle_positions: [N, 2] obstacle centers.
        obstacle_radii: [N] obstacle radii.
        margin: extra clearance beyond obstacle radius.

    Returns:
        Sum of squared penetrations into the margin zone.
    """
    ee_positions = np.asarray(ee_positions, dtype=np.float64)
    obstacle_positions = np.asarray(obstacle_positions, dtype=np.float64)
    obstacle_radii = np.asarray(obstacle_radii, dtype=np.float64)

    if ee_positions.ndim != 2 or ee_positions.shape[-1] != 2:
        raise ValueError(f"Expected ee_positions [T,2], got {ee_positions.shape}")

    if obstacle_positions.ndim != 2 or obstacle_positions.shape[-1] != 2:
        raise ValueError(
            f"Expected obstacle_positions [N,2], got {obstacle_positions.shape}"
        )

    n_obs = len(obstacle_positions)
    if obstacle_radii.shape != (n_obs,):
        raise ValueError(
            f"Expected obstacle_radii [{n_obs}], got {obstacle_radii.shape}"
        )

    total = 0.0
    for i in range(n_obs):
        diff = ee_positions - obstacle_positions[i]
        dist = np.linalg.norm(diff, axis=-1)
        penetration = (obstacle_radii[i] + margin) - dist
        total += float(np.sum(np.maximum(penetration, 0.0) ** 2))

    return total


def persistent_contact_cost(
    contact_flags: np.ndarray | None = None,
    ee_positions: np.ndarray | None = None,
    object_positions: np.ndarray | None = None,
    contact_distance_threshold: float = 0.035,
) -> float:
    """
    Encourage EE to maintain contact once it has been established.

    Penalises timesteps with no contact *after* the first contact.
    Before any contact has occurred, those timesteps are not penalised
    (that is the job of first_contact_cost).

    Returns:
        (number of post-first-contact no-contact timesteps) / T
        0.0 if contact is maintained after first touch, or if no contact at all.
    """
    if contact_flags is not None:
        contact_flags = np.asarray(contact_flags, dtype=np.float64)

        if not np.isfinite(contact_flags).all():
            raise ValueError("contact_flags contains non-finite values.")

        T = len(contact_flags)
        if T == 0:
            return 0.0

        contact_indices = np.where(contact_flags > 0.5)[0]
        if len(contact_indices) == 0:
            return 0.0  # no contact at all -- handled by no_contact_cost

        first_t = int(contact_indices[0])
        post_contact = contact_flags[first_t:]
        n_lost = float(np.sum(post_contact < 0.5))
        return n_lost / float(T)

    # Distance-based proxy
    if ee_positions is None or object_positions is None:
        return 0.0

    ee_positions = np.asarray(ee_positions, dtype=np.float64)
    object_positions = np.asarray(object_positions, dtype=np.float64)

    if ee_positions.shape != object_positions.shape:
        raise ValueError(
            f"Expected ee_positions and object_positions with same shape, "
            f"got {ee_positions.shape} and {object_positions.shape}"
        )

    if not np.isfinite(ee_positions).all() or not np.isfinite(object_positions).all():
        raise ValueError("persistent_contact_cost received non-finite positions.")

    T = len(ee_positions)
    if T == 0:
        return 0.0

    dists = np.linalg.norm(ee_positions - object_positions, axis=-1)
    in_contact = dists < contact_distance_threshold
    contact_indices = np.where(in_contact)[0]
    if len(contact_indices) == 0:
        return 0.0

    first_t = int(contact_indices[0])
    post_contact = in_contact[first_t:]
    n_lost = float(np.sum(~post_contact))
    return n_lost / float(T)


# =============================================================================
# Staged contact-obstacle-goal cost functions (v0.3)
# =============================================================================


def terminal_hold_cost(
    predicted_object_poses: np.ndarray,
    goal_pose: np.ndarray,
    last_k: int = 10,
    theta_weight: float = 0.2,
) -> float:
    """
    Compute terminal hold cost: average pose error over last K timesteps.

    Penalizes rollouts that pass through goal midway but drift away at the end.
    This solves the 'best_dist is small but final_dist is large' problem.

    Args:
        predicted_object_poses: [T, 3] object poses along rollout.
        goal_pose: [3] target pose.
        last_k: number of final timesteps to average over.
        theta_weight: weight for theta error relative to position error.

    Returns:
        Mean of (position_error_sq + theta_weight * theta_error_sq) over last K steps.
    """
    predicted_object_poses = np.asarray(predicted_object_poses, dtype=np.float64)
    goal_pose = np.asarray(goal_pose, dtype=np.float64)

    if predicted_object_poses.ndim != 2 or predicted_object_poses.shape[-1] != 3:
        raise ValueError(
            f"Expected predicted_object_poses [T,3], got {predicted_object_poses.shape}"
        )

    if not np.isfinite(predicted_object_poses).all() or not np.isfinite(goal_pose).all():
        raise ValueError("terminal_hold_cost received non-finite values.")

    T = len(predicted_object_poses)
    if T == 0:
        return 0.0

    k = min(last_k, T)
    last_poses = predicted_object_poses[-k:]

    errors = []
    for pose in last_poses:
        pos_err_sq, theta_err_sq = pose_error(pose, goal_pose)
        errors.append(pos_err_sq + theta_weight * theta_err_sq)

    return float(np.mean(errors))


def contact_duration_cost(
    contact_flags: np.ndarray | None = None,
    ee_positions: np.ndarray | None = None,
    object_positions: np.ndarray | None = None,
    contact_distance_threshold: float = 0.035,
) -> float:
    """
    Compute contact duration cost: 1 - contact_rate.

    Encourages longer contact throughout the rollout.
    Higher contact rate -> lower cost.

    Args:
        contact_flags: Optional [T], 1 means contact.
        ee_positions: Optional [T, 2] for distance-based proxy.
        object_positions: Optional [T, 2] for distance-based proxy.
        contact_distance_threshold: distance threshold for contact detection.

    Returns:
        1 - contact_rate. Range [0, 1].
    """
    if contact_flags is not None:
        contact_flags = np.asarray(contact_flags, dtype=np.float64)

        if not np.isfinite(contact_flags).all():
            raise ValueError("contact_duration_cost received non-finite contact_flags.")

        contact_rate = float(np.mean(contact_flags > 0.5))
        return 1.0 - contact_rate

    # Distance-based proxy
    if ee_positions is None or object_positions is None:
        return 1.0

    ee_positions = np.asarray(ee_positions, dtype=np.float64)
    object_positions = np.asarray(object_positions, dtype=np.float64)

    if ee_positions.shape != object_positions.shape:
        raise ValueError(
            f"Expected ee_positions and object_positions with same shape, "
            f"got {ee_positions.shape} and {object_positions.shape}"
        )

    if not np.isfinite(ee_positions).all() or not np.isfinite(object_positions).all():
        raise ValueError("contact_duration_cost received non-finite positions.")

    dists = np.linalg.norm(ee_positions - object_positions, axis=-1)
    contact_rate = float(np.mean(dists < contact_distance_threshold))
    return 1.0 - contact_rate


def object_obstacle_proximity_cost(
    object_positions: np.ndarray,
    obstacle_positions: np.ndarray,
    obstacle_radii: np.ndarray,
    object_radius: float = 0.034,
    margin: float = 0.03,
) -> float:
    """
    Smooth repulsive field for object approaching obstacles.

    Penalizes object proximity to obstacles (not EE).
    Gives gradient signal even before collision occurs.

    Args:
        object_positions: [T, 2] object xy positions.
        obstacle_positions: [N, 2] obstacle centers.
        obstacle_radii: [N] obstacle radii.
        object_radius: radius of the object being pushed.
        margin: extra clearance beyond combined radii.

    Returns:
        Sum of squared penetrations into the margin zone.
    """
    object_positions = np.asarray(object_positions, dtype=np.float64)
    obstacle_positions = np.asarray(obstacle_positions, dtype=np.float64)
    obstacle_radii = np.asarray(obstacle_radii, dtype=np.float64)

    if object_positions.ndim != 2 or object_positions.shape[-1] != 2:
        raise ValueError(
            f"Expected object_positions [T,2], got {object_positions.shape}"
        )

    if obstacle_positions.ndim != 2 or obstacle_positions.shape[-1] != 2:
        raise ValueError(
            f"Expected obstacle_positions [N,2], got {obstacle_positions.shape}"
        )

    n_obs = len(obstacle_positions)
    if obstacle_radii.shape != (n_obs,):
        raise ValueError(
            f"Expected obstacle_radii [{n_obs}], got {obstacle_radii.shape}"
        )

    T = len(object_positions)
    total = 0.0
    for i in range(n_obs):
        diff = object_positions - obstacle_positions[i]
        dist = np.linalg.norm(diff, axis=-1)
        combined_radius = object_radius + obstacle_radii[i]
        penetration = (combined_radius + margin) - dist
        total += float(np.sum(np.maximum(penetration, 0.0) ** 2))

    # Normalize by T for horizon invariance
    return total / max(T, 1)


def object_collision_cost(
    object_positions: np.ndarray,
    obstacle_positions: np.ndarray,
    obstacle_radii: np.ndarray,
    object_radius: float = 0.034,
) -> tuple[float, float, float]:
    """
    Detect object-obstacle collisions.

    Args:
        object_positions: [T, 2] object xy positions.
        obstacle_positions: [N, 2] obstacle centers.
        obstacle_radii: [N] obstacle radii.
        object_radius: radius of the object being pushed.

    Returns:
        collision_any: 1.0 if any collision occurs, else 0.0.
        collision_count: number of timesteps with collision.
        collision_rate: fraction of timesteps with collision.
    """
    object_positions = np.asarray(object_positions, dtype=np.float64)
    obstacle_positions = np.asarray(obstacle_positions, dtype=np.float64)
    obstacle_radii = np.asarray(obstacle_radii, dtype=np.float64)

    if object_positions.ndim != 2 or object_positions.shape[-1] != 2:
        raise ValueError(
            f"Expected object_positions [T,2], got {object_positions.shape}"
        )

    if obstacle_positions.ndim != 2 or obstacle_positions.shape[-1] != 2:
        raise ValueError(
            f"Expected obstacle_positions [N,2], got {obstacle_positions.shape}"
        )

    T = len(object_positions)
    if T == 0:
        return 0.0, 0.0, 0.0

    n_obs = len(obstacle_positions)
    collision_per_step = np.zeros(T, dtype=bool)

    for i in range(n_obs):
        diff = object_positions - obstacle_positions[i]
        dist = np.linalg.norm(diff, axis=-1)
        combined_radius = object_radius + obstacle_radii[i]
        collision_per_step |= (dist < combined_radius)

    collision_any = float(np.any(collision_per_step))
    collision_count = float(np.sum(collision_per_step))
    collision_rate = float(np.mean(collision_per_step))

    return collision_any, collision_count, collision_rate


def obstacle_aware_potential(
    object_xy: np.ndarray,
    goal_xy: np.ndarray,
    obstacle_positions: np.ndarray | None = None,
    obstacle_radii: np.ndarray | None = None,
    object_radius: float = 0.034,
    margin: float = 0.03,
    obstacle_scale: float = 1.0,
) -> float:
    """
    Compute obstacle-aware potential function for a single 2D position.

    potential(x) = ||x - goal||^2 + obstacle_repulsion(x)
    obstacle_repulsion = sum_obs max(0, object_radius + obs_radius + margin - dist(x, obs))^2 * obstacle_scale

    If no obstacles provided, returns just goal distance squared.

    Args:
        object_xy: [2] object xy position.
        goal_xy: [2] goal xy position.
        obstacle_positions: Optional [N, 2] obstacle centers.
        obstacle_radii: Optional [N] obstacle radii.
        object_radius: radius of the object being pushed.
        margin: extra clearance beyond combined radii.
        obstacle_scale: scaling factor for obstacle repulsion.

    Returns:
        Scalar potential value.
    """
    object_xy = np.asarray(object_xy, dtype=np.float64)
    goal_xy = np.asarray(goal_xy, dtype=np.float64)

    if object_xy.shape != (2,):
        raise ValueError(f"Expected object_xy [2], got {object_xy.shape}")
    if goal_xy.shape != (2,):
        raise ValueError(f"Expected goal_xy [2], got {goal_xy.shape}")

    if not np.isfinite(object_xy).all() or not np.isfinite(goal_xy).all():
        raise ValueError("obstacle_aware_potential received non-finite values.")

    # Goal distance squared
    goal_dist_sq = float(np.sum((object_xy - goal_xy) ** 2))

    # Obstacle repulsion
    obstacle_repulsion = 0.0
    if obstacle_positions is not None and obstacle_radii is not None:
        obstacle_positions = np.asarray(obstacle_positions, dtype=np.float64)
        obstacle_radii = np.asarray(obstacle_radii, dtype=np.float64)

        n_obs = len(obstacle_positions)
        for i in range(n_obs):
            diff = object_xy - obstacle_positions[i]
            dist = float(np.linalg.norm(diff))
            combined_radius = object_radius + obstacle_radii[i]
            penetration = (combined_radius + margin) - dist
            obstacle_repulsion += max(penetration, 0.0) ** 2 * obstacle_scale

    return goal_dist_sq + obstacle_repulsion


def obstacle_aware_progress_cost(
    object_positions: np.ndarray,
    goal_pose: np.ndarray,
    obstacle_positions: np.ndarray | None = None,
    obstacle_radii: np.ndarray | None = None,
    object_radius: float = 0.034,
    margin: float = 0.03,
    progress_margin: float = 1e-4,
) -> float:
    """
    Compute obstacle-aware progress cost (dense progress shaping).

    Penalizes *lack of progress* rather than just potential increase.
    A static trajectory has cost > 0, unlike the old version.

    cost = mean(max(0, progress_margin - (potential_t - potential_{t+1})))

    This means:
    - If potential drops by >= progress_margin each step, cost = 0.
    - If potential stays flat or rises, cost > 0.
    - Replaces straight-line push_alignment for obstacle-aware navigation.

    Args:
        object_positions: [T, 2] object xy positions.
        goal_pose: [3] goal pose (only xy used).
        obstacle_positions: Optional [N, 2] obstacle centers.
        obstacle_radii: Optional [N] obstacle radii.
        object_radius: radius of the object being pushed.
        margin: extra clearance beyond combined radii.
        progress_margin: minimum expected potential decrease per step.

    Returns:
        Mean progress deficit. 0 if sufficient progress each step.
    """
    object_positions = np.asarray(object_positions, dtype=np.float64)
    goal_pose = np.asarray(goal_pose, dtype=np.float64)

    if object_positions.ndim != 2 or object_positions.shape[-1] != 2:
        raise ValueError(
            f"Expected object_positions [T,2], got {object_positions.shape}"
        )

    if not np.isfinite(object_positions).all() or not np.isfinite(goal_pose).all():
        raise ValueError("obstacle_aware_progress_cost received non-finite values.")

    T = len(object_positions)
    if T <= 1:
        return 0.0

    goal_xy = goal_pose[:2]

    # Compute potential for each timestep
    potentials = []
    for t in range(T):
        pot = obstacle_aware_potential(
            object_xy=object_positions[t],
            goal_xy=goal_xy,
            obstacle_positions=obstacle_positions,
            obstacle_radii=obstacle_radii,
            object_radius=object_radius,
            margin=margin,
        )
        potentials.append(pot)

    potentials = np.array(potentials)

    # progress_decrease = potential_t - potential_{t+1} (positive = good)
    progress_decrease = potentials[:-1] - potentials[1:]

    # Penalize insufficient progress
    deficit = np.maximum(progress_margin - progress_decrease, 0.0)

    return float(np.mean(deficit))


def staged_contact_obstacle_goal_cost(
    predicted_object_poses: np.ndarray,
    ee_positions: np.ndarray,
    action_sequence: np.ndarray,
    goal_pose: np.ndarray,
    weights: CostWeights | None = None,
    contact_flags: np.ndarray | None = None,
    collision_flags: np.ndarray | None = None,
    obstacle_positions: np.ndarray | None = None,
    obstacle_radii: np.ndarray | None = None,
    object_radius: float = 0.034,
    proximity_margin: float = 0.03,
    terminal_last_k: int = 10,
    return_breakdown: bool = False,
) -> float | dict:
    """
    Staged contact-obstacle-goal cost for MuJoCo Oracle CEM/MPPI sweep.

    Stages:
    1. Pusher approaches object (early_contact)
    2. Pusher contacts object (first_contact, persistent_contact)
    3. Object moves toward goal avoiding obstacles (obstacle_aware_progress)
    4. Object reaches and stays at goal (terminal_hold, final_pose)

    This cost does NOT use push_alignment (straight-line assumption fails with obstacles).
    This cost does NOT use no_contact or reach (handled by contact stages).

    Args:
        predicted_object_poses: [T, 3] object poses along rollout.
        ee_positions: [T, 2] end-effector positions.
        action_sequence: [H, action_dim] action sequence.
        goal_pose: [3] target pose.
        weights: CostWeights instance (uses defaults if None).
        contact_flags: Optional [T], 1 means contact.
        collision_flags: Optional [T], 1 means EE collision.
        obstacle_positions: Optional [N, 2] obstacle centers.
        obstacle_radii: Optional [N] obstacle radii.
        object_radius: radius of the object being pushed.
        proximity_margin: margin for proximity cost.
        terminal_last_k: number of final timesteps for terminal hold cost.
        return_breakdown: if True, return dict with cost breakdown.

    Returns:
        Total cost (float), or dict with breakdown if return_breakdown=True.
    """
    if weights is None:
        weights = make_staged_cost_weights()

    predicted_object_poses = np.asarray(predicted_object_poses, dtype=np.float64)
    ee_positions = np.asarray(ee_positions, dtype=np.float64)
    action_sequence = np.asarray(action_sequence, dtype=np.float64)
    goal_pose = np.asarray(goal_pose, dtype=np.float64)

    # Validate inputs
    if predicted_object_poses.ndim != 2 or predicted_object_poses.shape[-1] != 3:
        raise ValueError(
            f"Expected predicted_object_poses [T,3], got {predicted_object_poses.shape}"
        )

    if ee_positions.ndim != 2 or ee_positions.shape[-1] != 2:
        raise ValueError(f"Expected ee_positions [T,2], got {ee_positions.shape}")

    if action_sequence.ndim != 2:
        raise ValueError(f"Expected action_sequence [H,A], got {action_sequence.shape}")

    if len(predicted_object_poses) != len(ee_positions):
        raise ValueError(
            f"Expected predicted_object_poses and ee_positions to have same T, "
            f"got {len(predicted_object_poses)} and {len(ee_positions)}"
        )

    if not (
        np.isfinite(predicted_object_poses).all()
        and np.isfinite(ee_positions).all()
        and np.isfinite(action_sequence).all()
        and np.isfinite(goal_pose).all()
    ):
        raise ValueError("staged_contact_obstacle_goal_cost received non-finite inputs.")

    object_positions = predicted_object_poses[:, :2]
    T = len(predicted_object_poses)

    # === Cost terms ===

    # 1. Final goal cost (position + theta)
    object_final_pose = predicted_object_poses[-1]
    pos_error_sq, theta_error_sq = pose_error(object_final_pose, goal_pose)
    final_pos = weights.w_pos * pos_error_sq
    final_theta = weights.w_theta * theta_error_sq

    # 2. Terminal hold cost
    terminal_hold = weights.w_terminal_hold * terminal_hold_cost(
        predicted_object_poses=predicted_object_poses,
        goal_pose=goal_pose,
        last_k=terminal_last_k,
    )

    # 3. Early contact cost
    early_contact = weights.w_early_contact * first_contact_cost(
        contact_flags=contact_flags,
        ee_positions=ee_positions,
        object_positions=object_positions,
    )

    # 4. Persistent contact cost
    persistent_contact = weights.w_persistent_contact * persistent_contact_cost(
        contact_flags=contact_flags,
        ee_positions=ee_positions,
        object_positions=object_positions,
    )

    # 5. Contact duration cost
    contact_duration = weights.w_contact_duration * contact_duration_cost(
        contact_flags=contact_flags,
        ee_positions=ee_positions,
        object_positions=object_positions,
    )

    # 6. Obstacle-aware progress cost
    obstacle_aware_progress = weights.w_progress * obstacle_aware_progress_cost(
        object_positions=object_positions,
        goal_pose=goal_pose,
        obstacle_positions=obstacle_positions,
        obstacle_radii=obstacle_radii,
        object_radius=object_radius,
        margin=proximity_margin,
    )

    # 7. Object obstacle proximity cost
    object_obstacle_proximity = 0.0
    if obstacle_positions is not None and obstacle_radii is not None:
        object_obstacle_proximity = weights.w_object_obstacle * object_obstacle_proximity_cost(
            object_positions=object_positions,
            obstacle_positions=obstacle_positions,
            obstacle_radii=obstacle_radii,
            object_radius=object_radius,
            margin=proximity_margin,
        )

    # 8. Object collision cost (normalize count to rate)
    object_collision_any = 0.0
    object_collision_step = 0.0
    if obstacle_positions is not None and obstacle_radii is not None:
        obj_coll_any, obj_coll_count, _ = object_collision_cost(
            object_positions=object_positions,
            obstacle_positions=obstacle_positions,
            obstacle_radii=obstacle_radii,
            object_radius=object_radius,
        )
        object_collision_any = weights.w_object_collision * obj_coll_any
        # Normalize: count -> count/T (collision rate)
        object_collision_step = weights.w_object_collision_step * (obj_coll_count / max(T, 1))

    # 9. EE obstacle proximity cost (normalize by T)
    ee_obstacle_proximity = 0.0
    if obstacle_positions is not None and obstacle_radii is not None and weights.w_ee_obstacle > 0.0:
        raw_ee_prox = obstacle_proximity_cost(
            ee_positions=ee_positions,
            obstacle_positions=obstacle_positions,
            obstacle_radii=obstacle_radii,
            margin=proximity_margin,
        )
        # Normalize: sum -> mean over T
        ee_obstacle_proximity = weights.w_ee_obstacle * (raw_ee_prox / max(T, 1))

    # 10. EE collision cost (normalize count to rate)
    collision_any = 0.0
    collision_step = 0.0
    if collision_flags is not None:
        coll_any, coll_count, _ = collision_cost(collision_flags)
        collision_any = weights.w_collision * coll_any
        # Normalize: count -> count/T (collision rate)
        collision_step = weights.w_collision_step * (coll_count / max(T, 1))

    # 11. Action regularization
    action = weights.w_action * action_regularization_cost(action_sequence)

    # 12. Action smoothness
    smooth = weights.w_smooth * action_smoothness_cost(action_sequence)

    # === Total ===
    total = (
        final_pos
        + final_theta
        + terminal_hold
        + early_contact
        + persistent_contact
        + contact_duration
        + obstacle_aware_progress
        + object_obstacle_proximity
        + object_collision_any
        + object_collision_step
        + ee_obstacle_proximity
        + collision_any
        + collision_step
        + action
        + smooth
    )

    if not np.isfinite(total):
        raise ValueError(f"Non-finite staged cost: {total}")

    if return_breakdown:
        return {
            "final_pos": float(final_pos),
            "final_theta": float(final_theta),
            "terminal_hold": float(terminal_hold),
            "early_contact": float(early_contact),
            "persistent_contact": float(persistent_contact),
            "contact_duration": float(contact_duration),
            "obstacle_aware_progress": float(obstacle_aware_progress),
            "object_obstacle_proximity": float(object_obstacle_proximity),
            "object_collision_any": float(object_collision_any),
            "object_collision_step": float(object_collision_step),
            "ee_obstacle_proximity": float(ee_obstacle_proximity),
            "collision_any": float(collision_any),
            "collision_step": float(collision_step),
            "action": float(action),
            "smooth": float(smooth),
            "total": float(total),
        }

    return float(total)


# =============================================================================
# Original rollout_cost with cost_mode dispatch (v0.3)
# =============================================================================


def rollout_cost(
    predicted_object_poses: np.ndarray,
    ee_positions: np.ndarray,
    action_sequence: np.ndarray,
    goal_pose: np.ndarray,
    weights: CostWeights | None = None,
    contact_flags: np.ndarray | None = None,
    collision_flags: np.ndarray | None = None,
    subgoal_pose: np.ndarray | None = None,
    subgoal_index: int | None = None,
    obstacle_positions: np.ndarray | None = None,
    obstacle_radii: np.ndarray | None = None,
    proximity_margin: float = 0.03,
    cost_mode: str = "current",
) -> float:
    """
    Compute planner cost for one rollout.

    Args:
        predicted_object_poses:
            [T, 3] object poses along the rollout.
            predicted_object_poses[0] should be the current / initial object pose.
            predicted_object_poses[-1] is used for final goal cost.

        ee_positions:
            [T, 2] predicted end-effector positions.

        action_sequence:
            [H, action_dim]

        goal_pose:
            [3]

        contact_flags:
            Optional [T], 1 means contact.

        collision_flags:
            Optional [T], 1 means collision.

        subgoal_pose:
            Optional [3]. Used only when weights.w_subgoal > 0.

        subgoal_index:
            Which predicted timestep to compare to subgoal.

        cost_mode:
            "current" for original cost (backward-compatible).
            "staged_contact_obstacle_goal" for new staged cost.
    """
    # Dispatch to staged cost if requested
    if cost_mode == "staged_contact_obstacle_goal":
        return staged_contact_obstacle_goal_cost(
            predicted_object_poses=predicted_object_poses,
            ee_positions=ee_positions,
            action_sequence=action_sequence,
            goal_pose=goal_pose,
            weights=weights,
            contact_flags=contact_flags,
            collision_flags=collision_flags,
            obstacle_positions=obstacle_positions,
            obstacle_radii=obstacle_radii,
            proximity_margin=proximity_margin,
        )

    if cost_mode not in ("current", "clean_default"):
        raise ValueError(f"Unknown cost_mode: {cost_mode}. Use 'current', 'clean_default', or 'staged_contact_obstacle_goal'.")

    # Original cost logic (backward-compatible)
    if weights is None:
        weights = CostWeights()

    predicted_object_poses = np.asarray(predicted_object_poses, dtype=np.float64)
    ee_positions = np.asarray(ee_positions, dtype=np.float64)
    action_sequence = np.asarray(action_sequence, dtype=np.float64)
    goal_pose = np.asarray(goal_pose, dtype=np.float64)

    if predicted_object_poses.ndim != 2 or predicted_object_poses.shape[-1] != 3:
        raise ValueError(
            f"Expected predicted_object_poses [T,3], got {predicted_object_poses.shape}"
        )

    if ee_positions.ndim != 2 or ee_positions.shape[-1] != 2:
        raise ValueError(f"Expected ee_positions [T,2], got {ee_positions.shape}")

    if action_sequence.ndim != 2:
        raise ValueError(f"Expected action_sequence [H,A], got {action_sequence.shape}")

    if len(predicted_object_poses) != len(ee_positions):
        raise ValueError(
            f"Expected predicted_object_poses and ee_positions to have same T, "
            f"got {len(predicted_object_poses)} and {len(ee_positions)}"
        )

    if not (
        np.isfinite(predicted_object_poses).all()
        and np.isfinite(ee_positions).all()
        and np.isfinite(action_sequence).all()
        and np.isfinite(goal_pose).all()
    ):
        raise ValueError("rollout_cost received non-finite inputs.")

    object_initial_pose = predicted_object_poses[0]
    object_final_pose = predicted_object_poses[-1]

    pos_error_sq, theta_error_sq = pose_error(object_final_pose, goal_pose)

    object_positions = predicted_object_poses[:, :2]

    total = 0.0
    total += weights.w_pos * pos_error_sq
    total += weights.w_theta * theta_error_sq
    total += weights.w_reach * reach_cost(ee_positions, object_positions)
    total += weights.w_no_contact * no_contact_cost(
        contact_flags=contact_flags,
        ee_positions=ee_positions,
        object_positions=object_positions,
    )
    total += weights.w_push_alignment * push_alignment_cost(
        object_initial_pose=object_initial_pose,
        object_final_pose=object_final_pose,
        goal_pose=goal_pose,
    )
    total += weights.w_early_contact * first_contact_cost(
        contact_flags=contact_flags,
        ee_positions=ee_positions,
        object_positions=object_positions,
    )
    total += weights.w_persistent_contact * persistent_contact_cost(
        contact_flags=contact_flags,
        ee_positions=ee_positions,
        object_positions=object_positions,
    )
    total += weights.w_action * action_regularization_cost(action_sequence)
    total += weights.w_smooth * action_smoothness_cost(action_sequence)

    if collision_flags is not None:
        collision_any, collision_count, _ = collision_cost(collision_flags)
        total += weights.w_collision * collision_any
        total += weights.w_collision_step * collision_count

    if obstacle_positions is not None and obstacle_radii is not None and weights.w_proximity > 0.0:
        total += weights.w_proximity * obstacle_proximity_cost(
            ee_positions=ee_positions,
            obstacle_positions=obstacle_positions,
            obstacle_radii=obstacle_radii,
            margin=proximity_margin,
        )

    if subgoal_pose is not None and weights.w_subgoal > 0.0:
        subgoal_pose = np.asarray(subgoal_pose, dtype=np.float64)

        if not np.isfinite(subgoal_pose).all():
            raise ValueError("subgoal_pose contains non-finite values.")

        if subgoal_index is None:
            subgoal_index = max(0, len(predicted_object_poses) // 2)

        subgoal_index = int(np.clip(subgoal_index, 0, len(predicted_object_poses) - 1))
        sub_pos_error_sq, sub_theta_error_sq = pose_error(
            predicted_object_poses[subgoal_index],
            subgoal_pose,
        )
        total += weights.w_subgoal * (sub_pos_error_sq + 0.2 * sub_theta_error_sq)

    if not np.isfinite(total):
        raise ValueError(f"Non-finite rollout cost: {total}")

    return float(total)
